# Milestone 3 Task 2: Mixed-Workload Eviction

This extension mixes the eight Task 1 workloads into one trace and replays
that trace under TLRU, LRU, LFU, and FIFO on one shared cache. It uses the
same v0 online simulator. Task 1's separate-workload runs are unchanged.

Input data, formats, and the simulator environment are the ones in
[the Task 1 guide](milestone3-task1.md). Task 1's prepared directories cannot
be reused: their `max_model_len` values differ, and one server has one
context limit. Prepare all eight sources again at 32768.

## What the mix changes

`mixed_prepare` writes one directory per label (no per-seed copies) with
think time 0. `mix_workload` then rebuilds arrivals and refuses a mix whose
sources disagree on model, revision, tokenizer, `max_model_len`, or chat
template. Each conversation is tagged with its label. Text requests keep
`chatcmpl-` trace ids; block traces keep `cmpl-…-0` ids. Both go out with
the label, chat through `/v1/chat/completions` and block traces through
`/v1/completions`.

Conversations are shuffled with `--seed`. Only the first turn of each
conversation draws a Poisson arrival at `--request-rate`. Each later turn is
placed at the previous turn's planned time plus `--think-time`, so it can
land among other conversations' first turns. Recorded Bailian and Mooncake
timestamps are not kept. Replay still sends a later turn only after the
previous reply, then waits out the think time.

The cache is `--cache-ratio` times the mixed trace's working set: the
distinct full KV blocks of prompt plus generated tokens, excluding the final
EOS. All four policies share that capacity. The ratio is an offline size for
the experiment. The evictor does not receive the unique-block list. A cache
that cannot hold the longest request is raised and marked `cache_clamped`.

## TLRU

TLRU orders victims; it does not lock any block in. `evict()` always returns
a block while the free table is non-empty. Lower tier first, and within a
tier the oldest `last_accessed`, then the larger `num_hashed_tokens`, then
the smaller block id:

<!-- Previous order: within a tier, only the oldest last_accessed. -->

| Tier | Which blocks |
| --- | --- |
| 0 | One-shot labels (`quality`, `code`, `mooncake-agent`, `chat`), unlabeled blocks, and multi-turn blocks whose hold has expired or that fall outside the budget |
| 1 | A block whose only label is `agent`, `coding-agent`, `bailian-chat`, or `bailian-coder`, accessed within `--hold-s`, and kept by the budget |
| 2 | A block seen by two or more workload labels |

<!-- Previous tier 1 also included chat, and the five multi-turn labels shared one budget. -->

The four protected labels share one budget. `--hold-budget` is a fraction of
the current free table; when more blocks qualify, the newest
`int(hold_budget * len(free_table))` stay in tier 1 and the older ones drop
to tier 0. A budget of 0 protects nothing. LRU, LFU, FIFO, TFIFO, HFIFO, and
RFIFO ignore `--hold-s` and `--hold-budget`. HLRU uses the same hold and
budget as TLRU.

`--hold-s` is set on its own, longer than `--think-time`, so a next turn that
spends time queued can still find its prefix. `last_accessed` is wall-clock
time, and `evict()` reads the clock when it selects a victim, so time since
the last access still counts against the hold. One-shot, protected, and
shared blocks are kept in heaps ordered by `last_accessed`; the oldest entry
is the victim, and expired or over-budget protected blocks are taken from
that end without scanning the free table. Mooncake has no session id, so each of its
requests is one-shot even when hash prefixes repeat. A block reaches tier 2
only when two labels have used it, which includes the shared chat-template
prefix across text workloads.

## TFIFO

TFIFO uses the same three tiers, but the order inside a tier is insertion
order. A block receives a serial the first time it becomes cached. Pinning it
and touching it again do not change that serial, and a smaller serial is
evicted first. Tier 0 is emptied before tier 1, and tier 1 before tier 2.
`--hold-s` and `--hold-budget` do not move a block between tiers.

| Tier | Which blocks |
| --- | --- |
| 0 | One-shot labels (`quality`, `code`, `mooncake-agent`, `chat`) and unlabeled blocks |
| 1 | A block whose only label is `agent`, `coding-agent`, `bailian-chat`, or `bailian-coder` |
| 2 | A block seen by two or more workload labels |

## HFIFO and HLRU

HFIFO and HLRU do not read workload labels. A block starts in tier 0. The
first time that block is reused, it moves to tier 1; other blocks of the
same label stay in tier 0. There is no tier 2. Eviction clears the hit, so
a recycled block id starts in tier 0 again.

HFIFO orders each tier by the serial assigned at insertion. A later access
does not change it, and tier 0 is emptied before tier 1. HLRU orders each
tier by `last_accessed`, with the longer prefix broken first on a tie. A
reused block stays out of tier 0 until `--hold-s` has passed or the reused
tier exceeds `--hold-budget` of the free table. After that it competes with
tier 0 by recency.

## RFIFO

RFIFO admits a label from the reuse of its own recent blocks, then keeps
insertion order. Each label remembers the last 100 blocks that have either
been hit or evicted with no hit. Until that window is full, the label's new
blocks stay in tier 0. After that, a hit rate of at least 5% puts new blocks
of that label straight into tier 1. A lower rate leaves them in tier 0, which
is emptied first. A later hit does not move a block that is already cached,
and it does not change its serial. The sweep below also runs windows of
20, 100, and 500 and thresholds of 2%, 5%, and 20%.

## Run Task 2

Same exports as Task 1 (`VLLM_USE_V1=0`, `MODEL_REVISION`, and the thread
limits). `EXP` is the run directory; prepared data, the mix, and the results
are `$EXP/prepared`, `$EXP/mixed`, and `$EXP/runs`. Change `EXP` to point a
second run at another directory. Those three directories must be empty; a
rerun needs a new `EXP` or the old tree removed. Chain the commands with
`&&` so a failed prepare does not start the mix.

ShareGPT has 1000 raw records and 602 eligible conversations at 32768 tokens
(most of the rest are not alternating user/assistant turns), so the chat
limit is 600. The other limits are smaller than Task 1 and weighted toward
multi-turn sessions. Task 1's full per-source limits, mixed into one cache,
make TLRU's free-table scan take many hours. This cohort is meant to finish
in about one to two hours. `--cache-ratio 0.1` is one pressure level on the
mixed working set.

```bash
EXP=experiments/m3-task2
AB=data/m3
python -m vllm.simulator.mixed_prepare \
  --source chat chat data/sharegpt1000.jsonl \
  --source agent chat ${AB}/agentbank_alfworld.parquet,${AB}/agentbank_webshop.parquet,${AB}/agentbank_intercode_bash.parquet,${AB}/agentbank_mind2web.parquet \
  --source coding-agent claude-code ${AB}/cc_bench.parquet \
  --source quality document-qa ${AB}/quality_validation.parquet \
  --source bailian-chat block-trace ${AB}/bailian_traceA.jsonl \
  --source bailian-coder block-trace ${AB}/bailian_coder.jsonl \
  --source mooncake-agent block-trace ${AB}/mooncake_toolagent.jsonl \
  --source code instruction data/code_alpaca.json \
  --set chat limit=600 \
  --set agent limit=100 \
  --set coding-agent limit=50 \
  --set code limit=20 \
  --set quality limit=10 \
  --set bailian-chat limit=20 \
  --set bailian-coder limit=20 \
  --set mooncake-agent limit=40 \
  --set mooncake-agent trace_block_size=512 \
  --set mooncake-agent timestamp_unit=ms \
  --limit 25 --max-model-len 32768 --max-output-tokens 32 \
  --output "$EXP/prepared" \
  --revision "$MODEL_REVISION"

P=$EXP/prepared
python -m vllm.simulator.mix_workload \
  --source chat $P/chat \
  --source agent $P/agent \
  --source coding-agent $P/coding-agent \
  --source quality $P/quality \
  --source code $P/code \
  --source bailian-chat $P/bailian-chat \
  --source bailian-coder $P/bailian-coder \
  --source mooncake-agent $P/mooncake-agent \
  --output "$EXP/mixed" \
  --think-time 1 --request-rate 100 --seed 0

python -m vllm.simulator.mixed_study \
  --workload "$EXP/mixed" \
  --output "$EXP/runs" \
  --cache-ratio 0.1 --block-size 16 \
  --hold-s 8 --hold-budget 1
```

<!-- Previous run used --hold-budget 0.5. -->

`--source` and `--set` mean the same thing as in Task 1. Prepare prints the
kept count per label and stops if a text source yields fewer conversations
than its limit. Block-trace limits are a time-ordered window and may keep
fewer requests. The mix prints conversations and turns per label. The study
runs `tlru`, `lru`, `lfu`, `fifo`, `tfifo`, `hlru`, `hfifo`, and `rfifo` once each, in that order, and replays print
progress every 100 requests. The bar chart sorts run directories, so its
x-axis is alphabetical.

The primary score is the same token-weighted shared fraction as Task 1:

```text
shared_fraction = sum(cached_prompt_tokens) / sum(prompt_tokens)
```

`by_workload.csv` breaks that fraction out by label. `elapsed_s` and
`mean_reuse_gap_s` are wall-clock times and include CPU work, HTTP, and the
synthetic delays (0.01 ms per uncached prefill token, 0.1 ms per decoding
step). They are not measured GPU time. A longer reuse gap means a retained
block was used again after a longer wait; it is not the policy's run time.
The score used here is the shared fraction.

## Outputs

| Output | Contents |
| --- | --- |
| `prepared/LABEL/` | Selected conversations, exclusions, and exact token traces. |
| `mixed/` | Interleaved workload, retagged trace, per-label counts, and source hashes. |
| `runs/POLICY/` | Server log, `run.json` (working set and cache blocks), requests, block events, and `summary.json`. `blocks.jsonl` records `hold_s` and `hold_budget`. |
| `runs/comparison.json`, `comparison.csv` | One row per policy, including the per-label breakdown. |
| `runs/by_workload.csv` | Policy, label, shared fraction, and mean reuse gap. |
| `runs/cache_comparison.png` | Overall shared fraction. CDFs: `shared_fraction_cdf`, `block_hits_cdf`, `reuse_gap_cdf`. |

There is no `ranking.json`. Task 1's ranker only accepts LRU, LFU, and FIFO.

## Results

Two mixes, both with no failures. The skewed mix is the one from the command
above: 3,322 requests, 10,517,812 prompt tokens, and a cache of 12,360 blocks
(10% of 123,593, not clamped). `coding-agent` is 79.2% of those tokens. The
equal-token mix uses the same replay settings after each of the eight labels
is cut to about 260,000 prompt tokens (6,117 requests, 2,083,061 tokens,
8,351 blocks). The plots are
[skewed](assets/m3-task2/cache_comparison_heavy.png) and
[equal-token](assets/m3-task2/cache_comparison_balanced.png). Values are
token-weighted shared fractions, in percent.

| Policy | Skewed | Equal |
| --- | ---: | ---: |
| LRU | 46.1 | 37.8 |
| LFU | 20.4 | 33.3 |
| FIFO | 51.7 | 36.8 |
| TFIFO | **60.4** | 29.5 |
| RFIFO | 58.3 | **38.4** |

Skewed mix, per label. `chat` is 13.2% of tokens; `agent` is 2.5%.

| Workload | LRU | LFU | FIFO | TFIFO | RFIFO |
| --- | ---: | ---: | ---: | ---: | ---: |
| `coding-agent` | 56.2 | 23.7 | 63.2 | **73.2** | 67.7 |
| `agent` | 11.0 | 9.4 | 20.7 | **55.3** | 47.0 |
| `bailian-chat` | 11.5 | 11.6 | 10.0 | **13.2** | 7.0 |
| `bailian-coder` | 16.4 | 16.9 | 16.3 | **17.0** | 16.6 |
| `quality` | **20.6** | **20.6** | **20.6** | 9.3 | 9.3 |
| `mooncake-agent` | 19.8 | **22.4** | 15.8 | 13.7 | 16.4 |
| `chat` | 2.4 | 2.4 | 2.6 | 2.3 | **20.9** |
| `code` | 28.8 | 28.8 | 28.8 | 28.8 | 28.8 |

Equal-token mix, per label. Each label is 12.2–12.8% of tokens.

| Workload | LRU | LFU | FIFO | TFIFO | RFIFO |
| --- | ---: | ---: | ---: | ---: | ---: |
| `coding-agent` | 84.2 | 70.8 | 84.2 | **89.2** | **89.2** |
| `chat` | 63.1 | 33.5 | 61.0 | 2.9 | **63.8** |
| `agent` | 64.6 | 47.6 | 62.7 | **68.7** | 66.6 |
| `bailian-coder` | 5.4 | 11.9 | 5.0 | 20.1 | **24.8** |
| `bailian-chat` | 16.9 | **17.9** | 14.2 | 17.3 | 15.1 |
| `code` | **30.0** | **30.0** | **30.0** | 29.8 | 29.9 |
| `quality` | 27.2 | **36.9** | 27.7 | 5.1 | 15.4 |
| `mooncake-agent` | 9.2 | **16.8** | 8.4 | 0.3 | 0.3 |

Observations:

- TFIFO is highest on the skewed mix and lowest on the equal-token mix. On
  the skewed mix the 8.7 point gain over FIFO is mostly `coding-agent` (73.2
  vs 63.2) and `agent` (55.3 vs 20.7). `chat` is 2.3–2.6 for LRU, LFU, FIFO,
  and TFIFO, so putting `chat` in tier 0 costs almost nothing. On the
  equal-token mix the same rule drops `chat` from 61.0 (FIFO) to 2.9, and
  that label is 12.5% of tokens, so TFIFO falls to 29.5.
- RFIFO is second on the skewed mix (58.3) and first on the equal-token mix
  (38.4). A label enters tier 1 only after 100 resolved blocks and a hit rate
  of at least 5%, in insertion order. On the equal-token mix, `chat` stays at
  63.8 and `coding-agent` ties TFIFO at 89.2. On the skewed mix, `chat` rises
  to 20.9 and `coding-agent` is 67.7 rather than 73.2: once `chat` crosses
  5%, its new blocks occupy tier 1.
- FIFO remains the strongest baseline (51.7 and 36.8). LRU is next on the
  skewed mix (46.1) and close to FIFO on the equal-token mix (37.8 vs 36.8).
  LFU is last on the skewed mix (20.4; `coding-agent` 23.7) because a new
  block starts with the lowest count.
- TLRU, HLRU, and HFIFO are not in the tables. TLRU and HLRU stay next to LRU
  (43.5 and 42.6 on the skewed mix; 38.2 and 38.0 on the equal-token mix).
  HFIFO is 37.8 and 32.9. Waiting for a block's own hit before protecting it
  evicts the prefix that the next turn still needs.
- `code` is 28.8 on the skewed mix for every policy, and about 30 on the
  equal-token mix. The reused part is the shared template. `quality` and
  `mooncake-agent` stay low under both TFIFO and RFIFO when their blocks do
  not clear the tier-1 rule. Replay is about 8 minutes per policy on the
  skewed mix and about 2.5 minutes on the equal-token mix.

## Hold and admission sweep

The two mixes above use one hold (8 s, budget 1) and one RFIFO window
(100 blocks, threshold 5%). This sweep varies both on five mixes. Each mix
is replayed 25 times: LRU, LFU, FIFO, and TFIFO once; RFIFO at windows 20,
100, and 500 and thresholds 0.02, 0.05, and 0.2; TLRU and HLRU at holds of
3, 8, and 30 seconds and budgets 0.25 and 1. HFIFO stays at the two-mix
result above. Cache ratio, block size, think time, request rate, seed, and
concurrency match that run. All 125 replays finished with no failed
requests, and no cache was clamped.

The reported run is `experiments/m3-hold-grid3`. A smaller run,
`experiments/m3-hold-grid2`, uses half of these limits. `coding-agent` stays
at 12 on the protected and balanced mixes in both runs, because the next
CC-Bench sessions exceed 32,768 tokens. Chat is read from `data/sharegpt.json`.
Same exports as Task 1:

```bash
python -m vllm.simulator.hold_grid \
  --output experiments/m3-hold-grid3 \
  --revision "$MODEL_REVISION"
```

Every mix keeps a small background (chat 40, agent 30, coding-agent 6,
quality 10, code 400, bailian-chat 40, bailian-coder 16, mooncake-agent 8)
except where that mix raises the label. Mooncake's limit of 62 keeps 60
requests; two scanned records exceed the context limit.

| Mix | Requests | Prompt tokens | Cache blocks | What is large |
| --- | ---: | ---: | --- | --- |
| chat-heavy | 4,682 | 3,235,776 | 11,474 / 114,738 | `chat` 86.5% (limit 1200; `code` 600) |
| protected-heavy | 1,958 | 2,040,836 | 7,289 / 72,888 | `bailian-coder` 28.6%, `bailian-chat` 24.9%, `agent` 22.7%, `coding-agent` 13.1% |
| trace-heavy | 1,874 | 3,733,235 | 15,287 / 152,868 | `bailian-coder` 43.3%, `bailian-chat` 35.4%, `mooncake-agent` 12.9% |
| template-heavy | 10,248 | 1,469,927 | 7,284 / 72,832 | `code` 36.0%, `quality` 35.1% |
| balanced | 12,029 | 3,849,896 | 15,686 / 156,854 | 6.9–15.2% per label; `coding-agent` is the 6.9% |

Plots:
[chat-heavy](assets/m3-task2/hold_chat_heavy.png),
[protected-heavy](assets/m3-task2/hold_protected_heavy.png),
[trace-heavy](assets/m3-task2/hold_trace_heavy.png),
[template-heavy](assets/m3-task2/hold_template_heavy.png),
and [balanced](assets/m3-task2/hold_balanced.png).
Values are token-weighted shared fractions, in percent. Bold is the best
of that mix.

| Setting | Chat | Protected | Trace | Template | Balanced |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lru` | 13.7 | 37.7 | 25.4 | 33.2 | 37.9 |
| `lfu` | 12.3 | 30.1 | 27.7 | **47.6** | 37.3 |
| `fifo` | 13.1 | 36.5 | 23.1 | 32.2 | 35.9 |
| `tfifo` | 8.4 | 34.8 | 20.3 | 24.5 | 25.5 |
| `rfifo-w20-t0.02` | 18.0 | 30.5 | 19.7 | 37.5 | 31.4 |
| `rfifo-w20-t0.05` | 8.9 | 31.1 | 19.6 | 40.3 | 34.3 |
| `rfifo-w20-t0.2` | 13.7 | 23.6 | 16.7 | 40.7 | 37.3 |
| `rfifo-w100-t0.02` | 21.5 | 35.0 | 21.1 | 40.7 | 35.9 |
| `rfifo-w100-t0.05` | 22.6 | 30.0 | 17.8 | 44.0 | 36.2 |
| `rfifo-w100-t0.2` | 22.7 | 21.3 | 18.0 | 45.6 | 36.8 |
| `rfifo-w500-t0.02` | 21.3 | 31.0 | 16.9 | 42.5 | 30.1 |
| `rfifo-w500-t0.05` | **23.6** | 28.0 | 18.7 | 42.0 | 38.3 |
| `rfifo-w500-t0.2` | 15.2 | 21.7 | 21.5 | 47.4 | 36.6 |
| `tlru-h3-b0.25` | 13.1 | 37.5 | 25.3 | 33.6 | 37.9 |
| `tlru-h3-b1` | 13.0 | 37.4 | 25.5 | 33.7 | 37.9 |
| `tlru-h8-b0.25` | 12.6 | 37.7 | 25.5 | 33.6 | 37.9 |
| `tlru-h8-b1` | 12.9 | 39.1 | 22.8 | 33.6 | 37.9 |
| `tlru-h30-b0.25` | 12.8 | 37.6 | 25.4 | 32.5 | 38.0 |
| `tlru-h30-b1` | 12.7 | 37.3 | 22.0 | 31.7 | 36.6 |
| `hlru-h3-b0.25` | 13.3 | 37.6 | 25.4 | 33.6 | 37.8 |
| `hlru-h3-b1` | 12.9 | 37.7 | 25.4 | 33.6 | 37.8 |
| `hlru-h8-b0.25` | 13.2 | 39.1 | 26.3 | 33.4 | 37.8 |
| `hlru-h8-b1` | 13.3 | **39.3** | 26.5 | 33.6 | 37.7 |
| `hlru-h30-b0.25` | 13.2 | 36.2 | 27.7 | 39.6 | 40.0 |
| `hlru-h30-b1` | 13.4 | 35.8 | **27.8** | 45.0 | **40.1** |

`hlru-h30-b1` has the highest mean of the five scores, 32.4, and the
smallest mean gap to the per-mix best, 3.3. Its worst gap is chat-heavy,
10.2. `hlru-h30-b0.25` averages 31.3 (mean gap 4.3). LFU averages 31.0
(mean gap 4.7). `rfifo-w100-t0.02` averages 30.8 and has the smallest worst
gap in the sweep, 6.9. LRU averages 29.6 (mean gap 6.1). `tlru-h8-b0.25`
averages 29.5 (mean gap 6.2). TFIFO averages 22.7 (mean gap 13.0).

Selected per-label scores on the larger run. Share is that label's fraction
of prompt tokens.

| Mix | Label | Share | LRU | LFU | TFIFO | HLRU h8 b1 | HLRU h30 b1 | TLRU h8 b0.25 | RFIFO w500 t0.05 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chat-heavy | `chat` | 86.5 | 13.3 | 11.7 | 4.3 | 12.9 | 12.9 | 12.1 | 26.1 |
| protected-heavy | `coding-agent` | 13.1 | 94.2 | 67.0 | 94.4 | 94.0 | 86.6 | 94.2 | 48.7 |
| protected-heavy | `agent` | 22.7 | 61.3 | 42.0 | 65.8 | 61.5 | 53.5 | 60.8 | 56.7 |
| protected-heavy | `bailian-coder` | 28.6 | 11.4 | 15.6 | 8.7 | 14.9 | 14.5 | 11.6 | 10.2 |
| trace-heavy | `bailian-coder` | 43.3 | 20.8 | 24.4 | 20.1 | 22.6 | 24.6 | 20.8 | 19.0 |
| trace-heavy | `mooncake-agent` | 12.9 | 24.2 | 28.5 | 1.5 | 26.2 | 28.5 | 24.2 | 19.5 |
| template-heavy | `quality` | 35.1 | 35.7 | 76.1 | 12.4 | 35.7 | 68.7 | 35.6 | 79.2 |
| template-heavy | `code` | 36.0 | 29.9 | 30.0 | 29.7 | 29.9 | 29.9 | 29.9 | 29.7 |
| balanced | `quality` | 13.4 | 37.3 | 69.2 | 3.5 | 37.3 | 47.4 | 37.3 | 58.8 |
| balanced | `chat` | 12.9 | 55.8 | 28.5 | 3.0 | 55.1 | 54.8 | 55.6 | 62.9 |
| balanced | `coding-agent` | 6.9 | 86.6 | 74.4 | 87.7 | 85.4 | 86.9 | 86.6 | 50.6 |

The smaller run has about half the tokens: chat-heavy 1,598,533,
protected-heavy 1,164,767, trace-heavy 1,746,870, template-heavy 722,573,
balanced 2,072,368. Cache blocks are 5,928, 3,853, 8,063, 3,880, and 8,323.
`coding-agent` is 23.0% of protected tokens and 12.9% of balanced tokens.
Scores below are from that run. A percent here and the same cell above are
separate measurements: each cache is 10% of its own working set, and the
protected and balanced mixes have different label shares.

| Setting | Chat | Protected | Trace | Template | Balanced |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lru` | 11.0 | 43.2 | 17.2 | 30.0 | 38.6 |
| `lfu` | 11.2 | 30.1 | **21.6** | **37.1** | 31.4 |
| `hlru-h8-b0.25` | 11.0 | 43.6 | 19.6 | 30.0 | 39.5 |
| `hlru-h30-b0.25` | 11.1 | 39.5 | 21.2 | 31.1 | 40.4 |
| `hlru-h30-b1` | 11.3 | 31.3 | **21.6** | 34.7 | **40.5** |
| `tlru-h8-b0.25` | 11.4 | 44.7 | 17.3 | 31.0 | 38.8 |
| `tlru-h8-b1` | 11.3 | **45.7** | 14.2 | 29.9 | 39.0 |
| `tlru-h30-b0.25` | 12.0 | 42.0 | 17.2 | 31.6 | 38.8 |
| `tlru-h30-b1` | 11.9 | **45.7** | 14.4 | 27.5 | 33.8 |
| `rfifo-w500-t0.02` | **21.1** | 29.5 | 11.4 | 34.5 | 25.2 |

Observations:

- On the larger run, HLRU hold 30 s and budget 1 is first on trace-heavy
  (27.8) and balanced (40.1), and 45.0 on template-heavy, 2.6 behind LFU.
  Protected-heavy is 35.8, 3.5 behind `hlru-h8-b1`. `coding-agent` there
  falls from 94.0 at hold 8 to 86.6 at hold 30, and that label is 13.1% of
  the tokens.
- Hold 8 barely moves with the budget. HLRU hold 8 is 39.3 at budget 1 and
  39.1 at budget 0.25 on protected-heavy, and 33.6 against 33.4 on
  template-heavy. Hold 30 does move: template-heavy falls from 45.0 to 39.6
  at budget 0.25, and `quality` falls from 68.7 to 53.2.
- TLRU stays next to LRU. `tlru-h8-b0.25` averages 29.5 against LRU's 29.6.
  Budget 1 at hold 8 or 30 raises protected-heavy to 39.1 and 37.3, and
  drops trace-heavy to 22.8 and 22.0, because `mooncake-agent` falls from
  24.2 to 1.4 and 1.2. Mooncake is a tier-0 label.
- Chat-heavy moves under RFIFO. `chat` is 86.5% of tokens and is not a TLRU
  label, so LRU, HLRU, and TLRU stay at 12.6–13.7. Window 500 and threshold
  0.05 reaches 23.6, with `chat` at 26.1. Window 20 and threshold 0.05 stays
  at 8.9.
- LFU wins template-heavy through `quality` (76.1, against LRU 35.7). The
  same count order puts protected-heavy at 30.1 (`coding-agent` 67.0,
  `agent` 42.0). On trace-heavy LFU is 27.7, 0.1 behind HLRU hold 30
  budget 1, from `bailian-coder` (24.4 vs LRU 20.8) and `mooncake-agent`
  (28.5 vs 24.2).
- RFIFO's window and threshold change which labels are admitted. On
  protected-heavy, window 500 and threshold 0.05 puts `coding-agent` at 48.7
  (LRU 94.2) and the mix at 28.0. Window 100 and threshold 0.02 keeps
  `coding-agent` at 94.6 and the mix at 35.0. Window 100 and threshold 0.2
  is the low end of that mix, 21.3. The window fills while the label's
  blocks are still in tier 0, so an early miss streak leaves later blocks
  in tier 0.
- TFIFO is the low end on chat-heavy (8.4), trace-heavy (20.3),
  template-heavy (24.5), and balanced (25.5). `chat` on chat-heavy is 4.3.
  `code` stays near 30 on every mix and setting. The reused tokens are the
  shared template.
- On the smaller run the high region is budget 0.25. HLRU hold 8 and hold
  30 at that budget both average 28.7 (mean gaps 4.5). TLRU hold 8 budget
  0.25 averages 28.6 and scores 44.7 on protected-heavy. TLRU hold 30 budget
  0.25 has the smallest worst gap among those rows, 9.1, at mean 28.3.
  HLRU hold 30 budget 1 is first on balanced and tied with LFU on trace, and
  scores 31.3 on protected-heavy, 14.4 behind TLRU, with `coding-agent` at
  56.3. Doubling the other labels cuts that label from 23.0% to 13.1% of
  protected tokens, so the hold-30 drop costs 3.5 points on the larger run.
  Chat-heavy's RFIFO winner also moves, from window 500 threshold 0.02
  (21.1) to window 500 threshold 0.05 (23.6).

## Limitations

- Dataset limits from Task 1 still apply: missing agent system prompts, CC-Bench
  truncated at the context limit, synthetic trace outputs, and Mooncake's
  512-token hash blocks.
- The two-mix cohort is the first eligible records, at the limits in Run
  Task 2, and one arrival seed. Bailian and Mooncake are short windows, not
  their full traces. The sweep uses the same seed and the limits in the
  sweep section.
- TLRU compares `--hold-s` with wall-clock time. A prefix that waits longer
  than the hold is evicted with the one-shot tier. TFIFO does not use the
  hold. On the skewed mix the median dispatch lag is about 150 s, far past
  the 8 s hold.
- Tier 2 is "two labels," so the shared chat-template prefix is protected as a
  shared prefix even when the rest of a request is one-shot.
- Text arrivals are Poisson at 100 requests/s with concurrency 8, which
  saturates the server the same way Task 1 does. Every policy sees the same
  requests.
- The sweep cannot raise `coding-agent` past 12. The larger protected and
  balanced mixes therefore give that label a smaller share than the
  half-size run (13.1% and 6.9%, against 23.0% and 12.9%).
- Shared fractions from the two sweep scales are separate measurements. Each
  cache is 10% of that mix's working set, and the label shares differ.

## Validation

```bash
python -m pytest tests/simulator/test_mix_workload.py tests/simulator/test_eviction_policies.py tests/simulator/test_hold_grid.py -q
```

The mix tests check unique retagged ids for chat and block traces, stable
order and arrivals for a seed, and `cmpl-` trace ids. The hold-grid tests
check five mixes, 25 settings, 125 runs, and a `coding-agent` limit of at
most 12. The policy tests check
that an expired multi-turn block goes before a fresh one, that the budget
keeps the newest protected blocks, that a one-shot block goes before a
protected one, that `CacheConfig` accepts `tlru`, `tfifo`, `hfifo`, `hlru`, and `rfifo`, and that
TFIFO evicts a later one-shot before an older protected block without
changing insertion order on a pin or an access. HFIFO and HLRU tests check
that an unhit block goes before a hit block of any label, and that eviction
clears the hit.
