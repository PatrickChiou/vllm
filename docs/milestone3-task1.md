# Milestone 3 Task 1: Eviction Policies for Separate Workloads

This extension evaluates LRU, LFU, and FIFO separately on eight workloads,
each with a different prefix-reuse pattern. It uses the existing v0 online
simulator. It does not combine workloads or introduce a new eviction policy.

| Label | Source | Format | Reuse pattern |
| --- | --- | --- | --- |
| `chat` | ShareGPT | `chat` | Multi-turn human chat; each turn extends its history |
| `agent` | AgentBank (ALFWorld, WebShop, InterCode-Bash, Mind2Web) | `chat` | Short agent thought/action/observation loops |
| `coding-agent` | CC-Bench trajectories | `claude-code` | Long Claude Code sessions with tool calls |
| `quality` | QuALITY | `document-qa` | Many independent questions over the same long article (RAG-like) |
| `bailian-chat` | Qwen Bailian to-C trace A | `block-trace` | Production chat traffic, recorded arrivals |
| `bailian-coder` | Qwen Bailian coder trace | `block-trace` | Production code-generation traffic, recorded arrivals |
| `mooncake-agent` | Mooncake tool-agent trace | `block-trace` | Production tool-agent traffic from a second provider |
| `code` | Code Alpaca | `instruction` | Single-turn control with no reuse beyond the template |

AgentBank, CC-Bench and the Bailian traces are the assignment's suggested
workloads; QuALITY and Mooncake add shared-document reuse and an independent
production trace.

## Setup

Use the simulator environment in [the project guide](comp536.md). In particular,
use Python 3.11, PyTorch 2.5.1, Transformers 4.47.1, an editable source install
with `VLLM_TARGET_DEVICE=empty`, and `VLLM_USE_V1=0`. No model weights or GPU are
needed. The first run downloads model configuration and tokenizer metadata.

```bash
export VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
export VLLM_RPC_BASE_PATH=/tmp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
```

Install `pytest`, `pytest-asyncio`, and `requirements-simulator.txt` if they are
not already present. `pyarrow` reads the Parquet datasets; `matplotlib` is
needed unless `--no-plots` is supplied. Activate the environment so subprocess
tests can find the `vllm` executable.

## Download the input data

The ShareGPT command downloads the first 1000 raw records rather than the full
dataset; preparation later selects eligible conversations. The Bailian files
are Git LFS objects; their SHA-256 hashes must equal the `oid` values in the
repository's LFS pointers (`07cedc9e…` for trace A, `3d74974c…` for coder).

```bash
mkdir -p data/m3
python -m vllm.simulator.study sample --limit 1000 \
  --output data/sharegpt1000.jsonl
HF=https://huggingface.co/datasets
for task in alfworld webshop intercode_bash mind2web; do
  curl --fail --location -o data/m3/agentbank_$task.parquet \
    $HF/Solaris99/AgentBank/resolve/main/$task/train-00000-of-00001.parquet
done
curl --fail --location -o data/m3/cc_bench.parquet \
  $HF/zai-org/CC-Bench-trajectories/resolve/main/train.parquet
curl --fail --location -o data/m3/quality_validation.parquet \
  $HF/emozilla/quality/resolve/main/data/validation-00000-of-00001-77baeb9538209706.parquet
LFS=https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/main
curl --fail --location -o data/m3/bailian_traceA.jsonl \
  $LFS/qwen_traceA_blksz_16.jsonl
curl --fail --location -o data/m3/bailian_coder.jsonl \
  $LFS/qwen_coder_blksz_16.jsonl
curl --fail --location -o data/m3/mooncake_toolagent.jsonl \
  https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
curl --fail --location -o data/code_alpaca.json \
  https://raw.githubusercontent.com/sahil280114/codealpaca/0d265112c705f88848783dfdf9bdca9df64d8e35/data/code_alpaca_20k.json
```

Source repositories and dataset documentation:

- [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [AgentBank](https://huggingface.co/datasets/Solaris99/AgentBank)
- [CC-Bench trajectories](https://huggingface.co/datasets/zai-org/CC-Bench-trajectories)
- [QuALITY](https://huggingface.co/datasets/emozilla/quality)
- [Qwen Bailian usage traces](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)
- [Mooncake traces](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces)
- [Code Alpaca](https://github.com/sahil280114/codealpaca)

Respect each source's data terms when downloading or redistributing records.
The runner records a SHA-256 hash of each local input file in `sources.json`.

## Run Task 1

The command below sizes each workload's cache to 25% of its working set and
runs 4 text workloads x 3 policies x 3 seeds, `coding-agent` x 3 policies x 2
seeds, and 3 production traces x 3 policies x 1 seed = 51 configurations,
about 30 minutes on a laptop. The traces use one seed because recorded
arrivals make further seeds identical repeats; `coding-agent` is the slowest
text workload and its two seeds already agree. `--seeds 0` gives a quicker
check.
Task 1 only requires one fixed cache size; for a second pressure level, repeat
with e.g. `--cache-ratio 0.5` and another output directory.

```bash
AB=data/m3/agentbank
python -m vllm.simulator.eviction_study \
  --source chat chat data/sharegpt1000.jsonl \
  --source agent chat ${AB}_alfworld.parquet,${AB}_webshop.parquet,${AB}_intercode_bash.parquet,${AB}_mind2web.parquet \
  --source coding-agent claude-code data/m3/cc_bench.parquet \
  --source quality document-qa data/m3/quality_validation.parquet \
  --source bailian-chat block-trace data/m3/bailian_traceA.jsonl \
  --source bailian-coder block-trace data/m3/bailian_coder.jsonl \
  --source mooncake-agent block-trace data/m3/mooncake_toolagent.jsonl \
  --source code instruction data/code_alpaca.json \
  --set chat max_model_len=2048 --set agent max_model_len=2048 \
  --set coding-agent max_model_len=8192 --set coding-agent limit=50 \
  --set coding-agent seeds=0,1 \
  --set quality max_model_len=8192 --set quality limit=300 \
  --set bailian-chat max_model_len=32768 --set bailian-chat limit=1000 \
  --set bailian-chat time_scale=0.25 --set bailian-chat seeds=0 \
  --set bailian-coder max_model_len=32768 --set bailian-coder limit=1000 \
  --set bailian-coder time_scale=0.25 --set bailian-coder seeds=0 \
  --set mooncake-agent max_model_len=32768 --set mooncake-agent limit=1000 \
  --set mooncake-agent time_scale=0.5 --set mooncake-agent seeds=0 \
  --set mooncake-agent trace_block_size=512 \
  --set mooncake-agent timestamp_unit=ms \
  --max-output-tokens 32 \
  --output experiments/m3-task1 \
  --revision "$MODEL_REVISION" \
  --limit 100 --cache-ratio 0.25 --block-size 16 \
  --request-rate 100 --concurrency 8 --seeds 0 1 2
```

Each `--source` has three arguments: label, format, and local path. Chat
sources may list several comma-separated files; they are interleaved
round-robin so that a record limit samples each file. `--set LABEL KEY=VALUE`
overrides `limit`, `max_model_len`, `max_output_tokens`, `time_scale`,
`overflow`, `seeds` (comma-separated), `trace_block_size` or `timestamp_unit`
(`s` or `ms`) for one source. Supported formats are:

| Format | Input schema | Interpretation |
| --- | --- | --- |
| `chat` | ShareGPT `conversations` or alternating OpenAI `messages`, JSON/JSONL/Parquet | Retains all complete turns and their causal ordering. |
| `instruction` | String `instruction`, optional string `input`, string `output` | A single user/assistant exchange. Nonempty input follows the instruction after two newlines. |
| `claude-code` | CC-Bench records with a Claude Code `trajectory` event list | Main agent thread only; tool calls and results become text turns. |
| `document-qa` | QuALITY records: `article`, `question`, `options`, integer `answer` | One single-turn request per question: the article, the question and lettered options; the reply is the correct option. |
| `block-trace` | JSONL with `timestamp`, `input_length`, `output_length`, `hash_ids`; optional `chat_id`, `parent_chat_id` (Bailian) | Recorded arrivals; requests linked by `parent_chat_id` replay in causal order. |

`document-qa` shuffles the questions of each group of 10 consecutive articles
with a fixed seed. QuALITY lists an article's questions consecutively; replayed
in that order, every policy would keep the one active article, so interleaving
is what makes the comparison informative. `--limit` counts questions, so 300
covers the first groups.

Text formats are rendered with the model's chat template, with no fabricated
shared system prompt. A conversation exceeding the context limit is excluded
(`overflow=reject`), except `claude-code`, which keeps the turns before the
first turn that does not fit (`overflow=truncate`); counts are recorded in each
prepared workload.

### Block-hash traces

Production-trace prompts are anonymized: each block of tokens (16 for Bailian,
512 for Mooncake, set by `trace_block_size`) is a hash id. The adapter maps
every hash id to a fixed, deterministic sequence of ordinary token ids, so
requests share a token prefix exactly when they share a hash prefix, and
replays them through `/v1/completions` with token-id prompts (the hashes were
taken after the chat template was applied, so no template is added). The last
block is truncated to `input_length`. Mooncake records have no session fields,
so each request is its own conversation, and its timestamps are milliseconds
(`timestamp_unit=ms`). `--limit` counts requests: the first `limit` requests
form a contiguous time window. Generated tokens are a repeated filler capped
by `max_output_tokens`; they occupy KV cache during decoding, but later prompts
carry recorded hashes, so they are never reused. Seeds do not change recorded
arrivals, so they would only repeat the run.

Recorded arrivals are only meaningful if the simulated server keeps up, so
check `dispatch_lag_s` in `requests.jsonl`. With 128 generated tokens, Bailian
at `time_scale=0.25` (4x faster) fell behind by a median of 12 s. Capping
outputs at 32 tokens (`--max-output-tokens 32`) brought the median lag to about
0.3 s for Bailian at 4x and 0.55 s for Mooncake at 2x (`time_scale=0.5`), whose
prompts are longer. The cap only affects the block traces.

To report cached prompt tokens for these requests, the completions endpoint
now fills `usage.prompt_tokens_details.cached_tokens` when
`--enable-prompt-tokens-details` is set, as the chat endpoint already did.

### Cache sizing

`--cache-ratio R` sizes each workload's cache to `R` times its working set:
the distinct full KV blocks its requests would cache with unlimited capacity
(identified by their whole prefix, prompt plus generated tokens, excluding the
final EOS). Workloads differ by orders of magnitude in footprint, so a single
absolute size would leave some workloads without any eviction and others
thrashing; a common ratio applies the same relative pressure. A cache is
raised to the smallest size that admits the longest request; such runs are
marked `cache_clamped`. Without `--cache-ratio`, `--cache-gb` sets one capacity
for every workload as before. Block size is fixed for all runs, and capacity
is fixed within a workload, as Task 1 permits.

Each workload/seed is prepared once and replayed under all three policies.
Trace and workload hashes must match across policies. Every run starts a fresh
online simulator with strict trace matching and prefix caching. Policy execution
order is shuffled reproducibly per seed. This changes host execution order,
not the input dataset order. Seeds change Poisson desired arrivals for text
workloads; subsequent conversation turns also wait for their preceding replies.
Wall-clock scheduling means repeated runs are not bit-for-bit deterministic.

## Outputs and interpretation

| Output | Contents |
| --- | --- |
| `experiment.json`, `sources.json` | Arguments, per-source settings, Python/platform information, input hashes and conversion counts. |
| `prepared/LABEL/seed-N/` | Selected conversations, exclusions, desired arrivals, and exact token traces. |
| `LABEL/seed-N/POLICY/` | Server log, settings (including working set and cache blocks), raw requests/block events, summary, request/block-hit/reuse-gap CSV files. |
| `comparison.json`, `comparison.csv` | Every workload/policy/seed measurement. |
| `ranking.json`, `ranking.csv` | Per-workload mean sharing fraction and sample standard deviation across seeds; exact ties remain ties. |
| `policy_comparison.png`, `.pdf` | Mean sharing fraction with seed standard-deviation bars. |

The primary score for a run is:

```text
shared_fraction = sum(cached_prompt_tokens) / sum(prompt_tokens)
```

Policies are ranked by the arithmetic mean of this token-weighted score across
arrival seeds. Ranking refuses failed or empty runs, missing policies, different
seed sets, duplicate runs, a block size or (within a workload) a cache capacity
that varies, or mismatched request/trace cohorts. `observed_best` means highest
measured mean under the tested settings; it is not a proof of universal
optimality or significance. No forced winner is reported when scores tie.

For per-run distribution figures, the existing analyzer can be used on one
workload/seed directory:

```bash
python -m vllm.simulator.study analyze \
  --input experiments/m3-task1/chat/seed-0
```

Block hits count distinct requests skipping a logical full prefix block.
Reuse gaps are observed only at successful cache hits; they do not describe
all potential reuse after evictions. LFU's internal counter counts allocator
reuse operations, which differ from these request-level block hits.

The default delays (0.01 ms per uncached prefill token and 0.1 ms per decoding
sequence/step) are synthetic. Wall-clock throughput and reuse gaps include CPU
and HTTP overhead; they are not measured GPU performance. Inspect dispatch lag
and failures alongside cache scores. Text workloads use a Poisson rate of 100
requests/s, which exceeds what the server sustains, so they deliberately
replay as fast as 8 concurrent requests allow; their dispatch lag reflects
this saturation, and every policy receives the same requests.

## Results

One run of the command above (51 configurations, no failed requests, 30
minutes on a macOS laptop). The plot is in
[`docs/assets/m3-task1/`](assets/m3-task1/). Values are mean shared fractions across
seeds (± sample standard deviation where several seeds were run); the bound is
the shared fraction with unlimited cache, computed offline from the same trace.

| Workload | LRU | LFU | FIFO | Observed best | Bound |
| --- | --- | --- | --- | --- | --- |
| `coding-agent` | **47.6 ± 0.2** | 37.5 ± 0.5 | 46.5 ± 0.8 | LRU | 95.8 |
| `quality` | 49.9 ± 0.4 | 44.9 ± 0.4 | **50.6 ± 1.6** | FIFO (≈ LRU); LFU worst | 92.0 |
| `bailian-chat` | 28.4 | **29.5** | 26.7 | LFU | 38.5 |
| `bailian-coder` | 35.8 | **37.3** | 32.4 | LFU | 43.6 |
| `agent` | 50.3 ± 0.2 | **50.8 ± 0.4** | 49.5 ± 0.2 | LFU (≈ LRU) | 77.8 |
| `mooncake-agent` | 40.1 | **40.4** | 39.5 | near tie | 47.5 |
| `chat` | 18.8 ± 0.7 | 18.8 ± 0.4 | **19.1 ± 0.7** | tie within noise | 83.5 |
| `code` | 29.5 | 29.5 | 29.5 | tie | 29.5 |

Observations (pilot scale; see Limitations):

- LFU loses most where new content is reused soon after it is first cached.
  A new block starts with the lowest count, so LFU evicts it first, while
  blocks that were popular earlier keep high counts after they stop being
  used. In `coding-agent`, the newest turn of an active session is evicted
  before the next turn arrives (LFU is 10 points below LRU). In `quality`
  (seed 0 of an earlier run), LFU is already worst on the first group of
  articles (47.7% vs 51.0% LRU) and falls further once the next group starts
  (40.0% vs 46.5%), when the previous group's articles are no longer asked
  about. FIFO and LRU are within seed noise on `quality`.
- LFU wins on both Bailian traces, where many requests begin with the same
  blocks (e.g. shared system prompts). It keeps these hot prefixes across
  bursts of one-off requests; on the coder trace its hits have the longest
  mean reuse gap (3.2 s vs 2.3 s for LRU in an earlier run), i.e. it retains
  content that returns later.
- When reuse follows closely, the policy matters little: the Mooncake tool
  agent's mean reuse gap is under 1 s and all policies are within 0.9 points.
  AgentBank policies are within 1.3 points, and ShareGPT policies are within
  seed noise.
- With no reuse beyond the template's system prompt (`code`), all policies
  reach the bound and tie, so the eviction policy only matters when a
  workload has reuse that does not fit in the cache.

In an earlier run with 128 generated tokens per trace request, Bailian
dispatch lagged by a median of 12 s; the rankings of the workloads run in both
experiments were unchanged (`coding-agent`: LRU > FIFO > LFU; Bailian traces:
LFU > LRU > FIFO; `agent`: LFU > LRU > FIFO). In this run the median dispatch
lag is 0.001 s for `bailian-chat`, 0.7 s for `bailian-coder` and 0.5 s for
`mooncake-agent`.

## Limitations

- AgentBank and CC-Bench records do not include the agents' original system
  prompts or tool schemas, which dominate cross-session reuse in deployed
  agents. Measured agent reuse therefore comes from each trajectory's own
  growing context and underestimates shared-prompt reuse.
- CC-Bench thinking blocks and images are dropped, sub-agent sidechains are
  excluded, and trajectories are cut at 8192 tokens (Claude Code contexts reach
  far beyond the 0.5B model's context).
- Production-trace outputs are synthetic fillers capped at 32 tokens, and each
  replay is a 1000-request window (about 4 minutes of Bailian, 3 minutes of
  Mooncake traffic) replayed faster than recorded, not the full traces.
  Mooncake's 512-token hash blocks hide any sharing shorter than 512 tokens.
- QuALITY's interleaving (groups of 10 articles) is a modeling choice; real
  retrieval traffic may revisit documents at other intervals.
- Cohorts are the first eligible records of each source; general conclusions
  need other subsets, more seeds, and other arrival rates.

## Validation

```bash
python -m pytest tests/simulator -q
ruff check vllm/simulator/eviction_study.py tests/simulator/test_eviction_study.py
```

The tests check text, Claude Code and document-QA conversion, round-robin
interleaving, truncation, block-hash tokens, session linking and millisecond
timestamps, working-set and cache
sizing, ranking, ties, seed variability, and rejection of unfair or failed
comparisons. An online test replays token-id prompts through
`/v1/completions` and checks the reported cached tokens. As an end-to-end
check, a Bailian window replayed with a cache holding its whole working set
reported exactly the cached tokens computed offline from the hash ids
(79,264 of 332,138 prompt tokens for the first 200 trace A requests).
