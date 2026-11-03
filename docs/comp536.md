# COMP536: Milestone 2 and 3 experiment guide

Here's how to run our Milestone 2 experiments. Start from the repository root
on `comp536-sharing`. If you haven't set up Milestone 1, the steps below will
get you started. We're still working toward the [Milestone 3 workflow](#milestone-3-status).

## Environment

I used **Python 3.11.9 and Transformers 4.47.1** in my local
environment. Feel free to use the same setup to make things easier 😀  

If you use different versions, just note them alongside your results so we can
compare runs.

### Starting from scratch

These commands are for macOS or Linux; use WSL if you're on Windows. With
Python 3.11 installed, let's create an environment first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

There's no single `requirements.txt` in this checkout. We need the build and
shared Python dependencies as well as the simulator's plotting dependency:

```bash
python -m pip install \
  -r requirements-build.txt \
  -r requirements-common.txt \
  -r requirements-simulator.txt \
  "transformers==4.47.1"

VLLM_TARGET_DEVICE=empty python -m pip install -e . \
  --no-build-isolation --no-deps
```

`requirements-build.txt` pins PyTorch to 2.5.1. The `empty` build skips compiled
GPU extensions, and `-e .` installs our checkout so the simulator changes are
available.


### Settings for either setup

```bash
python --version
export VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
python -m vllm.simulator.study --help
```

You won't need model weights or a GPU for this. The simulator just needs the
model configuration and tokenizer, which it can download from Hugging Face on
the first run. Once those are cached, you can turn on offline mode to skip
repeated metadata requests:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Milestone 2 workflow

### 1. Download the dataset

You can use the pinned ShareGPT JSON array or a JSONL subset. To download the
full source (about 673 MB) and set up the paths:

```bash
mkdir -p data
curl --fail --location \
  --output data/sharegpt.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab21/ShareGPT_V3_unfiltered_cleaned_split.json

export DATASET=data/sharegpt.json
export OUT=experiments/sharegpt-m2-seed0
```

Point `DATASET` at your chosen dataset and give `OUT` a fresh name for each run.
If you already have a JSONL subset, you can skip the download and use that path
instead. Dataset files and experiment outputs aren't bundled with a fresh
checkout.

A small detail about `--limit`: it takes the first N **eligible conversations**,
not N requests. Preparation skips malformed conversations and any conversation
with a turn exceeding `--max-model-len` (prompt + response + EOS). Both modes
use the same filter, so we're comparing the same starting conversations.
You can find the exclusions in the prepared workload. Also, `--seed` controls
arrival times; it doesn't shuffle the input dataset.

### 2. Run the cache sweep

Once your dataset is ready, here's the sweep. Give it a new or empty output
directory each time; it can't resume a partial run yet.

```bash
python -m vllm.simulator.study sweep \
  --dataset "$DATASET" --output "$OUT" \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --limit 200 --max-model-len 4096 \
  --arrival poisson --request-rate 100 --seed 0 \
  --think-time 0 --time-scale 1 \
  --modes single multi \
  --cache-gb 0.05 0.10 --block-sizes 8 16 32 \
  --policies lru lfu fifo \
  --concurrency 64 --max-num-seqs 64 \
  --prefill-ms 0.01 --decode-ms 0.1 \
  --startup-timeout 120 --no-plots
```

This runs **36 configurations**: 2 modes × 2 capacities × 3 block sizes × 3
policies. Each configuration starts a fresh online server with prefix caching,
strict trace matching, and the selected eviction policy. The sweep prepares
traces, replays requests, checks responses, and exports measurements.

With `--limit 200`, we're studying the first 200 eligible conversations in the
input file. For a random subset, sample the input separately before the sweep.
To run the whole dataset, point `DATASET` at the full
source and raise the limit enough to cover all eligible conversations. For our
comparisons, run a few seeds and give each one its own output directory.

### 3. Generate figures and inspect outputs

The sweep already saves CSV/JSON with `--no-plots`. When you're ready to look
at the figures, run:

```bash
python -m vllm.simulator.study analyze --input "$OUT"
```

You'll get sharing-fraction, block-hit, and reuse-interval CDFs, plus a cache
comparison bar chart, all in PNG and PDF. If you just want to regenerate the
tables, use:

```bash
python -m vllm.simulator.study analyze --input "$OUT" --no-plots
```

| Location under the output directory | Contents |
| --- | --- |
| `comparison.csv`, `comparison.json` | Configuration settings, sharing, failures, block statistics, throughput. |
| `prepared/{single,multi}/workload.json`, `trace.jsonl` | Selected conversations, arrivals, exclusions, tokenizer/template settings, replay tokens. |
| Each configuration's `requests.csv`, `requests.jsonl` | Per-request success, cached/prompt tokens, sharing, dispatch lag, latency. |
| Each configuration's `block_hits.csv`, `reuse_gaps.csv` | Per-block hit distribution and hit-associated reuse intervals. |
| Each configuration's `run.json`, `summary.json`, `blocks.jsonl`, `server.log` | Cache geometry, summary metrics, raw block events, server diagnostics. |

Before comparing results, take a look at `failed_requests` in the comparison
table. The client checks response text, prompt/response token counts, and stop
reason. If a turn fails, its later turns are skipped and the sweep stops with
an error.

## Settings and measurement notes

| Setting | Description |
| --- | --- |
| `--cache-gb` | Virtual KV capacity in GiB, rounded to whole blocks. Must hold the longest selected request. Use this flag to vary simulator capacity. |
| `--block-sizes` | Tokens per block; capacity in blocks changes with block size. Compare policies at fixed capacity and block size. |
| `--policies` | `lru`, `lfu`, or `fifo`. LFU frequency and FIFO insertion history survive pin/release and reset on eviction. |
| `--arrival poisson --request-rate R --seed S` | Seeded desired arrivals at mean rate R. Actual sends can be delayed by concurrency and previous turns. First-turn desired times match between modes. |
| `--arrival recorded` | Requires timestamps on all selected user turns: numeric seconds or timezone-aware ISO 8601. Times must be nondecreasing within a conversation. |
| `--think-time`, `--time-scale` | Later turns wait for their previous reply; think time adds a delay. Time scale zero is suitable only for functional checks. |
| `--prefill-ms`, `--decode-ms` | Synthetic delays per uncached prefill token and per decoding sequence/step. The example's coefficients are not calibrated GPU timings. |

When reading the results, sharing fraction is cached prompt tokens divided by
prompt length. The **token-weighted** summary is total cached tokens divided by total prompt
tokens; the mean per-request fraction is a separate metric.

Block hits count distinct requests skipping a logical full prefix block during
initial prefill, including zero-hit blocks in the distribution. Block identity
includes the preceding prefix and aggregates across cache residencies. Reuse
intervals measure monotonic wall time since the previous distinct request used
the block, sampled at hits. They include CPU execution and synthetic delays.
LFU allocator frequency differs from these measured block hits.

If you share a run with the team, it'd be handy to jot down which conversations you
used and which got skipped, the model/tokenizer revision, context limit, arrival
settings, cache size/block size, and dispatch lag. That'll make it easier for others
to compare policies using the same workload and settings 🙂‍↕️🙂‍↕️

You might see slightly different results when rerunning the same setup, since scheduling uses wall
time.

## Milestone 3 status

For now, we have the ShareGPT workflow and LRU/LFU/FIFO comparisons for
**Milestone 2**.

We don't have the Milestone 3 workload generation or mixed-policy workflow
implemented yet. We can use the sweep above to collect baseline results in the
meantime, then add the Milestone 3 commands here once they're ready.

## Tests and troubleshooting

If you'd like to check the simulator tests, run:

```bash
python -m pip install pytest pytest-asyncio
VLLM_RPC_BASE_PATH=/tmp python -m pytest tests/simulator -q
```

I got 86 passing tests on commit `54caccf7d08371fae82db6520a73b607560a9dc9`.
Hope things work just as well on your machine 😎  

If you get stuck, here are a few things to check:

- **Model files unavailable offline:** unset `HF_HUB_OFFLINE` and
  `TRANSFORMERS_OFFLINE`, cache the pinned configuration/tokenizer, then retry.
- **Cache cannot hold a request:** increase `--cache-gb` or choose a smaller
  context/cohort. Preserve a common cohort across the comparisons.
- **macOS multiprocessing socket path too long:** use
  `VLLM_RPC_BASE_PATH=/tmp`. The sweep itself uses the direct asynchronous engine.
- **macOS `psutil.AccessDenied` on shutdown:** this can happen during SIGTERM
  port inspection. Check request failures and final simulator
  statistics to distinguish shutdown errors from replay errors.
