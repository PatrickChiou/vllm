# Milestone 3 Task 1: Eviction Policies for Separate Workloads

This extension evaluates LRU, LFU, and FIFO separately on conversational chat,
coding instructions, and general instructions. It uses the existing v0 online
simulator. It does not combine workloads or introduce a new eviction policy.

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
not already present. Activate the environment so subprocess tests can find the
`vllm` executable. `matplotlib` is needed unless `--no-plots` is supplied.

## Download the input data

These URLs pin the source data versions. The ShareGPT command downloads the
first 100 raw records rather than the full dataset; preparation later selects
eligible conversations. Increase the sample size for larger experiments.

```bash
mkdir -p data
python -m vllm.simulator.study sample --limit 100 \
  --output data/sharegpt.jsonl
curl --fail --location \
  https://raw.githubusercontent.com/sahil280114/codealpaca/0d265112c705f88848783dfdf9bdca9df64d8e35/data/code_alpaca_20k.json \
  --output data/code_alpaca.json
curl --fail --location \
  https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/f13496211289def0ff88ae673389ae14a818b4b3/alpaca_data.json \
  --output data/alpaca.json
```

Source repositories and dataset documentation:

- [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
- [Code Alpaca](https://github.com/sahil280114/codealpaca)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

Respect each source's data terms when downloading or redistributing records.
The runner records a SHA-256 hash of each local input file in `sources.json`.

## Run Task 1

```bash
python -m vllm.simulator.eviction_study \
  --source chat chat data/sharegpt.jsonl \
  --source code instruction data/code_alpaca.json \
  --source instruction instruction data/alpaca.json \
  --output experiments/m3-task1 \
  --revision "$MODEL_REVISION" \
  --limit 30 --max-model-len 1024 \
  --cache-gb 0.02 --block-size 16 \
  --request-rate 100 --concurrency 8 --seeds 0 1 2
```

Each `--source` has three arguments: label, format, and local JSON/JSONL path.
Supported formats are:

| Format | Input schema | Interpretation |
| --- | --- | --- |
| `chat` | ShareGPT `conversations` or alternating OpenAI `messages` | Retains all complete turns and their causal ordering. |
| `instruction` | String `instruction`, optional string `input`, string `output` | A single user/assistant exchange. Nonempty input follows the instruction after two newlines. |

No shared system prompt is fabricated by the instruction adapter. The model's
own chat template still applies. Unsupported or empty instruction text is
counted in conversion statistics. Existing chat validation and model context
filtering are recorded in each prepared workload. Native tool-call trajectories
and anonymized block-hash traces need different adapters and are not claimed
as supported here. The assignment's suggested datasets are examples; these
experiments use additional public text workloads instead.

The command runs 3 workloads x 3 policies x 3 arrival seeds = 27 configurations.
It requires a fresh output directory and stops on insufficient eligible data,
server errors, or response validation failures. Cache capacity and block size
are scalar arguments and remain fixed throughout the experiment. Realized
capacity is rounded down to complete KV blocks and recorded in bytes/blocks.

Each workload/seed is prepared once and replayed under all three policies.
Trace and workload hashes must match across policies. Every run starts a fresh
online simulator with strict trace matching and prefix caching. Policy execution
order is shuffled reproducibly per seed. This changes host execution order,
not the input dataset order. Seeds change Poisson desired arrivals; subsequent
conversation turns also wait for their preceding replies. Wall-clock scheduling
means repeated runs are not bit-for-bit deterministic.

## Outputs and interpretation

| Output | Contents |
| --- | --- |
| `experiment.json`, `sources.json` | Arguments, Python/platform information, input hashes and conversion counts. |
| `prepared/LABEL/seed-N/` | Selected conversations, exclusions, desired arrivals, and exact token traces. |
| `LABEL/seed-N/POLICY/` | Server log, settings, raw requests/block events, summary, request/block-hit/reuse-gap CSV files. |
| `comparison.json`, `comparison.csv` | Every workload/policy/seed measurement. |
| `ranking.json`, `ranking.csv` | Per-workload mean sharing fraction and sample standard deviation across seeds; exact ties remain ties. |
| `policy_comparison.png`, `.pdf` | Mean sharing fraction with seed standard-deviation bars. |

The primary score for a run is:

```text
shared_fraction = sum(cached_prompt_tokens) / sum(prompt_tokens)
```

Policies are ranked by the arithmetic mean of this token-weighted score across
arrival seeds. Ranking refuses failed or empty runs, missing policies, different
seed sets, duplicate runs, different actual cache geometries, or mismatched
request/trace cohorts. `observed_best` means highest measured mean under the
tested settings; it is not a proof of universal optimality or significance.
No forced winner is reported when scores tie.

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
and failures alongside cache scores. A first-30-eligible subset is a pilot;
general conclusions require other subsets, larger cohorts, more seeds, and
additional arrival rates. Changing context limits also changes the eligible
population and should be reported explicitly.

## Validation

```bash
python -m pytest tests/simulator -q
ruff check vllm/simulator/eviction_study.py tests/simulator/test_eviction_study.py
```

The new tests check text conversion, exclusions, preserved context, ranking,
ties, seed variability, and rejection of unfair or failed comparisons. The
public-data experiment exercises the new entry point through real online
simulator servers. Existing simulator tests cover token replay, scheduling,
preemption, prefix reuse, and both online frontend modes.
