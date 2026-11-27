# November 27 update

Latest English [report and presentation](deliverables/final-2026-11-27/README.md) integrate commit `ad1d826542`. Current simulator tests: **153 passed, 0 failed**. The new 125-cell sweep is repository-reported from rounded summaries; its raw logs were not supplied, so it is kept separate from the 175 historical raw-validated runs. See the dated README for current findings, provenance, rebuild commands and the grid resume-completeness caveat. The following original November24 guide and its 146-test/24-slide references are historical snapshots.

# A Study of Prefix Sharing in LLM Serving

Rice University — COMP536. Authors, in order: Thomas Lin, Yu-Ting Chung, Patrick Chiou and Yu-Xin Lai.

This fork studies prefix-cache behavior using a **CPU-only vLLM v0 online simulator**. It retains real request management, scheduling, continuous batching, sequence/block state and prefix lookup while replacing model computation with recorded token replay. Model configuration and tokenizer files are required; model weights and a GPU are not. Synthetic delays and CPU replay throughput are not measured GPU performance.

## Quick Start for Teaching Assistants

Use the submitted project branch `comp536-sharing`, base commit `3d501ab40fb9a6b1bed5688d1db338b21f4a185f` with the final implementation and deliverables committed on this branch. The assignment baseline `94d545a1a18e20ea8763a6760194589b8a3c9065` is a verified ancestor. Full original Git history has been recovered/preserved. The final source patch records changes against that base; the branch includes those changes, so do not apply the patch twice. Evidence and archive manifests retain their pre-publication snapshot state.

Observed runtime: Linux aarch64, Python3.11.17, PyTorch2.5.1, Transformers4.47.1, NumPy1.26.4, PyArrow19.0.1. Linux/macOS empty-target builds are supported by the existing setup; Windows users should use an appropriate Linux environment. Fresh OS installation was not repeated in this final phase. The current environment and exact package list are preserved in the final evidence directory.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt -r requirements-common.txt \
  -r requirements-simulator.txt 'transformers==4.47.1' 'pyarrow==19.0.1'
python -m pip install pytest pytest-asyncio
VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation --no-deps
export VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
python -m vllm.simulator.study --help
```

If the submitted source already contains a working environment, activate it and skip installation. Do not install the upstream prebuilt `vllm` package over this fork. Installation commands correspond to actual requirement/setup files; the original editable setup and current runtime were exercised, but the complete installation recipe was not repeated in a fresh OS image.

A small **correctness** smoke check (not a benchmark) runs four concurrent trace requests and verifies exact tokens/EOS:

```bash
python -m pytest tests/simulator/test_sim_engine.py::test_concurrent_requests_replay_exact_tokens -q
```

The first execution needs public model config/tokenizer downloads. The full applicable simulator suite was actually run:

```bash
python -m pytest tests/simulator -q -p no:cacheprovider
```

Expected at the final tested source: **146 passed, 0 failed**. Both direct and multiprocessing API frontends are included. The former `workload` regression is fixed through client → RPC serialization → engine submission; do not describe it as still unresolved. Missing labels remain supported. New integration checks confirm labels on both endpoint types and block metrics. Historical reports are preserved but their older136/5 status is superseded.

## Repository and final artifacts

```text
vllm/executor/sim_executor.py             virtual KV geometry and trace executor
vllm/simulator/                          traces, client, metrics, analysis, studies
vllm/core/                              scheduler, allocator and policies
vllm/engine/multiprocessing/             fixed workload metadata transport
tests/simulator/                        unit, engine and online API validation
tools/comp536_final_study.py             resumable M2 / mixed experiment driver
tools/comp536_final_figures.py           raw validation + figure/table generation
tools/comp536_counting/sitecustomize.py  opt-in mixed eviction counter
tools/comp536_diagrams.py                source-based PDF/SVG/PNG diagrams
deliverables/final-2026-11-24/            final report/slides/notes/QA/evidence
```

The final report uses the supplied official MLSys2025 style, with four confirmed names/emails and Rice affiliation. No corresponding author or presenter split is designated. See the final requirements checklist, changelog and contribution table for scope and limitations. Older deliverables are historical snapshots; new artifacts supersede their evidence gaps when supported by the final logs.

## Model and dataset preparation

Set model metadata/tokenizer cache paths if desired (`HF_HOME`). Only after required files are cached, set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to avoid network requests. Do not enable offline mode before the first download.

```bash
mkdir -p data/m3
python -m vllm.simulator.study sample --limit 1000 \
  --output data/m3/sharegpt1000.jsonl
```

This streams the pinned ShareGPT revision192ab21. The final M2 cohort selects the first200 eligible conversations from this1000-record subset, using full-conversation context 4096 filtering for both modes. It produces200 single-turn and608 multi-turn requests per configuration; arrival seeds change timing, not the dataset sample.

The heterogeneous dataset URLs/formats and source limits are in [Task 1 guide](docs/milestone3-task1.md). The final evidence includes exact dataset SHA256 and URLs; several published sources use mutable main URLs, so verify file hashes instead of assuming later downloads are identical. All four AgentBank shards are interleaved. CC-Bench, QuALITY, Bailian chat/coder, Mooncake and Code Alpaca require their documented formats. No large dataset or model cache belongs in Git.

## Minimal standalone online workflow

The following uses actual subcommands/flags. The standalone prepare/API/replay subcommands were additionally exercised with two conversations and seven successful requests through the default multiprocessing frontend. Exact smoke commands/results are in the evidence directory; this correctness check is excluded from experiment totals.

```bash
python -m vllm.simulator.study prepare --dataset data/m3/sharegpt1000.jsonl \
  --output experiments/smoke-prepared-new --model "$MODEL" --revision "$MODEL_REVISION" \
  --limit 2 --max-model-len 4096 --mode multi --arrival poisson --request-rate 10
```

Start a server in another terminal with the same environment/model revision:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION" \
  --distributed-executor-backend sim --load-format dummy \
  --sim-trace-path experiments/smoke-prepared-new/trace.jsonl --sim-strict-trace \
  --sim-metrics-path experiments/smoke-server-new/blocks.jsonl \
  --sim-kv-cache-gb 0.1 --max-model-len 4096 --block-size 16 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --host 127.0.0.1 --port 8000
```

Then replay:

```bash
python -m vllm.simulator.study replay \
  --workload experiments/smoke-prepared-new/workload.json \
  --base-url http://127.0.0.1:8000 --output experiments/smoke-requests-new.jsonl \
  --concurrency 2
```

Expected: all selected turns succeed and recorded responses/token counts/stop reasons match. Use fresh output names. The automated study helper still selects the direct frontend for consistent historical experiments; both frontend modes now pass correctness tests.

## Required experiments and resumability

Give `--data` the directory containing the exact downloaded files; `--output` is a **new** experiment root. Existing historical outputs are never replaced. The new driver retains interrupted attempts and resumes only completed attempts whose input/control/implementation fingerprints match. An incompatible implementation/input raises an error rather than silently pooling results.

Milestone 2, actual final configuration (108 runs):

```bash
python tools/comp536_final_study.py --phase m2 \
  --data data/m3 --output experiments/final-new --seeds 0 1 2
```

This reuses the original documented36-cell-per-seed sweep: modes single/multi;0.05/0.10 GiB; blocks 8/16/32; LRU/LFU/FIFO;200 conversations; context 4096; target 100/s; concurrency/max-sequences 64. M2-A reuses matching LRU/block 16 cells rather than running duplicate experiments. The `--pilot` option executes one M2 multi/block 16/0.05 GiB/LRU cell, then that completed cell is resumed in the full batch. Absolute byte capacity is approximately fixed across block sizes, rounded to whole blocks.

Milestone 3 Task 1: the final analysis **reuses** the existing validated51-run experiment at the base source; the fix changes the previously broken multiprocessing path and strict-zip validation, not the direct-mode policy/executor decisions used by that study. Exact expanded command and input hashes are preserved in final evidence. To deliberately rerun it, use the [Task 1 guide](docs/milestone3-task1.md) with a fresh output root; do not rerun it merely to regenerate figures.

Milestone 3 Task 2: prepare the original skewed eight-source cohort at context 32768 using the [Task 2 guide](docs/milestone3-task2.md#run-task-2). All prepared sources must agree on model/revision/tokenizer/template/context. The final run reused the already prepared valid skewed cohort. The balanced preparation expands only labels with insufficient tokens (agent 300, code 8000, quality300, Bailian-chat 1000, Bailian-coder1000) using the same source formats, context 32768 and output cap32. Exact executed expansion command is recorded in final evidence; select a new output directory when repeating preparation.

```bash
python tools/comp536_final_study.py --phase mixed \
  --data data/m3 --output experiments/final-new \
  --mixed-prepared /path/to/skewed/prepared \
  --expanded-prepared /path/to/expanded/prepared --seeds 0
```

The driver builds both skewed and equal-token mixtures. Equal-token means the first causal prefix reaching260,000 prompt tokens per label, not equal request counts; overshoot is preserved. It runs all eight policies with a fresh cache, randomized order, block 16, ratio 0.10, context 32768, target 100/s, concurrency 8 and think time 1s. New mixed runs additionally count successful allocator eviction allocations with an isolated opt-in hook; count files are required before declaring a run complete. No victim ordering/reference-count behavior changes. Promotion counters remain unavailable.

This final evaluation uses one seed per mixture (16 runs); it reports no mixed SD/CI. Extra repetitions/ablations are not silently claimed completed. Additional seeds are supported by the driver; do not pool them with incompatible cohorts/settings.

## Regenerate final figures without benchmarks

```bash
python tools/comp536_final_figures.py \
  --m2 /path/to/final/experiments/m2 \
  --mixed /path/to/final/experiments/mixed \
  --task1 /path/to/existing/task1 \
  --output /path/to/new/final-figures
python tools/comp536_diagrams.py --output /path/to/new/diagrams
```

After extracting the raw-evidence archive elsewhere, point the three input arguments at its `m2`, `mixed` and `task1` directories. The validator resolves the adjacent prepared cohorts without rewriting historical absolute provenance paths.

The figure tool reads completed-run markers/raw logs, validates IDs, token arithmetic, cohort/capacity compatibility and block/gap tables, then writes aggregated CSV/JSON, vector PDFs and high-resolution PNGs. It does not invoke vLLM inference or rewrite source experimental summaries. Keep generated analysis in a separate path. Historical M2/Task 1 eviction counts are null, never inferred from misses. Gap CDFs are successful-hit-associated wall time, not unconditional potential recurrence.

## Troubleshooting and known limits

- PyArrow26 failed with the observed NumPy1.26 environment; PyArrow19.0.1 worked. The final phase restores missing Python/model metadata caches without upgrading major simulator dependencies.
- If a moved environment's interpreter symlink is broken, recreate an equivalent Python3.11 environment from the recorded package list; do not assume a `/tmp` interpreter persists across reboots.
- Strict trace mismatches usually require checking IDs, template/date kwargs, tokenizer revision and truncation. Do not use fallback fillers to claim exact replay success.
- The cache must hold the longest selected request including response/EOS constraints; ratio experiments record feasibility clamps. Capacity in blocks is not the same as bytes across block sizes.
- One benchmark server runs at a time; real CPU scheduling/HTTP/backlog still introduce timing variation. Identical seeds do not imply identical wall-clock cache retention or hold expiry.
- The simulator explicitly rejects LoRA, prompt adapters, speculative and multistep scheduling. GPU numerical quality, kernels, calibrated latency and distributed communication are not modeled.
- Existing project-wide legacy RPC modules have unrelated pyupgrade lint warnings; no mass refactor was performed. The prior B905 and targeted study/regression-test checks pass.
- The presentation PDF is exported from the editable PPTX with an isolated LibreOffice24.2.7.2 Impress renderer. A separate shared-source layout companion is retained for comparison. Fonts and wrapping can vary in Microsoft PowerPoint; rehearse in the actual presentation environment before delivery.

## Final evidence and preservation

Validated studies: **m2: 108 runs / 43,632 requests / 0 failures; mixed: 16 runs / 74,984 requests / 0 failures; task1: 51 runs / 21,834 requests / 0 failures**. These replay counts are repeated configurations, not independent dataset samples. New completed M2/mixed server/replay/analysis lifecycles total **109.44 minutes** (M2 38.51; mixed 70.93); preparation, tests, failed/interrupted attempts and document work are separate. Task1 was reused, not rerun. Mixed results use one seed per composition; no SD/CI is estimated.

Exact commands, dataset/model versions, hashes, source patch, test logs and evidence index accompany `deliverables/final-2026-11-24/`. Large raw request/block logs and prepared traces remain in their separate experiment trees; the artifact manifest gives exact locations. The original reports/slides, failed/interrupted historical runs, datasets and Git histories are preserved. No remote push was performed.

---

## Original upstream README

The upstream material below is retained for context. Its GPU installation/features concern upstream vLLM; use this fork's CPU-simulator instructions above for the course project.


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://discord.gg/jz7wjKhh6g"><b>Discord</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

See the [Milestone guide](docs/comp536.md) for setup, commands, metric definitions, and experiment limitations.

For separate-workload eviction comparisons, see the
[Milestone 3 Task 1 guide](docs/milestone3-task1.md).

---  
*Latest News* 🔥
- [2024/12] vLLM joins [pytorch ecosystem](https://pytorch.org/blog/vllm-joins-pytorch)! Easy, Fast, and Cheap LLM Serving for Everyone!
- [2024/11] We hosted [the seventh vLLM meetup](https://lu.ma/h0qvrajz) with Snowflake! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing), and Snowflake team [here](https://docs.google.com/presentation/d/1qF3RkDAbOULwz9WK5TOltt2fE9t6uIc_hVNLFAaQX6A/edit?usp=sharing).
- [2024/10] We have just created a developer slack ([slack.vllm.ai](https://slack.vllm.ai)) focusing on coordinating contributions and discussing features. Please feel free to join us there!
- [2024/10] Ray Summit 2024 held a special track for vLLM! Please find the opening talk slides from the vLLM team [here](https://docs.google.com/presentation/d/1B_KQxpHBTRa_mDF-tR6i8rWdOU5QoTZNcEg2MKZxEHM/edit?usp=sharing). Learn more from the [talks](https://www.youtube.com/playlist?list=PLzTswPQNepXl6AQwifuwUImLPFRVpksjR) from other vLLM contributors and users!
- [2024/09] We hosted [the sixth vLLM meetup](https://lu.ma/87q3nvnh) with NVIDIA! Please find the meetup slides [here](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing).
- [2024/07] We hosted [the fifth vLLM meetup](https://lu.ma/lp0gyjqr) with AWS! Please find the meetup slides [here](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing).
- [2024/07] In partnership with Meta, vLLM officially supports Llama 3.1 with FP8 quantization and pipeline parallelism! Please check out our blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
- [2024/06] We hosted [the fourth vLLM meetup](https://lu.ma/agivllm) with Cloudflare and BentoML! Please find the meetup slides [here](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing).
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) with IBM! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) with a16z! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

---
## About
vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8.
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

**Performance benchmark**: We include a performance benchmark at the end of [our blog post](https://blog.vllm.ai/2024/09/05/perf-update.html). It compares the performance of vLLM against other LLM serving engines ([TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [SGLang](https://github.com/sgl-project/sglang) and [LMDeploy](https://github.com/InternLM/lmdeploy)). The implementation is under [nightly-benchmarks folder](.buildkite/nightly-benchmarks/) and you can [reproduce](https://github.com/vllm-project/vllm/issues/8176) this benchmark using our one-click runnable script.

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism and pipeline parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
- Prefix caching support
- Multi-lora support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:
- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral)
- Embedding Models (e.g. E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://vllm.readthedocs.io/en/latest/) to learn more.
- [Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Supported Models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

## Sponsors

vLLM is a community project. Our compute resources for development and testing are supported by the following organizations. Thank you for your support!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/source/community/sponsors.md -->

- a16z
- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Dropbox
- Google Cloud
- Lambda Lab
- Nebius
- NVIDIA
- Replicate
- Roblox
- RunPod
- Sequoia Capital
- Skywork AI
- Trainy
- UC Berkeley
- UC San Diego
- ZhenFund

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):
```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

* For technical questions and feature requests, please use Github issues or discussions.
* For discussing with fellow users, please use Discord.
* For coordinating contributions and development, please use Slack.
* For security disclosures, please use Github's security advisory feature.
* For collaborations and partnerships, please contact us at vllm-questions AT lists.berkeley.edu.

## Media Kit

* If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).
