# A Study of Prefix Sharing in LLM Serving — COMP536

CPU-only trace simulation in the real vLLM v0 online request/scheduler/cache path. Current audited source: `3d501ab40fb9a6b1bed5688d1db338b21f4a185f` on `comp536-sharing`; assignment baseline: `94d545a1a18e20ea8763a6760194589b8a3c9065` (ancestry not locally verifiable in this shallow clone). Model weights/GPU are not needed; model configuration and tokenizer are needed. Synthetic timings are not measured GPU performance.

## Quick Start for Teaching Assistants

The commands below use actual files and flags. Existing-environment tests and full direct-mode Task1 execution were exercised; **fresh installation as a whole was not repeated during this documentation phase**. Use Linux/macOS with Python3.11 (observed Linux aarch64, Python3.11.17, Torch2.5.1, Transformers4.47.1, NumPy1.26.4, PyArrow19.0.1). Exact observed packages: [environment](deliverables/evidence/environment.txt).

```bash
git clone --branch comp536-sharing https://github.com/PatrickChiou/vllm.git
cd vllm
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-build.txt -r requirements-common.txt -r requirements-simulator.txt "transformers==4.47.1" "pyarrow==19.0.1"
VLLM_TARGET_DEVICE=empty python -m pip install -e . --no-build-isolation --no-deps
export VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MODEL=Qwen/Qwen2.5-0.5B-Instruct
export MODEL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
python -m vllm.simulator.study --help
```

Fast local correctness checks (no benchmark/data download):

```bash
python -m pytest tests/simulator/test_timing.py tests/simulator/test_trace_store.py tests/simulator/test_prefix_metrics.py tests/simulator/test_mix_workload.py tests/simulator/test_eviction_policies.py -q
```

Minimal engine simulation with known test token traces (downloads tokenizer/config on first use):

```bash
python -m pytest tests/simulator/test_sim_engine.py::test_concurrent_requests_replay_exact_tokens -q
```

This test is part of the previously executed suite and passed. It verifies exact trace tokens and EOS for four concurrent engine requests. For online coverage use the direct-mode API tests below. This smoke test is a correctness check, not a benchmark. A first run needs network access to the pinned model's public metadata/tokenizer. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` only **after** caching the required assets.

```bash
python -m pytest tests/simulator/test_sim_api_server.py -q -k True
python -m pytest tests/simulator -q
```

The full suite currently has **136 passed, 5 failed** at the audited commit. The five failures use frontend multiprocessing: serving handlers pass `workload`, but `MQLLMEngineClient.generate` does not accept it. Experiments explicitly use `--disable-frontend-multiprocessing` through `study.server`. Do not remove that flag when launching a simulator API server. No implementation fix has been applied. See [audit](deliverables/IMPLEMENTATION_AUDIT.md).

## Repository map and deliverables

```text
vllm/executor/sim_executor.py   CPU trace executor and virtual KV capacity
vllm/simulator/                traces, workload, timing, HTTP replay, metrics, studies
vllm/core/                    scheduler, block allocator and eviction policies
tests/simulator/              unit, engine and online API tests
docs/comp536.md                M2 setup/sweep instructions
docs/milestone3-task1.md       eight-workload baseline policy study
docs/milestone3-task2.md       mixed construction and custom policies
deliverables/                 audit, verification, figures, editable slides, report
```

See [experimental verification](deliverables/EXPERIMENT_VERIFICATION.md) before interpreting results. Task1 has checked raw evidence for51 configurations/21,834 requests. M2 experimental sweeps and complete mixed raw logs were not located; documentary mixed results are labeled accordingly. Do not assume every milestone is fully verified.

## Run real study commands (only when intentionally collecting new results)

Use a **fresh output directory** every time. Dataset files and large raw outputs are external and are not in a fresh Git clone. Never delete or overwrite prior runs. These examples document capabilities; this documentation phase did not execute them.

For single/multi ShareGPT, capacity, block size and baseline policy experiments, use the [M2 guide](docs/comp536.md). Its `study sample`, `prepare`, `replay` and `sweep` subcommands are present. Portable smaller sweep example below is source-checked, **not executed as this exact invocation**:

```bash
mkdir -p data
python -m vllm.simulator.study sample --limit 1000 --output data/sharegpt1000.jsonl
python -m vllm.simulator.study sweep \
  --dataset data/sharegpt1000.jsonl --output experiments/m2-new \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --limit 20 --max-model-len 2048 --arrival poisson --request-rate 100 --seed 0 \
  --modes single multi --cache-gb 0.05 0.10 --block-sizes 8 16 32 \
  --policies lru lfu fifo --concurrency 8 --max-num-seqs 8 --no-plots
```

`sample` requires an existing parent `data/` directory (`mkdir -p data` first); the published source is pinned to ShareGPT revision192ab21. `--limit` selects eligible conversations, not requests. Filtering checks full conversations in both modes. Synthetic delays default to .01ms/uncached-prefill-token and .1ms/decode-sequence-step. The original guide's36-run sweep is implemented but its raw results are unavailable for this audit.

The [Task1 guide](docs/milestone3-task1.md) includes exact dataset download URLs and eight-source arguments, per-source context/trace-block/time-scale overrides and seeds. Its complete study was executed in the preceding authorized run; exact expanded command is [saved here](deliverables/evidence/task1-command.json). Actual modules:

```bash
python -m vllm.simulator.eviction_study --help
python -m vllm.simulator.mixed_prepare --help
python -m vllm.simulator.mix_workload --help
python -m vllm.simulator.mixed_study --help
```

The [Task2 guide](docs/milestone3-task2.md#run-task-2) documents actual `mixed_prepare` → `mix_workload` → `mixed_study` commands. All sources must use the same model/tokenizer/revision/context/template. Preparation and skewed construction were exercised; the current eight-policy mixed run was interrupted and is **not** claimed reproduced. Complete guide results are repository-reported, not raw-verified. The equal-token cohort's exact selection command is missing. Policies: LRU/LFU/FIFO plus TLRU/TFIFO/HLRU/HFIFO/RFIFO; labels are supplied by dataset, not inferred by an online classifier.

## Reproduce figures without running benchmarks

Use a new output directory. This script only reads existing files and writes new figures/checks. It does not invoke the server or overwrite experimental summaries.

```bash
python deliverables/figures/generation_scripts/generate_figures.py \
  --input /path/to/existing/task1 \
  --output /path/to/new/figure-output
```

In this workspace, input was `../rerun-2026-11-23/task1`. Requires only NumPy/Matplotlib. Expected: PDF/300dpi PNG charts, architecture/workflow diagrams, `verification.json`, checked comparison/ranking tables and distribution statistics. The generator refuses a nonempty output directory. Requested matched single/multi and capacity/blocksize plots are omitted because data are missing. Archived mixed chart is clearly labeled **repository-reported**.

A fresh clone can inspect the existing figure06 PDF/PNG and compact tables in `deliverables/figures/publication/`; reproducing the charts from raw files requires the external result tree. Do not use `study analyze` against protected originals: it rewrites CSV/JSON/figures. Copy results to a new tree first if that analyzer is desired. Report build instructions and document-tool versions are in [final checklist](deliverables/FINAL_CHECKLIST.md).

## Troubleshooting and reproducibility limits

- Install this fork editable with `VLLM_TARGET_DEVICE=empty`; `pip install vllm` installs unrelated upstream GPU-serving code and does not reproduce this project.
- PyArrow26 failed to read AgentBank under the observed NumPy1.26 environment; PyArrow19.0.1 worked. This is an observed compatibility workaround, not a universal minimum requirement.
- No model weights are loaded, but config/tokenizer downloads and memory for prepared traces/raw event logs are required. Exact minimum RAM/runtime is not established; Task1 raw files here occupy about1.8GiB.
- Model revisions, source SHA256, context limits, arrivals, concurrency, truncation and virtual block counts affect results. Same seed does not guarantee identical wall-clock scheduling. Gaps and hold expiry depend on CPU/HTTP/backlog.
- Trace mismatches in strict mode fail; check request IDs/template/tokenizer revisions. Do not use fallback filler output to claim a successful exact replay.
- Ruff currently reports B905 in `eviction_study.py:193`; tests are not fully green. Neither issue was silently repaired.
- This local checkout is shallow. The assignment requires original full Git histories; preserve/recover them from the original repository before submission. Deliverable generation does not alter Git history.

---

## Original upstream README

The upstream information below is retained for context; its GPU installation/feature claims describe upstream vLLM rather than the restricted simulator support described above.

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
