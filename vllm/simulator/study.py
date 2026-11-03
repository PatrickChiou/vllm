"""Milestone 2 CLI: sample, prepare, replay, sweep, and analyze.

Run ``python -m vllm.simulator.study --help`` for commands.
"""
import argparse
import asyncio
import itertools
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from vllm.simulator.analysis import analyze
from vllm.simulator.replay import replay
from vllm.simulator.workload import prepare, sample_dataset

DATASET_URL = ("https://huggingface.co/datasets/anon8231489123/"
               "ShareGPT_Vicuna_unfiltered/resolve/192ab21/"
               "ShareGPT_V3_unfiltered_cleaned_split.json")


@contextmanager
def server(workload: dict, trace_path: Path, directory: Path, *,
           block_size: int, cache_gb: float, policy: str,
           prefill_ms: float, decode_ms: float, max_num_seqs: int,
           startup_timeout: float):
    directory.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", workload["model"], "--tokenizer", workload["tokenizer"],
        "--host", "127.0.0.1", "--port", str(port),
        "--distributed-executor-backend", "sim", "--load-format", "dummy",
        "--sim-trace-path", str(trace_path.resolve()), "--sim-strict-trace",
        "--sim-metrics-path", str((directory / "blocks.jsonl").resolve()),
        "--sim-kv-cache-gb", str(cache_gb), "--sim-eviction-policy", policy,
        "--sim-prefill-ms-per-token", str(prefill_ms),
        "--sim-decode-ms-per-seq", str(decode_ms),
        "--max-model-len", str(workload["max_model_len"]),
        "--block-size", str(block_size), "--max-num-seqs", str(max_num_seqs),
        "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--disable-frontend-multiprocessing", "--disable-log-requests",
        "--disable-sliding-window", "--swap-space", "0",
        "--chat-template-content-format",
        workload["chat_template_content_format"],
    ]
    for field in ("revision", "tokenizer_revision", "chat_template"):
        if workload.get(field):
            command.extend(["--" + field.replace("_", "-"), workload[field]])
    if workload["trust_remote_code"]:
        command.append("--trust-remote-code")
    env = dict(os.environ, VLLM_USE_V1="0", TOKENIZERS_PARALLELISM="false")
    base_url = f"http://127.0.0.1:{port}"
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited; inspect {directory / 'server.log'}")
                try:
                    response = requests.get(base_url + "/health", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Server startup timed out; inspect {directory}")
                time.sleep(0.2)
            # Reject impossible configurations instead of silently benchmarking
            # requests that the scheduler discards for insufficient capacity.
            with (directory / "blocks.jsonl").open() as stream:
                cache = next(json.loads(line) for line in stream
                             if json.loads(line)["event"] == "cache_config")
            capacity = (cache["num_gpu_blocks"] -
                        int(cache["num_gpu_blocks"] * 0.01)) * block_size
            longest = max(
                t["prompt_tokens"] + t["output_tokens"]
                for c in workload["conversations"] for t in c["turns"])
            if longest > capacity:
                raise ValueError(f"Cache holds {capacity} tokens but a request "
                                 f"needs {longest}; increase --cache-gb")
            yield base_url, cache
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def sweep(args) -> None:
    # Model metadata/tokenizers are loaded once; model weights are never loaded.
    from vllm.simulator.trace_builder import TraceBuilder
    builder = TraceBuilder(args.model, tokenizer=args.tokenizer,
                           chat_template=args.chat_template,
                           revision=args.revision,
                           trust_remote_code=args.trust_remote_code,
                           chat_template_kwargs=json.loads(
                               args.chat_template_kwargs))
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("A new sweep needs an empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        prepared = root / "prepared" / mode
        workload = prepare(args.dataset, prepared, builder, mode=mode,
                           limit=args.limit, max_model_len=args.max_model_len,
                           arrival=args.arrival, request_rate=args.request_rate,
                           seed=args.seed, think_time=args.think_time)
        for cache_gb, block_size, policy in itertools.product(
                args.cache_gb, args.block_sizes, args.policies):
            name = f"{mode}-{cache_gb:g}GiB-b{block_size}-{policy}"
            directory = root / name
            print(f"Running {name}", flush=True)
            with server(workload, prepared / "trace.jsonl", directory,
                        block_size=block_size, cache_gb=cache_gb, policy=policy,
                        prefill_ms=args.prefill_ms, decode_ms=args.decode_ms,
                        max_num_seqs=args.max_num_seqs,
                        startup_timeout=args.startup_timeout) as (url, cache):
                config = {
                    "name": name, "mode": mode, "cache_gb": cache_gb,
                    "block_size": block_size, "policy": policy,
                    "num_gpu_blocks": cache["num_gpu_blocks"],
                    "virtual_cache_bytes": (cache["num_gpu_blocks"] *
                                            cache["block_bytes"]),
                    "model": args.model, "seed": args.seed,
                    "request_rate": args.request_rate,
                    "time_scale": args.time_scale,
                    "prefill_ms_per_token": args.prefill_ms,
                    "decode_ms_per_seq": args.decode_ms,
                    "arrival": workload["arrival"],
                    "conversations": len(workload["conversations"]),
                }
                (directory / "run.json").write_text(
                    json.dumps(config, indent=2) + "\n")
                rows = asyncio.run(replay(
                    workload, url, directory / "requests.jsonl",
                    concurrency=args.concurrency, time_scale=args.time_scale))
            failures = [r for r in rows if not r["success"]]
            if failures:
                analyze(root, plots=False)
                raise RuntimeError(f"{len(failures)} failures in {name}: "
                                   f"{failures[0].get('error')}")
    analyze(root, plots=not args.no_plots)


def workload_args(parser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template")
    parser.add_argument("--chat-template-kwargs",
                        default='{"date_string": "26 Jul 2024"}')
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--arrival", choices=["auto", "recorded", "poisson"],
                        default="auto")
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--think-time", type=float, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Stream ShareGPT sample")
    sample.add_argument("--url", default=DATASET_URL)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--limit", type=int, default=100)
    prep = commands.add_parser("prepare", help="Build workload and trace")
    workload_args(prep)
    prep.add_argument("--mode", choices=["single", "multi"], default="multi")
    client = commands.add_parser("replay", help="Replay against a sim server")
    client.add_argument("--workload", type=Path, required=True)
    client.add_argument("--base-url", default="http://127.0.0.1:8000")
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--concurrency", type=int, default=256)
    client.add_argument("--time-scale", type=float, default=1.0)
    client.add_argument("--timeout", type=float, default=300.0)
    client.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    experiment = commands.add_parser("sweep", help="Compare cache settings")
    workload_args(experiment)
    experiment.add_argument("--modes", nargs="+", choices=["single", "multi"],
                            default=["single", "multi"])
    experiment.add_argument("--cache-gb", nargs="+", type=float,
                            default=[0.02, 0.04])
    experiment.add_argument("--block-sizes", nargs="+", type=int,
                            default=[8, 16, 32])
    experiment.add_argument("--policies", nargs="+",
                            choices=["lru", "lfu", "fifo"],
                            default=["lru", "lfu", "fifo"])
    experiment.add_argument("--concurrency", type=int, default=256)
    experiment.add_argument("--max-num-seqs", type=int, default=64)
    experiment.add_argument("--time-scale", type=float, default=1.0)
    experiment.add_argument("--prefill-ms", type=float, default=0.01)
    experiment.add_argument("--decode-ms", type=float, default=0.1)
    experiment.add_argument("--startup-timeout", type=float, default=120.0)
    experiment.add_argument("--no-plots", action="store_true")
    report = commands.add_parser("analyze", help="Export metrics and plots")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.command == "sample":
        count = sample_dataset(args.url, args.output, args.limit)
        print(f"Saved {count} conversations")
    elif args.command == "prepare":
        from vllm.simulator.trace_builder import TraceBuilder
        builder = TraceBuilder(args.model, tokenizer=args.tokenizer,
                               chat_template=args.chat_template,
                               revision=args.revision,
                               trust_remote_code=args.trust_remote_code,
                               chat_template_kwargs=json.loads(
                                   args.chat_template_kwargs))
        result = prepare(args.dataset, args.output, builder, mode=args.mode,
                         limit=args.limit, max_model_len=args.max_model_len,
                         arrival=args.arrival, request_rate=args.request_rate,
                         seed=args.seed, think_time=args.think_time)
        print(f"Prepared {len(result['conversations'])} conversations; "
              f"excluded: {result['excluded_conversations']}")
    elif args.command == "replay":
        result = asyncio.run(replay(json.loads(args.workload.read_text()),
                                   args.base_url, args.output,
                                   concurrency=args.concurrency,
                                   time_scale=args.time_scale,
                                   timeout=args.timeout,
                                   api_key=args.api_key))
        failures = sum(not r["success"] for r in result)
        print(f"{len(result) - failures} succeeded; {failures} failed")
        if failures:
            raise SystemExit(1)
    elif args.command == "sweep":
        sweep(args)
    else:
        analyze(args.input, plots=not args.no_plots)


if __name__ == "__main__":
    main()
