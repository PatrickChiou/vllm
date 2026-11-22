"""Milestone3 Task2: Replay one mixed workload under each eviction policy."""
import argparse
import asyncio
import json
from pathlib import Path
from vllm.simulator.analysis import analyze
from vllm.simulator.eviction_study import cache_blocks, working_set
from vllm.simulator.replay import replay
from vllm.simulator.study import server

POLICIES = ("tlru", "lru", "lfu", "fifo", "tfifo", "hlru", "hfifo", "rfifo")


def run(args) -> None:
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    source = args.workload.resolve()
    workload = json.loads((source / "workload.json").read_text())
    trace_path = source / "trace.jsonl"
    footprint = working_set(trace_path, args.block_size)
    sizing = cache_blocks(args.cache_ratio, footprint, args.block_size)
    root.mkdir(parents=True, exist_ok=True)
    for policy in POLICIES:
        directory = root / policy
        print(f"Running {policy}", flush=True)
        with server(workload, trace_path, directory,
                    block_size=args.block_size, cache_gb=args.cache_gb,
                    policy=policy, prefill_ms=args.prefill_ms,
                    decode_ms=args.decode_ms,
                    max_num_seqs=args.concurrency,
                    startup_timeout=args.startup_timeout,
                    num_gpu_blocks=sizing["num_gpu_blocks"],
                    hold_s=args.hold_s, 
                    hold_budget=args.hold_budget,) as state:
            url, cache = state
            metadata = {
                "name": policy, "policy": policy,
                "block_size": args.block_size, "cache_gb": None,
                "cache_ratio": args.cache_ratio,
                "cache_clamped": sizing["cache_clamped"],
                "working_set_blocks": footprint["unique_blocks"],
                "num_gpu_blocks": cache["num_gpu_blocks"],
                "virtual_cache_bytes": (cache["num_gpu_blocks"] *
                                        cache["block_bytes"]),
                "conversations": len(workload["conversations"]),
                "think_time_s": workload["think_time_s"],
                "concurrency": args.concurrency,
                "prefill_ms_per_token": args.prefill_ms,
                "decode_ms_per_seq": args.decode_ms,
            }
            (directory / "run.json").write_text(
                json.dumps(metadata, indent=2) + "\n")
            rows = asyncio.run(replay(
                workload, url, directory / "requests.jsonl",
                concurrency=args.concurrency, timeout=args.timeout))
        if any(not row["success"] for row in rows):
            raise RuntimeError(f"Failed requests in {directory}")
    analyze(root, plots=not args.no_plots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True, help="Directory from mix_workload.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-ratio", type=float, required=True)
    parser.add_argument("--cache-gb", type=float, default=0.02)
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prefill-ms", type=float, default=0.01)
    parser.add_argument("--decode-ms", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    # parser.add_argument("--hold-budget", type=float, default=0.5)
    parser.add_argument("--hold-budget", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not 0 < args.cache_ratio <= 1:
        parser.error("cache-ratio must be in (0, 1]")
    if args.hold_s < 0:
        parser.error("hold-s must be >= 0")
    if not 0 <= args.hold_budget <= 1:
        parser.error("hold-budget must be in [0, 1]")
    run(args)
if __name__ == "__main__":
    main()