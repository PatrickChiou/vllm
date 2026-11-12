"""Milestone 3 Task 1: compare eviction policies on separate text workloads."""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import random
import statistics
from pathlib import Path

from vllm.simulator.analysis import analyze, write_csv
from vllm.simulator.replay import replay
from vllm.simulator.study import server
from vllm.simulator.workload import prepare, read_records

POLICIES = ("lru", "lfu", "fifo")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_instructions(source: Path, destination: Path) -> dict:
    """Convert Alpaca-style records without adding a shared system prompt.

    Invalid text records are counted and excluded. Invalid JSON aborts the
    conversion. The existing preparation code applies model context filtering.
    """
    counts = {"accepted": 0, "invalid_text": 0}
    with destination.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(read_records(source)):
            instruction = record.get("instruction")
            context = record.get("input", "")
            reply = record.get("output")
            if (not all(isinstance(x, str)
                        for x in (instruction, context, reply))
                    or not instruction.strip() or not reply.strip()):
                counts["invalid_text"] += 1
                continue
            prompt = instruction + ("\n\n" + context if context else "")
            normalized = {"id": str(record.get("id", index)), "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply},
            ]}
            stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            counts["accepted"] += 1
    if not counts["accepted"]:
        raise ValueError("Instruction dataset has no usable records")
    return counts


def rank_results(rows: list) -> list:
    """Rank observed sharing, requiring a complete, comparable experiment."""
    if not rows:
        raise ValueError("No completed runs")
    geometry = {(r["block_size"], r["virtual_cache_bytes"]) for r in rows}
    if len(geometry) != 1:
        raise ValueError("Cache capacity and block size must be fixed")
    groups, cohorts, seen = {}, {}, set()
    for row in rows:
        if row["failed_requests"] or not row["successful_requests"]:
            raise ValueError("Cannot rank failed or empty runs")
        fraction = row["token_weighted_shared_fraction"]
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Invalid sharing fraction")
        identity = row["workload"], row["policy"], row["seed"]
        if identity in seen:
            raise ValueError("Duplicate workload/policy/seed")
        seen.add(identity)
        cohort = (row["successful_requests"], row["prompt_tokens"],
                  row["workload_sha256"], row["trace_sha256"])
        key = row["workload"], row["seed"]
        if cohorts.setdefault(key, cohort) != cohort:
            raise ValueError("Policies must use identical requests and traces")
        groups.setdefault((row["workload"], row["policy"]), []).append(row)
    ranking = []
    for workload in sorted({r["workload"] for r in rows}):
        policies = {p for label, p in groups if label == workload}
        if policies != set(POLICIES):
            raise ValueError("Each workload needs LRU, LFU, and FIFO")
        seed_sets = [{r["seed"] for r in groups[workload, p]}
                     for p in POLICIES]
        if any(seeds != seed_sets[0] for seeds in seed_sets):
            raise ValueError("Policies must use the same seeds")
        candidates = []
        for policy in POLICIES:
            runs = groups[workload, policy]
            fractions = [r["token_weighted_shared_fraction"] for r in runs]
            candidates.append({
                "workload": workload, "policy": policy,
                "seeds": sorted(seed_sets[0]), "runs": len(runs),
                "mean_shared_fraction": statistics.mean(fractions),
                "std_shared_fraction": (statistics.stdev(fractions)
                                        if len(runs) > 1 else 0.0),
                "prompt_tokens": sum(r["prompt_tokens"] for r in runs),
                "cached_tokens": sum(r["cached_tokens"] for r in runs),
            })
        candidates.sort(key=lambda r: (-r["mean_shared_fraction"], r["policy"]))
        best = candidates[0]["mean_shared_fraction"]
        for row in candidates:
            row["observed_best"] = (
                abs(row["mean_shared_fraction"] - best) <= 1e-12)
        ranking.extend(candidates)
    return ranking


def export_results(root: Path, *, plots: bool = True) -> list:
    """Summarize saved per-run measurements and preserve ties in ranking."""
    rows = analyze(root, plots=False)
    ranking = rank_results(rows)
    (root / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
    write_csv(root / "ranking.csv", ranking, list(ranking[0]))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = sorted({r["workload"] for r in ranking})
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.24
        for index, policy in enumerate(POLICIES):
            selected = [next(r for r in ranking if r["workload"] == label
                             and r["policy"] == policy) for label in labels]
            ax.bar([x + (index - 1) * width for x in range(len(labels))],
                   [r["mean_shared_fraction"] for r in selected], width,
                   yerr=[r["std_shared_fraction"] for r in selected],
                   capsize=3, label=policy.upper())
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 1),
               ylabel="Cached prompt tokens / all prompt tokens",
               title="Separate workloads; error bars: seed standard deviation")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(root / f"policy_comparison.{extension}", dpi=180)
        plt.close(fig)
    return ranking


def run(args) -> list:
    """Replay each source/policy with a fresh online server."""
    from vllm.simulator.trace_builder import TraceBuilder

    labels = [source[0] for source in args.source]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Use at least two sources with unique labels")
    if any(not label or any(c not in
           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
           for c in label) for label in labels):
        raise ValueError("Labels may contain only letters, digits, _ and -")
    if any(fmt not in ("chat", "instruction") for _, fmt, _ in args.source):
        raise ValueError("Source format must be chat or instruction")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Supply unique seeds")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    paths = [Path(path).resolve(strict=True) for _, _, path in args.source]
    root.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), python=platform.python_version(),
                  platform=platform.platform())
    (root / "experiment.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n")
    builder = TraceBuilder(args.model, revision=args.revision,
                           chat_template_kwargs={"date_string": "26 Jul 2024"})
    sources = {}
    for source_index, (label, fmt, _) in enumerate(args.source):
        source = paths[source_index]
        sources[label] = {"path": str(source), "format": fmt,
                          "sha256": file_hash(source)}
        dataset = source
        if fmt == "instruction":
            dataset = root / f"{label}-normalized.jsonl"
            sources[label]["conversion"] = convert_instructions(source, dataset)
        (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
        for seed in args.seeds:
            prepared = root / "prepared" / label / f"seed-{seed}"
            workload = prepare(dataset, prepared, builder, mode="multi",
                               limit=args.limit,
                               max_model_len=args.max_model_len,
                               arrival="poisson",
                               request_rate=args.request_rate,
                               seed=seed, think_time=args.think_time)
            if len(workload["conversations"]) < args.limit:
                raise ValueError(f"{label}: fewer than {args.limit} eligible "
                                 "conversations; lower --limit or add data")
            fingerprint = {
                "workload_sha256": file_hash(prepared / "workload.json"),
                "trace_sha256": file_hash(prepared / "trace.jsonl"),
            }
            # Avoid always measuring policies in the same host execution order.
            policies = random.Random(seed).sample(list(POLICIES), len(POLICIES))
            for order, policy in enumerate(policies):
                name = f"{label}-seed{seed}-{policy}"
                directory = root / label / f"seed-{seed}" / policy
                print(f"Running {name}", flush=True)
                with server(workload, prepared / "trace.jsonl", directory,
                            block_size=args.block_size, cache_gb=args.cache_gb,
                            policy=policy, prefill_ms=args.prefill_ms,
                            decode_ms=args.decode_ms,
                            max_num_seqs=args.concurrency,
                            startup_timeout=args.startup_timeout) as state:
                    url, cache = state
                    metadata = dict(
                        fingerprint, name=name, workload=label, policy=policy,
                        seed=seed, policy_order=order, model=args.model,
                        revision=args.revision, block_size=args.block_size,
                        cache_gb=args.cache_gb,
                        num_gpu_blocks=cache["num_gpu_blocks"],
                        virtual_cache_bytes=(cache["num_gpu_blocks"] *
                                             cache["block_bytes"]),
                        conversations=len(workload["conversations"]),
                        request_rate=args.request_rate,
                        concurrency=args.concurrency,
                        think_time_s=args.think_time,
                        prefill_ms_per_token=args.prefill_ms,
                        decode_ms_per_seq=args.decode_ms)
                    (directory / "run.json").write_text(
                        json.dumps(metadata, indent=2) + "\n")
                    rows = asyncio.run(replay(
                        workload, url, directory / "requests.jsonl",
                        concurrency=args.concurrency, timeout=args.timeout))
                if any(not row["success"] for row in rows):
                    raise RuntimeError(f"Failed requests in {directory}")
    return export_results(root, plots=not args.no_plots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=3, action="append", required=True,
                        metavar=("LABEL", "FORMAT", "JSON_PATH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--cache-gb", type=float, default=0.02)
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32],
                        default=16)
    parser.add_argument("--request-rate", type=float, default=100.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--think-time", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prefill-ms", type=float, default=0.01)
    parser.add_argument("--decode-ms", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if (not math.isfinite(args.cache_gb) or args.cache_gb <= 0
            or args.concurrency <= 0 or args.limit <= 0):
        parser.error("cache-gb, concurrency, and limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
