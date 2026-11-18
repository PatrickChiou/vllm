"""Milestone 3 Task 1: compare eviction policies on separate workloads."""
import argparse
import asyncio
import hashlib
import json
import math
import platform
import random
import statistics
from array import array
from itertools import zip_longest
from pathlib import Path

from vllm.simulator.analysis import analyze, write_csv
from vllm.simulator.replay import replay
from vllm.simulator.study import server
from vllm.simulator.workload import (prepare, prepare_block_trace,
                                     read_records)

POLICIES = ("lru", "lfu", "fifo")
FORMATS = ("chat", "instruction", "claude-code", "document-qa",
           "block-trace")
# Per-source settings accepted by --set LABEL KEY=VALUE.
SETTINGS = {"limit": int, "max_model_len": int, "max_output_tokens": int,
            "time_scale": float, "overflow": str,
            "seeds": lambda value: [int(s) for s in value.split(",")],
            "trace_block_size": int, "timestamp_unit": str}


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


def render_claude_content(content) -> str:
    """Render Claude Code message content as plain text.

    Tool calls and results use Qwen-style tags. Thinking blocks and images are
    omitted because they are not replayed as text context.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "tool_use":
            call = {"name": block.get("name"),
                    "arguments": block.get("input")}
            parts.append("<tool_call>\n" + json.dumps(call, ensure_ascii=False)
                         + "\n</tool_call>")
        elif kind == "tool_result":
            result = block.get("content")
            if not isinstance(result, str):
                result = "\n".join(item.get("text", "") for item in result or []
                                   if item.get("type") == "text")
            parts.append("<tool_response>\n" + result + "\n</tool_response>")
    return "\n".join(p for p in parts if p.strip())


def convert_claude_code(source: Path, destination: Path) -> dict:
    """Convert CC-Bench Claude Code trajectories into alternating messages.

    Only the main agent thread is kept; sidechains (sub-agents), summaries and
    system events are dropped. Consecutive events of one role, such as one
    response split into text and tool calls, are merged into one message.
    """
    counts = {"accepted": 0, "no_complete_turn": 0}
    with destination.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(read_records(source)):
            events = record.get("trajectory")
            if isinstance(events, str):
                events = json.loads(events)
            messages = []
            for event in events or []:
                if (event.get("isSidechain")
                        or event.get("type") not in ("user", "assistant")):
                    continue
                message = event.get("message") or {}
                role = message.get("role")
                text = render_claude_content(message.get("content"))
                if role not in ("user", "assistant") or not text.strip():
                    continue
                if not messages and role != "user":
                    continue
                if messages and messages[-1]["role"] == role:
                    messages[-1]["content"] += "\n" + text
                else:
                    messages.append({"role": role, "content": text})
            if not any(m["role"] == "assistant" for m in messages):
                counts["no_complete_turn"] += 1
                continue
            name = "{}-{}".format(record.get("model_name", "model"),
                                  record.get("task_id", index))
            stream.write(json.dumps({"id": name, "messages": messages},
                                    ensure_ascii=False) + "\n")
            counts["accepted"] += 1
    if not counts["accepted"]:
        raise ValueError("Claude Code dataset has no usable trajectories")
    return counts


def convert_document_qa(source: Path, destination: Path, *,
                        group: int = 10, seed: int = 0) -> dict:
    """Convert QuALITY-style multiple-choice questions over long articles.

    Each question is an independent single-turn request whose prompt starts
    with the full article, as in retrieval-augmented serving. Questions of
    consecutive groups of ``group`` articles are shuffled together with a
    fixed seed, so questions on different articles interleave instead of
    arriving back to back; a request limit therefore covers whole groups.
    """
    articles: dict = {}
    counts = {"accepted": 0, "invalid": 0}
    for record in read_records(source):
        article, question = record.get("article"), record.get("question")
        options, answer = record.get("options"), record.get("answer")
        if (not isinstance(article, str) or not isinstance(question, str)
                or not isinstance(options, list) or not options
                or not isinstance(answer, int)
                or not 0 <= answer < len(options)):
            counts["invalid"] += 1
            continue
        letters = "ABCDEFGHIJ"
        choices = "\n".join(f"{letters[i]}. {option}"
                            for i, option in enumerate(options))
        articles.setdefault(article, []).append({
            "messages": [
                {"role": "user",
                 "content": f"{article}\n\nQuestion: {question}\n{choices}\n"
                 "Answer with the letter of the correct option."},
                {"role": "assistant",
                 "content": f"{letters[answer]}. {options[answer]}"},
            ]})
    rng = random.Random(seed)
    texts = list(articles)
    with destination.open("w", encoding="utf-8") as stream:
        for start in range(0, len(texts), group):
            batch = [(index, number, item)
                     for index, text in enumerate(texts[start:start + group],
                                                  start)
                     for number, item in enumerate(articles[text])]
            rng.shuffle(batch)
            for article_index, number, item in batch:
                item = dict(item, id=f"article{article_index}-q{number}")
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts["accepted"] += 1
    if not counts["accepted"]:
        raise ValueError("Document QA dataset has no usable questions")
    counts["articles"] = len(texts)
    return counts


def interleave(sources: list, destination: Path) -> dict:
    """Merge chat datasets round-robin so a record limit samples each one."""
    counts = {}
    streams = [read_records(path) for path in sources]
    with destination.open("w", encoding="utf-8") as stream:
        for row in zip_longest(*streams):
            for path, record in zip(sources, row):
                if record is None:
                    continue
                raw = record.get("conversations", record.get("messages"))
                index = counts.get(path.stem, 0)
                counts[path.stem] = index + 1
                stream.write(json.dumps({
                    "id": f"{path.stem}-{record.get('id', index)}",
                    "conversations": raw}, ensure_ascii=False) + "\n")
    return counts


def working_set(trace_path: Path, block_size: int) -> dict:
    """Distinct full KV blocks the trace would cache with unlimited capacity.

    A request caches the full blocks of its prompt plus generated tokens; the
    final EOS is never computed. Blocks are identified by their whole prefix,
    as in vLLM's prefix-caching allocator.
    """
    seen, longest = set(), 0
    for record in read_records(trace_path):
        tokens = array("q", record["prompt_token_ids"] +
                       record["output_token_ids"])
        longest = max(longest, len(tokens) + 1)
        digest = hashlib.sha256()
        for start in range(0, len(tokens) - block_size + 1, block_size):
            digest.update(tokens[start:start + block_size].tobytes())
            seen.add(digest.copy().digest())
    return {"unique_blocks": len(seen), "longest_request_tokens": longest}


def cache_blocks(ratio: float, footprint: dict, block_size: int) -> dict:
    """Size a cache as a fraction of the working set.

    The result is raised to the smallest capacity that still admits the
    longest request after vLLM's 1% watermark; such runs are marked clamped.
    """
    wanted = max(1, math.ceil(ratio * footprint["unique_blocks"]))
    needed = math.ceil(footprint["longest_request_tokens"] / block_size)
    minimum = needed
    while minimum - int(minimum * 0.01) < needed:
        minimum += 1
    return {"num_gpu_blocks": max(wanted, minimum),
            "cache_clamped": wanted < minimum}


def rank_results(rows: list) -> list:
    """Rank observed sharing, requiring a complete, comparable experiment."""
    if not rows:
        raise ValueError("No completed runs")
    if len({r["block_size"] for r in rows}) != 1:
        raise ValueError("Block size must be fixed")
    groups, cohorts, geometry, seen = {}, {}, {}, set()
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
        if geometry.setdefault(row["workload"], row["virtual_cache_bytes"]) \
                != row["virtual_cache_bytes"]:
            raise ValueError("Cache capacity must be fixed within a workload")
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
                "num_gpu_blocks": runs[0].get("num_gpu_blocks"),
                "working_set_blocks": runs[0].get("working_set_blocks"),
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
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 4))
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


def source_settings(args) -> dict:
    """Merge global defaults with --set LABEL KEY=VALUE overrides."""
    labels = [source[0] for source in args.source]
    settings = {label: {"limit": args.limit,
                        "max_model_len": args.max_model_len,
                        "max_output_tokens": args.max_output_tokens,
                        "time_scale": args.time_scale,
                        "seeds": list(args.seeds),
                        "trace_block_size": 16, "timestamp_unit": "s",
                        "overflow": ("truncate" if fmt == "claude-code"
                                     else "reject")}
                for label, fmt, _ in args.source}
    for label, assignment in args.set or []:
        key, _, value = assignment.partition("=")
        if label not in labels or key not in SETTINGS or not value:
            raise ValueError(f"Invalid --set {label} {assignment}")
        settings[label][key] = SETTINGS[key](value)
    for label, values in settings.items():
        if (values["limit"] <= 0 or values["max_model_len"] <= 0
                or values["max_output_tokens"] < 0
                or not 0 < values["time_scale"] < math.inf
                or values["overflow"] not in ("reject", "truncate")
                or not values["seeds"]
                or len(set(values["seeds"])) != len(values["seeds"])
                or values["trace_block_size"] <= 0
                or values["timestamp_unit"] not in ("s", "ms")):
            raise ValueError(f"Invalid settings for {label}: {values}")
    return settings


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
    if any(fmt not in FORMATS for _, fmt, _ in args.source):
        raise ValueError(f"Source format must be one of {FORMATS}")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Supply unique seeds")
    settings = source_settings(args)
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    paths = [[Path(p).resolve(strict=True) for p in spec.split(",")]
             for _, _, spec in args.source]
    root.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), settings=settings,
                  python=platform.python_version(),
                  platform=platform.platform())
    (root / "experiment.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n")
    builder = TraceBuilder(args.model, revision=args.revision,
                           chat_template_kwargs={"date_string": "26 Jul 2024"})
    sources = {}
    for source_index, (label, fmt, _) in enumerate(args.source):
        options = settings[label]
        files = paths[source_index]
        sources[label] = {"paths": [str(p) for p in files], "format": fmt,
                          "sha256": [file_hash(p) for p in files],
                          "settings": options}
        if len(files) > 1 and fmt != "chat":
            raise ValueError("Only chat sources may combine several files")
        dataset = files[0]
        if len(files) > 1:
            dataset = root / f"{label}-interleaved.jsonl"
            sources[label]["conversion"] = interleave(files, dataset)
        elif fmt == "instruction":
            dataset = root / f"{label}-normalized.jsonl"
            sources[label]["conversion"] = convert_instructions(
                files[0], dataset)
        elif fmt == "claude-code":
            dataset = root / f"{label}-normalized.jsonl"
            sources[label]["conversion"] = convert_claude_code(
                files[0], dataset)
        elif fmt == "document-qa":
            dataset = root / f"{label}-normalized.jsonl"
            sources[label]["conversion"] = convert_document_qa(
                files[0], dataset)
        (root / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
        for seed in options["seeds"]:
            prepared = root / "prepared" / label / f"seed-{seed}"
            if fmt == "block-trace":
                # Recorded arrivals: seeds repeat the same input schedule.
                workload = prepare_block_trace(
                    dataset, prepared, builder, limit=options["limit"],
                    max_model_len=options["max_model_len"],
                    max_output_tokens=options["max_output_tokens"],
                    trace_block_size=options["trace_block_size"],
                    timestamp_unit=options["timestamp_unit"])
                size = sum(len(c["turns"]) for c in workload["conversations"])
            else:
                workload = prepare(dataset, prepared, builder, mode="multi",
                                   limit=options["limit"],
                                   max_model_len=options["max_model_len"],
                                   arrival="poisson",
                                   request_rate=args.request_rate,
                                   seed=seed, think_time=args.think_time,
                                   overflow=options["overflow"])
                size = len(workload["conversations"])
            if size < options["limit"] and fmt != "block-trace":
                raise ValueError(f"{label}: fewer than {options['limit']} "
                                 "eligible conversations; lower the limit "
                                 "or add data")
            fingerprint = {
                "workload_sha256": file_hash(prepared / "workload.json"),
                "trace_sha256": file_hash(prepared / "trace.jsonl"),
            }
            footprint = working_set(prepared / "trace.jsonl", args.block_size)
            sizing = ({"num_gpu_blocks": None, "cache_clamped": False}
                      if args.cache_ratio is None else
                      cache_blocks(args.cache_ratio, footprint,
                                   args.block_size))
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
                            startup_timeout=args.startup_timeout,
                            num_gpu_blocks=sizing["num_gpu_blocks"]) as state:
                    url, cache = state
                    metadata = dict(
                        fingerprint, name=name, workload=label, policy=policy,
                        format=fmt, seed=seed, policy_order=order,
                        model=args.model, revision=args.revision,
                        block_size=args.block_size,
                        cache_gb=(args.cache_gb if args.cache_ratio is None
                                  else None),
                        cache_ratio=args.cache_ratio,
                        cache_clamped=sizing["cache_clamped"],
                        working_set_blocks=footprint["unique_blocks"],
                        num_gpu_blocks=cache["num_gpu_blocks"],
                        virtual_cache_bytes=(cache["num_gpu_blocks"] *
                                             cache["block_bytes"]),
                        conversations=len(workload["conversations"]),
                        arrival=workload["arrival"],
                        request_rate=(args.request_rate
                                      if fmt != "block-trace" else None),
                        time_scale=options["time_scale"],
                        concurrency=args.concurrency,
                        think_time_s=args.think_time,
                        prefill_ms_per_token=args.prefill_ms,
                        decode_ms_per_seq=args.decode_ms)
                    (directory / "run.json").write_text(
                        json.dumps(metadata, indent=2) + "\n")
                    rows = asyncio.run(replay(
                        workload, url, directory / "requests.jsonl",
                        concurrency=args.concurrency,
                        time_scale=options["time_scale"],
                        timeout=args.timeout))
                if any(not row["success"] for row in rows):
                    raise RuntimeError(f"Failed requests in {directory}")
    return export_results(root, plots=not args.no_plots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=3, action="append", required=True,
                        metavar=("LABEL", "FORMAT", "PATHS"),
                        help=f"FORMAT is one of {', '.join(FORMATS)}; chat "
                        "sources may list several comma-separated files")
    parser.add_argument("--set", nargs=2, action="append",
                        metavar=("LABEL", "KEY=VALUE"),
                        help="Per-source override of " + ", ".join(SETTINGS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--limit", type=int, default=100,
                        help="Conversations, or requests for block traces")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=128,
                        help="Generated-token cap for block traces")
    parser.add_argument("--cache-gb", type=float, default=0.02)
    parser.add_argument("--cache-ratio", type=float,
                        help="Size each workload's cache as this fraction of "
                        "its working set instead of --cache-gb")
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32],
                        default=16)
    parser.add_argument("--request-rate", type=float, default=100.0)
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Arrival-time multiplier, e.g. 0.1 replays a "
                        "recorded trace ten times faster")
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
    if args.cache_ratio is not None and not 0 < args.cache_ratio <= 1:
        parser.error("cache-ratio must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
