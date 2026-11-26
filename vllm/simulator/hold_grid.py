"""Replay mixed workloads across TLRU, HLRU, and RFIFO settings.

Baselines run once per mix. TLRU and HLRU read hold and budget. RFIFO reads
window and threshold. Progress is printed and written to ``progress.json``
under the output directory.

    python -m vllm.simulator.hold_grid --dry-run
    python -m vllm.simulator.hold_grid --revision "$MODEL_REVISION"
"""
import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

BASELINES = ("lru", "lfu", "fifo", "tfifo")
SWEEP_POLICIES = ("tlru", "hlru")
# 3s covers the 1s think time; 0s and 1s expired before the next turn.
# budget 0.5 sat between 0.25 and 1.
HOLDS = (3, 8, 30)
BUDGETS = (0.25, 1.0)
RFIFO_WINDOWS = (20, 100, 500)
RFIFO_THRESHOLDS = (0.02, 0.05, 0.2)
# Mixes are about twice the previous token mass, except coding-agent.
REPLAY_MINUTES = 6
PREPARE_MINUTES = 15
SOURCES = (
    ("chat", "chat", "data/sharegpt.json"),
    ("agent", "chat",
     "data/m3/agentbank_alfworld.parquet,data/m3/agentbank_webshop.parquet,"
     "data/m3/agentbank_intercode_bash.parquet,"
     "data/m3/agentbank_mind2web.parquet"),
    ("coding-agent", "claude-code", "data/m3/cc_bench.parquet"),
    ("quality", "document-qa", "data/m3/quality_validation.parquet"),
    ("bailian-chat", "block-trace", "data/m3/bailian_traceA.jsonl"),
    ("bailian-coder", "block-trace", "data/m3/bailian_coder.jsonl"),
    ("mooncake-agent", "block-trace", "data/m3/mooncake_toolagent.jsonl"),
    ("code", "instruction", "data/code_alpaca.json"),
)
# Filler limits, twice the previous grid. coding-agent stays at or below 12:
# CC-Bench has 370 trajectories, but the 16th kept session is the long one.
_SMALL = {
    "chat": 40,
    "agent": 30,
    "coding-agent": 6,
    "quality": 10,
    "code": 400,
    "bailian-chat": 40,
    "bailian-coder": 16,
    "mooncake-agent": 8,
}


def _limits(**overrides):
    limits = dict(_SMALL)
    limits.update(overrides)
    return limits


# Twice the m3-hold-grid2 limits, except coding-agent. Ceilings checked
# against the source files are in validate_mixes.
MIXES = (
    ("chat-heavy", _limits(chat=1200, code=600)),
    ("protected-heavy", _limits(**{
        "agent": 200,
        "coding-agent": 12,
        "bailian-chat": 310,
        "bailian-coder": 124,
    })),
    ("trace-heavy", _limits(**{
        "bailian-chat": 800,
        "bailian-coder": 300,
        "mooncake-agent": 62,
    })),
    ("template-heavy", _limits(code=9800, quality=80)),
    # coding-agent stays at 12, so its token share is smaller than in grid2.
    ("balanced", _limits(**{
        "chat": 220,
        "agent": 200,
        "coding-agent": 12,
        "quality": 80,
        "code": 9800,
        "bailian-chat": 310,
        "bailian-coder": 124,
        "mooncake-agent": 62,
    })),
)


def _spec(policy, hold=None, budget=None, window=None, threshold=None):
    return {"policy": policy, "hold": hold, "budget": budget,
            "window": window, "threshold": threshold}


def run_specs():
    """Baselines once, each RFIFO window and threshold, then a short hold grid."""
    specs = [_spec(policy) for policy in BASELINES]
    for window in RFIFO_WINDOWS:
        for threshold in RFIFO_THRESHOLDS:
            specs.append(_spec("rfifo", window=window, threshold=threshold))
    for policy in SWEEP_POLICIES:
        for hold in HOLDS:
            for budget in BUDGETS:
                specs.append(_spec(policy, hold, budget))
    return specs


def run_name(spec) -> str:
    if spec["policy"] == "rfifo":
        return f"rfifo-w{spec['window']:g}-t{spec['threshold']:g}"
    if spec["hold"] is None:
        return spec["policy"]
    return f"{spec['policy']}-h{spec['hold']:g}-b{spec['budget']:g}"


def bar_order(names) -> list:
    """Put LRU, LFU, and FIFO first, then the other baselines, then the grid."""
    rank = {run_name(spec): index for index, spec in enumerate(run_specs())}
    names = list(names)
    unknown = [name for name in names if name not in rank]
    if unknown:
        raise ValueError(f"Unknown run names: {unknown}")
    return sorted(names, key=rank.__getitem__)


def prompt_token_count(workload: dict) -> int:
    return sum(turn["prompt_tokens"]
               for conversation in workload["conversations"]
               for turn in conversation["turns"])


def label_fractions(rows) -> dict:
    totals = {}
    for row in rows:
        label = row.get("workload")
        if not row.get("success") or not label:
            continue
        prompt, cached = totals.setdefault(label, [0, 0])
        totals[label] = [prompt + row["prompt_tokens"],
                         cached + row["cached_tokens"]]
    return {label: (cached / prompt if prompt else 0.0)
            for label, (prompt, cached) in totals.items()}


def validate_mixes() -> None:
    labels = {label for label, _, _ in SOURCES}
    for name, limits in MIXES:
        if set(limits) != labels:
            raise ValueError(f"{name} limits do not cover the eight sources")
        if any(value <= 0 for value in limits.values()):
            raise ValueError(f"{name} has a non-positive limit")
        # CC-Bench row 16 is the ~500,000-token session. The file has 370 rows.
        if limits["coding-agent"] > 12:
            raise ValueError(f"{name} coding-agent limit exceeds 12")
        # 1,300 eligible conversations were found in the first 2,162 records.
        if limits["chat"] > 1300:
            raise ValueError(f"{name} chat limit exceeds 1300")
        # Sum of the four AgentBank files. The first 250 records all fit.
        if limits["agent"] > 16249:
            raise ValueError(f"{name} agent limit exceeds 16249")
        if limits["quality"] > 2086:
            raise ValueError(f"{name} quality limit exceeds 2086")
        if limits["code"] > 20016:
            raise ValueError(f"{name} code limit exceeds 20016")
        if limits["bailian-chat"] > 43058:
            raise ValueError(f"{name} bailian-chat limit exceeds 43058")
        if limits["bailian-coder"] > 43011:
            raise ValueError(f"{name} bailian-coder limit exceeds 43011")
        if limits["mooncake-agent"] > 23608:
            raise ValueError(f"{name} mooncake-agent limit exceeds 23608")


def estimate_minutes() -> float:
    return len(MIXES) * (len(run_specs()) * REPLAY_MINUTES + PREPARE_MINUTES)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def eta_seconds(completed: int, elapsed_s: float, total: int) -> float:
    """Remaining time. Before the first finish, subtract elapsed from the estimate."""
    if completed <= 0:
        return max(0.0, estimate_minutes() * 60 - elapsed_s)
    return elapsed_s / completed * (total - completed)


def progress_line(completed: int, total: int, mix: str, mix_index: int,
                  mix_count: int, run, elapsed_s: float, phase: str) -> str:
    where = f"{mix} ({mix_index}/{mix_count})"
    if run:
        where = f"{where} {run}"
    eta = eta_seconds(completed, elapsed_s, total)
    return (f"[{completed}/{total}] {phase} {where} "
            f"elapsed {format_duration(elapsed_s)} "
            f"eta {format_duration(eta)}")


def print_plan() -> None:
    per_mix = len(run_specs())
    print(f"{len(MIXES)} mixes x {per_mix} runs = {len(MIXES) * per_mix}")
    print(f"RFIFO windows {RFIFO_WINDOWS} thresholds {RFIFO_THRESHOLDS}")
    print(f"TLRU/HLRU holds {HOLDS} budgets {BUDGETS}")
    print(f"Estimated time: {estimate_minutes() / 60:.1f} hours")
    for name, limits in MIXES:
        shown = ", ".join(f"{label}={limits[label]}" for label, _, _ in SOURCES)
        print(f"{name}: {shown}")


def prepared_ready(root: Path) -> bool:
    return all((root / label / "workload.json").is_file()
               and (root / label / "trace.jsonl").is_file()
               for label, _, _ in SOURCES)


def limit_stamp(limits: dict) -> dict:
    return {label: limits[label] for label, _, _ in SOURCES}


def prepared_matches(root: Path, limits: dict) -> bool:
    stamp_path = root / "limits.json"
    if not prepared_ready(root) or not stamp_path.is_file():
        return False
    saved = json.loads(stamp_path.read_text(encoding="utf-8"))
    return saved == limit_stamp(limits)


def prepare_namespace(output: Path, revision: str, limits: dict):
    sets = [(label, f"limit={limits[label]}") for label, _, _ in SOURCES]
    sets.append(("mooncake-agent", "trace_block_size=512"))
    sets.append(("mooncake-agent", "timestamp_unit=ms"))
    return argparse.Namespace(
        source=list(SOURCES),
        set=sets,
        output=output,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        revision=revision,
        limit=1,
        max_model_len=32768,
        max_output_tokens=32,
        request_rate=100.0,
        seed=0,
    )


def read_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_succeeded(directory: Path) -> bool:
    path = directory / "requests.jsonl"
    if not path.is_file():
        return False
    rows = read_jsonl(path)
    return bool(rows) and all(row.get("success") for row in rows)


def plot_bars(root: Path, comparisons: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {name: index for index, name in enumerate(
        bar_order(row["name"] for row in comparisons))}
    ordered = sorted(comparisons, key=lambda row: order[row["name"]])
    fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 0.45), 5))
    ax.bar([row["name"] for row in ordered],
           [row["token_weighted_shared_fraction"] for row in ordered])
    ax.set(ylabel="Cached prompt tokens / all prompt tokens", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(root / "cache_comparison.png", dpi=180)
    fig.savefig(root / "cache_comparison.pdf")
    plt.close(fig)


def collect_summary(root: Path) -> dict:
    mixes = []
    for name, _limits in MIXES:
        mix_dir = root / name
        workload_path = mix_dir / "mixed" / "workload.json"
        entry = {"name": name, "runs": []}
        if not workload_path.is_file():
            mixes.append(entry)
            continue
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        entry["prompt_tokens"] = prompt_token_count(workload)
        runs_dir = mix_dir / "runs"
        if runs_dir.is_dir():
            for spec in run_specs():
                directory = runs_dir / run_name(spec)
                if not run_succeeded(directory):
                    continue
                rows = read_jsonl(directory / "requests.jsonl")
                successful = [row for row in rows if row["success"]]
                prompt = sum(row["prompt_tokens"] for row in successful)
                cached = sum(row["cached_tokens"] for row in successful)
                entry["runs"].append({
                    "name": run_name(spec),
                    "policy": spec["policy"],
                    "hold_s": spec["hold"],
                    "hold_budget": spec["budget"],
                    "rfifo_window": spec["window"],
                    "rfifo_threshold": spec["threshold"],
                    "token_weighted_shared_fraction": (
                        cached / prompt if prompt else 0.0),
                    "by_workload": label_fractions(rows),
                })
        mixes.append(entry)
    return {"mixes": mixes}


class Progress:
    """Print a grid position and keep the same record in progress.json."""

    def __init__(self, root: Path, started: float):
        self.root = root
        self.started = started
        self.total = len(MIXES) * len(run_specs())
        self.finished = []
        for mix_name, _limits in MIXES:
            for spec in run_specs():
                name = run_name(spec)
                directory = root / mix_name / "runs" / name
                if run_succeeded(directory):
                    self.finished.append(f"{mix_name}/{name}")
        self.mix = None
        self.mix_index = 0
        self.run = None
        self.phase = "starting"

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def line(self) -> str:
        return progress_line(
            len(self.finished), self.total, self.mix or "-",
            self.mix_index, len(MIXES), self.run, self.elapsed(), self.phase)

    def update(self, mix: str, mix_index: int, phase: str, run=None) -> None:
        self.mix = mix
        self.mix_index = mix_index
        self.phase = phase
        self.run = run
        payload = {
            "completed": len(self.finished),
            "total": self.total,
            "phase": phase,
            "mix": mix,
            "mix_index": mix_index,
            "mix_count": len(MIXES),
            "run": run,
            "elapsed_s": round(self.elapsed(), 1),
            "eta_s": round(eta_seconds(
                len(self.finished), self.elapsed(), self.total), 1),
            "finished": self.finished,
        }
        path = self.root / "progress.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(self.line(), flush=True)

    def complete(self, mix: str, run: str) -> None:
        key = f"{mix}/{run}"
        if key not in self.finished:
            self.finished.append(key)


def replay_mix(args, mix_dir: Path, workload: dict, progress: Progress,
               mix_index: int) -> None:
    """Replay every cell of one mix."""
    from vllm.simulator.analysis import analyze
    from vllm.simulator.eviction_study import cache_blocks, working_set
    from vllm.simulator.replay import replay
    from vllm.simulator.study import server

    trace_path = mix_dir / "mixed" / "trace.jsonl"
    footprint = working_set(trace_path, args.block_size)
    sizing = cache_blocks(args.cache_ratio, footprint, args.block_size)
    runs = mix_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for spec in run_specs():
        name = run_name(spec)
        directory = runs / name
        if run_succeeded(directory):
            progress.complete(mix_dir.name, name)
            progress.update(mix_dir.name, mix_index, "skip", name)
            continue
        if directory.exists():
            shutil.rmtree(directory)
        progress.update(mix_dir.name, mix_index, "run", name)
        try:
            with server(workload, trace_path, directory,
                        block_size=args.block_size, cache_gb=args.cache_gb,
                        policy=spec["policy"], prefill_ms=args.prefill_ms,
                        decode_ms=args.decode_ms,
                        max_num_seqs=args.concurrency,
                        startup_timeout=args.startup_timeout,
                        num_gpu_blocks=sizing["num_gpu_blocks"],
                        hold_s=spec["hold"], hold_budget=spec["budget"],
                        rfifo_window=spec["window"],
                        rfifo_threshold=spec["threshold"]) as state:
                url, cache = state
                metadata = {
                    "name": name, "policy": spec["policy"],
                    "hold_s": spec["hold"], "hold_budget": spec["budget"],
                    "rfifo_window": spec["window"],
                    "rfifo_threshold": spec["threshold"],
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
                done = len(progress.finished) + 1
                prefix = (f"[{done}/{progress.total}] "
                          f"{mix_dir.name}/{name}")
                rows = asyncio.run(replay(
                    workload, url, directory / "requests.jsonl",
                    concurrency=args.concurrency, timeout=args.timeout,
                    progress_prefix=prefix))
        except Exception:
            progress.update(mix_dir.name, mix_index, "failed", name)
            raise
        if any(not row["success"] for row in rows):
            progress.update(mix_dir.name, mix_index, "failed", name)
            raise RuntimeError(f"Failed requests in {directory}")
        progress.complete(mix_dir.name, name)
        progress.update(mix_dir.name, mix_index, "done", name)
    if any(runs.glob("*/run.json")):
        progress.update(mix_dir.name, mix_index, "plot")
        comparisons = analyze(runs, plots=True)
        plot_bars(runs, comparisons)


def run_grid(args, progress: Progress) -> None:
    from vllm.simulator.mix_workload import mix
    from vllm.simulator.mixed_prepare import prepare_sources

    for mix_index, (name, limits) in enumerate(MIXES, start=1):
        mix_dir = args.output / name
        prepared = mix_dir / "prepared"
        if prepared_matches(prepared, limits):
            progress.update(name, mix_index, "reuse-prepared")
        elif prepared.exists() and any(prepared.iterdir()):
            raise FileExistsError(
                f"{prepared} does not match these limits; use a new --output")
        else:
            if prepared.exists():
                prepared.rmdir()
            progress.update(name, mix_index, "prepare")
            prepare_sources(prepare_namespace(prepared, args.revision, limits))
            (prepared / "limits.json").write_text(
                json.dumps(limit_stamp(limits), indent=2) + "\n",
                encoding="utf-8")
        mixed = mix_dir / "mixed"
        if ((mixed / "workload.json").is_file()
                and (mixed / "trace.jsonl").is_file()):
            progress.update(name, mix_index, "reuse-mixed")
        else:
            progress.update(name, mix_index, "mix")
            mix([(label, prepared / label) for label, _, _ in SOURCES],
                mixed, 100.0, 1.0, 0)
        workload = json.loads(
            (mixed / "workload.json").read_text(encoding="utf-8"))
        tokens = prompt_token_count(workload)
        print(f"{name}: {tokens} prompt tokens", flush=True)
        replay_mix(args, mix_dir, workload, progress, mix_index)
    progress.update(MIXES[-1][0], len(MIXES), "finished")


def write_summary(args) -> None:
    summary = collect_summary(args.output)
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/m3-hold-grid3"))
    parser.add_argument("--revision",
                        default=os.environ.get("MODEL_REVISION"))
    parser.add_argument("--cache-ratio", type=float, default=0.1)
    parser.add_argument("--cache-gb", type=float, default=0.02)
    parser.add_argument("--block-size", type=int, choices=[8, 16, 32],
                        default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prefill-ms", type=float, default=0.01)
    parser.add_argument("--decode-ms", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_mixes()
    if not 0 < args.cache_ratio <= 1:
        parser.error("cache-ratio must be in (0, 1]")
    if args.dry_run:
        print_plan()
        return
    if not args.revision:
        parser.error("pass --revision or set MODEL_REVISION")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.output, time.monotonic())
    print(f"Progress file: {args.output / 'progress.json'}", flush=True)
    progress.update("-", 0, "starting")
    try:
        run_grid(args, progress)
    finally:
        write_summary(args)


if __name__ == "__main__":
    main()
