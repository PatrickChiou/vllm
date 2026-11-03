"""Export study measurements and publication-friendly matplotlib figures."""
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cdf(values: Iterable[float]):
    ordered = sorted(values)
    return ordered, [(i + 1) / len(ordered) for i in range(len(ordered))]


def summarize(rows: List[dict], events: List[dict]) -> dict:
    successful = [r for r in rows if r["success"]]
    prompt = sum(r["prompt_tokens"] for r in successful)
    cached = sum(r["cached_tokens"] for r in successful)
    blocks: Dict[str, dict] = {}
    gaps = []
    for event in events:
        if event["event"] != "block":
            continue
        block = blocks.setdefault(event["block_key"], {
            "block_key": event["block_key"], "accesses": 0, "hits": 0})
        block["accesses"] += 1
        block["hits"] += int(event["hit"])
        if event["reuse_gap_s"] is not None:
            gaps.append(event["reuse_gap_s"])
    elapsed = max((r.get("finished_s", 0) for r in rows), default=0)
    result = {
        "requests": len(rows), "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "prompt_tokens": prompt, "cached_tokens": cached,
        "token_weighted_shared_fraction": cached / prompt if prompt else 0.0,
        "mean_request_shared_fraction": (
            sum(r["shared_fraction"] for r in successful) / len(successful)
            if successful else 0.0),
        "unique_blocks": len(blocks),
        "block_hits": sum(b["hits"] for b in blocks.values()),
        "zero_hit_blocks": sum(b["hits"] == 0 for b in blocks.values()),
        "reuse_gap_samples": len(gaps),
        "mean_reuse_gap_s": sum(gaps) / len(gaps) if gaps else None,
        "elapsed_s": elapsed,
        "requests_per_s": len(successful) / elapsed if elapsed else 0.0,
    }
    return {"summary": result, "blocks": list(blocks.values()), "gaps": gaps}


def analyze(root: Path, *, plots: bool = True) -> List[dict]:
    runs = sorted(root.rglob("run.json"))
    if not runs:
        raise ValueError(f"No run.json files under {root}")
    comparisons, distributions = [], []
    for config_path in runs:
        directory = config_path.parent
        config = json.loads(config_path.read_text())
        rows = read_jsonl(directory / "requests.jsonl")
        events = read_jsonl(directory / "blocks.jsonl")
        result = summarize(rows, events)
        summary = dict(config, **result["summary"])
        (directory / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        write_csv(directory / "requests.csv", rows, [
            "request_id", "conversation_id", "turn", "success", "prompt_tokens",
            "completion_tokens", "cached_tokens", "shared_fraction",
            "planned_s",
            "sent_s", "dispatch_lag_s", "latency_s", "finished_s", "error"])
        write_csv(directory / "block_hits.csv", result["blocks"],
                  ["block_key", "accesses", "hits"])
        write_csv(directory / "reuse_gaps.csv", [
            e for e in events
            if e["event"] == "block" and e["reuse_gap_s"] is not None],
            ["block_key", "request_id", "time_s", "reuse_gap_s"])
        comparisons.append(summary)
        fractions = [r["shared_fraction"] for r in rows if r["success"]]
        hits = [b["hits"] for b in result["blocks"]]
        distributions.append((config["name"], fractions, hits, result["gaps"]))
    (root / "comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n")
    write_csv(root / "comparison.csv", comparisons, list(comparisons[0]))
    if plots:
        plot(root, comparisons, distributions)
    return comparisons


def plot(root: Path, comparisons: List[dict], distributions: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires requirements-simulator.txt; "
                          "use analyze --no-plots for CSV/JSON only") from exc
    for index, (filename, xlabel) in enumerate([
            ("shared_fraction_cdf", "Prompt fraction served by prefix cache"),
            ("block_hits_cdf", "Cache hits per logical prefix block"),
            ("reuse_gap_cdf", "Seconds since previous request used block")], 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        for distribution in distributions:
            values, fractions = cdf(distribution[index])
            if values:
                ax.step(values, fractions, where="post", label=distribution[0])
        ax.set(xlabel=xlabel, ylabel="CDF", ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(0.5, 0.5, "No observed reuse", ha="center",
                    transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(root / f"{filename}.png", dpi=180)
        fig.savefig(root / f"{filename}.pdf")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, len(comparisons) * 0.45), 5))
    ax.bar([c["name"] for c in comparisons],
           [c["token_weighted_shared_fraction"] for c in comparisons])
    ax.set(ylabel="Cached prompt tokens / all prompt tokens", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(root / "cache_comparison.png", dpi=180)
    fig.savefig(root / "cache_comparison.pdf")
    plt.close(fig)
