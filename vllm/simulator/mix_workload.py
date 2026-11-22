"""Merge prepared Task 1 workloads into one interleaved mixed workload."""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_prepared(path: Path) -> tuple:
    """Read one prepared directory. Traces are indexed by their request id."""
    workload_path = path / "workload.json"
    trace_path = path / "trace.jsonl"
    if not workload_path.is_file() or not trace_path.is_file():
        raise ValueError(f"{path} needs workload.json and trace.jsonl")
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    traces = {}
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            entry = json.loads(line)
            request_id = entry["request_id"]
            if request_id in traces:
                raise ValueError(f"Duplicate trace request id: {request_id}")
            traces[request_id] = entry
    return workload, traces

def same_setup(workload: dict) -> tuple:
    """Fields that must agree before two prepared runs can share one server."""
    kwargs = workload.get("chat_template_kwargs") or {}
    return (
        workload["model"],
        workload.get("revision"),
        workload.get("tokenizer"),
        workload.get("max_model_len"),
        json.dumps(kwargs, sort_keys=True),
    )

def retag(conversation: dict, label: str, traces: dict) -> tuple:
    """Give this conversation ids that cannot collide with another source.
    
    The server stores chat requests as chatcmpl-<body request_id>, and the 
    trace store looks them up with that same string.
    """
    copied = copy.deepcopy(conversation)
    copied["id"] = f"{label}:{copied['id']}"
    copied["workload"] = label
    rewritten = []
    for turn in copied["turns"]:
        old = turn["request_id"]
        new = f"m3t2-{label}-{old}"
        if "prompt_token_ids" in turn:
            trace_key = f"cmpl-{old}-0"
            new_trace_id = f"cmpl-{new}-0"
        else:
            trace_key = f"chatcmpl-{old}"
            new_trace_id = f"chatcmpl-{new}"
        if trace_key not in traces:
            raise ValueError(f"Missing trace for {trace_key}")
        entry = dict(traces[trace_key])
        entry["request_id"] = new_trace_id
        turn["request_id"] = new
        turn["workload"] = label
        rewritten.append(entry)
    return copied, rewritten

def schedule(conversations: list, request_rate: float, think_time: float, seed: int) -> list:
    """Shuffle conversations, then place each follow-up one think time later.

    Only the first turn draws from the Poisson stream. A later turn is placed
    at the previous turn's planned time plus think_time, so it can sit among
    other conversations' first turns.
    """
    rng = random.Random(seed)
    ordered = conversations[:]
    rng.shuffle(ordered)
    offset = 0.0
    for conversation in ordered:
        offset += rng.expovariate(request_rate)
        arrival = offset
        previous = -1.0
        for turn in conversation["turns"]:
            if arrival <= previous:
                raise ValueError("Turn arrivals must increase")
            turn["arrival_time_s"] = arrival
            previous = arrival
            arrival += think_time
    return ordered

def mix(sources: list, output: Path, request_rate: float, think_time: float, seed: int) -> dict:

    if not math.isfinite(think_time) or think_time <= 0.0:
        raise ValueError("think_time must be finite and positive")
    if not math.isfinite(request_rate) or request_rate <= 0.0:
        raise ValueError("request_rate must be finite and positive")
    labels = [label for label,_ in sources]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Use at least two sources with unique labels")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")

    loaded = []
    for label,path in sources:
        workload, traces = load_prepared(path)
        loaded.append((label, path, workload, traces))
    expected = same_setup(loaded[0][2])
    for label, _, workload, _ in loaded[1:]:
        if same_setup(workload) != expected:
            raise ValueError(f"{label} was prepared with a different model setup")
    
    conversations = []
    seen = set()
    for label, _, workload, traces in loaded:
        for conversation in workload["conversations"]:
            copied, entries = retag(conversation, label, traces)
            for turn in copied["turns"]:
                if turn["request_id"] in seen:
                    raise ValueError(f"Duplicate request id: {turn['request_id']}")
                seen.add(turn["request_id"])
            copied["_traces"] = entries
            conversations.append(copied)
    conversations = schedule(conversations, request_rate, think_time, seed)

    traces_out = []
    for conversation in conversations:
        traces_out.extend(conversation.pop("_traces"))
    first = loaded[0][2]
    manifest = {
        "version": 1,
        "model": first["model"],
        "tokenizer": first.get("tokenizer"),
        "revision": first.get("revision"),
        "tokenizer_revision": first.get("tokenizer_revision"),
        "trust_remote_code": first.get("trust_remote_code", False),
        "chat_template": first.get("chat_template"),
        "chat_template_content_format": first.get("chat_template_content_format", "auto"),
        "chat_template_kwargs": first.get("chat_template_kwargs") or {},
        "mode": "mixed",
        "max_model_len": first.get("max_model_len"),
        "arrival": "poisson",
        "request_rate": request_rate,
        "seed": seed,
        "think_time_s": think_time,
        "sources": [
            {
                "label": label,
                "prepared": str(path.resolve()),
                "workload_sha256": file_hash(path / "workload.json"),
                "trace_sha256": file_hash(path / "trace.jsonl"),
                "conversations": len(workload["conversations"]),
            }
            for label, path, workload, _ in loaded
        ],
        "conversations": conversations,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "workload.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for entry in traces_out:
            stream.write(json.dumps(entry) + "\n")
    counts = {}
    for conversation in conversations:
        label = conversation["workload"]
        turns = counts.setdefault(label, {"conversations": 0, "turns": 0})
        turns["conversations"] += 1
        turns["turns"] += len(conversation["turns"])
    print(json.dumps(counts, indent=2))
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=2, action="append", required=True,
                        metavar=("LABEL", "PREPARED_DIR"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, default=100.0)
    parser.add_argument("--think-time", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sources = [(label, Path(path)) for label, path in args.source]
    mix(sources, args.output, args.request_rate, args.think_time, args.seed)

if __name__ == "__main__":
    main() 