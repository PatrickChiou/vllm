"""Prepare Task 1 sources at one max_model_len for mix_workload.

Sources use the same ``--source`` and ``--set`` flags as eviction_study.
Every source must end up with the same max_model_len so mix_workload can
share one server. One prepared directory is written per label.
"""
import argparse
from pathlib import Path

from vllm.simulator.eviction_study import (
    FORMATS, SETTINGS, convert_claude_code, convert_document_qa,
    convert_instructions, interleave, source_settings)
from vllm.simulator.trace_builder import TraceBuilder
from vllm.simulator.workload import prepare, prepare_block_trace

_LABEL_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def prepare_sources(args) -> None:
    labels = [source[0] for source in args.source]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Use at least two sources with unique labels")
    if any(not label or any(char not in _LABEL_CHARS for char in label)
           for label in labels):
        raise ValueError("Labels may contain only letters, digits, _ and -")
    if any(fmt not in FORMATS for _, fmt, _ in args.source):
        raise ValueError(f"Source format must be one of {FORMATS}")
    args.seeds = [args.seed]
    args.time_scale = 1.0
    settings = source_settings(args)
    lengths = {settings[label]["max_model_len"] for label in labels}
    if len(lengths) != 1:
        raise ValueError(
            "Every source must use the same max_model_len so the mix can "
            "share one server")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    paths = [[Path(path).resolve(strict=True) for path in spec.split(",")]
             for _, _, spec in args.source]
    root.mkdir(parents=True)
    builder = TraceBuilder(args.model, revision=args.revision,
                           chat_template_kwargs={"date_string": "26 Jul 2024"})
    for source_index, (label, fmt, _) in enumerate(args.source):
        options = settings[label]
        files = paths[source_index]
        if len(files) > 1 and fmt != "chat":
            raise ValueError("Only chat sources may combine several files")
        dataset = files[0]
        if len(files) > 1:
            dataset = root / f"{label}-interleaved.jsonl"
            interleave(files, dataset)
        elif fmt == "instruction":
            dataset = root / f"{label}-normalized.jsonl"
            convert_instructions(files[0], dataset)
        elif fmt == "claude-code":
            dataset = root / f"{label}-normalized.jsonl"
            convert_claude_code(files[0], dataset)
        elif fmt == "document-qa":
            dataset = root / f"{label}-normalized.jsonl"
            convert_document_qa(files[0], dataset)
        prepared = root / label
        if fmt == "block-trace":
            workload = prepare_block_trace(
                dataset, prepared, builder, limit=options["limit"],
                max_model_len=options["max_model_len"],
                max_output_tokens=options["max_output_tokens"],
                trace_block_size=options["trace_block_size"],
                timestamp_unit=options["timestamp_unit"])
            size = sum(len(conversation["turns"])
                       for conversation in workload["conversations"])
        else:
            workload = prepare(dataset, prepared, builder, mode="multi",
                               limit=options["limit"],
                               max_model_len=options["max_model_len"],
                               arrival="poisson",
                               request_rate=args.request_rate,
                               seed=args.seed, think_time=0,
                               overflow=options["overflow"])
            size = len(workload["conversations"])
        if size < options["limit"] and fmt != "block-trace":
            raise ValueError(f"{label}: fewer than {options['limit']} "
                             "eligible conversations; lower the limit "
                             "or add data")
        print(f"{label}: {size}", flush=True)


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
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Shared context length; --set may not make "
                        "sources differ")
    parser.add_argument("--max-output-tokens", type=int, default=32,
                        help="Generated-token cap for block traces")
    parser.add_argument("--request-rate", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_model_len <= 0:
        parser.error("limit and max-model-len must be positive")
    prepare_sources(args)


if __name__ == "__main__":
    main()
