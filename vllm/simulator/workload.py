"""ShareGPT normalization and model-specific trace preparation."""
import codecs
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from vllm.simulator.timing import assign_arrival_times, timestamp, validate_arrival_options


def array_records(chunks: Iterable[str]) -> Iterator[dict]:
    """Stream a JSON array without loading the full ShareGPT file into RAM."""
    decoder = json.JSONDecoder()
    chunks = iter(chunks)
    buffer = ""
    eof = False

    def refill() -> None:
        nonlocal buffer, eof
        try:
            buffer += next(chunks)
        except StopIteration:
            eof = True

    def whitespace() -> None:
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer and not eof:
            refill()
            buffer = buffer.lstrip()

    whitespace()
    if not buffer.startswith("["):
        raise ValueError("Expected a JSON array")
    buffer = buffer[1:]
    first = True
    while True:
        whitespace()
        if buffer.startswith("]"):
            buffer = buffer[1:]
            whitespace()
            if buffer:
                raise ValueError("Unexpected data after JSON array")
            return
        if not first:
            if not buffer.startswith(","):
                raise ValueError("Expected ',' between dataset records")
            buffer = buffer[1:]
            whitespace()
        while True:
            try:
                record, end = decoder.raw_decode(buffer)
                break
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(
                        "Incomplete or invalid dataset JSON") from None
                refill()
        if not isinstance(record, dict):
            raise ValueError("Dataset records must be JSON objects")
        buffer = buffer[end:]
        first = False
        yield record


def read_records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        first = " "
        while first and first.isspace():
            first = stream.read(1)
        stream.seek(0)
        if first == "[":
            yield from array_records(iter(lambda: stream.read(65536), ""))
        else:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Dataset records must be JSON objects")
                    yield record


def sample_dataset(url: str, output: Path, limit: int) -> int:
    """Download only enough of a JSON array to obtain the requested sample."""
    import requests

    if limit <= 0:
        raise ValueError("limit must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with requests.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        chunks = codecs.iterdecode(response.iter_content(65536), "utf-8")
        with output.open("w", encoding="utf-8") as stream:
            for record in array_records(chunks):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count >= limit:
                    break
    return count


def normalize(record: dict) -> List[dict]:
    """Accept ShareGPT human/gpt turns or OpenAI user/assistant messages.

    A final unanswered user turn is ignored. Malformed role sequences are
    rejected rather than silently joining unrelated parts of a conversation.
    """
    raw = record.get("conversations", record.get("messages"))
    if not isinstance(raw, list):
        raise ValueError("Missing conversations/messages list")
    roles = {"human": "user", "gpt": "assistant", "system": "system",
             "user": "user", "assistant": "assistant"}
    messages = []
    expected = "user"
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Messages must be JSON objects")
        role = roles.get(item.get("from", item.get("role")))
        content = item.get("value", item.get("content"))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be non-empty text")
        if role == "system" and not messages:
            pass
        elif role != expected:
            raise ValueError("Expected alternating user/assistant messages")
        else:
            expected = "assistant" if role == "user" else "user"
        message = {"role": role, "content": content}
        if role == "user":
            value = item.get("timestamp")
            if value is None and not any(m["role"] == "user" for m in messages):
                value = record.get("timestamp")
            message["timestamp"] = timestamp(value)
        messages.append(message)
    if messages and messages[-1]["role"] == "user":
        messages.pop()
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("No complete user/assistant turn")
    return messages


def prepare(dataset: Path, output: Path, builder: Any, *, mode: str = "multi",
            limit: int = 100, max_model_len: int = 4096,
            arrival: str = "auto", request_rate: float = 10.0,
            seed: int = 0, think_time: float = 0.0) -> dict:
    if mode not in ("single", "multi"):
        raise ValueError("Unknown mode")
    validate_arrival_options(arrival, request_rate, think_time)
    if limit <= 0 or max_model_len <= 0:
        raise ValueError("limit and max_model_len must be positive")
    output.mkdir(parents=True, exist_ok=True)
    conversations, traces = [], []
    skipped: Dict[str, int] = {}
    for index, record in enumerate(read_records(dataset)):
        try:
            messages = normalize(record)
            entries = builder.chat_entries([
                {k: v for k, v in m.items() if k != "timestamp"}
                for m in messages])
            turns = []
            for turn, entry in enumerate(entries):
                assistant_index = [i for i, m in enumerate(messages) if m["role"] == "assistant"][turn]
                # Filter whole conversations identically in both modes, so their first-turn cohorts remain comparable.
                if (len(entry["prompt_token_ids"]) + len(entry["output_token_ids"]) + 1 > max_model_len):
                    raise ValueError("context_length")
                request_id = f"m2-{index}-{turn}"
                entry = dict(entry, request_id=f"chatcmpl-{request_id}")
                turns.append({
                    "request_id": request_id,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in messages[:assistant_index]],
                    "prompt_tokens": len(entry["prompt_token_ids"]),
                    "output_tokens": len(entry["output_token_ids"]) + 1,
                    "expected_text": builder.tokenizer.decode(entry["output_token_ids"], skip_special_tokens=True),
                    "timestamp": messages[assistant_index - 1]["timestamp"],
                })
                if mode == "multi" or turn == 0:
                    traces.append(entry)
            conversations.append({"id": str(record.get("id", index)), "turns": (turns if mode == "multi" else turns[:1])})
            if len(conversations) >= limit:
                break
        except (ValueError, TypeError, KeyError) as exc:
            # Do not retain traces from partially processed conversations.
            prefix = f"chatcmpl-m2-{index}-"
            traces = [e for e in traces if not e["request_id"].startswith(prefix)]
            reason = str(exc)
            skipped[reason] = skipped.get(reason, 0) + 1
    if not conversations:
        raise ValueError(f"No usable conversations; exclusions: {skipped}")
    arrival = assign_arrival_times(conversations, arrival=arrival, request_rate=request_rate, seed=seed)
    manifest = {
        "version": 1, "model": builder.model_config.model,
        "tokenizer": builder.model_config.tokenizer,
        "revision": getattr(builder.model_config, "revision", None),
        "tokenizer_revision": getattr(builder.model_config, "tokenizer_revision", None),
        "trust_remote_code": getattr(builder.model_config, "trust_remote_code", False),
        "chat_template": getattr(builder, "chat_template", None),
        "chat_template_content_format": getattr(builder, "content_format", "auto"),
        "dataset_path": str(dataset.resolve()),
        "mode": mode, "max_model_len": max_model_len,
        "chat_template_kwargs": builder.chat_template_kwargs,
        "arrival": arrival, "request_rate": request_rate, "seed": seed,
        "think_time_s": think_time, "excluded_conversations": skipped,
        "conversations": conversations,
    }
    (output / "workload.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for entry in traces:
            stream.write(json.dumps(entry) + "\n")
    return manifest
