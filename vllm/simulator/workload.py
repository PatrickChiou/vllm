"""ShareGPT normalization and model-specific trace preparation."""
import codecs
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

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
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
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


def model_fields(builder: Any) -> dict:
    """Manifest fields the replay server needs to load the same tokenizer."""
    config = builder.model_config
    return {
        "version": 1, "model": config.model, "tokenizer": config.tokenizer,
        "revision": getattr(config, "revision", None),
        "tokenizer_revision": getattr(config, "tokenizer_revision", None),
        "trust_remote_code": getattr(config, "trust_remote_code", False),
        "chat_template": getattr(builder, "chat_template", None),
        "chat_template_content_format": getattr(builder, "content_format",
                                                "auto"),
    }


def prepare(dataset: Path, output: Path, builder: Any, *, mode: str = "multi",
            limit: int = 100, max_model_len: int = 4096,
            arrival: str = "auto", request_rate: float = 10.0,
            seed: int = 0, think_time: float = 0.0,
            overflow: str = "reject") -> dict:
    """Build a replay manifest and exact token trace.

    ``overflow="reject"`` drops a conversation with any turn exceeding
    ``max_model_len``. ``"truncate"`` keeps the turns before the first such
    turn, which suits long agent trajectories.
    """
    if mode not in ("single", "multi"):
        raise ValueError("Unknown mode")
    if overflow not in ("reject", "truncate"):
        raise ValueError("Unknown overflow handling")
    validate_arrival_options(arrival, request_rate, think_time)
    if limit <= 0 or max_model_len <= 0:
        raise ValueError("limit and max_model_len must be positive")
    output.mkdir(parents=True, exist_ok=True)
    conversations, traces = [], []
    skipped: Dict[str, int] = {}
    truncated = 0
    for index, record in enumerate(read_records(dataset)):
        try:
            messages = normalize(record)
            plain = [{k: v for k, v in m.items() if k != "timestamp"}
                     for m in messages]
            if overflow == "truncate":
                entries = builder.chat_entries(plain,
                                               max_model_len=max_model_len)
                if not entries:
                    raise ValueError("context_length")
            else:
                entries = builder.chat_entries(plain)
            was_truncated = len(entries) < sum(
                m["role"] == "assistant" for m in messages)
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
            truncated += was_truncated
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
        **model_fields(builder),
        "dataset_path": str(dataset.resolve()),
        "mode": mode, "max_model_len": max_model_len,
        "chat_template_kwargs": builder.chat_template_kwargs,
        "arrival": arrival, "request_rate": request_rate, "seed": seed,
        "think_time_s": think_time, "excluded_conversations": skipped,
        "overflow": overflow, "truncated_conversations": truncated,
        "conversations": conversations,
    }
    write_prepared(output, manifest, traces)
    return manifest


def write_prepared(output: Path, manifest: dict, traces: List[dict],
                   indent: Optional[int] = 2) -> None:
    (output / "workload.json").write_text(
        json.dumps(manifest, indent=indent), encoding="utf-8")
    with (output / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for entry in traces:
            stream.write(json.dumps(entry) + "\n")


class BlockTokens:
    """Deterministic stand-in tokens for anonymized KV block hashes.

    Equal hash ids map to equal tokens, so prefix reuse in the trace becomes
    prefix reuse in vLLM; distinct ids collide with negligible probability.
    Special token ids are never produced.
    """

    def __init__(self, tokenizer: Any, block_size: int = 16):
        self.block_size = block_size
        self.vocab_size = tokenizer.vocab_size
        self.special = set(tokenizer.all_special_ids)
        self._cache: Dict[int, List[int]] = {}

    def block(self, hash_id: int) -> List[int]:
        tokens = self._cache.get(hash_id)
        if tokens is None:
            rng, tokens = random.Random(hash_id), []
            while len(tokens) < self.block_size:
                token = rng.randrange(self.vocab_size)
                if token not in self.special:
                    tokens.append(token)
            self._cache[hash_id] = tokens
        return tokens

    def prompt(self, hash_ids: List[int], length: int) -> List[int]:
        if not (self.block_size * (len(hash_ids) - 1) < length
                <= self.block_size * len(hash_ids)):
            raise ValueError("input_length does not match hash_ids")
        tokens = [t for h in hash_ids for t in self.block(h)]
        return tokens[:length]


def prepare_block_trace(dataset: Path, output: Path, builder: Any, *,
                        limit: int = 1000, max_model_len: int = 32768,
                        max_output_tokens: int = 128,
                        trace_block_size: int = 16,
                        timestamp_unit: str = "s") -> dict:
    """Prepare a Bailian-style block-hash trace for /v1/completions replay.

    Uses the first ``limit`` requests in trace order, i.e. a time window, with
    their recorded arrival times. Requests linked by ``parent_chat_id`` form
    one conversation replayed in causal order; traces without session fields
    (e.g. Mooncake, whose timestamps are milliseconds) treat every request as
    its own conversation. Prompts are the already
    templated token blocks, so no chat template is applied. Generated tokens
    are a repeated filler token capped at ``max_output_tokens``; they occupy
    KV cache but are not reused because later prompts carry recorded hashes.
    """
    if limit <= 0 or max_model_len <= 0 or max_output_tokens < 0:
        raise ValueError("limit and max_model_len must be positive, "
                         "max_output_tokens non-negative")
    units = {"s": 1.0, "ms": 1e-3}
    if timestamp_unit not in units:
        raise ValueError("timestamp_unit must be s or ms")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = builder.tokenizer
    blocks = BlockTokens(tokenizer, trace_block_size)
    filler = tokenizer(" a", add_special_tokens=False).input_ids[-1]
    sessions: Dict[int, dict] = {}
    root_of: Dict[int, int] = {}
    traces, skipped, previous, count = [], {}, 0.0, 0
    for record in read_records(dataset):
        if count >= limit:
            break
        chat_id = int(record.get("chat_id", count))
        parent = int(record.get("parent_chat_id", -1))
        count += 1
        arrival = timestamp(record["timestamp"])
        if arrival is None or arrival * units[timestamp_unit] < previous:
            raise ValueError("Trace must be sorted by timestamp")
        arrival *= units[timestamp_unit]
        previous = arrival
        # Keep sessions linked through requests excluded below.
        root_of[chat_id] = root_of.get(parent, chat_id)
        prompt = blocks.prompt([int(h) for h in record["hash_ids"]],
                               int(record["input_length"]))
        outputs = min(int(record["output_length"]), max_output_tokens)
        if len(prompt) + outputs + 1 > max_model_len:
            skipped["context_length"] = skipped.get("context_length", 0) + 1
            continue
        request_id = f"bt-{chat_id}"
        output_ids = [filler] * outputs
        traces.append({"prompt_token_ids": prompt,
                       "output_token_ids": output_ids,
                       "request_id": f"cmpl-{request_id}-0"})
        session = sessions.setdefault(root_of[chat_id], {
            "id": str(root_of[chat_id]), "turns": []})
        session["turns"].append({
            "request_id": request_id, "prompt_token_ids": prompt,
            "prompt_tokens": len(prompt), "output_tokens": outputs + 1,
            "expected_text": tokenizer.decode(output_ids,
                                              skip_special_tokens=True),
            "timestamp": arrival,
            "type": record.get("type"),
        })
    conversations = list(sessions.values())
    if not conversations:
        raise ValueError(f"No usable requests; exclusions: {skipped}")
    arrival = assign_arrival_times(conversations, arrival="recorded")
    manifest = {
        **model_fields(builder),
        "dataset_path": str(dataset.resolve()), "format": "block-trace",
        "mode": "multi", "max_model_len": max_model_len,
        "chat_template_kwargs": builder.chat_template_kwargs,
        "arrival": arrival, "request_rate": None, "seed": None,
        "think_time_s": 0.0, "excluded_requests": skipped,
        "max_output_tokens": max_output_tokens,
        "trace_block_size": trace_block_size,
        "timestamp_unit": timestamp_unit,
        "conversations": conversations,
    }
    # Token-id prompts make an indented manifest needlessly large.
    write_prepared(output, manifest, traces, indent=None)
    return manifest
