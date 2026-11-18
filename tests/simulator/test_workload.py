import json
from types import SimpleNamespace

import pytest

from vllm.simulator.workload import (BlockTokens, array_records, normalize,
                                     prepare, prepare_block_trace,
                                     read_records)


class Builder:
    model_config = SimpleNamespace(model="test", tokenizer="test")
    chat_template_kwargs = {"date_string": "26 Jul 2024"}
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))

    def chat_entries(self, messages, max_model_len=None):
        entries = []
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            entry = {"prompt_token_ids": [1] * (i + 1),
                     "output_token_ids": [2] * len(message["content"])}
            if (max_model_len is not None and len(entry["prompt_token_ids"])
                    + len(entry["output_token_ids"]) + 1 > max_model_len):
                break
            entries.append(entry)
        return entries


def conversation(identity="a", timestamps=False):
    result = {"id": identity, "conversations": [
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello"},
        {"from": "human", "value": "Again"},
        {"from": "gpt", "value": "Reply"},
    ]}
    if timestamps:
        result["conversations"][0]["timestamp"] = 100
        result["conversations"][2]["timestamp"] = 104
    return result


def test_stream_json_array_handles_chunk_boundaries_and_trailing_data(tmp_path):
    records = [conversation(), conversation("b")]
    text = json.dumps(records, ensure_ascii=False)
    chunks = (text[i:i + 3] for i in range(0, len(text), 3))
    assert list(array_records(chunks)) == records
    path = tmp_path / "dataset.json"
    path.write_text(text)
    assert list(read_records(path)) == records
    with pytest.raises(ValueError):
        list(array_records([text + "junk"]))
    with pytest.raises(ValueError):
        list(array_records([text[:-1]]))


def test_single_and_multi_share_first_turn_cohort_and_seeded_arrivals(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(
        json.dumps(conversation(x)) for x in ["a", "b"]))
    single = prepare(source, tmp_path / "single", Builder(), mode="single")
    multi = prepare(source, tmp_path / "multi", Builder())
    assert [len(c["turns"]) for c in single["conversations"]] == [1, 1]
    assert [len(c["turns"]) for c in multi["conversations"]] == [2, 2]
    for a, b in zip(single["conversations"], multi["conversations"]):
        assert a["turns"][0] == b["turns"][0]
        assert b["turns"][1]["arrival_time_s"] > b["turns"][0]["arrival_time_s"]
    trace = list(read_records(tmp_path / "multi" / "trace.jsonl"))
    assert len({e["request_id"] for e in trace}) == 4


def test_preparation_uses_recorded_arrivals(tmp_path):
    source = tmp_path / "data.json"
    record = conversation(timestamps=True)
    source.write_text(json.dumps([record]))
    workload = prepare(source, tmp_path / "recorded", Builder())
    assert workload["arrival"] == "recorded"
    times = [t["arrival_time_s"]
             for t in workload["conversations"][0]["turns"]]
    assert times == [0, 4]


def test_context_filter_excludes_whole_conversation(tmp_path):
    source = tmp_path / "data.json"
    too_long = conversation("long")
    too_long["conversations"][3]["value"] = "x" * 100
    source.write_text(json.dumps([too_long, conversation()]))
    for mode in ["single", "multi"]:
        result = prepare(source, tmp_path / mode, Builder(),
                         mode=mode, max_model_len=20)
        assert result["excluded_conversations"] == {"context_length": 1}
        assert len(result["conversations"]) == 1
        assert all(e["request_id"].startswith("chatcmpl-m2-1-")
                   for e in read_records(tmp_path / mode / "trace.jsonl"))


def test_malformed_roles_rejected_and_unanswered_user_ignored():
    record = conversation()
    record["conversations"].append({"from": "human", "value": "Unanswered"})
    assert len(normalize(record)) == 4
    record["conversations"][2]["from"] = "gpt"
    with pytest.raises(ValueError, match="alternating"):
        normalize(record)


def test_truncation_keeps_turns_before_the_first_overflow(tmp_path):
    source = tmp_path / "data.json"
    too_long = conversation("long")
    too_long["conversations"][3]["value"] = "x" * 100
    source.write_text(json.dumps([too_long]))
    result = prepare(source, tmp_path / "out", Builder(), max_model_len=20,
                     overflow="truncate")
    assert result["excluded_conversations"] == {}
    assert result["truncated_conversations"] == 1
    assert [len(c["turns"]) for c in result["conversations"]] == [1]
    assert len(list(read_records(tmp_path / "out" / "trace.jsonl"))) == 1


class Tokenizer:
    vocab_size = 50
    all_special_ids = [0, 1]

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[7])

    def decode(self, ids, **kwargs):
        return "a" * len(ids)


class BlockBuilder(Builder):
    tokenizer = Tokenizer()


def test_block_tokens_are_deterministic_and_avoid_special_ids():
    blocks = BlockTokens(Tokenizer(), 4)
    assert blocks.block(3) == BlockTokens(Tokenizer(), 4).block(3)
    assert blocks.block(3) != blocks.block(4)
    assert all(1 < t < 50 for h in range(20) for t in blocks.block(h))
    prompt = blocks.prompt([3, 4], 6)
    assert prompt == blocks.block(3) + blocks.block(4)[:2]
    with pytest.raises(ValueError, match="input_length"):
        blocks.prompt([3, 4], 9)


def test_block_trace_links_sessions_and_caps_outputs(tmp_path):
    source = tmp_path / "trace.jsonl"
    records = [
        dict(chat_id=5, parent_chat_id=-1, timestamp=10.0, input_length=16,
             output_length=300, type="text", turn=1, hash_ids=[1, 2]),
        dict(chat_id=6, parent_chat_id=-1, timestamp=10.5, input_length=60,
             output_length=2, type="file", turn=1,
             hash_ids=[1, 3, 4, 5, 6, 7, 8, 9]),
        dict(chat_id=7, parent_chat_id=5, timestamp=12.0, input_length=30,
             output_length=2, type="text", turn=2, hash_ids=[1, 2, 10, 11]),
        dict(chat_id=8, parent_chat_id=-1, timestamp=13.0, input_length=8,
             output_length=1, type="text", turn=1, hash_ids=[12]),
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    result = prepare_block_trace(source, tmp_path / "out", BlockBuilder(),
                                 limit=3, max_model_len=40,
                                 max_output_tokens=4, trace_block_size=8)
    assert result["excluded_requests"] == {"context_length": 1}
    [session] = result["conversations"]
    assert [t["request_id"] for t in session["turns"]] == ["bt-5", "bt-7"]
    assert [t["arrival_time_s"] for t in session["turns"]] == [0.0, 2.0]
    first, second = session["turns"]
    assert first["output_tokens"] == 5 and first["expected_text"] == "aaaa"
    assert second["prompt_token_ids"][:16] == first["prompt_token_ids"][:16]
    trace = list(read_records(tmp_path / "out" / "trace.jsonl"))
    assert [e["request_id"] for e in trace] == ["cmpl-bt-5-0", "cmpl-bt-7-0"]
    assert trace[0]["output_token_ids"] == [7] * 4


def test_block_trace_without_sessions_uses_millisecond_timestamps(tmp_path):
    source = tmp_path / "mooncake.jsonl"
    source.write_text("".join(json.dumps(dict(
        timestamp=t, input_length=10, output_length=1, hash_ids=[0, i]))
        + "\n" for i, t in [(1, 1000), (2, 1000), (3, 3500)]))
    result = prepare_block_trace(source, tmp_path / "out", BlockBuilder(),
                                 trace_block_size=8, timestamp_unit="ms")
    assert [len(c["turns"]) for c in result["conversations"]] == [1, 1, 1]
    assert [c["turns"][0]["arrival_time_s"]
            for c in result["conversations"]] == [0.0, 0.0, 2.5]
    prompts = [c["turns"][0]["prompt_token_ids"]
               for c in result["conversations"]]
    assert prompts[0][:8] == prompts[1][:8] and prompts[0] != prompts[1]


def test_block_trace_rejects_unsorted_timestamps(tmp_path):
    source = tmp_path / "trace.jsonl"
    source.write_text("".join(json.dumps(dict(
        chat_id=i, parent_chat_id=-1, timestamp=t, input_length=8,
        output_length=1, hash_ids=[i])) + "\n" for i, t in [(0, 5), (1, 4)]))
    with pytest.raises(ValueError, match="sorted"):
        prepare_block_trace(source, tmp_path / "out", BlockBuilder(),
                            trace_block_size=8)
