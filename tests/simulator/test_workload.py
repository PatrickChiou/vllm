import json
from types import SimpleNamespace

import pytest

from vllm.simulator.workload import (array_records, normalize, prepare,
                                     read_records)


class Builder:
    model_config = SimpleNamespace(model="test", tokenizer="test")
    chat_template_kwargs = {"date_string": "26 Jul 2024"}
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))

    def chat_entries(self, messages):
        return [{"prompt_token_ids": [1] * (i + 1),
                 "output_token_ids": [2] * len(message["content"])}
                for i, message in enumerate(messages)
                if message["role"] == "assistant"]


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
