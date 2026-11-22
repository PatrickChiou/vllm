import json

from vllm.simulator.mix_workload import mix


def prepared(path, turns, *, block=False):
    path.mkdir()
    conversations, traces = [], []
    for index, contents in enumerate(turns):
        conv_turns = []
        for turn_index, content in enumerate(contents):
            request_id = (f"bt-{index}-{turn_index}" if block
                          else f"m2-{index}-{turn_index}")
            turn = {"request_id": request_id, "prompt_tokens": 4,
                    "output_tokens": 2, "expected_text": "ok",
                    "timestamp": None}
            if block:
                turn["prompt_token_ids"] = content
                trace_id = f"cmpl-{request_id}-0"
                trace = {"request_id": trace_id,
                         "prompt_token_ids": content,
                         "output_token_ids": [3]}
            else:
                turn["messages"] = [{"role": "user", "content": content}]
                trace_id = f"chatcmpl-{request_id}"
                trace = {"request_id": trace_id,
                         "prompt_token_ids": [1, 2, 3, 4],
                         "output_token_ids": [5]}
            conv_turns.append(turn)
            traces.append(trace)
        conversations.append({"id": str(index), "turns": conv_turns})
    manifest = {"model": "m", "revision": None, "tokenizer": "m",
                "max_model_len": 128, "chat_template_kwargs": {},
                "conversations": conversations}
    (path / "workload.json").write_text(json.dumps(manifest))
    (path / "trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in traces))


def test_mix_keeps_unique_ids_and_turn_order(tmp_path):
    prepared(tmp_path / "chat", [["a", "b"], ["c"]])
    prepared(tmp_path / "code", [["d"]])
    manifest = mix([("chat", tmp_path / "chat"), ("code", tmp_path / "code")],
                   tmp_path / "mixed", 10, 1.0, 0)
    ids = [turn["request_id"] for conv in manifest["conversations"]
           for turn in conv["turns"]]
    assert len(ids) == len(set(ids))
    for conv in manifest["conversations"]:
        suffixes = [turn["request_id"].rsplit("-", 1)[-1]
                    for turn in conv["turns"]]
        assert suffixes == [str(i) for i in range(len(suffixes))]


def test_mix_repeats_the_same_seed(tmp_path):
    prepared(tmp_path / "chat", [["a"], ["b"], ["c"]])
    prepared(tmp_path / "code", [["d"], ["e"]])
    sources = [("chat", tmp_path / "chat"), ("code", tmp_path / "code")]
    first = mix(sources, tmp_path / "one", 10, 1.5, 3)
    second = mix(sources, tmp_path / "two", 10, 1.5, 3)
    assert [c["id"] for c in first["conversations"]] == [
        c["id"] for c in second["conversations"]]
    assert [t["arrival_time_s"] for c in first["conversations"]
            for t in c["turns"]] == [
                t["arrival_time_s"] for c in second["conversations"]
                for t in c["turns"]]


def test_mix_rewrites_block_trace_ids(tmp_path):
    prepared(tmp_path / "bailian", [[[9, 8, 7, 6]]], block=True)
    prepared(tmp_path / "chat", [["hi"]])
    output = tmp_path / "mixed"
    manifest = mix([("bailian-chat", tmp_path / "bailian"),
                    ("chat", tmp_path / "chat")], output, 10, 1.0, 0)
    turn = next(conv["turns"][0] for conv in manifest["conversations"]
                if conv["workload"] == "bailian-chat")
    assert turn["prompt_token_ids"] == [9, 8, 7, 6]
    traces = [json.loads(line) for line in
              (output / "trace.jsonl").read_text().splitlines()]
    assert any(row["request_id"] == f"cmpl-{turn['request_id']}-0"
               and row["prompt_token_ids"] == [9, 8, 7, 6]
               for row in traces)