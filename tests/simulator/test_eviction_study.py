import json

import pytest

from vllm.simulator.eviction_study import (cache_blocks, convert_claude_code,
                                           convert_document_qa,
                                           convert_instructions, file_hash,
                                           interleave, rank_results,
                                           working_set)
from vllm.simulator.workload import normalize, read_records


def results():
    return [dict(workload=label, policy=policy, seed=seed,
                 failed_requests=0, successful_requests=10, prompt_tokens=100,
                 cached_tokens=hits, token_weighted_shared_fraction=hits / 100,
                 block_size=16, virtual_cache_bytes=1024,
                 workload_sha256=f"{label}-{seed}", trace_sha256=label)
            for label in ("chat", "code") for seed in (0, 1)
            for policy, hits in (("lru", 40), ("lfu", 60), ("fifo", 60))]


def test_instruction_conversion_preserves_context_and_counts_invalid(tmp_path):
    source, output = tmp_path / "data.json", tmp_path / "normalized.jsonl"
    source.write_text(json.dumps([
        {"instruction": "Fix code", "input": "  print(1)\n", "output": "ok"},
        {"instruction": "Explain", "output": "answer"},
        {"instruction": "invalid", "output": ["not text"]},
        {"instruction": "", "output": "empty instruction"},
    ]))
    counts = convert_instructions(source, output)
    assert counts == {"accepted": 2, "invalid_text": 2}
    records = list(read_records(output))
    first = normalize(records[0])
    assert first[0]["content"] == "Fix code\n\n  print(1)\n"
    assert first[1]["content"] == "ok"
    assert len(first) == 2  # No fabricated shared system prompt.
    assert normalize(records[1])[0]["content"] == "Explain"
    assert file_hash(source) != file_hash(output)


def test_conversion_rejects_dataset_with_no_valid_replies(tmp_path):
    source = tmp_path / "data.jsonl"
    source.write_text('{"instruction": "Fix", "output": ""}\n')
    with pytest.raises(ValueError, match="no usable"):
        convert_instructions(source, tmp_path / "out.jsonl")


def test_ranking_preserves_ties_and_reports_seed_variability():
    rows = results()
    rows[0].update(cached_tokens=20, token_weighted_shared_fraction=0.2)
    ranking = rank_results(rows)
    for label in ("chat", "code"):
        best = {r["policy"] for r in ranking
                if r["workload"] == label and r["observed_best"]}
        assert best == {"lfu", "fifo"}
    lru = next(r for r in ranking
               if r["workload"] == "chat" and r["policy"] == "lru")
    assert lru["mean_shared_fraction"] == pytest.approx(0.3)
    assert lru["std_shared_fraction"] == pytest.approx(0.1414213562)
    assert lru["cached_tokens"] == 60


@pytest.mark.parametrize("problem", [
    "failure", "empty", "geometry", "trace", "cohort", "seed", "policy",
    "duplicate", "nan"])
def test_ranking_refuses_incomplete_or_unfair_comparisons(problem):
    rows = results()
    if problem == "failure":
        rows[0]["failed_requests"] = 1
    elif problem == "empty":
        rows[0]["successful_requests"] = 0
    elif problem == "geometry":
        rows[0]["virtual_cache_bytes"] = 2048
    elif problem == "trace":
        rows[0]["trace_sha256"] = "changed"
    elif problem == "cohort":
        rows[0]["prompt_tokens"] = 101
    elif problem == "seed":
        rows[0]["seed"] = 7
    elif problem == "policy":
        rows = [r for r in rows if r["policy"] != "fifo"]
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["token_weighted_shared_fraction"] = float("nan")
    with pytest.raises(ValueError):
        rank_results(rows)


def test_ranking_rejects_no_runs():
    with pytest.raises(ValueError, match="No completed"):
        rank_results([])


def test_ranking_allows_cache_capacity_to_differ_between_workloads():
    rows = results()
    for row in rows:
        if row["workload"] == "code":
            row["virtual_cache_bytes"] = 4096
    assert len(rank_results(rows)) == 6


def event(role, content, sidechain=False, kind=None):
    return {"type": kind or role, "isSidechain": sidechain,
            "message": {"role": role, "content": content}}


def test_claude_code_conversion_merges_main_thread_and_renders_tools(tmp_path):
    trajectory = [
        event("user", "Build a game"),
        event("assistant", [{"type": "text", "text": "Plan first."}]),
        event("assistant", [{"type": "tool_use", "name": "Write",
                             "input": {"file_path": "a.html"}}]),
        event("user", [{"type": "tool_result", "content": [
            {"type": "text", "text": "created"}, {"type": "image"}]}]),
        event("assistant", [{"type": "text", "text": "sub"}], sidechain=True),
        event("assistant", [{"type": "thinking", "thinking": "hidden"},
                            {"type": "text", "text": "Done."}]),
        {"type": "summary", "summary": "ignored"},
    ]
    source, output = tmp_path / "cc.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps({"model_name": "m", "task_id": 3,
                                  "trajectory": json.dumps(trajectory)}) +
                      "\n" + json.dumps({"trajectory": []}) + "\n")
    assert convert_claude_code(source, output) == {
        "accepted": 1, "no_complete_turn": 1}
    record = next(read_records(output))
    messages = normalize(record)
    assert record["id"] == "m-3"
    assert [m["role"] for m in messages] == [
        "user", "assistant", "user", "assistant"]
    assert messages[1]["content"] == (
        'Plan first.\n<tool_call>\n{"name": "Write", "arguments": '
        '{"file_path": "a.html"}}\n</tool_call>')
    assert messages[2]["content"] == (
        "<tool_response>\ncreated\n</tool_response>")
    assert messages[3]["content"] == "Done."


def test_document_qa_shares_article_prefix_and_interleaves(tmp_path):
    rows = [dict(article=f"Article {a}", question=f"Q{a}{q}?",
                 options=["no", "yes"], answer=1)
            for a in range(3) for q in range(4)]
    rows.append(dict(article="Article 0", question="bad", options=["x"],
                     answer=3))
    source, output = tmp_path / "quality.jsonl", tmp_path / "out.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = convert_document_qa(source, output, group=2)
    assert counts == {"accepted": 12, "invalid": 1, "articles": 3}
    records = list(read_records(output))
    first = normalize(records[0])
    assert first[0]["content"].startswith("Article ")
    assert first[0]["content"].endswith(
        "\nA. no\nB. yes\nAnswer with the letter of the correct option.")
    assert first[1]["content"] == "B. yes"
    # The first group's 8 questions come first, in a shuffled order that
    # mixes both articles; the last article follows as its own group.
    articles = [r["id"].split("-")[0] for r in records]
    assert set(articles[:8]) == {"article0", "article1"}
    assert articles[:8] != sorted(articles[:8])
    assert set(articles[8:]) == {"article2"}
    assert convert_document_qa(source, tmp_path / "again.jsonl",
                               group=2) == counts
    assert list(read_records(tmp_path / "again.jsonl")) == records


def test_interleave_samples_each_file_round_robin(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_text("".join(json.dumps({"id": i, "conversations": []}) + "\n"
                             for i in range(3)))
    second.write_text(json.dumps({"id": 9, "messages": []}) + "\n")
    output = tmp_path / "mixed.jsonl"
    assert interleave([first, second], output) == {"a": 3, "b": 1}
    assert [r["id"] for r in read_records(output)] == [
        "a-0", "b-9", "a-1", "a-2"]


def test_working_set_counts_prefix_blocks_and_cache_sizing(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        # 2 full blocks of prompt+reply; the EOS is not cached.
        {"prompt_token_ids": [1] * 4 + [2] * 3, "output_token_ids": [3]},
        # Shares the first block; the same tokens later in a different
        # prefix are a distinct block.
        {"prompt_token_ids": [1] * 4 + [1] * 4 + [1] * 4,
         "output_token_ids": []},
    ]
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    footprint = working_set(trace, 4)
    assert footprint == {"unique_blocks": 4, "longest_request_tokens": 13}
    assert cache_blocks(1.0, footprint, 4) == {
        "num_gpu_blocks": 4, "cache_clamped": False}
    assert cache_blocks(0.5, footprint, 4) == {
        "num_gpu_blocks": 4, "cache_clamped": True}
    # The 1% watermark must still leave room for the longest request.
    assert cache_blocks(0.01, {"unique_blocks": 1000,
                               "longest_request_tokens": 400}, 4) == {
        "num_gpu_blocks": 101, "cache_clamped": True}
