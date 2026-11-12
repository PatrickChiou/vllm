import json

import pytest

from vllm.simulator.eviction_study import (convert_instructions, file_hash,
                                           rank_results)
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
