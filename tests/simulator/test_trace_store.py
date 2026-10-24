import json

import pytest

from vllm.simulator import TraceEntry, TraceStore


def test_exact_lookup_is_order_sensitive():
    store = TraceStore([{
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [7, 8]
    }])
    assert store.lookup([1, 2, 3]) == TraceEntry(output_token_ids=(7, 8),
                                                 output_len=2)
    assert store.lookup([3, 2, 1]) is None
    assert store.lookup([1, 2]) is None


def test_output_len_only_entry():
    store = TraceStore([{"prompt_token_ids": [5], "output_len": 4}])
    assert store.lookup([5]) == TraceEntry(output_token_ids=None, output_len=4)


def test_duplicate_prompts_cycle_through_responses():
    store = TraceStore([
        {
            "prompt_token_ids": [1],
            "output_token_ids": [10]
        },
        {
            "prompt_token_ids": [2],
            "output_token_ids": [20]
        },
        {
            "prompt_token_ids": [1],
            "output_token_ids": [11, 12]
        },
    ])
    assert len(store) == 3
    assert store.num_unique_prompts == 2
    replayed = [store.lookup([1]).output_token_ids for _ in range(3)]
    assert replayed == [(10, ), (11, 12), (10, )]


@pytest.mark.parametrize("fmt", ["jsonl", "json"])
def test_from_file(tmp_path, fmt):
    entries = [
        {
            "prompt_token_ids": [1, 2],
            "output_token_ids": [3]
        },
        {
            "prompt_token_ids": [4],
            "output_len": 2
        },
    ]
    path = tmp_path / f"trace.{fmt}"
    if fmt == "jsonl":
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n\n")
    else:
        path.write_text(json.dumps(entries))

    store = TraceStore.from_file(str(path))
    assert len(store) == 2
    assert store.lookup([1, 2]).output_token_ids == (3, )
    assert store.lookup([4]).output_len == 2


@pytest.mark.parametrize("entry, message", [
    ({
        "output_token_ids": [1]
    }, "prompt_token_ids"),
    ({
        "prompt_token_ids": [1]
    }, "output_token_ids"),
    ({
        "prompt_token_ids": [1],
        "output_len": -1
    }, "non-negative"),
])
def test_invalid_entries(entry, message):
    with pytest.raises(ValueError, match=message):
        TraceStore([entry])
