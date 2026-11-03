from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.simulator.analysis import read_jsonl, summarize
from vllm.simulator.metrics import PrefixMetrics
from vllm.config import SimulatorConfig
from vllm.simulator import Simulator, TraceStore


def metadata(request_id, tokens, cached=0, **kwargs):
    return SequenceGroupMetadata(request_id=request_id, is_prompt=True,
                                 seq_data={0: SequenceData.from_seqs(tokens)},
                                 sampling_params=None, block_tables={0: []},
                                 computed_block_nums=list(range(cached)),
                                 **kwargs)


def test_counts_distinct_request_reuse_with_stable_block_identity(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = PrefixMetrics(str(path), block_size=4)
    first = metadata("a", list(range(8)))
    metrics.observe(first, now=1)
    second = metadata("b", list(range(12)), cached=2)
    metrics.observe(second, now=3)
    # Chunk/decode/recompute visits must not count extra cross-request hits.
    metrics.observe(second, now=4)
    metrics.finish(["a", "b"])
    # Identical local block tokens with a different prefix are different blocks.
    metrics.observe(metadata("c", [99] * 4 + list(range(4, 8))), now=5)
    metrics.close()
    events = read_jsonl(path)
    blocks = [e for e in events if e["event"] == "block"]
    hits = [e for e in blocks if e["hit"]]
    assert len(blocks) == 7
    assert len(hits) == 2
    assert [e["reuse_gap_s"] for e in hits] == [2, 2]
    result = summarize([], events)
    assert result["summary"]["unique_blocks"] == 5
    assert result["summary"]["zero_hit_blocks"] == 3


def test_decode_records_only_computed_full_blocks(tmp_path):
    path = tmp_path / "metrics.jsonl"
    metrics = PrefixMetrics(str(path), 4)
    first = metadata("a", [1, 2, 3, 4, 5])
    metrics.observe(first, now=1)
    data = SequenceData.from_seqs([1, 2, 3, 4, 5], [6, 7])
    data.update_num_computed_tokens(6)
    decode = SequenceGroupMetadata(request_id="a", is_prompt=False,
                                    seq_data={0: data}, sampling_params=None,
                                    block_tables={0: []})
    metrics.observe(decode, now=2)
    # The next sampled token would fill a block but has not been computed.
    metrics.close()
    blocks = [e for e in read_jsonl(path) if e["event"] == "block"]
    assert len(blocks) == 1


def test_summary_distinguishes_weighted_fraction_and_failed_requests():
    result = summarize([
        {"success": True, "prompt_tokens": 100, "cached_tokens": 50,
         "shared_fraction": 0.5, "finished_s": 2},
        {"success": True, "prompt_tokens": 10, "cached_tokens": 0,
         "shared_fraction": 0, "finished_s": 4},
        {"success": False, "finished_s": 4},
    ], [])
    assert result["summary"]["mean_request_shared_fraction"] == 0.25
    assert result["summary"]["token_weighted_shared_fraction"] == 50 / 110
    assert result["summary"]["failed_requests"] == 1


def test_recompute_does_not_overwrite_initial_prompt_reuse():
    simulator = Simulator(SimulatorConfig(), TraceStore(), 4, 0)
    group = metadata("a", list(range(8)), cached=1)
    simulator._prefill_step(group)
    assert group.seq_data[0].get_num_cached_tokens() == 4
    data = SequenceData.from_seqs(list(range(8)), [9, 10, 11, 12])
    data.update_num_cached_tokens(4)
    group.seq_data[0] = data
    group.computed_block_nums = [0, 1, 2]
    group.token_chunk_size = 12
    simulator._prefill_step(group)
    assert data.get_num_cached_tokens() == 4
