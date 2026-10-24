"""Replays traces through the real v0 LLMEngine with the sim executor."""
import json
from typing import Dict, List

import pytest
from transformers import AutoTokenizer

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.executor.sim_executor import SimExecutor
from vllm.inputs import TokensPrompt
from vllm.outputs import CompletionOutput

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16


@pytest.fixture(scope="module")
def eos_token_id() -> int:
    return AutoTokenizer.from_pretrained(MODEL).eos_token_id


def _tokens(start: int, n: int) -> List[int]:
    # Plain vocabulary ids, clear of Qwen's special tokens.
    return list(range(1000 + start, 1000 + start + n))


def _make_engine(tmp_path, entries: List[dict], **kwargs) -> LLMEngine:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    engine_args = EngineArgs(model=MODEL,
                             load_format="dummy",
                             distributed_executor_backend="sim",
                             sim_trace_path=str(trace_path),
                             max_model_len=512,
                             block_size=BLOCK_SIZE,
                             **kwargs)
    return LLMEngine.from_engine_args(engine_args)


def _run(engine: LLMEngine, prompts: Dict[str, List[int]],
         max_tokens: int) -> Dict[str, CompletionOutput]:
    for request_id, prompt in prompts.items():
        engine.add_request(request_id, TokensPrompt(prompt_token_ids=prompt),
                           SamplingParams(max_tokens=max_tokens))
    finished: Dict[str, CompletionOutput] = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = output.outputs[0]
    return finished


def test_uses_sim_executor_without_model(tmp_path):
    engine = _make_engine(tmp_path, [], num_gpu_blocks_override=123)
    assert isinstance(engine.model_executor, SimExecutor)
    assert engine.cache_config.num_gpu_blocks == 123
    assert engine.model_config.use_async_output_proc is False


def test_concurrent_requests_replay_exact_tokens(tmp_path, eos_token_id):
    prompts = {f"req-{i}": _tokens(100 * i, 20 + i) for i in range(4)}
    responses = {
        rid: _tokens(5000 + 100 * i, 5 + 3 * i)
        for i, rid in enumerate(prompts)
    }
    entries = [{
        "prompt_token_ids": prompts[rid],
        "output_token_ids": responses[rid]
    } for rid in prompts]
    engine = _make_engine(tmp_path, entries)

    finished = _run(engine, prompts, max_tokens=100)

    for rid, response in responses.items():
        assert list(finished[rid].token_ids) == response + [eos_token_id]
        assert finished[rid].finish_reason == "stop"
    stats = engine.model_executor.simulator.stats
    assert stats.num_trace_hits == 4
    assert stats.num_trace_misses == 0


def test_max_tokens_truncates_replay(tmp_path):
    prompt, response = _tokens(0, 10), _tokens(5000, 8)
    engine = _make_engine(tmp_path, [{
        "prompt_token_ids": prompt,
        "output_token_ids": response
    }])

    finished = _run(engine, {"r": prompt}, max_tokens=3)

    assert list(finished["r"].token_ids) == response[:3]
    assert finished["r"].finish_reason == "length"


def test_output_len_entry_and_trace_miss(tmp_path, eos_token_id):
    known, unknown = _tokens(0, 10), _tokens(300, 10)
    engine = _make_engine(tmp_path, [{
        "prompt_token_ids": known,
        "output_len": 4
    }],
                          sim_filler_token_id=42,
                          sim_default_output_len=2)

    finished = _run(engine, {"known": known, "unknown": unknown},
                    max_tokens=50)

    assert list(finished["known"].token_ids) == [42] * 4 + [eos_token_id]
    assert list(finished["unknown"].token_ids) == [42] * 2 + [eos_token_id]
    stats = engine.model_executor.simulator.stats
    assert (stats.num_trace_hits, stats.num_trace_misses) == (1, 1)


def test_chunked_prefill_samples_only_on_last_chunk(tmp_path, eos_token_id):
    prompt, response = _tokens(0, 50), _tokens(5000, 6)
    engine = _make_engine(tmp_path, [{
        "prompt_token_ids": prompt,
        "output_token_ids": response
    }],
                          enable_chunked_prefill=True,
                          max_num_batched_tokens=16,
                          max_num_seqs=4)

    finished = _run(engine, {"r": prompt}, max_tokens=100)

    assert list(finished["r"].token_ids) == response + [eos_token_id]
    # ceil(50 / 16) prefill steps, then one step per remaining token.
    assert engine.model_executor.simulator.stats.num_steps == 4 + len(
        response)


def test_replay_survives_recompute_preemption(tmp_path, eos_token_id):
    prompts = {f"req-{i}": _tokens(100 * i, 20) for i in range(4)}
    responses = {
        rid: _tokens(5000 + 100 * i, 40)
        for i, rid in enumerate(prompts)
    }
    entries = [{
        "prompt_token_ids": prompts[rid],
        "output_token_ids": responses[rid]
    } for rid in prompts]
    # 8 blocks hold all four prompts but not their responses.
    engine = _make_engine(tmp_path,
                          entries,
                          num_gpu_blocks_override=8,
                          max_num_seqs=4,
                          preemption_mode="recompute")

    finished = _run(engine, prompts, max_tokens=100)

    assert engine.scheduler[0].num_cumulative_preemption > 0
    for rid, response in responses.items():
        assert list(finished[rid].token_ids) == response + [eos_token_id]


def test_prefix_cache_hit_skips_cached_prefill_tokens(tmp_path):
    shared = _tokens(0, 4 * BLOCK_SIZE)
    first, second = shared + _tokens(900, 5), shared + _tokens(950, 10)
    engine = _make_engine(tmp_path, [
        {
            "prompt_token_ids": first,
            "output_token_ids": _tokens(5000, 3)
        },
        {
            "prompt_token_ids": second,
            "output_token_ids": _tokens(6000, 3)
        },
    ],
                          enable_prefix_caching=True)
    stats = engine.model_executor.simulator.stats

    _run(engine, {"first": first}, max_tokens=100)
    assert stats.num_uncached_prefill_tokens == len(first)

    _run(engine, {"second": second}, max_tokens=100)
    assert stats.num_prefill_tokens == len(first) + len(second)
    assert stats.num_uncached_prefill_tokens == len(first) + len(second) - len(
        shared)


def test_simulated_latency(tmp_path):
    prompt, response = _tokens(0, 20), _tokens(5000, 5)
    engine = _make_engine(tmp_path, [{
        "prompt_token_ids": prompt,
        "output_token_ids": response
    }],
                          sim_prefill_ms_per_token=0.05,
                          sim_decode_ms_base=0.5,
                          sim_decode_ms_per_seq=0.25)

    _run(engine, {"r": prompt}, max_tokens=100)

    # The prefill step emits the first token; the remaining response tokens
    # and the EOS each take one single-sequence decode step.
    expected_ms = 0.05 * len(prompt) + len(response) * (0.5 + 0.25)
    assert engine.model_executor.simulator.stats.simulated_time_s == (
        pytest.approx(expected_ms / 1000))
