from vllm.simulator.analysis import summarize


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
