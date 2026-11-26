from vllm.simulator.hold_grid import (
    MIXES, bar_order, eta_seconds, estimate_minutes, progress_line, run_name,
    run_specs)


def test_grid_has_125_runs():
    assert len(MIXES) == 5
    assert len(run_specs()) == 25
    assert len(MIXES) * len(run_specs()) == 125


def test_coding_agent_limit_stays_at_or_below_12():
    assert all(limits["coding-agent"] <= 12 for _, limits in MIXES)


def test_progress_line_names_the_run_and_the_position():
    line = progress_line(12, 116, "chat-heavy", 1, 4, "tlru-h8-b0.5", 3600, "run")
    assert line == ("[12/116] run chat-heavy (1/4) tlru-h8-b0.5 "
                    "elapsed 1h00m eta 8h40m")


def test_eta_before_the_first_finish_uses_the_estimate():
    assert eta_seconds(0, 0, 116) == estimate_minutes() * 60
    assert eta_seconds(0, 60, 116) == estimate_minutes() * 60 - 60


def test_bar_order_puts_lru_lfu_fifo_first():
    names = [run_name(spec) for spec in reversed(run_specs())]
    ordered = bar_order(names)
    assert ordered[:3] == ["lru", "lfu", "fifo"]
    assert ordered[3] == "tfifo"
    assert ordered[4] == "rfifo-w20-t0.02"
    assert ordered[-1] == "hlru-h30-b1"
