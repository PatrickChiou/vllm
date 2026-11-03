"""Arrival scheduling tests independent of dataset preparation and HTTP."""
import copy
import random

import pytest

from vllm.simulator.timing import (assign_arrival_times, request_target_time,
                                   timestamp, validate_arrival_options)


def conversations(*times):
    return [{"turns": [{"timestamp": value} for value in sequence]}
            for sequence in times]


@pytest.mark.parametrize("value,expected", [
    (None, None), (0, 0.0), (12.5, 12.5),
    ("1970-01-01T00:00:00Z", 0.0),
    ("1970-01-01T01:00:00+01:00", 0.0),
])
def test_timestamp_accepts_seconds_and_aware_iso_times(value, expected):
    assert timestamp(value) == expected


@pytest.mark.parametrize("value", [
    -1, float("nan"), float("inf"), "1970-01-01T00:00:00",
])
def test_timestamp_rejects_invalid_or_ambiguous_times(value):
    with pytest.raises(ValueError):
        timestamp(value)


def test_recorded_arrivals_share_an_origin_across_conversations():
    workload = conversations([104, 110], [100, 120])
    assert assign_arrival_times(workload) == "recorded"
    assert [[t["arrival_time_s"] for t in c["turns"]]
            for c in workload] == [[4, 10], [0, 20]]
    assert all("timestamp" not in t for c in workload for t in c["turns"])


def test_poisson_arrivals_are_seeded_and_preserve_first_turns():
    original = conversations([None, None, None], [None, None])
    workload = copy.deepcopy(original)
    repeated = copy.deepcopy(original)
    different = copy.deepcopy(original)
    first_turns = conversations([None], [None])
    global_rng_state = random.getstate()
    assert assign_arrival_times(workload, request_rate=2, seed=7) == "poisson"
    assign_arrival_times(repeated, request_rate=2, seed=7)
    assign_arrival_times(different, request_rate=2, seed=8)
    assign_arrival_times(first_turns, request_rate=2, seed=7)
    assert random.getstate() == global_rng_state
    assert workload == repeated
    assert workload != different
    assert [c["turns"][0] for c in workload] == [
        c["turns"][0] for c in first_turns]
    # Round order is a0, b0, a1, b1, a2, with positive interarrival times.
    order = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    times = [workload[c]["turns"][t]["arrival_time_s"] for c, t in order]
    assert times[0] > 0
    assert all(a < b for a, b in zip(times, times[1:]))


@pytest.mark.parametrize("arrival", ["auto", "recorded"])
def test_partial_timestamps_require_explicit_poisson(arrival):
    with pytest.raises(ValueError):
        assign_arrival_times(conversations([100, None]), arrival=arrival)
    workload = conversations([100, None])
    assert assign_arrival_times(workload, arrival="poisson") == "poisson"


def test_recorded_arrivals_reject_backwards_conversation_time():
    with pytest.raises(ValueError, match="timestamps must increase"):
        assign_arrival_times(conversations([100, 99]), arrival="recorded")


def test_empty_workload_cannot_be_scheduled():
    with pytest.raises(ValueError, match="empty workload"):
        assign_arrival_times([])


@pytest.mark.parametrize("options", [
    ("unknown", 1, 0), ("poisson", 0, 0),
    ("poisson", float("nan"), 0), ("recorded", float("inf"), 0),
    ("auto", 1, -1), ("auto", 1, float("nan")),
])
def test_invalid_arrival_options_are_rejected(options):
    with pytest.raises(ValueError):
        validate_arrival_options(*options)


@pytest.mark.parametrize("arrival,previous,scale,think,expected", [
    (2, None, 1, 3, 2),  # No think time before the first turn.
    (2, 4, 1, 3, 7),     # Previous reply and think time delay this turn.
    (10, 4, 1, 3, 10),   # A later desired arrival still controls dispatch.
    (2, 4, 2, 3, 10),    # Scale intervals, not the observed completion time.
    (2, 4, 0, 3, 4),     # Zero scale retains causal ordering.
])
def test_replay_targets_respect_arrivals_and_prior_completion(
        arrival, previous, scale, think, expected):
    assert request_target_time(arrival, previous, time_scale=scale,
                               think_time_s=think) == expected
