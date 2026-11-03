"""Timestamp parsing and arrival scheduling shared by preparation and replay.

These helpers depend only on the standard library. Dataset formatting, HTTP
requests, and actual waiting remain in their respective callers.
"""
import math
import random
from datetime import datetime
from typing import Any, List, Optional


def timestamp(value: Any) -> Optional[float]:
    """Convert numeric seconds or timezone-aware ISO text to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("ISO timestamps must include a timezone")
        result = parsed.timestamp()
    else:
        result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Timestamps must be finite, non-negative seconds")
    return result


def validate_arrival_options(arrival: str, request_rate: float, think_time: float = 0.0) -> None:
    """Validate scheduling options before preparing a dataset or schedule."""
    if arrival not in ("auto", "poisson", "recorded"):
        raise ValueError("Unknown arrival process")
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be finite and positive")
    if not math.isfinite(think_time) or think_time < 0:
        raise ValueError("think_time must be finite and non-negative")


def assign_arrival_times(conversations: List[dict], *, arrival: str = "auto", request_rate: float = 10.0, seed: int = 0) -> str:
    """Replace each turn's normalized timestamp with an arrival_time_s.

    Returns the resolved arrival process. Recorded times share a common origin;
    Poisson intervals form one stream assigned in turn-round order, preserving
    the first-turn schedule when follow-up turns are added. Uses a local RNG.
    """
    validate_arrival_options(arrival, request_rate)
    times = [t["timestamp"] for c in conversations for t in c["turns"]]
    if not times:
        raise ValueError("Cannot schedule an empty workload")
    if arrival == "auto":
        if (any(t is not None for t in times) and not all(t is not None for t in times)):
            raise ValueError("Partial timestamps: select --arrival poisson")
        arrival = "recorded" if all(t is not None for t in times) else "poisson"
    if arrival == "recorded" and any(t is None for t in times):
        raise ValueError("Recorded arrivals need timestamps on every user turn")
    rng, offset = random.Random(seed), 0.0
    origin = min(times) if arrival == "recorded" else None
    if arrival == "poisson":
        for i in range(max(len(c["turns"]) for c in conversations)):
            for conversation in conversations:
                if i < len(conversation["turns"]):
                    offset += rng.expovariate(request_rate)
                    conversation["turns"][i]["arrival_time_s"] = offset
    for conversation in conversations:
        previous = -1.0
        for turn in conversation["turns"]:
            target = (turn["timestamp"] - origin if origin is not None else turn["arrival_time_s"])
            if target < previous:
                raise ValueError("Conversation timestamps must increase")
            previous = target
            turn["arrival_time_s"] = target
            turn.pop("timestamp")
    return arrival


def request_target_time(arrival_time_s: float, previous_end_s: Optional[float] = None, *, time_scale: float = 1.0, think_time_s: float = 0.0) -> float:
    """Return a send target in seconds since replay started.

    The caller validates options and waits using its monotonic replay clock.
    Completion times are already in that clock's seconds and are not scaled;
    only desired arrivals and think-time intervals are scaled. None denotes a
    first turn, which has no preceding reply or think-time delay.
    """
    planned = arrival_time_s * time_scale
    earliest = (previous_end_s + think_time_s * time_scale if previous_end_s is not None else 0.0)
    return max(planned, earliest)
