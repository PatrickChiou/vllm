"""Arrival-driven online replay with causal ordering of conversation turns."""
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from vllm.simulator.timing import request_target_time


def validate_response(payload: dict, turn: dict) -> dict:
    choice, usage = payload["choices"][0], payload["usage"]
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0)
    prompt = usage["prompt_tokens"]
    if (choice["message"]["content"] != turn["expected_text"]
            or usage["completion_tokens"] != turn["output_tokens"]
            or prompt != turn["prompt_tokens"]
            or choice["finish_reason"] != "stop"):
        raise ValueError("Server response differs from recorded trace")
    if not 0 <= cached <= prompt:
        raise ValueError("Invalid cached-token count")
    return dict(success=True, prompt_tokens=prompt, cached_tokens=cached,
                shared_fraction=cached / prompt,
                completion_tokens=usage["completion_tokens"])


async def replay(workload: dict, base_url: str, output: Path, *,
                 concurrency: int = 256, time_scale: float = 1.0,
                 timeout: float = 300.0,
                 api_key: Optional[str] = None) -> List[dict]:
    if concurrency <= 0 or timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("concurrency and timeout must be positive")
    if time_scale < 0 or not math.isfinite(time_scale):
        raise ValueError("time_scale must be finite and non-negative")
    semaphore = asyncio.Semaphore(concurrency)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    start = time.monotonic()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = base_url.rstrip("/") + "/v1/chat/completions"
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    template_kwargs = workload["chat_template_kwargs"]

    async with aiohttp.ClientSession(
            timeout=client_timeout, headers=headers) as session:
        with output.open("w", encoding="utf-8") as stream:

            async def conversation_replay(conversation: dict) -> None:
                previous_end: Optional[float] = None
                failed = False
                for index, turn in enumerate(conversation["turns"]):
                    planned = turn["arrival_time_s"] * time_scale
                    target = request_target_time(
                        turn["arrival_time_s"], previous_end,
                        time_scale=time_scale,
                        think_time_s=workload["think_time_s"])
                    delay = target - (time.monotonic() - start)
                    if not failed:
                        await asyncio.sleep(max(0.0, delay))
                    row: Dict[str, Any] = {
                        "request_id": turn["request_id"],
                        "conversation_id": conversation["id"], "turn": index,
                        "planned_s": planned, "success": False,
                    }
                    if failed:
                        row["error"] = "Skipped after previous turn failed"
                    else:
                        async with semaphore:
                            sent = time.monotonic() - start
                            row.update(sent_s=sent,
                                       dispatch_lag_s=sent - planned)
                            try:
                                body = {
                                    "model": workload["model"],
                                    "messages": turn["messages"],
                                    "request_id": turn["request_id"],
                                    "chat_template_kwargs": template_kwargs,
                                    "max_tokens": turn["output_tokens"],
                                    "temperature": 0.0,
                                }
                                async with session.post(
                                        url, json=body) as response:
                                    payload = await response.json()
                                    if response.status != 200:
                                        status = response.status
                                        raise ValueError(
                                            f"HTTP {status}: {payload}")
                                row.update(validate_response(payload, turn))
                            except (aiohttp.ClientError, asyncio.TimeoutError,
                                    ValueError, KeyError, IndexError,
                                    TypeError) as exc:
                                row["error"] = str(exc)
                            row["finished_s"] = time.monotonic() - start
                            row["latency_s"] = row["finished_s"] - sent
                            previous_end = row["finished_s"]
                            failed = not row["success"]
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()

            await asyncio.gather(*(conversation_replay(c)
                                   for c in workload["conversations"]))
    return rows
