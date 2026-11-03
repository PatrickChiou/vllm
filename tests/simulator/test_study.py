import asyncio
import json

import pytest
from aiohttp import web

from vllm.simulator.analysis import analyze, read_jsonl
from vllm.simulator.replay import replay
from vllm.simulator.study import server
from vllm.simulator.trace_builder import TraceBuilder
from vllm.simulator.workload import prepare

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.parametrize("policy", ["lru", "lfu", "fifo"])
def test_online_study_duplicate_replies_and_reuse(tmp_path, policy):
    dataset = tmp_path / "data.json"
    # Identical first prompts with different replies must be keyed by request
    # identity; neither asynchronous scheduling nor the policy may swap replies.
    records = [{"id": name, "conversations": [
        {"from": "human", "value": "List many different colors."},
        {"from": "gpt", "value": reply},
        {"from": "human", "value": "Now choose one."},
        {"from": "gpt", "value": name},
    ]} for name, reply in [
        ("Red", ("Red orange yellow green blue indigo violet black white "
                 "gray pink brown. ") * 3),
        ("Blue", ("Cyan magenta silver gold turquoise lavender beige "
                  "navy purple. ") * 3),
    ]]
    dataset.write_text(json.dumps(records))
    prepared = tmp_path / "prepared"
    workload = prepare(dataset, prepared, TraceBuilder(MODEL),
                       max_model_len=512, request_rate=100, seed=3)
    directory = tmp_path / policy
    with server(workload, prepared / "trace.jsonl", directory,
                block_size=16, cache_gb=0.02, policy=policy,
                prefill_ms=0, decode_ms=0, max_num_seqs=4,
                startup_timeout=60) as (url, cache):
        rows = asyncio.run(replay(workload, url, directory / "requests.jsonl"))
    assert len(rows) == 4
    assert all(r["success"] for r in rows), rows
    for conversation in workload["conversations"]:
        by_id = {r["request_id"]: r for r in rows}
        first, second = (by_id[t["request_id"]] for t in conversation["turns"])
        assert second["sent_s"] >= first["finished_s"]
        assert second["cached_tokens"] > 0
    assert cache["eviction_policy"] == policy
    (directory / "run.json").write_text(json.dumps({"name": policy}))
    summaries = analyze(tmp_path, plots=False)
    assert summaries[0]["failed_requests"] == 0
    assert summaries[0]["block_hits"] > 0
    events = read_jsonl(directory / "blocks.jsonl")
    assert len([e for e in events if e["event"] == "request"]) == 4
    assert (directory / "reuse_gaps.csv").exists()


def test_replay_failure_is_saved_and_blocks_followup_requests(tmp_path):
    async def run():
        received = []

        async def handler(request):
            received.append(await request.json())
            return web.json_response({"error": "injected failure"}, status=500)

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        workload = {"model": MODEL, "chat_template_kwargs": {},
                    "think_time_s": 0,
                    "conversations": [{"id": "a", "turns": [
                        {"request_id": str(i), "arrival_time_s": 0,
                         "messages": [], "output_tokens": 2}
                        for i in range(2)]}]}
        try:
            rows = await replay(workload, f"http://127.0.0.1:{port}",
                                tmp_path / "requests.jsonl")
        finally:
            await runner.cleanup()
        assert len(received) == 1
        assert len(rows) == 2
        assert all(not row["success"] for row in rows)
        assert "HTTP 500" in rows[0]["error"]
        assert "previous turn" in rows[1]["error"]
    asyncio.run(run())
