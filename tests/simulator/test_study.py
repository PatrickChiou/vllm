import asyncio

from aiohttp import web

from vllm.simulator.replay import replay

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


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
