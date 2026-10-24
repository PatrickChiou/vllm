"""Online (OpenAI-compatible server) inference with the sim executor."""
import json

import pytest
from transformers import AutoTokenizer

from ..utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHAT = [{"role": "user", "content": "What is the capital of France?"}]
CHAT_REPLY = "The capital of France is Paris."
COMPLETION_PROMPT = "Once upon a time"
COMPLETION_REPLY = " there was a simulator."


@pytest.fixture(scope="module", params=[False, True])
def server(request, tmp_path_factory):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    entries = [
        {
            "prompt_token_ids":
            tokenizer.apply_chat_template(CHAT,
                                          add_generation_prompt=True,
                                          tokenize=True),
            "output_token_ids":
            tokenizer(CHAT_REPLY, add_special_tokens=False).input_ids,
        },
        {
            "prompt_token_ids":
            tokenizer(COMPLETION_PROMPT).input_ids,
            "output_token_ids":
            tokenizer(COMPLETION_REPLY, add_special_tokens=False).input_ids,
        },
    ]
    trace_path = tmp_path_factory.mktemp("trace") / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in entries))

    args = [
        "--load-format", "dummy", "--distributed-executor-backend", "sim",
        "--sim-trace-path",
        str(trace_path), "--max-model-len", "1024", "--enable-prefix-caching"
    ]
    if request.param:
        # Exercises AsyncLLMEngine + SimExecutorAsync instead of MQLLMEngine.
        args.append("--disable-frontend-multiprocessing")
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


def test_chat_completion_replays_trace(server):
    client = server.get_client()

    response = client.chat.completions.create(model=MODEL,
                                              messages=CHAT,
                                              max_tokens=50)

    choice = response.choices[0]
    assert choice.message.content == CHAT_REPLY
    assert choice.finish_reason == "stop"


def test_completion_replays_trace(server):
    client = server.get_client()

    response = client.completions.create(model=MODEL,
                                         prompt=COMPLETION_PROMPT,
                                         max_tokens=50)

    assert response.choices[0].text == COMPLETION_REPLY
    assert response.choices[0].finish_reason == "stop"


def test_streaming_chat_replays_trace(server):
    client = server.get_client()

    stream = client.chat.completions.create(model=MODEL,
                                            messages=CHAT,
                                            max_tokens=50,
                                            stream=True)

    text = "".join(chunk.choices[0].delta.content or ""
                   for chunk in stream if chunk.choices)
    assert text == CHAT_REPLY


def test_unknown_prompt_stops_at_max_tokens(server):
    client = server.get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "not in the trace"
        }],
        max_tokens=4)

    assert response.usage.completion_tokens == 4
    assert response.choices[0].finish_reason == "length"
