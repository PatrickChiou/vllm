"""Online (OpenAI-compatible server) inference with the sim executor."""
import json

import pytest
import requests

from vllm.simulator.trace_builder import TraceBuilder

from ..utils import RemoteOpenAIServer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16
# Sent with every chat request and given to the TraceBuilder, as a client
# replaying a trace must do.
CHAT_TEMPLATE_KWARGS = {"date_string": "26 Jul 2024"}
CONVERSATION = [
    {
        "role": "system",
        "content": "You are a terse assistant."
    },
    {
        "role": "user",
        "content": "List the planets of the solar system in order."
    },
    {
        "role": "assistant",
        "content": ("Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus "
                    "and Neptune.")
    },
    {
        "role": "user",
        "content": "Which one is the largest?"
    },
    {
        "role": "assistant",
        "content": "Jupiter is the largest planet."
    },
]
FIRST_TURN = CONVERSATION[:2]
FIRST_REPLY = CONVERSATION[2]["content"]
SECOND_TURN = CONVERSATION[:4]
SECOND_REPLY = CONVERSATION[4]["content"]
COMPLETION_PROMPT = "Once upon a time"
COMPLETION_REPLY = " there was a simulator."


@pytest.fixture(scope="module")
def builder():
    return TraceBuilder(MODEL, chat_template_kwargs=CHAT_TEMPLATE_KWARGS)


@pytest.fixture(scope="module", params=[False, True])
def server(request, builder, tmp_path_factory):
    entries = builder.chat_entries(CONVERSATION)
    entries.append(
        builder.completion_entry(COMPLETION_PROMPT, COMPLETION_REPLY))
    trace_path = tmp_path_factory.mktemp("trace") / "trace.jsonl"
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in entries))

    args = [
        "--load-format", "dummy", "--distributed-executor-backend", "sim",
        "--sim-trace-path",
        str(trace_path), "--max-model-len", "1024", "--block-size",
        str(BLOCK_SIZE), "--enable-prefix-caching",
        "--enable-prompt-tokens-details"
    ]
    if request.param:
        # Exercises AsyncLLMEngine + SimExecutorAsync instead of MQLLMEngine.
        args.append("--disable-frontend-multiprocessing")
    with RemoteOpenAIServer(MODEL, args) as remote_server:
        yield remote_server


def _chat(server, messages, **kwargs):
    return server.get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"chat_template_kwargs": CHAT_TEMPLATE_KWARGS},
        **kwargs)


@pytest.mark.parametrize("messages", [FIRST_TURN, SECOND_TURN])
def test_trace_builder_matches_server_chat_tokenization(
        server, builder, messages):
    response = requests.post(server.url_for("tokenize"),
                             json={
                                 "model": MODEL,
                                 "messages": messages,
                                 "add_generation_prompt": True,
                                 "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                             })
    response.raise_for_status()

    assert response.json()["tokens"] == builder.chat_prompt_token_ids(
        messages)


def test_trace_builder_matches_server_completion_tokenization(
        server, builder):
    response = requests.post(server.url_for("tokenize"),
                             json={
                                 "model": MODEL,
                                 "prompt": COMPLETION_PROMPT
                             })
    response.raise_for_status()

    assert response.json()["tokens"] == builder.completion_prompt_token_ids(
        COMPLETION_PROMPT)


def test_multi_turn_chat_replays_and_reuses_previous_turn(server, builder):
    first = _chat(server, FIRST_TURN, max_tokens=100)
    assert first.choices[0].message.content == FIRST_REPLY
    assert first.choices[0].finish_reason == "stop"

    second = _chat(server,
                   SECOND_TURN[:2] + [{
                       "role": "assistant",
                       "content": first.choices[0].message.content
                   }] + SECOND_TURN[3:],
                   max_tokens=100)
    assert second.choices[0].message.content == SECOND_REPLY

    # The second prompt reuses the first prompt and its generated reply. The
    # final sampled token (EOS) is never fed back, so its KV is never
    # computed and the block holding it is not reusable.
    computed_tokens = (first.usage.prompt_tokens +
                       first.usage.completion_tokens - 1)
    cached = second.usage.prompt_tokens_details.cached_tokens
    assert cached == computed_tokens // BLOCK_SIZE * BLOCK_SIZE
    assert cached > first.usage.prompt_tokens


def test_completion_replays_trace(server):
    response = server.get_client().completions.create(
        model=MODEL, prompt=COMPLETION_PROMPT, max_tokens=50)

    assert response.choices[0].text == COMPLETION_REPLY
    assert response.choices[0].finish_reason == "stop"


def test_streaming_chat_replays_trace(server):
    stream = _chat(server, FIRST_TURN, max_tokens=100, stream=True)

    text = "".join(chunk.choices[0].delta.content or ""
                   for chunk in stream if chunk.choices)
    assert text == FIRST_REPLY


def test_unknown_prompt_stops_at_max_tokens(server):
    response = _chat(server, [{
        "role": "user",
        "content": "not in the trace"
    }],
                     max_tokens=4)

    assert response.usage.completion_tokens == 4
    assert response.choices[0].finish_reason == "length"
