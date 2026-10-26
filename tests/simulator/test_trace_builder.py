import pytest

from vllm.simulator.trace_builder import TraceBuilder

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
# Same tokenizer and chat template as meta-llama/Llama-3.2-1B-Instruct,
# without the license gate.
LLAMA = "unsloth/Llama-3.2-1B-Instruct"

CONVERSATION = [
    {
        "role": "user",
        "content": "Hi"
    },
    {
        "role": "assistant",
        "content": "  Hello! How can I help?  "
    },
    {
        "role": "user",
        "content": "Tell me a joke."
    },
    {
        "role": "assistant",
        "content": "Why did the chicken cross the road?"
    },
]


@pytest.mark.parametrize("model", [QWEN, LLAMA])
def test_next_turn_prompt_extends_previous_turn_and_reply(model):
    builder = TraceBuilder(model)

    first, second = builder.chat_entries(CONVERSATION)

    replayed_turn = (first["prompt_token_ids"] + first["output_token_ids"] +
                     [builder.eos_token_id])
    assert second["prompt_token_ids"][:len(replayed_turn)] == replayed_turn
    assert builder.eos_token_id not in first["output_token_ids"]


def test_reply_matches_template_rendering_not_raw_text():
    # The Llama 3 template trims message content, so the reply the model
    # must have generated is the trimmed text.
    builder = TraceBuilder(LLAMA)

    reply = builder.reply_token_ids(CONVERSATION[:2])

    assert builder.tokenizer.decode(reply) == "Hello! How can I help?"


def test_chat_template_kwargs_pin_llama_date():
    pinned = TraceBuilder(LLAMA,
                          chat_template_kwargs={"date_string": "26 Jul 2024"})

    prompt = pinned.tokenizer.decode(
        pinned.chat_prompt_token_ids(CONVERSATION[:1]))

    assert "Today Date: 26 Jul 2024" in prompt
