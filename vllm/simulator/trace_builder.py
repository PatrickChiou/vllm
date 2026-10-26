"""Builds simulator trace entries whose prompt token ids match what vLLM's
OpenAI-compatible server feeds the engine.

The simulator looks prompts up by exact token ids, so traces must be
tokenized the way the server tokenizes requests. `TraceBuilder` reuses the
server's own preprocessing (chat message parsing, chat template rendering
and tokenization flags) instead of re-implementing it.

Some chat templates render values that change between runs. Llama 3.x, for
example, inserts today's date, so a trace built on one day misses every
request on the next. Pin such values with `chat_template_kwargs` (e.g.
`{"date_string": "26 Jul 2024"}`) and send the same `chat_template_kwargs`
with every chat request.
"""
from typing import Any, Dict, List, Optional, Sequence

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         load_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
from vllm.transformers_utils.tokenizer import get_tokenizer


class TraceBuilder:

    def __init__(
        self,
        model: str,
        *,
        tokenizer: Optional[str] = None,
        chat_template: Optional[str] = None,
        chat_template_content_format: ChatTemplateContentFormatOption = "auto",
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
    ) -> None:
        """Arguments mirror the `vllm serve` flags of the same name; pass the
        values the server will be started with. `chat_template_kwargs` must
        equal the field of the same name in every chat request."""
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.model_config = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            revision=revision,
        ).create_model_config()
        self.tokenizer = get_tokenizer(
            self.model_config.tokenizer,
            tokenizer_mode=self.model_config.tokenizer_mode,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.tokenizer_revision)
        self.chat_template = load_chat_template(chat_template)
        self.content_format = resolve_chat_template_content_format(
            self.chat_template, chat_template_content_format, self.tokenizer)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def chat_prompt_token_ids(
            self,
            messages: Sequence[ChatCompletionMessageParam],
            add_generation_prompt: bool = True) -> List[int]:
        """Prompt token ids of a /v1/chat/completions request."""
        conversation, mm_data = parse_chat_messages(list(messages),
                                                    self.model_config,
                                                    self.tokenizer,
                                                    self.content_format)
        if mm_data:
            raise ValueError("Multimodal messages are not supported.")
        # Same keyword arguments, in the same precedence, as
        # OpenAIServing._preprocess_chat.
        template_kwargs: Dict[str, Any] = dict(
            chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=False,
            tools=None,
            documents=None,
        )
        template_kwargs.update(self.chat_template_kwargs)
        prompt = apply_hf_chat_template(self.tokenizer,
                                        conversation=conversation,
                                        **template_kwargs)
        # ChatCompletionRequest.add_special_tokens defaults to False.
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def completion_prompt_token_ids(self, prompt: str) -> List[int]:
        """Prompt token ids of a /v1/completions request."""
        # CompletionRequest.add_special_tokens defaults to True.
        return self.tokenizer(prompt, add_special_tokens=True).input_ids

    def reply_token_ids(
            self, messages: Sequence[ChatCompletionMessageParam]) -> List[int]:
        """Token ids of the final (assistant) message, excluding EOS.

        Taken from how the reply is tokenized inside the rendered
        conversation, so that when the simulator emits these tokens followed
        by EOS, the next turn's prompt extends this turn's prompt plus its
        generated tokens. That is the prefix the KV cache can reuse across
        turns.
        """
        prompt = self.chat_prompt_token_ids(messages[:-1])
        rendered = self.chat_prompt_token_ids(messages,
                                              add_generation_prompt=False)
        eos = self.eos_token_id
        if rendered[:len(prompt)] == prompt:
            reply = rendered[len(prompt):]
            if eos in reply:
                return reply[:reply.index(eos)]
        # The template does not render the reply after the generation prompt
        # (or ends it without EOS); fall back to tokenizing the text alone.
        content = messages[-1].get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Only string message content is supported.")
        return self.tokenizer(content, add_special_tokens=False).input_ids

    def chat_entries(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> List[Dict[str, Any]]:
        """One trace entry per assistant message of a conversation, keyed by
        the chat prompt that precedes it."""
        return [{
            "prompt_token_ids": self.chat_prompt_token_ids(messages[:i]),
            "output_token_ids": self.reply_token_ids(messages[:i + 1]),
        } for i, message in enumerate(messages)
                if message["role"] == "assistant"]

    def completion_entry(self, prompt: str,
                         completion: str) -> Dict[str, Any]:
        return {
            "prompt_token_ids":
            self.completion_prompt_token_ids(prompt),
            "output_token_ids":
            self.tokenizer(completion, add_special_tokens=False).input_ids,
        }
