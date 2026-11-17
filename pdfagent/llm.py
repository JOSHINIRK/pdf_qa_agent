"""Thin wrappers around the OpenAI API, behind small protocols so tests can use fakes.

- Embedder:  texts -> vectors
- ChatModel: streams one model turn; yields text deltas (str) and finally an AssistantTurn
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A failure talking to the model provider, with a message fit for end users."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model produced it

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments are not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("arguments must be a JSON object")
        return value


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str = "auto"
    ) -> Iterator[str | AssistantTurn]: ...


def _friendly(exc: Exception) -> LLMError:
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return LLMError("OpenAI rejected the API key (401). Check OPENAI_API_KEY in .env.")
    if isinstance(exc, openai.RateLimitError):
        return LLMError("OpenAI rate limit or quota reached (429). Wait a minute, or add credit to the account.")
    if isinstance(exc, openai.NotFoundError):
        return LLMError(f"OpenAI model not found or not available to this account: {exc}")
    if isinstance(exc, openai.BadRequestError):
        return LLMError(f"OpenAI rejected the request: {exc}")
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return LLMError("Could not reach the OpenAI API (network error or timeout). Try again.")
    if isinstance(exc, openai.APIStatusError):
        return LLMError(f"OpenAI API error {exc.status_code}. Try again in a moment.")
    return LLMError(f"Unexpected error calling OpenAI: {exc}")


def _client(api_key: str, timeout: float):
    from openai import OpenAI

    # The SDK retries connection errors, 429 and 5xx responses with exponential back-off.
    return OpenAI(api_key=api_key, timeout=timeout, max_retries=3)


class OpenAIEmbedder:
    BATCH = 128

    def __init__(self, api_key: str, model: str, timeout: float = 90):
        self._client = _client(api_key, timeout)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), self.BATCH):
                batch = [t if t.strip() else " " for t in texts[i : i + self.BATCH]]
                resp = self._client.embeddings.create(model=self.model, input=batch)
                vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        except Exception as exc:
            raise _friendly(exc) from exc
        return vectors


class OpenAIChatModel:
    def __init__(self, api_key: str, model: str, temperature: float | None, max_tokens: int, timeout: float = 90):
        self._client = _client(api_key, timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: str = "auto"
    ) -> Iterator[str | AssistantTurn]:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
            params["parallel_tool_calls"] = True
        if self.temperature is not None:
            params["temperature"] = self.temperature

        content: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        prompt_tokens = completion_tokens = 0
        try:
            for event in self._client.chat.completions.create(**params):
                if event.usage:
                    prompt_tokens = event.usage.prompt_tokens
                    completion_tokens = event.usage.completion_tokens
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    content.append(delta.content)
                    yield delta.content
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except Exception as exc:
            raise _friendly(exc) from exc

        yield AssistantTurn(
            content="".join(content),
            tool_calls=[ToolCall(c["id"], c["name"], c["arguments"]) for _, c in sorted(calls.items())],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
