"""
LLM client for any OpenAI-compatible chat endpoint.

"""
from __future__ import annotations

import logging
import time
from typing import Protocol

import openai

from ragbot.config import Settings
from ragbot.errors import LLMError, LLMOutputError

log = logging.getLogger(__name__)

_TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
              openai.InternalServerError)


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False,
                 max_tokens: int | None = None) -> str: ...


class OpenAICompatibleLLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client = openai.OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=0,              # retries are handled below with explicit backoff
        )
        self._json_supported = settings.llm_json_mode

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False,
                 max_tokens: int | None = None) -> str:
        attempts = self.s.llm_max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            kwargs: dict = {
                "model": self.s.llm_model,
                "messages": messages,
                "temperature": self.s.llm_temperature,
                "max_tokens": max_tokens or self.s.llm_max_tokens,
            }
            if json_mode and self._json_supported:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except _TRANSIENT as exc:
                last_exc = exc
                log.warning("LLM transient failure (attempt %d/%d): %s", attempt + 1, attempts, exc)
                if attempt < attempts - 1:
                    time.sleep(min(2 ** attempt, 8))
                continue
            except openai.BadRequestError as exc:
                if "response_format" in kwargs:
                    log.warning("server rejected response_format; falling back to prompt-only JSON")
                    self._json_supported = False
                    last_exc = exc
                    continue
                raise LLMError(f"LLM rejected request: {exc}",
                               "The language model rejected the request (the input may be too long).") from exc
            except openai.APIStatusError as exc:   # auth errors, unknown model, ...
                raise LLMError(f"LLM API error {exc.status_code}: {exc}",
                               "The language model service returned an error. Check the model configuration.") from exc

            content = (resp.choices[0].message.content or "") if resp.choices else ""
            if not content.strip():
                raise LLMOutputError("LLM returned empty content", "The language model returned an empty response.")
            return content

        raise LLMError(f"LLM unavailable after {attempts} attempts: {last_exc}",
                       "The language model is unavailable right now. Please try again shortly.")
