"""LLM client for the tree search solver — supports OpenAI-compatible APIs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


def _detect_provider(model: str) -> str:
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    return "openai"


_PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "anthropic": "https://api.anthropic.com/v1/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "o4-mini",
        base_url: str | None = None,
        provider: str | None = None,
    ):
        self.model = model
        self.provider = provider or _detect_provider(model)
        effective_base_url = base_url or _PROVIDER_BASE_URLS.get(self.provider)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if effective_base_url:
            kwargs["base_url"] = effective_base_url
        self._client = OpenAI(**kwargs)
        logger.info(
            "LLMClient: provider=%s, model=%s, base_url=%s",
            self.provider, self.model, effective_base_url,
        )

    def generate(self, *, system: str, user: str, temperature: float = 1.0) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        model_lower = self.model.lower()
        is_reasoning = any(model_lower.startswith(p) for p in ("o1", "o3", "o4"))
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if self.provider == "anthropic":
            kwargs["max_tokens"] = 16384

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return LLMResponse(text=text, usage=usage)

    def generate_code(self, *, system: str, user: str, temperature: float = 1.0) -> str:
        resp = self.generate(system=system, user=user, temperature=temperature)
        return self._extract_code(resp.text)

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            code = text.split("```python", 1)[1].split("```", 1)[0]
            return code.strip()
        if "```" in text:
            code = text.split("```", 1)[1].split("```", 1)[0]
            return code.strip()
        return text.strip()
