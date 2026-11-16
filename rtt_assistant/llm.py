"""Chat model clients. Every provider here speaks the OpenAI chat-completions protocol, so one
client covers them all; only the address, the key and the model name change.

Choose with LLM_PROVIDER (environment or Streamlit secrets) and optionally LLM_MODEL:

    provider      address                              key variable     cost
    groq          https://api.groq.com/openai/v1       GROQ_API_KEY     free tier (rate-limited), open-weight models
    ollama        http://localhost:11434/v1            (none)           free, runs on your own machine
    huggingface   https://router.huggingface.co/v1     HF_TOKEN         pay-as-you-go credits
    openai        https://api.openai.com/v1            OPENAI_API_KEY   paid

The default models are open-weight models published on Hugging Face, except for OpenAI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import requests


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None
    default_model: str


PROVIDERS = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "ollama": Provider("ollama", "http://localhost:11434/v1", None, "qwen2.5-coder:7b"),
    "huggingface": Provider("huggingface", "https://router.huggingface.co/v1", "HF_TOKEN",
                            "Qwen/Qwen2.5-Coder-32B-Instruct"),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.2"),
}
DEFAULT_PROVIDER = "groq"
THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class OpenAICompatibleChat:
    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout_s: float = 120.0, settings: dict | None = None):
        env = {**os.environ, **(settings or {})}
        self.provider = PROVIDERS[(provider or env.get("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()]
        self.base_url = (base_url or env.get("LLM_BASE_URL") or self.provider.base_url).rstrip("/")
        self.model = model or env.get("LLM_MODEL") or self.provider.default_model
        self.api_key = api_key if api_key is not None else (env.get(self.provider.key_env, "")
                                                            if self.provider.key_env else "none")
        self.timeout_s = timeout_s
        self._json_mode_ok = True

    @property
    def name(self) -> str:
        return f"{self.provider.name}:{self.model}"

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.api_key:
            raise LLMError(f"No API key set for {self.provider.name} ({self.provider.key_env}).")
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}]}
        if json_mode and self._json_mode_ok:
            body["response_format"] = {"type": "json_object"}
        r = self._post(body)
        if r.status_code == 400 and "response_format" in body and "response_format" in r.text:
            self._json_mode_ok = False  # this provider/model doesn't support JSON mode; the parser copes
            body = {k: v for k, v in body.items() if k != "response_format"}
            r = self._post(body)
        if r.status_code != 200:
            raise LLMError(f"{self.provider.name} API error {r.status_code}: {r.text[:300]}")
        content = r.json()["choices"][0]["message"].get("content") or ""
        return THINK.sub("", content).strip()  # some open models include their reasoning in <think> tags

    def _post(self, body: dict) -> requests.Response:
        try:
            return requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout_s,
                                 headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as e:
            raise LLMError(f"Could not reach {self.provider.name} at {self.base_url}: {e}") from e


class ScriptedChat:
    """Returns pre-written responses in order. Each response is a string, or a function of
    (system, user) -> str. Records every call, so tests can inspect the prompts."""

    def __init__(self, responses: list[str | Callable[[str, str], str]], name: str = "scripted"):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self.name = name

    available = True

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("ScriptedChat ran out of responses.")
        r = self.responses.pop(0)
        return r(system, user) if callable(r) else r
