"""A small OpenAI chat client (plain HTTPS, no SDK) and a scripted stand-in for tests.

Configuration comes from the environment or Streamlit secrets:
    OPENAI_API_KEY   required for live use
    RTT_LLM_MODEL    model name (default below)
"""

from __future__ import annotations

import os
from typing import Callable, Protocol

import requests

DEFAULT_MODEL = "gpt-5.2"
API_URL = "https://api.openai.com/v1/chat/completions"


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class OpenAIChat:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout_s: float = 90.0):
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.name = model or os.environ.get("RTT_LLM_MODEL", DEFAULT_MODEL)
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        if not self.api_key:
            raise LLMError("No OpenAI API key set (OPENAI_API_KEY).")
        body = {"model": self.name, "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(API_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                              timeout=self.timeout_s)
        except requests.RequestException as e:
            raise LLMError(f"Could not reach the OpenAI API: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"OpenAI API error {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


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
