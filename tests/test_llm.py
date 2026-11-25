import json

import pytest

from rtt_assistant import llm as llm_mod
from rtt_assistant.llm import LLMError, OpenAICompatibleChat


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text or json.dumps(payload or {})

    def json(self):
        return self._payload


def reply(content):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_provider_presets_and_overrides():
    c = OpenAICompatibleChat(provider="ollama")
    assert c.base_url == "http://localhost:11434/v1" and c.available  # no key needed locally
    g = OpenAICompatibleChat(settings={"LLM_PROVIDER": "groq", "GROQ_API_KEY": "k", "LLM_MODEL": "m"})
    assert g.name == "groq:m" and g.base_url.startswith("https://api.groq.com")
    assert not OpenAICompatibleChat(provider="openai", api_key="").available


def test_strips_reasoning_and_falls_back_when_json_mode_is_unsupported(monkeypatch):
    sent = []

    def fake_post(url, json, timeout, headers):
        sent.append(json)
        if "response_format" in json:
            return FakeResponse(400, text='{"error": "response_format is not supported"}')
        return reply('<think>let me see {draft}</think>{"sql": "SELECT 1"}')

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    c = OpenAICompatibleChat(provider="huggingface", api_key="t")
    assert c.complete("s", "u", json_mode=True) == '{"sql": "SELECT 1"}'
    assert "response_format" in sent[0] and "response_format" not in sent[1]
    c.complete("s", "u", json_mode=True)
    assert "response_format" not in sent[-1]  # remembered


def test_errors_are_reported(monkeypatch):
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: FakeResponse(429, text="rate limited"))
    with pytest.raises(LLMError, match="429"):
        OpenAICompatibleChat(provider="groq", api_key="k").complete("s", "u")
