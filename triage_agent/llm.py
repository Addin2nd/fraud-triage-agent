"""Minimal, dependency-free LLM clients with native tool calling.

Two wire formats cover almost every provider:
* Anthropic Messages API  (Claude)
* OpenAI Chat Completions (OpenAI, Gemini's OpenAI-compatible endpoint, Groq,
  OpenRouter, local Ollama / LM Studio / vLLM)

The agent speaks a provider-neutral message format:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "results": [{"id", "name", "content": str}]}
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

PRESETS = {
    # provider: (wire format, base url, api-key env var, default model)
    "anthropic": ("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-sonnet-4-5"),
    "openai": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "groq": ("openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "anthropic/claude-sonnet-4.5"),
    "ollama": ("openai", "http://localhost:11434/v1", "", "qwen2.5:7b-instruct"),
}


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[dict]
    usage: dict = field(default_factory=dict)
    stop_reason: str = ""


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, temperature: float = 0.0, max_tokens: int = 1500, timeout: int = 90):
        if provider not in PRESETS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {sorted(PRESETS)}")
        self.wire, default_url, key_env, default_model = PRESETS[provider]
        self.provider = provider
        self.model = model or os.getenv("LLM_MODEL") or default_model
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or default_url).rstrip("/")
        self.api_key = api_key or (os.getenv(key_env) if key_env else "") or os.getenv("LLM_API_KEY", "")
        if key_env and not self.api_key:
            raise LLMError(f"Missing API key: set {key_env} (or LLM_API_KEY) in your environment / .env")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    # ------------------------------------------------------------------ http
    def _post(self, url: str, payload: dict, headers: dict) -> dict:
        data = json.dumps(payload).encode()
        last = None
        for attempt in range(4):
            req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:500]
                last = LLMError(f"HTTP {e.code} from {self.provider}: {body}")
                if e.code not in (408, 429, 500, 502, 503, 504, 529):
                    raise last
            except (urllib.error.URLError, TimeoutError) as e:
                last = LLMError(f"Network error calling {self.provider}: {e}")
            time.sleep(min(2 ** attempt * 1.5, 20))
        raise last  # type: ignore[misc]

    # ------------------------------------------------------------------ chat
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        return self._anthropic(system, messages, tools) if self.wire == "anthropic" else self._openai(system, messages, tools)

    def _anthropic(self, system, messages, tools) -> AssistantTurn:
        msgs = []
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in m["tool_calls"]]
                msgs.append({"role": "assistant", "content": blocks})
            else:
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]} for r in m["results"]]})
        payload = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                   "system": system, "messages": msgs,
                   "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]}
        r = self._post(f"{self.base_url}/messages", payload,
                       {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in r.get("content", []) if b["type"] == "text")
        calls = [{"id": b["id"], "name": b["name"], "args": b.get("input") or {}} for b in r.get("content", []) if b["type"] == "tool_use"]
        u = r.get("usage", {})
        return AssistantTurn(text, calls, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)},
                             r.get("stop_reason", ""))

    def _openai(self, system, messages, tools) -> AssistantTurn:
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("text") or None}
                if m["tool_calls"]:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in m["tool_calls"]]
                msgs.append(msg)
            else:
                msgs += [{"role": "tool", "tool_call_id": r["id"], "content": r["content"]} for r in m["results"]]
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "messages": msgs,
                   "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "auto"}
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self._post(f"{self.base_url}/chat/completions", payload, headers)
        choice = r["choices"][0]
        msg = choice["message"]
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c["function"].get("arguments")}
            calls.append({"id": c.get("id") or f"call_{i}", "name": c["function"]["name"], "args": args})
        u = r.get("usage") or {}
        return AssistantTurn(msg.get("content") or "", calls,
                             {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)},
                             choice.get("finish_reason", ""))


class ScriptedLLM:
    """Deterministic fake LLM for tests and demos: replays a list of AssistantTurns."""

    def __init__(self, turns: list[AssistantTurn], name: str = "scripted"):
        self.turns, self._i, self._name = turns, 0, name

    @property
    def name(self) -> str:
        return self._name

    def chat(self, system, messages, tools) -> AssistantTurn:
        turn = self.turns[min(self._i, len(self.turns) - 1)]
        self._i += 1
        return turn
