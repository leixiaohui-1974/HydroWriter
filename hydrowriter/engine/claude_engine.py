"""Claude Engine — Anthropic API integration."""

from __future__ import annotations

import time
from typing import Any

from hydrowriter.engine.base_engine import BaseEngine, EngineResponse


class ClaudeEngine(BaseEngine):
    """LLM engine using Anthropic Claude API."""

    def __init__(self, name: str = "claude", model: str = "claude-opus-4-6",
                 api_key: str = "", base_url: str = ""):
        super().__init__(name, model, api_key, base_url)
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                kwargs: dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = anthropic.AsyncAnthropic(**kwargs)
            except ImportError:
                self._client = None
        return self._client

    async def generate(
        self, prompt: str, system: str = "",
        max_tokens: int = 4096, temperature: float = 0.7,
    ) -> EngineResponse:
        start = time.perf_counter()
        client = self._get_client()

        if client is None:
            return self._fallback_generate(prompt, system, max_tokens)

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = await client.messages.create(**kwargs)
            text = response.content[0].text if response.content else ""
            tokens = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
            return self._make_response(text, tokens=tokens, start_time=start)
        except Exception as e:
            return self._make_response(f"[Claude Error] {e}", start_time=start)

    async def _fallback_generate(self, prompt: str, system: str, max_tokens: int) -> EngineResponse:
        """Fallback using urllib when anthropic SDK not installed."""
        import json
        from urllib.request import Request, urlopen

        url = (self.base_url or "https://api.anthropic.com") + "/v1/messages"
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        start = time.perf_counter()
        try:
            req = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data.get("content", [{}])[0].get("text", "")
            tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
            return self._make_response(text, tokens=tokens, start_time=start)
        except Exception as e:
            return self._make_response(f"[Claude Fallback Error] {e}", start_time=start)
