"""OpenAI-Compatible Engine — works with GPT, DeepSeek, Qwen, etc."""

from __future__ import annotations

import time
from typing import Any

from hydrowriter.engine.base_engine import BaseEngine, EngineResponse


class OpenAIEngine(BaseEngine):
    """LLM engine using OpenAI-compatible API."""

    def __init__(self, name: str = "gpt", model: str = "gpt-4.1",
                 api_key: str = "", base_url: str = ""):
        super().__init__(name, model, api_key, base_url or "https://api.openai.com/v1")
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
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
            return await self._fallback_generate(prompt, system, max_tokens, temperature)

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return self._make_response(text, tokens=tokens, start_time=start)
        except Exception as e:
            return self._make_response(f"[OpenAI Error] {e}", start_time=start)

    async def _fallback_generate(self, prompt: str, system: str,
                                  max_tokens: int, temperature: float) -> EngineResponse:
        """Fallback using urllib."""
        import json
        from urllib.request import Request, urlopen

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        start = time.perf_counter()
        try:
            req = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return self._make_response(text, tokens=tokens, start_time=start)
        except Exception as e:
            return self._make_response(f"[OpenAI Fallback Error] {e}", start_time=start)
