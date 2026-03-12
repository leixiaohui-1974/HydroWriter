"""Gemini Engine — Google GenAI API integration."""

from __future__ import annotations

import time
from typing import Any

from hydrowriter.engine.base_engine import BaseEngine, EngineResponse


class GeminiEngine(BaseEngine):
    """LLM engine using Google Gemini API."""

    def __init__(self, name: str = "gemini", model: str = "gemini-2.5-pro",
                 api_key: str = "", base_url: str = ""):
        super().__init__(name, model, api_key, base_url)
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                kwargs: dict[str, Any] = {"api_key": self.api_key}
                if self.base_url:
                    # Custom endpoint support
                    kwargs["transport"] = "rest"
                self._client = genai.Client(**kwargs)
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
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )
            text = response.text or ""
            tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = getattr(response.usage_metadata, "total_token_count", 0)
            return self._make_response(text, tokens=tokens, start_time=start)
        except Exception as e:
            return self._make_response(f"[Gemini Error] {e}", start_time=start)

    async def _fallback_generate(self, prompt: str, system: str,
                                  max_tokens: int, temperature: float) -> EngineResponse:
        """Fallback using REST API."""
        import json
        from urllib.request import Request, urlopen

        base = self.base_url or "https://generativelanguage.googleapis.com"
        url = f"{base}/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        body = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        headers = {"Content-Type": "application/json"}
        start = time.perf_counter()
        try:
            req = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
            return self._make_response(text, tokens=tokens, start_time=start)
        except Exception as e:
            return self._make_response(f"[Gemini Fallback Error] {e}", start_time=start)
