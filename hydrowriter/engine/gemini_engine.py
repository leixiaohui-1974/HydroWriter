"""Google Gemini engine implementation."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
import urllib.parse
import urllib.request
from typing import Any

from hydrowriter.engine.base_engine import BaseEngine, EngineResponse


def _read_value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _flatten_parts(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""

    chunks: list[str] = []
    for part in parts:
        text = _read_value(part, "text", "")
        if text:
            chunks.append(str(text))
    return "".join(chunks).strip()


class GeminiEngine(BaseEngine):
    """Gemini engine using google.genai SDK or HTTP fallback."""

    provider = "google"

    def _load_sdk(self) -> Any | None:
        try:
            return importlib.import_module("google.genai")
        except ImportError:
            return None

    def _load_httpx(self) -> Any | None:
        try:
            return importlib.import_module("httpx")
        except ImportError:
            return None

    def _build_generate_content_url(self) -> str:
        root = self.base_url.rstrip("/") if self.base_url else "https://generativelanguage.googleapis.com"
        if not root.endswith("/v1beta"):
            root = f"{root}/v1beta"
        model = urllib.parse.quote(self.model, safe="")
        return f"{root}/models/{model}:generateContent"

    async def _generate_with_sdk(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, dict[str, Any]]:
        genai = self._load_sdk()
        if genai is None:
            raise ImportError("google.genai SDK is not installed")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["http_options"] = {"baseUrl": self.base_url}

        client = genai.Client(**client_kwargs)
        if hasattr(genai, "types") and hasattr(genai.types, "GenerateContentConfig"):
            config_obj = genai.types.GenerateContentConfig(
                maxOutputTokens=max_tokens,
                temperature=temperature,
                systemInstruction=system or None,
            )
        else:
            config_obj = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "system_instruction": system or None,
            }

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config_obj,
        )

        text = _read_value(response, "text", "")
        if not text:
            candidates = _read_value(response, "candidates", [])
            if candidates:
                first_candidate = candidates[0]
                content = _read_value(first_candidate, "content", {})
                text = _flatten_parts(_read_value(content, "parts", []))

        usage = _read_value(response, "usage_metadata") or _read_value(response, "usageMetadata", {})
        tokens = int(
            _read_value(usage, "total_token_count", _read_value(usage, "totalTokenCount", 0)) or 0
        )
        metadata = {"transport": "sdk"}
        return text, tokens, metadata

    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        httpx = self._load_httpx()
        if httpx is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, params={"key": self.api_key})
                response.raise_for_status()
                return response.json()

        def _send_request() -> dict[str, Any]:
            encoded_key = urllib.parse.quote(self.api_key, safe="")
            request_url = f"{url}?key={encoded_key}"
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                request_url,
                data=data,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))

        return await asyncio.to_thread(_send_request)

    async def _generate_with_http(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, dict[str, Any]]:
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        response = await self._post_json(self._build_generate_content_url(), payload=payload)
        candidates = response.get("candidates", [])
        text = ""
        if candidates:
            text = _flatten_parts(candidates[0].get("content", {}).get("parts", []))

        usage = response.get("usageMetadata", {})
        tokens = int(usage.get("totalTokenCount", 0) or 0)
        metadata = {"transport": "http"}
        return text, tokens, metadata

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> EngineResponse:
        start_time = time.perf_counter()
        try:
            if self._load_sdk() is not None:
                text, tokens, metadata = await self._generate_with_sdk(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                text, tokens, metadata = await self._generate_with_http(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            return self._make_response(text, tokens=tokens, start_time=start_time, **metadata)
        except Exception as exc:
            return self._make_response(
                "",
                tokens=0,
                start_time=start_time,
                error=str(exc),
                error_type=type(exc).__name__,
                provider=self.provider,
            )
