"""OpenAI-compatible engine implementation."""

from __future__ import annotations

import asyncio
import importlib
import json
import time
import urllib.request
from typing import Any

from hydrowriter.engine.base_engine import BaseEngine, EngineResponse


def _read_value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _message_to_text(message: Any) -> str:
    content = _read_value(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        text = _read_value(item, "text", "")
        if text:
            parts.append(str(text))
    return "".join(parts).strip()


class OpenAIEngine(BaseEngine):
    """OpenAI-compatible engine using SDK or HTTP fallback."""

    provider = "openai"

    def _load_sdk(self) -> Any | None:
        try:
            return importlib.import_module("openai")
        except ImportError:
            return None

    def _load_httpx(self) -> Any | None:
        try:
            return importlib.import_module("httpx")
        except ImportError:
            return None

    def _build_chat_completions_url(self) -> str:
        if not self.base_url:
            return "https://api.openai.com/v1/chat/completions"

        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    async def _generate_with_sdk(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, dict[str, Any]]:
        openai = self._load_sdk()
        if openai is None:
            raise ImportError("openai SDK is not installed")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client_cls = getattr(openai, "AsyncOpenAI", None)
        if client_cls is None:
            raise RuntimeError("openai.AsyncOpenAI is unavailable")

        client = client_cls(**client_kwargs)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choices = _read_value(response, "choices", [])
        text = ""
        if choices:
            text = _message_to_text(_read_value(choices[0], "message", {}))

        usage = _read_value(response, "usage")
        tokens = int(_read_value(usage, "total_tokens", 0) or 0)
        metadata = {
            "transport": "sdk",
            "finish_reason": _read_value(choices[0], "finish_reason") if choices else None,
        }
        return text, tokens, metadata

    async def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        httpx = self._load_httpx()
        if httpx is not None:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()

        def _send_request() -> dict[str, Any]:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._post_json(
            self._build_chat_completions_url(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            payload={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        choices = response.get("choices", [])
        text = ""
        if choices:
            text = _message_to_text(choices[0].get("message", {}))

        usage = response.get("usage", {})
        tokens = int(usage.get("total_tokens", 0) or 0)
        metadata = {
            "transport": "http",
            "finish_reason": choices[0].get("finish_reason") if choices else None,
        }
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
