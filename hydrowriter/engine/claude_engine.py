"""Anthropic Claude engine implementation."""

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


def _read_text_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        item_type = _read_value(item, "type")
        if item_type == "text":
            text = _read_value(item, "text", "")
            if text:
                parts.append(str(text))
    return "".join(parts).strip()


def _read_total_tokens(usage: Any, *field_names: str) -> int:
    total = 0
    for field_name in field_names:
        value = _read_value(usage, field_name, 0)
        if isinstance(value, int):
            total += value
    return total


class ClaudeEngine(BaseEngine):
    """Claude engine using Anthropic SDK or HTTP fallback."""

    provider = "anthropic"

    def _load_sdk(self) -> Any | None:
        try:
            return importlib.import_module("anthropic")
        except ImportError:
            return None

    def _load_httpx(self) -> Any | None:
        try:
            return importlib.import_module("httpx")
        except ImportError:
            return None

    def _build_messages_url(self) -> str:
        if not self.base_url:
            return "https://api.anthropic.com/v1/messages"

        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/messages"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/messages"
        return f"{base_url}/v1/messages"

    async def _generate_with_sdk(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, dict[str, Any]]:
        anthropic = self._load_sdk()
        if anthropic is None:
            raise ImportError("anthropic SDK is not installed")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client_cls = getattr(anthropic, "AsyncAnthropic", None)
        if client_cls is None:
            raise RuntimeError("anthropic.AsyncAnthropic is unavailable")

        client = client_cls(**client_kwargs)
        response = await client.messages.create(
            model=self.model,
            system=system or None,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _read_text_blocks(_read_value(response, "content", []))
        usage = _read_value(response, "usage")
        tokens = _read_total_tokens(usage, "input_tokens", "output_tokens")
        metadata = {"transport": "sdk", "stop_reason": _read_value(response, "stop_reason")}
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
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        response = await self._post_json(
            self._build_messages_url(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            payload=payload,
        )
        usage = response.get("usage", {})
        tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
        metadata = {"transport": "http", "stop_reason": response.get("stop_reason")}
        return _read_text_blocks(response.get("content", [])), tokens, metadata

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
