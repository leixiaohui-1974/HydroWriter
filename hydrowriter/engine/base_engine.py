"""Base LLM Engine — abstract interface for all LLM providers.

所有 LLM 引擎的抽象基类，统一接口:
- generate(): 生成文本
- review(): 评审文本
- chat(): 对话
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineResponse:
    """Unified response from any LLM engine."""
    text: str
    engine_name: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.text)


class BaseEngine(ABC):
    """Abstract base class for LLM engines."""

    def __init__(self, name: str, model: str, api_key: str = "", base_url: str = ""):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> EngineResponse:
        """Generate text from prompt."""
        ...

    async def review(
        self,
        content: str,
        role: str = "reviewer",
        criteria: List[str] | None = None,
    ) -> EngineResponse:
        """Review content from a specific role perspective."""
        criteria_text = "\n".join(f"- {c}" for c in (criteria or ["质量", "逻辑", "可读性"]))
        prompt = (
            f"请从{role}的角度评审以下内容，重点关注:\n{criteria_text}\n\n"
            f"---\n{content}\n---\n\n"
            f"请给出评分(1-10)和具体改进建议。"
        )
        return await self.generate(prompt, max_tokens=4096)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
    ) -> EngineResponse:
        """Multi-turn chat. Default: convert to single prompt."""
        combined = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in messages
        )
        return await self.generate(combined, max_tokens=max_tokens)

    def _make_response(
        self, text: str, tokens: int = 0, start_time: float = 0, **meta
    ) -> EngineResponse:
        elapsed = (time.perf_counter() - start_time) * 1000 if start_time else 0
        return EngineResponse(
            text=text,
            engine_name=self.name,
            model=self.model,
            tokens_used=tokens,
            latency_ms=round(elapsed, 1),
            metadata=meta,
        )
