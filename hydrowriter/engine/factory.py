"""Factory helpers for LLM engines."""

from __future__ import annotations

from hydrowriter.config import EngineConfig, WriterConfig
from hydrowriter.engine.base_engine import BaseEngine
from hydrowriter.engine.claude_engine import ClaudeEngine
from hydrowriter.engine.gemini_engine import GeminiEngine
from hydrowriter.engine.openai_engine import OpenAIEngine


def create_engine(config: EngineConfig) -> BaseEngine:
    """Create a single engine from config."""
    provider = config.provider.strip().lower()

    if provider in {"anthropic", "claude"}:
        return ClaudeEngine(
            name=config.name,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    if provider in {"google", "gemini"}:
        return GeminiEngine(
            name=config.name,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    if provider in {"openai", "openai_compatible", "deepseek", "qwen", "dashscope"}:
        return OpenAIEngine(
            name=config.name,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    raise ValueError(f"Unsupported engine provider: {config.provider}")


def create_all_engines(writer_config: WriterConfig) -> dict[str, BaseEngine]:
    """Create all configured engines."""
    return {
        name: create_engine(config)
        for name, config in writer_config.engines.items()
        if config.is_available
    }
