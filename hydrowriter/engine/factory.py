"""Engine Factory — create LLM engines from configuration."""

from __future__ import annotations

from hydrowriter.config import EngineConfig, WriterConfig
from hydrowriter.engine.base_engine import BaseEngine
from hydrowriter.engine.claude_engine import ClaudeEngine
from hydrowriter.engine.gemini_engine import GeminiEngine
from hydrowriter.engine.openai_engine import OpenAIEngine


_PROVIDER_MAP = {
    "anthropic": ClaudeEngine,
    "google": GeminiEngine,
    "openai": OpenAIEngine,
    "openai_compatible": OpenAIEngine,
}


def create_engine(config: EngineConfig) -> BaseEngine:
    """Create an engine instance from configuration."""
    cls = _PROVIDER_MAP.get(config.provider, OpenAIEngine)
    return cls(
        name=config.name,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )


def create_all_engines(writer_config: WriterConfig) -> dict[str, BaseEngine]:
    """Create all configured engines, skipping those without API keys."""
    engines = {}
    for name, econfig in writer_config.engines.items():
        if econfig.is_available:
            engines[name] = create_engine(econfig)
    return engines
