"""Multi-engine configuration loader.

支持从 YAML 加载多个 LLM 引擎配置，每个引擎有独立的:
- provider: anthropic / google / openai / openai_compatible
- model: 模型 ID
- api_key: API 密钥 (支持环境变量引用 ${VAR})
- base_url: 可选代理 URL
- role: architect / reviewer / engineer / specialist
- strengths: 擅长领域列表
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class EngineConfig:
    """Single LLM engine configuration."""
    name: str
    provider: str  # anthropic, google, openai, openai_compatible
    model: str
    api_key: str = ""
    base_url: str = ""
    role: str = "general"  # architect, reviewer, engineer, specialist
    strengths: List[str] = field(default_factory=list)
    max_tokens: int = 8192
    temperature: float = 0.7

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)


@dataclass
class WriterConfig:
    """Full writer configuration."""
    engines: Dict[str, EngineConfig] = field(default_factory=dict)
    default_engine: str = "claude"
    parallel_drafting: bool = True
    review_roles: List[str] = field(default_factory=lambda: ["teacher", "engineer", "reader"])
    consensus_threshold: int = 2  # ≥N engines agree = must fix
    output_dir: str = "./output"
    knowledge_base_path: str = ""


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} with environment variable values."""
    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return re.sub(r'\$\{(\w+)\}', replacer, str(value))


def load_config(path: str | Path) -> WriterConfig:
    """Load writer configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        return WriterConfig()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    engines = {}
    for name, econf in raw.get("engines", {}).items():
        engines[name] = EngineConfig(
            name=name,
            provider=econf.get("provider", "openai_compatible"),
            model=econf.get("model", ""),
            api_key=_resolve_env_vars(econf.get("api_key", "")),
            base_url=_resolve_env_vars(econf.get("base_url", "")),
            role=econf.get("role", "general"),
            strengths=econf.get("strengths", []),
            max_tokens=econf.get("max_tokens", 8192),
            temperature=econf.get("temperature", 0.7),
        )

    return WriterConfig(
        engines=engines,
        default_engine=raw.get("default_engine", "claude"),
        parallel_drafting=raw.get("parallel_drafting", True),
        review_roles=raw.get("review_roles", ["teacher", "engineer", "reader"]),
        consensus_threshold=raw.get("consensus_threshold", 2),
        output_dir=raw.get("output_dir", "./output"),
        knowledge_base_path=raw.get("knowledge_base_path", ""),
    )
