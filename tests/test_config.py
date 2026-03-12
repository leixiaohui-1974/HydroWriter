"""Tests for HydroWriter configuration and engine factory."""

from __future__ import annotations

import os
import pytest
from pathlib import Path

from hydrowriter.config import EngineConfig, WriterConfig, load_config, _resolve_env_vars
from hydrowriter.engine import (
    BaseEngine, EngineResponse,
    ClaudeEngine, GeminiEngine, OpenAIEngine,
    create_engine, create_all_engines,
)


class TestEngineConfig:
    def test_available_with_key(self):
        cfg = EngineConfig(name="test", provider="openai", model="gpt-4", api_key="sk-xxx")
        assert cfg.is_available is True

    def test_unavailable_without_key(self):
        cfg = EngineConfig(name="test", provider="openai", model="gpt-4")
        assert cfg.is_available is False


class TestResolveEnvVars:
    def test_resolve_existing_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "hello123")
        assert _resolve_env_vars("${TEST_KEY}") == "hello123"

    def test_resolve_missing_var(self):
        assert _resolve_env_vars("${NONEXISTENT_VAR_XYZ}") == ""

    def test_no_var(self):
        assert _resolve_env_vars("plain text") == "plain text"


class TestLoadConfig:
    def test_load_nonexistent(self, tmp_path: Path):
        cfg = load_config(tmp_path / "nope.yaml")
        assert isinstance(cfg, WriterConfig)
        assert len(cfg.engines) == 0

    def test_load_valid_yaml(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "key123")
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
engines:
  test_engine:
    provider: openai
    model: gpt-4
    api_key: ${TEST_API_KEY}
    role: engineer
    strengths: [code, math]
default_engine: test_engine
""", encoding="utf-8")

        cfg = load_config(yaml_file)
        assert "test_engine" in cfg.engines
        assert cfg.engines["test_engine"].api_key == "key123"
        assert cfg.engines["test_engine"].role == "engineer"
        assert cfg.default_engine == "test_engine"


class TestEngineFactory:
    def test_create_claude(self):
        cfg = EngineConfig(name="c", provider="anthropic", model="claude-opus-4-6", api_key="k")
        engine = create_engine(cfg)
        assert isinstance(engine, ClaudeEngine)

    def test_create_gemini(self):
        cfg = EngineConfig(name="g", provider="google", model="gemini-pro", api_key="k")
        engine = create_engine(cfg)
        assert isinstance(engine, GeminiEngine)

    def test_create_openai(self):
        cfg = EngineConfig(name="o", provider="openai", model="gpt-4", api_key="k")
        engine = create_engine(cfg)
        assert isinstance(engine, OpenAIEngine)

    def test_create_openai_compatible(self):
        cfg = EngineConfig(name="d", provider="openai_compatible", model="deepseek", api_key="k")
        engine = create_engine(cfg)
        assert isinstance(engine, OpenAIEngine)

    def test_create_all_skips_unavailable(self):
        wcfg = WriterConfig(engines={
            "available": EngineConfig(name="a", provider="openai", model="m", api_key="k"),
            "unavailable": EngineConfig(name="u", provider="openai", model="m"),
        })
        engines = create_all_engines(wcfg)
        assert "available" in engines
        assert "unavailable" not in engines


class TestEngineResponse:
    def test_success(self):
        r = EngineResponse(text="hello", engine_name="test", model="m")
        assert r.success is True

    def test_empty_is_not_success(self):
        r = EngineResponse(text="", engine_name="test", model="m")
        assert r.success is False
