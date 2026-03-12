from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hydrowriter.config import EngineConfig, WriterConfig
from hydrowriter.engine.claude_engine import ClaudeEngine
from hydrowriter.engine.factory import create_all_engines, create_engine
from hydrowriter.engine.gemini_engine import GeminiEngine
from hydrowriter.engine.openai_engine import OpenAIEngine


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTPXClient:
    def __init__(self, *, payload, capture):
        self.payload = payload
        self.capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.capture["url"] = url
        self.capture["kwargs"] = kwargs
        return FakeHTTPResponse(self.payload)


class FakeURLLibResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


@pytest.mark.asyncio
async def test_openai_generate_uses_sdk(monkeypatch):
    engine = OpenAIEngine("openai-main", "gpt-4o-mini", "sk-openai", "https://proxy.example/v1")
    capture = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            capture["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="openai sdk text"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(total_tokens=77),
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            capture["client_kwargs"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(engine, "_load_sdk", lambda: SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    response = await engine.generate("Write a draft", system="Be concise", max_tokens=321, temperature=0.2)

    assert response.text == "openai sdk text"
    assert response.tokens_used == 77
    assert response.metadata["transport"] == "sdk"
    assert capture["client_kwargs"]["base_url"] == "https://proxy.example/v1"
    assert capture["request"]["messages"][0] == {"role": "system", "content": "Be concise"}
    assert capture["request"]["messages"][1] == {"role": "user", "content": "Write a draft"}
    assert capture["request"]["max_tokens"] == 321
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_openai_generate_uses_httpx_fallback(monkeypatch):
    engine = OpenAIEngine("compatible", "deepseek-chat", "sk-deepseek", "https://proxy.example")
    capture = {}

    fake_httpx = SimpleNamespace(
        AsyncClient=lambda timeout=60.0: FakeHTTPXClient(
            payload={
                "choices": [{"message": {"content": "fallback text"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 48},
            },
            capture=capture,
        )
    )

    monkeypatch.setattr(engine, "_load_sdk", lambda: None)
    monkeypatch.setattr(engine, "_load_httpx", lambda: fake_httpx)

    response = await engine.generate("Explain", system="System hint", max_tokens=99, temperature=0.5)

    assert response.text == "fallback text"
    assert response.tokens_used == 48
    assert response.metadata["transport"] == "http"
    assert capture["url"] == "https://proxy.example/v1/chat/completions"
    assert capture["kwargs"]["headers"]["authorization"] == "Bearer sk-deepseek"
    assert capture["kwargs"]["json"]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_openai_generate_returns_error_response(monkeypatch):
    engine = OpenAIEngine("openai-main", "gpt-4o-mini", "sk-openai")

    class FakeCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("rate limited")

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(engine, "_load_sdk", lambda: SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))

    response = await engine.generate("Hello")

    assert response.text == ""
    assert not response.success
    assert response.metadata["error"] == "rate limited"
    assert response.metadata["error_type"] == "RuntimeError"
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_claude_generate_uses_sdk(monkeypatch):
    engine = ClaudeEngine("claude-arch", "claude-3-7-sonnet", "sk-anthropic", "https://claude.proxy")
    capture = {}

    class FakeMessages:
        async def create(self, **kwargs):
            capture["request"] = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="claude sdk text")],
                usage=SimpleNamespace(input_tokens=12, output_tokens=18),
                stop_reason="end_turn",
            )

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            capture["client_kwargs"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setattr(engine, "_load_sdk", lambda: SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic))

    response = await engine.generate("Plan it", system="Architect mode", max_tokens=222, temperature=0.3)

    assert response.text == "claude sdk text"
    assert response.tokens_used == 30
    assert response.metadata["transport"] == "sdk"
    assert capture["client_kwargs"]["base_url"] == "https://claude.proxy"
    assert capture["request"]["system"] == "Architect mode"
    assert capture["request"]["messages"] == [{"role": "user", "content": "Plan it"}]


@pytest.mark.asyncio
async def test_claude_generate_uses_urllib_fallback(monkeypatch):
    from hydrowriter.engine import claude_engine as claude_module

    engine = ClaudeEngine("claude-review", "claude-3-5-haiku", "sk-anthropic", "https://proxy.anthropic")
    capture = {}

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_urlopen(request, timeout=60):
        capture["url"] = request.full_url
        capture["headers"] = dict(request.header_items())
        capture["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeURLLibResponse(
            {
                "content": [{"type": "text", "text": "claude urllib text"}],
                "usage": {"input_tokens": 4, "output_tokens": 9},
                "stop_reason": "stop_sequence",
            }
        )

    monkeypatch.setattr(engine, "_load_sdk", lambda: None)
    monkeypatch.setattr(engine, "_load_httpx", lambda: None)
    monkeypatch.setattr(claude_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(claude_module.urllib.request, "urlopen", fake_urlopen)

    response = await engine.generate("Review this", system="Strict reviewer", max_tokens=111, temperature=0.1)

    assert response.text == "claude urllib text"
    assert response.tokens_used == 13
    assert response.metadata["transport"] == "http"
    assert capture["url"] == "https://proxy.anthropic/v1/messages"
    assert capture["payload"]["system"] == "Strict reviewer"
    assert capture["payload"]["messages"][0]["content"] == "Review this"


@pytest.mark.asyncio
async def test_gemini_generate_uses_sdk(monkeypatch):
    engine = GeminiEngine("gemini-main", "gemini-2.0-flash", "sk-gemini", "https://gemini.proxy")
    capture = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            capture["config_kwargs"] = kwargs

    class FakeModels:
        async def generate_content(self, **kwargs):
            capture["request"] = kwargs
            return SimpleNamespace(
                text="gemini sdk text",
                usage_metadata=SimpleNamespace(total_token_count=55),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            capture["client_kwargs"] = kwargs
            self.aio = SimpleNamespace(models=FakeModels())

    fake_genai = SimpleNamespace(
        Client=FakeClient,
        types=SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig),
    )
    monkeypatch.setattr(engine, "_load_sdk", lambda: fake_genai)

    response = await engine.generate("Summarize", system="System prompt", max_tokens=444, temperature=0.4)

    assert response.text == "gemini sdk text"
    assert response.tokens_used == 55
    assert response.metadata["transport"] == "sdk"
    assert capture["client_kwargs"]["http_options"] == {"baseUrl": "https://gemini.proxy"}
    assert capture["config_kwargs"]["maxOutputTokens"] == 444
    assert capture["config_kwargs"]["systemInstruction"] == "System prompt"
    assert capture["request"]["contents"] == "Summarize"


@pytest.mark.asyncio
async def test_gemini_generate_uses_httpx_fallback(monkeypatch):
    engine = GeminiEngine("gemini-main", "gemini-2.0-flash", "sk-gemini", "https://proxy.gemini")
    capture = {}

    fake_httpx = SimpleNamespace(
        AsyncClient=lambda timeout=60.0: FakeHTTPXClient(
            payload={
                "candidates": [{"content": {"parts": [{"text": "gemini fallback text"}]}}],
                "usageMetadata": {"totalTokenCount": 67},
            },
            capture=capture,
        )
    )

    monkeypatch.setattr(engine, "_load_sdk", lambda: None)
    monkeypatch.setattr(engine, "_load_httpx", lambda: fake_httpx)

    response = await engine.generate("Draft", system="Use bullets", max_tokens=200, temperature=0.8)

    assert response.text == "gemini fallback text"
    assert response.tokens_used == 67
    assert response.metadata["transport"] == "http"
    assert capture["url"] == "https://proxy.gemini/v1beta/models/gemini-2.0-flash:generateContent"
    assert capture["kwargs"]["params"] == {"key": "sk-gemini"}
    assert capture["kwargs"]["json"]["systemInstruction"]["parts"][0]["text"] == "Use bullets"


def test_create_engine_routes_providers():
    claude = create_engine(
        EngineConfig(name="claude", provider="anthropic", model="claude-3-7-sonnet", api_key="a")
    )
    gemini = create_engine(
        EngineConfig(name="gemini", provider="google", model="gemini-2.0-flash", api_key="b")
    )
    compatible = create_engine(
        EngineConfig(name="deepseek", provider="openai_compatible", model="deepseek-chat", api_key="c")
    )

    assert isinstance(claude, ClaudeEngine)
    assert isinstance(gemini, GeminiEngine)
    assert isinstance(compatible, OpenAIEngine)


def test_create_all_engines_builds_named_mapping():
    config = WriterConfig(
        engines={
            "claude": EngineConfig(name="claude", provider="anthropic", model="claude-3", api_key="a"),
            "gemini": EngineConfig(name="gemini", provider="google", model="gemini-2.0", api_key="b"),
            "qwen": EngineConfig(name="qwen", provider="qwen", model="qwen-max", api_key="c"),
        }
    )

    engines = create_all_engines(config)

    assert set(engines) == {"claude", "gemini", "qwen"}
    assert isinstance(engines["claude"], ClaudeEngine)
    assert isinstance(engines["gemini"], GeminiEngine)
    assert isinstance(engines["qwen"], OpenAIEngine)


def test_create_engine_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported engine provider"):
        create_engine(EngineConfig(name="bad", provider="unknown", model="none"))
