"""Tests for writing pipelines."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from hydrowriter.config import EngineConfig, WriterConfig
from hydrowriter.engine.base_engine import EngineResponse
from hydrowriter.pipeline.book_pipeline import BookPipeline, PipelineResult
from hydrowriter.pipeline.ppt_pipeline import PPTPipeline, PPTResult


# --- Fixtures ---

def _make_engine_response(text: str, success: bool = True, engine_name: str = "mock") -> EngineResponse:
    return EngineResponse(
        text=text,
        engine_name=engine_name,
        success=success,
        tokens=100,
        latency=0.5,
        metadata={},
    )


def _make_mock_engine(name: str = "mock", text: str = "Generated content.") -> MagicMock:
    engine = MagicMock()
    engine.name = name

    async def mock_generate(prompt, system="", **kwargs):
        return _make_engine_response(text, engine_name=name)

    async def mock_review(content, role="", **kwargs):
        return _make_engine_response(
            f"Score: 8.5/10\n- Good structure\n- Need more examples\n- Fix references",
            engine_name=name,
        )

    engine.generate = AsyncMock(side_effect=mock_generate)
    engine.review = AsyncMock(side_effect=mock_review)
    return engine


def _make_config() -> WriterConfig:
    return WriterConfig(
        engines={
            "mock": EngineConfig(name="mock", provider="openai", model="gpt-4"),
        },
        default_engine="mock",
        review_roles=["teacher", "engineer"],
        consensus_threshold=2,
    )


# --- BookPipeline Tests ---

class TestBookPipeline:
    @pytest.mark.asyncio(loop_scope="function")
    async def test_pipeline_result_properties(self):
        result = PipelineResult(topic="test")
        assert not result.success
        result.final = "Some content"
        assert result.success

    @pytest.mark.asyncio(loop_scope="function")
    async def test_draft_only(self):
        engine = _make_mock_engine()
        config = _make_config()
        pipeline = BookPipeline({"mock": engine}, config)
        draft = await pipeline.draft_only("Water hydraulics")
        assert isinstance(draft, str)

    @pytest.mark.asyncio(loop_scope="function")
    async def test_review_only(self):
        engine = _make_mock_engine()
        config = _make_config()
        pipeline = BookPipeline({"mock": engine}, config)
        result = await pipeline.review_only("Some chapter content")
        assert "consensus" in result or "reviews" in result


# --- PPTPipeline Tests ---

class TestPPTPipeline:
    def test_parse_markdown_to_sections(self):
        pipeline = PPTPipeline()
        sections = pipeline._parse_markdown_to_sections(
            "# Title\nIntro\n## Section 1\n- Bullet A\n- Bullet B\n## Section 2\n- Bullet C"
        )
        assert len(sections) == 3
        assert sections[0]["title"] == "Title"
        assert sections[1]["title"] == "Section 1"
        assert "Bullet A" in sections[1]["bullets"]

    def test_parse_empty_markdown(self):
        pipeline = PPTPipeline()
        sections = pipeline._parse_markdown_to_sections("")
        assert len(sections) == 1
        assert sections[0]["title"] == "Untitled"

    @pytest.mark.asyncio(loop_scope="function")
    async def test_generate_from_outline_no_pptx(self):
        pipeline = PPTPipeline()
        pipeline._pptx = None
        result = await pipeline.generate_from_outline("Test", [])
        assert not result.success
        assert "python-pptx" in result.error

    @pytest.mark.asyncio(loop_scope="function")
    async def test_generate_with_llm_no_engines(self):
        pipeline = PPTPipeline()
        result = await pipeline.generate_with_llm("Test topic")
        assert not result.success
        assert "No engines" in result.error

    @pytest.mark.asyncio(loop_scope="function")
    async def test_ppt_result_defaults(self):
        result = PPTResult()
        assert not result.success
        assert result.slide_count == 0


# --- MCP Server Tests ---

class TestHydroWriterMCPServer:
    def test_get_tools(self):
        from hydrowriter.mcp_server import HydroWriterMCPServer
        server = HydroWriterMCPServer()
        tools = server.get_tools()
        assert len(tools) == 4
        tool_names = {t["name"] for t in tools}
        assert "write_chapter" in tool_names
        assert "review_content" in tool_names
        assert "generate_ppt" in tool_names
        assert "get_engine_status" in tool_names

    @pytest.mark.asyncio(loop_scope="function")
    async def test_generate_ppt_no_input(self):
        from hydrowriter.mcp_server import HydroWriterMCPServer
        server = HydroWriterMCPServer()
        server._config = WriterConfig()
        server._engines = {}
        result = await server.generate_ppt()
        assert not result["success"]
        assert "Need topic" in result["error"]

    @pytest.mark.asyncio(loop_scope="function")
    async def test_get_engine_status_empty(self):
        from hydrowriter.mcp_server import HydroWriterMCPServer
        server = HydroWriterMCPServer()
        server._config = WriterConfig()
        server._engines = {}
        result = await server.get_engine_status()
        assert result["success"]
        assert result["results"]["engine_count"] == 0


# --- CLI Tests ---

class TestCLI:
    def test_cli_group_exists(self):
        from hydrowriter.cli.main import cli
        assert cli.name == "cli"

    def test_cli_commands(self):
        from hydrowriter.cli.main import cli
        command_names = set(cli.commands.keys())
        assert "write" in command_names
        assert "review" in command_names
        assert "ppt" in command_names
        assert "info" in command_names
