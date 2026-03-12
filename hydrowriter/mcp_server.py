"""HydroWriter MCP Server — exposes writing capabilities as MCP tools.

Port: 8033 (for HydroClaw integration)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from hydrowriter.config import WriterConfig, load_config
from hydrowriter.engine.factory import create_all_engines


class HydroWriterMCPServer:
    """MCP-compatible server wrapping HydroWriter pipelines."""

    PORT = 8033

    def __init__(self, config_path: str = "configs/engines.yaml"):
        self.config_path = config_path
        self._config: WriterConfig | None = None
        self._engines: dict[str, Any] | None = None

    def _ensure_loaded(self) -> tuple[dict[str, Any], WriterConfig]:
        if self._config is None:
            self._config = load_config(self.config_path)
            self._engines = create_all_engines(self._config)
        return self._engines, self._config

    async def write_chapter(
        self,
        topic: str,
        context: str = "",
        style_guide: str = "",
        chapter_num: int = 1,
    ) -> Dict[str, Any]:
        """Multi-engine collaborative chapter writing."""
        from hydrowriter.pipeline.book_pipeline import BookPipeline

        engines, config = self._ensure_loaded()
        if not engines:
            return {"success": False, "method": "write_chapter", "error": "No engines available"}

        pipeline = BookPipeline(engines, config)
        result = await pipeline.run(
            topic=topic, context=context, style_guide=style_guide, chapter_num=chapter_num,
        )
        return {
            "success": result.success,
            "method": "write_chapter",
            "results": {
                "content": result.final,
                "draft_length": len(result.draft),
                "final_length": len(result.final),
                "average_score": result.metadata.get("average_score", 0),
                "engines_used": result.metadata.get("draft_engines", []),
            },
        }

    async def review_content(
        self,
        content: str,
        roles: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Multi-engine multi-role content review."""
        from hydrowriter.pipeline.book_pipeline import BookPipeline

        engines, config = self._ensure_loaded()
        if not engines:
            return {"success": False, "method": "review_content", "error": "No engines available"}

        pipeline = BookPipeline(engines, config)
        result = await pipeline.review_only(content, roles=roles)
        return {
            "success": True,
            "method": "review_content",
            "results": {
                "average_score": result.get("average_score", 0),
                "consensus": result.get("consensus", []),
                "divergence": result.get("divergence", []),
            },
        }

    async def generate_ppt(
        self,
        topic: str = "",
        markdown_content: str = "",
        slide_count: int = 10,
        output_path: str = "output.pptx",
    ) -> Dict[str, Any]:
        """Generate PPT from topic or markdown."""
        from hydrowriter.pipeline.ppt_pipeline import PPTPipeline

        engines, config = self._ensure_loaded()
        pipeline = PPTPipeline(engines=engines, config=config)

        if markdown_content:
            result = await pipeline.generate_from_markdown(markdown_content, output_path)
        elif topic:
            result = await pipeline.generate_with_llm(topic, slide_count, output_path)
        else:
            return {"success": False, "method": "generate_ppt", "error": "Need topic or markdown_content"}

        return {
            "success": result.success,
            "method": "generate_ppt",
            "results": {
                "output_path": result.output_path,
                "slide_count": result.slide_count,
                "error": result.error,
            },
        }

    async def get_engine_status(self) -> Dict[str, Any]:
        """Return status of all configured engines."""
        engines, config = self._ensure_loaded()
        engine_info = {}
        for name, ec in config.engines.items():
            engine_info[name] = {
                "provider": ec.provider,
                "model": ec.model,
                "role": ec.role,
                "available": ec.is_available,
                "loaded": name in engines,
            }
        return {
            "success": True,
            "method": "get_engine_status",
            "results": {
                "engine_count": len(engines),
                "engines": engine_info,
                "default_engine": config.default_engine,
            },
        }

    def get_tools(self) -> list[Dict[str, Any]]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "write_chapter",
                "description": "Multi-engine collaborative chapter writing (draft→review→edit→number)",
                "parameters": {
                    "topic": {"type": "string", "required": True},
                    "context": {"type": "string", "default": ""},
                    "style_guide": {"type": "string", "default": ""},
                    "chapter_num": {"type": "integer", "default": 1},
                },
            },
            {
                "name": "review_content",
                "description": "Multi-engine multi-role content review with consensus extraction",
                "parameters": {
                    "content": {"type": "string", "required": True},
                    "roles": {"type": "array", "items": {"type": "string"}, "default": None},
                },
            },
            {
                "name": "generate_ppt",
                "description": "Generate PPT slides from topic or markdown content",
                "parameters": {
                    "topic": {"type": "string", "default": ""},
                    "markdown_content": {"type": "string", "default": ""},
                    "slide_count": {"type": "integer", "default": 10},
                    "output_path": {"type": "string", "default": "output.pptx"},
                },
            },
            {
                "name": "get_engine_status",
                "description": "Get status of all configured LLM engines",
                "parameters": {},
            },
        ]
