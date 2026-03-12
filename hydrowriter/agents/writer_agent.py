"""Drafting agent that coordinates multiple engines."""

from __future__ import annotations

import asyncio
from typing import Any

from hydrowriter.config import WriterConfig
from hydrowriter.engine.base_engine import BaseEngine, EngineResponse


class WriterAgent:
    """Coordinate parallel drafting across multiple engines."""

    def __init__(self, engines: dict[str, BaseEngine], config: WriterConfig):
        self.engines = engines
        self.config = config

    async def draft(self, topic: str, context: str, style_guide: str) -> dict[str, Any]:
        """Generate drafts from all configured engines and merge successful responses."""
        prompt = self._build_prompt(topic=topic, context=context, style_guide=style_guide)
        engine_items = list(self.engines.items())

        if self.config.parallel_drafting:
            results = await asyncio.gather(
                *(engine.generate(prompt, system="You are a precise long-form writer.") for _, engine in engine_items),
                return_exceptions=True,
            )
        else:
            results = []
            for _, engine in engine_items:
                try:
                    results.append(await engine.generate(prompt, system="You are a precise long-form writer."))
                except Exception as exc:  # pragma: no cover - same handling as gather path
                    results.append(exc)

        drafts: dict[str, EngineResponse] = {}
        failures: dict[str, str] = {}
        for (engine_name, _), result in zip(engine_items, results):
            if isinstance(result, Exception):
                failures[engine_name] = str(result)
                continue
            if result.success:
                drafts[engine_name] = result
            else:
                failures[engine_name] = "empty response"

        merged = await self.merge_drafts(list(drafts.values()))
        return {
            "topic": topic,
            "drafts": drafts,
            "failures": failures,
            "merged": merged,
        }

    async def merge_drafts(self, drafts: list[EngineResponse]) -> str:
        """Fuse the best draft with unique high-signal paragraphs from other drafts."""
        if not drafts:
            return ""
        if len(drafts) == 1:
            return drafts[0].text

        base = max(drafts, key=self._draft_score)
        merged_parts: list[str] = []
        seen: set[str] = set()

        for draft in [base, *[item for item in drafts if item is not base]]:
            for paragraph in self._split_paragraphs(draft.text):
                normalized = self._normalize_paragraph(paragraph)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged_parts.append(paragraph.strip())

        return "\n\n".join(merged_parts).strip()

    def _build_prompt(self, topic: str, context: str, style_guide: str) -> str:
        return (
            f"Topic:\n{topic}\n\n"
            f"Context:\n{context}\n\n"
            f"Style Guide:\n{style_guide}\n\n"
            "Write a polished draft. Keep the structure clear, arguments specific, and examples concrete."
        )

    def _draft_score(self, response: EngineResponse) -> tuple[float, int]:
        quality = response.metadata.get("quality_score", 0)
        try:
            quality_value = float(quality)
        except (TypeError, ValueError):
            quality_value = 0.0
        return quality_value, len(response.text.strip())

    def _split_paragraphs(self, text: str) -> list[str]:
        return [part for part in text.split("\n\n") if part.strip()]

    def _normalize_paragraph(self, paragraph: str) -> str:
        return " ".join(paragraph.split()).casefold()
