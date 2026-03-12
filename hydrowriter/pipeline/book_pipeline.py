"""Book / chapter writing pipeline — draft → review → edit → number."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydrowriter.agents.editor_agent import EditorAgent
from hydrowriter.agents.numbering_agent import NumberingAgent
from hydrowriter.agents.reviewer_agent import ReviewerAgent
from hydrowriter.agents.writer_agent import WriterAgent
from hydrowriter.config import WriterConfig
from hydrowriter.engine.base_engine import BaseEngine


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    topic: str
    draft: str = ""
    reviews: dict[str, Any] = field(default_factory=dict)
    merged_review: dict[str, Any] = field(default_factory=dict)
    edited: str = ""
    final: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.final.strip())


class BookPipeline:
    """End-to-end book chapter pipeline: draft → review → edit → number."""

    def __init__(self, engines: dict[str, BaseEngine], config: WriterConfig):
        self.engines = engines
        self.config = config
        self.writer = WriterAgent(engines, config)
        self.reviewer = ReviewerAgent(engines, config)
        self.editor = EditorAgent(engines, config)
        self.numbering = NumberingAgent()

    async def run(
        self,
        topic: str,
        context: str = "",
        style_guide: str = "",
        chapter_num: int = 1,
        ref_list: list[str] | None = None,
    ) -> PipelineResult:
        """Execute full pipeline."""
        result = PipelineResult(topic=topic)

        # Step 1: Multi-engine parallel drafting
        draft_result = await self.writer.draft(
            topic=topic, context=context, style_guide=style_guide,
        )
        result.draft = draft_result.get("merged", "")
        result.metadata["draft_engines"] = list(draft_result.get("drafts", {}).keys())
        result.metadata["draft_failures"] = draft_result.get("failures", {})

        if not result.draft.strip():
            return result

        # Step 2: Multi-role review
        reviews = await self.reviewer.review(result.draft)
        result.reviews = reviews
        result.merged_review = await self.reviewer.merge_reviews(reviews)
        result.metadata["average_score"] = result.merged_review.get("average_score", 0)

        # Step 3: Edit based on consensus
        result.edited = await self.editor.edit(
            content=result.draft,
            review_result=result.merged_review,
            original_context=context,
        )

        # Step 4: Numbering
        numbered = self.numbering.update_chapter_numbers(result.edited, chapter_num)
        numbered = self.numbering.update_figure_numbers(numbered, chapter_num)
        numbered = self.numbering.update_equation_numbers(numbered, chapter_num)
        if ref_list:
            numbered = self.numbering.update_references(numbered, ref_list)
        result.final = numbered

        return result

    async def draft_only(self, topic: str, context: str = "", style_guide: str = "") -> str:
        """Run only the drafting step."""
        draft_result = await self.writer.draft(topic=topic, context=context, style_guide=style_guide)
        return draft_result.get("merged", "")

    async def review_only(self, content: str, roles: list[str] | None = None) -> dict[str, Any]:
        """Run only the review step."""
        reviews = await self.reviewer.review(content, roles=roles)
        return await self.reviewer.merge_reviews(reviews)
