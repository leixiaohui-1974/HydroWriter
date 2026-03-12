from __future__ import annotations

import asyncio
import time

import pytest

from hydrowriter.agents import EditorAgent, NumberingAgent, ReviewerAgent, WriterAgent
from hydrowriter.config import EngineConfig, WriterConfig
from hydrowriter.engine.base_engine import BaseEngine, EngineResponse
from hydrowriter.merge import calc_quality_score, extract_consensus, extract_divergence


class MockEngine(BaseEngine):
    def __init__(
        self,
        name: str,
        *,
        generate_text: str | None = None,
        review_text: str | None = None,
        delay: float = 0.0,
        fail_generate: bool = False,
        fail_review: bool = False,
        metadata: dict | None = None,
    ):
        super().__init__(name=name, model=f"{name}-model")
        self.generate_text = generate_text or f"{name} draft"
        self.review_text = review_text or "Score: 8\n- tighten structure"
        self.delay = delay
        self.fail_generate = fail_generate
        self.fail_review = fail_review
        self.metadata = metadata or {}
        self.prompts: list[dict] = []
        self.review_calls: list[dict] = []

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> EngineResponse:
        self.prompts.append(
            {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_generate:
            raise RuntimeError(f"{self.name} generate failed")
        return EngineResponse(
            text=self.generate_text,
            engine_name=self.name,
            model=self.model,
            metadata=dict(self.metadata),
        )

    async def review(
        self,
        content: str,
        role: str = "reviewer",
        criteria: list[str] | None = None,
    ) -> EngineResponse:
        self.review_calls.append({"content": content, "role": role, "criteria": criteria})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_review:
            raise RuntimeError(f"{self.name} review failed")
        return EngineResponse(text=self.review_text, engine_name=self.name, model=self.model)


def make_config(*engine_roles: tuple[str, str], default_engine: str = "writer") -> WriterConfig:
    return WriterConfig(
        engines={name: EngineConfig(name=name, provider="mock", model=name, role=role) for name, role in engine_roles},
        default_engine=default_engine,
    )


@pytest.mark.asyncio
async def test_writer_agent_draft_runs_engines_in_parallel():
    config = make_config(("writer", "general"), ("alt", "general"), default_engine="writer")
    engines = {
        "writer": MockEngine("writer", generate_text="Intro\n\nShared", delay=0.05),
        "alt": MockEngine("alt", generate_text="Shared\n\nUnique", delay=0.05),
    }
    agent = WriterAgent(engines, config)

    start = time.perf_counter()
    result = await agent.draft("Hydropower", "Reservoir operations", "Formal")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.09
    assert set(result["drafts"]) == {"writer", "alt"}
    assert "Unique" in result["merged"]


@pytest.mark.asyncio
async def test_writer_agent_merge_drafts_prefers_high_quality_base():
    config = make_config(("writer", "general"))
    agent = WriterAgent({}, config)
    drafts = [
        EngineResponse(
            text="Short draft",
            engine_name="a",
            model="a",
            metadata={"quality_score": 6},
        ),
        EngineResponse(
            text="Best opening\n\nUnique evidence",
            engine_name="b",
            model="b",
            metadata={"quality_score": 9},
        ),
        EngineResponse(
            text="Best opening\n\nDifferent close",
            engine_name="c",
            model="c",
            metadata={"quality_score": 7},
        ),
    ]

    merged = await agent.merge_drafts(drafts)

    assert merged.startswith("Best opening")
    assert "Unique evidence" in merged
    assert "Different close" in merged


@pytest.mark.asyncio
async def test_writer_agent_draft_collects_failures():
    config = make_config(("writer", "general"), ("broken", "general"), default_engine="writer")
    engines = {
        "writer": MockEngine("writer", generate_text="Primary"),
        "broken": MockEngine("broken", fail_generate=True),
    }
    agent = WriterAgent(engines, config)

    result = await agent.draft("Topic", "Context", "Style")

    assert result["merged"] == "Primary"
    assert "broken" in result["failures"]


@pytest.mark.asyncio
async def test_reviewer_agent_uses_role_mapped_engines():
    config = make_config(
        ("teacher_engine", "teacher"),
        ("engineer_engine", "engineer"),
        ("reader_engine", "reader"),
    )
    engines = {
        "teacher_engine": MockEngine("teacher_engine"),
        "engineer_engine": MockEngine("engineer_engine"),
        "reader_engine": MockEngine("reader_engine"),
    }
    agent = ReviewerAgent(engines, config)

    result = await agent.review("Draft content")

    assert result["teacher"]["engine"] == "teacher_engine"
    assert result["engineer"]["engine"] == "engineer_engine"
    assert result["reader"]["engine"] == "reader_engine"


@pytest.mark.asyncio
async def test_reviewer_agent_parses_score_and_suggestions():
    config = make_config(("reviewer", "teacher"))
    engine = MockEngine(
        "reviewer",
        review_text="Score: 9\n- clarify the thesis\n- add one example",
    )
    agent = ReviewerAgent({"reviewer": engine}, config)

    result = await agent.review("Draft", roles=["teacher"])

    assert result["teacher"]["score"] == 9
    assert result["teacher"]["suggestions"] == ["clarify the thesis", "add one example"]


@pytest.mark.asyncio
async def test_reviewer_agent_merge_reviews_returns_consensus_and_divergence():
    config = make_config(("reviewer", "teacher"))
    agent = ReviewerAgent({}, config)
    reviews = {
        "teacher": {"score": 8, "suggestions": ["clarify thesis", "fix transitions"], "engine": "a"},
        "engineer": {"score": 7, "suggestions": ["clarify thesis"], "engine": "b"},
        "reader": {"score": 9, "suggestions": ["add a case study"], "engine": "c"},
    }

    merged = await agent.merge_reviews(reviews)

    assert merged["average_score"] == 8.0
    assert merged["consensus"][0]["issue"] == "clarify thesis"
    assert any(item["issue"] == "add a case study" for item in merged["divergence"])


@pytest.mark.asyncio
async def test_editor_agent_builds_prompt_with_consensus_and_divergence():
    config = make_config(("writer", "general"), default_engine="writer")
    engine = MockEngine("writer", generate_text="Revised content")
    agent = EditorAgent({"writer": engine}, config)
    review_result = {
        "consensus": [{"issue": "clarify thesis", "count": 2}],
        "divergence": [{"issue": "add more technical detail", "count": 1}],
    }

    result = await agent.edit("Original", review_result, "Source notes")

    prompt = engine.prompts[-1]["prompt"]
    assert result == "Revised content"
    assert "clarify thesis" in prompt
    assert "add more technical detail" in prompt
    assert "Must Fix" in prompt


@pytest.mark.asyncio
async def test_editor_agent_normalizes_raw_review_mapping():
    config = make_config(("writer", "general"), default_engine="writer")
    engine = MockEngine("writer", generate_text="Edited")
    agent = EditorAgent({"writer": engine}, config)

    await agent.edit(
        "Original",
        {
            "teacher": {"score": 8, "suggestions": ["tighten intro"], "engine": "a"},
            "reader": {"score": 8, "suggestions": ["tighten intro"], "engine": "b"},
        },
        "Context",
    )

    prompt = engine.prompts[-1]["prompt"]
    assert "tighten intro" in prompt
    assert "mentioned by 2 reviewers" in prompt


def test_numbering_agent_updates_chapter_numbers():
    agent = NumberingAgent()
    content = "# Chapter 2\nSee Chapter 2 for methods.\n## 第2章 背景"

    updated = agent.update_chapter_numbers(content, 5)

    assert "# Chapter 5" in updated
    assert "See Chapter 5" in updated
    assert "## 第5章 背景" in updated


def test_numbering_agent_updates_figure_numbers_and_references():
    agent = NumberingAgent()
    content = (
        "Figure 1.1: System layout\n"
        "As shown in Figure 1.1, the pump starts first.\n"
        "Fig. 1.2: Backup pipeline\n"
    )

    updated = agent.update_figure_numbers(content, 3)

    assert "Figure 3.1: System layout" in updated
    assert "Figure 3.1, the pump starts first." in updated
    assert "Fig. 3.2: Backup pipeline" in updated


def test_numbering_agent_updates_equation_numbers_and_references():
    agent = NumberingAgent()
    content = (
        "$$\nE = mc^2\n$$ (1.1)\n\n"
        "Equation 1.1 shows the core relation.\n"
        "$$\nF = ma\n$$ (1.2)\n"
    )

    updated = agent.update_equation_numbers(content, 4)

    assert "$$ (4.1)" in updated
    assert "Equation 4.1 shows the core relation." in updated
    assert "$$ (4.2)" in updated


def test_numbering_agent_updates_references_section():
    agent = NumberingAgent()
    content = "## References\n\n[9] Old source\n\n## Appendix\n\nText"

    updated = agent.update_references(content, ["Alpha 2024", "Beta 2025"])

    assert "[1] Alpha 2024" in updated
    assert "[2] Beta 2025" in updated
    assert "Old source" not in updated
    assert "## Appendix" in updated


def test_consensus_helpers_group_feedback_and_average_scores():
    responses = [
        {"role": "teacher", "engine": "a", "score": 8, "suggestions": ["clarify thesis", "fix transitions"]},
        {"role": "engineer", "engine": "b", "score": 7, "suggestions": ["clarify thesis"]},
        {"role": "reader", "engine": "c", "score": 9, "suggestions": ["add anecdote"]},
    ]

    consensus = extract_consensus(responses, threshold=2)
    divergence = extract_divergence(responses)
    quality = calc_quality_score(responses)

    assert consensus == [
        {
            "issue": "clarify thesis",
            "count": 2,
            "roles": ["teacher", "engineer"],
            "engines": ["a", "b"],
        }
    ]
    assert any(item["issue"] == "add anecdote" for item in divergence)
    assert quality == 8.0
