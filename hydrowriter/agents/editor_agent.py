"""Editor agent that rewrites content from review synthesis."""

from __future__ import annotations

from typing import Any

from hydrowriter.config import WriterConfig
from hydrowriter.engine.base_engine import BaseEngine
from hydrowriter.merge.consensus import extract_consensus, extract_divergence


class EditorAgent:
    """Apply review consensus and optionally address divergent comments."""

    def __init__(self, engines: dict[str, BaseEngine], config: WriterConfig):
        self.engines = engines
        self.config = config

    async def edit(self, content: str, review_result: dict[str, Any], original_context: str) -> str:
        """Revise content by enforcing consensus issues and considering divergent issues."""
        if not self.engines:
            raise ValueError("EditorAgent requires at least one engine.")

        normalized_review = self._normalize_review_result(review_result)
        engine = self.engines.get(self.config.default_engine) or next(iter(self.engines.values()))

        consensus_items = normalized_review.get("consensus", [])
        divergence_items = normalized_review.get("divergence", [])
        prompt = self._build_prompt(
            content=content,
            original_context=original_context,
            consensus_items=consensus_items,
            divergence_items=divergence_items,
        )
        response = await engine.generate(prompt, system="You are a strict editor. Apply must-fix items without fail.")
        return response.text if response.success else content

    def _normalize_review_result(self, review_result: dict[str, Any]) -> dict[str, Any]:
        if "consensus" in review_result or "divergence" in review_result:
            return review_result

        responses = [{"role": role, **payload} for role, payload in review_result.items()]
        return {
            "reviews": review_result,
            "consensus": extract_consensus(responses, threshold=self.config.consensus_threshold),
            "divergence": extract_divergence(responses),
        }

    def _build_prompt(
        self,
        content: str,
        original_context: str,
        consensus_items: list[dict[str, Any]],
        divergence_items: list[dict[str, Any]],
    ) -> str:
        must_fix = "\n".join(
            f"- {item['issue']} (mentioned by {item['count']} reviewers)"
            for item in consensus_items
        ) or "- No consensus issue was detected. Preserve the draft unless the context requires cleanup."
        optional_fix = "\n".join(f"- {item['issue']}" for item in divergence_items) or "- No divergent issue."
        return (
            f"Original Context:\n{original_context}\n\n"
            f"Current Content:\n{content}\n\n"
            "Must Fix (consensus, mandatory):\n"
            f"{must_fix}\n\n"
            "Optional Fix (divergence, apply only if it materially improves the draft):\n"
            f"{optional_fix}\n\n"
            "Return only the revised content."
        )
