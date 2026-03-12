"""Review agent that routes roles across multiple engines."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from hydrowriter.config import WriterConfig
from hydrowriter.engine.base_engine import BaseEngine
from hydrowriter.merge.consensus import calc_quality_score, extract_consensus, extract_divergence


class ReviewerAgent:
    """Collect structured review feedback from multiple reviewer roles."""

    def __init__(self, engines: dict[str, BaseEngine], config: WriterConfig):
        self.engines = engines
        self.config = config

    async def review(self, content: str, roles: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Request role-based reviews and normalize the results."""
        review_roles = roles or list(self.config.review_roles)
        assignments = self._assign_engines(review_roles)

        results = await asyncio.gather(
            *(engine.review(content, role=role) for role, (_, engine) in assignments.items()),
            return_exceptions=True,
        )

        normalized: dict[str, dict[str, Any]] = {}
        for role, result in zip(review_roles, results):
            engine_name, _ = assignments[role]
            if isinstance(result, Exception):
                normalized[role] = {
                    "score": 0.0,
                    "suggestions": [str(result)],
                    "engine": engine_name,
                    "raw": "",
                }
                continue
            normalized[role] = {
                "score": self._extract_score(result.text),
                "suggestions": self._extract_suggestions(result.text),
                "engine": result.engine_name,
                "raw": result.text,
                "role": role,
            }
        return normalized

    async def merge_reviews(self, reviews: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Extract consensus issues and divergent opinions from review results."""
        responses = [{"role": role, **payload} for role, payload in reviews.items()]
        return {
            "reviews": reviews,
            "average_score": calc_quality_score(responses),
            "consensus": extract_consensus(responses, threshold=self.config.consensus_threshold),
            "divergence": extract_divergence(responses),
        }

    def _assign_engines(self, roles: list[str]) -> dict[str, tuple[str, BaseEngine]]:
        if not self.engines:
            raise ValueError("ReviewerAgent requires at least one engine.")

        assignments: dict[str, tuple[str, BaseEngine]] = {}
        used: set[str] = set()
        engine_names = list(self.engines)

        for index, role in enumerate(roles):
            preferred = [
                name
                for name, engine_config in self.config.engines.items()
                if engine_config.role == role and name in self.engines and name not in used
            ]
            if preferred:
                engine_name = preferred[0]
            else:
                available = [name for name in engine_names if name not in used]
                engine_name = available[0] if available else engine_names[index % len(engine_names)]
            used.add(engine_name)
            assignments[role] = (engine_name, self.engines[engine_name])
        return assignments

    def _extract_score(self, text: str) -> float:
        patterns = [
            r"(?:score|rating|评分|得分)\s*[:：]?\s*(\d+(?:\.\d+)?)",
            r"\b(\d+(?:\.\d+)?)\s*/\s*10\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    def _extract_suggestions(self, text: str) -> list[str]:
        suggestions: list[str] = []
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if re.search(r"(?:score|rating|评分|得分)\s*[:：]?\s*\d", cleaned, re.IGNORECASE):
                continue
            cleaned = re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", cleaned)
            cleaned = cleaned.strip("：: ")
            if cleaned:
                suggestions.append(cleaned)

        if suggestions:
            return suggestions

        fragments = re.split(r"[。.!?]\s*", text)
        return [fragment.strip() for fragment in fragments if fragment.strip()]
