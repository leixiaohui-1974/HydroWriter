"""Consensus extraction utilities for review outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def extract_consensus(responses: list[dict[str, Any]], threshold: int = 2) -> list[dict[str, Any]]:
    """Return issues raised by at least ``threshold`` responses."""
    grouped = _group_suggestions(responses)
    consensus = [item for item in grouped.values() if item["count"] >= threshold]
    return sorted(consensus, key=lambda item: (-item["count"], item["issue"]))


def extract_divergence(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return issues raised by exactly one response."""
    grouped = _group_suggestions(responses)
    divergence = []
    for item in grouped.values():
        if item["count"] == 1:
            divergence.append(
                {
                    "issue": item["issue"],
                    "count": item["count"],
                    "roles": item["roles"],
                    "engines": item["engines"],
                }
            )
    return sorted(divergence, key=lambda item: item["issue"])


def calc_quality_score(responses: list[dict[str, Any]]) -> float:
    """Average the numeric scores across responses."""
    scores: list[float] = []
    for response in responses:
        score = response.get("score")
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 2)


def _group_suggestions(responses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"issue": "", "count": 0, "roles": [], "engines": []}
    )

    for response in responses:
        seen: set[str] = set()
        for suggestion in _extract_suggestions(response):
            normalized = _normalize_suggestion(suggestion)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            item = grouped[normalized]
            item["issue"] = item["issue"] or suggestion.strip()
            item["count"] += 1
            role = response.get("role")
            engine = response.get("engine") or response.get("engine_name")
            if role and role not in item["roles"]:
                item["roles"].append(role)
            if engine and engine not in item["engines"]:
                item["engines"].append(engine)

    return grouped


def _extract_suggestions(response: dict[str, Any]) -> list[str]:
    suggestions = response.get("suggestions", [])
    if isinstance(suggestions, str):
        return [line.strip(" -*") for line in suggestions.splitlines() if line.strip()]
    if isinstance(suggestions, list):
        return [str(item).strip() for item in suggestions if str(item).strip()]
    return []


def _normalize_suggestion(suggestion: str) -> str:
    cleaned = " ".join(suggestion.split()).casefold()
    return cleaned.strip(" .,:;!?")
