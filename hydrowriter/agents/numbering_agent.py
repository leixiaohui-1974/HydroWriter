"""Markdown numbering helpers."""

from __future__ import annotations

import re


class NumberingAgent:
    """Update chapter, figure, equation, and reference numbering in Markdown."""

    def update_chapter_numbers(self, content: str, chapter_num: int) -> str:
        """Update chapter headings and chapter references."""
        updated = re.sub(
            r"(?mi)^(#{1,6}\s*)Chapter\s+\d+\b",
            lambda m: f"{m.group(1)}Chapter {chapter_num}",
            content,
        )
        updated = re.sub(
            r"(?mi)^(#{1,6}\s*)第\s*\d+\s*章",
            lambda m: f"{m.group(1)}第{chapter_num}章",
            updated,
        )
        updated = re.sub(r"(?i)\bChapter\s+\d+\b", f"Chapter {chapter_num}", updated)
        updated = re.sub(r"第\s*\d+\s*章", f"第{chapter_num}章", updated)
        return updated

    def update_figure_numbers(self, content: str, chapter_num: int) -> str:
        """Renumber Markdown figure captions and matching inline references."""
        pattern = re.compile(
            r"(?mi)^(?P<indent>\s*)(?P<label>Figure|Fig\.|图)\s*(?P<number>\d+(?:\.\d+)?)"
            r"(?P<suffix>\s*[:：].*)$"
        )
        mapping: dict[tuple[str, str], str] = {}
        counter = 0

        def replace_caption(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            label = match.group("label")
            old_number = match.group("number")
            new_number = f"{chapter_num}.{counter}"
            mapping[(label, old_number)] = new_number
            if label == "图":
                return f"{match.group('indent')}图{new_number}{match.group('suffix')}"
            return f"{match.group('indent')}{label} {new_number}{match.group('suffix')}"

        updated = pattern.sub(replace_caption, content)
        for (label, old_number), new_number in mapping.items():
            if label == "图":
                updated = re.sub(rf"图\s*{re.escape(old_number)}\b", f"图{new_number}", updated)
            else:
                updated = re.sub(
                    rf"\b{re.escape(label)}\s*{re.escape(old_number)}\b",
                    f"{label} {new_number}",
                    updated,
                )
        return updated

    def update_equation_numbers(self, content: str, chapter_num: int) -> str:
        """Renumber display-equation labels and matching inline references."""
        pattern = re.compile(
            r"(?s)(?P<body>\$\$.*?\$\$\s*)(?P<open>[（(])(?P<number>\d+(?:\.\d+)?)(?P<close>[）)])"
        )
        mapping: dict[str, str] = {}
        counter = 0

        def replace_equation(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            old_number = match.group("number")
            new_number = f"{chapter_num}.{counter}"
            mapping[old_number] = new_number
            return f"{match.group('body')}{match.group('open')}{new_number}{match.group('close')}"

        updated = pattern.sub(replace_equation, content)
        for old_number, new_number in mapping.items():
            updated = re.sub(rf"\bEquation\s*{re.escape(old_number)}\b", f"Equation {new_number}", updated)
            updated = re.sub(rf"\bEq\.\s*{re.escape(old_number)}\b", f"Eq. {new_number}", updated)
            updated = re.sub(rf"式\s*[（(]?{re.escape(old_number)}[）)]?", f"式({new_number})", updated)
        return updated

    def update_references(self, content: str, ref_list: list[str]) -> str:
        """Rewrite the Markdown references section with normalized numbering."""
        rendered_refs = "\n".join(f"[{index}] {ref}" for index, ref in enumerate(ref_list, start=1))
        replacement = f"## References\n\n{rendered_refs}".strip()
        pattern = re.compile(r"(?ms)^##\s*(References|参考文献)\s*$.*?(?=^##\s|\Z)")
        if pattern.search(content):
            return pattern.sub(f"{replacement}\n\n", content).rstrip()

        separator = "\n\n" if content.strip() else ""
        return f"{content.rstrip()}{separator}{replacement}".strip()
