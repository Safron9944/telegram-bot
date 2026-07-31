from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable


def normalize_match_text(value: str) -> str:
    normalized = str(value or "").casefold()
    normalized = normalized.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    normalized = re.sub(
        r"[^0-9a-zа-яіїєґ']+",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    return " ".join(normalized.split())


@dataclass(frozen=True)
class ReferenceQuestion:
    source: str
    question: str
    choices: list[str]
    correct: list[int]


@dataclass(frozen=True)
class MatchEvidence:
    source: str
    match_type: str
    score: float
    question: str
    choices: list[str]
    correct: list[int]


def match_references(
    question: str,
    choices: list[str],
    references: Iterable[ReferenceQuestion],
) -> list[MatchEvidence]:
    normalized_question = normalize_match_text(question)
    normalized_choices = [normalize_match_text(choice) for choice in choices]
    results: list[MatchEvidence] = []

    for reference in references:
        reference_question = normalize_match_text(reference.question)
        reference_choices = [
            normalize_match_text(choice) for choice in reference.choices
        ]
        exact = (
            normalized_question == reference_question
            and normalized_choices == reference_choices
        )
        score = SequenceMatcher(
            None,
            normalized_question,
            reference_question,
        ).ratio()
        if exact or score >= 0.65:
            results.append(
                MatchEvidence(
                    source=reference.source,
                    match_type="exact" if exact else "fuzzy",
                    score=score,
                    question=reference.question,
                    choices=list(reference.choices),
                    correct=list(reference.correct),
                )
            )

    return sorted(
        results,
        key=lambda item: (item.match_type == "exact", item.score),
        reverse=True,
    )[:3]


def quality_issues(
    question: str,
    choices: list[str],
    confidences: list[float],
    crossed_page: bool,
    ended_before_next_number: bool,
) -> list[str]:
    issues: list[str] = []
    values = [str(question or ""), *(str(choice or "") for choice in choices)]

    if not values[0].strip() or any(not choice.strip() for choice in values[1:]):
        issues.append("empty_text")
    if confidences and min(confidences) < 0.70:
        issues.append("low_ocr_confidence")
    if crossed_page and not ended_before_next_number:
        issues.append("page_break_not_closed")
    if any(
        re.search(r"\b[а-яіїєґ]{1,2}$", value.strip(), re.IGNORECASE)
        for value in values
        if value.strip()
    ):
        issues.append("suspicious_word_break")

    return sorted(set(issues))


def completeness_report(
    numbers_by_section: dict[str, list[int]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for section, raw_numbers in numbers_by_section.items():
        numbers = [int(number) for number in raw_numbers if int(number) > 0]
        counts = Counter(numbers)
        maximum = max(numbers, default=0)
        result[section] = {
            "found": len(numbers),
            "missing": [
                number for number in range(1, maximum + 1) if counts[number] == 0
            ],
            "duplicates": sorted(
                number for number, count in counts.items() if count > 1
            ),
        }
    return result


def accounting_report(
    verified: list[dict],
    reviews: list[dict],
) -> dict[str, object]:
    numbers_by_section: dict[str, list[int]] = {}
    for item in [*verified, *reviews]:
        section = str(item.get("section") or "")
        qnum = int(item.get("qnum") or 0)
        if section and qnum > 0:
            numbers_by_section.setdefault(section, []).append(qnum)

    sections = completeness_report(numbers_by_section)
    unaccounted_gaps = {
        section: list(values["missing"])
        for section, values in sections.items()
        if values["missing"]
    }
    return {
        "sections": sections,
        "verified": len(verified),
        "needs_review": sum(
            item.get("status") == "needs_review" for item in reviews
        ),
        "unaccounted_gaps": unaccounted_gaps,
    }
