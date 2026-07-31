from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attestation import SECTION_KEYS
from attestation_quality import (
    ReferenceQuestion,
    accounting_report,
    match_references,
    normalize_match_text,
    quality_issues,
)
from questions import QuestionBank


# The first two physical PDF pages contain the cover and the table of contents.
# The ranges below are physical, one-based page numbers from the source document.
SECTION_PAGES: dict[str, range] = {
    "constitution": range(3, 31),
    "civil_service": range(31, 62),
    "customs_code": range(62, 93),
    "anti_corruption": range(93, 129),
}
EXPECTED_QUESTION_COUNTS = {section: 200 for section in SECTION_PAGES}

NUMBER_RE = re.compile(r"^\s*(\d{1,4})[.)]?\s*$")
BULLET_RE = re.compile(r"^\s*[•●▪◦·]+\s*")
HEADER_PARTS = (
    "перелік тестових питань",
    "для першого етапу атестації",
    "питання на перевірку знання",
)


def _box_bounds(line: dict[str, Any]) -> tuple[float, float, float, float]:
    points = line.get("box") or []
    xs = [float(point[0]) for point in points if len(point) >= 2]
    ys = [float(point[1]) for point in points if len(point) >= 2]
    return (
        min(xs, default=0.0),
        min(ys, default=0.0),
        max(xs, default=0.0),
        max(ys, default=0.0),
    )


def _clean_text(value: Any) -> str:
    text = str(value or "").replace("\u00ad", "")
    return " ".join(text.split()).strip()


def _strip_bullet(value: str) -> str:
    return BULLET_RE.sub("", value, count=1).strip()


def _is_header_or_page_number(text: str) -> bool:
    normalized = text.casefold().strip()
    return (
        not normalized
        or bool(re.fullmatch(r"[-–—]?\s*\d{1,3}\s*[-–—]?", normalized))
        or any(part in normalized for part in HEADER_PARTS)
    )


def _new_row(qnum: int | None, page_number: int) -> dict[str, Any]:
    return {
        "qnum": qnum,
        "question": "",
        "choices": [],
        "source_pages": [int(page_number)],
        "confidences": [],
        "choice_ink_scores": [],
        "ended_before_next_number": False,
    }


def group_page_lines(
    lines: Iterable[dict[str, Any]],
    page_number: int,
) -> list[dict[str, Any]]:
    """Convert OCR lines from one page into question-shaped rows.

    A line explicitly marked as a choice (or beginning with a bullet) starts a
    new answer. Every later non-bulleted line is appended to that answer. This
    is deliberately conservative: doubtful structure is kept as a candidate
    and will be routed to the review queue by ``classify_candidates``.
    """

    ordered = sorted(
        (dict(line) for line in lines),
        key=lambda line: (_box_bounds(line)[1], _box_bounds(line)[0]),
    )
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in ordered:
        text = _clean_text(line.get("text"))
        if not text:
            continue
        left, _top, _right, _bottom = _box_bounds(line)
        number_match = NUMBER_RE.fullmatch(text)
        # Question numbers live in the left margin. A numeric answer elsewhere
        # must not accidentally split the current question.
        if number_match and (left <= 180 or line.get("synthetic_structure")):
            if current is not None:
                current["ended_before_next_number"] = True
                rows.append(current)
            current = _new_row(int(number_match.group(1)), page_number)
            current["confidences"].append(float(line.get("confidence") or 0.0))
            continue

        is_choice = bool(line.get("is_choice")) or bool(BULLET_RE.match(text))
        if current is None:
            if _is_header_or_page_number(text):
                continue
            current = _new_row(None, page_number)

        current["confidences"].append(float(line.get("confidence") or 0.0))
        if is_choice:
            choice = _strip_bullet(text)
            current["choices"].append(choice)
            current["choice_ink_scores"].append(
                float(line.get("ink_per_character") or 0.0)
            )
        elif current["choices"]:
            if not current["choices"][-1]:
                current["choice_ink_scores"][-1] = float(
                    line.get("ink_per_character") or 0.0
                )
            current["choices"][-1] = _clean_text(
                f'{current["choices"][-1]} {text}'
            )
        else:
            current["question"] = _clean_text(f'{current["question"]} {text}')

    if current is not None:
        rows.append(current)
    return rows


def join_cross_page_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge unnumbered leading content with the prior unfinished row."""

    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["choices"] = list(raw.get("choices") or [])
        row["source_pages"] = list(raw.get("source_pages") or [])
        row["confidences"] = list(raw.get("confidences") or [])
        row["choice_ink_scores"] = list(raw.get("choice_ink_scores") or [])
        if row.get("qnum") is not None:
            result.append(row)
            continue

        previous = result[-1] if result else None
        if previous is None or previous.get("ended_before_next_number"):
            # Keep the orphan so the review queue can account for it.
            result.append(row)
            continue

        orphan_question = _clean_text(row.get("question"))
        orphan_choices = list(row.get("choices") or [])
        if previous.get("choices"):
            if orphan_question:
                previous["choices"][-1] = _clean_text(
                    f'{previous["choices"][-1]} {orphan_question}'
                )
            if orphan_choices:
                previous["choices"][-1] = _clean_text(
                    f'{previous["choices"][-1]} {orphan_choices[0]}'
                )
                previous["choices"].extend(orphan_choices[1:])
                previous["choice_ink_scores"].extend(
                    list(row.get("choice_ink_scores") or [])[1:]
                )
        else:
            previous["question"] = _clean_text(
                f'{previous.get("question", "")} {orphan_question}'
            )
            previous["choices"].extend(orphan_choices)
            previous["choice_ink_scores"].extend(
                list(row.get("choice_ink_scores") or [])
            )
        previous["source_pages"] = sorted(
            set(previous.get("source_pages") or [])
            | set(row.get("source_pages") or [])
        )
        previous["confidences"].extend(row.get("confidences") or [])
        previous["ended_before_next_number"] = bool(
            row.get("ended_before_next_number")
        )
    return result


def estimate_bold_choice(
    ink_scores: Iterable[float],
    margin_ratio: float = 0.12,
) -> tuple[int | None, bool]:
    scores = [float(value) for value in ink_scores]
    if not scores:
        return None, False
    winner = max(range(len(scores)), key=scores.__getitem__)
    if len(scores) == 1:
        return winner + 1, False
    runner_up = max(score for index, score in enumerate(scores) if index != winner)
    margin = (scores[winner] - runner_up) / max(abs(runner_up), 1e-9)
    return winner + 1, margin >= margin_ratio


def _candidate_hash(candidate: dict[str, Any]) -> str:
    payload = {
        "section": candidate.get("section"),
        "qnum": candidate.get("qnum"),
        "question": _clean_text(candidate.get("question")),
        "choices": [_clean_text(choice) for choice in candidate.get("choices") or []],
        "source_pages": candidate.get("source_pages") or [],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_question_id(section: str, qnum: int) -> int:
    section_index = list(SECTION_KEYS).index(section) + 1
    return section_index * 100_000 + int(qnum)


def _review_record(
    candidate: dict[str, Any],
    issues: Iterable[str],
    evidence: list[dict[str, Any]],
    proposed_correct: int | None,
) -> dict[str, Any]:
    pages = [int(page) for page in candidate.get("source_pages") or []]
    section = str(candidate.get("section") or "")
    return {
        "section": section,
        "section_title": SECTION_KEYS.get(section, section),
        "qnum": int(candidate.get("qnum") or 0),
        "extracted_question": _clean_text(candidate.get("question")),
        "extracted_choices": [
            _clean_text(choice) for choice in candidate.get("choices") or []
        ],
        "proposed_correct": proposed_correct,
        "source_page": min(pages, default=1),
        "source_pages": pages,
        "source_hash": _candidate_hash(candidate),
        "issues": sorted(set(issues)),
        "matches": evidence,
        "status": "needs_review",
    }


def classify_candidates(
    candidates: Iterable[dict[str, Any]],
    references: Iterable[ReferenceQuestion] = (),
    expected_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = [dict(candidate) for candidate in candidates]
    references = list(references)
    key_counts = Counter(
        (candidate.get("section"), int(candidate.get("qnum") or 0))
        for candidate in candidates
        if candidate.get("section") and int(candidate.get("qnum") or 0) > 0
    )
    verified: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    processed_keys: set[tuple[str, int]] = set()

    for candidate in candidates:
        section = str(candidate.get("section") or "")
        qnum = int(candidate.get("qnum") or 0)
        candidate_key = (section, qnum)
        if qnum > 0 and candidate_key in processed_keys:
            continue
        if qnum > 0:
            processed_keys.add(candidate_key)
        question = _clean_text(candidate.get("question"))
        choices = [_clean_text(choice) for choice in candidate.get("choices") or []]
        pages = [int(page) for page in candidate.get("source_pages") or []]
        issues = quality_issues(
            question,
            choices,
            [float(value) for value in candidate.get("confidences") or []],
            crossed_page=len(set(pages)) > 1,
            ended_before_next_number=bool(
                candidate.get("ended_before_next_number")
            ),
        )
        if section not in SECTION_KEYS:
            issues.append("unknown_section")
        if qnum < 1:
            issues.append("missing_question_number")
        if len(choices) < 2:
            issues.append("too_few_choices")
        if qnum > 0 and key_counts[(section, qnum)] > 1:
            issues.append("duplicate_number")

        proposed_correct, clear_bold = estimate_bold_choice(
            candidate.get("choice_ink_scores") or []
        )
        if proposed_correct is None or proposed_correct > len(choices):
            issues.append("missing_correct_answer")
        elif not clear_bold:
            issues.append("bold_choice_ambiguous")

        matches = match_references(question, choices, references)
        evidence = [asdict(match) for match in matches]
        for match in matches:
            if (
                match.match_type == "exact"
                and match.correct
                and match.correct != [proposed_correct]
            ):
                issues.append("correct_answer_conflict")

        if issues:
            reviews.append(
                _review_record(candidate, issues, evidence, proposed_correct)
            )
            continue

        verified.append(
            {
                "id": _stable_question_id(section, qnum),
                "section": section,
                "section_title": SECTION_KEYS[section],
                "qnum": qnum,
                "question": question,
                "choices": choices,
                "correct": [int(proposed_correct)],
                "source_page": min(pages, default=1),
                "source_hash": _candidate_hash(candidate),
                "verification_method": "pdf_bold",
                "match_evidence": evidence,
            }
        )

    # Every number up to the highest observed number in a section must be
    # represented once, even when OCR skipped it entirely. These placeholders
    # make omissions visible and block them from the public bank.
    represented = Counter(
        (str(item.get("section") or ""), int(item.get("qnum") or 0))
        for item in [*verified, *reviews]
        if int(item.get("qnum") or 0) > 0
    )
    for section in SECTION_KEYS:
        observed = [
            qnum
            for candidate in candidates
            if str(candidate.get("section") or "") == section
            and (qnum := int(candidate.get("qnum") or 0)) > 0
        ]
        expected_max = int(
            (expected_counts or {}).get(section, max(observed, default=0))
        )
        for qnum in range(1, expected_max + 1):
            if represented[(section, qnum)]:
                continue
            placeholder = {
                "section": section,
                "qnum": qnum,
                "question": "",
                "choices": [],
                "source_pages": [SECTION_PAGES[section].start],
            }
            reviews.append(
                _review_record(
                    placeholder,
                    ["missing_from_extraction"],
                    [],
                    None,
                )
            )

    verified.sort(key=lambda item: (list(SECTION_KEYS).index(item["section"]), item["qnum"]))
    reviews.sort(
        key=lambda item: (
            list(SECTION_KEYS).index(item["section"])
            if item["section"] in SECTION_KEYS
            else len(SECTION_KEYS),
            item["qnum"],
            item["source_page"],
        )
    )
    return verified, reviews, accounting_report(verified, reviews)


def section_for_page(page_number: int) -> str | None:
    for section, pages in SECTION_PAGES.items():
        if page_number in pages:
            return section
    return None


def extract_pdf_page_image(pdf_page: Any, destination: Path) -> Path:
    """Extract the largest embedded scan from a pypdf page."""

    from PIL import Image

    images = list(pdf_page.images)
    if not images:
        raise RuntimeError("PDF page has no embedded image")
    source = max(images, key=lambda item: len(item.data))
    with Image.open(io.BytesIO(source.data)) as image:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(destination, format="PNG", optimize=True)
    return destination


def extract_page_images(
    pdf_path: Path,
    destination: Path,
    page_numbers: Iterable[int] = range(3, 129),
) -> dict[int, Path]:
    """Extract selected physical PDF pages to deterministic PNG paths."""

    from pypdf import PdfReader

    pdf = PdfReader(str(pdf_path))
    result: dict[int, Path] = {}
    for physical_page in page_numbers:
        if physical_page < 1 or physical_page > len(pdf.pages):
            raise RuntimeError(
                f"physical page {physical_page} is outside a {len(pdf.pages)} page PDF"
            )
        image_path = destination / f"page-{physical_page:03d}.png"
        if not image_path.exists():
            extract_pdf_page_image(pdf.pages[physical_page - 1], image_path)
        result[int(physical_page)] = image_path
    return result


def _ink_per_character(image: Any, box: Any, text: str) -> float:
    from PIL import ImageStat

    points = box or []
    xs = [max(0, int(point[0])) for point in points if len(point) >= 2]
    ys = [max(0, int(point[1])) for point in points if len(point) >= 2]
    if not xs or not ys:
        return 0.0
    crop = image.crop((min(xs), min(ys), max(xs) + 1, max(ys) + 1)).convert("L")
    # Darkness is less sensitive than a hard black/white threshold to JPEG
    # artifacts in the scan.
    darkness = 255.0 - float(ImageStat.Stat(crop).mean[0])
    glyphs = max(1, sum(character.isalnum() for character in text))
    return darkness * crop.width * crop.height / glyphs / 100.0


def _group_close_values(values: Iterable[int], distance: int = 4) -> list[int]:
    groups: list[list[int]] = []
    for value in sorted(set(int(item) for item in values)):
        if not groups or value - groups[-1][-1] > distance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group)) for group in groups]


def annotate_page_structure(
    image: Any,
    lines: list[dict[str, Any]],
    starting_question_number: int,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Recover table rows, question numbers and answer bullets from pixels.

    The source is a scan, and generic OCR frequently ignores the narrow number
    column and isolated bullet glyphs. The document's printed table is much
    more stable than OCR, so we use its horizontal/vertical rules as anchors
    and inject deterministic structural markers before text grouping.
    """

    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = binary.shape

    # Find the three long vertical table rules first. A small horizontal
    # dilation absorbs scan skew; genuine rules cover nearly the full page.
    expanded_vertical = cv2.dilate(
        binary, np.ones((1, 13), dtype=np.uint8)
    )
    y_start, y_stop = round(height * 0.07), round(height * 0.96)
    vertical_coverage = np.count_nonzero(
        expanded_vertical[y_start:y_stop, :], axis=0
    ) / max(1, y_stop - y_start)
    rule_x = np.flatnonzero(vertical_coverage >= 0.75)
    vertical_bands: list[list[int]] = []
    for raw_x in rule_x:
        x = int(raw_x)
        if not vertical_bands or x - vertical_bands[-1][-1] > 1:
            vertical_bands.append([x])
        else:
            vertical_bands[-1].append(x)
    vertical_rules = [
        round(sum(band) / len(band)) for band in vertical_bands
    ]
    plausible_rules = [
        value for value in vertical_rules if width * 0.03 < value < width * 0.35
    ]
    if len(plausible_rules) >= 2:
        table_left, divider = plausible_rules[:2]
    else:
        table_left, divider = round(width * 0.072), round(width * 0.126)

    right_rules = [value for value in vertical_rules if value > width * 0.65]
    table_right = right_rules[-1] if right_rules else round(width * 0.95)

    # The row rules are a little tilted/broken. Dilating vertically turns each
    # one into a solid band; a real rule covers almost the full table width,
    # while text baselines do not. This recovers every row on the source scan.
    expanded = cv2.dilate(binary, np.ones((13, 1), dtype=np.uint8))
    coverage = np.count_nonzero(
        expanded[:, table_left:table_right], axis=1
    ) / max(1, table_right - table_left)
    rule_y = np.flatnonzero(coverage >= 0.75)
    boundary_bands: list[list[int]] = []
    for raw_y in rule_y:
        y = int(raw_y)
        if not boundary_bands or y - boundary_bands[-1][-1] > 1:
            boundary_bands.append([y])
        else:
            boundary_bands[-1].append(y)
    boundaries = [round(sum(band) / len(band)) for band in boundary_bands]
    boundaries = [
        value for value in boundaries if height * 0.06 < value < height * 0.98
    ]

    if len(boundaries) < 2:
        return lines, starting_question_number, {
            "boundaries": boundaries,
            "vertical_rules": vertical_rules,
            "table_left": table_left,
            "divider": divider,
            "table_right": table_right,
            "numbered_rows": 0,
            "bullets": 0,
            "structure_confidence": 0.0,
        }

    table_top, table_bottom = min(boundaries), max(boundaries)

    # Remove unreliable OCR readings of question numbers. They are replaced by
    # row markers derived from the table and the known sequential numbering.
    cleaned_lines = []
    for line in lines:
        left, _top, _right, _bottom = _box_bounds(line)
        if left < divider and NUMBER_RE.fullmatch(_clean_text(line.get("text"))):
            continue
        cleaned_lines.append(line)

    next_question = max(1, int(starting_question_number))
    markers: list[dict[str, Any]] = []
    bullet_centers: list[int] = []
    numbered_rows = 0
    row_boundaries = [
        value for value in boundaries if table_top <= value <= table_bottom
    ]
    for top, bottom in zip(row_boundaries, row_boundaries[1:]):
        if bottom - top < 32:
            continue
        number_crop = binary[
            top + 3:min(bottom - 2, top + max(48, height // 28)),
            table_left + 5:max(table_left + 6, divider - 5),
        ]
        has_number = bool(number_crop.size and cv2.countNonZero(number_crop) >= 18)
        if has_number:
            markers.append(
                {
                    "text": str(next_question),
                    "confidence": 1.0,
                    "box": [
                        [float(table_left + 8), float(top + 5)],
                        [float(divider - 8), float(top + 5)],
                        [float(divider - 8), float(top + 32)],
                        [float(table_left + 8), float(top + 32)],
                    ],
                    "ink_per_character": 0.0,
                    "is_choice": False,
                    "synthetic_structure": True,
                }
            )
            next_question += 1
            numbered_rows += 1

        row_roi = binary[top + 2:bottom - 1, divider + 28:divider + 62]
        contours, _ = cv2.findContours(
            row_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            fill = area / max(1.0, float(w * h))
            if (
                5 <= w <= 26
                and 5 <= h <= 26
                and 18 <= area <= 500
                and 0.55 <= w / max(h, 1) <= 1.7
                and fill >= 0.35
            ):
                bullet_centers.append(top + 2 + y + h // 2)

    bullet_centers = _group_close_values(bullet_centers, distance=5)
    for bullet_y in bullet_centers:
        candidates = []
        for line in cleaned_lines:
            left, top, right, bottom = _box_bounds(line)
            center_y = (top + bottom) / 2
            if left >= divider + 80 and abs(center_y - bullet_y) <= max(18, (bottom - top)):
                candidates.append((abs(center_y - bullet_y), left, line))
        if candidates:
            _distance, _left, target = min(candidates, key=lambda item: (item[0], item[1]))
            target["is_choice"] = True

    output = cleaned_lines + markers
    return output, next_question, {
        "boundaries": row_boundaries,
        "vertical_rules": vertical_rules,
        "table_left": table_left,
        "divider": divider,
        "table_right": table_right,
        "numbered_rows": numbered_rows,
        "bullets": len(bullet_centers),
        "bullet_centers": bullet_centers,
        "structure_confidence": 1.0 if numbered_rows else 0.5,
    }


def structured_text_boxes(
    image: Any,
    layout: dict[str, Any],
) -> list[tuple[list[int], bool]]:
    """Build OCR line boxes directly from table pixels.

    EasyOCR's generic scene-text detector is both slow and distracted by the
    printed grid. The scan has strong row geometry, so projection-based line
    segmentation is faster and preserves wrapped answer lines reliably.
    """

    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    divider = int(layout.get("divider") or 0)
    table_right = int(layout.get("table_right") or image.width)
    boundaries = [int(value) for value in layout.get("boundaries") or []]
    bullets = [int(value) for value in layout.get("bullet_centers") or []]
    result: list[tuple[list[int], bool]] = []

    for top, bottom in zip(boundaries, boundaries[1:]):
        y_start, y_stop = top + 8, bottom - 8
        x_start, x_stop = divider + 8, table_right - 8
        if y_stop <= y_start or x_stop <= x_start:
            continue
        roi = binary[y_start:y_stop, x_start:x_stop]
        projection = np.count_nonzero(roi, axis=1)
        ink_rows = np.flatnonzero(projection >= 8)
        bands: list[list[int]] = []
        for raw_y in ink_rows:
            y = int(raw_y) + y_start
            if not bands or y - bands[-1][-1] > 10:
                bands.append([y])
            else:
                bands[-1].append(y)

        for band in bands:
            band_top, band_bottom = band[0], band[-1]
            if band_bottom - band_top < 12:
                continue
            line_roi = binary[
                max(top + 2, band_top - 4):min(bottom - 2, band_bottom + 5),
                x_start:x_stop,
            ]
            ink_columns = np.flatnonzero(np.count_nonzero(line_roi, axis=0) >= 2)
            if not ink_columns.size:
                continue
            left = x_start + int(ink_columns[0])
            right = x_start + int(ink_columns[-1]) + 1
            center_y = (band_top + band_bottom) / 2
            is_choice = any(
                abs(center_y - bullet_y) <= max(20, band_bottom - band_top)
                for bullet_y in bullets
            )
            if is_choice:
                # Do not ask the recognizer to interpret the bullet itself.
                left = max(left, divider + 105)
            result.append(
                (
                    [
                        max(divider + 2, left - 5),
                        min(table_right - 2, right + 6),
                        max(top + 2, band_top - 5),
                        min(bottom - 2, band_bottom + 6),
                    ],
                    is_choice,
                )
            )
    return result


def run_ocr(
    image_path: Path,
    reader: Any,
    starting_question_number: int | None = None,
) -> list[dict[str, Any]]:
    from PIL import Image
    import numpy as np

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        if starting_question_number is None:
            _markers, _next, probe_layout = annotate_page_structure(
                image, [], 1
            )
            boundaries = list(probe_layout.get("boundaries") or [])
            table_left = int(probe_layout.get("table_left") or 0)
            divider = int(probe_layout.get("divider") or 0)
            if len(boundaries) >= 2 and divider > table_left:
                x_start = max(0, table_left - 20)
                x_stop = min(image.width, divider + 20)
                number_crop = np.asarray(image)[:, x_start:x_stop]
                detected = reader.readtext(
                    number_crop,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789",
                    canvas_size=4096,
                    mag_ratio=1.5,
                    text_threshold=0.3,
                    low_text=0.2,
                    link_threshold=0.3,
                )
                first_detected: tuple[int, int] | None = None
                for number_box, number_text, _confidence in detected:
                    cleaned_number = re.sub(r"\D", "", str(number_text))
                    if not cleaned_number:
                        continue
                    value = int(cleaned_number)
                    center_y = sum(float(point[1]) for point in number_box) / max(
                        1, len(number_box)
                    )
                    for row_index, (top, bottom) in enumerate(
                        zip(boundaries, boundaries[1:])
                    ):
                        if top <= center_y <= bottom and 1 <= value <= 200:
                            if first_detected is None or row_index < first_detected[0]:
                                first_detected = (row_index, value)
                            break
                if first_detected is not None:
                    first_row, first_value = first_detected
                    # A page can start with a continuation row. Count only
                    # numbered cells before the first OCR-readable number.
                    import cv2

                    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
                    binary = cv2.threshold(
                        gray,
                        0,
                        255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
                    )[1]
                    prior_numbered = 0
                    for top, bottom in list(
                        zip(boundaries, boundaries[1:])
                    )[:first_row]:
                        cell = binary[
                            top + 3:min(
                                bottom - 2,
                                top + max(48, image.height // 28),
                            ),
                            table_left + 5:max(table_left + 6, divider - 5),
                        ]
                        prior_numbered += int(
                            bool(cell.size and cv2.countNonZero(cell) >= 18)
                        )
                    starting_question_number = first_value - prior_numbered

        _probe_markers, _probe_next, layout = annotate_page_structure(
            image, [], starting_question_number or 1
        )
        structured_boxes = structured_text_boxes(image, layout)
        horizontal_list = [box for box, _is_choice in structured_boxes]
        recognized = reader.recognize(
            np.asarray(image.convert("L")),
            horizontal_list=horizontal_list,
            free_list=[],
            detail=1,
            paragraph=False,
            reformat=False,
        )
        result: list[dict[str, Any]] = []
        for index, (box, raw_text, confidence) in enumerate(recognized):
            text = _clean_text(raw_text)
            if not text:
                continue
            structural_choice = (
                structured_boxes[index][1]
                if index < len(structured_boxes)
                else False
            )
            result.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "box": [[float(x), float(y)] for x, y in box],
                    "ink_per_character": _ink_per_character(image, box, text),
                    "is_choice": structural_choice or bool(BULLET_RE.match(text)),
                }
            )
        if starting_question_number is not None:
            result, _next_question, layout = annotate_page_structure(
                image,
                result,
                starting_question_number,
            )
            layout["detected_starting_question_number"] = (
                starting_question_number
            )
            for line in result:
                line.setdefault("page_layout", layout)
    return result


def load_file_references(
    questions_path: Path | None,
    test_exam_path: Path | None,
) -> list[ReferenceQuestion]:
    references: list[ReferenceQuestion] = []
    if questions_path and questions_path.exists():
        bank = QuestionBank(str(questions_path))
        bank.load()
        references.extend(
            ReferenceQuestion(
                source="questions",
                question=item.question,
                choices=list(item.choices),
                correct=list(item.correct),
            )
            for item in bank.by_id.values()
        )
    if test_exam_path and test_exam_path.exists():
        payload = json.loads(test_exam_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("items", [])
        for item in items:
            question = _clean_text(item.get("question"))
            answer = _clean_text(item.get("correct_answer"))
            if question:
                references.append(
                    ReferenceQuestion(
                        source="test_exam_questions",
                        question=question,
                        choices=[answer] if answer else [],
                        correct=[1] if answer else [],
                    )
                )
    return references


def _deduplicate_references(
    references: Iterable[ReferenceQuestion],
) -> list[ReferenceQuestion]:
    result: list[ReferenceQuestion] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for reference in references:
        key = (
            reference.source,
            normalize_match_text(reference.question),
            tuple(normalize_match_text(choice) for choice in reference.choices),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(reference)
    return result


async def load_database_references(database_url: str) -> list[ReferenceQuestion]:
    import asyncpg

    connection = await asyncpg.connect(database_url)
    references: list[ReferenceQuestion] = []
    try:
        table_names = {
            str(row["tablename"])
            for row in await connection.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            )
        }
        if "questions" in table_names:
            for row in await connection.fetch(
                "SELECT question, choices, correct FROM questions"
            ):
                references.append(
                    ReferenceQuestion(
                        "database.questions",
                        row["question"] or "",
                        list(row["choices"] or []),
                        list(row["correct"] or []),
                    )
                )
        if "test_exam_questions" in table_names:
            for row in await connection.fetch(
                "SELECT question, correct_answer FROM test_exam_questions"
            ):
                answer = row["correct_answer"] or ""
                references.append(
                    ReferenceQuestion(
                        "database.test_exam_questions",
                        row["question"] or "",
                        [answer] if answer else [],
                        [1] if answer else [],
                    )
                )
        if "case_questions" in table_names:
            for row in await connection.fetch(
                "SELECT question, answers, correct_answer FROM case_questions"
            ):
                answers = list(row["answers"] or [])
                correct_text = _clean_text(row["correct_answer"])
                correct = [
                    index + 1
                    for index, answer in enumerate(answers)
                    if _clean_text(answer) == correct_text
                ]
                references.append(
                    ReferenceQuestion(
                        "database.case_questions",
                        row["question"] or "",
                        answers,
                        correct,
                    )
                )
    finally:
        await connection.close()
    return references


def load_references(
    questions_path: Path,
    test_exam_path: Path,
    database_url: str = "",
) -> list[ReferenceQuestion]:
    references = load_file_references(questions_path, test_exam_path)
    if database_url:
        references.extend(asyncio.run(load_database_references(database_url)))
    return _deduplicate_references(references)


def extract_document(
    pdf_path: Path,
    ocr_cache: Path,
    languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    from pypdf import PdfReader
    import easyocr

    pdf = PdfReader(str(pdf_path))
    if len(pdf.pages) < max(page.stop - 1 for page in SECTION_PAGES.values()):
        raise RuntimeError(f"PDF has only {len(pdf.pages)} pages; expected at least 128")
    ocr_cache.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(languages or ["uk", "ru", "en"], gpu=False)
    rows: list[dict[str, Any]] = []

    for physical_page in range(3, 129):
        section = section_for_page(physical_page)
        if section is None:
            continue
        image_path = ocr_cache / f"page-{physical_page:03d}.png"
        ocr_path = ocr_cache / f"page-{physical_page:03d}.json"
        if not image_path.exists():
            extract_pdf_page_image(pdf.pages[physical_page - 1], image_path)
        if ocr_path.exists():
            lines = json.loads(ocr_path.read_text(encoding="utf-8"))
        else:
            lines = run_ocr(image_path, reader)
            write_json(ocr_path, lines)
        page_rows = group_page_lines(lines, physical_page)
        for row in page_rows:
            row["section"] = section
        rows.extend(page_rows)
    return join_cross_page_rows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and classify first-stage attestation questions"
    )
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", default="attestation_questions.json", type=Path)
    parser.add_argument(
        "--reviews", default="attestation_review_candidates.json", type=Path
    )
    parser.add_argument(
        "--report", default="data/attestation_quality_report.json", type=Path
    )
    parser.add_argument("--ocr-cache", default="tmp/attestation-ocr", type=Path)
    parser.add_argument("--questions", default="questions_flat.json", type=Path)
    parser.add_argument(
        "--test-exam", default="test_exam_questions.json", type=Path
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    parser.add_argument("--languages", default="uk,ru,en")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    references = load_references(
        args.questions,
        args.test_exam,
        args.database_url,
    )
    candidates = extract_document(
        args.pdf,
        args.ocr_cache,
        [item.strip() for item in args.languages.split(",") if item.strip()],
    )
    verified, reviews, report = classify_candidates(
        candidates,
        references,
        expected_counts=EXPECTED_QUESTION_COUNTS,
    )
    report["pdf"] = str(args.pdf.resolve())
    report["physical_pages_processed"] = 126
    report["candidate_rows"] = len(candidates)
    write_json(args.output, verified)
    write_json(args.reviews, reviews)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["unaccounted_gaps"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
