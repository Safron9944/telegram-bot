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
        if number_match and left <= 180:
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
            if choice:
                current["choices"].append(choice)
                current["choice_ink_scores"].append(
                    float(line.get("ink_per_character") or 0.0)
                )
        elif current["choices"]:
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

    for candidate in candidates:
        section = str(candidate.get("section") or "")
        qnum = int(candidate.get("qnum") or 0)
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
        for qnum in range(1, max(observed, default=0) + 1):
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


def run_ocr(image_path: Path, reader: Any) -> list[dict[str, Any]]:
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        result: list[dict[str, Any]] = []
        for box, raw_text, confidence in reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
        ):
            text = _clean_text(raw_text)
            if not text:
                continue
            result.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "box": [[float(x), float(y)] for x, y in box],
                    "ink_per_character": _ink_per_character(image, box, text),
                    "is_choice": bool(BULLET_RE.match(text)),
                }
            )
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
    verified, reviews, report = classify_candidates(candidates, references)
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
