from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attestation import AttestationBank, AttestationQuestion, SECTION_KEYS
from attestation_quality import accounting_report


def load_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [dict(item) for item in payload]


def check_files(
    questions_path: Path,
    reviews_path: Path,
) -> dict[str, Any]:
    verified = load_list(questions_path)
    reviews = load_list(reviews_path)
    AttestationBank(AttestationQuestion(**item) for item in verified)

    invalid_sections = sorted(
        {
            str(item.get("section") or "")
            for item in [*verified, *reviews]
            if item.get("section") not in SECTION_KEYS
        }
    )
    if invalid_sections:
        raise ValueError(f"unknown sections: {invalid_sections}")
    non_pending_reviews = [
        item for item in reviews if item.get("status") != "needs_review"
    ]
    if non_pending_reviews:
        raise ValueError("source review file may contain only needs_review records")

    keys = [
        (str(item.get("section") or ""), int(item.get("qnum") or 0))
        for item in [*verified, *reviews]
        if int(item.get("qnum") or 0) > 0
    ]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate section/question numbers: {duplicates[:20]}")

    report = accounting_report(verified, reviews)
    if report["unaccounted_gaps"]:
        raise ValueError(f'unaccounted gaps: {report["unaccounted_gaps"]}')
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate attestation source JSON")
    parser.add_argument(
        "--questions", default="attestation_questions.json", type=Path
    )
    parser.add_argument(
        "--reviews", default="attestation_review_candidates.json", type=Path
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = check_files(args.questions, args.reviews)
    except (OSError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
