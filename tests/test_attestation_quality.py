from attestation_quality import (
    ReferenceQuestion,
    accounting_report,
    completeness_report,
    match_references,
    normalize_match_text,
    quality_issues,
)


def test_normalize_match_text_preserves_numbers_and_words():
    assert normalize_match_text("  Стаття 19 — Конституції  ") == "стаття 19 конституції"
    assert normalize_match_text("обов’язок") == normalize_match_text("обов'язок")


def test_exact_match_requires_full_question_and_all_choices():
    refs = [
        ReferenceQuestion(
            "questions",
            "Чи є Україна унітарною державою?",
            ["так", "ні"],
            [1],
        )
    ]

    result = match_references(
        "Чи є Україна унітарною державою?",
        ["так", "ні"],
        refs,
    )
    assert result[0].match_type == "exact"

    changed = match_references(
        "Чи є Україна унітарною державою?",
        ["так", "інколи"],
        refs,
    )
    assert not any(item.match_type == "exact" for item in changed)


def test_quality_flags_low_confidence_and_page_break_truncation():
    issues = quality_issues(
        question="Громадянин має пра",
        choices=["повну відповідь", "частина відповіді"],
        confidences=[0.98, 0.42, 0.97],
        crossed_page=True,
        ended_before_next_number=False,
    )

    assert "low_ocr_confidence" in issues
    assert "page_break_not_closed" in issues


def test_quality_flags_empty_and_suspicious_word_breaks():
    issues = quality_issues(
        question="Текст із об",
        choices=["повна відповідь", ""],
        confidences=[0.99],
        crossed_page=False,
        ended_before_next_number=True,
    )

    assert issues == ["empty_text", "suspicious_word_break"]


def test_completeness_reports_missing_and_duplicate_numbers():
    report = completeness_report({"constitution": [1, 2, 2, 4]})

    assert report["constitution"]["missing"] == [3]
    assert report["constitution"]["duplicates"] == [2]


def test_accounting_report_counts_verified_reviews_and_unaccounted_gaps():
    verified = [
        {"section": "constitution", "qnum": 1},
        {"section": "constitution", "qnum": 2},
    ]
    reviews = [
        {"section": "constitution", "qnum": 4, "status": "needs_review"},
    ]

    report = accounting_report(verified, reviews)

    assert report["verified"] == 2
    assert report["needs_review"] == 1
    assert report["unaccounted_gaps"] == {"constitution": [3]}
