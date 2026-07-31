from attestation_quality import ReferenceQuestion
from scripts.extract_attestation_questions import (
    classify_candidates,
    estimate_bold_choice,
    group_page_lines,
    join_cross_page_rows,
)


def test_group_page_lines_keeps_wrapped_answers_with_question(load_fixture_json):
    lines = load_fixture_json("attestation_page_ocr.json")

    rows = group_page_lines(lines, page_number=3)

    assert [row["qnum"] for row in rows] == [1, 2]
    assert rows[0]["question"] == "Якою є Україна за державним устроєм?"
    assert rows[0]["choices"] == [
        "федеративною державою",
        "конфедеративною державою",
        "унітарною державою",
        "республікою",
    ]
    assert rows[0]["ended_before_next_number"] is True
    assert rows[1]["ended_before_next_number"] is False


def test_group_page_lines_appends_wrapped_choice_text():
    lines = [
        {"text": "7", "confidence": 0.99, "box": [[100, 10], [120, 10], [120, 30], [100, 30]]},
        {"text": "Оберіть відповідь", "confidence": 0.99, "box": [[200, 10], [500, 10], [500, 30], [200, 30]]},
        {"text": "• перша частина", "confidence": 0.99, "box": [[220, 50], [500, 50], [500, 70], [220, 70]], "is_choice": True},
        {"text": "продовження відповіді", "confidence": 0.99, "box": [[255, 75], [600, 75], [600, 95], [255, 95]]},
        {"text": "• інший варіант", "confidence": 0.99, "box": [[220, 100], [500, 100], [500, 120], [220, 120]], "is_choice": True},
    ]

    row = group_page_lines(lines, page_number=4)[0]

    assert row["choices"] == ["перша частина продовження відповіді", "інший варіант"]


def test_join_cross_page_rows_merges_orphan_continuation():
    rows = [
        {"qnum": 9, "question": "Питання", "choices": ["довга відповідь"], "source_pages": [5], "confidences": [0.9], "choice_ink_scores": [20.0], "ended_before_next_number": False},
        {"qnum": None, "question": "", "choices": ["продовження", "друга відповідь"], "source_pages": [6], "confidences": [0.9], "choice_ink_scores": [20.0, 21.0], "ended_before_next_number": True},
        {"qnum": 10, "question": "Наступне питання", "choices": ["так", "ні"], "source_pages": [6], "confidences": [0.9], "choice_ink_scores": [21.0, 20.0], "ended_before_next_number": False},
    ]

    merged = join_cross_page_rows(rows)

    assert len(merged) == 2
    assert merged[0]["choices"] == ["довга відповідь продовження", "друга відповідь"]
    assert merged[0]["source_pages"] == [5, 6]


def test_estimate_bold_choice_requires_clear_margin():
    assert estimate_bold_choice([20.0, 20.4, 28.8, 20.1]) == (3, True)
    assert estimate_bold_choice([20.0, 20.4, 21.0, 20.1]) == (3, False)


def test_classify_candidates_keeps_conflict_out_of_verified_bank():
    candidate = {
        "section": "constitution",
        "qnum": 1,
        "question": "Якою є Україна?",
        "choices": ["унітарною", "федеративною"],
        "source_pages": [3],
        "confidences": [0.99, 0.99, 0.99],
        "choice_ink_scores": [28.0, 20.0],
        "ended_before_next_number": True,
    }
    references = [
        ReferenceQuestion(
            source="questions",
            question="Якою є Україна?",
            choices=["унітарною", "федеративною"],
            correct=[2],
        )
    ]

    verified, reviews, report = classify_candidates([candidate], references)

    assert verified == []
    assert reviews[0]["status"] == "needs_review"
    assert "correct_answer_conflict" in reviews[0]["issues"]
    assert report["needs_review"] == 1
