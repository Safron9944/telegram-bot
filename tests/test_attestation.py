import random

import pytest

from attestation import (
    AttestationBank,
    AttestationQuestion,
    AttestationValidationError,
    SECTION_KEYS,
    validate_question,
)


def question(section="constitution", number=1, qid=1):
    return AttestationQuestion(
        id=qid,
        section=section,
        section_title=SECTION_KEYS[section],
        qnum=number,
        question="Яке твердження правильне?",
        choices=["Перша відповідь", "Друга відповідь"],
        correct=[2],
        source_page=3,
        source_hash=f"hash-{section}-{number}",
        verification_method="pdf_visual",
    )


def test_validate_question_rejects_empty_choice():
    broken = question()
    broken.choices = ["Повна відповідь", ""]

    with pytest.raises(AttestationValidationError, match="порожній варіант"):
        validate_question(broken)


def test_validate_question_rejects_duplicate_choices():
    broken = question()
    broken.choices = ["Та сама відповідь", "  та сама відповідь "]

    with pytest.raises(AttestationValidationError, match="повторений варіант"):
        validate_question(broken)


def test_bank_builds_parts_random_and_fixed_demo():
    items = [question(number=n, qid=n) for n in range(1, 121)]
    bank = AttestationBank(items)

    assert [q.qnum for q in bank.select("constitution", "part", part=2)] == list(range(51, 101))
    assert [q.qnum for q in bank.select("constitution", "demo")] == list(range(1, 11))
    picked = bank.select("constitution", "random", rng=random.Random(7))
    assert len(picked) == 50
    assert len({q.id for q in picked}) == 50


def test_all_demo_contains_ten_from_each_section():
    items = []
    qid = 1
    for section in SECTION_KEYS:
        for number in range(1, 13):
            items.append(question(section=section, number=number, qid=qid))
            qid += 1

    bank = AttestationBank(items)
    demo = bank.select("all", "demo")

    assert len(demo) == 40
    assert {key: sum(q.section == key for q in demo) for key in SECTION_KEYS} == {
        key: 10 for key in SECTION_KEYS
    }


def test_bank_rejects_duplicate_number_inside_section():
    with pytest.raises(AttestationValidationError, match="повторений номер"):
        AttestationBank([question(qid=1), question(qid=2)])
