from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable


SECTION_KEYS = {
    "constitution": "Конституція України",
    "civil_service": "Закон України «Про державну службу»",
    "customs_code": "Митний кодекс України",
    "anti_corruption": "Закон України «Про запобігання корупції»",
}


class AttestationValidationError(ValueError):
    pass


@dataclass
class AttestationQuestion:
    id: int
    section: str
    section_title: str
    qnum: int
    question: str
    choices: list[str]
    correct: list[int]
    source_page: int
    source_hash: str
    verification_method: str
    match_evidence: list[dict] = field(default_factory=list)

    @property
    def correct_texts(self) -> list[str]:
        return [self.choices[index - 1] for index in self.correct]

    @property
    def topic(self) -> str:
        return self.section_title

    @property
    def ok(self):
        return None

    @property
    def level(self):
        return None


def validate_question(question: AttestationQuestion) -> None:
    if question.section not in SECTION_KEYS:
        raise AttestationValidationError("невідомий розділ")
    if question.qnum < 1 or not question.question.strip():
        raise AttestationValidationError("відсутній номер або текст питання")
    if len(question.choices) < 2:
        raise AttestationValidationError("потрібно щонайменше два варіанти")
    if any(not choice.strip() for choice in question.choices):
        raise AttestationValidationError("порожній варіант відповіді")
    normalized_choices = [" ".join(choice.casefold().split()) for choice in question.choices]
    if len(set(normalized_choices)) != len(normalized_choices):
        raise AttestationValidationError("повторений варіант відповіді")
    if (
        len(question.correct) != 1
        or question.correct[0] < 1
        or question.correct[0] > len(question.choices)
    ):
        raise AttestationValidationError("має бути рівно одна правильна відповідь")
    if question.source_page < 1:
        raise AttestationValidationError("відсутня сторінка PDF")
    if not question.source_hash.strip():
        raise AttestationValidationError("відсутній контрольний хеш")


class AttestationBank:
    def __init__(self, questions: Iterable[AttestationQuestion] = ()):
        self.by_id: dict[int, AttestationQuestion] = {}
        self.by_section: dict[str, list[AttestationQuestion]] = {
            section: [] for section in SECTION_KEYS
        }

        for question in questions:
            validate_question(question)
            if question.id in self.by_id:
                raise AttestationValidationError(f"повторений id {question.id}")
            if any(
                existing.qnum == question.qnum
                for existing in self.by_section[question.section]
            ):
                raise AttestationValidationError(
                    f"повторений номер {question.section}:{question.qnum}"
                )
            self.by_id[question.id] = question
            self.by_section[question.section].append(question)

        for section_questions in self.by_section.values():
            section_questions.sort(key=lambda item: (item.qnum, item.id))

    def pool(self, section: str) -> list[AttestationQuestion]:
        if section == "all":
            return [
                question
                for section_key in SECTION_KEYS
                for question in self.by_section[section_key]
            ]
        if section not in SECTION_KEYS:
            raise AttestationValidationError("невідомий розділ")
        return list(self.by_section[section])

    def select(
        self,
        section: str,
        mode: str,
        part: int = 1,
        rng=None,
    ) -> list[AttestationQuestion]:
        if section != "all" and section not in SECTION_KEYS:
            raise AttestationValidationError("невідомий розділ")
        if mode == "demo":
            sections = SECTION_KEYS if section == "all" else [section]
            return [
                question
                for section_key in sections
                for question in self.by_section[section_key][:10]
            ]

        questions = self.pool(section)
        if mode == "part":
            start = (max(1, int(part)) - 1) * 50
            return questions[start:start + 50]
        if mode == "random":
            picker = rng or random
            return picker.sample(questions, min(50, len(questions)))
        raise AttestationValidationError("невідомий режим")
