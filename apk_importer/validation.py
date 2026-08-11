from __future__ import annotations

from dataclasses import dataclass

from .models import ParsedBank


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    source_key: str
    message: str


class BankValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.code for issue in issues))


def _normalized(value: str) -> str:
    return " ".join(str(value).casefold().split())


def validate_bank(bank: ParsedBank) -> None:
    issues: list[ValidationIssue] = []
    section_titles = {section.title.strip() for section in bank.sections if section.title.strip()}
    seen_keys: set[str] = set()

    for question in bank.questions:
        key = question.source_key.strip()
        if not key:
            issues.append(ValidationIssue("missing_source_key", key, "Немає source key."))
        elif key in seen_keys:
            issues.append(ValidationIssue("duplicate_source_key", key, "Source key повторюється."))
        seen_keys.add(key)

        topic = question.topic.strip()
        if not topic:
            issues.append(ValidationIssue("missing_topic", key, "Немає розділу."))
        if topic not in section_titles:
            issues.append(ValidationIssue("unknown_section", key, "Розділ питання відсутній у банку."))
        if not question.question.strip():
            issues.append(ValidationIssue("empty_question", key, "Порожній текст питання."))

        choices = tuple(choice.strip() for choice in question.choices)
        if len(choices) < 2:
            issues.append(ValidationIssue("too_few_choices", key, "Потрібно щонайменше дві відповіді."))
        if len({_normalized(choice) for choice in choices}) != len(choices):
            issues.append(ValidationIssue("duplicate_choices", key, "Варіанти відповіді повторюються."))

        if not question.correct:
            issues.append(ValidationIssue("missing_correct_answer", key, "Немає правильної відповіді."))
        elif len(question.correct) != 1:
            issues.append(ValidationIssue("invalid_correct_count", key, "Має бути одна правильна відповідь."))

        valid_indexes = all(1 <= index <= len(choices) for index in question.correct)
        if question.correct and not valid_indexes:
            issues.append(ValidationIssue("correct_index_out_of_range", key, "Індекс правильної відповіді некоректний."))

        expected_texts = tuple(
            choices[index - 1]
            for index in question.correct
            if 1 <= index <= len(choices)
        )
        if tuple(question.correct_texts) != expected_texts:
            issues.append(ValidationIssue("correct_text_mismatch", key, "Текст правильної відповіді не збігається."))

    if issues:
        raise BankValidationError(issues)
