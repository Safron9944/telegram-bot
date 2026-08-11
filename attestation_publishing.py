from __future__ import annotations

import inspect
import re
from pathlib import PurePath
from typing import Awaitable, Callable

from apk_importer.models import ParsedBank


class AttestationPublishError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _slug_for(bank: ParsedBank) -> str:
    stem = PurePath(bank.source or "assessment").stem.casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or f"assessment-{bank.source_hash[:12]}"


def validate_publishable_bank(bank: ParsedBank, title: str) -> str:
    clean_title = (title or "").strip()
    if not clean_title:
        raise AttestationPublishError("title_required", "Вкажіть назву розділу.")
    if not bank.questions or not bank.sections:
        raise AttestationPublishError("bank_empty", "У банку немає питань або розділів.")
    for question in bank.questions:
        if len(question.choices) < 2:
            raise AttestationPublishError(
                "answer_choices_required",
                f"Питання {question.qnum} повинно мати щонайменше два варіанти відповіді.",
            )
        if not question.correct or any(index < 1 or index > len(question.choices) for index in question.correct):
            raise AttestationPublishError(
                "correct_answer_invalid",
                f"У питанні {question.qnum} неправильно визначена правильна відповідь.",
            )
    return clean_title


class AttestationPublishingService:
    def __init__(self, store, reload_catalog: Callable[[], Awaitable[None] | None]):
        self.store = store
        self.reload_catalog = reload_catalog

    async def publish(self, bank: ParsedBank, title: str, *, changed_by: str) -> dict:
        clean_title = validate_publishable_bank(bank, title)
        result = await self.store.publish_attestation_bank(
            bank,
            title=clean_title,
            slug=_slug_for(bank),
            changed_by=str(changed_by),
        )
        reload_result = self.reload_catalog()
        if inspect.isawaitable(reload_result):
            await reload_result
        return result
