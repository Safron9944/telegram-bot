from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveBank:
    id: str
    path: str
    filename: str
    size: int
    supported: bool = False
    adapter: str | None = None
    status: str = "unsupported"
    title: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "filename": self.filename,
            "size": self.size,
            "supported": self.supported,
            "adapter": self.adapter,
            "status": self.status,
            "title": self.title,
        }


@dataclass(frozen=True)
class ParsedSection:
    title: str
    sort_order: int
    questions_count: int

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "sort_order": self.sort_order,
            "questions_count": self.questions_count,
        }


@dataclass(frozen=True)
class ParsedQuestion:
    source_key: str
    qnum: int
    topic: str
    question: str
    choices: tuple[str, ...]
    correct: tuple[int, ...]
    correct_texts: tuple[str, ...]
    shuffle_choices: bool = True

    def to_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "qnum": self.qnum,
            "topic": self.topic,
            "question": self.question,
            "choices": list(self.choices),
            "correct": list(self.correct),
            "correct_texts": list(self.correct_texts),
            "shuffle_choices": self.shuffle_choices,
        }


@dataclass(frozen=True)
class BankSummary:
    sections_count: int
    questions_count: int
    no_shuffle_count: int

    def to_dict(self) -> dict:
        return {
            "sections_count": self.sections_count,
            "questions_count": self.questions_count,
            "no_shuffle_count": self.no_shuffle_count,
        }


@dataclass(frozen=True)
class ParsedBank:
    adapter: str
    source: str
    source_version: str
    source_hash: str
    sections: tuple[ParsedSection, ...]
    questions: tuple[ParsedQuestion, ...]

    @property
    def summary(self) -> BankSummary:
        return BankSummary(
            sections_count=len(self.sections),
            questions_count=len(self.questions),
            no_shuffle_count=sum(not question.shuffle_choices for question in self.questions),
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_version": self.source_version,
            "source_hash": self.source_hash,
            "count": len(self.questions),
            "sections": [section.to_dict() for section in self.sections],
            "questions": [question.to_dict() for question in self.questions],
        }
