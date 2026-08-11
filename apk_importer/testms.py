from __future__ import annotations

from collections import OrderedDict
import re

from .models import ParsedBank, ParsedQuestion, ParsedSection
from .validation import validate_bank


_HEADER_RE = re.compile(r"^(testms[a-z0-9_-]*)\s+(\S+)\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^~([^\.]+)\.\s*(.+?)\s*$")
_QUESTION_RE = re.compile(r"^#([^\.]+)\.\s*(\d+)\.\s*(.+?)\s*$")


class TestMsParseError(ValueError):
    def __init__(self, code: str, message: str, *, line: int | None = None):
        self.code = code
        self.line = line
        super().__init__(message)


def _macro_index(first: str, second: str) -> tuple[int, str, str]:
    first_code = ord(first)
    second_code = ord(second)
    if not 37 <= second_code < 127 or not 37 <= first_code < 127:
        raise TestMsParseError("unresolved_macro", "Некоректне посилання на словник TestMS.")
    if first_code < 57:
        return (first_code - 37) * 90 + second_code - 37, " ", " "
    if first_code < 77:
        return (first_code - 57) * 90 + second_code - 37, " ", ""
    if first_code < 97:
        return (first_code - 77) * 90 + second_code - 37, "", " "
    return (first_code - 97) * 90 + second_code - 37, "", ""


def _expand_line(line: str, dictionary: tuple[str, ...], line_number: int) -> str:
    cursor = 0
    result: list[str] = []
    while True:
        marker = line.find("$", cursor)
        if marker < 0:
            result.append(line[cursor:])
            return "".join(result)
        result.append(line[cursor:marker])
        if marker + 2 >= len(line):
            raise TestMsParseError(
                "unresolved_macro",
                "Неповне посилання на словник TestMS.",
                line=line_number,
            )
        try:
            index, prefix, suffix = _macro_index(line[marker + 1], line[marker + 2])
            token = dictionary[index]
        except (IndexError, TestMsParseError) as exc:
            raise TestMsParseError(
                "unresolved_macro",
                "Посилання на відсутній елемент словника TestMS.",
                line=line_number,
            ) from exc
        result.extend((prefix, token, suffix))
        cursor = marker + 3


def expand_testms_macros(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if len(lines) < 2 or not _HEADER_RE.match(lines[0].strip()):
        raise TestMsParseError("invalid_header", "Некоректний заголовок банку TestMS.", line=1)
    if not lines[1].startswith("$"):
        raise TestMsParseError("missing_dictionary", "У банку TestMS відсутній словник.", line=2)

    dictionary = tuple(lines[1][1:].split("$"))
    expanded = lines[:2]
    expanded.extend(_expand_line(line, dictionary, number) for number, line in enumerate(lines[2:], 3))
    return "\n".join(expanded)


def parse_testms_bank(text: str, *, source: str, source_hash: str, default_section_title: str = "Питання") -> ParsedBank:
    expanded = expand_testms_macros(text)
    lines = expanded.splitlines()
    header = _HEADER_RE.match(lines[0].strip())
    assert header is not None
    version = header.group(2)

    sections: OrderedDict[str, str] = OrderedDict()
    section_counts: dict[str, int] = {}
    questions: list[ParsedQuestion] = []
    active_section: tuple[str, str, str] | None = None
    pending: dict | None = None

    def finish_question(line_number: int) -> None:
        nonlocal pending
        if pending is None:
            return
        choices = tuple(pending["choices"])
        correct_indexes = tuple(pending["correct"])
        if len(choices) < 2:
            raise TestMsParseError(
                "too_few_choices", "Питання має менше двох варіантів відповіді.", line=line_number
            )
        if len(correct_indexes) != 1:
            raise TestMsParseError(
                "invalid_correct_count", "Питання повинно мати одну правильну відповідь.", line=line_number
            )
        correct_texts = tuple(choices[index - 1] for index in correct_indexes)
        section_code = pending["section_code"]
        questions.append(
            ParsedQuestion(
                source_key=f"{source}:{section_code}:{pending['qnum']}",
                qnum=pending["qnum"],
                topic=pending["topic"],
                question=pending["question"],
                choices=choices,
                correct=correct_indexes,
                correct_texts=correct_texts,
                shuffle_choices=pending["shuffle_choices"],
            )
        )
        section_counts[section_code] += 1
        pending = None

    for line_number, raw_line in enumerate(lines[2:], 3):
        line = raw_line.strip()
        if not line:
            finish_question(line_number)
            continue

        if pending is not None and line[0] in "+^-":
            answer = line[1:].strip()
            if not answer:
                raise TestMsParseError("empty_choice", "Порожній варіант відповіді.", line=line_number)
            pending["choices"].append(answer)
            if line[0] in "+^":
                pending["correct"].append(len(pending["choices"]))
                if line[0] == "^":
                    pending["shuffle_choices"] = False
            continue
        if pending is not None and line.startswith("*"):
            continue
        if pending is not None:
            finish_question(line_number)

        section = _SECTION_RE.match(line)
        if section:
            code, title = section.group(1).strip(), section.group(2).strip()
            internal_code = code
            suffix = 2
            while internal_code in sections:
                internal_code = f"{code}#{suffix}"
                suffix += 1
            sections[internal_code] = title
            section_counts[internal_code] = 0
            active_section = (code, internal_code, title)
            continue

        question = _QUESTION_RE.match(line)
        if question:
            code, qnum, question_text = question.group(1).strip(), int(question.group(2)), question.group(3).strip()
            if active_section is None:
                sections[code] = default_section_title
                section_counts[code] = 0
                active_section = (code, code, default_section_title)
            if active_section[0] != code:
                raise TestMsParseError("unknown_section", "Питання не належить активному розділу.", line=line_number)
            pending = {
                "section_code": active_section[1],
                "qnum": qnum,
                "topic": active_section[2],
                "question": question_text,
                "choices": [],
                "correct": [],
                "shuffle_choices": True,
            }
            continue

        if line[0] in "+^-*":
            raise TestMsParseError("orphan_answer", "Варіант відповіді не має питання.", line=line_number)
        # Metadata records before the first section are not question content.
        if active_section is not None:
            raise TestMsParseError("unknown_record", "Невідомий запис у банку TestMS.", line=line_number)

    finish_question(len(lines) + 1)
    if not sections or not questions:
        raise TestMsParseError("empty_bank", "Банк TestMS не містить розділів або питань.")

    parsed_sections = tuple(
        ParsedSection(title=title, sort_order=order, questions_count=section_counts[code])
        for order, (code, title) in enumerate(sections.items())
    )
    bank = ParsedBank(
        adapter="testms",
        source=source,
        source_version=version,
        source_hash=source_hash,
        sections=parsed_sections,
        questions=tuple(questions),
    )
    validate_bank(bank, allow_duplicate_choices=True)
    return bank
