# First Stage Attestation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Витягнути й перевірити питання зі сканованого PDF, додати окремий тест «Перший етап атестації» з full/admin-доступом, фіксованим демо та адмін-чергою проблемних питань.

**Architecture:** OCR і звірка виконуються офлайн та створюють валідований банк і чергу кандидатів. Production завантажує лише перевірені питання в окрему таблицю, а чинний механізм сесій отримує явний селектор банку, щоб не змішувати ідентифікатори та статистику. Неперевірені кандидати зберігаються окремо, редагуються в адмін-панелі й потрапляють у тест лише після повторної серверної валідації.

**Tech Stack:** Python 3.12+ для OCR, FastAPI, Pydantic, asyncpg/PostgreSQL, vanilla ES modules, Node built-in test runner, pytest/pytest-asyncio, pypdf, Pillow, EasyOCR (тільки офлайн).

**Design:** `docs/superpowers/specs/2026-07-31-first-stage-attestation-design.md`

---

## File map

- Create `attestation.py` — модель перевіреного питання, чотири розділи, банк, валідація, вибір part/random/demo.
- Create `attestation_quality.py` — нормалізація, пошук збігів, проблеми OCR, звіт повноти.
- Create `scripts/extract_attestation_questions.py` — витяг з PDF, OCR, групування рядків і формування кандидатів.
- Create `scripts/check_attestation_questions.py` — фінальна машинна перевірка JSON і звіту.
- Create `requirements-dev.txt`, `requirements-ocr.txt`, `pytest.ini` — тестові та офлайн-залежності без зміни production requirements.
- Create `tests/test_attestation.py`, `tests/test_attestation_quality.py`, `tests/test_attestation_storage.py`, `tests/test_attestation_service.py` — доменні, SQL і сервісні тести.
- Create `static/js/screens/attestation.js` — користувацькі екрани розділів і частин.
- Create `static/js/screens/admin-attestation.js` — черга проблемних питань, редактор і звіт повноти.
- Create `tests/js/attestation.test.mjs`, `package.json` — тести чистих frontend-функцій через `node --test`.
- Create `attestation_questions.json` — тільки перевірені записи.
- Create `attestation_review_candidates.json` — кандидати `needs_review` для початкового імпорту.
- Create `data/attestation_quality_report.json` — відтворюваний звіт по розділах і пропусках.
- Modify `storage.py` — таблиці, імпорт, читання банку, review CRUD, окрема статистика.
- Modify `app.py` — runtime bank, доступ, resolver питань, сесії, public/admin API, startup sync.
- Modify `static/js/app.js`, `static/js/core/state.js`, `static/js/screens/session.js`, `static/js/screens/user.js`, `static/js/screens/admin.js` — маршрутизація та підключення нових екранів.
- Modify `static/styles/components.css`, `static/styles/responsive.css`, `static/index.html` — review diff/editor і cache-busting.

### Task 1: Test harness and attestation domain model

**Files:**
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `attestation.py`
- Create: `tests/conftest.py`
- Create: `tests/test_attestation.py`

- [ ] **Step 1: Add the failing domain tests**

```python
# tests/test_attestation.py
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


def test_validate_question_rejects_truncated_or_invalid_content():
    broken = question()
    broken.choices = ["Повна відповідь", ""]
    with pytest.raises(AttestationValidationError, match="порожній варіант"):
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
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `python -m pytest tests/test_attestation.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'attestation'`.

- [ ] **Step 3: Add the dev test configuration**

```text
# requirements-dev.txt
pytest>=8.3,<9.0
pytest-asyncio>=0.24,<1.0
```

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

```python
# tests/conftest.py
import json
from pathlib import Path
import pytest


@pytest.fixture
def load_fixture_json():
    fixtures = Path(__file__).parent / "fixtures"
    return lambda name: json.loads((fixtures / name).read_text(encoding="utf-8"))
```

- [ ] **Step 4: Install the pinned development test tools**

Run: `python -m pip install -r requirements-dev.txt`

Expected: pytest 8.x and pytest-asyncio install successfully.

- [ ] **Step 5: Implement the domain model and selection rules**

```python
# attestation.py
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


def validate_question(q: AttestationQuestion) -> None:
    if q.section not in SECTION_KEYS:
        raise AttestationValidationError("невідомий розділ")
    if q.qnum < 1 or not q.question.strip():
        raise AttestationValidationError("відсутній номер або текст питання")
    if len(q.choices) < 2:
        raise AttestationValidationError("потрібно щонайменше два варіанти")
    if any(not choice.strip() for choice in q.choices):
        raise AttestationValidationError("порожній варіант відповіді")
    if len({choice.casefold().strip() for choice in q.choices}) != len(q.choices):
        raise AttestationValidationError("повторений варіант відповіді")
    if len(q.correct) != 1 or q.correct[0] < 1 or q.correct[0] > len(q.choices):
        raise AttestationValidationError("має бути рівно одна правильна відповідь")


class AttestationBank:
    def __init__(self, questions: Iterable[AttestationQuestion] = ()):
        self.by_id: dict[int, AttestationQuestion] = {}
        self.by_section: dict[str, list[AttestationQuestion]] = {key: [] for key in SECTION_KEYS}
        for q in questions:
            validate_question(q)
            if q.id in self.by_id:
                raise AttestationValidationError(f"повторений id {q.id}")
            if any(existing.qnum == q.qnum for existing in self.by_section[q.section]):
                raise AttestationValidationError(f"повторений номер {q.section}:{q.qnum}")
            self.by_id[q.id] = q
            self.by_section[q.section].append(q)
        for items in self.by_section.values():
            items.sort(key=lambda item: (item.qnum, item.id))

    def pool(self, section: str) -> list[AttestationQuestion]:
        if section == "all":
            return [q for key in SECTION_KEYS for q in self.by_section[key]]
        if section not in SECTION_KEYS:
            raise AttestationValidationError("невідомий розділ")
        return list(self.by_section[section])

    def select(self, section: str, mode: str, part: int = 1, rng=None) -> list[AttestationQuestion]:
        if mode == "demo":
            keys = SECTION_KEYS if section == "all" else [section]
            return [q for key in keys for q in self.by_section[key][:10]]
        items = self.pool(section)
        if mode == "part":
            start = (max(1, part) - 1) * 50
            return items[start:start + 50]
        if mode == "random":
            picker = rng or random
            return picker.sample(items, min(50, len(items)))
        raise AttestationValidationError("невідомий режим")
```

- [ ] **Step 6: Run the domain tests**

Run: `python -m pytest tests/test_attestation.py -q`

Expected: `4 passed`.

- [ ] **Step 7: Commit the domain slice**

```powershell
git add requirements-dev.txt pytest.ini attestation.py tests/conftest.py tests/test_attestation.py
git commit -m "feat: add attestation question domain"
```

### Task 2: Quality checks, matching, and completeness report

**Files:**
- Create: `attestation_quality.py`
- Create: `tests/test_attestation_quality.py`

- [ ] **Step 1: Write failing tests for exact/fuzzy matches and gaps**

```python
# tests/test_attestation_quality.py
from attestation_quality import ReferenceQuestion, match_references, quality_issues, completeness_report


def test_exact_match_requires_full_question_and_all_choices():
    refs = [ReferenceQuestion("questions", "Чи є Україна унітарною державою?", ["так", "ні"], [1])]
    result = match_references("Чи є Україна унітарною державою?", ["так", "ні"], refs)
    assert result[0].match_type == "exact"
    changed = match_references("Чи є Україна унітарною державою?", ["так", "інколи"], refs)
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


def test_completeness_reports_missing_and_duplicate_numbers():
    report = completeness_report({"constitution": [1, 2, 2, 4]})
    assert report["constitution"]["missing"] == [3]
    assert report["constitution"]["duplicates"] == [2]
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_attestation_quality.py -q`

Expected: FAIL because `attestation_quality` does not exist.

- [ ] **Step 3: Implement deterministic normalization and evidence types**

```python
# attestation_quality.py
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import Counter


def normalize_match_text(value: str) -> str:
    value = value.casefold().replace("’", "'").replace("`", "'")
    value = re.sub(r"[^0-9a-zа-яіїєґ']+", " ", value, flags=re.IGNORECASE)
    return " ".join(value.split())


@dataclass(frozen=True)
class ReferenceQuestion:
    source: str
    question: str
    choices: list[str]
    correct: list[int]


@dataclass(frozen=True)
class MatchEvidence:
    source: str
    match_type: str
    score: float
    question: str
    choices: list[str]
    correct: list[int]


def match_references(question: str, choices: list[str], references: list[ReferenceQuestion]) -> list[MatchEvidence]:
    nq = normalize_match_text(question)
    nc = [normalize_match_text(choice) for choice in choices]
    results = []
    for ref in references:
        rq = normalize_match_text(ref.question)
        rc = [normalize_match_text(choice) for choice in ref.choices]
        exact = nq == rq and nc == rc
        score = SequenceMatcher(None, nq, rq).ratio()
        if exact or score >= 0.65:
            results.append(MatchEvidence(ref.source, "exact" if exact else "fuzzy", score, ref.question, ref.choices, ref.correct))
    return sorted(results, key=lambda item: (item.match_type == "exact", item.score), reverse=True)[:3]


def quality_issues(question, choices, confidences, crossed_page, ended_before_next_number):
    issues = []
    if not question.strip() or any(not item.strip() for item in choices):
        issues.append("empty_text")
    if confidences and min(confidences) < 0.70:
        issues.append("low_ocr_confidence")
    if crossed_page and not ended_before_next_number:
        issues.append("page_break_not_closed")
    if any(re.search(r"\b[а-яіїєґ]{1,2}$", item.strip(), re.IGNORECASE) for item in [question, *choices]):
        issues.append("suspicious_word_break")
    return sorted(set(issues))


def completeness_report(numbers_by_section):
    result = {}
    for section, numbers in numbers_by_section.items():
        counts = Counter(numbers)
        maximum = max(numbers, default=0)
        result[section] = {
            "found": len(numbers),
            "missing": [number for number in range(1, maximum + 1) if counts[number] == 0],
            "duplicates": sorted(number for number, count in counts.items() if count > 1),
        }
    return result


def accounting_report(verified, reviews):
    numbers = {}
    for item in [*verified, *reviews]:
        numbers.setdefault(item["section"], []).append(int(item["qnum"]))
    sections = completeness_report(numbers)
    return {
        "sections": sections,
        "verified": len(verified),
        "needs_review": sum(item.get("status") == "needs_review" for item in reviews),
        "unaccounted_gaps": {
            section: values["missing"]
            for section, values in sections.items()
            if values["missing"]
        },
    }
```

- [ ] **Step 4: Run quality tests**

Run: `python -m pytest tests/test_attestation_quality.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the quality slice**

```powershell
git add attestation_quality.py tests/test_attestation_quality.py
git commit -m "feat: validate attestation extraction quality"
```

### Task 3: Offline PDF/OCR extraction pipeline

**Files:**
- Create: `requirements-ocr.txt`
- Create: `scripts/extract_attestation_questions.py`
- Create: `scripts/check_attestation_questions.py`
- Create: `tests/fixtures/attestation_page.png`
- Create: `tests/fixtures/attestation_page_ocr.json`
- Create: `tests/test_attestation_extractor.py`

- [ ] **Step 1: Add a small page fixture and failing row-grouping test**

Use physical PDF page 3 to create a cropped fixture containing questions 1 and 2. Store the OCR response as deterministic JSON so unit tests never download an OCR model.

```python
# tests/test_attestation_extractor.py
from scripts.extract_attestation_questions import group_page_lines


def test_group_page_lines_keeps_wrapped_answers_with_their_question(load_fixture_json):
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
```

- [ ] **Step 2: Run and confirm extractor failure**

Run: `python -m pytest tests/test_attestation_extractor.py -q`

Expected: FAIL because the extractor module/function does not exist.

- [ ] **Step 3: Add isolated OCR dependencies**

```text
# requirements-ocr.txt
easyocr>=1.7,<2.0
pypdf>=5.0,<7.0
Pillow>=11.0,<13.0
```

- [ ] **Step 4: Implement the CLI contract and deterministic grouping**

The script must expose `extract_page_images`, `run_ocr`, `group_page_lines`, `join_cross_page_rows`, `estimate_bold_choice`, and `main`. OCR output records must use this stable shape:

```python
line = {
    "text": "унітарною державою",
    "confidence": 0.98,
    "box": [[250, 710], [710, 710], [710, 760], [250, 760]],
    "ink_per_character": 34.2,
}
```

Implement the section page map and CLI exactly as follows:

```python
SECTION_PAGES = {
    "constitution": range(3, 31),
    "civil_service": range(31, 62),
    "customs_code": range(62, 93),
    "anti_corruption": range(93, 129),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", default="attestation_questions.json")
    parser.add_argument("--reviews", default="attestation_review_candidates.json")
    parser.add_argument("--report", default="data/attestation_quality_report.json")
    parser.add_argument("--ocr-cache", default="tmp/attestation-ocr")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    args = parser.parse_args()
    candidates = extract_document(Path(args.pdf), Path(args.ocr_cache), ["uk", "ru", "en"])
    references = load_references(Path("questions_flat.json"), Path("test_exam_questions.json"), args.database_url)
    verified, reviews, report = classify_candidates(candidates, references)
    write_json(Path(args.output), verified)
    write_json(Path(args.reviews), reviews)
    write_json(Path(args.report), report)
```

Classification rules must be deterministic: structural errors, OCR confidence below `0.70`, bold-score margin below `0.12`, page-boundary ambiguity, numbering gaps, or any exact-match content conflict produce `needs_review`. A fuzzy match is evidence only and never changes extracted text or status.

`load_references` must load all currently available sources. Load `questions_flat.json` through `QuestionBank.load()` and convert every `Q` to `ReferenceQuestion("questions", q.question, q.choices, q.correct)`. Load `test_exam_questions.json` as fuzzy-only references because it has no complete choice list. When `--database-url` is set, read `questions`, `test_exam_questions`, and `case_questions` with read-only `SELECT` statements; case and test-exam rows without full choices can produce only fuzzy evidence. Deduplicate references by normalized source/question/choices before matching.

- [ ] **Step 5: Implement the standalone final checker**

```python
# scripts/check_attestation_questions.py
import json
from pathlib import Path

from attestation import AttestationBank, AttestationQuestion
from attestation_quality import accounting_report


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    verified = load_json(Path("attestation_questions.json"))
    reviews = load_json(Path("attestation_review_candidates.json"))
    bank = AttestationBank(AttestationQuestion(**item) for item in verified)
    report = accounting_report(verified, reviews)
    if report["unaccounted_gaps"]:
        raise SystemExit(f"Unaccounted question numbers: {report['unaccounted_gaps']}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run extractor tests without invoking EasyOCR**

Run: `python -m pytest tests/test_attestation_extractor.py tests/test_attestation_quality.py -q`

Expected: all tests PASS using cached fixture JSON.

- [ ] **Step 7: Commit the extraction tooling**

```powershell
git add requirements-ocr.txt scripts/extract_attestation_questions.py scripts/check_attestation_questions.py tests/fixtures tests/test_attestation_extractor.py
git commit -m "feat: add attestation PDF extraction pipeline"
```

### Task 4: PostgreSQL schema and verified-bank persistence

**Files:**
- Modify: `storage.py:20-155`
- Modify: `storage.py:350-390`
- Create: `tests/test_attestation_storage.py`

- [ ] **Step 1: Write failing storage tests with async fakes**

```python
# tests/test_attestation_storage.py
import pytest
from unittest.mock import AsyncMock
from storage import Storage


@pytest.mark.asyncio
async def test_import_attestation_upserts_verified_but_preserves_admin_row():
    store = Storage("postgresql://unused")
    con = AsyncMock()
    con.fetchrow.return_value = {"verification_method": "admin"}
    imported = await store._upsert_attestation_row(con, {
        "section": "constitution", "qnum": 1, "question": "Текст",
        "choices": ["А", "Б"], "correct": [1], "source_page": 3,
        "source_hash": "new", "verification_method": "pdf_visual",
    })
    assert imported is False
    con.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_attestation_questions_decodes_json():
    store = Storage("postgresql://unused")
    store._fetch = AsyncMock(return_value=[{
        "id": 4, "section": "constitution", "section_title": "Конституція України",
        "qnum": 1, "question": "Текст", "choices": '["А", "Б"]',
        "correct": "[1]", "source_page": 3, "source_hash": "h",
        "verification_method": "pdf_visual",
    }])
    rows = await store.fetch_attestation_questions()
    assert rows[0]["choices"] == ["А", "Б"]
    assert rows[0]["correct"] == [1]
```

- [ ] **Step 2: Run and confirm missing methods**

Run: `python -m pytest tests/test_attestation_storage.py -q`

Expected: FAIL for missing attestation storage methods.

- [ ] **Step 3: Add idempotent schema creation**

Add to `Storage.init()`:

```sql
CREATE TABLE IF NOT EXISTS attestation_questions (
    id BIGSERIAL PRIMARY KEY,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    qnum INT NOT NULL,
    question TEXT NOT NULL,
    choices JSONB NOT NULL,
    correct JSONB NOT NULL,
    source_page INT NOT NULL,
    source_hash TEXT NOT NULL,
    verification_method TEXT NOT NULL,
    match_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    verified_by TEXT,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(section, qnum)
);
CREATE INDEX IF NOT EXISTS idx_attestation_questions_section
ON attestation_questions(section, qnum);
```

- [ ] **Step 4: Implement import/load and protect admin-approved rows**

Add `import_attestation_questions(items, force=False)`, `_upsert_attestation_row`, `fetch_attestation_questions`, and `attestation_counts`. Store JSON using `json.dumps(..., ensure_ascii=False)`. An existing row with `verification_method='admin'` returns `False` unless `force=True`. `fetch_attestation_questions` selects only fields accepted by `AttestationQuestion` (`id`, section/title, qnum, text, choices, correct, source page/hash, verification method, match evidence) and decodes every JSON field.

The upsert must use:

```sql
INSERT INTO attestation_questions
    (section, section_title, qnum, question, choices, correct, source_page,
     source_hash, verification_method, match_evidence, verified_by, verified_at, updated_at)
VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7,$8,$9,$10::jsonb,$11,now(),now())
ON CONFLICT (section, qnum) DO UPDATE SET
    section_title=EXCLUDED.section_title,
    question=EXCLUDED.question,
    choices=EXCLUDED.choices,
    correct=EXCLUDED.correct,
    source_page=EXCLUDED.source_page,
    source_hash=EXCLUDED.source_hash,
    verification_method=EXCLUDED.verification_method,
    match_evidence=EXCLUDED.match_evidence,
    verified_by=EXCLUDED.verified_by,
    verified_at=now(),
    updated_at=now()
```

- [ ] **Step 5: Add test type and metadata without breaking old statistics**

Migrate `tests` with `ALTER TABLE tests ADD COLUMN IF NOT EXISTS test_type TEXT NOT NULL DEFAULT 'standard'` and `ALTER TABLE tests ADD COLUMN IF NOT EXISTS meta JSONB NOT NULL DEFAULT '{}'::jsonb`. Change `save_test(..., test_type='standard', meta=None)` and make `stats()` filter `WHERE user_id=$1 AND test_type='standard'`. Add `attestation_stats(user_id)` filtered by `test_type='attestation'`.

- [ ] **Step 6: Run storage tests**

Run: `python -m pytest tests/test_attestation_storage.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit verified-bank persistence**

```powershell
git add storage.py tests/test_attestation_storage.py
git commit -m "feat: persist verified attestation questions"
```

### Task 5: Review queue persistence and approval transaction

**Files:**
- Modify: `storage.py`
- Modify: `tests/test_attestation_storage.py`

- [ ] **Step 1: Add failing approval tests**

```python
@pytest.mark.asyncio
async def test_approve_review_validates_then_moves_question_atomically():
    store = Storage("postgresql://unused")
    con = FakeReviewConnection(candidate=valid_review_candidate())
    store._upsert_attestation_row = AsyncMock(return_value=True)
    result = await store._approve_attestation_review(con, review_id=9, payload={
        "question": "Повне питання?", "choices": ["А", "Б"], "correct": [2]
    }, admin_id=123)
    assert result["status"] == "approved"
    assert con.transaction_entered
    inserted = store._upsert_attestation_row.await_args.args[1]
    assert inserted["verification_method"] == "admin"


@pytest.mark.asyncio
async def test_approve_review_rejects_empty_choice_without_changing_status():
    store = Storage("postgresql://unused")
    con = FakeReviewConnection(candidate=valid_review_candidate())
    store._upsert_attestation_row = AsyncMock(return_value=True)
    with pytest.raises(ValueError, match="порожній варіант"):
        await store._approve_attestation_review(con, 9, {
            "question": "Питання?", "choices": ["А", ""], "correct": [1]
        }, 123)
    assert con.status_updates == []
```

Define the helpers in the same test file before these tests:

```python
class FakeTransaction:
    def __init__(self, con):
        self.con = con
    async def __aenter__(self):
        self.con.transaction_entered = True
    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeReviewConnection:
    def __init__(self, candidate):
        self.candidate = candidate
        self.transaction_entered = False
        self.status_updates = []
    def transaction(self):
        return FakeTransaction(self)
    async def fetchrow(self, sql, *args):
        return self.candidate if "FOR UPDATE" in sql else None
    async def execute(self, sql, *args):
        if "UPDATE attestation_question_reviews" in sql:
            self.status_updates.append(args)
        return "UPDATE 1"


def valid_review_candidate():
    return {
        "id": 9,
        "section": "constitution",
        "section_title": "Конституція України",
        "qnum": 1,
        "extracted_question": "Повне питання?",
        "extracted_choices": ["А", "Б"],
        "proposed_correct": [2],
        "source_page": 3,
        "source_hash": "source-9",
        "issues": ["low_ocr_confidence"],
        "matches": [],
        "status": "needs_review",
    }
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_attestation_storage.py -q`

Expected: FAIL for missing review methods/schema.

- [ ] **Step 3: Add the review table**

```sql
CREATE TABLE IF NOT EXISTS attestation_question_reviews (
    id BIGSERIAL PRIMARY KEY,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    qnum INT NOT NULL,
    extracted_question TEXT NOT NULL,
    extracted_choices JSONB NOT NULL,
    proposed_correct JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_page INT NOT NULL,
    source_hash TEXT NOT NULL,
    issues JSONB NOT NULL DEFAULT '[]'::jsonb,
    matches JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL DEFAULT 'needs_review',
    resolved_by BIGINT,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(section, qnum, source_hash),
    CHECK (status IN ('needs_review', 'approved', 'rejected'))
);
```

- [ ] **Step 4: Implement review import, list, summary, approve, and reject**

Add these exact methods: `import_attestation_reviews`, `list_attestation_reviews(status, offset, limit)`, `get_attestation_review`, `attestation_review_summary`, `approve_attestation_review`, and `reject_attestation_review`. Approval uses one transaction, `SELECT ... FOR UPDATE`, domain validation, attestation upsert with `verification_method='admin'`, then status update. Rejection only updates status/resolver fields. Startup review import may refresh only rows still in `needs_review`; it must never revert an `approved` or `rejected` row. The summary combines review statuses with verified `match_evidence` so «підтверджено збігом із базою» is a real count rather than an estimate.

- [ ] **Step 5: Run storage tests**

Run: `python -m pytest tests/test_attestation_storage.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit review persistence**

```powershell
git add storage.py tests/test_attestation_storage.py
git commit -m "feat: add attestation review queue storage"
```

### Task 6: Runtime bank, catalog, access, and secure selection

**Files:**
- Modify: `app.py:248-266`
- Modify: `app.py:403-575`
- Create: `tests/test_attestation_service.py`

- [ ] **Step 1: Add failing access and catalog tests**

```python
# tests/test_attestation_service.py
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from app import AuthContext, MiniAppService, StartAttestationRequest, AnswerRequest
from attestation import AttestationBank, AttestationQuestion, SECTION_KEYS
from utils import now


def auth(tier="none", admin=False):
    user = {}
    if tier == "full":
        user = {"sub_tier": "full", "sub_infinite": True}
    elif tier == "trial_full":
        user = {"trial_end": now() + timedelta(days=1)}
    return AuthContext({}, user, 7, admin)


def question(section="constitution", number=1, qid=1):
    return AttestationQuestion(
        id=qid,
        section=section,
        section_title=SECTION_KEYS[section],
        qnum=number,
        question=f"Питання {section} {number}?",
        choices=["Варіант А", "Варіант Б"],
        correct=[2],
        source_page=3,
        source_hash=f"hash-{section}-{number}",
        verification_method="pdf_visual",
    )


class MemoryStore:
    def __init__(self):
        self.state = {}
        self.save_test = AsyncMock()
    async def get_ui(self, user_id):
        return {"state": self.state}
    async def set_state(self, user_id, state):
        self.state = state
    async def stats(self, user_id):
        return {"count": 0, "avg": 0.0, "last": None}
    async def get_setting(self, key, default=None):
        return default


@pytest.fixture
def fake_runtime():
    questions = []
    qid = 1
    for section in SECTION_KEYS:
        for number in range(1, 121):
            questions.append(question(section=section, number=number, qid=qid))
            qid += 1
    return SimpleNamespace(
        store=MemoryStore(),
        qb=SimpleNamespace(by_id={}, law_groups={}, ok_modules={}, law=[]),
        attestation_qb=AttestationBank(questions),
        admin_ids=set(), bot_token="", webapp_url="", allow_dev_auth=True,
        auth_max_age_seconds=0,
    )


def completed_attestation_state(section="constitution", mode="demo"):
    return {
        "mode": "attestation",
        "pending": [],
        "correct_count": 8,
        "total": 10,
        "answers": {str(number): number <= 8 for number in range(1, 11)},
        "chosen": {str(number): 0 for number in range(1, 11)},
        "started_at": now().isoformat(),
        "meta": {"bank": "attestation", "section": section, "selection_mode": mode, "access": "demo"},
    }


def test_catalog_exposes_four_sections_and_all(fake_runtime):
    service = MiniAppService(fake_runtime)
    catalog = service.serialize_attestation_catalog(auth())
    assert [item["key"] for item in catalog["sections"]] == [
        "constitution", "civil_service", "customs_code", "anti_corruption", "all"
    ]
    assert catalog["access"] == "demo"


@pytest.mark.asyncio
async def test_demo_cannot_request_random_or_part(fake_runtime):
    service = MiniAppService(fake_runtime)
    with pytest.raises(Exception) as exc:
        await service.start_attestation(auth(), StartAttestationRequest(section="constitution", mode="random"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_full_and_admin_may_request_full_modes(fake_runtime):
    service = MiniAppService(fake_runtime)
    assert len(service.select_attestation(auth("full"), "constitution", "random", 1)) == 50
    assert len(service.select_attestation(auth(admin=True), "constitution", "part", 1)) == 50
```

- [ ] **Step 2: Run and confirm missing runtime/service support**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: FAIL for missing runtime field and service methods.

- [ ] **Step 3: Extend runtime and add explicit access predicate**

```python
@dataclass
class RuntimeContext:
    store: Storage
    qb: QuestionBank
    attestation_qb: AttestationBank
    # existing fields remain unchanged


def has_attestation_full_access(auth: AuthContext) -> bool:
    return auth.is_admin or access_tier(auth.user) == "full"
```

- [ ] **Step 4: Add request model, catalog, and selection guard**

```python
class StartAttestationRequest(BaseModel):
    section: Literal["constitution", "civil_service", "customs_code", "anti_corruption", "all"]
    mode: Literal["part", "random", "demo"]
    part: int = 1


def select_attestation(self, auth, section, mode, part):
    full = has_attestation_full_access(auth)
    if not full and mode != "demo":
        require_http(403, "attestation_full_required", "Повний банк доступний за повною підпискою.")
    if full and mode == "demo":
        mode = "part"
        part = 1
    selected = self.runtime.attestation_qb.select(section, mode, part)
    if not selected:
        require_http(404, "attestation_selection_empty", "У вибраному наборі немає питань.")
    return selected
```

Catalog entries include `key`, `title`, `count`, `parts`, `demo_count`, `locked`, and top-level `access` (`full` or `demo`). For `all`, count is the sum of all four sections and demo_count is 40.

- [ ] **Step 5: Run service access tests**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: access/catalog tests PASS.

- [ ] **Step 6: Commit runtime catalog/access**

```powershell
git add app.py tests/test_attestation_service.py
git commit -m "feat: expose attestation catalog and access rules"
```

### Task 7: Attestation session resolver, immediate feedback, review, and stats

**Files:**
- Modify: `app.py:540-810`
- Modify: `app.py:960-1095`
- Modify: `storage.py:350-375`
- Modify: `tests/test_attestation_service.py`

- [ ] **Step 1: Add failing session tests**

```python
@pytest.mark.asyncio
async def test_attestation_answer_always_returns_feedback(fake_runtime):
    service = MiniAppService(fake_runtime)
    await service.start_attestation(auth(), StartAttestationRequest(section="constitution", mode="demo"))
    view = await service.answer(auth(), AnswerRequest(choice=0))
    assert view["mode"] == "attestation"
    assert view["screen"] == "feedback"
    assert any(option["status"] == "correct" for option in view["question"]["options"])


@pytest.mark.asyncio
async def test_attestation_finish_uses_separate_stats(fake_runtime):
    service = MiniAppService(fake_runtime)
    fake_runtime.store.save_test.reset_mock()
    state = completed_attestation_state(section="all", mode="demo")
    result = await service.finish_attestation(7, state)
    assert result["mode"] == "attestation_result"
    fake_runtime.store.save_test.assert_awaited_once()
    assert fake_runtime.store.save_test.await_args.kwargs["test_type"] == "attestation"
```

- [ ] **Step 2: Run and confirm session failures**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: new tests FAIL because `attestation` is not a session mode.

- [ ] **Step 3: Add a bank-aware question resolver**

```python
def resolve_question(self, state: dict[str, Any], qid: int):
    bank = str((state.get("meta") or {}).get("bank") or "questions")
    if bank == "attestation":
        return self.runtime.attestation_qb.by_id.get(int(qid))
    return self.qb.by_id.get(int(qid))
```

Replace direct `self.qb.by_id.get(...)` calls in active session, feedback, result review, and answer paths with `resolve_question(state, qid)`. Do not change question lookup in unrelated admin endpoints.

- [ ] **Step 4: Add state-aware session authorization**

```python
def ensure_session_access(self, auth, state):
    meta = dict(state.get("meta") or {})
    if meta.get("bank") == "attestation":
        if meta.get("access") == "full" and not has_attestation_full_access(auth):
            require_http(403, "attestation_full_required", "Повний доступ завершився.")
        return
    self.ensure_access(auth)
```

Use this in answer, feedback-next, session restore, and review endpoints. This allows demo after trial expiry but never allows a saved full session after full access expires.

- [ ] **Step 5: Implement `mode='attestation'`**

`start_attestation` writes `pending`, `feedback`, `correct_count`, `answers`, `chosen`, `started_at`, and `meta={bank:'attestation', section, selection_mode, access}`. `answer` stores the choice and always sets feedback. `feedback_next` clears feedback and finishes when pending is empty. `finish_attestation` saves `test_type='attestation'`, builds `attestation_result`, and retains both `wrong_qids` and the unchanged `meta` so review resolves the correct bank. Extend saved-view/result-review mode sets accordingly; returning from `attestation_review` must restore `attestation_result`, not `test_result`.

- [ ] **Step 6: Run all service/storage tests**

Run: `python -m pytest tests/test_attestation_service.py tests/test_attestation_storage.py -q`

Expected: all tests PASS and standard stats tests remain unchanged.

- [ ] **Step 7: Commit session support**

```powershell
git add app.py storage.py tests/test_attestation_service.py tests/test_attestation_storage.py
git commit -m "feat: run isolated attestation sessions"
```

### Task 8: Public attestation API and startup synchronization

**Files:**
- Modify: `app.py:1560-1610`
- Modify: `app.py:1635-1705`
- Modify: `tests/test_attestation_service.py`

- [ ] **Step 1: Add failing bootstrap/start contract tests**

Assert that bootstrap includes `catalog.attestation`, `/api/attestation/start` delegates the exact payload, and missing/empty bank returns code `attestation_bank_empty` rather than an internal error.

```python
@pytest.mark.asyncio
async def test_bootstrap_contains_attestation_catalog(fake_runtime):
    result = await MiniAppService(fake_runtime).bootstrap(auth())
    assert result["catalog"]["attestation"]["access"] == "demo"
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: bootstrap assertion FAILS.

- [ ] **Step 3: Load and sync verified/review JSON during startup**

After normal questions and `test_exam_questions.json` imports:

```python
verified_path = BASE_DIR / "attestation_questions.json"
reviews_path = BASE_DIR / "attestation_review_candidates.json"
if verified_path.exists():
    await store.import_attestation_questions(load_json_file(verified_path))
if reviews_path.exists():
    await store.import_attestation_reviews(load_json_file(reviews_path))
attestation_qb = AttestationBank(
    AttestationQuestion(**row) for row in await store.fetch_attestation_questions()
)
```

Pass `attestation_qb` into `RuntimeContext`. An empty attestation bank is allowed at process startup and reported cleanly by its API.

Add the loader once near the existing path helpers:

```python
def load_json_file(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]
```

- [ ] **Step 4: Add public start endpoint and bootstrap catalog**

```python
@app.post("/api/attestation/start")
async def api_attestation_start(
    payload: StartAttestationRequest,
    auth: AuthContext = Depends(get_auth_context),
    runtime: RuntimeContext = Depends(get_runtime),
):
    return await MiniAppService(runtime).start_attestation(auth, payload)
```

- [ ] **Step 5: Run service tests and compile Python**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: PASS.

Run: `python -m compileall -q app.py storage.py attestation.py attestation_quality.py scripts`

Expected: exit code 0.

- [ ] **Step 6: Commit API/startup support**

```powershell
git add app.py tests/test_attestation_service.py
git commit -m "feat: add attestation public API"
```

### Task 9: Admin review API

**Files:**
- Modify: `app.py:300-325`
- Modify: `app.py:1800-1935`
- Modify: `tests/test_attestation_service.py`

- [ ] **Step 1: Add failing admin tests**

Test non-admin 403 for every review route, paginated list response, invalid approval 400 with field errors, successful approval reloads `runtime.attestation_qb`, and rejection never changes the verified bank.

```python
@pytest.mark.asyncio
async def test_admin_approval_reloads_runtime_bank(fake_runtime):
    fake_runtime.store.approve_attestation_review = AsyncMock(return_value={"status": "approved"})
    existing = list(fake_runtime.attestation_qb.by_id.values())
    new_question = question(section="constitution", number=121, qid=9999)
    fake_runtime.store.fetch_attestation_questions = AsyncMock(
        return_value=[item.__dict__ for item in [*existing, new_question]]
    )
    service = MiniAppService(fake_runtime)
    before = len(fake_runtime.attestation_qb.by_id)
    await service.admin_approve_attestation_review(
        auth(admin=True), 9,
        AttestationReviewPatch(question="Повне питання?", choices=["А", "Б"], correct=[2]),
    )
    assert len(fake_runtime.attestation_qb.by_id) == before + 1
```

Extend the imports in `tests/test_attestation_service.py` with `AttestationReviewPatch` when its model is added.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: FAIL for missing request model/service methods.

- [ ] **Step 3: Add the review patch model**

```python
class AttestationReviewPatch(BaseModel):
    question: str
    choices: list[str]
    correct: list[int]
```

- [ ] **Step 4: Add admin routes**

Implement:

```text
GET  /api/admin/attestation/reviews?status=needs_review&offset=0&limit=20
GET  /api/admin/attestation/reviews/{review_id}
POST /api/admin/attestation/reviews/{review_id}/approve
POST /api/admin/attestation/reviews/{review_id}/reject
GET  /api/admin/attestation/summary
```

Every handler checks `auth.is_admin`. Approval calls storage, reloads the runtime bank from verified rows, and returns the approved record plus updated summary.

- [ ] **Step 5: Run service tests**

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: all admin tests PASS.

- [ ] **Step 6: Commit admin API**

```powershell
git add app.py tests/test_attestation_service.py
git commit -m "feat: add attestation review API"
```

### Task 10: User-facing attestation screens

**Files:**
- Create: `static/js/screens/attestation.js`
- Create: `tests/js/attestation.test.mjs`
- Create: `package.json`
- Modify: `static/js/app.js:1-250`
- Modify: `static/js/core/state.js`
- Modify: `static/js/screens/user.js:45-115`
- Modify: `static/js/screens/session.js`
- Modify: `static/index.html`

- [ ] **Step 1: Write pure frontend tests**

```javascript
// tests/js/attestation.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { partRows, startOptions } from "../../static/js/screens/attestation.js";

test("partRows makes 50-question ranges", () => {
  assert.deepEqual(partRows(121), [
    { part: 1, start: 1, end: 50 },
    { part: 2, start: 51, end: 100 },
    { part: 3, start: 101, end: 121 },
  ]);
});

test("demo exposes only the fixed demo action", () => {
  assert.deepEqual(startOptions({ access: "demo", key: "constitution", demo_count: 10 }), [
    { mode: "demo", count: 10, locked: false },
    { mode: "random", count: 50, locked: true },
  ]);
});
```

- [ ] **Step 2: Add Node test command and confirm failure**

```json
{
  "private": true,
  "type": "module",
  "scripts": { "test:js": "node --test tests/js/*.test.mjs" }
}
```

Run: `npm run test:js`

Expected: FAIL because `attestation.js` does not exist.

- [ ] **Step 3: Implement pure helpers and screens**

`attestation.js` exports `partRows`, `startOptions`, `renderAttestation`, `renderAttestationParts`, and `startAttestation`. Render five catalog cells. Full users see random 50 and part rows. Demo users see a primary demo button (10 or 40) and locked full actions that call `ctx.openPayment("full")`.

Use this API call:

```javascript
export async function startAttestation(ctx, section, mode, part = 1) {
  try {
    ctx.state.currentView = await ctx.api("/api/attestation/start", {
      method: "POST",
      body: { section, mode, part },
    });
    ctx.render();
  } catch (error) {
    ctx.setMessage("error", error.message);
  }
}
```

- [ ] **Step 4: Wire navigation and home tile**

Add screens `attestation` and `attestation-parts` to router/history cleanup. Add a home cell visible to every user:

```javascript
ctx.cell({
  title: "Перший етап атестації",
  subtitle: "4 розділи · демо або повний банк",
  icon: "✅",
  tint: "purple",
  screen: "attestation",
})
```

Extend `session.js` result action so both `test_result` and `attestation_result` may open wrong-answer review.

- [ ] **Step 5: Bump every changed ES-module query version consistently**

Update imports and `static/index.html` to one new value such as `v=20260731-attestation-01`; do not leave mixed versions for changed modules.

- [ ] **Step 6: Run frontend and Python tests**

Run: `npm run test:js`

Expected: `2 passed`.

Run: `python -m pytest tests/test_attestation_service.py -q`

Expected: PASS.

- [ ] **Step 7: Commit user UI**

```powershell
git add package.json tests/js/attestation.test.mjs static/js static/index.html
git commit -m "feat: add first-stage attestation screens"
```

### Task 11: Admin problem-review UI

**Files:**
- Create: `static/js/screens/admin-attestation.js`
- Modify: `static/js/screens/admin.js:1-75`
- Modify: `static/js/app.js`
- Modify: `static/styles/components.css`
- Modify: `static/styles/responsive.css`
- Modify: `tests/js/attestation.test.mjs`
- Modify: `static/index.html`

- [ ] **Step 1: Add failing view-model test for review reasons/differences**

```javascript
import { reviewReasonLabels, changedFields } from "../../static/js/screens/admin-attestation.js";

test("review model exposes OCR and answer differences", () => {
  assert.deepEqual(reviewReasonLabels(["low_ocr_confidence", "page_break_not_closed"]), [
    "Низька впевненість OCR",
    "Незавершене перенесення між сторінками",
  ]);
  assert.deepEqual(changedFields(
    { question: "Питання", choices: ["А", "Б"] },
    { question: "Питання?", choices: ["А", "В"] },
  ), ["question", "choices"]);
});
```

- [ ] **Step 2: Run and confirm missing admin module**

Run: `npm run test:js`

Expected: FAIL for missing exports/module.

- [ ] **Step 3: Implement list, summary, and editor**

The admin screen loads summary and 20 unresolved candidates. Each row shows section, number, PDF page, reason chips, and best match. Opening a row renders editable question textarea, one textarea per answer, a delete control for every answer, an «Додати відповідь» control, radio selection for the single correct answer, and Approve/Reject actions. Approval sends the complete payload; no partial PATCH is allowed.

```javascript
await ctx.api(`/api/admin/attestation/reviews/${review.id}/approve`, {
  method: "POST",
  body: { question, choices, correct: [selectedIndex] },
});
```

- [ ] **Step 4: Add admin hub tile and router integration**

Add `admin-attestation-reviews` to `ensureScreenData`, `render`, context loader, admin route guard, and Admin Hub with subtitle showing unresolved count when loaded.

- [ ] **Step 5: Add responsive diff/editor styles**

Use `.attestation-review`, `.attestation-review__source`, `.attestation-review__match`, `.attestation-review__choices`, and `.issue-chip`. Below 640px, stack source/match columns and keep action buttons full-width. Reuse existing CSS variables and focus styles.

- [ ] **Step 6: Run frontend tests**

Run: `npm run test:js`

Expected: all tests PASS.

- [ ] **Step 7: Commit admin UI**

```powershell
git add static/js static/styles static/index.html tests/js/attestation.test.mjs
git commit -m "feat: add admin attestation review screen"
```

### Task 12: Extract the real PDF and account for every question

**Files:**
- Create: `attestation_questions.json`
- Create: `attestation_review_candidates.json`
- Create: `data/attestation_quality_report.json`
- Source: `C:/Users/Lucky/Desktop/2048/додаток_до_наказу_3125_Перелік_питань.pdf`

- [ ] **Step 1: Install only offline OCR dependencies in the bundled Python environment**

```powershell
$ocrPython = 'C:\Users\Lucky\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $ocrPython -m pip install -r requirements-ocr.txt
```

Expected: EasyOCR, pypdf, Pillow and their required runtime packages install successfully.

- [ ] **Step 2: Run extraction with OCR cache**

```powershell
$ocrPython = 'C:\Users\Lucky\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $ocrPython scripts/extract_attestation_questions.py `
  --pdf 'C:\Users\Lucky\Desktop\2048\додаток_до_наказу_3125_Перелік_питань.pdf' `
  --output attestation_questions.json `
  --reviews attestation_review_candidates.json `
  --report data/attestation_quality_report.json `
  --ocr-cache tmp/attestation-ocr
```

Expected: 128 pages processed; physical pages 3-128 assigned to exactly four sections; three JSON files created.

- [ ] **Step 3: Verify pages 3-30 (Constitution)**

Render the source pages, compare every question, every answer line and bold correct choice to the candidate. Fix verified JSON only when the scan is unambiguous; otherwise add/update a `needs_review` record with page and reason. Confirm the report has no unaccounted number in this section.

- [ ] **Step 4: Verify pages 31-61 (Civil Service)**

Repeat the same full text/answer/correct-choice comparison for every row. Pay special attention to questions and answers crossing page 61 boundaries. Confirm every source number is verified or in review.

- [ ] **Step 5: Verify pages 62-92 (Customs Code)**

Repeat the same comparison. Cross-check exact matches against `questions_flat.json`, but keep the PDF wording whenever source and bank differ. Record every meaningful conflict instead of silently selecting one version.

- [ ] **Step 6: Verify pages 93-128 (Anti-corruption)**

Repeat the same comparison through the final page. Confirm no final answer is truncated by the end of the document.

- [ ] **Step 7: Run the final completeness checker**

Run: `python scripts/check_attestation_questions.py`

Expected: exit code 0; `unaccounted_gaps` is empty; each question number is either verified or represented once in the review queue; no duplicate section/number pair exists.

- [ ] **Step 8: Run domain validation over all verified questions**

Run: `python -m pytest tests/test_attestation.py tests/test_attestation_quality.py tests/test_attestation_extractor.py -q`

Expected: all tests PASS.

- [ ] **Step 9: Commit source-derived data and report**

```powershell
git add attestation_questions.json attestation_review_candidates.json data/attestation_quality_report.json
git commit -m "data: add verified first-stage attestation bank"
```

### Task 13: End-to-end regression and security verification

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Run all Python tests**

Run: `python -m pytest -q`

Expected: all tests PASS, no skipped attestation tests.

- [ ] **Step 2: Run all frontend tests**

Run: `npm run test:js`

Expected: all tests PASS.

- [ ] **Step 3: Run the existing Rijndael sanity suite**

Run: `python -m unittest discover -s rijndael/tests -p "*tests.py"`

Expected: existing sanity tests PASS.

- [ ] **Step 4: Compile Python and validate generated data**

Run: `python -m compileall -q app.py storage.py attestation.py attestation_quality.py scripts`

Expected: exit code 0.

Run: `python scripts/check_attestation_questions.py`

Expected: exit code 0 and no unaccounted gaps.

- [ ] **Step 5: Start the app against a disposable PostgreSQL database**

Set `DATABASE_URL`, `ALLOW_DEV_AUTH=1`, `QUESTIONS_AUTO_IMPORT=1`, then run `uvicorn app:app --host 127.0.0.1 --port 8000`. Expected: startup creates both attestation tables, imports verified/review JSON, and `/healthz` returns `{"ok":true}`.

- [ ] **Step 6: Verify demo security with a no-access debug user**

Check that catalog reports demo counts 10/10/10/10 and 40 for all. Start each demo and verify only the fixed ids are returned. Direct requests for `part` and `random` must return 403. Editing client state must not bypass the server check.

- [ ] **Step 7: Verify full/admin behavior**

Grant `full` to one debug user and verify parts of 50, last partial part, random 50 without duplicates, immediate feedback, result, restore, and wrong-answer review. Verify an admin receives the same full access regardless of subscription.

- [ ] **Step 8: Verify review workflow**

Open «Проблемні питання», edit one candidate, prove invalid content cannot be approved, then approve a valid correction and confirm it immediately appears in the catalog/test bank. Reject a second candidate and confirm it never appears to users.

- [ ] **Step 9: Verify statistics isolation**

Complete one standard test and one attestation test. `/api/stats` must reflect only the standard test; attestation storage query must contain only the attestation attempt with section/mode metadata.

- [ ] **Step 10: Inspect desktop and narrow mobile layouts**

Check Home, five attestation sections, parts, demo paywall, question/feedback/result, admin summary, review list and editor at desktop width and at 390px. Expected: no clipping, horizontal overflow, hidden choices, or unreachable approval controls.

- [ ] **Step 11: Review final diff and commit only real fixes**

Run: `git diff --check`

Expected: no whitespace errors.

If this task required fixes, stage only those files and commit:

```powershell
git commit -m "fix: harden attestation end-to-end flow"
```

If no files changed, do not create an empty commit.
