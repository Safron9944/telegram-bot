import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

from fastapi import HTTPException

from app import AuthContext, MiniAppService
from questions import Q, QuestionBank


ROOT = Path(__file__).resolve().parents[1]


class PracticeStore:
    def __init__(self):
        self.state = {"existing": "unchanged"}
        self.set_calls = 0

    async def set_state(self, user_id, state):
        self.set_calls += 1
        self.state = state


def practice_question(
    qid: int,
    number: int,
    *,
    topic: str = "Практичні завдання",
    instructions: bool = True,
) -> Q:
    question = f"TASK. {number}. Тема {number}."
    if instructions:
        question += f"\n\nІнструкція {number}"
    return Q(
        id=qid,
        section="Навчальні матеріали",
        topic=topic,
        ok=None,
        level=None,
        qnum=number,
        question=question,
        choices=[],
        correct=[],
        correct_texts=[],
        shuffle_choices=False,
        practice_answer=f"Зразок відповіді {number}",
    )


def regular_question(qid: int) -> Q:
    return Q(
        id=qid,
        section="Навчальні матеріали",
        topic="Тестові питання",
        ok=None,
        level=None,
        qnum=1,
        question="Тестове питання",
        choices=["А", "Б"],
        correct=[1],
        correct_texts=["А"],
    )


class OpenPracticeBrowsingTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        bank = QuestionBank("unused.json")
        bank.register_attestation_bank(
            "practice-bank",
            "Навчальні матеріали",
            [
                practice_question(20_001_938, 1),
                practice_question(20_001_939, 2, instructions=False),
                regular_question(20_001_940),
                practice_question(20_001_941, 3, topic="Приклади", instructions=False),
                practice_question(20_001_942, 4, topic="Приклади"),
            ],
            source_id="bundled-practice-test",
            manual_grant_section_key="practice_materials",
        )
        store = PracticeStore()
        service = MiniAppService(SimpleNamespace(qb=bank, store=store))
        granted = AuthContext({}, {"section_access": ["practice_materials"]}, 42, False)
        return service, store, granted

    async def test_returns_topic_detail_without_creating_a_session(self):
        service, store, auth = self.make_service()

        detail = await service.attestation_practice_detail(auth, "practice-bank", 20_001_938)

        self.assertEqual("browse", detail["mode"])
        self.assertEqual("open-practice-detail", detail["screen"])
        self.assertEqual("Практичні завдання", detail["header"])
        self.assertEqual("Тема 1.", detail["item"]["title"])
        self.assertEqual("Інструкція 1", detail["item"]["question"])
        self.assertEqual("Зразок відповіді 1", detail["item"]["sample_answer"])
        self.assertEqual({"existing": "unchanged"}, store.state)
        self.assertEqual(0, store.set_calls)

    async def test_single_line_practice_shows_topic_only_once(self):
        service, _, auth = self.make_service()
        for question_id, title in ((20_001_939, "Тема 2."), (20_001_941, "Тема 3.")):
            with self.subTest(question_id=question_id):
                detail = await service.attestation_practice_detail(auth, "practice-bank", question_id)
                self.assertEqual(title, detail["item"]["title"])
                self.assertEqual("", detail["item"]["question"])

    async def test_topic_keeps_its_instructions(self):
        service, _, auth = self.make_service()
        detail = await service.attestation_practice_detail(auth, "practice-bank", 20_001_942)
        self.assertEqual("Тема 4.", detail["item"]["title"])
        self.assertEqual("Інструкція 4", detail["item"]["question"])

    async def test_requires_explicit_admin_grant(self):
        service, _, _ = self.make_service()
        denied = AuthContext({}, {"section_access": [], "access_tier": "full"}, 43, False)
        with self.assertRaises(HTTPException) as raised:
            await service.attestation_practice_detail(denied, "practice-bank", 20_001_938)
        self.assertEqual(403, raised.exception.status_code)
        self.assertEqual("protected_materials_required", raised.exception.detail["code"])

    async def test_rejects_non_practice_or_foreign_topic(self):
        service, _, auth = self.make_service()
        for question_id in (20_001_940, 99_999_999):
            with self.subTest(question_id=question_id), self.assertRaises(HTTPException) as raised:
                await service.attestation_practice_detail(auth, "practice-bank", question_id)
            self.assertEqual(404, raised.exception.status_code)

    def test_bundled_bank_separates_practice_topics_from_test_blocks(self):
        bank = QuestionBank("unused.json")
        questions = [
            {
                "id": 1,
                "section": "Практичні завдання",
                "type": "open_answer",
                "question": "TASK. 1. Перша тема.\n\nІнструкція",
                "sample_answer": "Перший зразок відповіді",
            },
            {
                "id": 2,
                "section": "Практичні завдання",
                "type": "open_answer",
                "question": "TASK. 2. Друга тема.",
                "sample_answer": "Другий зразок відповіді",
            },
            {
                "id": 3,
                "section": "Тестові питання",
                "question": "Тестове питання?",
                "options": ["А", "Б"],
                "correct": [1],
            },
        ]
        with patch("questions.Path.open", mock_open(read_data=json.dumps(questions, ensure_ascii=False))):
            loaded = bank.load_bundled_attestation_bank(
                "practice.json",
                slug="practice-bank",
                title="Навчальні матеріали",
                source_id="bundled-practice-test",
                id_offset=20_000_000,
                manual_grant_section_key="practice_materials",
            )

        sections = {item["title"]: item for item in bank.attestation_sections(loaded.slug)}
        self.assertEqual({"Практичні завдання", "Тестові питання"}, set(sections))
        practice = sections["Практичні завдання"]
        self.assertTrue(practice["practice"])
        self.assertEqual(2, practice["count"])
        self.assertEqual([], practice["blocks"])
        self.assertEqual(
            [
                {"id": 20_000_001, "title": "Перша тема."},
                {"id": 20_000_002, "title": "Друга тема."},
            ],
            practice["items"],
        )
        tests = sections["Тестові питання"]
        self.assertFalse(tests["practice"])
        self.assertEqual(1, tests["count"])
        self.assertEqual([], tests["items"])
        self.assertEqual([{"key": "1-1", "title": "1-1"}], tests["blocks"])
        self.assertEqual([20_000_003], bank.attestation_combined_test_qids(loaded.slug, 3))


class OpenPracticeAssetsTests(unittest.TestCase):
    def test_frontend_wires_read_only_topic_catalog(self):
        session = (ROOT / "static" / "js" / "screens" / "session.js").read_text(encoding="utf-8")
        user = (ROOT / "static" / "js" / "screens" / "user.js").read_text(encoding="utf-8")
        server = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("section.items", user)
        self.assertIn("оберіть тему для перегляду", user)
        self.assertIn("/practice/${item.id}", user)
        self.assertIn('screen === "open-practice-detail"', session)
        self.assertIn("Повернутися до списку тем", session)
        self.assertIn('@app.get("/api/attestation/{bank_slug}/practice/{question_id}")', server)
        self.assertNotIn("/api/session/open-practice/", session)
        self.assertNotIn('@app.post("/api/session/open-practice/', server)


if __name__ == "__main__":
    unittest.main()
