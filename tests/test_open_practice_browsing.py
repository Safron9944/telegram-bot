from pathlib import Path
from types import SimpleNamespace
import unittest

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


def practice_question(qid: int, number: int, *, topic: str = "Написання тексту на визначену тему") -> Q:
    prefix = "XC" if topic.startswith("Написання") else "C"
    question = f"{prefix}. {number}. Тема {number}.\n\nІнструкція {number}" if prefix == "XC" else f"{prefix}. {number}. Тема {number}."
    return Q(
        id=qid,
        section="Державна мова",
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
        section="Державна мова",
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
            "ukrainian-language",
            "Державна мова",
            [
                practice_question(20_001_938, 1),
                practice_question(20_001_939, 2, topic="Говоріння"),
                regular_question(20_001_940),
            ],
            source_id="bundled-ukrainian-language-test",
            manual_grant_section_key="ukrainian_language",
        )
        store = PracticeStore()
        service = MiniAppService(SimpleNamespace(qb=bank, store=store))
        granted = AuthContext({}, {"section_access": ["ukrainian_language"]}, 42, False)
        return service, store, granted

    async def test_returns_topic_detail_without_creating_a_session(self):
        service, store, auth = self.make_service()

        detail = await service.attestation_practice_detail(auth, "ukrainian-language", 20_001_938)

        self.assertEqual("browse", detail["mode"])
        self.assertEqual("open-practice-detail", detail["screen"])
        self.assertEqual("Написання тексту на визначену тему", detail["header"])
        self.assertEqual("Тема 1.", detail["item"]["title"])
        self.assertEqual("Інструкція 1", detail["item"]["question"])
        self.assertEqual("Зразок відповіді 1", detail["item"]["sample_answer"])
        self.assertEqual({"existing": "unchanged"}, store.state)
        self.assertEqual(0, store.set_calls)

    async def test_keeps_single_line_speaking_task_as_detail_text(self):
        service, _, auth = self.make_service()
        detail = await service.attestation_practice_detail(auth, "ukrainian-language", 20_001_939)
        self.assertEqual("Тема 2.", detail["item"]["question"])

    async def test_requires_explicit_admin_grant(self):
        service, _, _ = self.make_service()
        denied = AuthContext({}, {"section_access": [], "access_tier": "full"}, 43, False)
        with self.assertRaises(HTTPException) as raised:
            await service.attestation_practice_detail(denied, "ukrainian-language", 20_001_938)
        self.assertEqual(403, raised.exception.status_code)
        self.assertEqual("protected_materials_required", raised.exception.detail["code"])

    async def test_rejects_non_practice_or_foreign_topic(self):
        service, _, auth = self.make_service()
        for question_id in (20_001_940, 99_999_999):
            with self.subTest(question_id=question_id), self.assertRaises(HTTPException) as raised:
                await service.attestation_practice_detail(auth, "ukrainian-language", question_id)
            self.assertEqual(404, raised.exception.status_code)

    def test_bundled_bank_exposes_all_topics_by_visible_name(self):
        bank = QuestionBank("unused.json")
        loaded = bank.load_bundled_attestation_bank(
            str(ROOT / "data" / "ukrainian_language_questions"),
            slug="ukrainian-language",
            title="Державна мова",
            source_id="bundled-ukrainian-language-3.8.26",
            id_offset=20_000_000,
            manual_grant_section_key="ukrainian_language",
        )

        sections = {item["title"]: item for item in bank.attestation_sections(loaded.slug)}
        writing = sections["Написання тексту на визначену тему"]
        speaking = sections["Говоріння"]
        self.assertEqual(11, len(writing["items"]))
        self.assertEqual(113, len(speaking["items"]))
        self.assertEqual([], writing["blocks"])
        self.assertEqual([], speaking["blocks"])
        self.assertEqual("Мистецтво спілкування.", writing["items"][0]["title"])
        self.assertFalse(writing["items"][0]["title"].startswith("XC."))
        self.assertFalse(speaking["items"][0]["title"].startswith("C."))


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
