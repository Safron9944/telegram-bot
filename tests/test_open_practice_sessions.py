from pathlib import Path
from types import SimpleNamespace
import unittest

from app import AuthContext, MiniAppService, StartAttestationRequest
from questions import Q, QuestionBank


ROOT = Path(__file__).resolve().parents[1]


class PracticeStore:
    def __init__(self):
        self.state = {}

    async def set_state(self, user_id, state):
        self.state = state

    async def get_ui(self, user_id):
        return {"state": self.state}


def practice_question(qid: int, number: int) -> Q:
    return Q(
        id=qid,
        section="Державна мова",
        topic="Написання тексту на визначену тему",
        ok=None,
        level=None,
        qnum=number,
        question=f"Практичне завдання {number}",
        choices=[],
        correct=[],
        correct_texts=[],
        shuffle_choices=False,
        practice_answer=f"Зразок відповіді {number}",
    )


class OpenPracticeSessionTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        bank = QuestionBank("unused.json")
        bank.register_attestation_bank(
            "ukrainian-language",
            "Державна мова",
            [practice_question(20_001_938, 1), practice_question(20_001_939, 2)],
            source_id="bundled-ukrainian-language-test",
            manual_grant_section_key="ukrainian_language",
        )
        store = PracticeStore()
        service = MiniAppService(SimpleNamespace(qb=bank, store=store))
        auth = AuthContext({}, {"section_access": ["ukrainian_language"]}, 42, False)
        return service, store, auth

    async def test_starts_reveals_advances_and_finishes_open_practice(self):
        service, store, auth = self.make_service()

        first = await service.start_attestation(
            auth,
            "ukrainian-language",
            StartAttestationRequest(section="Написання тексту на визначену тему", block="1-2"),
        )
        self.assertEqual("open-practice", first["screen"])
        self.assertEqual({"current": 1, "total": 2}, first["progress"])
        self.assertIsNone(first["question"]["sample_answer"])

        restored = await service.saved_view(auth)
        self.assertEqual("open-practice", restored["screen"])

        revealed = await service.reveal_open_practice_answer(auth)
        self.assertEqual("Зразок відповіді 1", revealed["question"]["sample_answer"])

        second = await service.next_open_practice(auth)
        self.assertEqual({"current": 2, "total": 2}, second["progress"])
        self.assertIsNone(second["question"]["sample_answer"])

        result = await service.next_open_practice(auth)
        self.assertEqual("open-practice-result", result["screen"])
        self.assertEqual({"title": "Практику завершено", "completed": 2, "total": 2}, result["summary"])
        self.assertEqual({}, store.state)

    def test_bundled_bank_contains_both_open_practice_sections(self):
        bank = QuestionBank("unused.json")
        loaded = bank.load_bundled_attestation_bank(
            str(ROOT / "data" / "ukrainian_language_questions"),
            slug="ukrainian-language",
            title="Державна мова",
            source_id="bundled-ukrainian-language-3.8.26",
            id_offset=20_000_000,
            manual_grant_section_key="ukrainian_language",
        )

        practice = [bank.by_id[qid] for qid in loaded.qids if bank.by_id[qid].is_open_practice]
        self.assertEqual(124, len(practice))
        self.assertEqual(11, sum(q.topic == "Написання тексту на визначену тему" for q in practice))
        self.assertEqual(113, sum(q.topic == "Говоріння" for q in practice))


class OpenPracticeAssetsTests(unittest.TestCase):
    def test_frontend_wires_open_practice_actions(self):
        session = (ROOT / "static" / "js" / "screens" / "session.js").read_text(encoding="utf-8")
        user = (ROOT / "static" / "js" / "screens" / "user.js").read_text(encoding="utf-8")
        server = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("Показати зразок відповіді", session)
        self.assertIn("/api/session/open-practice/reveal", session)
        self.assertIn("/api/session/open-practice/next", session)
        self.assertIn('screen === "open-practice"', session)
        self.assertIn("section.practice", user)
        self.assertIn('@app.post("/api/session/open-practice/reveal")', server)


if __name__ == "__main__":
    unittest.main()
