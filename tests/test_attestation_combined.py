import json
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from app import AnswerRequest, AuthContext, MiniAppService, StartAttestationRequest
from questions import Q, QuestionBank


ROOT = Path(__file__).resolve().parents[1]


def make_bank() -> QuestionBank:
    bank = QuestionBank("unused.json")
    questions = []
    for topic_index in range(4):
        for qnum in range(1, 51):
            qid = topic_index * 100 + qnum
            questions.append(Q(
                id=qid,
                section="Атестація — 2 етап",
                topic=f"Розділ {topic_index + 1}",
                ok=None,
                level=None,
                qnum=qnum,
                question=f"Питання {qid}?",
                choices=["А", "Б", "В", "Г"],
                correct=[1],
                correct_texts=["А"],
                shuffle_choices=False,
            ))
    bank.register_attestation_bank(
        "stage-2",
        "Атестація — 2 етап",
        questions,
        source_id="apk-stage-2",
        db_id=7,
    )
    return bank


class StateStore:
    def __init__(self, settings=""):
        self.settings = settings
        self.states = {}
        self.saved_tests = []

    async def get_setting(self, key, default=None):
        return self.settings or default

    async def set_state(self, user_id, state):
        self.states[user_id] = state

    async def get_ui(self, user_id):
        return {"state": self.states.get(user_id, {})}

    async def save_test(self, *args):
        self.saved_tests.append(args)


class AttestationCombinedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bank = make_bank()

    def test_combined_test_is_balanced_across_four_sections(self):
        qids = self.bank.attestation_combined_test_qids("stage-2", 100)
        counts = Counter(self.bank.by_id[qid].topic for qid in qids)
        self.assertEqual(100, len(qids))
        self.assertEqual({25}, set(counts.values()))

    async def test_admin_question_count_is_used_per_bank(self):
        settings = json.dumps({"attestation:7": {"combined_test_question_count": 80}})
        store = StateStore(settings)
        service = MiniAppService(SimpleNamespace(store=store, qb=self.bank))
        auth = AuthContext({"id": 1}, {"user_id": 1}, 1, True)

        await service.start_attestation(
            auth,
            "stage-2",
            StartAttestationRequest(section="", block="combined"),
        )

        state = store.states[1]
        qids = state["pending"]
        counts = Counter(self.bank.by_id[qid].topic for qid in qids)
        self.assertEqual("test", state["mode"])
        self.assertEqual(80, len(qids))
        self.assertEqual({20}, set(counts.values()))

    async def test_answers_are_hidden_until_the_final_result(self):
        settings = json.dumps({"attestation:7": {"combined_test_question_count": 4}})
        store = StateStore(settings)
        service = MiniAppService(SimpleNamespace(store=store, qb=self.bank))
        auth = AuthContext(
            {"id": 3},
            {"user_id": 3, "sub_infinite": True, "sub_tier": "full"},
            3,
            False,
        )

        view = await service.start_attestation(
            auth,
            "stage-2",
            StartAttestationRequest(section="", block="combined"),
        )
        self.assertEqual(("test", "question"), (view["mode"], view["screen"]))

        view = await service.answer(auth, AnswerRequest(choice=1))
        self.assertEqual("question", view["screen"])
        self.assertIsNone(store.states[3]["feedback"])

        for _ in range(3):
            view = await service.answer(auth, AnswerRequest(choice=0))

        self.assertEqual(("test_result", "result"), (view["mode"], view["screen"]))
        self.assertEqual(3, view["summary"]["correct"])
        self.assertEqual(4, view["summary"]["total"])
        self.assertEqual(1, view["wrong_count"])
        self.assertEqual(1, len(store.saved_tests))

    async def test_hidden_combined_test_is_rejected_for_user(self):
        settings = json.dumps({"attestation:7": {"combined_test_enabled": False}})
        store = SimpleNamespace(get_setting=AsyncMock(return_value=settings))
        service = MiniAppService(SimpleNamespace(store=store, qb=self.bank))
        auth = AuthContext(
            {"id": 2},
            {"user_id": 2, "sub_infinite": True, "sub_tier": "full"},
            2,
            False,
        )

        with self.assertRaisesRegex(HTTPException, "Загальний тест вимкнено"):
            await service.start_attestation(
                auth,
                "stage-2",
                StartAttestationRequest(section="", block="combined"),
            )

    def test_user_and_admin_ui_expose_combined_test_controls(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        admin = (ROOT / "static/js/admin_sections.js").read_text(encoding="utf-8")
        self.assertIn('id="attestation-combined-start"', user)
        self.assertIn('header: "Загальний тест"', user)
        self.assertIn('class="cell cell--accent"', user)
        self.assertIn('id="admin-section-test-count"', admin)
        self.assertIn('id="admin-section-test-enabled"', admin)
        self.assertIn("attestation_combined_tests", user)


if __name__ == "__main__":
    unittest.main()
