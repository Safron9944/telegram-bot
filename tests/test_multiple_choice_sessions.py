import unittest
from types import SimpleNamespace

from app import AnswerRequest, AuthContext, MiniAppService, build_option_review, serialize_question
from questions import Q, QuestionBank


def multiple_question() -> Q:
    return Q(
        101,
        "Державна мова",
        "Лексика",
        None,
        None,
        1,
        "Оберіть три правильні варіанти.",
        ["А", "Б", "В", "Г"],
        [1, 2, 3],
        ["А", "Б", "В"],
        False,
    )


class SessionStore:
    def __init__(self, state):
        self.state = state
        self.wrong = []

    async def get_ui(self, user_id):
        return {"state": self.state}

    async def set_state(self, user_id, state):
        self.state = state

    async def bump_wrong(self, user_id, qid):
        self.wrong.append((user_id, qid))


class MultipleChoiceSessionTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self):
        qb = QuestionBank("unused.json")
        question = multiple_question()
        qb.register_attestation_bank(
            "ukrainian-language",
            "Державна мова",
            [question],
            source_id="bundled-ukrainian-language-test",
            manual_grant_section_key="ukrainian_language",
        )
        state = {
            "mode": "learn",
            "header": "Державна мова",
            "pending": [question.id],
            "skipped": [],
            "phase": "pending",
            "feedback": None,
            "current_qid": question.id,
            "correct_count": 0,
            "total": 1,
            "answers": {},
            "choice_orders": {},
            "meta": {
                "kind": "attestation",
                "bank_slug": "ukrainian-language",
            },
        }
        store = SessionStore(state)
        service = MiniAppService(SimpleNamespace(qb=qb, store=store))
        auth = AuthContext({}, {"section_access": ["ukrainian_language"]}, 42, False)
        return service, store, auth, question

    async def test_exact_selected_set_is_accepted(self):
        service, store, auth, _ = self.make_service()

        result = await service.answer(auth, AnswerRequest(choices=[2, 0, 1]))

        self.assertEqual("result", result["screen"])
        self.assertEqual(1, result["summary"]["correct"])
        self.assertEqual([], store.wrong)

    async def test_wrong_selected_set_is_shown_in_feedback(self):
        service, store, auth, _ = self.make_service()

        result = await service.answer(auth, AnswerRequest(choices=[0, 1, 3]))

        self.assertEqual("feedback", result["screen"])
        self.assertEqual(["correct", "correct", "correct", "chosen"], [item["status"] for item in result["question"]["options"]])
        self.assertEqual([(42, 101)], store.wrong)

    def test_serialized_question_declares_required_choice_count(self):
        payload = serialize_question(multiple_question())

        self.assertTrue(payload["multiple"])
        self.assertEqual(3, payload["required_choices"])

    def test_option_review_accepts_legacy_single_and_new_list_values(self):
        question = multiple_question()

        multiple = build_option_review(question, [0, 3])
        legacy = build_option_review(question, 3)

        self.assertEqual(["correct", "correct", "correct", "chosen"], [item["status"] for item in multiple])
        self.assertEqual("chosen", legacy[3]["status"])


if __name__ == "__main__":
    unittest.main()
