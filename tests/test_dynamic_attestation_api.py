import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from app import AuthContext, MiniAppService, StartAttestationRequest
from questions import Q, QuestionBank


class StateStore:
    def __init__(self):
        self.states = {}

    async def set_state(self, user_id, state):
        self.states[user_id] = state

    async def get_ui(self, user_id):
        return {"state": self.states.get(user_id, {})}


def question(qid, number):
    return Q(qid, "Stage 2", "Topic", None, None, number, f"Q{number}", ["A", "B", "C", "D"], [1], ["A"])


class DynamicAttestationApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        bank = QuestionBank("unused.json")
        bank.register_attestation_bank("stage-2", "Stage 2", [question(index, index) for index in range(1, 53)])
        self.store = StateStore()
        self.service = MiniAppService(SimpleNamespace(qb=bank, store=self.store))
        self.auth = AuthContext({}, {"sub_infinite": 1, "sub_tier": "cases"}, 77, False)

    async def test_starts_dynamic_partial_block(self):
        result = await self.service.start_attestation(
            self.auth, "stage-2", StartAttestationRequest(section="Topic", block="51-52")
        )
        self.assertEqual(2, result["progress"]["total"])
        self.assertEqual("stage-2", self.store.states[77]["meta"]["bank_slug"])

    async def test_rejects_unknown_bank_or_block(self):
        for slug, block, code in (
            ("missing", "1-50", "attestation_bank_not_found"),
            ("stage-2", "bad", "attestation_block_not_found"),
        ):
            with self.subTest(code=code), self.assertRaises(HTTPException) as raised:
                await self.service.start_attestation(
                    self.auth, slug, StartAttestationRequest(section="Topic", block=block)
                )
            self.assertEqual(code, raised.exception.detail["code"])

    async def test_trial_cannot_start_dynamic_bank(self):
        trial = AuthContext({}, {"trial_end": __import__("utils").now()}, 78, False)
        with self.assertRaises(HTTPException) as raised:
            await self.service.start_attestation(
                trial, "stage-2", StartAttestationRequest(section="Topic", block="1-50")
            )
        self.assertEqual("attestation_access_required", raised.exception.detail["code"])


if __name__ == "__main__":
    unittest.main()
