from types import SimpleNamespace
import unittest

from app import AuthContext, MiniAppService
from questions import QuestionBank


class PublishedStore:
    async def list_published_attestation_banks(self):
        return [{
            "slug": "stage-2",
            "title": "Атестація — 2 етап",
            "source_id": "testms2.enc",
            "questions": [{
                "id": 9001,
                "source_key": "s:q1",
                "qnum": 1,
                "topic": "Новий розділ",
                "question": "Питання?",
                "choices": ["A", "B", "C", "D"],
                "correct": [2],
                "correct_texts": ["B"],
                "shuffle_choices": False,
            }],
        }]


class DynamicAttestationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_published_database_banks(self):
        bank = QuestionBank("unused.json")
        await bank.load_published_attestation_banks(PublishedStore())

        self.assertEqual("Атестація — 2 етап", bank.attestation_banks["stage-2"].title)
        self.assertEqual([-1_000_009_001], bank.attestation_banks["stage-2"].qids)
        self.assertFalse(bank.by_id[-1_000_009_001].shuffle_choices)

    async def test_catalog_serializes_all_published_banks(self):
        bank = QuestionBank("unused.json")
        await bank.load_published_attestation_banks(PublishedStore())
        runtime = SimpleNamespace(qb=bank, store=None)
        auth = AuthContext({}, {"ok_modules": [], "ok_last_levels": []}, 1, False)

        catalog = MiniAppService(runtime).serialize_catalog(auth)

        self.assertEqual(["stage-1", "stage-2"], [item["slug"] for item in catalog["attestation_banks"]])
        self.assertTrue(catalog["attestation_banks"][0]["system"])
        self.assertFalse(catalog["attestation_banks"][1]["system"])
        self.assertEqual("Новий розділ", catalog["attestation_banks"][1]["sections"][0]["title"])


if __name__ == "__main__":
    unittest.main()
