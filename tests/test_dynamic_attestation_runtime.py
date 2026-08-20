from types import SimpleNamespace
import asyncio
import unittest

from app import AuthContext, MiniAppService
from questions import Q, QuestionBank


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


class LegacyQuestionStore:
    async def fetch_questions(self):
        return [{
            "id": 101,
            "section": "Атестація посадових осіб — 1 етап",
            "topic": "Старий розділ",
            "question": "Старе питання?",
            "choices": ["A", "B"],
            "correct": [1],
            "correct_texts": ["A"],
        }]


class PausedPublishedStore(PublishedStore):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def list_published_attestation_banks(self):
        self.started.set()
        await self.release.wait()
        return await super().list_published_attestation_banks()


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

        self.assertEqual(["stage-2"], [item["slug"] for item in catalog["attestation_banks"]])
        self.assertFalse(catalog["attestation_banks"][0]["system"])
        self.assertEqual("Новий розділ", catalog["attestation_banks"][0]["sections"][0]["title"])

    async def test_catalog_has_no_legacy_system_bank(self):
        bank = QuestionBank("unused.json")
        await bank.load_published_attestation_banks(PublishedStore())
        runtime = SimpleNamespace(qb=bank, store=None)
        auth = AuthContext({}, {"ok_modules": [], "ok_last_levels": []}, 1, False)

        catalog = MiniAppService(runtime).serialize_catalog(auth)

        self.assertEqual(["stage-2"], [item["slug"] for item in catalog["attestation_banks"]])
        self.assertNotIn("attestation_stage_1", catalog)

    async def test_legacy_questions_in_database_are_ignored(self):
        bank = QuestionBank("unused.json")
        await bank.load_from_db(LegacyQuestionStore())

        self.assertEqual({}, bank.by_id)
        self.assertEqual({}, bank.attestation_banks)

    async def test_database_reload_preserves_other_bundled_banks(self):
        bank = QuestionBank("unused.json")
        question = Q(
            20_000_001, "Державна мова", "Тема", None, None, 1,
            "Питання?", ["A", "B"], [1], ["A"],
        )
        bank.register_attestation_bank(
            "ukrainian-language",
            "Державна мова",
            [question],
            source_id="bundled-ukrainian-language-3.8.26",
            manual_grant_section_key="ukrainian_language",
        )

        await bank.load_published_attestation_banks(PublishedStore())

        self.assertIn("ukrainian-language", bank.attestation_banks)
        self.assertIn(20_000_001, bank.by_id)

    async def test_reload_keeps_current_catalog_available_while_fetching(self):
        bank = QuestionBank("unused.json")
        bank.register_attestation_bank("existing", "Existing", [], published=True)
        bank.attestation_banks["existing"].qids = [123]
        store = PausedPublishedStore()

        reload_task = asyncio.create_task(bank.load_published_attestation_banks(store))
        await store.started.wait()

        self.assertIn("existing", bank.attestation_banks)

        store.release.set()
        await reload_task
        self.assertNotIn("existing", bank.attestation_banks)
        self.assertIn("stage-2", bank.attestation_banks)


if __name__ == "__main__":
    unittest.main()
