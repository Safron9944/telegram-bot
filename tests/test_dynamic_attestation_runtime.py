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


class ConflictingStage1Store:
    async def list_published_attestation_banks(self):
        row = dict((await PublishedStore().list_published_attestation_banks())[0])
        row.update({"slug": "stage-1", "title": "Старий запис", "source_id": "stage-1.enc"})
        return [row]


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

        self.assertEqual(["stage-1", "stage-2"], [item["slug"] for item in catalog["attestation_banks"]])
        self.assertTrue(catalog["attestation_banks"][0]["system"])
        self.assertFalse(catalog["attestation_banks"][1]["system"])
        self.assertEqual("Новий розділ", catalog["attestation_banks"][1]["sections"][0]["title"])

    async def test_catalog_can_exclude_deleted_bundled_stage_1(self):
        bank = QuestionBank("unused.json")
        await bank.load_published_attestation_banks(PublishedStore())
        runtime = SimpleNamespace(qb=bank, store=None)
        auth = AuthContext({}, {"ok_modules": [], "ok_last_levels": []}, 1, False)

        catalog = MiniAppService(runtime).serialize_catalog(auth, include_stage_1=False)

        self.assertEqual(["stage-2"], [item["slug"] for item in catalog["attestation_banks"]])
        self.assertEqual(0, catalog["attestation_stage_1"]["count"])

    async def test_database_row_cannot_replace_bundled_stage_1_questions(self):
        bank = QuestionBank("unused.json")
        bank.load_attestation_stage_1("attestation_stage_1.json")

        await bank.load_published_attestation_banks(ConflictingStage1Store())

        self.assertEqual(800, len(bank.attestation_banks["stage-1"].qids))
        self.assertEqual(800, len(bank.attestation_stage_1))

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
