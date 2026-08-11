from dataclasses import replace
import unittest

from apk_importer.models import ParsedBank, ParsedQuestion, ParsedSection
from attestation_publishing import AttestationPublishError, AttestationPublishingService


def parsed_bank(*, choices=("A", "B", "C", "D"), correct=(1,)):
    question = ParsedQuestion(
        source_key="section-1:q1",
        qnum=1,
        topic="Section 1",
        question="Question?",
        choices=choices,
        correct=correct,
        correct_texts=(choices[correct[0] - 1],) if correct and 1 <= correct[0] <= len(choices) else (),
        shuffle_choices=False,
    )
    return ParsedBank(
        adapter="testms",
        source="testms2.enc",
        source_version="v1",
        source_hash="abc123",
        sections=(ParsedSection("Section 1", 0, 1),),
        questions=(question,),
    )


class FakeStore:
    def __init__(self):
        self.calls = []

    async def publish_attestation_bank(self, bank, *, title, slug, changed_by):
        self.calls.append((bank, title, slug, changed_by))
        return {"id": 7, "slug": slug, "title": title, "count": len(bank.questions), "updated": False}


class AttestationPublishingTests(unittest.IsolatedAsyncioTestCase):
    async def test_publishes_valid_bank_and_reloads_catalog(self):
        store = FakeStore()
        reloads = 0

        async def reload_catalog():
            nonlocal reloads
            reloads += 1

        result = await AttestationPublishingService(store, reload_catalog).publish(
            parsed_bank(), "Атестація — 2 етап", changed_by="99"
        )

        self.assertEqual("testms2", result["slug"])
        self.assertEqual("Атестація — 2 етап", result["title"])
        self.assertEqual(1, reloads)
        self.assertEqual("99", store.calls[0][3])

    async def test_rejects_invalid_title_choices_and_correct_index(self):
        service = AttestationPublishingService(FakeStore(), lambda: None)
        cases = (
            (parsed_bank(), " ", "title_required"),
            (parsed_bank(choices=("A",)), "Stage 2", "answer_choices_required"),
            (parsed_bank(correct=(5,)), "Stage 2", "correct_answer_invalid"),
        )
        for bank, title, code in cases:
            with self.subTest(code=code), self.assertRaises(AttestationPublishError) as raised:
                await service.publish(bank, title, changed_by="99")
            self.assertEqual(code, raised.exception.code)

    async def test_accepts_real_bank_question_with_five_choices(self):
        store = FakeStore()
        bank = parsed_bank(choices=("A", "B", "C", "D", "E"))

        result = await AttestationPublishingService(store, lambda: None).publish(
            bank, "Митних органів", changed_by="99"
        )

        self.assertEqual(1, result["count"])

    async def test_reserved_stage_1_slug_cannot_replace_bundled_stage(self):
        store = FakeStore()
        bank = replace(parsed_bank(), source="stage-1.enc")

        result = await AttestationPublishingService(store, lambda: None).publish(
            bank, "Інший тест", changed_by="99"
        )

        self.assertNotEqual("stage-1", result["slug"])


if __name__ == "__main__":
    unittest.main()
