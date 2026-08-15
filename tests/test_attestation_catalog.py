import unittest
from pathlib import Path

from questions import Q, QuestionBank


ROOT = Path(__file__).resolve().parents[1]


def make_question(qid, topic, qnum, *, shuffle=True):
    return Q(
        id=qid,
        section="dynamic",
        topic=topic,
        ok=None,
        level=None,
        qnum=qnum,
        question=f"Question {qid}",
        choices=["A", "B", "C", "D"],
        correct=[1],
        correct_texts=["A"],
        shuffle_choices=shuffle,
    )


class AttestationCatalogTests(unittest.TestCase):
    def setUp(self):
        self.bank = QuestionBank("unused.json")
        questions = [make_question(i, "Section A", i) for i in range(1, 53)]
        questions.append(make_question(100, "Section B", 1, shuffle=False))
        self.bank.register_attestation_bank("stage-2", "Stage 2", questions)

    def test_lists_sections_and_dynamic_blocks_in_source_order(self):
        sections = self.bank.attestation_sections("stage-2")
        self.assertEqual(["Section A", "Section B"], [item["title"] for item in sections])
        self.assertEqual([52, 1], [item["count"] for item in sections])
        self.assertEqual(["1-50", "51-52"], [item["key"] for item in sections[0]["blocks"]])

    def test_selects_partial_and_random_blocks(self):
        self.assertEqual([51, 52], self.bank.attestation_block_qids("stage-2", "Section A", "51-52"))
        random_ids = self.bank.attestation_block_qids("stage-2", "Section A", "random")
        self.assertEqual(50, len(random_ids))
        self.assertTrue(set(random_ids).issubset(set(range(1, 53))))

    def test_preserves_order_sensitive_choice_flag(self):
        self.assertFalse(self.bank.by_id[100].shuffle_choices)

    def test_rejects_unknown_bank_or_block(self):
        self.assertEqual([], self.bank.attestation_sections("missing"))
        self.assertEqual([], self.bank.attestation_block_qids("stage-2", "Section A", "bad"))

    def test_loads_bundled_ukrainian_language_test_questions(self):
        bank = QuestionBank("unused.json")

        loaded = bank.load_bundled_attestation_bank(
            str(ROOT / "data" / "ukrainian_language_questions"),
            slug="ukrainian-language",
            title="Державна мова",
            source_id="bundled-ukrainian-language-3.8.26",
            id_offset=20_000_000,
            manual_grant_section_key="ukrainian_language",
        )

        self.assertEqual(1937, len(loaded.qids))
        self.assertEqual("ukrainian_language", loaded.manual_grant_section_key)
        self.assertEqual(307, sum(len(bank.by_id[qid].correct) > 1 for qid in loaded.qids))
        self.assertTrue(all(bank.by_id[qid].choices for qid in loaded.qids))


if __name__ == "__main__":
    unittest.main()
