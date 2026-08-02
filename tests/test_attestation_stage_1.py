import json
import unittest
from pathlib import Path

from questions import ATTESTATION_STAGE_1_SECTION, QuestionBank


ROOT = Path(__file__).resolve().parents[1]


class AttestationStage1BankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = QuestionBank(str(ROOT / "questions_flat.json"))
        cls.bank.load()
        cls.base_law_count = len(cls.bank.law)
        cls.bank.load_attestation_stage_1(str(ROOT / "attestation_stage_1.json"))

    def test_all_800_questions_are_loaded_separately(self):
        self.assertEqual(800, len(self.bank.attestation_stage_1))
        self.assertEqual(self.base_law_count, len(self.bank.law))
        self.assertTrue(
            all(self.bank.by_id[qid].section == ATTESTATION_STAGE_1_SECTION for qid in self.bank.attestation_stage_1)
        )

    def test_every_question_has_four_unique_choices_and_valid_answer(self):
        for qid in self.bank.attestation_stage_1:
            question = self.bank.by_id[qid]
            self.assertEqual(4, len(question.choices), qid)
            self.assertEqual(4, len({choice.casefold().strip() for choice in question.choices}), qid)
            self.assertEqual(1, len(question.correct), qid)
            self.assertIn(question.correct[0], range(1, 5), qid)
            self.assertEqual(question.correct_texts[0], question.choices[question.correct[0] - 1], qid)

    def test_source_topics_are_complete(self):
        source = json.loads((ROOT / "attestation_stage_1.json").read_text(encoding="utf-8"))["questions"]
        topic_counts = {}
        for item in source:
            topic_counts[item["topic"]] = topic_counts.get(item["topic"], 0) + 1
        self.assertEqual(4, len(topic_counts))
        self.assertEqual({200}, set(topic_counts.values()))

    def test_catalog_sections_are_available_in_source_order(self):
        sections = self.bank.attestation_stage_1_sections()
        self.assertEqual(
            [
                "Конституція України",
                "Закон України «Про державну службу»",
                "Митний кодекс України",
                "Закон України «Про запобігання корупції»",
            ],
            [item["title"] for item in sections],
        )
        self.assertEqual([200, 200, 200, 200], [item["count"] for item in sections])
if __name__ == "__main__":
    unittest.main()
