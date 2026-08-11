import json
import re
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

    def test_question_text_does_not_contain_ocr_garbage(self):
        repeated_letter = re.compile(r"([А-Яа-яІіЇїЄєҐґ])\1\1")
        zero_prefix = re.compile(r"^\s*[0OО]\s+[А-Яа-яІіЇїЄєҐґ]")
        ukrainian_word = re.compile(r"[А-Яа-яІіЇїЄєҐґ]{8,}")
        vowels = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")

        for qid in self.bank.attestation_stage_1:
            question = self.bank.by_id[qid]
            for text in [question.question, *question.choices]:
                self.assertNotIn("|", text, qid)
                self.assertIsNone(repeated_letter.search(text), qid)
                self.assertIsNone(zero_prefix.search(text), qid)
                low_vowel_words = [
                    word for word in ukrainian_word.findall(text)
                    if sum(char in vowels for char in word) <= 1
                ]
                self.assertEqual([], low_vowel_words, f"{qid}: {low_vowel_words}")

    def test_verified_answer_mappings_are_not_shifted_by_ocr(self):
        source = json.loads((ROOT / "attestation_stage_1.json").read_text(encoding="utf-8"))["questions"]

        def find(topic, number):
            return next(item for item in source if item["topic"] == topic and item["qnum"] == number)

        constitution_167 = find("Конституція України", 167)
        self.assertEqual("Президент України", constitution_167["choices"][3])
        self.assertEqual([4], constitution_167["correct"])

        customs_167 = find("Митний кодекс України", 167)
        self.assertIn("150 євро", customs_167["choices"][2])
        self.assertEqual([3], customs_167["correct"])

        corruption_198 = find("Закон України «Про запобігання корупції»", 198)
        self.assertEqual("у разі звернення фізичної особи щодо отримання відомостей про себе", corruption_198["choices"][2])

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

    def test_each_section_is_split_into_four_numbered_blocks(self):
        for section in self.bank.attestation_stage_1_sections():
            self.assertEqual(
                ["1-50", "51-100", "101-150", "151-200"],
                [item["key"] for item in section["blocks"]],
            )
            title = section["title"]
            for block, expected_numbers in {
                "1-50": range(1, 51),
                "51-100": range(51, 101),
                "101-150": range(101, 151),
                "151-200": range(151, 201),
            }.items():
                qids = self.bank.attestation_stage_1_block_qids(title, block)
                self.assertEqual(list(expected_numbers), [self.bank.by_id[qid].qnum for qid in qids])

    def test_random_block_contains_50_questions_from_selected_section(self):
        title = "Митний кодекс України"
        qids = self.bank.attestation_stage_1_block_qids(title, "random")
        self.assertEqual(50, len(qids))
        self.assertTrue(all(self.bank.by_id[qid].topic == title for qid in qids))
if __name__ == "__main__":
    unittest.main()
