import unittest

from apk_importer.models import ParsedBank, ParsedQuestion, ParsedSection
from apk_importer.validation import BankValidationError, validate_bank


def make_question(**overrides):
    values = {
        "source_key": "constitution:1",
        "qnum": 1,
        "topic": "Конституція України",
        "question": "Питання?",
        "choices": ("A", "B", "C", "D"),
        "correct": (3,),
        "correct_texts": ("C",),
        "shuffle_choices": False,
    }
    values.update(overrides)
    return ParsedQuestion(**values)


def make_bank(*questions):
    selected = questions or (make_question(),)
    return ParsedBank(
        adapter="testms",
        source="testmsat.enc",
        source_version="3",
        source_hash="a" * 64,
        sections=(ParsedSection("Конституція України", 0, len(selected)),),
        questions=tuple(selected),
    )


class ParsedBankTests(unittest.TestCase):
    def test_serializes_stable_import_contract(self):
        bank = make_bank()

        validate_bank(bank)
        payload = bank.to_dict()

        self.assertEqual(
            ["source", "source_version", "source_hash", "count", "sections", "questions"],
            list(payload),
        )
        self.assertEqual(1, payload["count"])
        self.assertEqual(
            {
                "source_key": "constitution:1",
                "qnum": 1,
                "topic": "Конституція України",
                "question": "Питання?",
                "choices": ["A", "B", "C", "D"],
                "correct": [3],
                "correct_texts": ["C"],
                "shuffle_choices": False,
            },
            payload["questions"][0],
        )

    def test_reports_all_invalid_question_fields_with_stable_codes(self):
        question = make_question(
            source_key="",
            topic="",
            question=" ",
            choices=("A", "A"),
            correct=(3,),
            correct_texts=("wrong",),
        )

        with self.assertRaises(BankValidationError) as raised:
            validate_bank(make_bank(question))

        self.assertEqual(
            {
                "missing_source_key",
                "missing_topic",
                "empty_question",
                "duplicate_choices",
                "correct_index_out_of_range",
                "correct_text_mismatch",
                "unknown_section",
            },
            {issue.code for issue in raised.exception.issues},
        )

    def test_rejects_too_few_choices_and_missing_correct_answer(self):
        question = make_question(choices=("A",), correct=(), correct_texts=())

        with self.assertRaises(BankValidationError) as raised:
            validate_bank(make_bank(question))

        self.assertEqual(
            {"too_few_choices", "missing_correct_answer"},
            {issue.code for issue in raised.exception.issues},
        )

    def test_rejects_duplicate_source_keys(self):
        duplicate = make_question(qnum=2)
        bank = ParsedBank(
            adapter="testms",
            source="testmsat.enc",
            source_version="3",
            source_hash="a" * 64,
            sections=(ParsedSection("Конституція України", 0, 2),),
            questions=(make_question(), duplicate),
        )

        with self.assertRaises(BankValidationError) as raised:
            validate_bank(bank)

        self.assertIn("duplicate_source_key", {issue.code for issue in raised.exception.issues})


if __name__ == "__main__":
    unittest.main()
