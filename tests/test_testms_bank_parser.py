from pathlib import Path
import unittest

from apk_importer.testms import TestMsParseError, expand_testms_macros, parse_testms_bank


FIXTURE = Path(__file__).parent / "fixtures" / "testms_plaintext_small.txt"


class TestMsBankParserTests(unittest.TestCase):
    def test_parses_questions_and_omits_explanations(self):
        bank = parse_testms_bank(
            FIXTURE.read_text(encoding="utf-8"),
            source="testmsat.enc",
            source_hash="a" * 64,
        )

        self.assertEqual("3", bank.source_version)
        self.assertEqual(2, len(bank.sections))
        self.assertEqual(2, len(bank.questions))
        self.assertEqual("Що таке?", bank.questions[0].question)
        self.assertEqual(("правильна відповідь", "неправильна відповідь"), bank.questions[0].choices)
        self.assertEqual((1,), bank.questions[0].correct)
        self.assertEqual(("правильна відповідь",), bank.questions[0].correct_texts)
        self.assertTrue(bank.questions[0].shuffle_choices)
        self.assertFalse(bank.questions[1].shuffle_choices)
        self.assertEqual("testmsat.enc:I:1", bank.questions[0].source_key)
        self.assertEqual("testmsat.enc:II:1", bank.questions[1].source_key)
        self.assertNotIn("explanation", bank.to_dict()["questions"][0])

    def test_expands_all_four_macro_spacing_modes(self):
        text = "testmsat 3\n$слово\n$%%X|$9%X|$M%X|$a%X"

        expanded = expand_testms_macros(text).splitlines()[-1]

        self.assertEqual(" слово X| словоX|слово X|словоX", expanded)

    def test_rejects_unresolved_macro(self):
        text = "testmsat 3\n$одне\n\n~I. Розділ\n#I. 1. $a&\n+так\n-ні"

        with self.assertRaises(TestMsParseError) as raised:
            parse_testms_bank(text, source="testmsat.enc", source_hash="a" * 64)

        self.assertEqual("unresolved_macro", raised.exception.code)

    def test_rejects_question_without_exactly_one_correct_answer(self):
        text = "testmsat 3\n$слово\n\n~I. Розділ\n#I. 1. Питання?\n-так\n-ні"

        with self.assertRaises(TestMsParseError) as raised:
            parse_testms_bank(text, source="testmsat.enc", source_hash="a" * 64)

        self.assertEqual("invalid_correct_count", raised.exception.code)


if __name__ == "__main__":
    unittest.main()
