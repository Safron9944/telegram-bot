import unittest
from pathlib import Path

from questions import Q, QuestionBank


ROOT = Path(__file__).resolve().parents[1]

UKRAINIAN_LANGUAGE_SECTIONS = [
    ("Слова близькі за значенням", 22),
    ("Слова протилежні за значенням", 21),
    ("Можливі обидва слова, наведені в дужках", 60),
    ("Правильно вжито всі слова та форми", 73),
    ("Рядок містить помилку", 63),
    ("Точність висловлювання", 42),
    ("Пряма мова в реченні", 18),
    ("Норми культури мови у словосполученнях", 74),
    ("Офіційне мовлення", 32),
    ("Відмінювання прізвищ, імен та по батькові", 69),
    ("Виправлення у реченнях", 60),
    ("Розділові знаки", 72),
    ("Іншомовні слова", 206),
    ("Значення слів", 175),
    ("Значення висловів", 83),
    ("Відповідність між текстом і метою мовлення", 227),
    ("Пропущені літери", 85),
    ("Пропущені слова", 200),
    ("Пропущені вислови", 115),
    ("Розуміння тексту (правда чи неправда)", 175),
    ("Розуміння тексту (вибір відповіді)", 65),
    ("Написання тексту на визначену тему", 11),
    ("Говоріння", 113),
]


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

        self.assertEqual(2061, len(loaded.qids))
        self.assertEqual("ukrainian_language", loaded.manual_grant_section_key)
        self.assertEqual(307, sum(len(bank.by_id[qid].correct) > 1 for qid in loaded.qids))
        practice_qids = [qid for qid in loaded.qids if bank.by_id[qid].is_open_practice]
        test_qids = [qid for qid in loaded.qids if not bank.by_id[qid].is_open_practice]
        self.assertEqual(124, len(practice_qids))
        self.assertEqual(1937, len(test_qids))
        self.assertTrue(all(bank.by_id[qid].choices for qid in test_qids))
        self.assertEqual(
            UKRAINIAN_LANGUAGE_SECTIONS,
            [(item["title"], item["count"]) for item in bank.attestation_sections(loaded.slug)],
        )
        visible_text = [
            value
            for qid in loaded.qids
            for value in [
                bank.by_id[qid].topic,
                bank.by_id[qid].question,
                bank.by_id[qid].practice_answer,
                *bank.by_id[qid].choices,
            ]
        ]
        self.assertTrue(all("\u0301" not in value for value in visible_text))
        practice_sections = [item["title"] for item in bank.attestation_sections(loaded.slug) if item["practice"]]
        self.assertEqual(["Написання тексту на визначену тему", "Говоріння"], practice_sections)
        sections = {item["title"]: item for item in bank.attestation_sections(loaded.slug)}
        self.assertEqual(11, len(sections["Написання тексту на визначену тему"]["items"]))
        self.assertEqual(113, len(sections["Говоріння"]["items"]))
        self.assertEqual([], sections["Говоріння"]["blocks"])


if __name__ == "__main__":
    unittest.main()
