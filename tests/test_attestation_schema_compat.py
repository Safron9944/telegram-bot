from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class AttestationSchemaCompatibilityTests(unittest.TestCase):
    def test_dynamic_banks_do_not_reuse_legacy_attestation_questions_table(self):
        source = (ROOT / "storage.py").read_text(encoding="utf-8")

        self.assertIn("published_attestation_questions", source)
        self.assertNotIn(
            "idx_attestation_questions_bank ON attestation_questions(bank_id)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
