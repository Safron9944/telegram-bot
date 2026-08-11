from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AttestationCmsAssetsTests(unittest.TestCase):
    def test_cms_screens_are_registered_and_opened_from_admin(self):
        app = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        state = (ROOT / "static/js/core/state.js").read_text(encoding="utf-8")

        self.assertIn('"admin-attestation-bank"', app)
        self.assertIn('"admin-attestation-question"', app)
        self.assertIn("renderAdminAttestationQuestion", app)
        self.assertIn("Керувати питаннями", admin)
        self.assertIn("selectedAttestationAdminQuestionId", state)

    def test_cms_supports_search_and_full_question_crud(self):
        module = (ROOT / "static/js/admin_attestation_banks.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/components.css").read_text(encoding="utf-8")

        self.assertIn("/questions?topic=", module)
        self.assertIn("Додати питання", module)
        self.assertIn("Зберегти зміни", module)
        self.assertIn("Видалити питання", module)
        self.assertIn("Додати варіант", module)
        self.assertIn("attestation-managed-question-form", styles)
        self.assertIn("attestation-bank-filters", styles)


if __name__ == "__main__":
    unittest.main()
