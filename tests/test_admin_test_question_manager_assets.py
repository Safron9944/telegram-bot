from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminTestQuestionManagerAssetsTests(unittest.TestCase):
    def setUp(self):
        self.script = (ROOT / "static/js/admin_test_exam_crud.js").read_text(encoding="utf-8")

    def test_toolbar_only_offers_json_import(self):
        toolbar = self.script.split("function injectToolbar()", 1)[1]
        self.assertIn('id="test-q-manager-import"', toolbar)
        self.assertNotIn('id="test-q-manager-add"', toolbar)
        self.assertNotIn("renderQuestionEditor(null)", toolbar)

    def test_import_is_a_bounded_dialog_instead_of_a_fullscreen_page(self):
        self.assertIn('overlay.setAttribute("role", "dialog")', self.script)
        self.assertIn("max-height:min(86dvh,900px)", self.script)
        self.assertIn("place-items:center", self.script)
        self.assertIn("overflow-y:auto", self.script)

    def test_file_picker_uses_a_native_label_target(self):
        self.assertIn('for="tqi-file"', self.script)
        self.assertNotIn('querySelector("#tqi-pick")', self.script)


if __name__ == "__main__":
    unittest.main()
