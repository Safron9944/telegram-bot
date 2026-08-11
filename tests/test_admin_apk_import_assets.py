from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class AdminApkImportAssetsTests(unittest.TestCase):
    def test_module_is_loaded_and_exposes_complete_workflow(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        module = (ROOT / "static" / "js" / "admin_apk_import.js").read_text(encoding="utf-8")

        self.assertIn("/static/js/admin_apk_import.js", html)
        self.assertIn("admin-apk-import-entry", module)
        for fragment in ("/sessions", "/banks/", "/parse", "/preview", "/download"):
            self.assertIn(fragment, module)
        for label in ("Завантажити APK", "Оберіть банк", "Пошук питань", "Правильна відповідь"):
            self.assertIn(label, module)


if __name__ == "__main__":
    unittest.main()
