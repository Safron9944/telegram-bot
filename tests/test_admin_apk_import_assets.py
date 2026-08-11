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

    def test_importer_has_touch_friendly_mobile_layout_and_busy_state(self):
        css = (ROOT / "static" / "styles" / "components.css").read_text(encoding="utf-8")
        module = (ROOT / "static" / "js" / "admin_apk_import.js").read_text(encoding="utf-8")

        for fragment in (
            "@media (max-width: 767px)",
            "height: 100dvh",
            "overflow-y: auto",
            "-webkit-overflow-scrolling: touch",
            "min-height: 44px",
            "font-size: 16px",
        ):
            self.assertIn(fragment, css)
        self.assertIn('document.body.classList.add("apk-import-open")', module)
        self.assertIn('button.className = "cell apk-bank"', module)
        self.assertIn('message("")', module)


if __name__ == "__main__":
    unittest.main()
