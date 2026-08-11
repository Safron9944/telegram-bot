from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class AdminApkImportAssetsTests(unittest.TestCase):
    def test_module_is_loaded_and_exposes_complete_workflow(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        admin = (ROOT / "static" / "js" / "screens" / "admin.js").read_text(encoding="utf-8")
        module = (ROOT / "static" / "js" / "admin_apk_import.js").read_text(encoding="utf-8")

        self.assertNotIn("/static/js/admin_apk_import.js", html)
        self.assertIn("./admin_apk_import.js", app)
        self.assertIn('case "admin-apk-import"', app)
        self.assertIn('state.currentScreen === "admin-apk-import"', app)
        self.assertIn('screen: "admin-apk-import"', admin)
        self.assertIn("renderAdminApkImport", module)
        for fragment in ("/sessions", "/banks/", "/parse", "/preview", "/download"):
            self.assertIn(fragment, module)
        for label in ("Вибрати файл", "Оберіть банк", "Пошук питань", "Правильна відповідь"):
            self.assertIn(label, module)

    def test_importer_has_touch_friendly_mobile_layout_and_busy_state(self):
        css = (ROOT / "static" / "styles" / "components.css").read_text(encoding="utf-8")
        module = (ROOT / "static" / "js" / "admin_apk_import.js").read_text(encoding="utf-8")

        for fragment in (
            "@media (max-width: 767px)",
            ".apk-import-screen",
            ".apk-file-picker",
            ".apk-import-message:empty",
            "min-height: 44px",
            "font-size: 16px",
        ):
            self.assertIn(fragment, css)
        self.assertIn("cleanupAdminApkImport", module)
        self.assertIn('button.className = "cell apk-bank"', module)
        self.assertIn("bank.title || bank.filename", module)
        self.assertIn('message("")', module)

    def test_preview_can_create_a_test_section(self):
        module = (ROOT / "static" / "js" / "admin_apk_import.js").read_text(encoding="utf-8")
        css = (ROOT / "static" / "styles" / "components.css").read_text(encoding="utf-8")

        self.assertIn('id="apk-publish-title"', module)
        self.assertIn('id="apk-publish"', module)
        self.assertIn("Створити розділ", module)
        self.assertIn("publishing", module)
        self.assertIn("openAttestationBank", module)
        self.assertIn(".apk-publish", css)


if __name__ == "__main__":
    unittest.main()
