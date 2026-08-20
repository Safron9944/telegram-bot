from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminInformationArchitectureTests(unittest.TestCase):
    def test_hub_keeps_secondary_tools_out_of_primary_tabs(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        hub = admin.split("/* ===================== ADMIN ATTESTATION BANKS", 1)[0]

        self.assertEqual(1, hub.count('title: "Пошук по всіх питаннях"'))
        self.assertEqual(3, hub.count("ctx.cell({"))
        self.assertIn('screen: "admin-attestation-banks"', hub)
        self.assertNotIn('screen: "admin-stats"', hub)
        self.assertNotIn('screen: "admin-messages"', hub)
        self.assertIn("Повний доступ", admin)
        self.assertIn("admin-full-price-form", admin)
        self.assertNotIn('title: "Атестації"', hub)
        self.assertNotIn('title: "Питання з APK"', hub)
        self.assertNotIn('title: "Розділи атестації"', hub)
        self.assertNotIn("Атестація посадових осіб — 1 етап", hub)

    def test_section_settings_are_not_duplicated_in_general_settings(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertNotIn("renderAdminSettings", admin)
        self.assertNotIn("loadAdminSettings", admin)
        sections = (ROOT / "static/js/admin_sections.js").read_text(encoding="utf-8")
        self.assertIn("admin-section-visible", sections)
        self.assertIn("admin-section-price", sections)

    def test_global_search_includes_dynamic_question_sections(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        storage = (ROOT / "storage.py").read_text(encoding="utf-8")

        self.assertIn('screen: "admin-global-search"', admin)
        self.assertIn("data.attestation", admin)
        self.assertIn("search_attestation_questions_all", app)
        self.assertIn("JOIN attestation_banks", storage)

    def test_legacy_stage_one_admin_overlay_is_removed(self):
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")
        self.assertFalse((ROOT / "static/js/admin_attestation_stage1.js").exists())
        self.assertNotIn("admin_attestation_stage1", html)


if __name__ == "__main__":
    unittest.main()
