from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminInformationArchitectureTests(unittest.TestCase):
    def test_hub_has_only_the_three_primary_tools(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        hub = admin.split("/* ===================== ADMIN ATTESTATION BANKS", 1)[0]

        self.assertEqual(1, hub.count('title: "Користувачі"'))
        self.assertEqual(1, hub.count('title: "Розділи"'))
        self.assertEqual(3, hub.count("ctx.cell({"))
        self.assertNotIn('title: "Атестації"', hub)
        self.assertNotIn('title: "Питання з APK"', hub)
        self.assertNotIn('title: "Розділи атестації"', hub)
        self.assertNotIn("Атестація посадових осіб — 1 етап", hub)

    def test_section_settings_are_not_duplicated_in_general_settings(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        settings = admin.split("/* ===================== ADMIN SETTINGS", 1)[1]

        self.assertNotIn('screen: "admin-settings"', admin.split("/* ===================== ADMIN ATTESTATION BANKS", 1)[0])
        sections = (ROOT / "static/js/admin_sections.js").read_text(encoding="utf-8")
        self.assertIn("admin-section-visible", sections)
        self.assertIn("admin-section-price", sections)

    def test_legacy_stage_one_admin_overlay_is_removed(self):
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")
        self.assertFalse((ROOT / "static/js/admin_attestation_stage1.js").exists())
        self.assertNotIn("admin_attestation_stage1", html)


if __name__ == "__main__":
    unittest.main()
