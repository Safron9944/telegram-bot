from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminInformationArchitectureTests(unittest.TestCase):
    def test_hub_does_not_duplicate_attestation_tools(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        hub = admin.split("/* ===================== ADMIN ATTESTATION BANKS", 1)[0]

        self.assertEqual(1, hub.count('title: "Атестації"'))
        self.assertNotIn('title: "Питання з APK"', hub)
        self.assertNotIn('title: "Розділи атестації"', hub)
        self.assertNotIn("Атестація посадових осіб — 1 етап", hub)

    def test_attestation_visibility_is_not_duplicated_in_general_settings(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        settings = admin.split("/* ===================== ADMIN SETTINGS", 1)[1]

        self.assertNotIn('["attestation",', settings)
        self.assertIn("Атестації та їх видимість керуються", settings)

    def test_legacy_stage_one_admin_overlay_is_removed(self):
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")
        self.assertFalse((ROOT / "static/js/admin_attestation_stage1.js").exists())
        self.assertNotIn("admin_attestation_stage1", html)


if __name__ == "__main__":
    unittest.main()
