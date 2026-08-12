from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class DynamicAttestationAssetTests(unittest.TestCase):
    def test_home_and_test_flow_use_published_bank_catalog(self):
        user = (ROOT / "static" / "js" / "screens" / "user.js").read_text(encoding="utf-8")
        app = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        state = (ROOT / "static" / "js" / "core" / "state.js").read_text(encoding="utf-8")

        self.assertIn("bootstrap.sections", user)
        self.assertIn("data-home-section", user)
        self.assertIn("selectedAttestationBankSlug", state)
        self.assertIn("section.screen", user)
        self.assertIn('case "attestation-bank"', app)
        self.assertIn("/api/attestation/${bank.slug}/start", user)
        self.assertIn("section.blocks", user)
        self.assertIn("openAttestationBank", app)
        self.assertIn("attestation-start-error", user)
        styles = (ROOT / "static" / "styles" / "components.css").read_text(encoding="utf-8")
        self.assertIn(".attestation-start-error[hidden]", styles)

    def test_admin_has_one_consolidated_attestation_area(self):
        admin = (ROOT / "static" / "js" / "screens" / "admin.js").read_text(encoding="utf-8")
        app = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        detail = (ROOT / "static" / "js" / "admin_attestation_banks.js").read_text(encoding="utf-8")
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('screen: "admin-attestation-banks"', admin)
        self.assertIn('screen: "admin-apk-import"', admin)
        self.assertIn('case "admin-attestation-banks"', app)
        self.assertIn('/api/admin/sections', admin)
        self.assertIn('ctx.navigate("admin-section")', admin)
        self.assertIn('method: "DELETE"', detail)
        self.assertIn('moveBank("up")', detail)
        self.assertIn('moveBank("down")', detail)
        self.assertIn('bank.questions_count', admin)
        self.assertNotIn("Захищено", admin)
        self.assertNotIn("admin_attestation_stage1.js", html)
        self.assertNotIn("Атестація посадових осіб — 1 етап", admin)


if __name__ == "__main__":
    unittest.main()
