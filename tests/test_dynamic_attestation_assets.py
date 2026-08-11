from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


class DynamicAttestationAssetTests(unittest.TestCase):
    def test_home_and_test_flow_use_published_bank_catalog(self):
        user = (ROOT / "static" / "js" / "screens" / "user.js").read_text(encoding="utf-8")
        app = (ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
        state = (ROOT / "static" / "js" / "core" / "state.js").read_text(encoding="utf-8")

        self.assertIn("catalog.attestation_banks", user)
        self.assertIn("data-attestation-bank", user)
        self.assertIn("selectedAttestationBankSlug", state)
        self.assertIn('screen: "attestation-bank"', user)
        self.assertIn('case "attestation-bank"', app)
        self.assertIn("/api/attestation/${bank.slug}/start", user)
        self.assertIn("section.blocks", user)
        self.assertIn("openAttestationBank", app)


if __name__ == "__main__":
    unittest.main()
