from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProtectedMaterialsAssetsTests(unittest.TestCase):
    def test_home_hides_manual_grant_sections_without_access(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        self.assertIn("section.manual_grant_only && !user.is_admin && !section.has_access", user)

    def test_admin_user_has_individual_materials_controls(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("Додаткові матеріали", admin)
        self.assertIn("section_controls", admin)
        self.assertIn("/sections/", admin)
        self.assertIn('section.control_mode === "visibility"', admin)

    def test_payment_copy_describes_all_visible_paid_sections(self):
        access = (ROOT / "access.py").read_text(encoding="utf-8")
        self.assertIn("Безлімітний доступ до всіх показаних платних розділів", access)
        self.assertNotIn("навчання, тести, кейси та атестація", access)

    def test_locked_sections_offer_individual_or_full_purchase(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        self.assertIn("Купити лише цей розділ", user)
        self.assertIn("Купити повний доступ до всіх розділів", user)
        self.assertIn('ctx.navigate("purchase-options")', user)

    def test_admin_can_edit_full_access_price(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("Повний доступ до всіх розділів, ⭐", admin)
        self.assertIn("price_full: priceFull", admin)

    def test_server_routes_require_explicit_protected_access(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        storage = (ROOT / "storage.py").read_text(encoding="utf-8")
        self.assertIn('"protected_materials_required"', app)
        self.assertIn('/protected-materials")', app)
        self.assertIn("protected_materials_access_v1", storage)


if __name__ == "__main__":
    unittest.main()
