from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminUsersUiTests(unittest.TestCase):
    def test_notice_form_is_collapsible_and_user_list_stays_separate(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn('class="admin-notice-panel"', admin)
        self.assertIn('id="admin-users-list-label"', admin)
        self.assertIn("adminNoticeOpen", admin)

    def test_user_detail_uses_compact_account_and_access_cards(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/components.css").read_text(encoding="utf-8")
        prototype = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        self.assertIn("admin-account-card", admin)
        self.assertIn("admin-access-actions--grid", admin)
        self.assertIn("admin-materials-status", admin)
        self.assertIn(".admin-access-actions--grid", styles)
        self.assertIn(".ui-prototype .admin-account-card", prototype)

    def test_user_detail_manages_all_sections_individually(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("Керування розділами", admin)
        self.assertIn("section_controls", admin)
        self.assertIn("/sections/", admin)
        self.assertIn("показ і оплачений доступ керуються окремо", admin)

    def test_admin_screen_has_current_cache_version(self):
        index = (ROOT / "static/index.html").read_text(encoding="utf-8")
        entry = (ROOT / "static/app.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")
        module = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("static/app.js?v=20260816-purchase-access-01", index)
        self.assertIn("static/styles.css?v=20260816-section-access-02", index)
        self.assertIn("static/js/app.js?v=20260816-purchase-access-01", entry)
        self.assertIn("styles/components.css?v=20260816-section-access-02", styles)
        self.assertIn("screens/admin.js?v=20260816-purchase-access-01", module)
        self.assertIn("screens/user.js?v=20260816-purchase-access-01", module)


if __name__ == "__main__":
    unittest.main()
