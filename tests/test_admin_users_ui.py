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

    def test_admin_assets_have_fresh_cache_version(self):
        index = (ROOT / "static/index.html").read_text(encoding="utf-8")
        app = (ROOT / "static/app.js").read_text(encoding="utf-8")
        module = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("20260814-admin-users-01", index)
        self.assertIn("20260814-admin-users-01", app)
        self.assertIn("screens/admin.js?v=20260814-admin-users-01", module)


if __name__ == "__main__":
    unittest.main()
