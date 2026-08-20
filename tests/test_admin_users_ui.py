from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminUsersUiTests(unittest.TestCase):
    def test_notice_form_has_its_own_screen_and_user_list_stays_clean(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        users_screen = admin.split("export function renderAdminMessages", 1)[0]
        self.assertIn("export function renderAdminMessages", admin)
        self.assertIn('id="admin-users-list-label"', admin)
        self.assertNotIn('id="admin-mini-app-notice"', users_screen)
        self.assertIn('id="admin-mini-app-notice"', admin)

    def test_admin_has_mobile_bottom_navigation(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        app = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("const ADMIN_TABS", admin)
        self.assertIn('screen: "admin-messages"', admin)
        self.assertIn('screen: "admin-stats"', admin)
        self.assertIn(".ui-prototype .admin-tab-bar", styles)
        self.assertIn('case "admin-messages"', app)
        self.assertIn('case "admin-stats"', app)

    def test_user_detail_uses_compact_profile_and_tier_selector(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/components.css").read_text(encoding="utf-8")
        prototype = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        self.assertIn("admin-tier-selector", admin)
        self.assertIn('label: "Немає"', admin)
        self.assertIn('label: "1 етап"', admin)
        self.assertIn('label: "Повний"', admin)
        self.assertIn("admin-materials-status", admin)
        self.assertIn(".admin-tier-selector", styles)
        self.assertIn(".ui-prototype .admin-user-profile", prototype)

    def test_user_detail_uses_three_separate_tabs(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        self.assertIn('data-admin-user-tab="access"', admin)
        self.assertIn('data-admin-user-tab="results"', admin)
        self.assertIn('data-admin-user-tab="info"', admin)
        self.assertIn("adminUserTab", admin)
        self.assertIn(".ui-prototype .admin-profile-tabs", styles)

    def test_user_detail_manages_all_sections_individually(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("Керуйте окремими розділами перемикачами", admin)
        self.assertIn("section_controls", admin)
        self.assertIn("/sections/", admin)
        self.assertIn("section.control_mode", admin)
        self.assertIn('class="switch admin-section-access-row__switch"', admin)

    def test_tier_selector_confirms_access_downgrade(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("const tierRank", admin)
        self.assertIn("const isDowngrade", admin)
        self.assertIn("window.confirm", admin)
        self.assertIn("await updateAccess(option.key, option.message)", admin)

    def test_admin_screen_has_current_cache_version(self):
        index = (ROOT / "static/index.html").read_text(encoding="utf-8")
        entry = (ROOT / "static/app.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles.css").read_text(encoding="utf-8")
        module = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("static/app.js?v=20260820-admin-mobile-01", index)
        self.assertIn("static/styles.css?v=20260820-admin-mobile-01", index)
        self.assertIn("static/js/app.js?v=20260820-admin-mobile-01", entry)
        self.assertIn("styles/components.css?v=20260820-admin-mobile-01", styles)
        self.assertIn("screens/admin.js?v=20260820-admin-mobile-01", module)
        self.assertIn("screens/user.js?v=20260820-admin-mobile-01", module)


if __name__ == "__main__":
    unittest.main()
