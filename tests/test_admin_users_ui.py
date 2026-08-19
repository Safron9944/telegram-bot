from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminUsersUiTests(unittest.TestCase):
    def test_notice_form_is_collapsible_and_user_list_stays_separate(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn('class="admin-notice-panel"', admin)
        self.assertIn('id="admin-users-list-label"', admin)
        self.assertIn("adminNoticeOpen", admin)

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

    def test_user_detail_sections_and_account_info_are_collapsible(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/components.css").read_text(encoding="utf-8")
        self.assertIn('disclosure.className = "admin-section-group"', admin)
        self.assertIn('class="admin-user-info"', admin)
        self.assertIn("adminUserSectionGroupsOpen", admin)
        self.assertIn(".admin-section-group__summary", styles)

    def test_user_detail_manages_all_sections_individually(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("Керування розділами", admin)
        self.assertIn("section_controls", admin)
        self.assertIn("/sections/", admin)
        self.assertIn("section.control_mode", admin)

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
        self.assertIn("static/app.js?v=20260819-admin-user-compact-01", index)
        self.assertIn("static/styles.css?v=20260819-admin-user-compact-01", index)
        self.assertIn("static/js/app.js?v=20260819-admin-user-compact-01", entry)
        self.assertIn("styles/components.css?v=20260819-admin-user-compact-01", styles)
        self.assertIn("screens/admin.js?v=20260819-admin-user-compact-01", module)
        self.assertIn("screens/user.js?v=20260816-paid-functions-01", module)


if __name__ == "__main__":
    unittest.main()
