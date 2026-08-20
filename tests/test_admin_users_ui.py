from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AdminUsersUiTests(unittest.TestCase):
    def test_notice_form_has_its_own_screen_and_user_list_stays_clean(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        users_screen = admin.split("export function renderAdminMessages", 1)[0]
        self.assertIn("export function renderAdminMessages", admin)
        self.assertIn('id="admin-users-list-label"', admin)
        self.assertIn('id="admin-users-notice"', users_screen)
        self.assertIn("Надіслати повідомлення", users_screen)
        self.assertIn("admin-users-notice-button", styles)
        self.assertNotIn('id="admin-user-search"', users_screen)
        self.assertNotIn('id="admin-users-more"', users_screen)
        self.assertNotIn('id="admin-message-fab"', users_screen)
        self.assertNotIn('id="admin-mini-app-notice"', users_screen)
        self.assertIn('id="admin-mini-app-notice"', admin)

    def test_admin_has_no_persistent_bottom_navigation(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        styles = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        app = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertNotIn("const ADMIN_TABS", admin)
        self.assertNotIn("mountAdminTabs", admin)
        self.assertNotIn("admin-tab-bar", styles)
        self.assertIn('screen: "admin-attestation-banks"', admin)
        self.assertNotIn('screen: "admin-stats"', admin)
        self.assertIn('case "admin-messages"', app)
        self.assertNotIn('case "admin-stats"', app)

    def test_user_detail_uses_compact_profile_and_separate_access_screen(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        prototype = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        self.assertIn("export function renderAdminUserAccess", admin)
        self.assertIn('ctx.navigate("admin-user-access")', admin)
        self.assertIn("admin-profile-hero", admin)
        self.assertIn("admin-quick-grid", admin)
        self.assertIn('label: "Немає"', admin)
        self.assertIn('label: "Повний"', admin)
        self.assertNotIn('label: "1 етап"', admin)
        self.assertIn(".ui-prototype .admin-profile-hero", prototype)
        self.assertIn(".ui-prototype .admin-access-list", prototype)

    def test_admin_access_uses_only_none_and_full_without_legacy_stage_duplicate(self):
        server = (ROOT / "app.py").read_text(encoding="utf-8")
        request_model = server.split("class AdminAccessUpdateRequest", 1)[1].split("class AdminProtectedMaterialsUpdateRequest", 1)[0]
        detail = server.split("async def admin_user_detail", 1)[1].split("async def admin_set_access", 1)[0]
        self.assertIn('Literal["full", "none"]', request_model)
        self.assertNotIn('"cases"', request_model)
        self.assertNotIn("ATTESTATION_STAGE_1_SECTION_KEY", detail)
        self.assertNotIn("Атестація посадових осіб — 1 етап", detail)

    def test_user_detail_does_not_use_legacy_tabs(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        detail = admin.split("export function renderAdminUserDetail", 1)[1].split("export function renderAdminUserAccess", 1)[0]
        self.assertNotIn("admin-profile-tabs", detail)
        self.assertNotIn("data-admin-user-tab", detail)
        self.assertIn('id="admin-manage-access"', detail)

    def test_user_detail_manages_all_sections_individually(self):
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")
        self.assertIn("section_controls", admin)
        self.assertIn("/sections/", admin)
        self.assertIn("section.control_mode", admin)
        self.assertIn('class="switch admin-access-row__switch"', admin)

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
        self.assertIn("static/app.js?v=20260820-admin-section-layout-08", index)
        self.assertIn("static/styles.css?v=20260820-admin-section-layout-08", index)
        self.assertIn("static/js/app.js?v=20260820-admin-section-layout-08", entry)
        self.assertIn("styles/components.css?v=20260820-admin-section-layout-08", styles)
        self.assertIn("screens/admin.js?v=20260820-admin-section-layout-08", module)
        self.assertIn("screens/user.js?v=20260820-admin-section-layout-08", module)


if __name__ == "__main__":
    unittest.main()
