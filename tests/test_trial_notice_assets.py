from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrialNoticeAssetsTests(unittest.TestCase):
    def test_home_shows_trial_notice_only_for_active_trial(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        self.assertIn('!user.is_admin && user.access?.tier === "trial_full"', user)
        self.assertIn("Безкоштовний тріал на 3 дні", user)
        self.assertIn("Після завершення доступ можна буде придбати", user)
        self.assertIn('class="trial-notice"', user)

    def test_trial_notice_has_mobile_card_styles(self):
        styles = (ROOT / "static/styles/prototype.css").read_text(encoding="utf-8")
        self.assertIn(".ui-prototype .trial-notice", styles)
        self.assertIn(".ui-prototype .trial-notice__badge", styles)

    def test_trial_notice_assets_have_fresh_cache_version(self):
        index = (ROOT / "static/index.html").read_text(encoding="utf-8")
        app = (ROOT / "static/app.js").read_text(encoding="utf-8")
        module = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("20260814-trial-notice-01", index)
        self.assertIn("20260814-trial-notice-01", app)
        self.assertIn("screens/user.js?v=20260814-trial-notice-01", module)


if __name__ == "__main__":
    unittest.main()
