from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VisibleMessagesTests(unittest.TestCase):
    def test_messages_are_fixed_inside_the_bottom_safe_area(self):
        css = (ROOT / "static/styles/layout.css").read_text(encoding="utf-8")

        block = css.split(".messages {", 1)[1].split("}", 1)[0]
        self.assertIn("position: fixed", block)
        self.assertIn("bottom: calc(16px + var(--content-safe-bottom", block)
        self.assertNotIn("position: sticky", block)

    def test_message_region_is_accessible_and_cache_is_bumped(self):
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")

        self.assertIn('id="messages-panel" role="status" aria-live="polite"', html)
        self.assertIn("20260812-visible-messages-01", html)

    def test_old_success_timer_cannot_hide_a_newer_message(self):
        source = (ROOT / "static/js/core/ui.js").read_text(encoding="utf-8")

        self.assertIn("clearTimeout(messageDismissTimer)", source)
        self.assertIn('messageDismissTimer = setTimeout(() => setMessage("", ""), 2400)', source)


if __name__ == "__main__":
    unittest.main()
