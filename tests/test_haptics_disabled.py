from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HapticsDisabledTests(unittest.TestCase):
    def test_mini_app_does_not_call_telegram_haptic_api(self):
        scripts = (ROOT / "static/js").rglob("*.js")
        source = "\n".join(path.read_text(encoding="utf-8") for path in scripts)

        self.assertNotIn("HapticFeedback", source)
        self.assertNotIn("impactOccurred", source)
        self.assertNotIn("notificationOccurred", source)

    def test_shared_impact_hook_is_a_noop(self):
        telegram = (ROOT / "static/js/core/telegram.js").read_text(encoding="utf-8")
        self.assertIn("export function impact() {}", telegram)


if __name__ == "__main__":
    unittest.main()
