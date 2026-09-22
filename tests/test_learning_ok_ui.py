from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LearningOkUiTests(unittest.TestCase):
    def test_ok_learning_shows_levels_inline(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        block = user.split("function renderOkTab", 1)[1].split("function renderMistakesTab", 1)[0]

        self.assertIn("data-ok-level-actions", block)
        self.assertIn('levelButton.textContent = `Рівень ${entry.level}`', block)
        self.assertIn('ctx.startLearning({ kind: "ok", module: item.name, level: entry.level })', block)
        self.assertNotIn('ctx.navigate("ok-levels")', block)

    def test_ok_badge_is_clear_and_last_level_is_highlighted(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        block = user.split("function renderOkTab", 1)[1].split("function renderMistakesTab", 1)[0]

        self.assertIn('return match ? "ОК-" + match[1] : "ОК"', user)
        self.assertIn("entry.level === item.last_level", block)
        self.assertIn('" is-selected"', block)

    def test_backend_remembers_last_selected_ok_level(self):
        server = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("await self.store.set_ok_last_level(auth.user_id, module, level)", server)


if __name__ == "__main__":
    unittest.main()
