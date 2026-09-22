from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LearningOkUiTests(unittest.TestCase):
    def test_ok_learning_shows_levels_inline_without_starting_on_new_selection(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        block = user.split("function renderOkTab", 1)[1].split("function renderMistakesTab", 1)[0]

        self.assertIn("data-ok-level-actions", block)
        self.assertIn('levelButton.textContent = `Рівень ${entry.level}`', block)
        self.assertIn('if (entry.level === selectedLevel)', block)
        self.assertIn('"/api/preferences/ok-level"', block)
        self.assertIn('body: { module: item.name, level: entry.level }', block)
        self.assertNotIn('ctx.navigate("ok-levels")', block)

    def test_ok_module_title_and_selected_level_start_learning(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        block = user.split("function renderOkTab", 1)[1].split("function renderMistakesTab", 1)[0]

        self.assertIn("data-ok-module-open", block)
        self.assertIn("const startSelectedLevel", block)
        self.assertIn('ctx.startLearning({ kind: "ok", module: item.name, level: selectedLevel })', block)
        self.assertIn('row.querySelector("[data-ok-module-open]")?.addEventListener("click", startSelectedLevel)', block)

    def test_ok_badge_is_clear_and_selected_level_is_highlighted(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        block = user.split("function renderOkTab", 1)[1].split("function renderMistakesTab", 1)[0]

        self.assertIn('return match ? "ОК-" + match[1] : "ОК"', user)
        self.assertIn("item.last_level || item.levels[0]?.level", block)
        self.assertIn('entry.level === selectedLevel ? " is-selected" : ""', block)

    def test_backend_can_save_ok_level_without_starting_session(self):
        server = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("class OkLevelUpdate(BaseModel)", server)
        self.assertIn("async def set_ok_level(self, auth: AuthContext, payload: OkLevelUpdate)", server)
        self.assertIn("await self.store.set_ok_last_level(auth.user_id, module, level)", server)
        self.assertIn('@app.post("/api/preferences/ok-level")', server)


if __name__ == "__main__":
    unittest.main()
