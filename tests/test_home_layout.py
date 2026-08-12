from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HomeLayoutTests(unittest.TestCase):
    def test_original_home_layout_is_preserved(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        home = user.split("/* ===================== CUSTOMS", 1)[0]
        self.assertIn('class="home-primary"', home)
        self.assertIn('header: "Матеріали"', home)
        self.assertIn('header: "Допомога"', home)
        self.assertNotIn('header: "Розділи"', home)
        self.assertIn("Митні компетенції, атестація та робота з матеріалами", home)


if __name__ == "__main__":
    unittest.main()
