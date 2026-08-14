from pathlib import Path
from types import SimpleNamespace
import unittest

from app import AuthContext, MiniAppService


ROOT = Path(__file__).resolve().parents[1]


class PreviewStore:
    def __init__(self):
        self.state = {}

    async def set_state(self, user_id, state):
        self.state[user_id] = state

    async def get_ui(self, user_id):
        return {"state": self.state.get(user_id, {})}


class CustomsPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_unpaid_user_can_start_only_first_50_questions(self):
        questions = {
            qid: SimpleNamespace(
                id=qid,
                section="Law",
                topic="Preview",
                ok=None,
                level=None,
                qnum=qid,
                question=f"Питання {qid}",
                choices=["A", "B"],
                correct=[1],
                correct_texts=["A"],
                shuffle_choices=False,
            )
            for qid in range(1, 81)
        }
        qb = SimpleNamespace(law=list(questions), ok_modules={}, by_id=questions)
        store = PreviewStore()
        service = MiniAppService(SimpleNamespace(qb=qb, store=store))
        auth = AuthContext({}, {"section_access": []}, 42, False)

        result = await service.start_customs_preview(auth)

        self.assertEqual(50, result["progress"]["total"])
        self.assertEqual(list(range(1, 51)), store.state[42]["pending"])
        self.assertEqual({"kind": "customs_preview", "preview": True}, store.state[42]["meta"])
        service.ensure_session_access(auth, store.state[42])


class CustomsPreviewAssetsTests(unittest.TestCase):
    def test_home_and_customs_screen_offer_first_50(self):
        user = (ROOT / "static/js/screens/user.js").read_text(encoding="utf-8")
        app = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        server = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("Перші 50 питань безкоштовно", user)
        self.assertIn("ctx.startCustomsPreview", user)
        self.assertIn("/api/learning/preview/start", app)
        self.assertIn('@app.post("/api/learning/preview/start")', server)
        self.assertNotIn("trial-notice", user)

    def test_removed_trial_and_free_sections_are_wired_server_side(self):
        access = (ROOT / "access.py").read_text(encoding="utf-8")
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        storage = (ROOT / "storage.py").read_text(encoding="utf-8")
        sections = (ROOT / "sections.py").read_text(encoding="utf-8")
        self.assertNotIn("trial_full", access)
        self.assertNotIn("start_trial_if_needed", app)
        self.assertIn("trial_removed_v1", storage)
        self.assertIn('ALWAYS_FREE_SECTION_KEYS = frozenset({"customs_code", "support"})', sections)

    def test_preview_assets_have_fresh_cache_version(self):
        index = (ROOT / "static/index.html").read_text(encoding="utf-8")
        app = (ROOT / "static/app.js").read_text(encoding="utf-8")
        module = (ROOT / "static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("20260814-no-trial-preview-01", index)
        self.assertIn("20260814-no-trial-preview-01", app)
        self.assertIn("screens/user.js?v=20260814-no-trial-preview-01", module)


if __name__ == "__main__":
    unittest.main()
