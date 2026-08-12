from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app import run_mini_app_notice


ROOT = Path(__file__).resolve().parents[1]


class BotMiniAppEntryTests(unittest.TestCase):
    def test_start_old_callbacks_and_old_text_open_current_mini_app(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn('@router.message(F.text.startswith("/start"))', app)
        self.assertIn("@router.callback_query()", app)
        self.assertIn("await query.answer", app)
        self.assertIn("@router.message(F.text)", app)
        self.assertIn("mini_app_markup(runtime.webapp_url)", app)

    def test_telegram_menu_button_is_configured_from_current_url(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertIn("MenuButtonWebApp", app)
        self.assertIn("runtime.bot.set_chat_menu_button", app)
        self.assertIn('text="Відкрити"', app)

    def test_admin_can_start_but_deploy_does_not_auto_start_notice(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")

        self.assertIn('@app.post("/api/admin/users/mini-app-notice")', app)
        self.assertIn("Текст повідомлення", admin)
        lifespan = app.split("async def lifespan", 1)[1].split("app = FastAPI", 1)[0]
        self.assertNotIn("run_mini_app_notice(runtime)", lifespan)


class MiniAppNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def test_notice_sends_current_button_to_every_registered_user(self):
        status = {
            "state": "running",
            "total": 0,
            "processed": 0,
            "sent": 0,
            "blocked": 0,
            "failed": 0,
        }
        runtime = SimpleNamespace(
            store=SimpleNamespace(),
            bot=SimpleNamespace(send_message=AsyncMock()),
            webapp_url="https://example.test/app",
            mini_app_notice_status=status,
        )

        with patch("app.asyncio.sleep", new=AsyncMock()):
            await run_mini_app_notice(runtime, [101, 202], "Власний текст")

        self.assertEqual("completed", status["state"])
        self.assertEqual(2, status["total"])
        self.assertEqual(2, status["processed"])
        self.assertEqual(2, status["sent"])
        self.assertEqual([101, 202], [call.kwargs["chat_id"] for call in runtime.bot.send_message.await_args_list])
        self.assertTrue(all(call.kwargs["text"] == "Власний текст" for call in runtime.bot.send_message.await_args_list))
        self.assertTrue(all(call.kwargs["parse_mode"] is None for call in runtime.bot.send_message.await_args_list))

    def test_admin_notice_supports_all_or_selected_users(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        admin = (ROOT / "static/js/screens/admin.js").read_text(encoding="utf-8")

        self.assertIn('audience: Literal["all", "selected"]', app)
        self.assertIn('payload.audience == "selected"', app)
        self.assertIn('audience: ctx.state.adminNoticeAudience', admin)
        self.assertIn('user_ids: selectedIds', admin)
        self.assertIn('text: messageText', admin)
        self.assertIn('max_length=4_000', app)


if __name__ == "__main__":
    unittest.main()
