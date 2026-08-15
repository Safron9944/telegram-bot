import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from app import AuthContext, MiniAppService
from access import access_status, access_tier, has_attestation_access
from utils import dt_to_iso, iso_to_dt, now


class AccessTierTests(unittest.TestCase):
    def test_infinite_cases_tier_stays_limited(self):
        user = {"sub_infinite": True, "sub_tier": "cases"}
        self.assertEqual("cases", access_tier(user))
        self.assertEqual((True, "sub_cases"), access_status(user))
        self.assertTrue(has_attestation_access(user))

    def test_infinite_full_tier_has_full_access(self):
        user = {"sub_infinite": True, "sub_tier": "full"}
        self.assertEqual("full", access_tier(user))
        self.assertEqual((True, "sub_infinite"), access_status(user))

    def test_legacy_trial_is_ignored(self):
        user = {"trial_end": now() + timedelta(days=3)}
        self.assertEqual("none", access_tier(user))
        self.assertEqual((False, "expired"), access_status(user))
        self.assertFalse(has_attestation_access(user))

    def test_expired_user_has_no_access(self):
        user = {"trial_end": now() - timedelta(seconds=1)}
        self.assertEqual("none", access_tier(user))
        self.assertEqual((False, "expired"), access_status(user))

    def test_new_full_subscription_does_not_unlock_protected_materials(self):
        service = MiniAppService(SimpleNamespace())
        auth = AuthContext(
            telegram_user={"id": 42},
            user={"sub_infinite": True, "sub_tier": "full", "section_access": []},
            user_id=42,
            is_admin=False,
        )

        with self.assertRaises(HTTPException) as cases_error:
            service.ensure_cases_access(auth)
        with self.assertRaises(HTTPException) as search_error:
            service.ensure_full_access(auth, "question_search")

        self.assertEqual("protected_materials_required", cases_error.exception.detail["code"])
        self.assertEqual("protected_materials_required", search_error.exception.detail["code"])

    def test_explicit_grant_unlocks_protected_materials(self):
        service = MiniAppService(SimpleNamespace())
        auth = AuthContext(
            telegram_user={"id": 42},
            user={
                "section_access": ["cases", "test_questions", "question_search"],
            },
            user_id=42,
            is_admin=False,
        )

        service.ensure_cases_access(auth)
        service.ensure_full_access(auth, "question_search")


class SavedAttestationAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_subscription_cannot_restore_attestation_session(self):
        state = {
            "mode": "learn",
            "meta": {"kind": "attestation_stage_1"},
            "last_activity_at": dt_to_iso(now()),
        }
        store = SimpleNamespace(get_ui=AsyncMock(return_value={"state": state}))
        service = MiniAppService(SimpleNamespace(store=store))
        service.build_session_view = AsyncMock(return_value={"screen": "question"})
        auth = AuthContext(
            telegram_user={"id": 42},
            user={"trial_end": now() - timedelta(seconds=1)},
            user_id=42,
            is_admin=False,
        )

        with self.assertRaises(HTTPException) as raised:
            await service.saved_view(auth)

        self.assertEqual(403, raised.exception.status_code)
        self.assertEqual("attestation_access_required", raised.exception.detail["code"])
        service.build_session_view.assert_not_awaited()


class SavedSessionResumeTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self, state):
        store = SimpleNamespace(
            get_ui=AsyncMock(return_value={"state": state}),
            set_state=AsyncMock(),
        )
        service = MiniAppService(SimpleNamespace(store=store))
        service.build_session_view = AsyncMock(return_value={"screen": "question"})
        auth = AuthContext(
            telegram_user={"id": 42},
            user={"section_access": []},
            user_id=42,
            is_admin=False,
        )
        return service, store, auth

    async def test_recent_preview_session_is_restored(self):
        state = {
            "mode": "test",
            "last_activity_at": dt_to_iso(now()),
            "meta": {"kind": "customs_preview", "preview": True},
        }
        service, store, auth = self.make_service(state)

        view = await service.saved_view(auth)

        self.assertEqual({"screen": "question"}, view)
        service.build_session_view.assert_awaited_once_with(auth, state)
        store.set_state.assert_not_awaited()

    async def test_session_older_than_24_hours_is_cleared(self):
        state = {
            "mode": "test",
            "last_activity_at": dt_to_iso(now() - timedelta(hours=25)),
            "meta": {"kind": "customs_preview", "preview": True},
        }
        service, store, auth = self.make_service(state)

        view = await service.saved_view(auth)

        self.assertIsNone(view)
        store.set_state.assert_awaited_once_with(42, {})
        service.build_session_view.assert_not_awaited()

    async def test_old_session_uses_legacy_started_at_for_expiry(self):
        state = {
            "mode": "test",
            "started_at": dt_to_iso(now() - timedelta(days=150)),
            "meta": {"kind": "customs_preview", "preview": True},
        }
        service, store, auth = self.make_service(state)

        view = await service.saved_view(auth)

        self.assertIsNone(view)
        store.set_state.assert_awaited_once_with(42, {})
        service.build_session_view.assert_not_awaited()

    async def test_legacy_session_without_timestamp_is_cleared(self):
        service, store, auth = self.make_service({"mode": "test"})

        view = await service.saved_view(auth)

        self.assertIsNone(view)
        store.set_state.assert_awaited_once_with(42, {})

    async def test_completed_result_is_not_restored(self):
        state = {
            "mode": "test_result",
            "summary": {"finished_at": dt_to_iso(now()), "percent": 80},
        }
        service, store, auth = self.make_service(state)

        view = await service.saved_view(auth)

        self.assertIsNone(view)
        store.set_state.assert_awaited_once_with(42, {})

    async def test_saved_active_state_gets_activity_timestamp(self):
        service, store, _ = self.make_service({})

        await service.set_state(42, {"mode": "learn"})

        saved = store.set_state.await_args.args[1]
        self.assertEqual("learn", saved["mode"])
        self.assertIsNotNone(iso_to_dt(saved["last_activity_at"]))


if __name__ == "__main__":
    unittest.main()
