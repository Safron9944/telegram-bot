import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from app import AuthContext, MiniAppService
from access import access_status, access_tier, has_attestation_access
from utils import now


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


if __name__ == "__main__":
    unittest.main()
