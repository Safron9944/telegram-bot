import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from app import (
    AuthContext,
    HOME_VISIBILITY_DEFAULTS,
    MiniAppService,
    get_home_visibility,
)
from sections import UKRAINIAN_LANGUAGE_SECTION_KEY
from storage import Storage


def admin_auth(user_id: int = 1) -> AuthContext:
    return AuthContext(
        telegram_user={"id": user_id},
        user={"user_id": user_id, "is_admin": 1},
        user_id=user_id,
        is_admin=True,
    )


class HomeVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_partial_saved_visibility_uses_defaults_for_other_items(self):
        store = SimpleNamespace(
            get_setting=AsyncMock(return_value=json.dumps({"cases": False, "support": True})),
        )

        visibility = await get_home_visibility(store)

        self.assertFalse(visibility["cases"])
        self.assertTrue(visibility["support"])
        self.assertTrue(visibility["attestation"])
        self.assertEqual(set(HOME_VISIBILITY_DEFAULTS), set(visibility))

    async def test_invalid_saved_visibility_falls_back_safely(self):
        store = SimpleNamespace(get_setting=AsyncMock(return_value="not-json"))

        visibility = await get_home_visibility(store)

        self.assertEqual(HOME_VISIBILITY_DEFAULTS, visibility)


class AdminDeleteUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_can_delete_regular_user(self):
        store = SimpleNamespace(
            get_user=AsyncMock(return_value={"user_id": 42, "is_admin": 0}),
            delete_user=AsyncMock(return_value=True),
        )
        service = MiniAppService(SimpleNamespace(store=store, admin_ids={1}))

        result = await service.admin_delete_user(admin_auth(), 42)

        self.assertEqual({"ok": True, "user_id": 42}, result)
        store.delete_user.assert_awaited_once_with(42)

    async def test_admin_cannot_delete_self(self):
        store = SimpleNamespace(
            get_user=AsyncMock(return_value={"user_id": 1, "is_admin": 1}),
            delete_user=AsyncMock(return_value=True),
        )
        service = MiniAppService(SimpleNamespace(store=store, admin_ids={1}))

        with self.assertRaises(HTTPException) as raised:
            await service.admin_delete_user(admin_auth(), 1)

        self.assertEqual(400, raised.exception.status_code)
        self.assertEqual("cannot_delete_self", raised.exception.detail["code"])
        store.delete_user.assert_not_awaited()


class AdminUkrainianLanguageAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_can_grant_ukrainian_language_access(self):
        store = SimpleNamespace(set_section_access=AsyncMock(return_value=True))
        service = MiniAppService(SimpleNamespace(store=store))
        service.admin_user_detail = AsyncMock(return_value={"ukrainian_language_access": True})

        result = await service.admin_set_ukrainian_language_access(admin_auth(), 42, True)

        self.assertTrue(result["ukrainian_language_access"])
        store.set_section_access.assert_awaited_once_with(42, UKRAINIAN_LANGUAGE_SECTION_KEY, True)
        service.admin_user_detail.assert_awaited_once_with(admin_auth(), 42)

    async def test_missing_user_is_reported(self):
        store = SimpleNamespace(set_section_access=AsyncMock(return_value=False))
        service = MiniAppService(SimpleNamespace(store=store))

        with self.assertRaises(HTTPException) as raised:
            await service.admin_set_ukrainian_language_access(admin_auth(), 404, True)

        self.assertEqual("user_not_found", raised.exception.detail["code"])


class _AsyncContext:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _DeleteConnection:
    def __init__(self, exists=True):
        self.exists = exists
        self.statements = []

    def transaction(self):
        return _AsyncContext()

    async def fetchval(self, sql, *params):
        self.statements.append((sql, params))
        return 1 if self.exists else None

    async def execute(self, sql, *params):
        self.statements.append((sql, params))


class StorageDeleteUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_user_removes_all_owned_rows_before_profile(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))

        deleted = await storage.delete_user(42)

        self.assertTrue(deleted)
        delete_tables = [
            sql.split("DELETE FROM ", 1)[1].split(" ", 1)[0]
            for sql, _ in connection.statements
            if sql.startswith("DELETE FROM ")
        ]
        self.assertEqual(["ui_state", "errors", "tests", "users"], delete_tables)
        self.assertTrue(all(params == (42,) for _, params in connection.statements))

    async def test_delete_user_does_nothing_when_profile_is_missing(self):
        connection = _DeleteConnection(exists=False)
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))

        deleted = await storage.delete_user(42)

        self.assertFalse(deleted)
        self.assertEqual(1, len(connection.statements))

    async def test_set_section_access_grants_and_revokes_exact_section(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))

        self.assertTrue(await storage.set_section_access(42, UKRAINIAN_LANGUAGE_SECTION_KEY, True))
        self.assertTrue(await storage.set_section_access(42, UKRAINIAN_LANGUAGE_SECTION_KEY, False))

        statements = [(" ".join(sql.split()), params) for sql, params in connection.statements]
        self.assertIn(
            (
                "INSERT INTO user_section_access(user_id, section_key) VALUES($1, $2) ON CONFLICT(user_id, section_key) DO NOTHING",
                (42, UKRAINIAN_LANGUAGE_SECTION_KEY),
            ),
            statements,
        )
        self.assertIn(
            (
                "DELETE FROM user_section_access WHERE user_id=$1 AND section_key=$2",
                (42, UKRAINIAN_LANGUAGE_SECTION_KEY),
            ),
            statements,
        )

    async def test_admin_cannot_delete_another_admin(self):
        store = SimpleNamespace(
            get_user=AsyncMock(return_value={"user_id": 2, "is_admin": 1}),
            delete_user=AsyncMock(return_value=True),
        )
        service = MiniAppService(SimpleNamespace(store=store, admin_ids={1, 2}))

        with self.assertRaises(HTTPException) as raised:
            await service.admin_delete_user(admin_auth(), 2)

        self.assertEqual(400, raised.exception.status_code)
        self.assertEqual("cannot_delete_admin", raised.exception.detail["code"])
        store.delete_user.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
