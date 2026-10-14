import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from app import (
    AuthContext,
    HOME_VISIBILITY_DEFAULTS,
    MiniAppService,
    get_home_visibility,
)
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
        store.delete_user.assert_awaited_once_with(42, actor_id=1)

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


class AdminSectionAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_controls_protected_section_visibility_without_granting_access(self):
        store = SimpleNamespace(
            get_user=AsyncMock(return_value={"user_id": 42}),
            set_section_visibility=AsyncMock(return_value=True),
            get_setting=AsyncMock(return_value="{}"),
            list_attestation_banks_for_admin=AsyncMock(return_value=[]),
        )
        service = MiniAppService(SimpleNamespace(store=store))
        service.admin_user_detail = AsyncMock(return_value={"section_controls": []})

        with patch("app.CommerceStore") as ledger:
            ledger.return_value.set_visibility = AsyncMock(return_value=True)
            await service.admin_set_section_access(admin_auth(), 42, "cases", True)
            ledger.return_value.set_visibility.assert_awaited_once_with(42, ["cases"], True, 1, "Ручна зміна видимості розділу")
        self.assertFalse(hasattr(store, "set_section_access_override"))


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
        return 1 if self.exists and ("FROM users" in sql or "FROM attestation_banks WHERE id" in sql) else None

    async def fetchrow(self, sql, *params):
        self.statements.append((sql, params))
        return None

    async def execute(self, sql, *params):
        self.statements.append((sql, params))


class StorageDeleteUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_user_removes_all_owned_rows_before_profile(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))
        storage._commerce_initialized = True

        deleted = await storage.delete_user(42)

        self.assertTrue(deleted)
        delete_tables = [
            sql.split("DELETE FROM ", 1)[1].split(" ", 1)[0]
            for sql, _ in connection.statements
            if sql.startswith("DELETE FROM ")
        ]
        self.assertEqual(["ui_state", "errors", "tests", "users"], delete_tables)
        self.assertTrue(all(params == (42,) for sql, params in connection.statements if sql.startswith("DELETE FROM ")))

    async def test_delete_user_does_nothing_when_profile_is_missing(self):
        connection = _DeleteConnection(exists=False)
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))
        storage._commerce_initialized = True

        deleted = await storage.delete_user(42)

        self.assertFalse(deleted)
        self.assertEqual(2, len(connection.statements))

    async def test_set_section_access_grants_and_revokes_exact_section(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))
        storage._commerce_initialized = True

        self.assertTrue(await storage.set_section_access(42, "attestation:7", True))
        self.assertTrue(await storage.set_section_access(42, "attestation:7", False))

        statements = [(" ".join(sql.split()), params) for sql, params in connection.statements]
        self.assertTrue(any("INSERT INTO manual_access_grants" in sql and params[:2] == (42,"section:attestation:7") for sql,params in statements))
        self.assertIn(("DELETE FROM manual_access_grants WHERE user_id=$1 AND product_key=$2",(42,"section:attestation:7")),statements)
        self.assertFalse(any("DELETE FROM user_section_access" in sql for sql,_ in statements))

    async def test_set_section_access_override_changes_only_a_manual_grant(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))
        storage._commerce_initialized = True
        self.assertTrue(await storage.set_section_access_override(42,"customs",False))
        statements = [(" ".join(sql.split()),params) for sql,params in connection.statements]
        self.assertIn(("DELETE FROM manual_access_grants WHERE user_id=$1 AND product_key=$2",(42,"section:customs")),statements)
        self.assertTrue(any("INSERT INTO access_events" in sql for sql,_ in statements))
        self.assertFalse(any("DELETE FROM payments" in sql or "DELETE FROM legacy_access" in sql for sql,_ in statements))

    async def test_set_section_visibility_upserts_independent_display_decision(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))
        storage._commerce_initialized = True
        self.assertTrue(await storage.set_section_visibility(42,"cases",True))
        self.assertTrue(any("INSERT INTO user_section_visibility_overrides" in sql and params == (42,"cases",True) for sql,params in connection.statements))
        self.assertTrue(any("INSERT INTO access_events" in sql for sql,_ in connection.statements))

    async def test_paid_section_grant_uses_durable_ledger(self):
        connection = _DeleteConnection()
        storage = Storage("postgresql://unused")
        storage.pool = SimpleNamespace(acquire=lambda: _AsyncContext(connection))
        storage._commerce_initialized = True
        await storage.grant_section_access(42,"customs",telegram_charge_id="charge-1")
        self.assertTrue(any("INSERT INTO payments" in sql and "charge-1" in params for sql,params in connection.statements))
        self.assertFalse(any("DELETE FROM payments" in sql for sql,_ in connection.statements))

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
