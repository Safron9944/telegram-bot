"""Commerce integration tests use an isolated in-memory SQL database, never production."""
import json
import re
import sqlite3
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from access import access_tier
from app import AuthContext, MiniAppService, access_payload
from sections import build_sections
from storage import Storage
from utils import now


class Context:
    def __init__(self, value=None):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        return False


class Transaction(Context):
    async def __aenter__(self):
        self.value.db.execute("BEGIN")
        return self.value

    async def __aexit__(self, kind, *args):
        self.value.db.rollback() if kind else self.value.db.commit()
        return False


class SQLConnection:
    """Adapt PostgreSQL placeholders/types only; execute the actual commerce SQL."""
    def __init__(self):
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.create_function("now", 0, lambda: now().isoformat())
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE users(user_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT,
                is_admin INT DEFAULT 0, created_at TEXT, trial_start TEXT, trial_end TEXT, sub_end TEXT, sub_infinite INT DEFAULT 0,
                sub_tier TEXT, ok_modules_json TEXT DEFAULT '[]', ok_last_levels_json TEXT DEFAULT '{}');
            CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE user_section_access(user_id INTEGER,section_key TEXT,purchased_at TEXT,
                telegram_charge_id TEXT, PRIMARY KEY(user_id,section_key));
            CREATE TABLE user_section_access_overrides(user_id INTEGER,section_key TEXT,enabled BOOLEAN,
                updated_at TEXT, PRIMARY KEY(user_id,section_key));
            CREATE TABLE user_section_visibility_overrides(user_id INTEGER,section_key TEXT,visible BOOLEAN,
                updated_at TEXT, PRIMARY KEY(user_id,section_key));
            CREATE TABLE attestation_banks(id INTEGER PRIMARY KEY, title TEXT, slug TEXT,
                source_id TEXT, status TEXT, display_order INTEGER, questions_count INTEGER,
                created_at TEXT, updated_at TEXT, ever_published BOOLEAN DEFAULT FALSE,
                source_hash TEXT, source_version TEXT);
            CREATE TABLE published_attestation_questions(id INTEGER PRIMARY KEY,bank_id INTEGER,
                source_key TEXT,qnum INTEGER,topic TEXT,question TEXT,choices TEXT,correct TEXT,
                correct_texts TEXT,shuffle_choices BOOLEAN DEFAULT FALSE);
            CREATE TABLE ui_state(user_id INTEGER PRIMARY KEY,chat_id INTEGER,main_message_id INTEGER,state_json TEXT);
            CREATE TABLE errors(user_id INTEGER,qid INTEGER,wrong_count INTEGER DEFAULT 0,in_mistakes INTEGER DEFAULT 0,PRIMARY KEY(user_id,qid));
            CREATE TABLE tests(id INTEGER PRIMARY KEY,user_id INTEGER,started_at TEXT,finished_at TEXT,total INTEGER,correct INTEGER,percent REAL);
        """)
        self.statements = []

    def transaction(self):
        return Transaction(self)

    def run(self, sql, args):
        self.statements.append((sql, args))
        if "pg_advisory_xact_lock" in sql:
            return None
        if sql.startswith("ALTER TABLE access_events ADD COLUMN IF NOT EXISTS product_title"):
            if any(row["name"] == "product_title" for row in self.db.execute("PRAGMA table_info(access_events)")):
                return None
            sql = sql.replace("ADD COLUMN IF NOT EXISTS","ADD COLUMN")
        sql = re.sub(r"\bBIGSERIAL PRIMARY KEY\b", "INTEGER PRIMARY KEY AUTOINCREMENT", sql)
        sql = sql.replace("FOR UPDATE", "").replace("DEFAULT now()", "DEFAULT (now())").replace("ILIKE","LIKE")
        bindings = []
        def bind(match):
            value = args[int(match.group(1)) - 1]
            bindings.append(value.isoformat() if isinstance(value, datetime) else value)
            return "?"
        sql = re.sub(r"\$(\d+)", bind, sql)
        return self.db.execute(sql, bindings)

    async def execute(self, sql, *args):
        self.run(sql, args)

    async def fetch(self, sql, *args):
        cursor = self.run(sql, args)
        rows = [dict(row) for row in cursor.fetchall()] if cursor else []
        for row in rows:
            for key in ("created_at", "paid_at", "purchased_at", "expires_at", "sub_end"):
                if isinstance(row.get(key), str):
                    row[key] = datetime.fromisoformat(row[key])
        return rows

    async def fetchrow(self, sql, *args):
        rows = await self.fetch(sql, *args)
        return rows[0] if rows else None

    async def fetchval(self, sql, *args):
        row = await self.fetchrow(sql, *args)
        return next(iter(row.values())) if row else None


def auth(admin=True):
    return AuthContext({"id": 1}, {"user_id": 1}, 1, admin)


class LegacySafetyRedTest(unittest.IsolatedAsyncioTestCase):
    async def test_old_admin_revoke_does_not_destroy_unknown_legacy_subscription(self):
        con = SQLConnection()
        self.addCleanup(con.db.close)
        store = Storage("unused")
        store.pool = SimpleNamespace(acquire=lambda: Context(con))
        con.db.execute("INSERT INTO users(user_id,sub_tier,sub_infinite) VALUES(42,'full',1)")
        await store.set_admin_access(42, "none")
        row = await con.fetchrow("SELECT * FROM users WHERE user_id=42")
        self.assertEqual("full", access_tier(row), "Ordinary gift revocation must preserve unknown legacy rights")


class CommerceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from commerce import CommerceStore
        self.con = SQLConnection()
        self.addCleanup(self.con.db.close)
        self.store = Storage("unused")
        self.store.pool = SimpleNamespace(acquire=lambda: Context(self.con))
        self.commerce = CommerceStore(self.store)
        await self.commerce.initialize(self.con)
        await self.store.ensure_user(42, first_name="Олена")

    async def buy(self, key="full", charge="charge-1", **changes):
        payload = dict(user_id=42, product_key=key, product_title="Назва на час оплати",
            amount=250, currency="XTR", reference=charge, source="telegram", paid_at=now())
        payload.update(changes)
        return await self.commerce.record_payment(**payload)

    async def test_duplicate_telegram_charge_is_one_immutable_payment(self):
        self.assertTrue(await self.buy())
        self.assertFalse(await self.buy(amount=999, product_title="Змінена назва"))
        rows = (await self.commerce.user_commerce(42))["purchases"]
        self.assertEqual(1, len(rows))
        self.assertEqual((250, "Назва на час оплати"), (rows[0]["amount"], rows[0]["product_title"]))

    async def test_payment_ensures_user_without_demoting_existing_admin(self):
        await self.buy(user_id=77)
        self.assertTrue(await self.store.get_user(77))
        await self.store.ensure_user(42, is_admin=True)
        await self.buy(charge="another")
        self.assertTrue((await self.store.get_user(42))["is_admin"])

    async def test_full_then_cases_does_not_reduce_stronger_tier(self):
        await self.buy()
        await self.buy("cases", "charge-2", amount=100)
        self.assertEqual("full", access_tier(await self.store.get_user(42)))

    async def test_paid_package_payload_reports_permanent_effective_access(self):
        await self.buy()
        payload = access_payload(await self.store.get_user(42))
        self.assertTrue(payload["sub_infinite"])
        self.assertEqual("sub_infinite",payload["state"])

    async def test_paid_section_ignores_false_access_and_visibility_overrides(self):
        await self.buy("section:cases")
        await self.store.set_section_access_override(42, "cases", False)
        self.con.db.execute("INSERT INTO user_section_access_overrides VALUES(42,'cases',FALSE,?)",(now().isoformat(),))
        self.con.db.execute("INSERT INTO user_section_visibility_overrides VALUES(42,'cases',FALSE,?)",(now().isoformat(),))
        user = await self.store.get_user(42)
        section = next(item for item in await build_sections(self.store, user) if item["key"] == "cases")
        self.assertTrue(section["has_access"])
        self.assertTrue(section["visible"])
        self.assertEqual("paid", section["access_source"])

    async def test_paid_visibility_mutation_records_the_preserved_result(self):
        await self.buy("section:cases")
        await self.store.set_section_visibility(42,"cases",False)
        user = await self.store.get_user(42)
        self.assertTrue(user["section_visibility_overrides"]["cases"])
        event = (await self.commerce.user_commerce(42))["access_events"][0]
        self.assertEqual("section_shown",event["action"])

    async def test_revoke_manual_gift_keeps_subsequent_purchase(self):
        await self.commerce.set_manual_grant(42, "full", "Повний доступ", True, 1, "Подарунок")
        await self.buy()
        await self.commerce.set_manual_grant(42, "full", "Повний доступ", False, 1, "Скасування подарунка")
        user = await self.store.get_user(42)
        self.assertEqual("full", access_tier(user))
        self.assertEqual([], (await self.commerce.user_commerce(42))["manual_grants"])

    async def test_old_direct_access_api_only_removes_manual_grants(self):
        await self.buy()
        await self.store.set_admin_access(42, "full")
        await self.store.set_admin_access(42, "none")
        self.assertEqual("full", access_tier(await self.store.get_user(42)))

    async def test_old_direct_section_api_cannot_delete_paid_rights(self):
        await self.buy("section:customs")
        await self.store.set_section_access(42, "customs", False)
        self.assertIn("customs", (await self.store.get_user(42))["section_access"])

    async def test_hidden_purchased_bank_stays_visible_but_new_draft_is_not_exposed(self):
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(7,'Куплений банк','second','second','hidden',0,5)")
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(8,'Чернетка','draft','draft','draft',1,0)")
        await self.buy("section:attestation:7")
        await self.buy("full", "full-charge")
        items = await build_sections(self.store, await self.store.get_user(42))
        self.assertTrue(next(row for row in items if row["key"] == "attestation:7")["visible"])
        self.assertFalse(next(row for row in items if row["key"] == "attestation:8")["visible"])

    async def test_refund_before_payment_never_activates_charge(self):
        await self.commerce.record_refund("charge-1", actor_id=1)
        await self.buy()
        self.assertEqual("none", access_tier(await self.store.get_user(42)))
        self.assertEqual("refunded", (await self.commerce.user_commerce(42))["purchases"][0]["status"])

    async def test_refund_then_duplicate_success_does_not_reactivate(self):
        await self.buy()
        await self.commerce.record_refund("charge-1", actor_id=1)
        await self.buy()
        self.assertEqual("none", access_tier(await self.store.get_user(42)))

    async def test_external_request_retry_is_idempotent_and_conflicting_retry_rejected(self):
        args = dict(source="admin_external", reference="bank-ref", request_id="request-1", actor_id=1, reason="Оплата на рахунок", currency="UAH")
        self.assertTrue(await self.buy(**args))
        self.assertFalse(await self.buy(**args))
        with self.assertRaises(ValueError):
            await self.buy(**args, amount=251)
        self.assertEqual(1, len((await self.commerce.user_commerce(42))["purchases"]))

    async def test_legacy_migration_preserves_unknown_amount_and_provenance(self):
        self.con.db.execute("DELETE FROM settings WHERE key='commerce_provenance_v1'")
        self.con.db.execute("UPDATE users SET sub_tier='full',sub_infinite=1 WHERE user_id=42")
        self.con.db.execute("INSERT INTO user_section_access VALUES(42,'customs',?,'old-charge')", (now().isoformat(),))
        self.con.db.execute("INSERT INTO user_section_access VALUES(42,'attestation:9',?,NULL)", (now().isoformat(),))
        await self.commerce.initialize(self.con)
        detail = await self.commerce.user_commerce(42)
        self.assertEqual(None, detail["purchases"][0]["amount"])
        self.assertEqual("legacy_telegram", detail["purchases"][0]["source"])
        self.assertEqual({"full", "section:attestation:9"}, {r["product_key"] for r in detail["legacy_access"]})
        await self.store.set_admin_access(42, "none")
        self.assertEqual("full", access_tier(await self.store.get_user(42)))

    async def test_deletion_is_blocked_for_payments_and_unknown_legacy(self):
        await self.buy()
        with self.assertRaises(ValueError):
            await self.store.delete_user(42)
        self.assertTrue(await self.store.get_user(42))

    async def test_deletion_is_blocked_for_unknown_legacy_section(self):
        self.con.db.execute("DELETE FROM settings WHERE key='commerce_provenance_v1'")
        self.con.db.execute("INSERT INTO user_section_access VALUES(42,'customs',?,NULL)",(now().isoformat(),))
        await self.commerce.initialize(self.con)
        with self.assertRaises(ValueError):
            await self.store.delete_user(42)
        self.assertTrue(await self.store.get_user(42))

    async def test_manual_only_user_can_be_deleted_and_audit_is_retained(self):
        await self.commerce.set_manual_grant(42,"full","Повний доступ",True,1,"Подарунок")
        self.assertTrue(await self.store.delete_user(42,actor_id=1))
        self.assertFalse(await self.store.get_user(42))
        events = await self.commerce.activity_page(0,100)
        self.assertEqual("user_deleted",events["items"][0]["action"])
        self.assertEqual(1,events["items"][0]["actor_id"])

    async def test_manual_mutation_rolls_back_when_audit_fails(self):
        execute = self.con.execute
        async def fail_audit(sql, *args):
            if "INSERT INTO access_events" in sql:
                raise RuntimeError("audit unavailable")
            return await execute(sql, *args)
        self.con.execute = fail_audit
        with self.assertRaises(RuntimeError):
            await self.commerce.set_manual_grant(42, "full", "Повний доступ", True, 1, "Подарунок")
        self.assertEqual([], (await self.commerce.user_commerce(42))["manual_grants"])

    async def test_restore_preserves_controls_for_unpurchased_sections(self):
        await self.buy("section:customs")
        await self.store.set_section_visibility(42,"cases",True)
        await self.commerce.restore_purchases(42,1,"Відновлення після перевірки")
        self.assertTrue((await self.store.get_user(42))["section_visibility_overrides"]["cases"])

    async def test_paid_bank_deletion_keeps_content_as_hidden_material(self):
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count,ever_published) VALUES(7,'Куплений банк','second','second','published',0,5,TRUE)")
        await self.buy("section:attestation:7")
        self.assertTrue(await self.store.delete_attestation_bank(7))
        bank = await self.con.fetchrow("SELECT * FROM attestation_banks WHERE id=7")
        self.assertIsNotNone(bank)
        self.assertEqual("hidden",bank["status"])

    async def test_migrated_manual_override_can_be_revoked(self):
        self.con.db.execute("DELETE FROM settings WHERE key='commerce_provenance_v1'")
        self.con.db.execute("INSERT INTO user_section_access_overrides VALUES(42,'customs',TRUE,?)",(now().isoformat(),))
        await self.commerce.initialize(self.con)
        await self.commerce.set_manual_grant(42,"section:customs","Митні компетенції",False,1,"Скасування подарунка")
        section = next(row for row in await build_sections(self.store,await self.store.get_user(42)) if row["key"] == "customs")
        self.assertFalse(section["has_access"])

    async def test_first_hide_remembers_that_a_new_bank_was_published(self):
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(7,'Новий банк','second','second','published',0,5)")
        await self.store.set_attestation_bank_visibility(7,visible=False)
        self.assertTrue((await self.con.fetchrow("SELECT * FROM attestation_banks WHERE id=7"))["ever_published"])

    async def test_sync_ignores_unrelated_transactions_and_reports_pagination(self):
        from commerce import sync_star_transactions
        partner = lambda payload: SimpleNamespace(type="user",transaction_type="invoice_payment",user=SimpleNamespace(id=42,first_name="Олена",last_name=None),invoice_payload=payload)
        bot = SimpleNamespace(get_star_transactions=AsyncMock(return_value=SimpleNamespace(transactions=[
            SimpleNamespace(id="sync-1", amount=100,date=now(), source=partner("cases"),receiver=None),
            SimpleNamespace(id="irrelevant", amount=5,date=now(), source=partner("unrecognized"),receiver=None),
        ])))
        result = await sync_star_transactions(bot, self.store, {"cases":"Атестація"}, offset=100, limit=2, actor_id=1)
        self.assertEqual((1,1,102,True), (result["imported"],result["skipped"],result["next_offset"],result["has_more"]))
        bot.get_star_transactions.assert_awaited_once_with(offset=100,limit=2)

    async def test_admin_detail_and_guarded_mutations_follow_contract(self):
        service = MiniAppService(SimpleNamespace(store=self.store,admin_ids={1}))
        detail = await service.admin_user_detail(auth(),42)
        for key in ("purchases","manual_grants","legacy_access","access_events","grant_products","commerce_summary"):
            self.assertIn(key,detail)
        with self.assertRaises(HTTPException) as raised:
            await service.admin_set_grant(auth(False),42,"full",True,"Подарунок")
        self.assertEqual(403,raised.exception.status_code)
        with self.assertRaises(HTTPException) as raised:
            await service.admin_set_grant(auth(),999,"full",True,"Подарунок")
        self.assertEqual(404,raised.exception.status_code)
        with self.assertRaises(HTTPException) as raised:
            await service.admin_set_grant(auth(),42,"full",True,"")
        self.assertEqual(400,raised.exception.status_code)

    async def test_user_filters_use_payment_and_manual_provenance(self):
        await self.buy()
        await self.store.ensure_user(43,first_name="Марія")
        await self.commerce.set_manual_grant(43,"cases","Атестація",True,1,"Подарунок")
        self.assertEqual([42],[row["user_id"] for row in await self.store.list_users(0,10,access_filter="paid")])
        self.assertEqual([43],[row["user_id"] for row in await self.store.list_users(0,10,access_filter="manual")])

    async def test_access_counts_include_confirmed_and_manual_packages(self):
        await self.buy()
        await self.store.ensure_user(43)
        await self.commerce.set_manual_grant(43,"full","Повний доступ",True,1,"Подарунок")
        self.assertEqual({"active":2,"expired":0},await self.store.users_access_counts())

    async def test_manual_only_bank_is_not_included_in_full_purchase(self):
        from questions import QuestionBank
        qb = QuestionBank("unused")
        qb.register_attestation_bank("private","Закритий",[],source_id="bundled-private",manual_grant_section_key="private")
        await self.buy()
        service = MiniAppService(SimpleNamespace(store=self.store,qb=qb))
        with self.assertRaises(HTTPException):
            service.ensure_attestation_access(AuthContext({},await self.store.get_user(42),42,False),"private")

    async def test_pagination_and_external_payment_validation(self):
        from app import AdminPaymentRequest
        service = MiniAppService(SimpleNamespace(store=self.store,admin_ids={1}))
        await service.admin_record_payment(auth(),42,AdminPaymentRequest(product_key="full",amount=12300,currency="UAH",reference="bank-1",reason="Оплата через банк",request_id="req-1"))
        page = await service.admin_payments(auth(),"",0,1)
        self.assertEqual((1,False,False),(len(page["items"]),page["has_next"],page["has_prev"]))
        self.assertEqual(12300,page["items"][0]["amount"])
        activity = await service.admin_activity(auth(),0,1)
        self.assertEqual(1,len(activity["items"]))
        self.assertEqual(1,activity["items"][0]["actor_id"])
        for kwargs in ({"amount":-1},{"amount":1.5},{"currency":"USD"},{"product_key":"section:unknown"},{"reason":""}):
            args = dict(product_key="full",amount=12300,currency="UAH",reference="bank-1",reason="Оплата",request_id="req-2")
            args.update(kwargs)
            with self.assertRaises(HTTPException) as raised:
                await service.admin_record_payment(auth(),42,AdminPaymentRequest(**args))
            self.assertEqual(400,raised.exception.status_code)

    async def test_sync_refund_before_invoice_is_safe_and_unknown_refunds_are_ignored(self):
        from commerce import sync_star_transactions
        partner = lambda payload: SimpleNamespace(type="user",transaction_type="invoice_payment",user=SimpleNamespace(id=42,first_name="Олена",last_name=None),invoice_payload=payload)
        bot = SimpleNamespace(get_star_transactions=AsyncMock(return_value=SimpleNamespace(transactions=[
            SimpleNamespace(id="sync-refund",amount=100,date=now(),source=None,receiver=partner("cases")),
            SimpleNamespace(id="irrelevant-refund",amount=100,date=now(),source=None,receiver=partner("other")),
            SimpleNamespace(id="sync-refund",amount=100,date=now(),source=partner("cases"),receiver=None),
        ])))
        result = await sync_star_transactions(bot,self.store,{"cases":"Атестація"},offset=0,limit=100,actor_id=1)
        self.assertEqual((2,1),(result["imported"],result["skipped"]))
        self.assertEqual("none",access_tier(await self.store.get_user(42)))
        self.assertEqual(0,(await self.store.get_user(42))["commerce_summary"]["paid_count"])

    async def test_bot_payment_and_refund_handlers_use_durable_ledger(self):
        from app import build_bot_router
        router = build_bot_router(SimpleNamespace(store=self.store))
        callbacks = {handler.callback.__name__:handler.callback for handler in router.message.handlers}
        message = SimpleNamespace(from_user=SimpleNamespace(id=42,first_name="Олена",last_name=None),date=now(),answer=AsyncMock(),
            successful_payment=SimpleNamespace(invoice_payload="full",total_amount=250,currency="XTR",telegram_payment_charge_id="bot-charge"))
        await callbacks["successful_payment"](message)
        await callbacks["successful_payment"](message)
        self.assertEqual(1,len((await self.commerce.user_commerce(42))["purchases"]))
        await callbacks["refunded_payment"](SimpleNamespace(refunded_payment=SimpleNamespace(telegram_payment_charge_id="bot-charge")))
        self.assertEqual("none",access_tier(await self.store.get_user(42)))

    async def test_accepted_late_payment_for_missing_bank_is_recorded_with_attention(self):
        from app import build_bot_router
        router = build_bot_router(SimpleNamespace(store=self.store))
        callback = next(handler.callback for handler in router.message.handlers if handler.callback.__name__ == "successful_payment")
        message = SimpleNamespace(from_user=SimpleNamespace(id=42,first_name="Олена",last_name=None),date=now(),answer=AsyncMock(),
            successful_payment=SimpleNamespace(invoice_payload="section:attestation:999",total_amount=100,currency="XTR",telegram_payment_charge_id="late-missing"))
        await callback(message)
        detail = await self.commerce.user_commerce(42)
        self.assertEqual(1,len(detail["purchases"]))
        self.assertIn("невідома",detail["purchases"][0]["product_title"])
        self.assertEqual("section:attestation:999",detail["unavailable_purchases"][0]["product_key"])
        self.assertGreater((await self.store.get_user(42))["commerce_summary"]["attention_count"],0)
        self.assertEqual([42],[row["user_id"] for row in await self.store.list_users(0,10,access_filter="attention")])
        self.assertNotIn("активовано",message.answer.await_args.args[0])

    async def test_accepted_precheckout_then_withdrawal_keeps_bank_for_late_payment(self):
        from app import build_bot_router
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(7,'Первісна назва','second','second','published',0,5)")
        router = build_bot_router(SimpleNamespace(store=self.store))
        query = SimpleNamespace(from_user=SimpleNamespace(id=42),invoice_payload="section:attestation:7",total_amount=100,answer=AsyncMock())
        await router.pre_checkout_query.handlers[0].callback(query)
        query.answer.assert_awaited_once_with(ok=True)
        deletion = await self.store.delete_attestation_bank(7)
        self.assertIsNotNone(await self.con.fetchrow("SELECT id FROM attestation_banks WHERE id=7"))
        self.assertEqual("archived",deletion["status"])
        callback = next(handler.callback for handler in router.message.handlers if handler.callback.__name__ == "successful_payment")
        message = SimpleNamespace(from_user=SimpleNamespace(id=42,first_name="Олена",last_name=None),date=now(),answer=AsyncMock(),
            successful_payment=SimpleNamespace(invoice_payload="section:attestation:7",total_amount=100,currency="XTR",telegram_payment_charge_id="late-retained"))
        await callback(message)
        detail = await self.commerce.user_commerce(42)
        self.assertEqual("Первісна назва",detail["purchases"][0]["product_title"])
        self.assertEqual([],detail["unavailable_purchases"])
        section = next(row for row in await build_sections(self.store,await self.store.get_user(42)) if row["key"] == "attestation:7")
        self.assertTrue(section["visible"])

    async def test_historical_hidden_migration_preserves_packages_but_never_marks_future_drafts(self):
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(7,'Старий прихований','second','second','hidden',0,5)")
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(8,'Чернетка','draft','draft','draft',1,0)")
        await self.store._migrate_attestation_publication_history(self.con)
        self.assertTrue((await self.con.fetchrow("SELECT ever_published FROM attestation_banks WHERE id=7"))["ever_published"])
        self.assertFalse((await self.con.fetchrow("SELECT ever_published FROM attestation_banks WHERE id=8"))["ever_published"])
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(9,'Майбутній прихований','future','future','hidden',2,0)")
        await self.store._migrate_attestation_publication_history(self.con)
        self.assertFalse((await self.con.fetchrow("SELECT ever_published FROM attestation_banks WHERE id=9"))["ever_published"])
        for field in ("paid_products","legacy_products"):
            for tier in ("full","cases"):
                items = await build_sections(self.store,{field:[tier]})
                self.assertTrue(next(row for row in items if row["key"] == "attestation:7")["visible"])
                self.assertFalse(next(row for row in items if row["key"] == "attestation:9")["visible"])
        self.assertEqual("archived",(await self.store.delete_attestation_bank(7))["status"])

    async def test_revoke_existing_gift_after_material_is_gone(self):
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(8,'Відомий подарунок','draft','draft','draft',1,0)")
        service = MiniAppService(SimpleNamespace(store=self.store,admin_ids={1}))
        await service.admin_set_grant(auth(),42,"section:attestation:8",True,"Подарунок")
        self.con.db.execute("DELETE FROM attestation_banks WHERE id=8")
        detail = await service.admin_set_grant(auth(),42,"section:attestation:8",False,"Скасування подарунка")
        self.assertEqual([],detail["manual_grants"])
        self.assertEqual("Відомий подарунок",detail["access_events"][0]["product_title"])
        with self.assertRaises(HTTPException):
            await service.admin_set_grant(auth(),42,"section:attestation:8",True,"Новий подарунок")

    async def test_gifted_draft_is_retained_when_withdrawn(self):
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(8,'Чернетка з дозволом','draft','draft','draft',1,0)")
        await self.commerce.set_manual_grant(42,"section:attestation:8","Чернетка з дозволом",True,1,"Подарунок")
        self.assertEqual("archived",(await self.store.delete_attestation_bank(8))["status"])
        self.assertIsNotNone(await self.con.fetchrow("SELECT id FROM attestation_banks WHERE id=8"))

    async def test_gift_creation_rechecks_material_after_catalog_validation(self):
        with self.assertRaises(ValueError):
            await self.commerce.set_manual_grant(42,"section:attestation:999","Розділ вже вилучено",True,1,"Подарунок")
        self.assertEqual([], (await self.commerce.user_commerce(42))["manual_grants"])

    async def test_manual_gifts_show_published_protected_material_for_fresh_users(self):
        for key in ("section:cases","full"):
            with self.subTest(product_key=key):
                await self.commerce.set_manual_grant(42,key,"Подарунок",True,1,"Подарунок")
                items = await build_sections(self.store,await self.store.get_user(42))
                controls = [row for row in items if row["key"] in ("cases","test_questions","question_search")]
                cases = next(row for row in controls if row["key"] == "cases")
                self.assertTrue(cases["has_access"])
                self.assertTrue(cases["visible"])
                if key == "full":
                    self.assertTrue(all(row["visible"] and row["has_access"] for row in controls))

    async def test_manual_full_gift_respects_global_withdrawal_and_new_drafts(self):
        await self.commerce.set_manual_grant(42,"full","Повний доступ",True,1,"Подарунок")
        self.con.db.execute("INSERT INTO attestation_banks(id,title,slug,source_id,status,display_order,questions_count) VALUES(8,'Нова чернетка','draft','draft','draft',1,0)")
        await self.store.set_setting("section_config",json.dumps({"cases":{"visible":False}}))
        items = await build_sections(self.store,await self.store.get_user(42))
        self.assertFalse(next(row for row in items if row["key"] == "cases")["visible"])
        self.assertFalse(next(row for row in items if row["key"] == "attestation:8")["visible"])


class RetainedCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_hidden_bank_is_loaded_only_in_buyers_request(self):
        from questions import QuestionBank
        qb = QuestionBank("unused")
        hidden = {"id":7,"slug":"removed","title":"Куплений","source_id":"removed","questions":[{"id":5,"topic":"Тема","qnum":1,"question":"Запитання?","choices":["A","B"],"correct":[1],"correct_texts":["A"]}]}
        store = SimpleNamespace(list_retained_attestation_banks=AsyncMock(return_value=[hidden]),list_published_attestation_banks=AsyncMock(return_value=[]))
        runtime = SimpleNamespace(store=store,qb=qb)
        buyer = MiniAppService(runtime)
        await buyer.prepare_attestation_catalog(AuthContext({}, {"paid_products":["section:attestation:7"]},42,False))
        self.assertIn("removed",buyer.qb.attestation_banks)
        self.assertNotIn("removed",MiniAppService(runtime).qb.attestation_banks)


class CommerceRouteGuardTests(unittest.TestCase):
    def test_section_withdrawal_returns_explicit_archive_message(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        import app as application
        result = {"id":7,"status":"archived","retained":True,"message":"Матеріали збережено."}
        store = SimpleNamespace(get_setting=AsyncMock(return_value='{}'),set_setting=AsyncMock(),
            list_attestation_banks_for_admin=AsyncMock(return_value=[{"id":7,"slug":"second","title":"Куплений банк","source_id":"second","status":"published","questions_count":5}]),
            delete_attestation_bank=AsyncMock(return_value=result))
        runtime = SimpleNamespace(store=store,qb=SimpleNamespace(load_published_attestation_banks=AsyncMock()))
        with patch.dict(application.app.dependency_overrides,{application.get_runtime:lambda:runtime,application.get_auth_context:lambda:auth()}):
            response = TestClient(application.app).delete("/api/admin/sections/attestation:7")
        self.assertEqual(200,response.status_code)
        self.assertEqual("archived",response.json()["status"])
        self.assertEqual("Матеріали збережено.",response.json()["message"])

    def test_all_commerce_routes_require_admin_and_mutations_verify_missing_user(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        import app as application
        store = SimpleNamespace(get_user=AsyncMock(return_value={}))
        runtime = SimpleNamespace(store=store)
        mutations = [
            ("/api/admin/users/999/grants",{"product_key":"full","enabled":True,"reason":"Подарунок"}),
            ("/api/admin/users/999/restore-purchases",{"reason":"Відновлення"}),
            ("/api/admin/users/999/payments",{"product_key":"full","amount":100,"currency":"XTR","reference":"bank-ref","reason":"Оплата","request_id":"req"}),
            ("/api/admin/users/999/access",{"access":"full"}),
            ("/api/admin/users/999/protected-materials",{"enabled":True}),
            ("/api/admin/users/999/sections/customs",{"enabled":True}),
        ]
        with patch.dict(application.app.dependency_overrides,{application.get_runtime:lambda:runtime,application.get_auth_context:lambda:auth(False)}):
            client = TestClient(application.app)
            for path,payload in mutations+[("/api/admin/payments/sync",{"offset":0,"limit":100})]:
                self.assertEqual(403,client.post(path,json=payload).status_code,path)
            for path in ("/api/admin/users","/api/admin/users/999","/api/admin/payments","/api/admin/activity"):
                self.assertEqual(403,client.get(path).status_code,path)
            self.assertEqual(403,client.delete("/api/admin/users/999").status_code)
        with patch.dict(application.app.dependency_overrides,{application.get_runtime:lambda:runtime,application.get_auth_context:lambda:auth()}):
            client = TestClient(application.app)
            for path,payload in mutations:
                self.assertEqual(404,client.post(path,json=payload).status_code,path)
            self.assertEqual(404,client.delete("/api/admin/users/999").status_code)


class StartupMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_does_not_turn_timed_packages_into_permanent_sections(self):
        from unittest.mock import patch
        connection = SimpleNamespace(execute=AsyncMock(),fetchrow=AsyncMock(return_value=None))
        pool = SimpleNamespace(acquire=lambda:Context(connection))
        store = Storage("postgresql://unused")
        with patch("storage.asyncpg.create_pool",new=AsyncMock(return_value=pool)),patch("storage.CommerceStore.initialize",new=AsyncMock()),patch.object(store,"_migrate_attestation_publication_history",new=AsyncMock()):
            await store.init()
        for call in connection.execute.await_args_list:
            sql = call.args[0]
            self.assertFalse("INSERT INTO user_section_access" in sql and "sub_end >= now()" in sql,
                "Timed subscriptions must stay timed legacy packages, not permanent section grants")
