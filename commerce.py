"""Durable payment provenance, gifts, and safe effective entitlements.

Payments never share storage with administrator gifts. Telegram charge locks also
serialize refund tombstones, so historical imports cannot resurrect a refund.
"""
from __future__ import annotations

import json

from utils import dt_to_iso, now


PACKAGE_TITLES = {"full": "Повний доступ", "cases": "Атестація"}
SYSTEM_TITLES = {"customs": "Митні компетенції", "cases": "Кейси", "test_questions": "Тестові питання", "question_search": "Пошук питань"}
FULL_ENTITLEMENT_SQL = """(EXISTS(SELECT 1 FROM payments p WHERE p.user_id=users.user_id AND p.status='confirmed' AND p.product_key='full')
    OR EXISTS(SELECT 1 FROM manual_access_grants m WHERE m.user_id=users.user_id AND m.product_key='full')
    OR EXISTS(SELECT 1 FROM legacy_access l WHERE l.user_id=users.user_id AND l.product_key='full' AND (l.expires_at IS NULL OR l.expires_at>=now())))"""
MISSING_MATERIAL_SQL = """(p.product_key NOT IN ('full','cases','section:customs','section:cases',
    'section:test_questions','section:question_search','section:customs_code','section:support')
    AND NOT EXISTS(SELECT 1 FROM attestation_banks b WHERE 'section:attestation:' || CAST(b.id AS TEXT)=p.product_key))"""


def product_covers(product: str, section: str, *, manual_grant_only: bool = False) -> bool:
    if product == f"section:{section}":
        return True
    if manual_grant_only:
        return False
    return product == "full" or (product == "cases" and section.startswith("attestation:"))


def entitlement_source(user: dict, section: str, *, manual_grant_only: bool = False) -> str:
    for field, label in (("paid_products", "paid"), ("legacy_products", "legacy"), ("manual_products", "manual")):
        if any(product_covers(key, section, manual_grant_only=manual_grant_only) for key in user.get(field, []) or []):
            return label
    return "none"


def serialize_rows(rows: list[dict]) -> list[dict]:
    return [{key: dt_to_iso(value) if key in {"paid_at", "created_at", "expires_at"} else value
             for key, value in dict(row).items()} for row in rows]


def commerce_summary(detail: dict) -> dict:
    paid = [row for row in detail.get("purchases", []) if row["status"] == "confirmed"]
    totals = {"XTR": 0, "UAH": 0}
    for row in paid:
        if row.get("amount") is not None:
            totals[row["currency"]] = totals.get(row["currency"], 0) + row["amount"]
    legacy = detail.get("legacy_access", [])
    unavailable = {row["id"] for row in detail.get("unavailable_purchases", [])}
    return {"paid_count": len(paid), "payment_totals": totals,
            "purchase_titles": list(dict.fromkeys(row["product_title"] for row in paid)),
            "purchase_sources": list(dict.fromkeys(row.get("source", "") for row in paid)),
            "unknown_amount_count": sum(row.get("amount") is None for row in paid),
            "manual_titles": list(dict.fromkeys(row["product_title"] for row in detail.get("manual_grants", []))),
            "legacy_titles": list(dict.fromkeys(row["product_title"] for row in legacy)),
            "manual_count": len(detail.get("manual_grants", [])), "legacy_count": len(legacy),
            "attention_count": len(legacy) + sum(row.get("amount") is None or row.get("id") in unavailable for row in paid)}


class CommerceStore:
    def __init__(self, store):
        self.store = store

    async def initialize(self, con):
        for statement in (
            """CREATE TABLE IF NOT EXISTS payments (
                id BIGSERIAL PRIMARY KEY, user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
                product_key TEXT NOT NULL, product_title TEXT NOT NULL, amount BIGINT,
                currency TEXT NOT NULL CHECK(currency IN ('XTR','UAH')), paid_at TIMESTAMPTZ NOT NULL,
                source TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('confirmed','refunded')),
                reference TEXT NOT NULL, telegram_charge_id TEXT UNIQUE, request_id TEXT UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(), CHECK(amount IS NULL OR amount > 0))""",
            """CREATE TABLE IF NOT EXISTS payment_refunds (
                telegram_charge_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS manual_access_grants (
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                product_key TEXT NOT NULL, product_title TEXT NOT NULL, actor_id BIGINT,
                reason TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY(user_id,product_key))""",
            """CREATE TABLE IF NOT EXISTS legacy_access (
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
                product_key TEXT NOT NULL, product_title TEXT NOT NULL, expires_at TIMESTAMPTZ,
                PRIMARY KEY(user_id,product_key))""",
            """CREATE TABLE IF NOT EXISTS access_events (
                id BIGSERIAL PRIMARY KEY, actor_id BIGINT, user_id BIGINT,
                action TEXT NOT NULL, product_key TEXT, product_title TEXT, reason TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
            "ALTER TABLE access_events ADD COLUMN IF NOT EXISTS product_title TEXT",
            "CREATE INDEX IF NOT EXISTS payments_user_idx ON payments(user_id,status)",
            "CREATE INDEX IF NOT EXISTS access_events_user_idx ON access_events(user_id,id)",
        ):
            await con.execute(statement)
        async with con.transaction():
            guard = await con.fetchval("""INSERT INTO settings(key,value) VALUES('commerce_provenance_v1','done')
                ON CONFLICT(key) DO NOTHING RETURNING key""")
            if guard:
                config_row = await con.fetchrow("SELECT value FROM settings WHERE key='section_config'")
                try:
                    config = json.loads(config_row["value"]) if config_row else {}
                except (TypeError, ValueError):
                    config = {}
                config = config if isinstance(config, dict) else {}
                banks = {f"attestation:{row['id']}": row["title"] for row in await con.fetch("SELECT id,title FROM attestation_banks")}
                def title(key):
                    saved = config.get(key, {})
                    return str(saved.get("title") or SYSTEM_TITLES.get(key) or banks.get(key) or key) if isinstance(saved,dict) else str(key)
                for user in await con.fetch("SELECT user_id,sub_tier,sub_infinite,sub_end FROM users"):
                    tier = user.get("sub_tier") or ("full" if user.get("sub_infinite") else "none")
                    if tier in PACKAGE_TITLES and (user.get("sub_infinite") or user.get("sub_end")):
                        await con.execute("""INSERT INTO legacy_access(user_id,product_key,product_title,expires_at)
                            VALUES($1,$2,$3,$4) ON CONFLICT(user_id,product_key) DO NOTHING""",
                            user["user_id"], tier, PACKAGE_TITLES[tier], None if user.get("sub_infinite") else user["sub_end"])
                for row in await con.fetch("SELECT user_id,section_key,purchased_at,telegram_charge_id FROM user_section_access"):
                    key = str(row["section_key"])
                    if row.get("telegram_charge_id"):
                        await con.execute("""INSERT INTO payments(user_id,product_key,product_title,amount,currency,paid_at,
                            source,status,reference,telegram_charge_id) VALUES($1,$2,$3,NULL,'XTR',$4,'legacy_telegram','confirmed',$5,$5)
                            ON CONFLICT(telegram_charge_id) DO NOTHING""", row["user_id"], f"section:{key}",title(key),row.get("purchased_at") or now(), row["telegram_charge_id"])
                    else:
                        await con.execute("""INSERT INTO legacy_access(user_id,product_key,product_title)
                            VALUES($1,$2,$3) ON CONFLICT(user_id,product_key) DO NOTHING""",row["user_id"],f"section:{key}",title(key))
                # Old positive administrator overrides are known manual evidence.
                for row in await con.fetch("SELECT user_id,section_key FROM user_section_access_overrides WHERE enabled=TRUE"):
                    await con.execute("""INSERT INTO manual_access_grants(user_id,product_key,product_title,reason)
                        VALUES($1,$2,$3,'Перенесено з рішення адміністратора') ON CONFLICT(user_id,product_key) DO NOTHING""",
                        row["user_id"],f"section:{row['section_key']}",title(row["section_key"]))
                await con.execute("DELETE FROM user_section_access_overrides WHERE enabled=TRUE")
        self.store._commerce_initialized = True

    async def ready(self):
        if not getattr(self.store, "_commerce_initialized", False):
            async with self.store.pool.acquire() as con:
                await self.initialize(con)

    async def user_commerce(self, user_id: int, *, con=None) -> dict:
        if con is None:
            async with self.store.pool.acquire() as connection:
                return await self.user_commerce(user_id, con=connection)
        result = {}
        for label, statement in (
            ("purchases", "SELECT id,product_key,product_title,amount,currency,paid_at,source,status,reference FROM payments WHERE user_id=$1 ORDER BY paid_at DESC,id DESC"),
            ("manual_grants", "SELECT product_key,product_title,created_at,actor_id,reason FROM manual_access_grants WHERE user_id=$1 ORDER BY created_at DESC,product_key"),
            ("legacy_access", "SELECT product_key,product_title,expires_at FROM legacy_access WHERE user_id=$1 ORDER BY product_key"),
            ("access_events", "SELECT id,actor_id,user_id,action,product_key,product_title,reason,created_at FROM access_events WHERE user_id=$1 ORDER BY id DESC LIMIT 100"),
        ):
            result[label] = [dict(row) for row in await con.fetch(statement,user_id)]
        result["unavailable_purchases"] = await self.unavailable_purchases(result["purchases"],con)
        return result

    async def unavailable_purchases(self,purchases,con):
        result = []
        for row in purchases:
            key = row["product_key"]
            if row["status"] != "confirmed" or key in PACKAGE_TITLES:
                continue
            section_key = key.removeprefix("section:")
            if section_key in SYSTEM_TITLES or section_key in {"customs_code","support"}:
                continue
            available = False
            if section_key.startswith("attestation:") and section_key.removeprefix("attestation:").isdigit():
                bank_id = int(section_key.removeprefix("attestation:"))
                available = key == f"section:attestation:{bank_id}" and bool(await con.fetchval("SELECT 1 FROM attestation_banks WHERE id=$1",bank_id))
            if not available:
                result.append({"id":row["id"],"product_key":key,"product_title":row["product_title"],"reason":"material_missing"})
        return result

    async def hydrate(self, user: dict) -> dict:
        detail = await self.user_commerce(user["user_id"])
        user["_commerce_ready"] = True
        user["paid_products"] = [row["product_key"] for row in detail["purchases"] if row["status"] == "confirmed"]
        user["manual_products"] = [row["product_key"] for row in detail["manual_grants"]]
        user["legacy_products"] = [row["product_key"] for row in detail["legacy_access"] if row["expires_at"] is None or row["expires_at"] >= now()]
        user["section_access"] = sorted({key.removeprefix("section:") for field in ("paid_products","manual_products","legacy_products") for key in user[field] if key.startswith("section:")})
        user["commerce_summary"] = commerce_summary(detail)
        all_products = set(user["paid_products"] + user["manual_products"] + user["legacy_products"])
        tier = "full" if "full" in all_products else "cases" if "cases" in all_products else "none"
        legacy_matches = [row for row in detail["legacy_access"] if row["product_key"] == tier and (row["expires_at"] is None or row["expires_at"] >= now())]
        infinite = tier != "none" and (tier in user["paid_products"] or tier in user["manual_products"] or any(row["expires_at"] is None for row in legacy_matches))
        user["_effective_infinite"] = infinite
        user["_effective_sub_end"] = None if infinite else max((row["expires_at"] for row in legacy_matches),default=None)
        return user

    async def audit(self, con, actor_id, user_id, action, product_key, reason, product_title=None):
        if product_key and not product_title:
            product_title = PACKAGE_TITLES.get(product_key) or SYSTEM_TITLES.get(product_key.removeprefix("section:")) or product_key
        await con.execute("""INSERT INTO access_events(actor_id,user_id,action,product_key,reason,product_title)
            VALUES($1,$2,$3,$4,$5,$6)""",actor_id,user_id,action,product_key,reason,product_title)

    async def record_payment(self, *, user_id: int, product_key: str, product_title: str,
                             amount: int | None, currency: str, reference: str, source: str,
                             paid_at=None, request_id=None, actor_id=None, reason="", first_name=None, last_name=None) -> bool:
        if currency not in {"XTR","UAH"} or (amount is not None and (isinstance(amount,bool) or not isinstance(amount,int) or amount <= 0)):
            raise ValueError("Вкажіть додатну цілу суму та валюту XTR або UAH.")
        if not reference or not product_key or not product_title:
            raise ValueError("Вкажіть товар та номер платежу.")
        if source == "admin_external" and (not request_id or not reason.strip() or amount is None):
            raise ValueError("Вкажіть причину та унікальний номер запиту.")
        telegram = source in {"telegram","telegram_sync","legacy_telegram"}
        async with self.store.pool.acquire() as con:
            async with con.transaction():
                lock_key = f"charge:{reference}" if telegram else f"external:{request_id}"
                await con.execute("SELECT pg_advisory_xact_lock(hashtext($1))",lock_key)
                await con.execute("SELECT pg_advisory_xact_lock(hashtext($1))",f"product:{product_key}")
                await con.execute("SELECT pg_advisory_xact_lock(hashtext($1))",f"user:{user_id}")
                previous = await con.fetchrow("SELECT user_id,product_key,amount,currency,reference FROM payments WHERE telegram_charge_id=$1" if telegram else
                    "SELECT user_id,product_key,amount,currency,reference FROM payments WHERE request_id=$1",reference if telegram else request_id)
                if previous:
                    if not telegram and any(previous[key] != value for key,value in {"user_id":user_id,"product_key":product_key,"amount":amount,"currency":currency,"reference":reference}.items()):
                        raise ValueError("Цей номер запиту вже використано для іншого платежу.")
                    return False
                await con.execute("""INSERT INTO users(user_id,first_name,last_name,created_at)
                    VALUES($1,$2,$3,now()) ON CONFLICT(user_id) DO NOTHING""",user_id,first_name,last_name)
                await con.fetchval("SELECT user_id FROM users WHERE user_id=$1 FOR UPDATE",user_id)
                refunded = telegram and bool(await con.fetchval("SELECT 1 FROM payment_refunds WHERE telegram_charge_id=$1",reference))
                await con.execute("""INSERT INTO payments(user_id,product_key,product_title,amount,currency,paid_at,
                    source,status,reference,telegram_charge_id,request_id) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                    user_id,product_key,product_title,amount,currency,paid_at or now(),source,"refunded" if refunded else "confirmed",reference,reference if telegram else None,request_id)
                await self.audit(con,actor_id,user_id,"payment_refunded" if refunded else "payment_confirmed",product_key,reason or f"Telegram: {reference}",product_title)
                return True

    async def record_refund(self, reference: str, *, actor_id=None) -> bool:
        async with self.store.pool.acquire() as con:
            async with con.transaction():
                await con.execute("SELECT pg_advisory_xact_lock(hashtext($1))",f"charge:{reference}")
                inserted = await con.fetchval("""INSERT INTO payment_refunds(telegram_charge_id) VALUES($1)
                    ON CONFLICT(telegram_charge_id) DO NOTHING RETURNING telegram_charge_id""",reference)
                payment = await con.fetchrow("SELECT user_id,product_key,product_title,status FROM payments WHERE telegram_charge_id=$1",reference)
                if payment:
                    await con.fetchval("SELECT user_id FROM users WHERE user_id=$1 FOR UPDATE",payment["user_id"])
                    await con.execute("UPDATE payments SET status='refunded' WHERE telegram_charge_id=$1",reference)
                if inserted:
                    await self.audit(con,actor_id,payment["user_id"] if payment else None,"payment_refunded",payment["product_key"] if payment else None,f"Повернення Telegram: {reference}",payment["product_title"] if payment else None)
                return bool(inserted)

    async def set_manual_grant(self, user_id, product_key, product_title, enabled, actor_id, reason):
        return await self.set_manual_grants(user_id,[(product_key,product_title,enabled)],actor_id,reason)

    async def set_manual_grants(self, user_id, changes, actor_id, reason):
        async with self.store.pool.acquire() as con:
            async with con.transaction():
                for key in sorted({change[0] for change in changes}):
                    await con.execute("SELECT pg_advisory_xact_lock(hashtext($1))",f"product:{key}")
                if not await con.fetchval("SELECT user_id FROM users WHERE user_id=$1 FOR UPDATE",user_id):
                    return False
                for key,title,enabled in changes:
                    if enabled and key.startswith("section:attestation:"):
                        bank_id = key.removeprefix("section:attestation:")
                        if not bank_id.isdigit() or not await con.fetchval("SELECT 1 FROM attestation_banks WHERE id=$1",int(bank_id)):
                            raise ValueError("Розділ вже вилучено. Оновіть список доступів.")
                    if enabled:
                        await con.execute("""INSERT INTO manual_access_grants(user_id,product_key,product_title,actor_id,reason)
                            VALUES($1,$2,$3,$4,$5) ON CONFLICT(user_id,product_key) DO UPDATE SET
                            actor_id=EXCLUDED.actor_id,reason=EXCLUDED.reason,created_at=now()""",user_id,key,title,actor_id,reason)
                    else:
                        await con.execute("DELETE FROM manual_access_grants WHERE user_id=$1 AND product_key=$2",user_id,key)
                    await self.audit(con,actor_id,user_id,"grant_added" if enabled else "grant_removed",key,reason,title)
                return True

    async def restore_purchases(self,user_id,actor_id,reason):
        async with self.store.pool.acquire() as con:
            async with con.transaction():
                if not await con.fetchval("SELECT user_id FROM users WHERE user_id=$1 FOR UPDATE",user_id):
                    return False
                # Reading effective rights rebuilds them from confirmed payments.
                # Unrelated gifts and display choices remain independently controlled.
                await self.audit(con,actor_id,user_id,"purchases_restored",None,reason)
                return True

    async def set_visibility(self,user_id,keys,visible,actor_id,reason):
        async with self.store.pool.acquire() as con:
            async with con.transaction():
                if not await con.fetchval("SELECT user_id FROM users WHERE user_id=$1 FOR UPDATE",user_id):
                    return False
                for key in keys:
                    paid = False if visible else bool(await con.fetchval("""SELECT 1 FROM payments WHERE user_id=$1
                        AND status='confirmed' AND (product_key=$2 OR (product_key='full' AND $3=TRUE)
                            OR (product_key='cases' AND $4=TRUE)) LIMIT 1""",user_id,f"section:{key}",
                            key in SYSTEM_TITLES or key.startswith("attestation:"),key.startswith("attestation:")))
                    effective_visible = bool(visible or paid)
                    await con.execute("""INSERT INTO user_section_visibility_overrides(user_id,section_key,visible,updated_at)
                        VALUES($1,$2,$3,now()) ON CONFLICT(user_id,section_key) DO UPDATE SET visible=EXCLUDED.visible,updated_at=now()""",user_id,key,effective_visible)
                    await self.audit(con,actor_id,user_id,"section_shown" if effective_visible else "section_hidden",f"section:{key}",reason + (" Куплений матеріал залишається доступним." if paid else ""))
                return True

    async def payments_page(self,query,offset,limit):
        rows = await self.store._fetch("""SELECT p.id,p.user_id,p.product_key,p.product_title,p.amount,p.currency,p.paid_at,
            p.source,p.status,p.reference,COALESCE(NULLIF(TRIM(COALESCE(u.first_name,'') || ' ' || COALESCE(u.last_name,'')),''),'—') AS display_name
            FROM payments p JOIN users u ON u.user_id=p.user_id
            WHERE ($1='' OR CAST(p.user_id AS TEXT) ILIKE $2 OR p.product_title ILIKE $2
                OR p.reference ILIKE $2 OR COALESCE(u.first_name,'') ILIKE $2 OR COALESCE(u.last_name,'') ILIKE $2)
            ORDER BY p.paid_at DESC,p.id DESC LIMIT $3 OFFSET $4""",query,f"%{query}%",limit+1,offset)
        return {"items":serialize_rows(rows[:limit]),"offset":offset,"limit":limit,"has_next":len(rows)>limit,"has_prev":offset>0}

    async def activity_page(self,offset,limit):
        rows = await self.store._fetch("SELECT id,actor_id,user_id,action,product_key,product_title,reason,created_at FROM access_events ORDER BY id DESC LIMIT $1 OFFSET $2",limit+1,offset)
        return {"items":serialize_rows(rows[:limit]),"offset":offset,"limit":limit,"has_next":len(rows)>limit,"has_prev":offset>0}


async def sync_star_transactions(bot,store,products: dict[str,str],*,offset: int,limit: int,actor_id: int):
    offset,limit = max(0,offset),max(1,min(100,limit))
    page = await bot.get_star_transactions(offset=offset,limit=limit)
    ledger = CommerceStore(store)
    imported = skipped = 0
    for transaction in page.transactions:
        source = transaction.source
        receiver = transaction.receiver
        # A refund's receiver is the user; its id is the original charge id.
        if receiver and getattr(receiver,"type",None) == "user" and getattr(receiver,"transaction_type",None) == "invoice_payment":
            recognized = getattr(receiver,"invoice_payload",None) in products
            if not recognized:
                recognized = bool(await store._fetchrow("SELECT id FROM payments WHERE telegram_charge_id=$1",transaction.id))
            changed = await ledger.record_refund(transaction.id,actor_id=actor_id) if recognized else False
        elif source and getattr(source,"type",None) == "user" and getattr(source,"transaction_type",None) == "invoice_payment" and getattr(source,"invoice_payload",None) in products and transaction.amount > 0:
            changed = await ledger.record_payment(user_id=source.user.id,product_key=source.invoice_payload,product_title=products[source.invoice_payload],
                amount=transaction.amount,currency="XTR",reference=transaction.id,source="telegram_sync",paid_at=transaction.date,
                actor_id=actor_id,reason="Звірка історії Telegram Stars",first_name=source.user.first_name,last_name=source.user.last_name)
        else:
            changed = False
        imported += int(changed)
        skipped += int(not changed)
    return {"imported":imported,"skipped":skipped,"next_offset":offset+len(page.transactions),
            "has_more":len(page.transactions)==limit,"offset":offset,"limit":limit,"partial":offset>0 or len(page.transactions)==limit,
            "explanation":"Звірено одну сторінку історії. Старі права без підтвердження платежу залишаються з невідомим походженням."}
