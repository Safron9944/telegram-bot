from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg

from utils import normalize_postgres_dsn, now, case_bank_sort_key


class Storage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: asyncpg.Pool | None = None
        self._rev_changed_by_is_int: bool | None = None

    async def init(self):
        dsn, ssl_param = normalize_postgres_dsn(self.dsn)
        self.pool = await asyncpg.create_pool(dsn=dsn, ssl=ssl_param, min_size=1, max_size=10)

        async with self.pool.acquire() as con:
            await con.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    first_name TEXT,
                    last_name TEXT,
                    is_admin INT DEFAULT 0,
                    trial_start TIMESTAMPTZ,
                    trial_end TIMESTAMPTZ,
                    sub_end TIMESTAMPTZ,
                    sub_infinite INT DEFAULT 0,
                    ok_modules_json TEXT DEFAULT '[]',
                    ok_last_levels_json TEXT DEFAULT '{}',
                    created_at TIMESTAMPTZ
                );
            """)
            await con.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name TEXT;")
            await con.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name TEXT;")
            await con.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS sub_tier TEXT;")

            await con.execute("""
                CREATE TABLE IF NOT EXISTS user_section_access (
                    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    section_key TEXT NOT NULL,
                    purchased_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    telegram_charge_id TEXT,
                    PRIMARY KEY (user_id, section_key)
                );
            """)

            await con.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS ui_state (
                    user_id BIGINT PRIMARY KEY,
                    chat_id BIGINT,
                    main_message_id BIGINT,
                    state_json TEXT DEFAULT '{}'
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    user_id BIGINT,
                    qid INT,
                    wrong_count INT DEFAULT 0,
                    in_mistakes INT DEFAULT 0,
                    PRIMARY KEY (user_id, qid)
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS tests (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT,
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    total INT,
                    correct INT,
                    percent DOUBLE PRECISION
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id INT PRIMARY KEY,
                    section TEXT,
                    topic TEXT,
                    ok TEXT,
                    level INT,
                    qnum INT,
                    question TEXT NOT NULL,
                    choices JSONB NOT NULL,
                    correct JSONB NOT NULL,
                    correct_texts JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now()
                );
            """)
            await con.execute("CREATE INDEX IF NOT EXISTS idx_questions_ok ON questions(ok);")
            await con.execute("CREATE INDEX IF NOT EXISTS idx_questions_level ON questions(level);")
            await con.execute("""
                CREATE TABLE IF NOT EXISTS attestation_banks (
                    id BIGSERIAL PRIMARY KEY,
                    slug TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    source_hash TEXT NOT NULL,
                    source_version TEXT,
                    status TEXT NOT NULL DEFAULT 'published',
                    display_order INT NOT NULL DEFAULT 0,
                    questions_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            await con.execute("""
                CREATE TABLE IF NOT EXISTS published_attestation_questions (
                    id BIGSERIAL PRIMARY KEY,
                    bank_id BIGINT NOT NULL REFERENCES attestation_banks(id) ON DELETE CASCADE,
                    source_key TEXT NOT NULL,
                    qnum INT,
                    topic TEXT NOT NULL,
                    question TEXT NOT NULL,
                    choices JSONB NOT NULL,
                    correct JSONB NOT NULL,
                    correct_texts JSONB NOT NULL DEFAULT '[]'::jsonb,
                    shuffle_choices BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE(bank_id, source_key)
                );
            """)
            await con.execute("CREATE INDEX IF NOT EXISTS idx_published_attestation_questions_bank ON published_attestation_questions(bank_id);")
            await con.execute("ALTER TABLE published_attestation_questions ADD COLUMN IF NOT EXISTS managed_manually BOOLEAN NOT NULL DEFAULT FALSE;")
            await con.execute("""
                CREATE TABLE IF NOT EXISTS question_revisions (
                    id BIGSERIAL PRIMARY KEY,
                    qid INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
                    version INT NOT NULL,
                    changed_by TEXT,
                    changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    before JSONB,
                    after JSONB NOT NULL
                );
            """)
            await con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_question_revisions_qid_version ON question_revisions(qid, version);")
            await con.execute("CREATE INDEX IF NOT EXISTS idx_question_revisions_qid ON question_revisions(qid);")
            await con.execute("""
                CREATE TABLE IF NOT EXISTS case_banks (
                    id BIGSERIAL PRIMARY KEY,
                    case_number TEXT NOT NULL,
                    case_title TEXT NOT NULL,
                    source_hash TEXT UNIQUE,
                    questions_count INT NOT NULL DEFAULT 0,
                    answers_count INT NOT NULL DEFAULT 0,
                    correct_count INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            await con.execute("CREATE INDEX IF NOT EXISTS idx_case_banks_number ON case_banks(case_number);")
            await con.execute("""
                CREATE TABLE IF NOT EXISTS case_questions (
                    id BIGSERIAL PRIMARY KEY,
                    case_id BIGINT NOT NULL REFERENCES case_banks(id) ON DELETE CASCADE,
                    position INT NOT NULL,
                    source_question_id INT NOT NULL,
                    question TEXT NOT NULL,
                    description TEXT,
                    question_type INT,
                    correct_answer TEXT NOT NULL DEFAULT '',
                    correct_count INT NOT NULL DEFAULT 0,
                    answers JSONB NOT NULL DEFAULT '[]'::jsonb
                );
            """)
            await con.execute("CREATE INDEX IF NOT EXISTS idx_case_questions_case_id ON case_questions(case_id);")
            await con.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_case_questions_case_source ON case_questions(case_id, source_question_id);")
            await con.execute("""
                CREATE TABLE IF NOT EXISTS test_exam_questions (
                    id BIGSERIAL PRIMARY KEY,
                    num TEXT,
                    module TEXT,
                    question TEXT NOT NULL,
                    correct_answer TEXT,
                    justification TEXT,
                    source TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            await con.execute("CREATE INDEX IF NOT EXISTS idx_test_exam_questions_source ON test_exam_questions(source);")
            await self._maybe_migrate_question_revisions_changed_by(con)

    async def _fetchrow(self, sql: str, *params):
        assert self.pool
        async with self.pool.acquire() as con:
            return await con.fetchrow(sql, *params)

    async def _maybe_migrate_question_revisions_changed_by(self, con) -> None:
        row = await con.fetchrow(
            """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='question_revisions'
              AND column_name='changed_by'
            """
        )
        if not row:
            return
        data_type = (row["data_type"] or "").lower()
        udt_name = (row["udt_name"] or "").lower()
        is_int = data_type in ("bigint", "integer") or udt_name in ("int8", "int4")
        if is_int:
            try:
                await con.execute(
                    "ALTER TABLE question_revisions "
                    "ALTER COLUMN changed_by TYPE TEXT "
                    "USING changed_by::text"
                )
            except Exception:
                pass
        row2 = await con.fetchrow(
            """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema='public'
              AND table_name='question_revisions'
              AND column_name='changed_by'
            """
        )
        if row2:
            dt2 = (row2["data_type"] or "").lower()
            udt2 = (row2["udt_name"] or "").lower()
            self._rev_changed_by_is_int = dt2 in ("bigint", "integer") or udt2 in ("int8", "int4")

    async def _rev_changed_by_param(self, con, changed_by):
        if self._rev_changed_by_is_int is None:
            row = await con.fetchrow(
                """
                SELECT data_type, udt_name
                FROM information_schema.columns
                WHERE table_schema='public'
                  AND table_name='question_revisions'
                  AND column_name='changed_by'
                """
            )
            if row:
                dt = (row["data_type"] or "").lower()
                udt = (row["udt_name"] or "").lower()
                self._rev_changed_by_is_int = dt in ("bigint", "integer") or udt in ("int8", "int4")
            else:
                self._rev_changed_by_is_int = False
        if self._rev_changed_by_is_int:
            if isinstance(changed_by, int):
                return changed_by
            if isinstance(changed_by, str) and changed_by.strip().isdigit():
                return int(changed_by.strip())
            return None
        else:
            if changed_by is None:
                return None
            s = str(changed_by).strip()
            return s or None

    async def _fetch(self, sql: str, *params):
        assert self.pool
        async with self.pool.acquire() as con:
            return await con.fetch(sql, *params)

    async def _exec(self, sql: str, *params):
        assert self.pool
        async with self.pool.acquire() as con:
            return await con.execute(sql, *params)

    async def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = await self._fetchrow("SELECT value FROM settings WHERE key=$1", key)
        return row["value"] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        await self._exec("""
            INSERT INTO settings(key, value) VALUES($1, $2)
            ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
        """, key, value)

    async def ensure_user(self, user_id: int, is_admin: bool = False, first_name: Optional[str] = None, last_name: Optional[str] = None):
        fn = (first_name or "").strip() or None
        ln = (last_name or "").strip() or None
        await self._exec("""
            INSERT INTO users (user_id, is_admin, created_at, first_name, last_name)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id) DO UPDATE SET
                is_admin=EXCLUDED.is_admin,
                first_name=COALESCE(EXCLUDED.first_name, users.first_name),
                last_name=COALESCE(EXCLUDED.last_name, users.last_name)
        """, user_id, 1 if is_admin else 0, now(), fn, ln)

    async def get_user(self, user_id: int) -> dict:
        r = await self._fetchrow("SELECT * FROM users WHERE user_id=$1", user_id)
        if not r:
            return {}
        d = dict(r)
        d["ok_modules"] = json.loads(d.get("ok_modules_json") or "[]")
        d["ok_last_levels"] = json.loads(d.get("ok_last_levels_json") or "{}")
        d["section_access"] = [
            str(row["section_key"])
            for row in await self._fetch(
                "SELECT section_key FROM user_section_access WHERE user_id=$1 ORDER BY section_key",
                user_id,
            )
        ]
        return d

    async def grant_section_access(
        self,
        user_id: int,
        section_key: str,
        *,
        telegram_charge_id: str | None = None,
    ) -> None:
        await self._exec(
            """
            INSERT INTO user_section_access(user_id, section_key, telegram_charge_id)
            VALUES($1, $2, $3)
            ON CONFLICT(user_id, section_key) DO UPDATE SET
                telegram_charge_id=COALESCE(EXCLUDED.telegram_charge_id, user_section_access.telegram_charge_id)
            """,
            user_id,
            section_key,
            telegram_charge_id,
        )

    async def start_trial_if_needed(self, user_id: int, *, days: int = 3, first_name: Optional[str] = None, last_name: Optional[str] = None):
        from datetime import timedelta
        r = await self._fetchrow(
            "SELECT trial_start, trial_end, sub_end, sub_infinite FROM users WHERE user_id=$1", user_id)
        if not r:
            return
        if r.get("trial_start") or r.get("trial_end"):
            return
        if r.get("sub_end") is not None or bool(r.get("sub_infinite")):
            return
        ts = now()
        te = ts + timedelta(days=int(days))
        fn = (first_name or "").strip() or None
        ln = (last_name or "").strip() or None
        await self._exec(
            """
            UPDATE users
            SET trial_start=$1, trial_end=$2,
                first_name=COALESCE($3, first_name),
                last_name=COALESCE($4, last_name)
            WHERE user_id=$5
            """,
            ts, te, fn, ln, user_id
        )

    async def set_ok_modules(self, user_id: int, modules: list[str]):
        await self._exec("""
            UPDATE users SET ok_modules_json=$1 WHERE user_id=$2
        """, json.dumps(modules, ensure_ascii=False), user_id)

    async def set_ok_last_level(self, user_id: int, module: str, level: int):
        user = await self.get_user(user_id)
        mp = user.get("ok_last_levels", {}) or {}
        mp[str(module)] = int(level)
        await self._exec("""
            UPDATE users SET ok_last_levels_json=$1 WHERE user_id=$2
        """, json.dumps(mp, ensure_ascii=False), user_id)

    async def set_subscription(self, user_id: int, sub_end: datetime | None, infinite: bool, tier: str = "full"):
        await self._exec("""
            UPDATE users SET sub_end=$1, sub_infinite=$2, sub_tier=$3 WHERE user_id=$4
        """, sub_end, 1 if infinite else 0, tier, user_id)

    async def set_admin_access(self, user_id: int, access: str, *, trial_days: int = 3):
        from datetime import timedelta

        ts = now()
        if access == "trial":
            await self._exec("""
                UPDATE users
                SET trial_start=$1, trial_end=$2,
                    sub_end=NULL, sub_infinite=0, sub_tier=NULL
                WHERE user_id=$3
            """, ts, ts + timedelta(days=int(trial_days)), user_id)
            return
        if access in ("cases", "full"):
            await self._exec("""
                UPDATE users
                SET sub_end=NULL, sub_infinite=1, sub_tier=$1
                WHERE user_id=$2
            """, access, user_id)
            return
        if access == "none":
            await self._exec("""
                UPDATE users
                SET trial_start=COALESCE(trial_start, $1), trial_end=$1,
                    sub_end=NULL, sub_infinite=0, sub_tier=NULL
                WHERE user_id=$2
            """, ts, user_id)
            return
        raise ValueError(f"Unknown access type: {access}")

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user and all user-owned progress in one transaction."""
        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                exists = await con.fetchval("SELECT 1 FROM users WHERE user_id=$1", user_id)
                if not exists:
                    return False
                await con.execute("DELETE FROM ui_state WHERE user_id=$1", user_id)
                await con.execute("DELETE FROM errors WHERE user_id=$1", user_id)
                await con.execute("DELETE FROM tests WHERE user_id=$1", user_id)
                await con.execute("DELETE FROM users WHERE user_id=$1", user_id)
        return True

    async def get_ui(self, user_id: int) -> dict:
        r = await self._fetchrow("SELECT * FROM ui_state WHERE user_id=$1", user_id)
        if not r:
            return {}
        d = dict(r)
        d["state"] = json.loads(d.get("state_json") or "{}")
        return d

    async def set_ui(self, user_id: int, chat_id: int, main_message_id: int):
        await self._exec("""
            INSERT INTO ui_state (user_id, chat_id, main_message_id, state_json)
            VALUES ($1, $2, $3, '{}')
            ON CONFLICT (user_id) DO UPDATE SET chat_id=EXCLUDED.chat_id, main_message_id=EXCLUDED.main_message_id
        """, user_id, chat_id, main_message_id)

    async def set_state(self, user_id: int, state: dict):
        await self._exec("""
            INSERT INTO ui_state (user_id, chat_id, main_message_id, state_json)
            VALUES ($1, NULL, NULL, $2)
            ON CONFLICT (user_id) DO UPDATE SET state_json=EXCLUDED.state_json
        """, user_id, json.dumps(state, ensure_ascii=False))

    async def bump_wrong(self, user_id: int, qid: int) -> tuple[int, int]:
        r = await self._fetchrow("SELECT wrong_count, in_mistakes FROM errors WHERE user_id=$1 AND qid=$2", user_id, qid)
        if not r:
            await self._exec("INSERT INTO errors (user_id, qid, wrong_count, in_mistakes) VALUES ($1,$2,1,0)", user_id, qid)
            return 1, 0
        wc = int(r["wrong_count"]) + 1
        im = int(r["in_mistakes"])
        if wc >= 5:
            im = 1
        await self._exec("UPDATE errors SET wrong_count=$1, in_mistakes=$2 WHERE user_id=$3 AND qid=$4", wc, im, user_id, qid)
        return wc, im

    async def list_mistakes(self, user_id: int) -> list[int]:
        rows = await self._fetch("SELECT qid FROM errors WHERE user_id=$1 AND in_mistakes=1", user_id)
        return [int(r["qid"]) for r in rows]

    async def remove_mistake(self, user_id: int, qid: int):
        await self._exec("DELETE FROM errors WHERE user_id=$1 AND qid=$2", user_id, qid)

    async def save_test(self, user_id: int, started_at: datetime, finished_at: datetime, total: int, correct: int):
        percent = (correct / total * 100.0) if total else 0.0
        await self._exec("""
            INSERT INTO tests (user_id, started_at, finished_at, total, correct, percent)
            VALUES ($1,$2,$3,$4,$5,$6)
        """, user_id, started_at, finished_at, total, correct, percent)

    async def stats(self, user_id: int) -> dict:
        rows = await self._fetch("SELECT * FROM tests WHERE user_id=$1 ORDER BY id DESC LIMIT 50", user_id)
        if not rows:
            return {"count": 0, "avg": 0.0, "last": None}
        perc = [float(r["percent"]) for r in rows]
        last = dict(rows[0])
        return {"count": len(rows), "avg": sum(perc) / len(perc), "last": last}

    async def list_users(self, offset: int, limit: int) -> list[dict]:
        rows = await self._fetch("""
            SELECT user_id, first_name, last_name, is_admin, trial_start, trial_end,
                   sub_end, sub_infinite, sub_tier, created_at
            FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2
        """, limit, offset)
        return [dict(r) for r in rows]

    async def users_access_counts(self) -> dict[str, int]:
        row = await self._fetchrow("""
            SELECT
                COUNT(*) FILTER (
                    WHERE COALESCE(sub_infinite, 0)=1 OR (sub_end IS NOT NULL AND sub_end >= NOW())
                ) AS active,
                COUNT(*) FILTER (
                    WHERE NOT (COALESCE(sub_infinite, 0)=1 OR (sub_end IS NOT NULL AND sub_end >= NOW()))
                      AND trial_end IS NOT NULL AND trial_end >= NOW()
                ) AS trial,
                COUNT(*) FILTER (
                    WHERE NOT (COALESCE(sub_infinite, 0)=1 OR (sub_end IS NOT NULL AND sub_end >= NOW()))
                      AND (trial_end IS NULL OR trial_end < NOW())
                ) AS expired
            FROM users
        """)
        return {
            "active": int(row["active"] or 0) if row else 0,
            "trial": int(row["trial"] or 0) if row else 0,
            "expired": int(row["expired"] or 0) if row else 0,
        }

    async def questions_count(self) -> int:
        r = await self._fetchrow("SELECT COUNT(*) AS c FROM questions")
        return int(r["c"]) if r else 0

    async def fetch_questions(self) -> list[dict]:
        rows = await self._fetch("""
            SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts
            FROM questions ORDER BY id
        """)
        return [dict(r) for r in rows]

    async def fetch_question(self, qid: int) -> Optional[dict]:
        row = await self._fetchrow(
            "SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts FROM questions WHERE id=$1",
            int(qid),
        )
        return dict(row) if row else None

    async def upsert_case_bank(self, payload: dict, *, changed_by: str = "admin") -> dict:
        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                row = await con.fetchrow(
                    """
                    INSERT INTO case_banks (case_number, case_title, source_hash, questions_count, answers_count, correct_count, updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6,now())
                    ON CONFLICT (source_hash) DO UPDATE SET
                        case_number=EXCLUDED.case_number,
                        case_title=EXCLUDED.case_title,
                        questions_count=EXCLUDED.questions_count,
                        answers_count=EXCLUDED.answers_count,
                        correct_count=EXCLUDED.correct_count,
                        updated_at=now()
                    RETURNING id, case_number, case_title, source_hash, questions_count, answers_count, correct_count, created_at, updated_at
                    """,
                    payload.get("case_number") or "Без номера",
                    payload.get("case_title") or "Кейс без назви",
                    payload.get("source_hash") or None,
                    int(payload.get("questions_count") or 0),
                    int(payload.get("answers_count") or 0),
                    int(payload.get("correct_count") or 0),
                )
                case_id = int(row["id"])
                await con.execute("DELETE FROM case_questions WHERE case_id=$1", case_id)
                for item in payload.get("questions") or []:
                    await con.execute(
                        """
                        INSERT INTO case_questions (case_id, position, source_question_id, question, description, question_type, correct_answer, correct_count, answers)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb)
                        """,
                        case_id, int(item.get("position") or 0), int(item.get("source_question_id") or 0),
                        item.get("question") or "", item.get("description") or "", item.get("question_type"),
                        item.get("correct_answer") or "", int(item.get("correct_count") or 0),
                        json.dumps(item.get("answers") or [], ensure_ascii=False),
                    )
                return dict(row)

    async def list_case_banks(self) -> list[dict]:
        rows = await self._fetch(
            "SELECT id, case_number, case_title, questions_count, answers_count, correct_count, created_at, updated_at FROM case_banks"
        )
        items = [dict(r) for r in rows]
        return sorted(items, key=case_bank_sort_key)

    async def get_case_bank(self, case_id: int) -> Optional[dict]:
        row = await self._fetchrow(
            "SELECT id, case_number, case_title, questions_count, answers_count, correct_count, created_at, updated_at FROM case_banks WHERE id=$1",
            int(case_id),
        )
        return dict(row) if row else None

    async def list_case_questions(self, case_id: int, offset: int = 0, limit: int = 50, query: str = "") -> list[dict]:
        if (query or "").strip():
            like = f"%{query.strip()}%"
            rows = await self._fetch(
                """
                SELECT id, case_id, position, source_question_id, question, description, question_type, correct_answer, correct_count, answers
                FROM case_questions
                WHERE case_id=$1 AND question ILIKE $2
                ORDER BY position ASC, id ASC LIMIT $3 OFFSET $4
                """,
                int(case_id), like, int(limit), int(offset),
            )
        else:
            rows = await self._fetch(
                """
                SELECT id, case_id, position, source_question_id, question, description, question_type, correct_answer, correct_count, answers
                FROM case_questions WHERE case_id=$1 ORDER BY position ASC, id ASC LIMIT $2 OFFSET $3
                """,
                int(case_id), int(limit), int(offset),
            )
        return [dict(r) for r in rows]

    async def search_case_questions_all(self, query: str, limit: int = 50, offset: int = 0) -> list[dict]:
        if not (query or "").strip():
            return []
        like = f"%{query.strip()}%"
        rows = await self._fetch(
            """
            SELECT cq.id, cq.case_id, cq.position, cq.source_question_id, cq.question, cq.description,
                   cq.question_type, cq.correct_answer, cq.correct_count, cq.answers, cb.case_number
            FROM case_questions cq
            JOIN case_banks cb ON cq.case_id = cb.id
            WHERE cq.question ILIKE $1
            ORDER BY cb.case_number, cq.position ASC
            LIMIT $2 OFFSET $3
            """,
            like, int(limit), int(offset),
        )
        return [dict(r) for r in rows]

    async def search_ok_questions(self, query: str, limit: int = 10, offset: int = 0) -> list[dict]:
        if (query or "").strip():
            like = f"%{query.strip()}%"
            rows = await self._fetch(
                """
                SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts
                FROM questions
                WHERE ok IS NOT NULL AND question ILIKE $1
                ORDER BY ok, COALESCE(qnum, 99999), id
                LIMIT $2 OFFSET $3
                """,
                like, int(limit), int(offset),
            )
        else:
            rows = await self._fetch(
                """
                SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts
                FROM questions
                WHERE ok IS NOT NULL
                ORDER BY ok, COALESCE(qnum, 99999), id
                LIMIT $1 OFFSET $2
                """,
                int(limit), int(offset),
            )
        return [dict(r) for r in rows]

    async def search_all_questions(self, query: str, limit: int = 15) -> list[dict]:
        if not (query or "").strip():
            return []
        like = f"%{query.strip()}%"
        rows = await self._fetch(
            """
            SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts
            FROM questions
            WHERE question ILIKE $1
            ORDER BY ok NULLS LAST, COALESCE(qnum, 99999), id
            LIMIT $2
            """,
            like, int(limit),
        )
        return [dict(r) for r in rows]

    async def delete_case_bank(self, case_id: int) -> bool:
        assert self.pool
        async with self.pool.acquire() as con:
            res = await con.execute("DELETE FROM case_banks WHERE id=$1", int(case_id))
        return not res.endswith(" 0")

    async def update_question_content(self, qid: int, *, question: Optional[str] = None, choices: Optional[List[str]] = None, correct: Optional[List[int]] = None, changed_by: Optional[str] = None) -> Optional[dict]:
        def _norm_json(v: Any):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return v
            return v

        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                existing = await con.fetchrow(
                    "SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts FROM questions WHERE id=$1 FOR UPDATE",
                    int(qid),
                )
                if not existing:
                    return None
                before = dict(existing)
                before["choices"] = _norm_json(before.get("choices"))
                before["correct"] = _norm_json(before.get("correct"))
                before["correct_texts"] = _norm_json(before.get("correct_texts"))
                after = {
                    "id": int(before.get("id")),
                    "section": before.get("section") or "",
                    "topic": before.get("topic") or "",
                    "ok": before.get("ok"),
                    "level": int(before.get("level")) if before.get("level") is not None else None,
                    "qnum": int(before.get("qnum")) if before.get("qnum") is not None else None,
                    "question": before.get("question") or "",
                    "choices": list(before.get("choices") or []),
                    "correct": [int(x) for x in (before.get("correct") or [])],
                    "correct_texts": list(before.get("correct_texts") or []),
                }
                if question is not None:
                    after["question"] = (question or "").strip()
                if choices is not None:
                    after["choices"] = [str(x).strip() for x in choices if str(x).strip()]
                if correct is not None:
                    after["correct"] = [int(x) for x in correct]
                ch = list(after.get("choices") or [])
                corr = [int(x) for x in (after.get("correct") or [])]
                after["correct_texts"] = [ch[i - 1] for i in corr if 1 <= i <= len(ch)]
                cmp_before = {k: before.get(k) for k in after.keys()}
                if cmp_before == after:
                    return after
                await con.execute(
                    """
                    UPDATE questions SET question=$2, choices=$3::jsonb, correct=$4::jsonb, correct_texts=$5::jsonb, updated_at=now()
                    WHERE id=$1
                    """,
                    after["id"], after["question"],
                    json.dumps(after["choices"], ensure_ascii=False),
                    json.dumps(after["correct"], ensure_ascii=False),
                    json.dumps(after["correct_texts"], ensure_ascii=False),
                )
                prev_ver = await con.fetchval("SELECT COALESCE(MAX(version), 0) FROM question_revisions WHERE qid=$1", after["id"])
                ver = int(prev_ver or 0) + 1
                changed_by_db = await self._rev_changed_by_param(con, changed_by)
                await con.execute(
                    "INSERT INTO question_revisions (qid, version, changed_by, before, after) VALUES ($1,$2,$3,$4::jsonb,$5::jsonb)",
                    after["id"], ver, changed_by_db,
                    json.dumps(before, ensure_ascii=False),
                    json.dumps(after, ensure_ascii=False),
                )
                return after

    async def import_questions_from_json(self, path: str, *, changed_by: str = "import", force: bool = False) -> int:
        if not path or not os.path.exists(path):
            raise RuntimeError(f"Questions file not found: {path}")

        from questions import QuestionBank
        qb = QuestionBank(path)
        qb.load()
        items = [q for q in qb.by_id.values() if q.is_valid_mcq]
        if not items:
            return 0

        def _norm_json(v: Any):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return v
            return v

        assert self.pool
        imported = 0
        async with self.pool.acquire() as con:
            async with con.transaction():
                for q in items:
                    existing = await con.fetchrow(
                        "SELECT id, section, topic, ok, level, qnum, question, choices, correct, correct_texts FROM questions WHERE id=$1",
                        int(q.id),
                    )
                    before = dict(existing) if existing else None
                    if before:
                        before["choices"] = _norm_json(before.get("choices"))
                        before["correct"] = _norm_json(before.get("correct"))
                        before["correct_texts"] = _norm_json(before.get("correct_texts"))
                    after = {
                        "id": int(q.id), "section": q.section or "", "topic": q.topic or "",
                        "ok": q.ok, "level": int(q.level) if q.level is not None else None,
                        "qnum": int(q.qnum) if q.qnum is not None else None,
                        "question": q.question or "", "choices": list(q.choices or []),
                        "correct": [int(x) for x in (q.correct or [])],
                        "correct_texts": list(q.correct_texts or []),
                    }
                    changed = True
                    if before:
                        cmp_before = {k: before.get(k) for k in after.keys()}
                        changed = (cmp_before != after)
                    if (not existing) or changed or force:
                        await con.execute(
                            """
                            INSERT INTO questions (id, section, topic, ok, level, qnum, question, choices, correct, correct_texts, updated_at)
                            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10::jsonb, now())
                            ON CONFLICT (id) DO UPDATE SET
                                section=EXCLUDED.section, topic=EXCLUDED.topic, ok=EXCLUDED.ok,
                                level=EXCLUDED.level, qnum=EXCLUDED.qnum, question=EXCLUDED.question,
                                choices=EXCLUDED.choices, correct=EXCLUDED.correct,
                                correct_texts=EXCLUDED.correct_texts, updated_at=now()
                            """,
                            after["id"], after["section"], after["topic"], after["ok"],
                            after["level"], after["qnum"], after["question"],
                            json.dumps(after["choices"], ensure_ascii=False),
                            json.dumps(after["correct"], ensure_ascii=False),
                            json.dumps(after["correct_texts"], ensure_ascii=False),
                        )
                        prev_ver = await con.fetchval("SELECT COALESCE(MAX(version), 0) FROM question_revisions WHERE qid=$1", after["id"])
                        ver = int(prev_ver or 0) + 1
                        changed_by_db = await self._rev_changed_by_param(con, changed_by)
                        await con.execute(
                            "INSERT INTO question_revisions (qid, version, changed_by, before, after) VALUES ($1,$2,$3,$4::jsonb,$5::jsonb)",
                            after["id"], ver, changed_by_db,
                            (json.dumps(before, ensure_ascii=False) if before else None),
                            json.dumps(after, ensure_ascii=False),
                        )
                        imported += 1
        return imported

    async def import_test_exam_questions(self, items: list[dict]) -> int:
        assert self.pool
        inserted = 0
        async with self.pool.acquire() as con:
            for item in items:
                row = await con.fetchrow(
                    "SELECT id FROM test_exam_questions WHERE num=$1 AND source=$2",
                    item.get("num"), item.get("source"),
                )
                if not row:
                    await con.execute(
                        """
                        INSERT INTO test_exam_questions (num, module, question, correct_answer, justification, source)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        item.get("num"), item.get("module"), item.get("question") or "",
                        item.get("correct_answer") or "", item.get("justification") or "",
                        item.get("source") or "",
                    )
                    inserted += 1
        return inserted

    async def list_published_attestation_banks(self) -> list[dict]:
        assert self.pool
        async with self.pool.acquire() as con:
            banks = await con.fetch(
                """
                SELECT id, slug, title, source_id, source_hash, source_version,
                       status, display_order, questions_count
                FROM attestation_banks
                WHERE status='published'
                ORDER BY display_order, id
                """
            )
            result = []
            for row in banks:
                item = dict(row)
                questions = await con.fetch(
                    """
                    SELECT id, source_key, qnum, topic, question, choices, correct,
                           correct_texts, shuffle_choices
                    FROM published_attestation_questions
                    WHERE bank_id=$1
                    ORDER BY topic, qnum NULLS LAST, id
                    """,
                    item["id"],
                )
                item["questions"] = [dict(question) for question in questions]
                result.append(item)
            return result

    async def list_attestation_banks_for_admin(self) -> list[dict]:
        assert self.pool
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                """
                SELECT id, slug, title, source_id, status, display_order, questions_count,
                       created_at, updated_at
                FROM attestation_banks
                ORDER BY display_order, id
                """
            )
            return [dict(row) for row in rows]

    async def set_attestation_bank_visibility(self, bank_id: int, *, visible: bool) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                UPDATE attestation_banks
                SET status=$2, updated_at=now()
                WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1'
                RETURNING id, slug, title, source_id, status, display_order, questions_count
                """,
                int(bank_id), "published" if visible else "hidden",
            )
            return dict(row) if row else None

    async def move_attestation_bank(self, bank_id: int, *, direction: str) -> bool:
        if direction not in {"up", "down"}:
            return False
        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                rows = [dict(row) for row in await con.fetch(
                    """
                    SELECT id, slug, source_id, display_order
                    FROM attestation_banks
                    WHERE slug <> 'stage-1' AND source_id <> 'bundled-stage-1'
                    ORDER BY display_order, id
                    FOR UPDATE
                    """
                )]
                index = next((i for i, row in enumerate(rows) if int(row["id"]) == int(bank_id)), None)
                if index is None:
                    return False
                target = index - 1 if direction == "up" else index + 1
                if target < 0 or target >= len(rows):
                    return True
                rows[index], rows[target] = rows[target], rows[index]
                await con.executemany(
                    "UPDATE attestation_banks SET display_order=$2, updated_at=now() WHERE id=$1",
                    [(int(row["id"]), order) for order, row in enumerate(rows)],
                )
                return True

    async def delete_attestation_bank(self, bank_id: int) -> bool:
        assert self.pool
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                DELETE FROM attestation_banks
                WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1'
                RETURNING id
                """,
                int(bank_id),
            )
            return row is not None

    async def get_attestation_bank_for_admin(self, bank_id: int) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                SELECT id, slug, title, source_id, status, display_order, questions_count,
                       created_at, updated_at
                FROM attestation_banks
                WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1'
                """,
                int(bank_id),
            )
            return dict(row) if row else None

    async def update_attestation_bank_title(self, bank_id: int, title: str) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                UPDATE attestation_banks
                SET title=$2, updated_at=now()
                WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1'
                RETURNING id, slug, title, source_id, status, display_order, questions_count,
                          created_at, updated_at
                """,
                int(bank_id), title,
            )
            return dict(row) if row else None

    async def rename_attestation_topic(self, bank_id: int, old_topic: str, new_topic: str) -> int | None:
        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                exists = await con.fetchval(
                    "SELECT 1 FROM attestation_banks WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1'",
                    int(bank_id),
                )
                if not exists:
                    return None
                result = await con.execute(
                    """
                    UPDATE published_attestation_questions
                    SET topic=$3, managed_manually=TRUE, updated_at=now()
                    WHERE bank_id=$1 AND topic=$2
                    """,
                    int(bank_id), old_topic, new_topic,
                )
                return int(result.rsplit(" ", 1)[-1])

    async def list_attestation_questions_for_admin(
        self,
        bank_id: int,
        *,
        topic: str = "",
        query: str = "",
        offset: int = 0,
        limit: int = 25,
    ) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            bank = await con.fetchrow(
                """
                SELECT id, slug, title, source_id, status, display_order, questions_count,
                       created_at, updated_at
                FROM attestation_banks
                WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1'
                """,
                int(bank_id),
            )
            if not bank:
                return None

            topics = await con.fetch(
                """
                SELECT topic, COUNT(*)::INT AS questions_count
                FROM published_attestation_questions
                WHERE bank_id=$1
                GROUP BY topic
                ORDER BY topic
                """,
                int(bank_id),
            )
            clean_topic = (topic or "").strip()
            clean_query = (query or "").strip()
            pattern = f"%{clean_query}%"
            total = int(await con.fetchval(
                """
                SELECT COUNT(*)
                FROM published_attestation_questions
                WHERE bank_id=$1
                  AND ($2='' OR topic=$2)
                  AND ($3='' OR question ILIKE $4 OR topic ILIKE $4 OR COALESCE(qnum::TEXT, '') ILIKE $4)
                """,
                int(bank_id), clean_topic, clean_query, pattern,
            ) or 0)
            rows = await con.fetch(
                """
                SELECT id, source_key, qnum, topic, question, choices, correct,
                       correct_texts, shuffle_choices, managed_manually,
                       created_at, updated_at
                FROM published_attestation_questions
                WHERE bank_id=$1
                  AND ($2='' OR topic=$2)
                  AND ($3='' OR question ILIKE $4 OR topic ILIKE $4 OR COALESCE(qnum::TEXT, '') ILIKE $4)
                ORDER BY topic, qnum NULLS LAST, id
                OFFSET $5 LIMIT $6
                """,
                int(bank_id), clean_topic, clean_query, pattern, int(offset), int(limit),
            )
            return {
                "bank": dict(bank),
                "topics": [dict(row) for row in topics],
                "items": [dict(row) for row in rows],
                "total": total,
                "offset": int(offset),
                "limit": int(limit),
            }

    async def get_attestation_question_for_admin(self, bank_id: int, question_id: int) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                SELECT id, bank_id, source_key, qnum, topic, question, choices, correct,
                       correct_texts, shuffle_choices, managed_manually,
                       created_at, updated_at
                FROM published_attestation_questions
                WHERE bank_id=$1 AND id=$2
                """,
                int(bank_id), int(question_id),
            )
            return dict(row) if row else None

    async def create_attestation_question(
        self,
        bank_id: int,
        *,
        topic: str,
        qnum: int | None,
        question: str,
        choices: list[str],
        correct: list[int],
        correct_texts: list[str],
        shuffle_choices: bool,
    ) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                bank = await con.fetchrow(
                    "SELECT id FROM attestation_banks WHERE id=$1 AND slug <> 'stage-1' AND source_id <> 'bundled-stage-1' FOR UPDATE",
                    int(bank_id),
                )
                if not bank:
                    return None
                final_qnum = qnum
                if final_qnum is None:
                    final_qnum = int(await con.fetchval(
                        "SELECT COALESCE(MAX(qnum), 0) + 1 FROM published_attestation_questions WHERE bank_id=$1 AND topic=$2",
                        int(bank_id), topic,
                    ) or 1)
                row = await con.fetchrow(
                    """
                    INSERT INTO published_attestation_questions(
                        bank_id, source_key, qnum, topic, question, choices, correct,
                        correct_texts, shuffle_choices, managed_manually
                    )
                    VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb,$9,TRUE)
                    RETURNING id, bank_id, source_key, qnum, topic, question, choices, correct,
                              correct_texts, shuffle_choices, managed_manually,
                              created_at, updated_at
                    """,
                    int(bank_id), f"manual:{uuid.uuid4().hex}", int(final_qnum), topic, question,
                    json.dumps(choices, ensure_ascii=False), json.dumps(correct),
                    json.dumps(correct_texts, ensure_ascii=False), bool(shuffle_choices),
                )
                await con.execute(
                    """
                    UPDATE attestation_banks
                    SET questions_count=(SELECT COUNT(*) FROM published_attestation_questions WHERE bank_id=$1),
                        updated_at=now()
                    WHERE id=$1
                    """,
                    int(bank_id),
                )
                return dict(row) if row else None

    async def update_attestation_question(
        self,
        bank_id: int,
        question_id: int,
        *,
        topic: str,
        qnum: int | None,
        question: str,
        choices: list[str],
        correct: list[int],
        correct_texts: list[str],
        shuffle_choices: bool,
    ) -> dict | None:
        assert self.pool
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                """
                UPDATE published_attestation_questions
                SET topic=$3, qnum=$4, question=$5, choices=$6::jsonb, correct=$7::jsonb,
                    correct_texts=$8::jsonb, shuffle_choices=$9,
                    managed_manually=TRUE, updated_at=now()
                WHERE bank_id=$1 AND id=$2
                RETURNING id, bank_id, source_key, qnum, topic, question, choices, correct,
                          correct_texts, shuffle_choices, managed_manually,
                          created_at, updated_at
                """,
                int(bank_id), int(question_id), topic, qnum, question,
                json.dumps(choices, ensure_ascii=False), json.dumps(correct),
                json.dumps(correct_texts, ensure_ascii=False), bool(shuffle_choices),
            )
            return dict(row) if row else None

    async def delete_attestation_question(self, bank_id: int, question_id: int) -> bool:
        assert self.pool
        async with self.pool.acquire() as con:
            async with con.transaction():
                row = await con.fetchrow(
                    "DELETE FROM published_attestation_questions WHERE bank_id=$1 AND id=$2 RETURNING id",
                    int(bank_id), int(question_id),
                )
                if not row:
                    return False
                await con.execute(
                    """
                    UPDATE attestation_banks
                    SET questions_count=(SELECT COUNT(*) FROM published_attestation_questions WHERE bank_id=$1),
                        updated_at=now()
                    WHERE id=$1
                    """,
                    int(bank_id),
                )
                return True

    async def publish_attestation_bank(self, bank, *, title: str, slug: str, changed_by: str) -> dict:
        assert self.pool
        source_id = (bank.source or slug).casefold()
        async with self.pool.acquire() as con:
            async with con.transaction():
                existing = await con.fetchrow(
                    "SELECT * FROM attestation_banks WHERE source_id=$1 FOR UPDATE",
                    source_id,
                )
                updated = existing is not None
                if existing:
                    bank_id = int(existing["id"])
                    stable_slug = existing["slug"]
                    await con.execute(
                        """
                        UPDATE attestation_banks
                        SET title=$2, source_hash=$3, source_version=$4,
                            status='published', questions_count=$5, updated_at=now()
                        WHERE id=$1
                        """,
                        bank_id, title, bank.source_hash, bank.source_version, len(bank.questions),
                    )
                else:
                    row = await con.fetchrow(
                        """
                        INSERT INTO attestation_banks(
                            slug, title, source_id, source_hash, source_version,
                            status, display_order, questions_count
                        )
                        VALUES($1,$2,$3,$4,$5,'published',
                               COALESCE((SELECT MAX(display_order) + 1 FROM attestation_banks), 0),$6)
                        RETURNING id, slug
                        """,
                        slug, title, source_id, bank.source_hash, bank.source_version, len(bank.questions),
                    )
                    bank_id = int(row["id"])
                    stable_slug = row["slug"]

                source_keys = [question.source_key for question in bank.questions]
                rows = [
                    (
                        bank_id, item.source_key, item.qnum, item.topic, item.question,
                        json.dumps(list(item.choices), ensure_ascii=False),
                        json.dumps(list(item.correct), ensure_ascii=False),
                        json.dumps(list(item.correct_texts), ensure_ascii=False),
                        bool(item.shuffle_choices),
                    )
                    for item in bank.questions
                ]
                await con.executemany(
                    """
                    INSERT INTO published_attestation_questions(
                        bank_id, source_key, qnum, topic, question, choices,
                        correct, correct_texts, shuffle_choices, managed_manually
                    )
                    VALUES($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8::jsonb,$9,FALSE)
                    ON CONFLICT(bank_id, source_key) DO UPDATE SET
                        qnum=EXCLUDED.qnum, topic=EXCLUDED.topic,
                        question=EXCLUDED.question, choices=EXCLUDED.choices,
                        correct=EXCLUDED.correct, correct_texts=EXCLUDED.correct_texts,
                        shuffle_choices=EXCLUDED.shuffle_choices, updated_at=now()
                    WHERE NOT published_attestation_questions.managed_manually
                    """,
                    rows,
                )
                await con.execute(
                    "DELETE FROM published_attestation_questions WHERE bank_id=$1 AND managed_manually=FALSE AND NOT(source_key = ANY($2::text[]))",
                    bank_id, source_keys,
                )
                await con.execute(
                    """
                    UPDATE attestation_banks
                    SET questions_count=(SELECT COUNT(*) FROM published_attestation_questions WHERE bank_id=$1),
                        updated_at=now()
                    WHERE id=$1
                    """,
                    bank_id,
                )
                return {
                    "id": bank_id,
                    "slug": stable_slug,
                    "title": title,
                    "count": len(bank.questions),
                    "updated": updated,
                }

    async def search_test_exam_questions(self, query: str, limit: int = 20, offset: int = 0) -> dict:
        assert self.pool
        async with self.pool.acquire() as con:
            if query:
                pattern = f"%{query}%"
                total = int(await con.fetchval(
                    "SELECT COUNT(*) FROM test_exam_questions WHERE question ILIKE $1 OR correct_answer ILIKE $1 OR num ILIKE $1",
                    pattern,
                ) or 0)
                rows = await con.fetch(
                    """
                    SELECT id, num, module, question, correct_answer, justification, source
                    FROM test_exam_questions
                    WHERE question ILIKE $1 OR correct_answer ILIKE $1 OR num ILIKE $1
                    ORDER BY id
                    LIMIT $2 OFFSET $3
                    """,
                    pattern, limit, offset,
                )
            else:
                total = int(await con.fetchval("SELECT COUNT(*) FROM test_exam_questions") or 0)
                rows = await con.fetch(
                    "SELECT id, num, module, question, correct_answer, justification, source FROM test_exam_questions ORDER BY id LIMIT $1 OFFSET $2",
                    limit, offset,
                )
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
        }
