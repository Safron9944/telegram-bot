from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import asyncpg

from .utils import utcnow
from .config import KYIV_TZ

DDL_CREATE = """
CREATE TABLE IF NOT EXISTS users (
  tg_id BIGINT PRIMARY KEY,
  phone TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  trial_until TIMESTAMPTZ,
  sub_until TIMESTAMPTZ,
  is_admin BOOLEAN NOT NULL DEFAULT FALSE,
  last_seen TIMESTAMPTZ NOT NULL,
  ok_code TEXT,
  ok_level INT,
  train_mode TEXT,
  position TEXT
);

CREATE TABLE IF NOT EXISTS stats (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  answered INT NOT NULL DEFAULT 0,
  correct INT NOT NULL DEFAULT 0,
  wrong INT NOT NULL DEFAULT 0,
  skipped INT NOT NULL DEFAULT 0,
  PRIMARY KEY (tg_id, mode)
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id UUID PRIMARY KEY,
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  question_ids JSONB NOT NULL,
  current_index INT NOT NULL DEFAULT 0,
  correct_count INT NOT NULL DEFAULT 0,
  wrong_count INT NOT NULL DEFAULT 0,
  skipped_count INT NOT NULL DEFAULT 0,
  started_at TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ,
  completed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS question_flags (
  question_id INT PRIMARY KEY,
  is_disabled BOOLEAN NOT NULL DEFAULT FALSE,
  note TEXT,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_prefs (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  ok_code TEXT NOT NULL,
  ok_level INT NOT NULL,
  topics JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (tg_id, mode, ok_code, ok_level)
);

CREATE TABLE IF NOT EXISTS ok_prefs (
  tg_id BIGINT NOT NULL REFERENCES users(tg_id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  ok_codes JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (tg_id, mode)
);

"""

DDL_MIGRATIONS = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ok_code TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ok_level INT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS train_mode TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS position TEXT",
]


async def db_init(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(DDL_CREATE)
        for stmt in DDL_MIGRATIONS:
            await conn.execute(stmt)

async def db_get_user(pool: asyncpg.Pool, tg_id: int) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_touch_user(pool: asyncpg.Pool, tg_id: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute("UPDATE users SET last_seen=$2 WHERE tg_id=$1", tg_id, utcnow())

async def db_upsert_user(pool: asyncpg.Pool, tg_id: int, phone: Optional[str], is_admin: bool) -> asyncpg.Record:
    now = utcnow()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)
        if existing is None:
            trial_until = now + timedelta(days=3)
            await conn.execute(
                """
                INSERT INTO users(tg_id, phone, created_at, trial_until, sub_until, is_admin, last_seen, ok_code, ok_level)
                VALUES($1, $2, $3, $4, NULL, $5, $3, NULL, NULL)
                """,
                tg_id, phone, now, trial_until, is_admin
            )
        else:
            await conn.execute(
                """
                UPDATE users
                SET phone = COALESCE($2, phone),
                    is_admin = (is_admin OR $3),
                    last_seen = $4
                WHERE tg_id=$1
                """,
                tg_id, phone, is_admin, now
            )
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_set_position(pool: asyncpg.Pool, tg_id: int, position: Optional[str]) -> asyncpg.Record:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET position=$2 WHERE tg_id=$1",
            tg_id,
            position,
        )
        return await conn.fetchrow(
            "SELECT * FROM users WHERE tg_id=$1",
            tg_id,
        )


async def db_set_scope(
    pool: asyncpg.Pool,
    tg_id: int,
    ok_code: str,
    ok_level: Optional[int] = None
) -> asyncpg.Record:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET ok_code=$2, ok_level=$3 WHERE tg_id=$1",
            tg_id, ok_code, ok_level
        )
        return await conn.fetchrow(
            "SELECT * FROM users WHERE tg_id=$1",
            tg_id
        )

async def db_has_access(user: asyncpg.Record) -> bool:
    now = utcnow()
    if user["is_admin"]:
        return True
    tu = user["trial_until"]
    su = user["sub_until"]
    if tu and tu > now:
        return True
    if su and su > now:
        return True
    return False

async def db_stats_add(pool: asyncpg.Pool, tg_id: int, mode: str, answered=0, correct=0, wrong=0, skipped=0) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO stats(tg_id, mode, answered, correct, wrong, skipped)
            VALUES($1,$2,$3,$4,$5,$6)
            ON CONFLICT (tg_id, mode)
            DO UPDATE SET
              answered = stats.answered + EXCLUDED.answered,
              correct  = stats.correct  + EXCLUDED.correct,
              wrong    = stats.wrong    + EXCLUDED.wrong,
              skipped  = stats.skipped  + EXCLUDED.skipped
            """,
            tg_id, mode, answered, correct, wrong, skipped
        )

async def db_stats_get(pool: asyncpg.Pool, tg_id: int) -> List[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM stats WHERE tg_id=$1 ORDER BY mode", tg_id)

async def db_create_session(pool: asyncpg.Pool, tg_id: int, mode: str, qids: List[int], expires_at: Optional[datetime]) -> uuid.UUID:
    sid = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET completed=TRUE WHERE tg_id=$1 AND mode=$2 AND completed=FALSE", tg_id, mode)
        await conn.execute(
            """
            INSERT INTO sessions(session_id, tg_id, mode, question_ids, current_index, correct_count, wrong_count, skipped_count, started_at, expires_at, completed)
            VALUES($1,$2,$3,$4,0,0,0,0,$5,$6,FALSE)
            """,
            sid, tg_id, mode, json.dumps(qids), utcnow(), expires_at
        )
    return sid

async def db_get_active_session(pool: asyncpg.Pool, tg_id: int, mode: str) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            """
            SELECT * FROM sessions
            WHERE tg_id=$1 AND mode=$2 AND completed=FALSE
            ORDER BY started_at DESC
            LIMIT 1
            """,
            tg_id, mode
        )

async def db_update_session_progress(
    pool: asyncpg.Pool,
    session_id: uuid.UUID,
    current_index: int,
    correct_delta: int = 0,
    wrong_delta: int = 0,
    skipped_delta: int = 0,
    completed: Optional[bool] = None,
) -> None:
    async with pool.acquire() as conn:
        if completed is None:
            await conn.execute(
                """
                UPDATE sessions
                SET current_index=$2,
                    correct_count=correct_count+$3,
                    wrong_count=wrong_count+$4,
                    skipped_count=skipped_count+$5
                WHERE session_id=$1
                """,
                session_id, current_index, correct_delta, wrong_delta, skipped_delta
            )
        else:
            await conn.execute(
                """
                UPDATE sessions
                SET current_index=$2,
                    correct_count=correct_count+$3,
                    wrong_count=wrong_count+$4,
                    skipped_count=skipped_count+$5,
                    completed=$6
                WHERE session_id=$1
                """,
                session_id, current_index, correct_delta, wrong_delta, skipped_delta, completed
            )



async def db_set_session_question_ids(
    pool: asyncpg.Pool,
    session_id: uuid.UUID,
    question_ids: List[int],
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET question_ids=$2
            WHERE session_id=$1
            """,
            session_id,
            json.dumps(question_ids),
        )

async def db_finish_session(pool: asyncpg.Pool, session_id: uuid.UUID) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        await conn.execute("UPDATE sessions SET completed=TRUE WHERE session_id=$1", session_id)
        return await conn.fetchrow("SELECT * FROM sessions WHERE session_id=$1", session_id)

async def db_list_users(pool: asyncpg.Pool, limit: int = 30) -> List[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT tg_id, phone, trial_until, sub_until, is_admin, last_seen, ok_code, ok_level
            FROM users
            ORDER BY last_seen DESC
            LIMIT $1
            """,
            limit
        )

async def db_set_sub_days(pool: asyncpg.Pool, tg_id: int, days: int) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        u = await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)
        if not u:
            return None
        base = u["sub_until"] if u["sub_until"] and u["sub_until"] > utcnow() else utcnow()
        new_until = base + timedelta(days=days)
        await conn.execute("UPDATE users SET sub_until=$2 WHERE tg_id=$1", tg_id, new_until)
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_revoke_sub(pool: asyncpg.Pool, tg_id: int) -> Optional[asyncpg.Record]:
    async with pool.acquire() as conn:
        u = await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)
        if not u:
            return None
        await conn.execute("UPDATE users SET sub_until=NULL WHERE tg_id=$1", tg_id)
        return await conn.fetchrow("SELECT * FROM users WHERE tg_id=$1", tg_id)

async def db_seed_problem_flags(pool: asyncpg.Pool, problem_ids: Set[int]) -> None:
    if not problem_ids:
        return
    now = utcnow()
    async with pool.acquire() as conn:
        rows = [(qid, True, "from_problem_questions.json", now) for qid in problem_ids]
        await conn.executemany(
            """
            INSERT INTO question_flags(question_id, is_disabled, note, updated_at)
            VALUES($1,$2,$3,$4)
            ON CONFLICT (question_id) DO UPDATE
            SET is_disabled=EXCLUDED.is_disabled,
                note=EXCLUDED.note,
                updated_at=EXCLUDED.updated_at
            """,
            rows
        )

async def db_get_disabled_ids(pool: asyncpg.Pool) -> Set[int]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT question_id FROM question_flags WHERE is_disabled=TRUE")
    return {int(r["question_id"]) for r in rows}

async def db_toggle_question(pool: asyncpg.Pool, qid: int, enable: bool, note: str) -> None:
    now = utcnow()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO question_flags(question_id, is_disabled, note, updated_at)
            VALUES($1,$2,$3,$4)
            ON CONFLICT (question_id) DO UPDATE
            SET is_disabled=EXCLUDED.is_disabled,
                note=EXCLUDED.note,
                updated_at=EXCLUDED.updated_at
            """,
            qid, (not enable), note, now
        )



async def db_get_topic_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_code: str, ok_level: int) -> Set[str]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT topics FROM topic_prefs WHERE tg_id=$1 AND mode=$2 AND ok_code=$3 AND ok_level=$4",
            tg_id, mode, ok_code, ok_level
        )
    if not row or row["topics"] is None:
        return set()
    topics = row["topics"]
    # asyncpg може віддати list напряму або str
    if isinstance(topics, str):
        try:
            topics = json.loads(topics)
        except Exception:
            topics = []
    if not isinstance(topics, list):
        topics = []
    return {str(t) for t in topics}

async def db_set_topic_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_code: str, ok_level: int, topics: Set[str]) -> None:
    now = utcnow()
    payload = json.dumps(sorted(list(topics)))
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO topic_prefs(tg_id, mode, ok_code, ok_level, topics, updated_at) "
            "VALUES($1,$2,$3,$4,$5,$6) "
            "ON CONFLICT (tg_id, mode, ok_code, ok_level) "
            "DO UPDATE SET topics=EXCLUDED.topics, updated_at=EXCLUDED.updated_at",
            tg_id, mode, ok_code, ok_level, payload, now
        )

async def db_clear_topic_prefs(pool: asyncpg.Pool, tg_id: int, mode: str, ok_code: str, ok_level: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM topic_prefs WHERE tg_id=$1 AND mode=$2 AND ok_code=$3 AND ok_level=$4",
            tg_id, mode, ok_code, ok_level
        )





async def db_defer_question_to_end(
    pool: asyncpg.Pool,
    session_id: uuid.UUID,
    new_qids: List[int],
    current_index: int,
    skipped_delta: int = 1,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE sessions
            SET question_ids=$2,
                current_index=$3,
                skipped_count=skipped_count+$4
            WHERE session_id=$1
            """,
            session_id,
            json.dumps(new_qids),
            current_index,
            skipped_delta,
        )
