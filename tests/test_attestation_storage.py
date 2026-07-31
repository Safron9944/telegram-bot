import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from storage import Storage


@pytest.mark.asyncio
async def test_import_attestation_upserts_verified_but_preserves_admin_row():
    store = Storage("postgresql://unused")
    con = AsyncMock()
    con.fetchrow.return_value = {"verification_method": "admin"}

    imported = await store._upsert_attestation_row(
        con,
        {
            "section": "constitution",
            "qnum": 1,
            "question": "Текст",
            "choices": ["А", "Б"],
            "correct": [1],
            "source_page": 3,
            "source_hash": "new",
            "verification_method": "pdf_visual",
        },
    )

    assert imported is False
    con.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_import_can_replace_admin_row():
    store = Storage("postgresql://unused")
    con = AsyncMock()
    con.fetchrow.return_value = {"verification_method": "admin"}

    imported = await store._upsert_attestation_row(
        con,
        {
            "section": "constitution",
            "section_title": "Конституція України",
            "qnum": 1,
            "question": "Текст",
            "choices": ["А", "Б"],
            "correct": [1],
            "source_page": 3,
            "source_hash": "new",
            "verification_method": "pdf_visual",
        },
        force=True,
    )

    assert imported is True
    con.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_attestation_questions_decodes_json():
    store = Storage("postgresql://unused")
    store._fetch = AsyncMock(
        return_value=[
            {
                "id": 4,
                "section": "constitution",
                "section_title": "Конституція України",
                "qnum": 1,
                "question": "Текст",
                "choices": '["А", "Б"]',
                "correct": "[1]",
                "source_page": 3,
                "source_hash": "h",
                "verification_method": "pdf_visual",
                "match_evidence": "[]",
            }
        ]
    )

    rows = await store.fetch_attestation_questions()

    assert rows[0]["choices"] == ["А", "Б"]
    assert rows[0]["correct"] == [1]
    assert rows[0]["match_evidence"] == []


@pytest.mark.asyncio
async def test_save_test_persists_type_and_metadata():
    store = Storage("postgresql://unused")
    store._exec = AsyncMock()
    instant = datetime(2026, 7, 31, tzinfo=timezone.utc)

    await store.save_test(
        7,
        instant,
        instant,
        10,
        8,
        test_type="attestation",
        meta={"section": "constitution", "mode": "demo"},
    )

    args = store._exec.await_args.args
    assert "test_type" in args[0]
    assert args[-2] == "attestation"
    assert json.loads(args[-1]) == {"section": "constitution", "mode": "demo"}


@pytest.mark.asyncio
async def test_standard_and_attestation_statistics_are_isolated():
    store = Storage("postgresql://unused")
    store._fetch = AsyncMock(return_value=[])

    await store.stats(5)
    standard_sql = store._fetch.await_args.args[0]
    await store.attestation_stats(5)
    attestation_sql = store._fetch.await_args.args[0]

    assert "test_type='standard'" in standard_sql
    assert "test_type='attestation'" in attestation_sql
