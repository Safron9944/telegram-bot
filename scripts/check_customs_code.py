#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "customs_code.db"


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    counts = {
        "sections": conn.execute("SELECT COUNT(*) FROM customs_sections").fetchone()[0],
        "chapters": conn.execute("SELECT COUNT(*) FROM customs_chapters").fetchone()[0],
        "articles": conn.execute("SELECT COUNT(*) FROM customs_articles").fetchone()[0],
    }
    print(counts)

    required_articles = ["1", "3-1", "4", "49", "257", "590", "XXI-1", "XXI-9-16", "XXI-10"]
    for number in required_articles:
        row = conn.execute(
            """
            SELECT a.number, a.title, s.number AS section_number, c.number AS chapter_number, length(a.text) AS text_len
            FROM customs_articles a
            JOIN customs_sections s ON s.id = a.section_id
            LEFT JOIN customs_chapters c ON c.id = a.chapter_id
            WHERE a.number = ?
            """,
            (number,),
        ).fetchone()
        if not row:
            raise SystemExit(f"Missing article {number}")
        print(f"article {row['number']}: section {row['section_number']}, chapter {row['chapter_number']}, {row['text_len']} chars")

    search_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM customs_articles_fts
        WHERE customs_articles_fts MATCH ?
        """,
        ("митна вартість",),
    ).fetchone()[0]
    if search_count < 1:
        raise SystemExit("FTS search failed for: митна вартість")
    print(f"search 'митна вартість': {search_count} matches")

    transitional = conn.execute(
        """
        SELECT s.number, COUNT(DISTINCT c.id) AS chapters_count, COUNT(DISTINCT a.id) AS articles_count
        FROM customs_sections s
        LEFT JOIN customs_chapters c ON c.section_id = s.id
        LEFT JOIN customs_articles a ON a.section_id = s.id
        WHERE s.number = 'XXI'
        GROUP BY s.id
        """
    ).fetchone()
    if not transitional or transitional["chapters_count"] < 1 or transitional["articles_count"] < 1:
        raise SystemExit("Missing Section XXI transitional provisions structure")
    print(f"section XXI: {transitional['chapters_count']} chapter, {transitional['articles_count']} articles")

    xxi_search_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM customs_articles_fts
        WHERE customs_articles_fts MATCH ?
        """,
        ("воєнного стану",),
    ).fetchone()[0]
    if xxi_search_count < 1:
        raise SystemExit("FTS search failed for transitional provisions")
    print(f"search 'воєнного стану': {xxi_search_count} matches")

    print("OK")


if __name__ == "__main__":
    main()
