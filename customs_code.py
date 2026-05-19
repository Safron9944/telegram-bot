from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CUSTOMS_CODE_DB = BASE_DIR / "data" / "customs_code.db"

_TRANSITIONAL_NUMBER_RE = re.compile(
    r"(?:ст(?:аття)?\.?\s*)?(?:розділ\s*)?(?:xxi|ххі|хxi|xxі)[\s-]*(?:пункт\s*)?([0-9]+(?:-[0-9]+)*)",
    re.IGNORECASE,
)
_TRANSITIONAL_POINT_RE = re.compile(
    r"(?:п(?:ункт)?\.?\s*)([0-9]+(?:-[0-9]+)*)(?:\s+розділу\s*(?:xxi|ххі|хxi|xxі))?",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?:ст(?:аття)?\.?\s*)?([0-9]+(?:-[0-9]+)*)", re.IGNORECASE)
# FTS5 treats hyphen as a query operator, so we split article-like values
# (for example 3-1 or XXI-9-16) into safe tokens before building MATCH queries.
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+")


def normalize_article_number(value: str) -> str | None:
    value = (value or "").strip().replace("–", "-").replace("—", "-")
    transitional = _TRANSITIONAL_NUMBER_RE.fullmatch(value) or _TRANSITIONAL_NUMBER_RE.search(value)
    if transitional:
        return f"XXI-{transitional.group(1)}"
    point = _TRANSITIONAL_POINT_RE.fullmatch(value)
    if point:
        return f"XXI-{point.group(1)}"
    match = _NUMBER_RE.fullmatch(value) or _NUMBER_RE.search(value)
    return match.group(1) if match else None


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def _fts_query(query: str) -> str:
    normalized = (query or "").lower().replace("–", " ").replace("—", " ").replace("-", " ")
    tokens = _TOKEN_RE.findall(normalized)[:8]
    safe_tokens = []
    for token in tokens:
        token = token.replace('"', "")
        if not token:
            continue
        if len(token) >= 2:
            safe_tokens.append(f'{token}*')
        elif token.isdigit():
            safe_tokens.append(token)
    return " ".join(safe_tokens)


class CustomsCodeRepository:
    def __init__(self, db_path: Path | str = DEFAULT_CUSTOMS_CODE_DB):
        self.db_path = Path(db_path)

    def exists(self) -> bool:
        return self.db_path.exists()

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Customs code DB not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def status(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"available": False, "message": "Базу Митного кодексу ще не створено."}
        with self._connect() as conn:
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM document_meta")}
            counts = {
                "sections": conn.execute("SELECT COUNT(*) FROM customs_sections").fetchone()[0],
                "chapters": conn.execute("SELECT COUNT(*) FROM customs_chapters").fetchone()[0],
                "articles": conn.execute("SELECT COUNT(*) FROM customs_articles").fetchone()[0],
            }
        return {"available": True, "meta": meta, "counts": counts}

    def sections(self) -> dict[str, Any]:
        with self._connect() as conn:
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM document_meta")}
            rows = conn.execute(
                """
                SELECT
                  s.id,
                  s.number,
                  s.title,
                  s.position,
                  COUNT(DISTINCT c.id) AS chapters_count,
                  COUNT(DISTINCT a.id) AS articles_count,
                  MIN(CAST(c.number AS INTEGER)) AS first_chapter,
                  MAX(CAST(c.number AS INTEGER)) AS last_chapter,
                  MIN(CAST(a.number AS INTEGER)) AS first_article,
                  MAX(CAST(a.number AS INTEGER)) AS last_article
                FROM customs_sections s
                LEFT JOIN customs_chapters c ON c.section_id = s.id
                LEFT JOIN customs_articles a ON a.section_id = s.id
                GROUP BY s.id
                ORDER BY s.position
                """
            ).fetchall()
        return {"meta": meta, "items": [dict(row) for row in rows]}

    def section_detail(self, section_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            section = _row_to_dict(
                conn.execute(
                    "SELECT id, number, title, position FROM customs_sections WHERE id = ?",
                    (section_id,),
                ).fetchone()
            )
            if not section:
                return None

            chapters_rows = conn.execute(
                """
                SELECT id, number, title, position, is_excluded
                FROM customs_chapters
                WHERE section_id = ?
                ORDER BY position
                """,
                (section_id,),
            ).fetchall()
            articles_rows = conn.execute(
                """
                SELECT id, chapter_id, number, title, position, is_excluded
                FROM customs_articles
                WHERE section_id = ?
                ORDER BY position
                """,
                (section_id,),
            ).fetchall()

        articles_by_chapter: dict[int, list[dict[str, Any]]] = {}
        for row in articles_rows:
            item = dict(row)
            articles_by_chapter.setdefault(int(item["chapter_id"] or 0), []).append(item)

        chapters = []
        for row in chapters_rows:
            chapter = dict(row)
            chapter["articles"] = articles_by_chapter.get(int(chapter["id"]), [])
            chapter["articles_count"] = len(chapter["articles"])
            chapters.append(chapter)

        return {"section": section, "chapters": chapters}

    def article(self, article_number: str) -> dict[str, Any] | None:
        number = normalize_article_number(article_number) or article_number
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  a.id,
                  a.number,
                  a.title,
                  a.text,
                  a.position,
                  a.is_excluded,
                  s.id AS section_id,
                  s.number AS section_number,
                  s.title AS section_title,
                  c.id AS chapter_id,
                  c.number AS chapter_number,
                  c.title AS chapter_title
                FROM customs_articles a
                JOIN customs_sections s ON s.id = a.section_id
                LEFT JOIN customs_chapters c ON c.id = a.chapter_id
                WHERE a.number = ?
                """,
                (number,),
            ).fetchone()
        return _row_to_dict(row)

    def search(self, query: str, limit: int = 25, offset: int = 0) -> dict[str, Any]:
        query = (query or "").strip()
        limit = max(1, min(int(limit or 25), 50))
        offset = max(0, int(offset or 0))
        if not query:
            return {"items": [], "query": query, "limit": limit, "offset": offset, "has_more": False}

        exact_number = normalize_article_number(query)
        with self._connect() as conn:
            items: list[dict[str, Any]] = []
            seen: set[int] = set()

            if exact_number:
                exact = conn.execute(
                    """
                    SELECT
                      a.id, a.number, a.title, a.is_excluded,
                      s.id AS section_id, s.number AS section_number, s.title AS section_title,
                      c.id AS chapter_id, c.number AS chapter_number, c.title AS chapter_title,
                      substr(a.text, 1, 220) AS snippet,
                      0 AS rank
                    FROM customs_articles a
                    JOIN customs_sections s ON s.id = a.section_id
                    LEFT JOIN customs_chapters c ON c.id = a.chapter_id
                    WHERE a.number = ?
                    """,
                    (exact_number,),
                ).fetchone()
                if exact:
                    item = dict(exact)
                    item["snippet"] = (item.get("snippet") or "").strip()
                    items.append(item)
                    seen.add(int(item["id"]))

            title_rows = [
                row for row in conn.execute(
                    """
                    SELECT
                      a.id, a.number, a.title, a.is_excluded,
                      s.id AS section_id, s.number AS section_number, s.title AS section_title,
                      c.id AS chapter_id, c.number AS chapter_number, c.title AS chapter_title,
                      substr(a.text, 1, 220) AS snippet,
                      1 AS rank
                    FROM customs_articles a
                    JOIN customs_sections s ON s.id = a.section_id
                    LEFT JOIN customs_chapters c ON c.id = a.chapter_id
                    ORDER BY a.position
                    """
                ).fetchall()
                if query.lower() in str(row["title"]).lower()
            ][:limit]
            for row in title_rows:
                item = dict(row)
                if int(item["id"]) in seen:
                    continue
                item["snippet"] = (item.get("snippet") or "").strip()
                items.append(item)
                seen.add(int(item["id"]))

            fts = _fts_query(query)
            if fts:
                rows = conn.execute(
                    """
                    SELECT
                      a.id, a.number, a.title, a.is_excluded,
                      s.id AS section_id, s.number AS section_number, s.title AS section_title,
                      c.id AS chapter_id, c.number AS chapter_number, c.title AS chapter_title,
                      snippet(customs_articles_fts, 2, '‹', '›', '…', 22) AS snippet,
                      bm25(customs_articles_fts) AS rank
                    FROM customs_articles_fts
                    JOIN customs_articles a ON a.id = customs_articles_fts.rowid
                    JOIN customs_sections s ON s.id = a.section_id
                    LEFT JOIN customs_chapters c ON c.id = a.chapter_id
                    WHERE customs_articles_fts MATCH ?
                    ORDER BY rank, a.position
                    LIMIT ? OFFSET ?
                    """,
                    (fts, limit + 1, offset),
                ).fetchall()
                for row in rows:
                    item = dict(row)
                    if int(item["id"]) in seen:
                        continue
                    item["snippet"] = (item.get("snippet") or "").strip()
                    items.append(item)
                    seen.add(int(item["id"]))
                    if len(items) >= limit + 1:
                        break

        return {
            "items": items[:limit],
            "query": query,
            "limit": limit,
            "offset": offset,
            "has_more": len(items) > limit,
        }


repository = CustomsCodeRepository()
