#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = BASE_DIR / "data" / "customs_code_source.htm"
DEFAULT_DB = BASE_DIR / "data" / "customs_code.db"

SECTION_RE = re.compile(r"^Розділ\s+([^\s]+)\s+(.+)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^Глава\s+([0-9]+(?:-[0-9]+)*)\s*\.\s*(.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Стаття\s+([0-9]+(?:-[0-9]+)*)\s*\.\s*(.*)$", re.IGNORECASE)
TRANSITIONAL_POINT_RE = re.compile(r"^([0-9]+(?:-[0-9]+)*)\.\s+(.+)$")
TRANSITIONAL_SECTION_NUMBER = "XXI"



def clean_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1-\2", value)
    value = re.sub(r"\s+([.,;:!?])", r"\1", value)
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\s+\)", ")", value)
    value = re.sub(r"№\s+", "№ ", value)
    return value.strip()


def is_excluded(title: str, text: str = "") -> bool:
    blob = title.strip().lower()
    return "виключено" in blob and ("статт" in blob or "глав" in blob)


@dataclass
class Article:
    number: str
    title: str
    section_id: int
    chapter_id: Optional[int]
    position: int
    anchor: str = ""
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in self.paragraphs if p).strip()


@dataclass
class Chapter:
    id: int
    section_id: int
    number: str
    title: str
    position: int
    anchor: str = ""
    is_excluded: bool = False


@dataclass
class Section:
    id: int
    number: str
    title: str
    position: int
    anchor: str = ""


def get_anchor(tag) -> str:
    anchor = tag.find("a") if tag else None
    return (anchor.get("name") or "").strip() if anchor else ""


def make_article_title(prefix: str, body: str, limit: int = 180) -> str:
    body = clean_text(body)
    if len(body) > limit:
        body = body[:limit].rstrip(" ,;:-") + "…"
    if prefix and body:
        return f"{prefix}. {body}"
    return body or prefix


def append_transitional_points(
    section: Optional[Section],
    raw_paragraphs: list[tuple[str, str]],
    chapters: list[Chapter],
    articles: list[Article],
) -> None:
    if section is None or not raw_paragraphs:
        return

    chapter = Chapter(
        id=len(chapters) + 1,
        section_id=section.id,
        number=TRANSITIONAL_SECTION_NUMBER,
        title=section.title,
        position=len(chapters) + 1,
        anchor=section.anchor,
        is_excluded=False,
    )
    chapters.append(chapter)

    current_number: Optional[str] = None
    current_title = ""
    current_anchor = ""
    current_paragraphs: list[str] = []

    def flush_point() -> None:
        nonlocal current_number, current_title, current_anchor, current_paragraphs
        if current_number is None:
            return
        article_number = f"{TRANSITIONAL_SECTION_NUMBER}-{current_number}"
        existing = next((article for article in articles if article.number == article_number), None)
        if existing is not None:
            existing.paragraphs.extend(current_paragraphs)
        else:
            articles.append(
                Article(
                    number=article_number,
                    title=make_article_title("", current_title),
                    section_id=section.id,
                    chapter_id=chapter.id,
                    position=len(articles) + 1,
                    anchor=current_anchor,
                    paragraphs=current_paragraphs,
                )
            )
        current_number = None
        current_title = ""
        current_anchor = ""
        current_paragraphs = []

    for text, anchor in raw_paragraphs:
        point_match = TRANSITIONAL_POINT_RE.match(text)
        if point_match:
            flush_point()
            current_number = point_match.group(1).strip()
            current_title = point_match.group(2).strip()
            current_anchor = anchor
            current_paragraphs = [text]
        elif current_number is not None:
            current_paragraphs.append(text)

    flush_point()


def parse_html(source: Path) -> tuple[list[Section], list[Chapter], list[Article], dict[str, str]]:
    html = source.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    meta = {
        "title": "Митний кодекс України",
        "document_id": "4495-VI / 4495-17",
        "edition_date": "2026-01-04",
        "source": "Офіційний текст Верховної Ради України",
        "source_file": source.name,
    }

    title_tag = soup.find("title")
    if title_tag:
        meta["html_title"] = clean_text(title_tag.get_text(" ", strip=True))

    sections: list[Section] = []
    chapters: list[Chapter] = []
    articles: list[Article] = []

    current_section: Optional[Section] = None
    current_chapter: Optional[Chapter] = None
    current_article: Optional[Article] = None
    transitional_section: Optional[Section] = None
    transitional_paragraphs: list[tuple[str, str]] = []
    started = False

    def finish_article() -> None:
        nonlocal current_article
        if current_article is not None:
            articles.append(current_article)
            current_article = None

    for tag in soup.find_all("p"):
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue

        section_match = SECTION_RE.match(text)
        if section_match:
            finish_article()
            started = True
            current_section = Section(
                id=len(sections) + 1,
                number=section_match.group(1).strip(),
                title=section_match.group(2).strip(),
                position=len(sections) + 1,
                anchor=get_anchor(tag),
            )
            sections.append(current_section)
            if current_section.number.upper() == TRANSITIONAL_SECTION_NUMBER:
                transitional_section = current_section
            current_chapter = None
            continue

        if not started:
            continue

        chapter_match = CHAPTER_RE.match(text)
        if chapter_match:
            finish_article()
            if current_section is None:
                continue
            title = chapter_match.group(2).strip()
            current_chapter = Chapter(
                id=len(chapters) + 1,
                section_id=current_section.id,
                number=chapter_match.group(1).strip(),
                title=title,
                position=len(chapters) + 1,
                anchor=get_anchor(tag),
                is_excluded=is_excluded(title),
            )
            chapters.append(current_chapter)
            continue

        article_match = ARTICLE_RE.match(text)
        if article_match:
            finish_article()
            if current_section is None:
                continue
            if current_chapter is None:
                current_chapter = Chapter(
                    id=len(chapters) + 1,
                    section_id=current_section.id,
                    number="0",
                    title="Без глави",
                    position=len(chapters) + 1,
                    anchor="",
                    is_excluded=False,
                )
                chapters.append(current_chapter)
            current_article = Article(
                number=article_match.group(1).strip(),
                title=article_match.group(2).strip(),
                section_id=current_section.id,
                chapter_id=current_chapter.id,
                position=len(articles) + 1,
                anchor=get_anchor(tag),
            )
            continue

        if current_article is not None:
            current_article.paragraphs.append(text)
        elif current_section is not None and current_section.number.upper() == TRANSITIONAL_SECTION_NUMBER:
            transitional_paragraphs.append((text, get_anchor(tag)))

    finish_article()
    append_transitional_points(transitional_section, transitional_paragraphs, chapters, articles)
    return sections, chapters, articles, meta


def build_db(db_path: Path, sections: list[Section], chapters: list[Chapter], articles: list[Article], meta: dict[str, str]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(
        """
        CREATE TABLE document_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE customs_sections (
          id INTEGER PRIMARY KEY,
          number TEXT NOT NULL,
          title TEXT NOT NULL,
          position INTEGER NOT NULL,
          anchor TEXT DEFAULT ''
        );

        CREATE TABLE customs_chapters (
          id INTEGER PRIMARY KEY,
          section_id INTEGER NOT NULL REFERENCES customs_sections(id) ON DELETE CASCADE,
          number TEXT NOT NULL,
          title TEXT NOT NULL,
          position INTEGER NOT NULL,
          anchor TEXT DEFAULT '',
          is_excluded INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE customs_articles (
          id INTEGER PRIMARY KEY,
          section_id INTEGER NOT NULL REFERENCES customs_sections(id) ON DELETE CASCADE,
          chapter_id INTEGER REFERENCES customs_chapters(id) ON DELETE SET NULL,
          number TEXT NOT NULL UNIQUE,
          title TEXT NOT NULL,
          text TEXT NOT NULL,
          position INTEGER NOT NULL,
          anchor TEXT DEFAULT '',
          is_excluded INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX idx_customs_chapters_section ON customs_chapters(section_id, position);
        CREATE INDEX idx_customs_articles_section ON customs_articles(section_id, position);
        CREATE INDEX idx_customs_articles_chapter ON customs_articles(chapter_id, position);
        CREATE INDEX idx_customs_articles_number ON customs_articles(number);

        CREATE VIRTUAL TABLE customs_articles_fts USING fts5(
          number,
          title,
          text,
          content='customs_articles',
          content_rowid='id',
          tokenize='unicode61'
        );
        """
    )

    conn.executemany(
        "INSERT INTO document_meta(key, value) VALUES(?, ?)",
        sorted(meta.items()),
    )
    conn.executemany(
        "INSERT INTO customs_sections(id, number, title, position, anchor) VALUES(?, ?, ?, ?, ?)",
        [(s.id, s.number, s.title, s.position, s.anchor) for s in sections],
    )
    conn.executemany(
        """
        INSERT INTO customs_chapters(id, section_id, number, title, position, anchor, is_excluded)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        [(c.id, c.section_id, c.number, c.title, c.position, c.anchor, int(c.is_excluded)) for c in chapters],
    )
    rows = []
    for a in articles:
        text = a.text
        rows.append((
            a.position,
            a.section_id,
            a.chapter_id,
            a.number,
            a.title,
            text,
            a.position,
            a.anchor,
            int(is_excluded(a.title, text)),
        ))
    conn.executemany(
        """
        INSERT INTO customs_articles(id, section_id, chapter_id, number, title, text, position, anchor, is_excluded)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.execute("INSERT INTO customs_articles_fts(customs_articles_fts) VALUES('rebuild')")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official Customs Code HTML into local SQLite DB.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to official HTML file")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Output SQLite DB path")
    args = parser.parse_args()

    sections, chapters, articles, meta = parse_html(args.source)
    if len(sections) < 10 or len(chapters) < 50 or len(articles) < 500:
        raise SystemExit(
            f"Parsed suspiciously little data: sections={len(sections)}, chapters={len(chapters)}, articles={len(articles)}"
        )
    build_db(args.db, sections, chapters, articles, meta)
    print(f"Created {args.db}")
    print(f"Sections: {len(sections)}")
    print(f"Chapters: {len(chapters)}")
    print(f"Articles: {len(articles)}")


if __name__ == "__main__":
    main()
