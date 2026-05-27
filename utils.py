from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

try:
    TZ = ZoneInfo("Europe/Kyiv")
except Exception:
    TZ = timezone.utc

# Links
GROUP_URL = _normalize_tme_url = None  # defined below after the function


def normalize_tme_url(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith(("http://", "https://", "tg://")):
        return s
    if s.startswith("t.me/"):
        return "https://" + s
    if s.startswith("@"):
        return "https://t.me/" + s.lstrip("@")
    return s


GROUP_URL = normalize_tme_url(os.getenv("GROUP_URL", "t.me/mytnytsia_test"))

# --- keys у state ---
ADMIN_PANEL_MSG_ID = "admin_panel_msg_id"
ADMIN_PANEL_CHAT_ID = "admin_panel_chat_id"

ADMIN_QWORK_AWAITING = "admin_qwork_awaiting"
ADMIN_QWORK_PAGE = "admin_qwork_page"
ADMIN_QEDIT = "admin_qedit"
ADMIN_QWORK_QUERY = "admin_qwork_query"

CASES_PER_PAGE = 8
CASE_QUESTIONS_PER_PAGE = 3
OK_QUESTIONS_PER_PAGE = 3

OK_SEARCH_AWAITING = "ok_search_awaiting"
OK_SEARCH_QUERY = "ok_search_query"


def get_admin_contact_url(admin_ids: set[int]) -> str:
    url = normalize_tme_url(os.getenv("ADMIN_CONTACT_URL", ""))
    if url:
        return url
    username = (os.getenv("ADMIN_USERNAME", "") or "").strip().lstrip("@")
    if username:
        return f"https://t.me/{username}"
    if admin_ids:
        admin_id = next(iter(admin_ids))
        return f"tg://user?id={admin_id}"
    return ""


def now() -> datetime:
    return datetime.now(TZ)


def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s)


def clamp_callback(s: str, max_bytes: int = 64) -> str:
    s = (s or "").strip()
    if not s:
        return "0"
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    return b[:max_bytes].decode("utf-8", errors="ignore")


def case_bank_sort_key(item: dict) -> tuple[int, str, int]:
    raw_number = str(item.get("case_number") or "")
    title = str(item.get("case_title") or "")
    match = re.search(r"\d+", f"{raw_number} {title}")
    numeric = int(match.group(0)) if match else 10**12
    return numeric, raw_number.casefold() or title.casefold(), int(item.get("id") or 0)


def normalize_postgres_dsn(dsn: str) -> tuple[str, object | None]:
    if not dsn:
        return dsn, None
    if dsn.startswith("postgres://"):
        dsn = "postgresql://" + dsn[len("postgres://"):]
    parsed = urlparse(dsn)
    qs = parse_qs(parsed.query)
    sslmode = (qs.get("sslmode", [None])[0] or "").lower() if qs else ""
    ssl_param = None
    if sslmode in {"require", "verify-ca", "verify-full"}:
        ssl_param = True
    for k in ["sslmode"]:
        qs.pop(k, None)
    new_query = urlencode({k: v[0] for k, v in qs.items()}) if qs else ""
    cleaned = urlunparse(parsed._replace(query=new_query))
    return cleaned, ssl_param
