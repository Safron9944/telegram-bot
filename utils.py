from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
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


OK_TITLES: dict[str, str] = {
    "ОК-1": "Кінологічне забезпечення",
    "ОК-2": "Технічні засоби здійснення митного контролю",
    "ОК-3": "Класифікація товарів",
    "ОК-4": "Митні платежі",
    "ОК-5": "Походження товарів",
    "ОК-6": "Митна вартість",
    "ОК-7": "Нетарифне регулювання",
    "ОК-8": "Контроль за міжнародними передачами стратегічних товарів",
    "ОК-9": "Авторизації",
    "ОК-10": "Митні процедури",
    "ОК-11": "Митний аудит",
    "ОК-12": "Митна статистика",
    "ОК-13": "Транзитні процедури",
    "ОК-14": "Протидія контрабанді та боротьба з порушеннями митних правил",
    "ОК-15": "Управління ризиками",
    "ОК-16": "Захист прав інтелектуальної власності",
    "ОК-17": "Підтримка митниці (організаційне забезпечення)",
}

_ok_code_any_re = re.compile(r"(?i)(?:\[\s*)?(?:ОК|OK)\s*[-–]?\s*(\d+)(?:\s*\])?")


def clean_law_title(title: str) -> str:
    t = (title or "").strip()
    for p in [
        "Питання на перевірку знання ",
        "Питання на перевірку знань ",
        "Питання на перевірку знання",
        "Питання на перевірку знань",
    ]:
        if t.startswith(p):
            t = t[len(p):].strip()
    return t


def ok_extract_code(s: str) -> Optional[str]:
    if not s:
        return None
    m = _ok_code_any_re.search(s.strip())
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    return f"ОК-{n}"


def ok_full_label(s: str) -> str:
    code = ok_extract_code(s) or (s or "").strip()
    if code in OK_TITLES:
        return f"[{code}] {OK_TITLES[code]}"
    return s


def ok_sort_key(name: str) -> Tuple:
    code = ok_extract_code(name)
    if code:
        try:
            return (0, int(code.split("-", 1)[1]))
        except Exception:
            pass
    return (1, (name or "").lower())


def _norm_qsearch(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.U)
    s = re.sub(r"\s+", " ", s, flags=re.U)
    return s.strip()


def find_question_ids_by_title(qb: "QuestionBank", query: str, limit: int = 12) -> List[int]:
    qn = _norm_qsearch(query)
    if not qn:
        return []
    q_tokens = [t for t in qn.split() if t]
    res: List[Tuple[float, int]] = []
    for qid, q in qb.by_id.items():
        qt = _norm_qsearch(getattr(q, "question", "") or "")
        if not qt:
            continue
        if q_tokens and not any(t in qt for t in q_tokens):
            continue
        ratio = SequenceMatcher(None, qn, qt).ratio()
        if qn in qt:
            ratio = max(ratio, 0.95)
        qtoks = set(qt.split())
        qs = set(q_tokens)
        jacc = (len(qs & qtoks) / max(1, len(qs | qtoks))) if qs else 0.0
        score = 0.75 * ratio + 0.25 * jacc
        res.append((score, int(qid)))
    res.sort(key=lambda x: x[0], reverse=True)
    return [qid for _, qid in res[: max(1, int(limit))]]


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
