from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .config import KYIV_TZ, OK_CODE_LAW, LEVEL_ALL


# -------------------------
# Допоміжні утиліти
# -------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def as_minutes_seconds(seconds: int) -> str:
    seconds = max(0, int(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def is_question_valid(q: Dict[str, Any]) -> bool:
    """Валідне питання: 1 правильна відповідь і ≥2 варіанти."""
    try:
        choices = q.get("choices") or []
        correct = q.get("correct") or []
        if len(choices) < 2:
            return False
        if len(correct) != 1:
            return False
        ci = int(correct[0])
        if ci < 0 or ci >= len(choices):
            return False
        if not (q.get("question") or "").strip():
            return False
        return True
    except Exception:
        return False

def scope_title(ok_code: str, level: int | None = None) -> str:
    if ok_code == OK_CODE_LAW:
        return "📜 Законодавство"
    if level is None:
        return ok_code
    if int(level) == LEVEL_ALL:
        return f"{ok_code} • всі рівні"
    return f"{ok_code} • рівень {int(level)}"

def truncate_button(text: str, max_len: int = 44) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"

def normalize_ok_code(raw_ok: Any) -> str:
    # у файлі законодавство має ok=None
    return OK_CODE_LAW if raw_ok is None else str(raw_ok)

def normalize_level(raw_level: Any, ok_code: str) -> int:
    if ok_code == OK_CODE_LAW:
        return 0
    if raw_level is None:
        # на випадок неконсистентних даних
        return 1
    return int(raw_level)
