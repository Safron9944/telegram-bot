from __future__ import annotations

import os
from zoneinfo import ZoneInfo
from typing import Dict, List, Set


# -------------------------
# Конфіг
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

ADMIN_IDS: Set[int] = set()
if os.getenv("ADMIN_IDS"):
    for x in os.getenv("ADMIN_IDS", "").split(","):
        x = x.strip()
        if x.isdigit():
            ADMIN_IDS.add(int(x))

TRAIN_QUESTIONS = int(os.getenv("TRAIN_QUESTIONS", "20"))
EXAM_QUESTIONS = int(os.getenv("EXAM_QUESTIONS", "100"))
EXAM_DURATION_MINUTES = int(os.getenv("EXAM_DURATION_MINUTES", "90"))

EXAM_LAW_QUESTIONS = int(os.getenv("EXAM_LAW_QUESTIONS", "50"))
EXAM_PER_TOPIC_QUESTIONS = int(os.getenv("EXAM_PER_TOPIC_QUESTIONS", "20"))


QUESTIONS_FILE = os.getenv("QUESTIONS_FILE", "questions_flat.json")
PROBLEMS_FILE = os.getenv("PROBLEMS_FILE", "problem_questions.json")

KYIV_TZ = ZoneInfo("Europe/Kyiv")
OK_CODE_LAW = "LAW"  # внутрішній код для "законодавства"
LEVEL_ALL = -1  # спеціальне значення: всі рівні для обраного ОК

PENDING_AFTER_OK: dict[int, str] = {}  # tg_id -> "train" | "exam"
REG_PROMPT_MSG_ID: dict[int, int] = {}  # tg_id -> message_id (реєстраційний текст)

POSITION_OK_MAP: Dict[str, Dict[str, int]] = {
    "Начальник відділу": {
        "ОК-4": 2,
        "ОК-10": 3,
        "ОК-14": 2,
        "ОК-15": 2,
    },
    "Головний державний інспектор": {
        "ОК-4": 2,
        "ОК-10": 3,
        "ОК-14": 2,
        "ОК-15": 2,
    },
    "Старший державний інспектор": {
        "ОК-4": 1,
        "ОК-10": 2,
        "ОК-14": 1,
        "ОК-15": 1,
    },
    "Державний інспектор": {
        "ОК-4": 1,
        "ОК-10": 2,
        "ОК-14": 1,
        "ОК-15": 1,
    },
}

POSITIONS: List[str] = list(POSITION_OK_MAP.keys())
POS_ID_BY_NAME: Dict[str, int] = {name: i for i, name in enumerate(POSITIONS)}
POS_NAME_BY_ID: Dict[int, str] = {i: name for name, i in POS_ID_BY_NAME.items()}

def pos_id(name: str) -> int:
    return POS_ID_BY_NAME.get(name, -1)

def pos_name(pid: int) -> str:
    return POS_NAME_BY_ID.get(pid, "")


