from __future__ import annotations

import json
from typing import Any

from access import access_tier


PROTECTED_SECTION_KEYS = frozenset({"cases", "test_questions", "question_search"})


SYSTEM_SECTIONS = (
    {"key": "customs", "title": "Митні компетенції", "screen": "customs", "icon": "graduation", "price": 250, "group": "primary", "default_order": 100, "content_screen": "admin-questions", "content_label": "Банк питань"},
    {"key": "cases", "title": "Кейси", "screen": "cases", "icon": "folder", "price": 100, "manual_grant_only": True, "group": "materials", "default_order": 200, "content_screen": "admin-cases", "content_label": "Кейси та питання"},
    {"key": "customs_code", "title": "Митний кодекс", "screen": "customs-code", "icon": "scale", "price": 0, "group": "materials", "default_order": 201},
    {"key": "test_questions", "title": "Тестові питання", "screen": "test-exam-questions", "icon": "clipboard", "price": 250, "visible": False, "manual_grant_only": True, "group": "materials", "default_order": 202, "content_screen": "admin-test-questions", "content_label": "Питання"},
    {"key": "question_search", "title": "Пошук питань", "screen": "question-search", "icon": "search", "price": 250, "manual_grant_only": True, "group": "materials", "default_order": 203},
    {"key": "support", "title": "Підтримка", "screen": "help", "icon": "support", "price": 0, "group": "help", "default_order": 300},
)


async def section_config(store) -> dict[str, dict[str, Any]]:
    raw = await store.get_setting("section_config", "{}")
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        value = {}
    return value if isinstance(value, dict) else {}


async def save_section_config(store, value: dict[str, dict[str, Any]]) -> None:
    await store.set_setting("section_config", json.dumps(value, ensure_ascii=False))


async def free_section_keys(store) -> list[str]:
    """Resolve free access without loading the full section/question catalog."""
    config = await section_config(store)
    keys = []
    for definition in SYSTEM_SECTIONS:
        if definition["key"] in PROTECTED_SECTION_KEYS:
            continue
        override = config.get(definition["key"], {})
        price = override.get("price", definition["price"]) if isinstance(override, dict) else definition["price"]
        if int(price or 0) == 0:
            keys.append(definition["key"])
    for key, override in config.items():
        if key.startswith("attestation:") and isinstance(override, dict) and int(override.get("price", 100) or 0) == 0:
            keys.append(key)
    return keys


def _has_access(user: dict[str, Any], section: dict[str, Any], is_admin: bool) -> bool:
    if is_admin:
        return True
    key = str(section["key"])
    if section.get("manual_grant_only") or key in PROTECTED_SECTION_KEYS:
        return key in set(user.get("section_access", []) or [])
    if int(section.get("price") or 0) == 0:
        return True
    tier = access_tier(user)
    if tier == "full":
        return True
    if key in set(user.get("section_access", []) or []):
        return True
    if tier == "trial_full" and key == "customs":
        return True
    if tier == "cases" and (key == "cases" or key.startswith("attestation:")):
        return True
    return False


async def build_sections(store, user: dict[str, Any] | None = None, *, is_admin: bool = False) -> list[dict[str, Any]]:
    config = await section_config(store)
    items: list[dict[str, Any]] = []
    for definition in SYSTEM_SECTIONS:
        row = dict(definition)
        row.update({"kind": "system", "deletable": False, "questions_count": None})
        items.append(row)

    dynamic = await store.list_attestation_banks_for_admin()
    for index, bank in enumerate(dynamic):
        if bank.get("slug") == "stage-1" or bank.get("source_id") == "bundled-stage-1":
            continue
        items.append({
            "key": f"attestation:{bank['id']}",
            "kind": "attestation",
            "bank_id": int(bank["id"]),
            "bank_slug": bank["slug"],
            "title": bank["title"],
            "screen": "attestation-bank",
            "icon": "document",
            "price": 100,
            "preview_count": min(50, int(bank.get("questions_count") or 0)),
            "visible": bank.get("status") == "published",
            "questions_count": int(bank.get("questions_count") or 0),
            "deletable": True,
            "group": "primary",
            "content_screen": "admin-section-questions",
            "content_label": "Питання",
            "default_order": index,
        })

    result = []
    for item in items:
        override = config.get(item["key"], {})
        if isinstance(override, dict):
            for field in ("title", "visible", "price", "order"):
                if field in override:
                    item[field] = override[field]
        item.setdefault("visible", True)
        item["order"] = int(item.get("order", item.pop("default_order", 0)) or 0)
        item["price"] = max(0, int(item.get("price") or 0))
        item["has_access"] = _has_access(user or {}, item, is_admin)
        result.append(item)
    return sorted(result, key=lambda row: (row["order"], row["key"]))


async def get_section(store, key: str, user: dict[str, Any] | None = None, *, is_admin: bool = False) -> dict[str, Any] | None:
    return next((item for item in await build_sections(store, user, is_admin=is_admin) if item["key"] == key), None)


async def update_section(store, key: str, changes: dict[str, Any]) -> dict[str, Any] | None:
    current = await get_section(store, key, is_admin=True)
    if not current:
        return None
    config = await section_config(store)
    row = dict(config.get(key, {}) or {})
    row.update(changes)
    config[key] = row
    await save_section_config(store, config)
    return await get_section(store, key, is_admin=True)


async def move_section(store, key: str, direction: str) -> bool:
    items = await build_sections(store, is_admin=True)
    current = next((item for item in items if item["key"] == key), None)
    if not current:
        return False
    items = [item for item in items if item["group"] == current["group"]]
    index = next((i for i, item in enumerate(items) if item["key"] == key), -1)
    target = index - 1 if direction == "up" else index + 1
    if index < 0 or target < 0 or target >= len(items):
        return True
    items[index], items[target] = items[target], items[index]
    config = await section_config(store)
    group_base = min(int(item["order"]) for item in items)
    for offset, item in enumerate(items):
        config.setdefault(item["key"], {})["order"] = group_base + offset
    await save_section_config(store, config)
    return True


async def reorder_section_group(store, group: str, ordered_keys: list[str]) -> bool:
    """Persist an exact drag-and-drop order for one home-screen group."""
    items = [item for item in await build_sections(store, is_admin=True) if item["group"] == group]
    current_keys = [str(item["key"]) for item in items]
    clean_keys = [str(key) for key in ordered_keys]
    if not items or len(clean_keys) != len(set(clean_keys)) or set(clean_keys) != set(current_keys):
        return False
    by_key = {str(item["key"]): item for item in items}
    group_base = min(int(item["order"]) for item in items)
    config = await section_config(store)
    for offset, key in enumerate(clean_keys):
        config.setdefault(key, {})["order"] = group_base + offset
    await save_section_config(store, config)
    return all(key in by_key for key in clean_keys)
