from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any


def normalized_choice_order(raw_order: Any, choice_count: int) -> list[int] | None:
    """Return a valid zero-based permutation or None."""
    if not isinstance(raw_order, list) or len(raw_order) != choice_count:
        return None
    try:
        order = [int(value) for value in raw_order]
    except (TypeError, ValueError):
        return None
    return order if sorted(order) == list(range(choice_count)) else None


def ordered_choice_indices(choice_count: int, raw_order: Any = None) -> list[int]:
    return normalized_choice_order(raw_order, choice_count) or list(range(choice_count))


def ensure_choice_order(
    state: dict[str, Any],
    question_id: int,
    choice_count: int,
    *,
    shuffle: Callable[[list[int]], None] = random.shuffle,
    shuffle_choices: bool = True,
) -> tuple[list[int], bool]:
    """Create and persist one random order per question in a session state."""
    orders = dict(state.get("choice_orders", {}) or {})
    key = str(int(question_id))
    if not shuffle_choices:
        order = list(range(choice_count))
        created = orders.get(key) != order
        orders[key] = order
        state["choice_orders"] = orders
        return order, created
    existing = normalized_choice_order(orders.get(key), choice_count)
    if existing is not None:
        return existing, False

    order = list(range(choice_count))
    shuffle(order)
    if choice_count > 1 and order == list(range(choice_count)):
        order = order[1:] + order[:1]
    orders[key] = order
    state["choice_orders"] = orders
    return order, True
