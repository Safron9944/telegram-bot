from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, Subscription, Block, Question


async def get_or_create_user(session: AsyncSession, tg_id: int, full_name: str | None) -> User:
    res = await session.execute(select(User).where(User.tg_id == tg_id))
    user = res.scalar_one_or_none()
    if user:
        # оновимо ім'я якщо ще нема
        if full_name and not user.full_name:
            user.full_name = full_name
            await session.commit()
        return user

    user = User(tg_id=tg_id, full_name=full_name)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def ensure_trial_subscription(session: AsyncSession, user: User, trial_days: int = 3) -> Subscription:
    res = await session.execute(select(Subscription).where(Subscription.user_id == user.id))
    sub = res.scalar_one_or_none()
    if sub:
        return sub

    sub = Subscription(
        user_id=user.id,
        trial_end=datetime.utcnow() + timedelta(days=trial_days),
        paid_until=None,
        lifetime=False,
    )
    session.add(sub)
    await session.commit()
    return sub


def is_subscription_active(sub: Subscription, now: datetime | None = None) -> bool:
    now = now or datetime.utcnow()
    if sub.lifetime:
        return True
    if now < sub.trial_end:
        return True
    if sub.paid_until and now < sub.paid_until:
        return True
    return False


async def set_phone(session: AsyncSession, user: User, phone: str) -> None:
    user.phone = phone
    await session.commit()


def _data_path(filename: str) -> Path:
    # app/db/repo.py -> app/db -> app -> repo root
    return Path(__file__).resolve().parents[2] / "data" / filename


async def seed_from_json(session: AsyncSession, json_path: Path | None = None) -> str:
    """Імпорт блоків/питань з questions_flat.json.

    - id з JSON використовується як primary key question.id
    - некоректні питання (нема варіантів або correct) пропускаємо
    - multi-answer зберігаємо в correct_json, а correct_index ставимо перший варіант
    """

    json_path = json_path or _data_path("questions_flat.json")
    if not json_path.exists():
        return f"Не знайдено файл з питаннями: {json_path}"

    items = json.loads(json_path.read_text(encoding="utf-8"))

    # existing blocks
    res = await session.execute(select(Block))
    existing_blocks = {b.title: b for b in res.scalars().all()}

    # create missing blocks (зберігаємо порядок першої появи)
    order = 0
    for it in items:
        title = (it.get("block") or "").strip() or "Без блоку"
        if title not in existing_blocks:
            order += 1
            b = Block(title=title, sort_order=order)
            session.add(b)
            existing_blocks[title] = b

    await session.commit()

    # refresh blocks to get ids
    res = await session.execute(select(Block))
    blocks = {b.title: b for b in res.scalars().all()}

    # existing question ids
    res = await session.execute(select(Question.id))
    existing_qids = set(res.scalars().all())

    inserted = 0
    skipped_existing = 0
    skipped_invalid = 0
    multi_count = 0

    new_questions: list[Question] = []

    for it in items:
        qid = int(it.get("id"))
        if qid in existing_qids:
            skipped_existing += 1
            continue

        choices = it.get("choices") or []
        correct = it.get("correct") or []

        # базова валідація
        if not isinstance(choices, list) or len(choices) < 2:
            skipped_invalid += 1
            continue
        if not isinstance(correct, list) or len(correct) < 1:
            skipped_invalid += 1
            continue
        if any((not isinstance(ci, int)) or ci < 0 or ci >= len(choices) for ci in correct):
            skipped_invalid += 1
            continue

        block_title = (it.get("block") or "").strip() or "Без блоку"
        block = blocks[block_title]

        correct_index = correct[0]
        if len(correct) > 1:
            multi_count += 1

        q = Question(
            id=qid,
            block_id=block.id,
            topic=(it.get("topic") or None),
            page_start=(it.get("page_start") if isinstance(it.get("page_start"), int) else None),
            q_number=(it.get("number") if isinstance(it.get("number"), int) else None),
            text=(it.get("question") or "").strip(),
            options_json=json.dumps(choices, ensure_ascii=False),
            correct_index=int(correct_index),
            correct_json=json.dumps(correct, ensure_ascii=False),
            explanation=None,
        )
        new_questions.append(q)

    if new_questions:
        session.add_all(new_questions)
        await session.commit()
        inserted = len(new_questions)

    return (
        f"Імпорт завершено ✅\n"
        f"Блоків у БД: {len(blocks)}\n"
        f"Додано питань: {inserted}\n"
        f"Multi-answer питань: {multi_count}\n"
        f"Пропущено (вже були): {skipped_existing}\n"
        f"Пропущено (некоректні): {skipped_invalid}"
    )


# зворотна сумісність: /seed викликає seed_from_json
async def seed_demo(session: AsyncSession) -> str:
    return await seed_from_json(session)
