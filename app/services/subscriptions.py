from __future__ import annotations

from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Subscription
from app.db.repo import is_subscription_active


async def grant_days(session: AsyncSession, user: User, days: int) -> Subscription:
    res = await session.execute(select(Subscription).where(Subscription.user_id == user.id))
    sub = res.scalar_one()
    now = datetime.utcnow()
    base = sub.paid_until if sub.paid_until and sub.paid_until > now else now
    sub.paid_until = base + timedelta(days=days)
    await session.commit()
    return sub


async def grant_lifetime(session: AsyncSession, user: User) -> Subscription:
    res = await session.execute(select(Subscription).where(Subscription.user_id == user.id))
    sub = res.scalar_one()
    sub.lifetime = True
    sub.paid_until = None
    await session.commit()
    return sub


async def revoke(session: AsyncSession, user: User) -> Subscription:
    res = await session.execute(select(Subscription).where(Subscription.user_id == user.id))
    sub = res.scalar_one()
    sub.paid_until = None
    sub.lifetime = False
    await session.commit()
    return sub


def format_status(sub: Subscription) -> str:
    now = datetime.utcnow()
    if sub.lifetime:
        return "✅ Активна (безстроково)"
    if now < sub.trial_end:
        return f"🟡 Trial активний до: {sub.trial_end:%Y-%m-%d %H:%M} UTC"
    if sub.paid_until and now < sub.paid_until:
        return f"✅ Оплачено до: {sub.paid_until:%Y-%m-%d %H:%M} UTC"
    return "❌ Немає активної підписки"
