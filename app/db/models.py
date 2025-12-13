from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    subscription: Mapped["Subscription"] = relationship(back_populates="user", uselist=False)
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="user")


class Subscription(Base):
    __tablename__ = "subscriptions"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    trial_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    paid_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    lifetime: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="subscription")


class Block(Base):
    __tablename__ = "blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(160), unique=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    questions: Mapped[list["Question"]] = relationship(back_populates="block")


class Question(Base):
    __tablename__ = "questions"

    # ВАЖЛИВО: ми використовуємо id з вашого JSON як primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    block_id: Mapped[int] = mapped_column(ForeignKey("blocks.id"), index=True)
    topic: Mapped[str | None] = mapped_column(String(220), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    q_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    text: Mapped[str] = mapped_column(Text)
    # options_json зберігаємо як JSON-рядок
    options_json: Mapped[str] = mapped_column(Text)

    # для сумісності (один правильний варіант)
    correct_index: Mapped[int] = mapped_column(Integer)
    # для multi-answer: JSON-рядок зі списком індексів (може бути "[]")
    correct_json: Mapped[str] = mapped_column(Text, default="[]")

    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    block: Mapped[Block] = relationship(back_populates="questions")


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    mode: Mapped[str] = mapped_column(String(16))  # training / exam
    block_id: Mapped[int | None] = mapped_column(ForeignKey("blocks.id"), nullable=True)
    total: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="attempts")
