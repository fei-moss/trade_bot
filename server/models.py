import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String, Text, Float, Integer, Boolean, DateTime, Enum as SAEnum,
    ForeignKey, Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    personality: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    evolution_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    data_fingerprint: Mapped[dict] = mapped_column(JSONB, nullable=False)

    status: Mapped[str] = mapped_column(
        SAEnum("pending", "verified", "rejected", name="bot_status"),
        default="pending",
    )

    verified_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verified_blowup_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    verifications: Mapped[list["Verification"]] = relationship(back_populates="bot", cascade="all, delete-orphan")
    trades: Mapped[list["TradeRecord"]] = relationship(back_populates="bot", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_bots_status", "status"),
        Index("ix_bots_verified_return", "verified_return"),
    )


class Verification(Base):
    __tablename__ = "verifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))

    submitted_result: Mapped[dict] = mapped_column(JSONB, nullable=False)
    verified_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evolution_log: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    match: Mapped[bool] = mapped_column(Boolean, default=False)
    tolerance_pct: Mapped[float] = mapped_column(Float, default=0.01)
    mismatch_details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    bot: Mapped["Bot"] = relationship(back_populates="verifications")


class TradeRecord(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("bots.id", ondelete="CASCADE"))

    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
    margin: Mapped[float] = mapped_column(Float, default=0.0)
    exit_reason: Mapped[str] = mapped_column(String(32), default="")

    bot: Mapped["Bot"] = relationship(back_populates="trades")

    __table_args__ = (
        Index("ix_trades_bot_id", "bot_id"),
    )
