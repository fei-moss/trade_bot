"""Bot CRUD endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from server.database import get_db
from server.models import Bot, TradeRecord
from server.schemas import BotSummary, BotDetail, PaginatedResponse, TradeSubmit

router = APIRouter(prefix="/api/bots", tags=["bots"])


@router.get("", response_model=PaginatedResponse)
async def list_bots(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    name: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Bot).order_by(Bot.created_at.desc())
    count_query = select(func.count(Bot.id))

    if status:
        query = query.where(Bot.status == status)
        count_query = count_query.where(Bot.status == status)
    if name:
        query = query.where(Bot.name.ilike(f"%{name}%"))
        count_query = count_query.where(Bot.name.ilike(f"%{name}%"))

    total = (await db.execute(count_query)).scalar() or 0
    query = query.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(query)).scalars().all()

    items = [
        BotSummary(
            id=str(b.id), name=b.name, personality=b.personality,
            status=b.status,
            verified_return=b.verified_return,
            verified_sharpe=b.verified_sharpe,
            verified_drawdown=b.verified_drawdown,
            verified_trades=b.verified_trades,
            created_at=b.created_at,
        )
        for b in rows
    ]

    return PaginatedResponse(total=total, page=page, page_size=page_size, items=items)


@router.get("/{bot_id}", response_model=BotDetail)
async def get_bot(bot_id: UUID, db: AsyncSession = Depends(get_db)):
    bot = await db.get(Bot, bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    trade_rows = (
        await db.execute(
            select(TradeRecord).where(TradeRecord.bot_id == bot_id).order_by(TradeRecord.entry_time)
        )
    ).scalars().all()

    trades = [
        TradeSubmit(
            entry_time=str(t.entry_time),
            exit_time=str(t.exit_time) if t.exit_time else None,
            direction=t.direction,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            pnl_pct=t.pnl_pct,
            leverage=t.leverage,
            margin=t.margin,
            exit_reason=t.exit_reason,
        )
        for t in trade_rows
    ]

    return BotDetail(
        id=str(bot.id), name=bot.name, personality=bot.personality,
        status=bot.status,
        params=bot.params,
        evolution_config=bot.evolution_config,
        data_fingerprint=bot.data_fingerprint,
        verified_return=bot.verified_return,
        verified_sharpe=bot.verified_sharpe,
        verified_drawdown=bot.verified_drawdown,
        verified_trades=bot.verified_trades,
        verified_blowup_count=bot.verified_blowup_count,
        created_at=bot.created_at,
        trades=trades,
    )


@router.delete("/{bot_id}", status_code=204)
async def delete_bot(bot_id: UUID, db: AsyncSession = Depends(get_db)):
    bot = await db.get(Bot, bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    await db.delete(bot)
    await db.commit()
