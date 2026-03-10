"""Leaderboard endpoint."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.database import get_db
from server.models import Bot
from server.schemas import LeaderboardEntry, PaginatedResponse

router = APIRouter(prefix="/api", tags=["leaderboard"])

SORT_COLUMNS = {
    "return": Bot.verified_return,
    "sharpe": Bot.verified_sharpe,
    "drawdown": Bot.verified_drawdown,
    "trades": Bot.verified_trades,
}


@router.get("/leaderboard", response_model=PaginatedResponse)
async def leaderboard(
    sort_by: str = Query("return", description="return | sharpe | drawdown | trades"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    col = SORT_COLUMNS.get(sort_by, Bot.verified_return)

    base = select(Bot).where(Bot.status == "verified")

    if sort_by == "drawdown":
        query = base.order_by(col.asc())
    else:
        query = base.order_by(col.desc().nulls_last())

    from sqlalchemy import func
    total = (await db.execute(select(func.count(Bot.id)).where(Bot.status == "verified"))).scalar() or 0

    offset = (page - 1) * page_size
    rows = (await db.execute(query.offset(offset).limit(page_size))).scalars().all()

    items = [
        LeaderboardEntry(
            rank=offset + i + 1,
            id=str(b.id),
            name=b.name,
            personality=b.personality,
            verified_return=b.verified_return or 0,
            verified_sharpe=b.verified_sharpe or 0,
            verified_drawdown=b.verified_drawdown or 0,
            verified_trades=b.verified_trades or 0,
            verified_blowup_count=b.verified_blowup_count or 0,
        )
        for i, b in enumerate(rows)
    ]

    return PaginatedResponse(total=total, page=page, page_size=page_size, items=items)
