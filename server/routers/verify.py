"""Upload verification endpoint."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.database import get_db
from server.models import Bot, Verification, TradeRecord
from server.schemas import UploadPackage, VerifyResponse
from server.services.verifier import verify_package

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["verify"])


@router.post("/verify", response_model=VerifyResponse)
async def verify_bot(package: UploadPackage, db: AsyncSession = Depends(get_db)):
    result = await verify_package(package)

    bot = Bot(
        name=package.bot.name,
        personality=package.bot.personality,
        params=package.bot.params,
        evolution_config=package.bot.evolution_config,
        data_fingerprint=package.data_fingerprint.model_dump(),
        status=result.status,
    )

    if result.match and result.verified_result:
        vr = result.verified_result
        bot.verified_return = vr.get("total_return")
        bot.verified_sharpe = vr.get("sharpe_ratio")
        bot.verified_drawdown = vr.get("max_drawdown")
        bot.verified_trades = vr.get("total_trades")
        bot.verified_blowup_count = vr.get("blowup_count", 0)

    db.add(bot)
    await db.flush()

    verification = Verification(
        bot_id=bot.id,
        submitted_result=package.backtest_result.model_dump(),
        verified_result=result.verified_result,
        evolution_log=[e.model_dump() for e in package.evolution_log] if package.evolution_log else None,
        match=result.match,
        tolerance_pct=0.01,
        mismatch_details=result.mismatch_details,
    )
    db.add(verification)

    if result.match:
        for t in package.trades:
            trade = TradeRecord(
                bot_id=bot.id,
                entry_time=datetime.fromisoformat(t.entry_time),
                exit_time=datetime.fromisoformat(t.exit_time) if t.exit_time else None,
                direction=t.direction,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                pnl_pct=t.pnl_pct,
                leverage=t.leverage,
                margin=t.margin,
                exit_reason=t.exit_reason,
            )
            db.add(trade)

    await db.commit()
    await db.refresh(bot)

    result.bot_id = str(bot.id)
    return result
