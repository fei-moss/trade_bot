"""Core verification service: fetch data -> fingerprint check -> replay backtest -> compare."""

import hashlib
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from server.config import settings
from server.schemas import (
    UploadPackage, VerifyResponse, SegmentResult,
)
from server.services.backtest_runner import (
    fetch_data, get_regime, run_backtest, run_backtest_segments,
)

logger = logging.getLogger(__name__)


def _compute_checksum(close_prices: pd.Series) -> str:
    raw = ",".join(f"{v:.2f}" for v in close_prices)
    return hashlib.sha256(raw.encode()).hexdigest()


def _fingerprint_match(df: pd.DataFrame, fp: dict) -> tuple[bool, str]:
    """Validate data fingerprint against fetched DataFrame."""
    errors = []

    if abs(len(df) - fp["bars"]) > len(df) * 0.02:
        errors.append(f"bars: expected {fp['bars']}, got {len(df)}")

    first_close = df["close"].iloc[0]
    last_close = df["close"].iloc[-1]
    if abs(first_close - fp["first_close"]) / first_close > 0.001:
        errors.append(f"first_close: expected {fp['first_close']}, got {first_close:.2f}")
    if abs(last_close - fp["last_close"]) / last_close > 0.001:
        errors.append(f"last_close: expected {fp['last_close']}, got {last_close:.2f}")

    if errors:
        return False, "; ".join(errors)

    if fp.get("checksum") and fp["checksum"]:
        actual_checksum = _compute_checksum(df["close"])
        if actual_checksum != fp["checksum"]:
            logger.warning("Checksum mismatch (soft check) — data fetched at different times may differ slightly")

    return True, ""


def _close_enough(
    actual: dict,
    expected: dict,
    tolerance: float,
    fields: list[str] = None,
) -> tuple[bool, dict]:
    """Compare two result dicts within tolerance. Returns (match, mismatch_details)."""
    if fields is None:
        fields = ["total_return", "total_trades", "win_rate"]

    mismatches = {}
    for f in fields:
        a = actual.get(f, 0)
        e = expected.get(f, 0)
        if f == "total_trades":
            if abs(a - e) > max(2, e * tolerance):
                mismatches[f] = {"expected": e, "actual": a}
        else:
            if e == 0:
                if abs(a) > tolerance:
                    mismatches[f] = {"expected": e, "actual": a}
            elif abs(a - e) / max(abs(e), 1e-6) > tolerance:
                mismatches[f] = {"expected": e, "actual": round(a, 4)}

    return len(mismatches) == 0, mismatches


async def verify_package(package: UploadPackage) -> VerifyResponse:
    fp = package.data_fingerprint
    bot = package.bot

    try:
        start_date = fp.start[:10] if len(fp.start) > 10 else fp.start
        df = fetch_data(
            symbol=fp.symbol,
            timeframe=fp.timeframe,
            since_date=start_date,
            days=max(1, fp.bars // 96 + 5),
            exchange_id=fp.exchange or settings.DEFAULT_EXCHANGE,
        )
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return VerifyResponse(
            bot_id="", status="rejected", match=False,
            mismatch_details={"error": f"无法获取行情数据: {e}"},
        )

    if "timestamp" in df.columns:
        start_ts = pd.Timestamp(fp.start)
        end_ts = pd.Timestamp(fp.end)
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)

    fp_ok, fp_err = _fingerprint_match(df, fp.model_dump())
    if not fp_ok:
        return VerifyResponse(
            bot_id="", status="rejected", match=False,
            mismatch_details={"fingerprint": fp_err},
        )

    regime = get_regime(df, version=settings.REGIME_VERSION, min_duration=settings.REGIME_MIN_DURATION)

    all_mismatches = {}

    if package.evolution_log:
        evo_dicts = [e.model_dump() for e in package.evolution_log]
        seg_results = run_backtest_segments(
            df, regime,
            initial_params=bot.params,
            evolution_log=evo_dicts,
        )

        for i, (actual, submitted) in enumerate(zip(seg_results, package.evolution_log)):
            submitted_dict = submitted.segment_result.model_dump()
            ok, mis = _close_enough(
                actual, submitted_dict,
                tolerance=settings.VERIFY_SEGMENT_TOLERANCE_PCT,
            )
            if not ok:
                all_mismatches[f"segment_{submitted.round}"] = mis

    final_params = bot.params
    if package.evolution_log:
        final_params = package.evolution_log[-1].params_after

    full_result = run_backtest(df, final_params, regime)
    full_dict = full_result.to_dict()

    submitted_full = package.backtest_result.model_dump()
    ok, full_mis = _close_enough(
        full_dict, submitted_full,
        tolerance=settings.VERIFY_TOLERANCE_PCT,
        fields=["total_return", "sharpe_ratio", "max_drawdown", "win_rate",
                "profit_factor", "total_trades"],
    )
    if not full_mis:
        pass
    else:
        all_mismatches["full_result"] = full_mis

    match = len(all_mismatches) == 0
    status = "verified" if match else "rejected"

    return VerifyResponse(
        bot_id="",
        status=status,
        match=match,
        verified_result=full_dict,
        mismatch_details=all_mismatches if all_mismatches else None,
    )
