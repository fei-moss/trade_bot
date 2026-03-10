"""
End-to-end test: build an upload package from Livermore data, run verification.

Usage:
    cd trade_bot && python -m server.test_verify_e2e

No database needed — tests the verifier service directly.
"""

import sys
import os
import json
import asyncio
import hashlib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from server.schemas import (
    UploadPackage, BotCreate, DataFingerprint, BacktestResultSchema,
    TradeSubmit,
)
from server.services.verifier import verify_package
from server.services.backtest_runner import fetch_data, get_regime, run_backtest


def build_package_from_scratch():
    """Build a real upload package by running the backtest ourselves."""
    print("=" * 60)
    print("  E2E Verification Test — 利弗莫尔")
    print("=" * 60)

    livermore_file = os.path.join(_ROOT, "agent_top20_evolve/top_01_利弗莫尔.json")
    with open(livermore_file) as f:
        bot_data = json.load(f)

    params = bot_data["params"]

    print("\n[1] Fetching data...")
    symbol = "BTC/USDT"
    timeframe = "15m"
    df = fetch_data(symbol, timeframe, since_date=None, days=148, exchange_id="okx")
    regime = get_regime(df, version="v1", min_duration=192)
    print(f"  {len(df)} bars | {df['close'].iloc[0]:,.0f} → {df['close'].iloc[-1]:,.0f}")

    print("\n[2] Running backtest...")
    result = run_backtest(df, params, regime)
    rd = result.to_dict()
    print(f"  Return: {rd['total_return']*100:+.1f}% | Sharpe: {rd['sharpe_ratio']:.2f} | "
          f"Trades: {rd['total_trades']} | Blowup: {rd['blowup_count']}")

    first_ts = str(df["timestamp"].iloc[0])
    last_ts = str(df["timestamp"].iloc[-1])
    checksum_raw = ",".join(f"{v:.2f}" for v in df["close"])
    checksum = hashlib.sha256(checksum_raw.encode()).hexdigest()

    trades_submit = []
    for t in result.trades[:50]:
        trades_submit.append(TradeSubmit(
            entry_time=t.entry_time or first_ts,
            exit_time=t.exit_time,
            direction="LONG" if t.direction == 1 else "SHORT",
            entry_price=round(t.entry_price, 2),
            exit_price=round(t.exit_price, 2) if t.exit_price else None,
            pnl_pct=round(t.pnl_pct, 4),
            leverage=t.leverage,
            margin=round(t.margin, 2),
            exit_reason=t.exit_reason,
        ))

    package = UploadPackage(
        version="1.0",
        bot=BotCreate(
            name="利弗莫尔",
            personality="趋势投机者，偏空，高杠杆",
            params=params,
        ),
        data_fingerprint=DataFingerprint(
            symbol=symbol,
            timeframe=timeframe,
            exchange="okx",
            start=first_ts,
            end=last_ts,
            bars=len(df),
            first_close=round(df["close"].iloc[0], 2),
            last_close=round(df["close"].iloc[-1], 2),
            checksum=checksum,
        ),
        backtest_result=BacktestResultSchema(
            total_return=round(rd["total_return"], 4),
            sharpe_ratio=round(rd["sharpe_ratio"], 4),
            max_drawdown=round(rd["max_drawdown"], 4),
            win_rate=round(rd["win_rate"], 4),
            profit_factor=round(rd["profit_factor"], 4),
            total_trades=rd["total_trades"],
            blowup_count=rd["blowup_count"],
        ),
        evolution_log=[],
        trades=trades_submit,
    )

    return package


async def run_test():
    package = build_package_from_scratch()

    print("\n[3] Running verification...")
    result = await verify_package(package)

    print(f"\n{'=' * 60}")
    print(f"  Verification Result")
    print(f"{'=' * 60}")
    print(f"  Status: {result.status}")
    print(f"  Match: {result.match}")
    if result.verified_result:
        vr = result.verified_result
        print(f"  Verified return: {vr.get('total_return', 0)*100:+.1f}%")
        print(f"  Verified sharpe: {vr.get('sharpe_ratio', 0):.2f}")
        print(f"  Verified trades: {vr.get('total_trades', 0)}")
    if result.mismatch_details:
        print(f"  Mismatches: {json.dumps(result.mismatch_details, indent=2)}")

    assert result.match, f"Verification failed: {result.mismatch_details}"
    print("\n  PASS — Verification succeeded!")

    print("\n[4] Testing tampered data (should reject)...")
    tampered = package.model_copy(deep=True)
    tampered.backtest_result.total_return = 9.99
    tampered_result = await verify_package(tampered)
    print(f"  Tampered status: {tampered_result.status}")
    print(f"  Tampered match: {tampered_result.match}")
    assert not tampered_result.match, "Tampered package should be rejected!"
    print("  PASS — Tampered package correctly rejected!")

    print(f"\n{'=' * 60}")
    print("  All tests passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(run_test())
