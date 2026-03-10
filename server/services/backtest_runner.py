"""Bridge to the shared backtest engine in src/."""

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime
from src.strategy.decision import DecisionParams, compute_signals
from src.backtest.engine import BacktestResult, Trade
from agent_backtest import run_agent_backtest


def fetch_data(
    symbol: str,
    timeframe: str,
    since_date: str,
    days: int = 0,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    return fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        exchange_id=exchange_id,
        since_date=since_date,
    )


def get_regime(df: pd.DataFrame, version: str = "v1", min_duration: int = 192) -> pd.Series:
    return classify_regime(df, version=version, min_duration=min_duration)


def run_backtest(
    df: pd.DataFrame,
    params_dict: dict,
    regime: pd.Series,
    initial_capital: float = 10000.0,
    precomputed_signals: pd.Series = None,
) -> BacktestResult:
    params = DecisionParams.from_dict(params_dict)
    return run_agent_backtest(df, params, regime, initial_capital, precomputed_signals)


def run_backtest_segments(
    df: pd.DataFrame,
    regime: pd.Series,
    initial_params: dict,
    evolution_log: list[dict],
    initial_capital: float = 10000.0,
) -> list[dict]:
    """Replay backtest segment by segment following the evolution log.

    Returns a list of segment results matching each evolution round.
    """
    timestamps = df["timestamp"] if "timestamp" in df.columns else None
    results = []

    current_params = initial_params.copy()

    for entry in evolution_log:
        t_start = pd.Timestamp(entry["time_range"][0])
        t_end = pd.Timestamp(entry["time_range"][1])

        if timestamps is not None:
            mask = (timestamps >= t_start) & (timestamps < t_end)
            seg_df = df.loc[mask].reset_index(drop=True)
            seg_regime = regime.loc[mask].reset_index(drop=True)
        else:
            seg_df = df
            seg_regime = regime

        if len(seg_df) == 0:
            results.append({"total_return": 0.0, "total_trades": 0, "win_rate": 0.0})
            current_params = entry.get("params_after", current_params)
            continue

        params = DecisionParams.from_dict(current_params)
        seg_result = run_agent_backtest(seg_df, params, seg_regime, initial_capital)

        results.append({
            "total_return": round(seg_result.total_return, 4),
            "total_trades": seg_result.total_trades,
            "win_rate": round(seg_result.win_rate, 4),
        })

        current_params = entry.get("params_after", current_params)

    return results
