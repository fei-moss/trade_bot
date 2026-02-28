"""
回测引擎

接收 BotConfig(42维权重向量) + 历史数据，模拟交易并输出绩效指标。
支持:
- 6种止损类型 (None/Fixed/ATR/Trailing/RegimeAdaptive/MaxDD)
- 5种止盈类型 (RR2/RR3/RR5/Trailing/None)
- 6种仓位管理 (Fixed/Kelly/Martingale/Anti-Martingale/AllIn/Volatility)
- 动态杠杆 (Fixed/RegimeScale/VolatilityScale)
- 极端化开关 (YOLO/反向逻辑/允许爆仓/regime覆盖)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.strategy.schema import BotConfig, WeightVector
from src.strategy.signals import generate_signals
from src.strategy.indicators import atr as compute_atr


@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: int              # 1=long, -1=short
    margin: float               # 保证金
    leverage: int
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trailing_high: Optional[float] = None
    trailing_low: Optional[float] = None


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    regime_performance: dict = field(default_factory=dict)
    blowup_count: int = 0
    total_deposited: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "blowup_count": self.blowup_count,
            "total_deposited": round(self.total_deposited, 2),
            "regime_performance": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.regime_performance.items()
            },
        }


def _unrealized_pnl_pct(pos: Trade, current_price: float) -> float:
    if pos.direction == 1:
        return (current_price - pos.entry_price) / pos.entry_price * pos.leverage
    else:
        return (pos.entry_price - current_price) / pos.entry_price * pos.leverage


def _calc_margin(
    capital: float,
    w: WeightVector,
    recent_trades: list[Trade],
    volatility_rank: float,
) -> float:
    """根据仓位管理策略计算保证金"""
    base = capital * w.risk_per_trade

    if w.position_sizing == "Fixed":
        margin = base

    elif w.position_sizing == "Kelly":
        if len(recent_trades) >= 10:
            wins = [t for t in recent_trades[-20:] if t.pnl > 0]
            losses = [t for t in recent_trades[-20:] if t.pnl <= 0]
            if wins and losses:
                win_rate = len(wins) / (len(wins) + len(losses))
                avg_win = np.mean([t.pnl_pct for t in wins])
                avg_loss = abs(np.mean([t.pnl_pct for t in losses]))
                if avg_loss > 0:
                    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                    kelly = max(0.0, min(kelly, 0.25))
                    margin = capital * kelly
                else:
                    margin = base
            else:
                margin = base
        else:
            margin = base

    elif w.position_sizing == "Martingale":
        if recent_trades and recent_trades[-1].pnl < 0:
            margin = base * 2
        else:
            margin = base

    elif w.position_sizing == "Anti-Martingale":
        if recent_trades and recent_trades[-1].pnl > 0:
            margin = base * 2
        else:
            margin = base

    elif w.position_sizing == "AllIn":
        margin = capital * w.max_position_per_trade

    elif w.position_sizing == "Volatility":
        if volatility_rank > 0:
            margin = base / max(volatility_rank, 0.1)
        else:
            margin = base

    else:
        margin = base

    yolo_mult = {"Off": 1.0, "Low": 1.5, "Medium": 2.5, "High": 5.0}
    margin *= yolo_mult.get(w.yolo_mode, 1.0)

    return min(margin, capital * w.max_position_per_trade, capital)


def run_backtest(
    df: pd.DataFrame,
    bot: BotConfig,
    regime: pd.Series,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """
    运行单个Bot的回测。

    Args:
        df: OHLCV DataFrame
        bot: Bot配置（含42维权重）
        regime: 行情regime序列
        initial_capital: 初始资金

    Returns:
        BacktestResult
    """
    df = df.copy().reset_index(drop=True)
    regime = regime.reset_index(drop=True)
    w = bot.weights

    signals = generate_signals(df, w, regime)

    atr_series = compute_atr(df, 14)
    atr_pct_series = atr_series / df["close"]
    vol_rank_series = atr_pct_series.rolling(200, min_periods=50).rank(pct=True).fillna(0.5)

    capital = initial_capital
    equity = [initial_capital]
    trades: list[Trade] = []
    open_pos: Optional[Trade] = None
    prev_regime = regime.iloc[0]

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        curr_regime = regime.iloc[i]
        atr_pct = atr_pct_series.iloc[i] if not np.isnan(atr_pct_series.iloc[i]) else 0.02
        vol_rank = vol_rank_series.iloc[i] if not np.isnan(vol_rank_series.iloc[i]) else 0.5

        # ---- 1. 处理持仓退出 ----
        if open_pos is not None:
            exit_price = None
            exit_reason = ""

            # 爆仓检查（用high/low判断，做多看low，做空看high）
            liq_price = low if open_pos.direction == 1 else high
            unreal_pct = _unrealized_pnl_pct(open_pos, liq_price)
            if unreal_pct <= -1.0:
                liq_threshold = open_pos.entry_price * (1 - 1.0 / open_pos.leverage) if open_pos.direction == 1 \
                    else open_pos.entry_price * (1 + 1.0 / open_pos.leverage)
                exit_price = liq_threshold
                exit_reason = "liquidation"

            # 止损
            if exit_price is None and open_pos.sl_price is not None:
                if open_pos.direction == 1 and low <= open_pos.sl_price:
                    exit_price = open_pos.sl_price
                    exit_reason = "stop_loss"
                elif open_pos.direction == -1 and high >= open_pos.sl_price:
                    exit_price = open_pos.sl_price
                    exit_reason = "stop_loss"

            # 止盈
            if exit_price is None and open_pos.tp_price is not None:
                if open_pos.direction == 1 and high >= open_pos.tp_price:
                    exit_price = open_pos.tp_price
                    exit_reason = "take_profit"
                elif open_pos.direction == -1 and low <= open_pos.tp_price:
                    exit_price = open_pos.tp_price
                    exit_reason = "take_profit"

            # 移动止损/移动止盈
            if exit_price is None and w.sl_type == "Trailing":
                trail_pct = max(atr_pct * 1.5, 0.01)
                if open_pos.direction == 1:
                    if open_pos.trailing_high is None or high > open_pos.trailing_high:
                        open_pos.trailing_high = high
                    trail_sl = open_pos.trailing_high * (1 - trail_pct)
                    if low <= trail_sl and open_pos.trailing_high > open_pos.entry_price * 1.005:
                        exit_price = trail_sl
                        exit_reason = "trailing_stop"
                else:
                    if open_pos.trailing_low is None or low < open_pos.trailing_low:
                        open_pos.trailing_low = low
                    trail_sl = open_pos.trailing_low * (1 + trail_pct)
                    if high >= trail_sl and open_pos.trailing_low < open_pos.entry_price * 0.995:
                        exit_price = trail_sl
                        exit_reason = "trailing_stop"

            if exit_price is None and w.tp_type == "Trailing":
                trail_pct = max(atr_pct * 2, 0.015)
                if open_pos.direction == 1:
                    if open_pos.trailing_high is None or high > open_pos.trailing_high:
                        open_pos.trailing_high = high
                    trail_tp = open_pos.trailing_high * (1 - trail_pct)
                    if low <= trail_tp and trail_tp > open_pos.entry_price:
                        exit_price = trail_tp
                        exit_reason = "trailing_tp"
                else:
                    if open_pos.trailing_low is None or low < open_pos.trailing_low:
                        open_pos.trailing_low = low
                    trail_tp = open_pos.trailing_low * (1 + trail_pct)
                    if high >= trail_tp and trail_tp < open_pos.entry_price:
                        exit_price = trail_tp
                        exit_reason = "trailing_tp"

            # regime变化退出
            if exit_price is None and w.exit_on_regime_change != "No":
                if curr_regime != prev_regime:
                    if w.exit_on_regime_change == "Yes":
                        exit_price = price
                        exit_reason = "regime_change"
                    elif w.exit_on_regime_change == "OnlyIfLoss":
                        if _unrealized_pnl_pct(open_pos, price) < 0:
                            exit_price = price
                            exit_reason = "regime_change_loss"

            # 最大回撤止损 (MaxDD类型)
            if exit_price is None and w.sl_type == "MaxDD":
                peak_eq = max(equity) if equity else initial_capital
                curr_eq = capital + open_pos.margin + open_pos.margin * _unrealized_pnl_pct(open_pos, price)
                if peak_eq > 0:
                    dd = (peak_eq - curr_eq) / peak_eq
                    if dd >= w.max_dd_tolerance:
                        exit_price = price
                        exit_reason = "max_drawdown"

            # 平仓
            if exit_price is not None:
                pnl_pct = _unrealized_pnl_pct(open_pos, exit_price)
                pnl_pct = max(pnl_pct, -1.0)
                pnl = open_pos.margin * pnl_pct
                open_pos.exit_idx = i
                open_pos.exit_price = exit_price
                open_pos.pnl = pnl
                open_pos.pnl_pct = pnl_pct
                open_pos.exit_reason = exit_reason
                capital += open_pos.margin + pnl
                trades.append(open_pos)
                open_pos = None

                if w.allow_blowup == "No" and capital <= 0:
                    break

        # ---- 2. 开新仓 ----
        if open_pos is None and capital > 0:
            sig = signals.iloc[i]
            if sig != 0:
                direction = sig
                leverage = w.get_effective_leverage(curr_regime, vol_rank)
                sl_pct = w.get_sl_pct(atr_pct, curr_regime)
                tp_pct = w.get_tp_pct(sl_pct)

                margin = _calc_margin(capital, w, trades, vol_rank)
                if margin <= 0 or margin > capital:
                    margin = min(capital * w.risk_per_trade, capital)

                if direction == 1:
                    sl_price = price * (1 - sl_pct) if sl_pct < 0.99 else None
                    tp_price = price * (1 + tp_pct) if tp_pct else None
                else:
                    sl_price = price * (1 + sl_pct) if sl_pct < 0.99 else None
                    tp_price = price * (1 - tp_pct) if tp_pct else None

                open_pos = Trade(
                    entry_idx=i,
                    entry_price=price,
                    direction=direction,
                    margin=margin,
                    leverage=leverage,
                    sl_price=sl_price,
                    tp_price=tp_price,
                )
                capital -= margin

        # 记录权益
        if open_pos is not None:
            unrealized = open_pos.margin * _unrealized_pnl_pct(open_pos, price)
            total_eq = capital + open_pos.margin + unrealized
        else:
            total_eq = capital
        equity.append(max(total_eq, 0))

        prev_regime = curr_regime

    # 强制平仓
    if open_pos is not None:
        price = df["close"].iloc[-1]
        pnl_pct = _unrealized_pnl_pct(open_pos, price)
        pnl_pct = max(pnl_pct, -1.0)
        pnl = open_pos.margin * pnl_pct
        open_pos.exit_idx = len(df) - 1
        open_pos.exit_price = price
        open_pos.pnl = pnl
        open_pos.pnl_pct = pnl_pct
        open_pos.exit_reason = "end_of_data"
        capital += open_pos.margin + pnl
        trades.append(open_pos)
    equity.append(capital)

    return _calc_metrics(trades, equity, initial_capital, regime)


def _calc_metrics(
    trades: list[Trade],
    equity: list[float],
    initial_capital: float,
    regime: pd.Series,
) -> BacktestResult:
    """计算回测绩效指标"""
    eq = pd.Series(equity)
    result = BacktestResult()
    result.trades = trades
    result.equity_curve = eq
    result.total_trades = len(trades)
    result.total_return = (eq.iloc[-1] - initial_capital) / initial_capital

    if trades:
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = len(wins) / len(pnls) if pnls else 0
        result.avg_trade_pnl = float(np.mean(pnls))
        result.avg_win = float(np.mean(wins)) if wins else 0
        result.avg_loss = float(np.mean(losses)) if losses else 0

        total_win = sum(wins)
        total_loss = abs(sum(losses))
        result.profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for p in pnls:
            if p > 0:
                streak = streak + 1 if streak > 0 else 1
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = streak - 1 if streak < 0 else -1
                max_loss_streak = max(max_loss_streak, abs(streak))
        result.max_consecutive_wins = max_win_streak
        result.max_consecutive_losses = max_loss_streak

    returns = eq.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        result.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252 * 24))
    else:
        result.sharpe_ratio = 0

    peak = eq.expanding().max()
    drawdown = (eq - peak) / peak
    result.max_drawdown = float(abs(drawdown.min()))

    for regime_type in regime.unique():
        mask = regime == regime_type
        regime_indices = set(mask[mask].index.tolist())
        regime_trades = [t for t in trades if t.entry_idx in regime_indices]
        if regime_trades:
            rpnls = [t.pnl_pct for t in regime_trades]
            rwins = [p for p in rpnls if p > 0]
            result.regime_performance[regime_type] = {
                "trades": len(regime_trades),
                "return": float(sum(rpnls)),
                "win_rate": len(rwins) / len(rpnls),
                "avg_pnl": float(np.mean(rpnls)),
            }

    return result


def batch_backtest(
    df: pd.DataFrame,
    bots: list[BotConfig],
    regime: pd.Series,
    initial_capital: float = 10000.0,
) -> dict[str, BacktestResult]:
    """批量回测多个Bot。"""
    results = {}
    for i, bot in enumerate(bots):
        try:
            result = run_backtest(df, bot, regime, initial_capital)
            results[bot.bot_id] = result
            print(
                f"[{i + 1}/{len(bots)}] {bot.bot_id}: "
                f"return={result.total_return:.2%}, "
                f"trades={result.total_trades}, "
                f"sharpe={result.sharpe_ratio:.2f}"
            )
        except Exception as e:
            print(f"[{i + 1}/{len(bots)}] {bot.bot_id}: FAILED - {e}")
    return results


def multi_period_backtest(
    bots: list[BotConfig],
    eval_datasets: dict,
    current_df: pd.DataFrame = None,
    current_regime: pd.Series = None,
    initial_capital: float = 10000.0,
    cache: dict = None,
) -> dict[str, dict[str, dict]]:
    """
    在多段历史行情上批量回测Bot。

    Returns:
        {bot_id: {period_name: result_dict, ...}, ...}
    """
    if cache is None:
        cache = {}

    all_results = {}
    periods = dict(eval_datasets)
    if current_df is not None and current_regime is not None:
        periods["current"] = {"df": current_df, "regime": current_regime, "label": "当前"}

    for bot in bots:
        fp = bot.weights.fingerprint()
        bot_results = {}
        for pname, pdata in periods.items():
            cache_key = (fp, pname)
            if pname != "current" and cache_key in cache:
                bot_results[pname] = cache[cache_key]
                continue
            try:
                r = run_backtest(pdata["df"], bot, pdata["regime"], initial_capital)
                rd = r.to_dict()
                bot_results[pname] = rd
                if pname != "current":
                    cache[cache_key] = rd
            except Exception:
                bot_results[pname] = {"total_return": 0, "sharpe_ratio": 0, "total_trades": 0}

        all_results[bot.bot_id] = bot_results

    return all_results
