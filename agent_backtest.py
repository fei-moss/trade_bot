"""
LLM 驱动的交易 Agent 回测

用法:
  OPENROUTER_API_KEY=sk-... python3 agent_backtest.py "单边趋势滚仓，高杠杆做多"
  OPENROUTER_API_KEY=sk-... python3 agent_backtest.py "保守震荡网格，低杠杆双向"

架构:
  慢脑 (LLM): 用户prompt + 行情上下文 → DecisionParams (初始 + regime变化时微调)
  快脑 (决策函数): compute_signals(df, params, regime) → 每根K线信号 → 引擎执行
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary
from src.strategy.decision import DecisionParams, compute_signals
from src.strategy.indicators import atr as compute_atr
from src.backtest.engine import Trade, BacktestResult


def _unrealized_pnl_pct(pos: Trade, current_price: float) -> float:
    if pos.direction == 1:
        return (current_price - pos.entry_price) / pos.entry_price * pos.leverage
    else:
        return (pos.entry_price - current_price) / pos.entry_price * pos.leverage


def run_agent_backtest(
    df: pd.DataFrame,
    params: DecisionParams,
    regime: pd.Series,
    initial_capital: float = 10000.0,
    precomputed_signals: pd.Series = None,
) -> BacktestResult:
    """
    用 DecisionParams 驱动的回测引擎。
    支持滚仓、移动止损、regime退出、爆仓复活。
    爆仓后自动注入新本金继续交易，总收益按累计投入计算。
    """
    df = df.copy().reset_index(drop=True)
    regime = regime.reset_index(drop=True)
    has_ts = "timestamp" in df.columns

    if precomputed_signals is not None:
        signals = precomputed_signals.reset_index(drop=True)
    else:
        signals = compute_signals(df, params, regime)

    atr_series = compute_atr(df, 14)
    atr_pct_series = (atr_series / df["close"]).fillna(0.02)

    capital = initial_capital
    total_deposited = initial_capital
    blowup_count = 0
    equity = [initial_capital]
    trades: list[Trade] = []
    positions: list[Trade] = []
    roll_count = 0
    prev_regime = regime.iloc[0]

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        curr_regime = regime.iloc[i]
        atr_pct = atr_pct_series.iloc[i] if not np.isnan(atr_pct_series.iloc[i]) else 0.02
        atr_val = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else price * 0.02

        # ---- 0. 爆仓复活：资金低于初始本金1%视为实质爆仓 ----
        if capital < initial_capital * 0.01 and not positions:
            blowup_count += 1
            capital = initial_capital
            total_deposited += initial_capital
            roll_count = 0

        # ---- 1. 检查所有持仓的退出条件 ----
        closed_positions = []
        for pos in positions:
            exit_price = None
            exit_reason = ""

            liq_check = low if pos.direction == 1 else high
            if _unrealized_pnl_pct(pos, liq_check) <= -1.0:
                liq_threshold = pos.entry_price * (1 - 1.0 / pos.leverage) if pos.direction == 1 \
                    else pos.entry_price * (1 + 1.0 / pos.leverage)
                exit_price = liq_threshold
                exit_reason = "liquidation"

            if exit_price is None and pos.sl_price is not None:
                if pos.direction == 1 and low <= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = "stop_loss"
                elif pos.direction == -1 and high >= pos.sl_price:
                    exit_price = pos.sl_price
                    exit_reason = "stop_loss"

            if exit_price is None and pos.tp_price is not None:
                if pos.direction == 1 and high >= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "take_profit"
                elif pos.direction == -1 and low <= pos.tp_price:
                    exit_price = pos.tp_price
                    exit_reason = "take_profit"

            if exit_price is None and params.trailing_enabled:
                trail_dist = params.trailing_distance_atr * atr_val
                if pos.direction == 1:
                    if pos.trailing_high is None or high > pos.trailing_high:
                        pos.trailing_high = high
                    if pos.trailing_high and pos.trailing_high > pos.entry_price * (1 + params.trailing_activation_pct):
                        trail_sl = pos.trailing_high - trail_dist
                        if low <= trail_sl:
                            exit_price = trail_sl
                            exit_reason = "trailing_stop"
                else:
                    if pos.trailing_low is None or low < pos.trailing_low:
                        pos.trailing_low = low
                    if pos.trailing_low and pos.trailing_low < pos.entry_price * (1 - params.trailing_activation_pct):
                        trail_sl = pos.trailing_low + trail_dist
                        if high >= trail_sl:
                            exit_price = trail_sl
                            exit_reason = "trailing_stop"

            if exit_price is None and params.exit_on_regime_change:
                if curr_regime != prev_regime:
                    exit_price = price
                    exit_reason = "regime_change"

            if exit_price is None:
                sig = signals.iloc[i]
                if sig != 0 and sig != pos.direction:
                    exit_price = price
                    exit_reason = "signal_reverse"

            if exit_price is not None:
                pnl_pct = _unrealized_pnl_pct(pos, exit_price)
                pnl_pct = max(pnl_pct, -1.0)
                pnl = pos.margin * pnl_pct
                pos.exit_idx = i
                pos.exit_price = exit_price
                pos.exit_time = str(df["timestamp"].iloc[i]) if has_ts else None
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct
                pos.exit_reason = exit_reason
                capital += pos.margin + pnl
                capital = max(capital, 0)
                trades.append(pos)
                closed_positions.append(pos)

        for cp in closed_positions:
            positions.remove(cp)
            if cp.exit_reason != "liquidation":
                roll_count = max(0, roll_count - 1)

        # ---- 2. 滚仓检查 ----
        if params.rolling_enabled and positions and roll_count < params.rolling_max_times:
            for pos in list(positions):
                unrealized = _unrealized_pnl_pct(pos, price)
                if unrealized >= params.rolling_trigger_pct:
                    float_profit = pos.margin * unrealized
                    new_margin = float_profit * params.rolling_reinvest_pct

                    if new_margin > 0 and capital >= new_margin:
                        leverage = min(params.base_leverage, params.max_leverage)
                        sl_dist = params.sl_atr_mult * atr_val
                        tp_dist = sl_dist * params.tp_rr_ratio

                        if pos.direction == 1:
                            sl_p = price - sl_dist
                            tp_p = price + tp_dist
                        else:
                            sl_p = price + sl_dist
                            tp_p = price - tp_dist

                        new_pos = Trade(
                            entry_idx=i,
                            entry_price=price,
                            direction=pos.direction,
                            leverage=leverage,
                            margin=new_margin,
                            sl_price=sl_p,
                            tp_price=tp_p,
                            entry_time=str(df["timestamp"].iloc[i]) if has_ts else None,
                        )
                        capital -= new_margin
                        positions.append(new_pos)
                        roll_count += 1

                        if params.rolling_move_stop:
                            pos.sl_price = pos.entry_price

        # ---- 3. 新开仓 ----
        if not positions and capital > 0:
            sig = signals.iloc[i]
            if sig != 0:
                direction = int(sig)
                leverage = min(params.base_leverage, params.max_leverage)
                margin = capital * params.risk_per_trade
                margin = min(margin, capital * params.max_position_pct, capital)

                sl_dist = params.sl_atr_mult * atr_val
                tp_dist = sl_dist * params.tp_rr_ratio

                if direction == 1:
                    sl_p = price - sl_dist
                    tp_p = price + tp_dist
                else:
                    sl_p = price + sl_dist
                    tp_p = price - tp_dist

                pos = Trade(
                    entry_idx=i,
                    entry_price=price,
                    direction=direction,
                    leverage=leverage,
                    margin=margin,
                    sl_price=sl_p,
                    tp_price=tp_p,
                    entry_time=str(df["timestamp"].iloc[i]) if has_ts else None,
                )
                capital -= margin
                positions.append(pos)
                roll_count = 0

        # 权益 = 当前资金 - 多次注入的"虚拟本金"，用于真实盈亏曲线
        total_eq = capital
        for pos in positions:
            unreal = _unrealized_pnl_pct(pos, price)
            total_eq += pos.margin * (1 + max(unreal, -1.0))
        equity.append(total_eq - (total_deposited - initial_capital))

        prev_regime = curr_regime

    for pos in positions:
        price = df["close"].iloc[-1]
        pnl_pct = _unrealized_pnl_pct(pos, price)
        pnl_pct = max(pnl_pct, -1.0)
        pos.exit_idx = len(df) - 1
        pos.exit_price = price
        pos.exit_time = str(df["timestamp"].iloc[-1]) if has_ts else None
        pos.pnl = pos.margin * pnl_pct
        pos.pnl_pct = pnl_pct
        pos.exit_reason = "end_of_data"
        capital += pos.margin + pos.pnl
        trades.append(pos)

    equity_series = pd.Series(equity)
    return _build_result(trades, equity_series, blowup_count, total_deposited, initial_capital)


def _build_result(
    trades: list[Trade],
    equity: pd.Series,
    blowup_count: int = 0,
    total_deposited: float = 0.0,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    # total_return 基于初始本金：爆仓N次 = -N*100% + 最后一条命的盈亏
    total_return = (equity.iloc[-1] - initial_capital) / initial_capital if initial_capital > 0 else 0

    peak = equity.expanding().max()
    dd = equity - peak
    safe_peak = peak.replace(0, np.nan)
    drawdown = (dd / safe_peak).min()
    drawdown = drawdown if not np.isnan(drawdown) else 0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) if trades else 0

    returns = equity.pct_change().dropna()
    valid_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (valid_returns.mean() / valid_returns.std() * np.sqrt(8760)) if len(valid_returns) > 1 and valid_returns.std() > 0 else 0

    total_win = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses))
    pf = total_win / total_loss if total_loss > 0 else float("inf")

    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0

    consec_w = consec_l = max_cw = max_cl = 0
    for t in trades:
        if t.pnl > 0:
            consec_w += 1
            consec_l = 0
            max_cw = max(max_cw, consec_w)
        else:
            consec_l += 1
            consec_w = 0
            max_cl = max(max_cl, consec_l)

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=abs(drawdown),
        win_rate=win_rate,
        profit_factor=pf,
        total_trades=len(trades),
        avg_trade_pnl=np.mean([t.pnl_pct for t in trades]) if trades else 0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_consecutive_wins=max_cw,
        max_consecutive_losses=max_cl,
        trades=trades,
        equity_curve=equity,
        regime_performance={},
        blowup_count=blowup_count,
        total_deposited=total_deposited if total_deposited > 0 else initial_capital,
    )


def _format_trade_log(trades: list[Trade], df: pd.DataFrame) -> str:
    """将交易记录格式化为 LLM 可读的复盘日志。"""
    if not trades:
        return "无交易"

    lines = []
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    total_pnl = sum(t.pnl for t in trades)

    lines.append(f"共 {len(trades)} 笔 | 盈利 {len(wins)} 笔 | 亏损 {len(losses)} 笔 | "
                 f"净盈亏 ${total_pnl:+,.0f}")
    if wins:
        lines.append(f"平均盈利: {np.mean([t.pnl_pct for t in wins])*100:+.1f}% | "
                     f"最大单笔盈利: {max(t.pnl_pct for t in wins)*100:+.1f}%")
    if losses:
        lines.append(f"平均亏损: {np.mean([t.pnl_pct for t in losses])*100:+.1f}% | "
                     f"最大单笔亏损: {min(t.pnl_pct for t in losses)*100:+.1f}%")

    liquidations = [t for t in trades if t.exit_reason == "liquidation"]
    if liquidations:
        lines.append(f"⚠ 爆仓 {len(liquidations)} 次!")

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    lines.append(f"出场原因: {exit_reasons}")

    lines.append("\n最近交易明细:")
    for t in trades[-8:]:
        dir_str = "LONG" if t.direction == 1 else "SHORT"
        price_in = f"${t.entry_price:,.0f}" if t.entry_price > 100 else f"${t.entry_price:.4f}"
        price_out = f"${t.exit_price:,.0f}" if t.exit_price and t.exit_price > 100 else "?"
        lines.append(f"  {dir_str} {t.leverage}x | {price_in}→{price_out} | "
                     f"{t.pnl_pct*100:+.1f}% | {t.exit_reason}")

    return "\n".join(lines)


def run_with_reflection(
    df: pd.DataFrame,
    params: DecisionParams,
    regime: pd.Series,
    tuner,
    user_prompt: str,
    reflection_interval: int = 24,
    initial_capital: float = 10000.0,
    verbose: bool = True,
) -> tuple[BacktestResult, list[dict]]:
    """
    带 24h 反思进化的回测。

    每 reflection_interval 根 K 线暂停一次，让 LLM 复盘调参。

    Returns:
        (final_result, evolution_log)
    """
    total_bars = len(df)
    n_segments = max(1, total_bars // reflection_interval)

    initial_params = params
    current_params = params
    all_signals = pd.Series(0, index=df.index, dtype=int)
    evolution_log = []
    cumulative_trades = []
    cumulative_pnl = 0.0
    peak_pnl = 0.0
    segment_returns = []

    btc_start_price = df["close"].iloc[0]

    for seg_idx in range(n_segments):
        seg_start = seg_idx * reflection_interval
        seg_end = min((seg_idx + 1) * reflection_interval, total_bars)
        if seg_end <= seg_start:
            break

        lookback = min(200, seg_start)
        calc_start = seg_start - lookback
        seg_df = df.iloc[calc_start:seg_end].reset_index(drop=True)
        seg_regime = regime.iloc[calc_start:seg_end].reset_index(drop=True)
        seg_sigs = compute_signals(seg_df, current_params, seg_regime)

        for j in range(lookback, len(seg_sigs)):
            global_idx = seg_start + (j - lookback)
            if global_idx < total_bars:
                all_signals.iloc[global_idx] = seg_sigs.iloc[j]

        if seg_end < total_bars and seg_idx < n_segments - 1:
            local_df = df.iloc[seg_start:seg_end].reset_index(drop=True)
            local_regime = regime.iloc[seg_start:seg_end].reset_index(drop=True)
            local_sigs = all_signals.iloc[seg_start:seg_end].reset_index(drop=True)
            local_result = run_agent_backtest(local_df, current_params, local_regime,
                                              precomputed_signals=local_sigs)
            cumulative_trades.extend(local_result.trades)

            seg_ret = local_result.total_return
            segment_returns.append(seg_ret)
            cumulative_pnl += seg_ret
            peak_pnl = max(peak_pnl, cumulative_pnl)

            # ─── 构建累计上下文 ───
            btc_now = df["close"].iloc[seg_end - 1]
            btc_change = (btc_now / btc_start_price - 1) * 100
            total_wins = sum(1 for t in cumulative_trades if t.pnl_pct > 0)
            total_count = len(cumulative_trades)
            cum_wr = total_wins / total_count * 100 if total_count else 0
            drawdown_from_peak = peak_pnl - cumulative_pnl

            cumulative_context = (
                f"进度: 第 {seg_idx+1}/{n_segments} 周期\n"
                f"累计收益: {cumulative_pnl*100:+.1f}% | 历史峰值: {peak_pnl*100:+.1f}% | "
                f"峰值回撤: {drawdown_from_peak*100:.1f}%\n"
                f"累计交易: {total_count} 笔 | 累计胜率: {cum_wr:.0f}%\n"
                f"BTC总变化: ${btc_start_price:,.0f} → ${btc_now:,.0f} ({btc_change:+.1f}%)\n"
                f"近3周期表现: {', '.join(f'{r*100:+.1f}%' for r in segment_returns[-3:])}"
            )

            # ─── 反思 ───
            trade_log = _format_trade_log(local_result.trades, local_df)
            seg_price_start = df["close"].iloc[seg_start]
            seg_price_end = df["close"].iloc[seg_end - 1]
            market_summary = (
                f"价格: ${seg_price_start:,.0f} → ${seg_price_end:,.0f} "
                f"({(seg_price_end/seg_price_start - 1)*100:+.1f}%) | "
                f"Regime: {regime.iloc[seg_end-1]}"
            )

            try:
                new_params, reflection = tuner.reflect(
                    user_prompt=user_prompt,
                    current_params=current_params,
                    trade_log=trade_log,
                    market_summary=market_summary,
                    generation=seg_idx + 1,
                    cumulative_context=cumulative_context,
                    initial_params=initial_params,
                )
                _lock_personality(new_params, initial_params)
                old_params = current_params
                current_params = new_params

                log_entry = {
                    "generation": seg_idx + 1,
                    "bar_range": [seg_start, seg_end],
                    "segment_return": local_result.total_return,
                    "trades_count": local_result.total_trades,
                    "win_rate": local_result.win_rate,
                    "reflection": reflection[:200],
                    "key_changes": _param_diff(old_params, new_params),
                }
                evolution_log.append(log_entry)

                if verbose:
                    print(f"    [反思 #{seg_idx+1}] "
                          f"前段: {local_result.total_return*100:+.1f}% "
                          f"({local_result.total_trades}笔) → "
                          f"{reflection[:60]}...")

            except Exception as e:
                if verbose:
                    print(f"    [反思 #{seg_idx+1}] 失败: {e}")

    # 用拼接后的信号做完整回测
    final = run_agent_backtest(df, current_params, regime,
                               initial_capital=initial_capital,
                               precomputed_signals=all_signals)
    return final, evolution_log


PERSONALITY_FIELDS = [
    "long_bias",
    "base_leverage", "max_leverage",
    "risk_per_trade", "max_position_pct",
    "rolling_enabled", "rolling_trigger_pct", "rolling_reinvest_pct",
    "rolling_max_times", "rolling_move_stop",
    "trend_weight", "momentum_weight", "mean_revert_weight",
    "volume_weight", "volatility_weight",
]


def _lock_personality(new_params: DecisionParams, initial_params: DecisionParams):
    """反思后强制锁定性格参数，只允许 LLM 调整战术参数。"""
    for field in PERSONALITY_FIELDS:
        setattr(new_params, field, getattr(initial_params, field))


def _param_diff(old: DecisionParams, new: DecisionParams) -> dict:
    """提取两组参数之间的关键变化。"""
    changes = {}
    for field in ["trend_weight", "momentum_weight", "mean_revert_weight",
                   "entry_threshold", "long_bias", "base_leverage",
                   "sl_atr_mult", "tp_rr_ratio", "rolling_enabled"]:
        old_val = getattr(old, field)
        new_val = getattr(new, field)
        if old_val != new_val:
            if isinstance(old_val, float):
                if abs(old_val - new_val) > 0.01:
                    changes[field] = f"{old_val:.2f} → {new_val:.2f}"
            else:
                changes[field] = f"{old_val} → {new_val}"
    return changes


def main():
    parser = argparse.ArgumentParser(description="LLM 驱动的交易 Agent 回测")
    parser.add_argument("prompt", nargs="?", default="单边趋势滚仓，高杠杆做多",
                        help="策略描述（自然语言）")
    parser.add_argument("--days", type=int, default=148, help="回测天数")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--no-retune", action="store_true",
                        help="不在regime变化时重新调参")
    parser.add_argument("--output", default="agent_result")
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM Agent 回测")
    print("=" * 60)
    print(f"  策略: {args.prompt}")
    print(f"  模型: {args.model}")

    # 加载数据
    print(f"\n[1] 加载行情数据...")
    df = fetch_ohlcv("BTC/USDT", "1h", args.days)
    regime = classify_regime(df, version="v1", min_duration=48)
    print(f"  {len(df)} 根K线 | BTC ${df['close'].iloc[0]:,.0f} → ${df['close'].iloc[-1]:,.0f}")

    summary = regime_summary(df, regime)
    for r, s in summary.items():
        print(f"  {r}: {s['pct']:.0%}")

    # LLM 生成初始参数（不传行情上下文，避免对历史数据过拟合）
    print(f"\n[2] LLM 生成策略参数...")
    from src.generator.llm_tuner import LLMTuner

    tuner = LLMTuner(model=args.model)
    params, reasoning = tuner.tune(args.prompt)
    print(f"  LLM理由: {reasoning}")
    print(f"  关键参数:")
    print(f"    趋势权重={params.trend_weight:.2f} 动量={params.momentum_weight:.2f} "
          f"回归={params.mean_revert_weight:.2f}")
    print(f"    杠杆={params.base_leverage:.0f}x 仓位={params.risk_per_trade:.0%} "
          f"方向偏好={params.long_bias:.2f}")
    print(f"    止损={params.sl_atr_mult:.1f}ATR 止盈RR={params.tp_rr_ratio:.1f} "
          f"移动止损={'开' if params.trailing_enabled else '关'}")
    print(f"    滚仓={'开' if params.rolling_enabled else '关'}"
          + (f" 触发={params.rolling_trigger_pct:.0%} "
             f"再投={params.rolling_reinvest_pct:.0%} "
             f"最多{params.rolling_max_times}次"
             if params.rolling_enabled else ""))

    # Regime 变化时重新调参
    composite_signals = None
    if not args.no_retune:
        regime_changes = []
        for i in range(1, len(regime)):
            if regime.iloc[i] != regime.iloc[i-1]:
                regime_changes.append(i)

        if regime_changes:
            print(f"\n[3] 检测到 {len(regime_changes)} 次 regime 变化，将在关键点重新调参...")

            segments = []
            prev = 0
            retune_points = regime_changes[::max(1, len(regime_changes)//5)][:5]
            for rc in retune_points:
                segments.append((prev, rc))
                prev = rc
            segments.append((prev, len(df)))

            all_params = []
            current_params = params
            for seg_idx, (start, end) in enumerate(segments):
                if seg_idx > 0:
                    seg_df = df.iloc[max(0, start-200):start]
                    seg_regime = regime.iloc[max(0, start-200):start]
                    if len(seg_df) > 50:
                        perf_str = ""
                        if all_params:
                            prev_start, prev_end, prev_params = all_params[-1]
                            prev_result = run_agent_backtest(
                                df.iloc[prev_start:prev_end],
                                prev_params,
                                regime.iloc[prev_start:prev_end],
                            )
                            perf_str = (
                                f"前一段表现: 收益={prev_result.total_return*100:+.1f}% "
                                f"交易={prev_result.total_trades}笔 "
                                f"胜率={prev_result.win_rate*100:.0f}%"
                            )

                        new_params, new_reason = tuner.tune(
                            args.prompt,
                            current_params=current_params,
                            recent_performance=perf_str,
                        )
                        current_params = new_params
                        new_regime = regime.iloc[start]
                        print(f"    Regime→{new_regime} @ bar {start}: {new_reason[:80]}")

                all_params.append((start, end, current_params))

            composite_signals = pd.Series(0, index=df.index)
            for start, end, seg_params in all_params:
                seg_df = df.iloc[max(0, start-200):end].reset_index(drop=True)
                seg_regime = regime.iloc[max(0, start-200):end].reset_index(drop=True)
                seg_sigs = compute_signals(seg_df, seg_params, seg_regime)
                offset = max(0, start - 200)
                for j in range(len(seg_sigs)):
                    global_idx = offset + j
                    if start <= global_idx < end:
                        composite_signals.iloc[global_idx] = seg_sigs.iloc[j]

            params = all_params[-1][2]
            longs = (composite_signals == 1).sum()
            shorts = (composite_signals == -1).sum()
            print(f"    分段信号汇总: 做多={longs} 做空={shorts} 总={longs+shorts}")

    # 回测
    step = '4' if composite_signals is not None else '3'
    print(f"\n[{step}] 回测中...")
    result = run_agent_backtest(df, params, regime, precomputed_signals=composite_signals)

    ret = result.total_return
    ret_cls = "+" if ret >= 0 else ""
    print(f"\n{'=' * 60}")
    print(f"  回测结果")
    print(f"{'=' * 60}")
    print(f"  策略: {args.prompt}")
    print(f"  收益: {ret_cls}{ret*100:.1f}%")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  最大回撤: {result.max_drawdown*100:.1f}%")
    print(f"  交易: {result.total_trades}笔 | 胜率: {result.win_rate*100:.0f}%")
    print(f"  盈亏比: {result.profit_factor:.2f}")
    print(f"  连胜/连亏: {result.max_consecutive_wins}/{result.max_consecutive_losses}")

    if result.trades:
        rolling_trades = sum(1 for t in result.trades if t.margin < 100)
        print(f"  滚仓笔数: ~{rolling_trades}")

    # 保存结果
    out_dir = os.path.join(ROOT_DIR, args.output)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({
            "prompt": args.prompt,
            "model": args.model,
            "params": params.to_dict(),
            "reasoning": reasoning,
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n  结果保存到: {out_dir}/")


if __name__ == "__main__":
    main()
