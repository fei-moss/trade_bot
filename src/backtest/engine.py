"""
回测引擎

接收 BotStrategy 配置 + 历史数据，模拟交易并输出绩效指标。
支持杠杆、多空、加仓、滚仓、止盈止损、移动止损等。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from src.strategy.schema import BotStrategy
from src.strategy.signals import generate_entry_signals, generate_exit_signals


@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: int          # 1=long, -1=short
    size: float             # 投入金额（不是百分比，是实际金额）
    leverage: int
    margin: float = 0.0     # 保证金
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    is_rolling: bool = False        # 是否是滚仓产生的仓位
    rolling_generation: int = 0     # 第几次滚仓（0=初始仓位）
    parent_trade_idx: int = -1      # 滚仓来源的trade在列表中的位置
    stop_loss_price: Optional[float] = None  # 动态止损价（滚仓时可能被移到成本价）


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
    rolling_trades: int = 0         # 滚仓交易次数
    regime_performance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "rolling_trades": self.rolling_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "regime_performance": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.regime_performance.items()
            },
        }


def _calc_unrealized_pnl(pos: Trade, current_price: float) -> float:
    """计算持仓浮盈（金额）"""
    if pos.direction == 1:
        price_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
    else:
        price_pnl_pct = (pos.entry_price - current_price) / pos.entry_price
    return pos.margin * pos.leverage * price_pnl_pct


def _calc_unrealized_pnl_pct(pos: Trade, current_price: float) -> float:
    """计算持仓浮盈百分比（相对保证金）"""
    if pos.direction == 1:
        return (current_price - pos.entry_price) / pos.entry_price * pos.leverage
    else:
        return (pos.entry_price - current_price) / pos.entry_price * pos.leverage


def _get_stop_loss_price(pos: Trade, exit_cfg) -> float:
    """获取止损价，优先使用动态止损价"""
    if pos.stop_loss_price is not None:
        return pos.stop_loss_price
    if pos.direction == 1:
        return pos.entry_price * (1 - exit_cfg.stop_loss_pct)
    else:
        return pos.entry_price * (1 + exit_cfg.stop_loss_pct)


def run_backtest(
    df: pd.DataFrame,
    strategy: BotStrategy,
    initial_capital: float = 10000.0,
    regime: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    运行回测。

    滚仓逻辑:
    当持仓浮盈达到 rolling_trigger_pct 时，将 rolling_reinvest_pct 比例的浮盈
    作为新仓位的保证金，以相同方向和杠杆开出新仓。同时可选将老仓止损移到成本价。

    例: 10x做多BTC, 保证金$1000
    - BTC涨30% → 浮盈 = $1000 * 10 * 30% = $3000
    - 滚仓: 取80%浮盈 = $2400 作为新仓保证金
    - 新仓: $2400保证金 * 10x = $24000名义价值，以当前价开多
    - 老仓止损移到开仓价（保本）
    - 如果继续涨，新仓也产生浮盈，可以再滚...
    - 如果回调，新仓被止损但老仓保本
    """
    df = df.copy().reset_index(drop=True)

    # 生成信号
    signals, df = generate_entry_signals(df, strategy)
    indicator_data = {}
    exit_signals = generate_exit_signals(df, strategy, indicator_data)

    # 回测状态
    capital = initial_capital
    equity = [initial_capital]
    trades = []
    open_positions: list[Trade] = []
    daily_loss = 0.0
    last_trade_idx = -999
    cool_down = strategy.risk.cool_down_bars

    pos_cfg = strategy.position
    exit_cfg = strategy.exit_rule
    risk_cfg = strategy.risk
    rolling_enabled = pos_cfg.rolling
    rolling_count = {}  # trade index -> 已滚仓次数

    for i in range(1, len(df)):
        current_price = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        # ---- 1. 检查爆仓 ----
        # 滚仓策略下如果整体亏损超过保证金，全部爆仓
        liquidated = []
        for pos in open_positions:
            unrealized_pct = _calc_unrealized_pnl_pct(pos, current_price)
            if unrealized_pct <= -1.0:
                pos.exit_idx = i
                pos.exit_price = current_price
                pos.pnl = -pos.margin  # 亏完保证金
                pos.pnl_pct = -1.0
                pos.exit_reason = "liquidation"
                capital -= pos.margin  # 保证金归零
                trades.append(pos)
                liquidated.append(pos)
        for pos in liquidated:
            open_positions.remove(pos)

        # ---- 2. 检查止盈/止损/移动止损/时间退出/信号退出 ----
        closed_positions = []
        for pos in open_positions:
            exit_price = None
            exit_reason = ""

            # 止损（使用动态止损价）
            sl_price = _get_stop_loss_price(pos, exit_cfg)
            if pos.direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"

            # 止盈
            if exit_price is None:
                if pos.direction == 1:
                    tp_price = pos.entry_price * (1 + exit_cfg.take_profit_pct)
                    if high >= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"
                else:
                    tp_price = pos.entry_price * (1 - exit_cfg.take_profit_pct)
                    if low <= tp_price:
                        exit_price = tp_price
                        exit_reason = "take_profit"

            # 移动止损
            if exit_price is None and exit_cfg.trailing_stop:
                if pos.direction == 1:
                    highest = df["high"].iloc[pos.entry_idx:i + 1].max()
                    trail_price = highest * (1 - exit_cfg.trailing_stop_pct)
                    if low <= trail_price and current_price < highest * 0.99:
                        exit_price = trail_price
                        exit_reason = "trailing_stop"
                else:
                    lowest = df["low"].iloc[pos.entry_idx:i + 1].min()
                    trail_price = lowest * (1 + exit_cfg.trailing_stop_pct)
                    if high >= trail_price and current_price > lowest * 1.01:
                        exit_price = trail_price
                        exit_reason = "trailing_stop"

            # 时间退出
            if exit_price is None and exit_cfg.time_exit_bars:
                if i - pos.entry_idx >= exit_cfg.time_exit_bars:
                    exit_price = current_price
                    exit_reason = "time_exit"

            # 信号退出
            if exit_price is None and exit_signals.iloc[i]:
                exit_price = current_price
                exit_reason = "signal_exit"

            # 平仓结算
            if exit_price is not None:
                pnl_pct = _calc_unrealized_pnl_pct(pos, exit_price)
                pnl_pct = max(pnl_pct, -1.0)
                pnl = pos.margin * pnl_pct
                pos.exit_idx = i
                pos.exit_price = exit_price
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct
                pos.exit_reason = exit_reason
                capital += pos.margin + pnl  # 归还保证金 + 盈亏
                daily_loss += min(0, pnl)
                closed_positions.append(pos)
                trades.append(pos)

        for pos in closed_positions:
            open_positions.remove(pos)

        # ---- 3. 滚仓检测 ----
        if rolling_enabled:
            new_rolling_trades = []
            for pos in open_positions:
                pos_id = id(pos)
                gen = rolling_count.get(pos_id, 0)
                if gen >= pos_cfg.rolling_max_times:
                    continue

                unrealized_pnl_pct = _calc_unrealized_pnl_pct(pos, current_price)
                if unrealized_pnl_pct >= pos_cfg.rolling_trigger_pct:
                    # 浮盈金额
                    unrealized_pnl = _calc_unrealized_pnl(pos, current_price)
                    # 拿出一部分浮盈作为新仓保证金
                    new_margin = unrealized_pnl * pos_cfg.rolling_reinvest_pct

                    if new_margin > 0 and capital >= 0:
                        # 开滚仓新仓位
                        rolling_trade = Trade(
                            entry_idx=i,
                            entry_price=current_price,
                            direction=pos.direction,
                            size=new_margin / initial_capital,
                            leverage=pos.leverage,
                            margin=new_margin,
                            is_rolling=True,
                            rolling_generation=gen + 1,
                            stop_loss_price=(
                                current_price * (1 - exit_cfg.stop_loss_pct)
                                if pos.direction == 1
                                else current_price * (1 + exit_cfg.stop_loss_pct)
                            ),
                        )
                        new_rolling_trades.append(rolling_trade)
                        rolling_count[pos_id] = gen + 1

                        # 老仓止损移到成本价（保本）
                        if pos_cfg.rolling_move_stop:
                            pos.stop_loss_price = pos.entry_price

                        # 新仓保证金不从capital扣（因为是浮盈，还在持仓里）
                        # 但我们需要记录这部分浮盈已被"锁定"用于滚仓
                        # 简化处理：从capital中预扣新仓保证金
                        capital -= new_margin

            open_positions.extend(new_rolling_trades)

        # ---- 4. 检查是否开新仓（信号触发） ----
        signal = signals.iloc[i]
        non_rolling_positions = [p for p in open_positions if not p.is_rolling or p.rolling_generation == 0]
        can_open = (
            signal != 0
            and len(non_rolling_positions) < pos_cfg.max_concurrent
            and capital > 0
            and (i - last_trade_idx) >= cool_down
            and abs(daily_loss) < risk_cfg.max_daily_loss_pct * initial_capital
        )

        # 最大回撤检查
        peak_equity = max(equity) if equity else initial_capital
        total_equity = capital + sum(_calc_unrealized_pnl(p, current_price) + p.margin for p in open_positions)
        if peak_equity > 0:
            current_dd = (peak_equity - total_equity) / peak_equity
            if current_dd >= risk_cfg.max_drawdown_pct:
                can_open = False

        if can_open:
            direction = 1 if signal > 0 else -1

            if pos_cfg.direction == "long_only" and direction == -1:
                direction = 0
            elif pos_cfg.direction == "short_only" and direction == 1:
                direction = 0

            if direction != 0:
                margin = capital * pos_cfg.max_position_pct
                margin = min(margin, capital)  # 不能超过可用资金

                if margin > 0:
                    trade = Trade(
                        entry_idx=i,
                        entry_price=current_price,
                        direction=direction,
                        size=pos_cfg.max_position_pct,
                        leverage=pos_cfg.leverage,
                        margin=margin,
                        rolling_generation=0,
                    )
                    open_positions.append(trade)
                    capital -= margin  # 扣除保证金
                    last_trade_idx = i

        # 记录权益（现金 + 所有持仓浮盈 + 所有持仓保证金）
        total_equity = capital + sum(
            _calc_unrealized_pnl(p, current_price) + p.margin
            for p in open_positions
        )
        equity.append(total_equity)

        # 日切重置daily_loss
        if i % 24 == 0:
            daily_loss = 0.0

    # 强制平仓所有未关闭的持仓
    for pos in open_positions:
        current_price = df["close"].iloc[-1]
        pnl_pct = _calc_unrealized_pnl_pct(pos, current_price)
        pnl_pct = max(pnl_pct, -1.0)
        pnl = pos.margin * pnl_pct
        pos.exit_idx = len(df) - 1
        pos.exit_price = current_price
        pos.pnl = pnl
        pos.pnl_pct = pnl_pct
        pos.exit_reason = "end_of_data"
        capital += pos.margin + pnl
        trades.append(pos)
    equity.append(capital)

    # ---- 计算绩效指标 ----
    equity_series = pd.Series(equity)
    result = BacktestResult()
    result.trades = trades
    result.equity_curve = equity_series
    result.total_trades = len(trades)
    result.rolling_trades = sum(1 for t in trades if t.is_rolling)
    result.total_return = (equity_series.iloc[-1] - initial_capital) / initial_capital

    if trades:
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate = len(wins) / len(pnls) if pnls else 0
        result.avg_trade_pnl = np.mean(pnls)
        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = np.mean(losses) if losses else 0

        total_win = sum(wins)
        total_loss = abs(sum(losses))
        result.profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

        # 连续赢/亏
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

    # 夏普比率
    returns = equity_series.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        result.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24)
    else:
        result.sharpe_ratio = 0

    # 最大回撤
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    result.max_drawdown = abs(drawdown.min())

    # 按regime统计绩效
    if regime is not None:
        for regime_type in regime.unique():
            mask = regime == regime_type
            regime_indices = mask[mask].index.tolist()
            regime_trades = [t for t in trades if t.entry_idx in regime_indices]
            if regime_trades:
                rpnls = [t.pnl_pct for t in regime_trades]
                rwins = [p for p in rpnls if p > 0]
                result.regime_performance[regime_type] = {
                    "trades": len(regime_trades),
                    "return": sum(rpnls),
                    "win_rate": len(rwins) / len(rpnls),
                    "avg_pnl": np.mean(rpnls),
                }

    return result


def batch_backtest(
    df: pd.DataFrame,
    strategies: list[BotStrategy],
    regime: Optional[pd.Series] = None,
) -> dict[str, BacktestResult]:
    """批量回测多个策略。"""
    results = {}
    for i, strategy in enumerate(strategies):
        try:
            result = run_backtest(df, strategy, regime=regime)
            results[strategy.bot_id] = result
            rolling_info = f", rolling={result.rolling_trades}" if result.rolling_trades > 0 else ""
            print(f"[{i + 1}/{len(strategies)}] {strategy.bot_id}: "
                  f"return={result.total_return:.2%}, trades={result.total_trades}{rolling_info}")
        except Exception as e:
            print(f"[{i + 1}/{len(strategies)}] {strategy.bot_id}: FAILED - {e}")
    return results
