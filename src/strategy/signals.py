"""
信号生成引擎

将策略配置(BotStrategy)中的entry/exit条件翻译成具体的买卖信号序列。
"""

import pandas as pd
import numpy as np
from src.strategy.schema import BotStrategy, IndicatorCondition, EntryRule
from src.strategy.indicators import compute_indicator


def evaluate_condition(df: pd.DataFrame, cond: IndicatorCondition,
                       indicator_data: dict[str, pd.Series]) -> pd.Series:
    """
    评估单个指标条件，返回布尔Series。
    """
    ind_type = cond.indicator
    op = cond.condition
    params = cond.params

    # 确保指标已计算
    if not any(k in df.columns for k in indicator_data):
        new_cols = compute_indicator(df, ind_type, params)
        indicator_data.update(new_cols)
        for k, v in new_cols.items():
            df[k] = v

    # 根据条件类型评估
    if ind_type == "ema_cross" or ind_type == "sma_cross":
        prefix = "ema" if "ema" in ind_type else "sma"
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        fast_col = f"{prefix}_{fast}"
        slow_col = f"{prefix}_{slow}"

        if op == "cross_above":
            return (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1))
        elif op == "cross_below":
            return (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1))
        elif op == "above":
            return df[fast_col] > df[slow_col]
        elif op == "below":
            return df[fast_col] < df[slow_col]

    elif ind_type == "rsi":
        col = "rsi"
        if op == "above":
            return df[col] > cond.value
        elif op == "below":
            return df[col] < cond.value
        elif op == "cross_above":
            return (df[col] > cond.value) & (df[col].shift(1) <= cond.value)
        elif op == "cross_below":
            return (df[col] < cond.value) & (df[col].shift(1) >= cond.value)
        elif op == "between":
            return (df[col] >= cond.value) & (df[col] <= cond.value2)
        elif op == "increasing":
            return df[col] > df[col].shift(1)
        elif op == "decreasing":
            return df[col] < df[col].shift(1)

    elif ind_type == "macd":
        if op == "cross_above":
            return (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        elif op == "cross_below":
            return (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        elif op == "above":
            return df["macd_hist"] > 0
        elif op == "below":
            return df["macd_hist"] < 0
        elif op == "increasing":
            return df["macd_hist"] > df["macd_hist"].shift(1)
        elif op == "decreasing":
            return df["macd_hist"] < df["macd_hist"].shift(1)

    elif ind_type == "bollinger":
        if op == "above":
            return df["close"] > df["bb_upper"]
        elif op == "below":
            return df["close"] < df["bb_lower"]
        elif op == "cross_above":
            return (df["close"] > df["bb_upper"]) & (df["close"].shift(1) <= df["bb_upper"].shift(1))
        elif op == "cross_below":
            return (df["close"] < df["bb_lower"]) & (df["close"].shift(1) >= df["bb_lower"].shift(1))
        elif op == "squeeze":
            # 布林带宽度低于历史N期最低
            percentile = df["bb_width"].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            return percentile < 0.2
        elif op == "expansion":
            percentile = df["bb_width"].rolling(50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            return percentile > 0.8

    elif ind_type == "adx":
        if op == "above":
            return df["adx"] > cond.value
        elif op == "below":
            return df["adx"] < cond.value
        elif op == "increasing":
            return df["adx"] > df["adx"].shift(1)

    elif ind_type == "stochastic":
        if op == "cross_above":
            return (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
        elif op == "cross_below":
            return (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
        elif op == "above":
            return df["stoch_k"] > cond.value
        elif op == "below":
            return df["stoch_k"] < cond.value

    elif ind_type == "supertrend":
        if op == "above":
            return df["supertrend_dir"] == 1
        elif op == "below":
            return df["supertrend_dir"] == -1
        elif op == "cross_above":
            return (df["supertrend_dir"] == 1) & (df["supertrend_dir"].shift(1) == -1)
        elif op == "cross_below":
            return (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) == 1)

    elif ind_type == "volume_spike":
        threshold = cond.value or 2.0
        if op == "above":
            return df["vol_ratio"] > threshold
        elif op == "below":
            return df["vol_ratio"] < threshold

    elif ind_type == "price_breakout":
        if op == "above" or op == "cross_above":
            return df["breakout_high"]
        elif op == "below" or op == "cross_below":
            return df["breakout_low"]

    elif ind_type == "vwap":
        if op == "above":
            return df["close"] > df["vwap"]
        elif op == "below":
            return df["close"] < df["vwap"]
        elif op == "cross_above":
            return (df["close"] > df["vwap"]) & (df["close"].shift(1) <= df["vwap"].shift(1))
        elif op == "cross_below":
            return (df["close"] < df["vwap"]) & (df["close"].shift(1) >= df["vwap"].shift(1))

    elif ind_type == "ichimoku":
        if op == "cross_above":
            # 转换线穿越基准线
            return (df["tenkan_sen"] > df["kijun_sen"]) & (df["tenkan_sen"].shift(1) <= df["kijun_sen"].shift(1))
        elif op == "cross_below":
            return (df["tenkan_sen"] < df["kijun_sen"]) & (df["tenkan_sen"].shift(1) >= df["kijun_sen"].shift(1))
        elif op == "above":
            # 价格在云上方
            cloud_top = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).max(axis=1)
            return df["close"] > cloud_top
        elif op == "below":
            cloud_bottom = pd.concat([df["senkou_a"], df["senkou_b"]], axis=1).min(axis=1)
            return df["close"] < cloud_bottom

    elif ind_type == "ema" or ind_type == "sma":
        prefix = ind_type
        period = params.get("period", 20)
        col = f"{prefix}_{period}"
        if op == "above":
            return df["close"] > df[col]
        elif op == "below":
            return df["close"] < df[col]
        elif op == "cross_above":
            return (df["close"] > df[col]) & (df["close"].shift(1) <= df[col].shift(1))
        elif op == "cross_below":
            return (df["close"] < df[col]) & (df["close"].shift(1) >= df[col].shift(1))

    elif ind_type == "atr":
        col = "atr"
        atr_pct = df[col] / df["close"]
        if op == "above":
            return atr_pct > cond.value
        elif op == "below":
            return atr_pct < cond.value

    elif ind_type == "keltner":
        if op == "above":
            return df["close"] > df["kc_upper"]
        elif op == "below":
            return df["close"] < df["kc_lower"]

    elif ind_type == "support_resistance":
        if op == "above":
            return df["close"] > df["resistance"] * (1 - (cond.value or 0.01))
        elif op == "below":
            return df["close"] < df["support"] * (1 + (cond.value or 0.01))

    elif ind_type == "candle_pattern":
        pattern = params.get("pattern", "hammer")
        if pattern in df.columns:
            return df[pattern].astype(bool)

    elif ind_type in ("cci", "williams_r", "mfi"):
        col = ind_type
        if op == "above":
            return df[col] > cond.value
        elif op == "below":
            return df[col] < cond.value
        elif op == "cross_above":
            return (df[col] > cond.value) & (df[col].shift(1) <= cond.value)
        elif op == "cross_below":
            return (df[col] < cond.value) & (df[col].shift(1) >= cond.value)

    # fallback
    return pd.Series(False, index=df.index)


def generate_entry_signals(df: pd.DataFrame, strategy: BotStrategy) -> pd.Series:
    """
    根据策略配置生成入场信号。
    entry_rules之间是OR关系，每个rule内的conditions是AND关系。

    返回: Series of int (1=做多信号, -1=做空信号, 0=无信号)
    """
    df = df.copy()
    indicator_data = {}
    direction = strategy.position.direction

    # 先计算所有需要的指标
    all_conditions = []
    for rule in strategy.entry_rules:
        all_conditions.extend(rule.conditions)
    for cond in strategy.exit_rule.signal_exit:
        all_conditions.append(cond)

    for cond in all_conditions:
        new_cols = compute_indicator(df, cond.indicator, cond.params)
        indicator_data.update(new_cols)
        for k, v in new_cols.items():
            if k not in df.columns:
                df[k] = v

    # 评估入场信号
    long_signal = pd.Series(False, index=df.index)
    short_signal = pd.Series(False, index=df.index)

    for rule in strategy.entry_rules:
        if not rule.conditions:
            continue

        # 每个rule内的conditions是AND
        rule_signal = pd.Series(True, index=df.index)
        for cond in rule.conditions:
            cond_result = evaluate_condition(df, cond, indicator_data)
            rule_signal = rule_signal & cond_result

        # 根据条件的语义判断方向
        # 简化：偶数rule做多，奇数rule做空（实际由条件语义决定）
        # 更好的方式：rule中的conditions含"上涨"语义→做多
        long_signal = long_signal | rule_signal

    # 生成方向信号
    signals = pd.Series(0, index=df.index)
    if direction in ("long_only", "both"):
        signals = signals.where(~long_signal, 1)
    if direction in ("short_only", "both"):
        # 对于做空，使用相同的信号但反向
        # 实际中应该有独立的做空规则，但在简化版本中反转信号
        if direction == "short_only":
            signals = signals.where(~long_signal, -1)

    return signals, df


def generate_exit_signals(df: pd.DataFrame, strategy: BotStrategy,
                          indicator_data: dict) -> pd.Series:
    """
    生成信号退出条件（不含止盈止损，那些在回测引擎中处理）。
    """
    exit_signal = pd.Series(False, index=df.index)

    for cond in strategy.exit_rule.signal_exit:
        cond_result = evaluate_condition(df, cond, indicator_data)
        exit_signal = exit_signal | cond_result

    return exit_signal
