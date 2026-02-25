"""
权重向量 → 交易信号 翻译器

将42维权重向量翻译为具体的买卖信号序列。
核心流程:
1. 根据 indicator_set 计算对应指标
2. 根据 entry_condition 组合信号 (Strict=AND, Loose=OR, ...)
3. 根据 entry_confirmation 确认信号
4. 根据 regime_focus 过滤信号
5. 应用 reverse_logic_prob 随机反转
"""

import pandas as pd
import numpy as np
from src.strategy.schema import WeightVector, parse_ma_range
from src.strategy.indicators import (
    ema, sma, rsi, macd, bollinger_bands, bollinger_width, atr, adx,
    stochastic, supertrend, vwap, obv, volume_spike, price_breakout,
    keltner_channels, ichimoku, cci, mfi,
    compute_indicator,
)


# ============ 指标集 → 信号映射 ============

def _signals_ma_rsi(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """MA+RSI: 趋势跟踪 + 动量过滤"""
    fast, slow = parse_ma_range(w.ma_period_range)
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)
    rsi_val = rsi(df, w.rsi_period)

    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    cross_up = trend_up & (ema(df, fast).shift(1) <= ema(df, slow).shift(1))
    cross_down = trend_down & (ema(df, fast).shift(1) >= ema(df, slow).shift(1))

    long_sig = cross_up & (rsi_val < 70)
    short_sig = cross_down & (rsi_val > 30)
    return long_sig, short_sig


def _signals_macd_bb(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """MACD+BB: 动量 + 布林带回归"""
    macd_line, signal_line, hist = macd(df)
    bb_upper, bb_mid, bb_lower = bollinger_bands(df)

    macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    long_sig = macd_cross_up & (df["close"] < bb_mid)
    short_sig = macd_cross_down & (df["close"] > bb_mid)
    return long_sig, short_sig


def _signals_supertrend_adx(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """Supertrend+ADX: 强趋势跟踪"""
    fast, _ = parse_ma_range(w.ma_period_range)
    st_line, st_dir = supertrend(df, period=max(fast, 7), multiplier=3.0)
    adx_val, plus_di, minus_di = adx(df, w.adx_period)

    strong_trend = adx_val > 25
    dir_up = (st_dir == 1) & (st_dir.shift(1) == -1)
    dir_down = (st_dir == -1) & (st_dir.shift(1) == 1)

    long_sig = dir_up & strong_trend
    short_sig = dir_down & strong_trend
    return long_sig, short_sig


def _signals_ichimoku_kdj(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """Ichimoku+KDJ(Stochastic): 云图趋势 + 随机指标"""
    ichi = ichimoku(df)
    stoch_k, stoch_d = stochastic(df, k_period=w.rsi_period, d_period=3)

    cloud_top = pd.concat([ichi["senkou_a"], ichi["senkou_b"]], axis=1).max(axis=1)
    cloud_bottom = pd.concat([ichi["senkou_a"], ichi["senkou_b"]], axis=1).min(axis=1)

    above_cloud = df["close"] > cloud_top
    below_cloud = df["close"] < cloud_bottom
    tk_cross_up = (ichi["tenkan_sen"] > ichi["kijun_sen"]) & (ichi["tenkan_sen"].shift(1) <= ichi["kijun_sen"].shift(1))
    tk_cross_down = (ichi["tenkan_sen"] < ichi["kijun_sen"]) & (ichi["tenkan_sen"].shift(1) >= ichi["kijun_sen"].shift(1))

    long_sig = above_cloud & tk_cross_up & (stoch_k < 80)
    short_sig = below_cloud & tk_cross_down & (stoch_k > 20)
    return long_sig, short_sig


def _signals_ema_volume(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """EMA+Volume: 趋势 + 成交量确认"""
    fast, slow = parse_ma_range(w.ma_period_range)
    ema_val = ema(df, fast)
    vol_ratio = volume_spike(df, period=20)

    price_above = df["close"] > ema_val
    price_below = df["close"] < ema_val
    cross_up = price_above & (~price_above.shift(1).fillna(False))
    cross_down = price_below & (~price_below.shift(1).fillna(False))
    vol_high = vol_ratio > 1.5

    long_sig = cross_up & vol_high
    short_sig = cross_down & vol_high
    return long_sig, short_sig


def _signals_stochastic_cci(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """Stochastic+CCI: 超买超卖 + 动量反转"""
    stoch_k, stoch_d = stochastic(df, k_period=w.rsi_period)
    cci_val = cci(df, period=w.adx_period)

    long_sig = (stoch_k < 20) & (stoch_k > stoch_k.shift(1)) & (cci_val < -100)
    short_sig = (stoch_k > 80) & (stoch_k < stoch_k.shift(1)) & (cci_val > 100)
    return long_sig, short_sig


def _signals_price_action(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """PurePriceAction: 支撑阻力 + 蜡烛形态"""
    _, slow = parse_ma_range(w.ma_period_range)
    resistance = df["high"].rolling(slow).max().shift(1)
    support = df["low"].rolling(slow).min().shift(1)

    body = df["close"] - df["open"]
    body_abs = body.abs()
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]

    bullish_pin = (lower_shadow > 2 * body_abs) & (upper_shadow < body_abs * 0.3)
    bearish_pin = (upper_shadow > 2 * body_abs) & (lower_shadow < body_abs * 0.3)

    near_support = (df["low"] - support).abs() / df["close"] < 0.01
    near_resistance = (df["high"] - resistance).abs() / df["close"] < 0.01

    long_sig = near_support & (bullish_pin | (body > 0))
    short_sig = near_resistance & (bearish_pin | (body < 0))
    return long_sig, short_sig


def _signals_multi3(df: pd.DataFrame, w: WeightVector) -> tuple[pd.Series, pd.Series]:
    """Multi3: RSI + MACD + Bollinger 三指标组合"""
    rsi_val = rsi(df, w.rsi_period)
    macd_line, signal_line, hist = macd(df)
    bb_upper, bb_mid, bb_lower = bollinger_bands(df)

    rsi_long = rsi_val < 40
    rsi_short = rsi_val > 60
    macd_long = hist > 0
    macd_short = hist < 0
    bb_long = df["close"] < bb_mid
    bb_short = df["close"] > bb_mid

    long_sig = rsi_long & macd_long & bb_long
    short_sig = rsi_short & macd_short & bb_short
    return long_sig, short_sig


INDICATOR_SET_MAP = {
    "MA+RSI": _signals_ma_rsi,
    "MACD+BB": _signals_macd_bb,
    "Supertrend+ADX": _signals_supertrend_adx,
    "Ichimoku+KDJ": _signals_ichimoku_kdj,
    "EMA+Volume": _signals_ema_volume,
    "Stochastic+CCI": _signals_stochastic_cci,
    "PurePriceAction": _signals_price_action,
    "Multi3": _signals_multi3,
}


# ============ 入场条件逻辑 ============

def _apply_entry_condition(
    long_raw: pd.Series,
    short_raw: pd.Series,
    df: pd.DataFrame,
    w: WeightVector,
    regime: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """根据 entry_condition 调整信号逻辑"""
    cond = w.entry_condition

    if cond == "Strict":
        return long_raw, short_raw

    elif cond == "Loose":
        rsi_val = rsi(df, w.rsi_period)
        extra_long = rsi_val < 30
        extra_short = rsi_val > 70
        return long_raw | extra_long, short_raw | extra_short

    elif cond == "MomentumBreak":
        fast, _ = parse_ma_range(w.ma_period_range)
        breakout_high, breakout_low = price_breakout(df, period=fast)
        return long_raw & breakout_high, short_raw & breakout_low

    elif cond == "MeanReversion":
        bb_upper, _, bb_lower = bollinger_bands(df)
        long_rev = df["close"] < bb_lower
        short_rev = df["close"] > bb_upper
        return long_raw | long_rev, short_raw | short_rev

    elif cond == "RegimeConfirmed":
        bull_mask = regime == "BULL"
        bear_mask = regime == "BEAR"
        return long_raw & bull_mask, short_raw & bear_mask

    return long_raw, short_raw


# ============ 入场确认 ============

def _apply_confirmation(
    long_sig: pd.Series,
    short_sig: pd.Series,
    df: pd.DataFrame,
    w: WeightVector,
) -> tuple[pd.Series, pd.Series]:
    """根据 entry_confirmation 过滤信号"""
    conf = w.entry_confirmation

    if conf == "SingleBar":
        return long_sig, short_sig

    elif conf == "MultiBar":
        long_confirmed = long_sig & long_sig.shift(1).fillna(False)
        short_confirmed = short_sig & short_sig.shift(1).fillna(False)
        return long_confirmed, short_confirmed

    elif conf == "VolumeSpike":
        vol_ratio = volume_spike(df, period=20)
        vol_high = vol_ratio > 1.5
        return long_sig & vol_high, short_sig & vol_high

    return long_sig, short_sig


# ============ 主信号生成函数 ============

def generate_signals(
    df: pd.DataFrame,
    weights: WeightVector,
    regime: pd.Series,
) -> pd.Series:
    """
    根据权重向量生成交易信号。

    Returns:
        Series of int: 1=做多, -1=做空, 0=无信号
    """
    df = df.copy()

    sig_func = INDICATOR_SET_MAP.get(weights.indicator_set, _signals_ma_rsi)
    long_raw, short_raw = sig_func(df, weights)

    long_raw = long_raw.fillna(False)
    short_raw = short_raw.fillna(False)

    long_sig, short_sig = _apply_entry_condition(long_raw, short_raw, df, weights, regime)

    long_sig, short_sig = _apply_confirmation(long_sig, short_sig, df, weights)

    # regime过滤
    for i in range(len(df)):
        r = regime.iloc[i]
        if not weights.should_trade_regime(r):
            rng = np.random.default_rng(seed=i)
            if rng.random() > weights.regime_override_prob:
                long_sig.iloc[i] = False
                short_sig.iloc[i] = False

    # 反向逻辑
    if weights.reverse_logic_prob > 0:
        rng = np.random.default_rng(seed=42)
        reverse_mask = rng.random(len(df)) < weights.reverse_logic_prob
        for i in range(len(df)):
            if reverse_mask[i]:
                long_sig.iloc[i], short_sig.iloc[i] = short_sig.iloc[i], long_sig.iloc[i]

    # bias_towards_action: 低值 = 更少信号，高值 = 更多信号
    if weights.bias_towards_action < 0.5:
        suppress_rate = 0.5 - weights.bias_towards_action
        rng = np.random.default_rng(seed=123)
        suppress_mask = rng.random(len(df)) < suppress_rate
        long_sig = long_sig & ~pd.Series(suppress_mask, index=df.index)
        short_sig = short_sig & ~pd.Series(suppress_mask, index=df.index)

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[long_sig] = 1
    signals[short_sig] = -1
    # 多空同时触发时，BULL优先做多，BEAR优先做空
    conflict = long_sig & short_sig
    if conflict.any():
        for i in conflict[conflict].index:
            signals.iloc[i] = 1 if regime.iloc[i] == "BULL" else -1

    return signals
