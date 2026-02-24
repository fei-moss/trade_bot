"""
行情Regime识别模块

将历史数据按行情特征分段标记：
- trending_up: 单边上涨
- trending_down: 单边下跌
- ranging: 横盘震荡
- volatile: 剧烈波动（大幅上下）
"""

import pandas as pd
import numpy as np
from src.strategy.indicators import ema, atr, adx


class RegimeType:
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


def classify_regime(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """
    对每根K线标记其所处的行情regime。

    逻辑:
    1. 趋势方向: EMA斜率
    2. 趋势强度: ADX
    3. 波动率: ATR相对值

    Args:
        df: OHLCV DataFrame
        window: 评估窗口大小

    Returns:
        Series of regime labels
    """
    df = df.copy()

    # 计算指标
    df["ema_50"] = ema(df, 50)
    df["ema_slope"] = df["ema_50"].pct_change(window)
    adx_val, _, _ = adx(df, 14)
    df["adx"] = adx_val
    atr_val = atr(df, 14)
    df["atr_pct"] = atr_val / df["close"]

    # 波动率的历史分位数
    df["atr_rank"] = df["atr_pct"].rolling(window * 4).rank(pct=True)

    # 收益率
    df["return"] = df["close"].pct_change(window)

    regime = pd.Series("", index=df.index)

    for i in range(window, len(df)):
        slope = df["ema_slope"].iloc[i]
        adx_v = df["adx"].iloc[i]
        atr_r = df["atr_rank"].iloc[i]
        ret = df["return"].iloc[i]

        # 高波动
        if atr_r is not None and atr_r > 0.8:
            regime.iloc[i] = RegimeType.VOLATILE
        # 强趋势
        elif adx_v is not None and adx_v > 25:
            if slope > 0.01:
                regime.iloc[i] = RegimeType.TRENDING_UP
            elif slope < -0.01:
                regime.iloc[i] = RegimeType.TRENDING_DOWN
            else:
                regime.iloc[i] = RegimeType.RANGING
        # 弱趋势/震荡
        else:
            if abs(ret) > 0.05:
                regime.iloc[i] = RegimeType.TRENDING_UP if ret > 0 else RegimeType.TRENDING_DOWN
            else:
                regime.iloc[i] = RegimeType.RANGING

    # 回填前面没有数据的部分
    regime = regime.replace("", np.nan).ffill().fillna(RegimeType.RANGING)
    return regime


def get_regime_segments(df: pd.DataFrame, regime: pd.Series) -> list[dict]:
    """
    将regime序列切割成连续段落。

    Returns:
        list of {"regime": str, "start_idx": int, "end_idx": int, "duration": int}
    """
    segments = []
    current_regime = regime.iloc[0]
    start_idx = 0

    for i in range(1, len(regime)):
        if regime.iloc[i] != current_regime:
            segments.append({
                "regime": current_regime,
                "start_idx": start_idx,
                "end_idx": i - 1,
                "duration": i - start_idx,
                "start_time": df["timestamp"].iloc[start_idx] if "timestamp" in df.columns else None,
                "end_time": df["timestamp"].iloc[i - 1] if "timestamp" in df.columns else None,
            })
            current_regime = regime.iloc[i]
            start_idx = i

    # 最后一段
    segments.append({
        "regime": current_regime,
        "start_idx": start_idx,
        "end_idx": len(regime) - 1,
        "duration": len(regime) - start_idx,
        "start_time": df["timestamp"].iloc[start_idx] if "timestamp" in df.columns else None,
        "end_time": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
    })

    return segments


def regime_summary(df: pd.DataFrame, regime: pd.Series) -> dict:
    """
    统计各regime占比和特征。
    """
    summary = {}
    for r in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN, RegimeType.RANGING, RegimeType.VOLATILE]:
        mask = regime == r
        count = mask.sum()
        if count > 0:
            sub = df[mask]
            summary[r] = {
                "count": int(count),
                "pct": count / len(regime),
                "avg_return": float(sub["close"].pct_change().mean()),
                "volatility": float(sub["close"].pct_change().std()),
            }
    return summary
