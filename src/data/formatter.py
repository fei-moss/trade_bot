"""
K线数据LLM可读格式化模块

把DataFrame转成LLM可直接消费的紧凑文本格式。
用于后续滑动窗口逐根K线决策场景（50根窗口，~1500 token）。
"""

import pandas as pd
import numpy as np

from src.strategy.indicators import (
    ema, rsi, macd, bollinger_bands, atr, volume_spike,
)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始OHLCV上追加常用指标列。

    追加列：
    - EMA_20, EMA_50 — 趋势
    - RSI — 动量
    - MACD, MACD_S, MACD_H — 动量
    - BB_U, BB_L — 波动率
    - ATR — 波动率
    - VOL_R — 成交量比率(20周期)

    NaN行（指标预热期）保留不删除。
    """
    df = df.copy()

    # 趋势
    df["EMA_20"] = ema(df, 20)
    df["EMA_50"] = ema(df, 50)

    # 动量
    df["RSI"] = rsi(df, 14)
    macd_line, signal_line, histogram = macd(df, 12, 26, 9)
    df["MACD"] = macd_line
    df["MACD_S"] = signal_line
    df["MACD_H"] = histogram

    # 波动率
    bb_upper, _, bb_lower = bollinger_bands(df, 20, 2.0)
    df["BB_U"] = bb_upper
    df["BB_L"] = bb_lower
    df["ATR"] = atr(df, 14)

    # 成交量比率
    df["VOL_R"] = volume_spike(df, 20)

    return df


def format_candles_for_llm(df: pd.DataFrame, end_idx: int, window: int = 50) -> str:
    """
    取df[end_idx-window+1 : end_idx+1]这window行，输出紧凑CSV文本。

    价格保留整数，指标保留1-2位小数，减少token消耗。
    每行约80-100字符 × 50行 ≈ 4000-5000字符 ≈ ~1500 token。

    Args:
        df: enrich_dataframe()处理过的DataFrame
        end_idx: 窗口结束位置（包含）
        window: 窗口大小，默认50

    Returns:
        CSV格式文本字符串
    """
    start_idx = max(0, end_idx - window + 1)
    chunk = df.iloc[start_idx:end_idx + 1]

    header = "时间,开,高,低,收,量,EMA20,EMA50,RSI,MACD,MACD_S,MACD_H,BB_U,BB_L,ATR,VOL_R"
    lines = [header]

    for _, row in chunk.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts_str = ts.strftime("%Y-%m-%d %H:%M")
        else:
            ts_str = str(ts)

        def fmt_price(v):
            if pd.isna(v):
                return ""
            return f"{v:.0f}"

        def fmt_ind(v, decimals=1):
            if pd.isna(v):
                return ""
            return f"{v:.{decimals}f}"

        fields = [
            ts_str,
            fmt_price(row["open"]),
            fmt_price(row["high"]),
            fmt_price(row["low"]),
            fmt_price(row["close"]),
            fmt_ind(row["volume"], 1),
            fmt_price(row["EMA_20"]),
            fmt_price(row["EMA_50"]),
            fmt_ind(row["RSI"], 1),
            fmt_ind(row["MACD"], 1),
            fmt_ind(row["MACD_S"], 1),
            fmt_ind(row["MACD_H"], 1),
            fmt_price(row["BB_U"]),
            fmt_price(row["BB_L"]),
            fmt_ind(row["ATR"], 1),
            fmt_ind(row["VOL_R"], 2),
        ]
        lines.append(",".join(fields))

    return "\n".join(lines)


def format_market_context(df: pd.DataFrame) -> str:
    """
    输出整体市场上下文（几行文字）。

    包含：当前价位在全量数据中的位置、近期趋势方向、波动率水平。

    Args:
        df: enrich_dataframe()处理过的DataFrame

    Returns:
        市场上下文描述文本
    """
    close = df["close"]
    current = close.iloc[-1]
    high_all = close.max()
    low_all = close.min()

    # 当前价在区间中的百分位
    if high_all != low_all:
        pct_position = (current - low_all) / (high_all - low_all) * 100
    else:
        pct_position = 50.0

    # 近期趋势：用最近20根K线的EMA_20斜率判断
    ema20 = df["EMA_20"]
    if len(ema20) >= 20 and not ema20.iloc[-20:].isna().all():
        recent = ema20.iloc[-20:].dropna()
        if len(recent) >= 2:
            slope = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100
            if slope > 1:
                trend = f"上涨趋势 (EMA20近20根涨{slope:.1f}%)"
            elif slope < -1:
                trend = f"下跌趋势 (EMA20近20根跌{slope:.1f}%)"
            else:
                trend = f"横盘震荡 (EMA20近20根变化{slope:.1f}%)"
        else:
            trend = "数据不足"
    else:
        trend = "数据不足"

    # 波动率水平：ATR占价格的百分比
    atr_val = df["ATR"].iloc[-1]
    if pd.notna(atr_val) and current > 0:
        atr_pct = atr_val / current * 100
        if atr_pct > 3:
            vol_level = f"高波动 (ATR占价格{atr_pct:.2f}%)"
        elif atr_pct > 1:
            vol_level = f"中波动 (ATR占价格{atr_pct:.2f}%)"
        else:
            vol_level = f"低波动 (ATR占价格{atr_pct:.2f}%)"
    else:
        vol_level = "数据不足"

    # 数据范围
    ts_start = df["timestamp"].iloc[0]
    ts_end = df["timestamp"].iloc[-1]
    total_bars = len(df)

    lines = [
        f"数据范围: {ts_start} ~ {ts_end} ({total_bars}根K线)",
        f"价格区间: {low_all:.0f} ~ {high_all:.0f}, 当前: {current:.0f} (位于区间{pct_position:.0f}%位置)",
        f"近期趋势: {trend}",
        f"波动率: {vol_level}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from src.data.fetcher import fetch_ohlcv, summarize_dataframe, DATA_DIR
    import os

    parser = argparse.ArgumentParser(description="生成带技术指标的K线数据")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易对 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="K线周期 (默认: 1h)")
    parser.add_argument("--days", type=int, default=148, help="天数 (默认: 148)")
    args = parser.parse_args()

    df = fetch_ohlcv(args.symbol, args.timeframe, args.days)
    summarize_dataframe(df, args.symbol)

    df_enriched = enrich_dataframe(df)
    out_file = os.path.join(
        DATA_DIR,
        f"{args.symbol.replace('/', '_')}_{args.timeframe}_{args.days}d_enriched.csv"
    )
    df_enriched.to_csv(out_file, index=False)
    print(f"Enriched data saved to {out_file}")
    print(f"Columns: {list(df_enriched.columns)}")
    print(df_enriched[["timestamp", "close", "EMA_20", "EMA_50", "RSI", "MACD", "ATR"]].tail(5).to_string(index=False))
