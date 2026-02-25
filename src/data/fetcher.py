"""
数据采集模块

通过ccxt从交易所下载历史K线数据。
默认使用Binance，缓存到本地避免重复下载。
"""

import os
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_cache")


def get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 120,
    exchange_id: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    下载OHLCV数据。

    Args:
        symbol: 交易对
        timeframe: K线周期
        days: 下载天数（默认120天≈4个月）
        exchange_id: 交易所
        use_cache: 是否使用本地缓存

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file = os.path.join(
        DATA_DIR,
        f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_{days}d.csv"
    )

    # 检查缓存（24小时内有效）
    if use_cache and os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        if time.time() - mod_time < 86400:
            return pd.read_csv(cache_file, parse_dates=["timestamp"])

    exchange = get_exchange(exchange_id)
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())

    all_data = []
    limit = 1000

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            break

        if not ohlcv:
            break

        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # 下一根K线起始

        if len(ohlcv) < limit:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_data:
        raise ValueError(f"No data fetched for {symbol} {timeframe}")

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 缓存
    df.to_csv(cache_file, index=False)
    print(f"Fetched {len(df)} candles for {symbol} {timeframe}, cached to {cache_file}")

    return df


def fetch_multi_symbol(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 120,
    exchange_id: str = "binance",
) -> dict[str, pd.DataFrame]:
    """
    批量下载多个交易对数据。
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

    result = {}
    for symbol in symbols:
        try:
            result[symbol] = fetch_ohlcv(symbol, timeframe, days, exchange_id)
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")

    return result


def summarize_dataframe(df: pd.DataFrame, symbol: str = "") -> None:
    """打印DataFrame摘要信息。"""
    prefix = f"[{symbol}] " if symbol else ""
    print(f"{prefix}{len(df)}根K线, "
          f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}, "
          f"价格 {df['close'].min():.0f} ~ {df['close'].max():.0f}")


def fetch_multi_timeframe(
    symbol: str = "BTC/USDT",
    timeframes: list[str] = None,
    days: int = 120,
    exchange_id: str = "binance",
) -> dict[str, pd.DataFrame]:
    """
    下载同一交易对的多个时间周期数据。
    """
    if timeframes is None:
        timeframes = ["5m", "15m", "1h", "4h", "1d"]

    result = {}
    for tf in timeframes:
        try:
            result[tf] = fetch_ohlcv(symbol, tf, days, exchange_id)
        except Exception as e:
            print(f"Failed to fetch {symbol} {tf}: {e}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载交易所K线数据")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"],
                        help="交易对列表 (默认: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h",
                        help="K线周期 (默认: 1h)")
    parser.add_argument("--days", type=int, default=148,
                        help="下载天数 (默认: 148，约2025-10-01至今)")
    parser.add_argument("--exchange", default="binance",
                        help="交易所 (默认: binance)")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略缓存强制重新下载")
    args = parser.parse_args()

    for symbol in args.symbols:
        df = fetch_ohlcv(symbol, args.timeframe, args.days,
                         args.exchange, use_cache=not args.no_cache)
        summarize_dataframe(df, symbol)
