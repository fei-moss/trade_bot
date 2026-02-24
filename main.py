"""
策略发现工厂 - 主入口

完整流程:
1. 下载4个月真实行情数据
2. 识别行情regime
3. LLM批量生成多样化策略
4. 回测所有策略
5. 多样性筛选
6. 输出标准化Bot Profile
"""

import os
import sys
import argparse

# 项目根目录加入path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.data.fetcher import fetch_ohlcv, fetch_multi_symbol
from src.data.regime import classify_regime, regime_summary
from src.backtest.engine import run_backtest, batch_backtest
from src.generator.llm_generator import StrategyGenerator, summarize_market_data
from src.generator.diversity import select_diverse_subset, strategy_similarity_report
from src.output.profiler import generate_all_profiles


def run_pipeline(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 120,
    target_bots: int = 100,
    generate_count: int = 150,  # 多生成一些用于筛选
    output_dir: str = "profiles",
    api_key: str = None,
):
    """
    运行完整的策略发现流程。
    """
    if symbols is None:
        symbols = ["BTC/USDT"]

    print("=" * 60)
    print("  策略发现工厂 - Strategy Discovery Factory")
    print("=" * 60)

    # ---- Step 1: 数据采集 ----
    print("\n[Step 1/6] 下载行情数据...")
    market_data = {}
    for symbol in symbols:
        df = fetch_ohlcv(symbol, timeframe, days)
        market_data[symbol] = df
        print(f"  {symbol}: {len(df)} candles")

    # ---- Step 2: Regime识别 ----
    print("\n[Step 2/6] 识别行情regime...")
    regimes = {}
    for symbol, df in market_data.items():
        regime = classify_regime(df)
        regimes[symbol] = regime
        summary = regime_summary(df, regime)
        print(f"  {symbol} regime分布:")
        for r, s in summary.items():
            print(f"    {r}: {s['pct']:.1%} ({s['count']} bars)")

    # ---- Step 3: LLM生成策略 ----
    print(f"\n[Step 3/6] LLM生成 {generate_count} 个候选策略...")
    generator = StrategyGenerator(api_key=api_key)
    strategies = generator.generate_batch(
        market_data=market_data,
        regime={s: r for s, r in regimes.items()},
        count=generate_count,
    )
    print(f"  成功生成: {len(strategies)} 个策略")

    # ---- Step 4: 回测 ----
    print(f"\n[Step 4/6] 回测所有策略...")
    # 使用第一个symbol的数据做回测
    primary_symbol = symbols[0]
    primary_df = market_data[primary_symbol]
    primary_regime = regimes[primary_symbol]

    results = batch_backtest(primary_df, strategies, regime=primary_regime)
    print(f"  回测完成: {len(results)}/{len(strategies)} 成功")

    # ---- Step 5: 多样性筛选 ----
    print(f"\n[Step 5/6] 多样性筛选 → {target_bots} 个...")
    selected = select_diverse_subset(strategies, results, target_count=target_bots)
    print(f"  筛选后: {len(selected)} 个策略")

    # 多样性报告
    report = strategy_similarity_report(selected, results)
    print(f"\n  多样性报告:")
    print(f"    平均相关性: {report.get('avg_correlation', 0):.3f}")
    print(f"    平均多样性分数: {report.get('avg_diversity_score', 0):.3f}")
    print(f"    聚类数: {report.get('n_clusters', 0)}")
    print(f"    表现最好 Top 3:")
    for bid, ret in report.get("top_3_best", []):
        print(f"      {bid}: {ret:.2%}")
    print(f"    表现最差 Top 3:")
    for bid, ret in report.get("top_3_worst", []):
        print(f"      {bid}: {ret:.2%}")

    # ---- Step 6: 输出Profile ----
    print(f"\n[Step 6/6] 生成Bot Profile...")
    profile_dir = os.path.join(ROOT_DIR, output_dir)
    paths = generate_all_profiles(selected, results, output_dir=profile_dir)

    print("\n" + "=" * 60)
    print(f"  完成! 共生成 {len(paths)} 个Bot Profile")
    print(f"  输出目录: {profile_dir}")
    print("=" * 60)

    return selected, results


def main():
    parser = argparse.ArgumentParser(description="策略发现工厂")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"],
                        help="交易对列表")
    parser.add_argument("--timeframe", default="1h", help="K线周期")
    parser.add_argument("--days", type=int, default=120, help="数据天数")
    parser.add_argument("--bots", type=int, default=100, help="目标bot数量")
    parser.add_argument("--generate", type=int, default=150, help="候选策略数量")
    parser.add_argument("--output", default="profiles", help="输出目录")
    parser.add_argument("--api-key", default=None, help="Anthropic API key")

    args = parser.parse_args()

    run_pipeline(
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days,
        target_bots=args.bots,
        generate_count=args.generate,
        output_dir=args.output,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
