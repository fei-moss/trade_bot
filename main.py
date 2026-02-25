"""
策略发现工厂 - 主入口

三种模式:
  python main.py                 # 从零生成第0代Bot
  python main.py --evolve        # 单次LLM进化
  python main.py --loop          # 持续运行: 自动生成+每N小时进化一次
"""

import os
import sys
import time
import argparse
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.data.fetcher import fetch_ohlcv
from src.data.regime import classify_regime, regime_summary, Regime
from src.backtest.engine import batch_backtest
from src.generator.weight_generator import generate_batch
from src.generator.diversity import select_diverse_subset, strategy_similarity_report
from src.output.profiler import generate_all_profiles


def run_pipeline(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 120,
    target_bots: int = 100,
    generate_count: int = 150,
    regime_version: str = "v1",
    output_dir: str = "profiles",
):
    if symbols is None:
        symbols = ["BTC/USDT"]

    print("=" * 60)
    print("  策略发现工厂 v2.0 - 42维权重向量")
    print("=" * 60)

    # ---- Step 1: 数据采集 ----
    print("\n[Step 1/6] 下载行情数据...")
    market_data = {}
    for symbol in symbols:
        df = fetch_ohlcv(symbol, timeframe, days)
        market_data[symbol] = df
        print(f"  {symbol}: {len(df)} candles")

    # ---- Step 2: Regime识别 ----
    print(f"\n[Step 2/6] 识别行情regime (detector={regime_version})...")
    regimes = {}
    for symbol, df in market_data.items():
        regime = classify_regime(df, version=regime_version)
        regimes[symbol] = regime
        summary = regime_summary(df, regime)
        print(f"  {symbol} regime分布:")
        for r, s in summary.items():
            print(f"    {r}: {s['pct']:.1%} ({s['count']} bars)")

    # ---- Step 3: 生成Bot ----
    print(f"\n[Step 3/6] 生成 {generate_count} 个候选Bot...")
    bots = generate_batch(count=generate_count)
    print(f"  成功生成: {len(bots)} 个Bot")

    from collections import Counter
    sets = Counter(b.weights.indicator_set for b in bots)
    conds = Counter(b.weights.entry_condition for b in bots)
    print(f"  指标集分布: {dict(sets)}")
    print(f"  入场条件分布: {dict(conds)}")

    # ---- Step 4: 回测 ----
    print(f"\n[Step 4/6] 回测所有Bot...")
    primary_symbol = symbols[0]
    primary_df = market_data[primary_symbol]
    primary_regime = regimes[primary_symbol]

    results = batch_backtest(primary_df, bots, regime=primary_regime)
    print(f"  回测完成: {len(results)}/{len(bots)} 成功")

    # ---- Step 5: 多样性筛选 ----
    print(f"\n[Step 5/6] 多样性筛选 → {target_bots} 个...")
    selected = select_diverse_subset(bots, results, target_count=target_bots)
    print(f"  筛选后: {len(selected)} 个Bot")

    if len(selected) >= 2:
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


def run_evolve(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 120,
    target_bots: int = 10,
    llm_count: int = 5,
    model: str = "anthropic/claude-sonnet-4.6",
    regime_version: str = "v1",
    elite_pct: float = 0.2,
    mutation_rate: float = 0.2,
    fitness_metric: str = "Sharpe",
    profile_dir: str = "profiles",
):
    """自动找到最新一代profiles，通过LLM+遗传算法进化产生下一代。"""
    from src.generator.llm_evolver import (
        load_profiles, hybrid_evolve, append_generation_log,
        find_latest_generation,
    )

    if symbols is None:
        symbols = ["BTC/USDT"]

    abs_profile_dir = os.path.join(ROOT_DIR, profile_dir)

    print("=" * 60)
    print("  策略发现工厂 v2.0 - LLM进化模式")
    print("=" * 60)

    # ---- Step 1: 自动定位最新一代 ----
    print("\n[Step 1/5] 加载最新一代Bot...")
    latest_dir, latest_gen = find_latest_generation(abs_profile_dir)
    print(f"  定位到最新代: Gen {latest_gen} @ {latest_dir}")

    bots, prev_results = load_profiles(latest_dir)
    if not bots:
        print("  ERROR: 没有找到可用的Bot profiles, 请先运行 main.py 生成第0代")
        return None, None

    current_gen = max((b.generation for b in bots), default=latest_gen)
    returns = [r.get("total_return", 0) for r in prev_results.values()]
    avg_ret = sum(returns) / len(returns) if returns else 0
    print(f"  当前代: Gen {current_gen} | {len(bots)} Bots | 平均收益: {avg_ret:.2%}")

    # ---- Step 2: 加载行情数据 ----
    print(f"\n[Step 2/5] 加载行情数据...")
    market_df = fetch_ohlcv(symbols[0], timeframe, days)
    regime = classify_regime(market_df, version=regime_version)
    summary = regime_summary(market_df, regime)
    print(f"  {symbols[0]}: {len(market_df)} candles")
    for r, s in summary.items():
        print(f"    {r}: {s['pct']:.1%}")

    # ---- Step 3: 混合进化 ----
    print(f"\n[Step 3/5] 混合进化 (Elite + LLM + Genetic)...")
    print(f"  目标: {target_bots} Bots | LLM生成: {llm_count} | 模型: {model}")
    next_gen, evo_info = hybrid_evolve(
        bots=bots,
        results=prev_results,
        market_df=market_df,
        regime=regime,
        target_count=target_bots,
        elite_pct=elite_pct,
        llm_count=llm_count,
        mutation_rate=mutation_rate,
        model=model,
        fitness_metric=fitness_metric,
    )
    new_gen_num = evo_info["gen"]
    print(f"\n  进化结果: Gen {new_gen_num}")
    print(f"    Elite保留: {evo_info['elite_preserved']}")
    print(f"    LLM生成:   {evo_info['llm_bots_generated']}")
    print(f"    遗传填充:   {evo_info['genetic_filled']}")

    if evo_info.get("llm_analysis_summary"):
        print(f"\n  LLM分析摘要:\n    {evo_info['llm_analysis_summary'][:200]}")

    # ---- Step 4: 回测新一代 ----
    print(f"\n[Step 4/5] 回测新一代Bot...")
    new_results = batch_backtest(market_df, next_gen, regime=regime)
    print(f"  回测完成: {len(new_results)}/{len(next_gen)} 成功")

    new_returns = []
    for r in new_results.values():
        rd = r.to_dict() if hasattr(r, "to_dict") else r
        new_returns.append(rd.get("total_return", 0))
    if new_returns:
        print(f"  新一代表现: 平均={sum(new_returns)/len(new_returns):.2%}, "
              f"最好={max(new_returns):.2%}, 最差={min(new_returns):.2%}")

    # ---- Step 5: 输出新一代Profiles ----
    print(f"\n[Step 5/5] 输出新一代 Bot Profile...")
    output_dir = os.path.join(abs_profile_dir, f"gen_{new_gen_num}")
    paths = generate_all_profiles(next_gen, new_results, output_dir=output_dir)

    append_generation_log(
        evo_info,
        {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in new_results.items()},
        profile_dir=abs_profile_dir,
    )

    print("\n" + "=" * 60)
    print(f"  进化完成! Gen {current_gen} → Gen {new_gen_num}")
    print(f"  共生成 {len(paths)} 个Bot Profile")
    print(f"  输出目录: {output_dir}")
    print(f"  进化日志: {abs_profile_dir}/evolution_log.json")
    print("=" * 60)

    return next_gen, new_results


def run_walkforward(
    symbols: list[str] = None,
    timeframe: str = "1h",
    days: int = 120,
    target_bots: int = 10,
    generate_count: int = 25,
    llm_count: int = 5,
    model: str = "anthropic/claude-sonnet-4.6",
    regime_version: str = "v1",
    elite_pct: float = 0.2,
    mutation_rate: float = 0.2,
    fitness_metric: str = "Sharpe",
    profile_dir: str = "profiles",
    warmup_days: int = 30,
    step_hours: int = 24,
):
    """
    Walk-forward进化: 用历史数据模拟每天进化一次，一次跑完。

    流程:
    1. 加载全量历史数据
    2. 前warmup_days天作为初始回测窗口，生成Gen 0
    3. 每step_hours向前滑动，用[0:当前位置]回测当前Bot
    4. 基于回测结果进化产生下一代
    5. 重复直到数据用完
    """
    from src.generator.llm_evolver import (
        hybrid_evolve, append_generation_log,
    )

    if symbols is None:
        symbols = ["BTC/USDT"]

    abs_profile_dir = os.path.join(ROOT_DIR, profile_dir)
    bars_per_step = step_hours  # 1h timeframe: 24 bars = 24h
    if timeframe == "4h":
        bars_per_step = step_hours // 4
    elif timeframe == "1d":
        bars_per_step = step_hours // 24

    print("=" * 60)
    print("  策略发现工厂 v2.0 - Walk-Forward 进化")
    print("=" * 60)

    # ---- 加载全量数据 ----
    print(f"\n[准备] 加载全量行情数据...")
    full_df = fetch_ohlcv(symbols[0], timeframe, days)
    full_regime = classify_regime(full_df, version=regime_version)
    total_bars = len(full_df)
    warmup_bars = warmup_days * (24 if timeframe == "1h" else 6 if timeframe == "4h" else 1)
    warmup_bars = min(warmup_bars, total_bars - bars_per_step)

    total_steps = (total_bars - warmup_bars) // bars_per_step
    print(f"  总K线: {total_bars} | 预热: {warmup_bars} bars ({warmup_days}天)")
    print(f"  每步: {bars_per_step} bars ({step_hours}h) | 总进化次数: {total_steps}")
    print(f"  LLM模型: {model} | 每次LLM生成: {llm_count} 个Bot")

    summary = regime_summary(full_df, full_regime)
    print(f"  全量Regime分布:")
    for r, s in summary.items():
        print(f"    {r}: {s['pct']:.1%}")

    # ---- Gen 0: 用预热窗口生成+回测 ----
    print(f"\n{'=' * 60}")
    print(f"  Gen 0: 用前 {warmup_bars} bars 生成初始Bot")
    print(f"{'=' * 60}")

    warmup_df = full_df.iloc[:warmup_bars].reset_index(drop=True)
    warmup_regime = full_regime.iloc[:warmup_bars].reset_index(drop=True)

    bots = generate_batch(count=generate_count)
    results = batch_backtest(warmup_df, bots, regime=warmup_regime)

    selected = select_diverse_subset(bots, results, target_count=target_bots)
    selected_results = {b.bot_id: results[b.bot_id] for b in selected if b.bot_id in results}

    paths = generate_all_profiles(selected, selected_results, output_dir=abs_profile_dir)
    print(f"  Gen 0: {len(selected)} Bots, 平均收益={_avg_return(selected_results):.2%}")

    current_bots = selected
    current_results_dict = {
        bid: r.to_dict() if hasattr(r, "to_dict") else r
        for bid, r in selected_results.items()
    }

    # ---- Walk-Forward 进化循环 ----
    generation_summary = [{
        "gen": 0,
        "end_bar": warmup_bars,
        "bots": len(current_bots),
        "avg_return": _avg_return(selected_results),
        "best_return": _best_return(selected_results),
    }]

    for step in range(1, total_steps + 1):
        end_bar = warmup_bars + step * bars_per_step
        if end_bar > total_bars:
            break

        window_df = full_df.iloc[:end_bar].reset_index(drop=True)
        window_regime = full_regime.iloc[:end_bar].reset_index(drop=True)

        ts_end = full_df.iloc[end_bar - 1]["timestamp"] if "timestamp" in full_df.columns else f"bar_{end_bar}"

        print(f"\n{'─' * 60}")
        print(f"  Gen {step} | bars [0:{end_bar}] | 截至 {ts_end}")
        print(f"{'─' * 60}")

        # 混合进化
        try:
            next_bots, evo_info = hybrid_evolve(
                bots=current_bots,
                results=current_results_dict,
                market_df=window_df,
                regime=window_regime,
                target_count=target_bots,
                elite_pct=elite_pct,
                llm_count=llm_count,
                mutation_rate=mutation_rate,
                model=model,
                fitness_metric=fitness_metric,
            )
        except Exception as e:
            print(f"  进化失败: {e}, 跳过本轮")
            continue

        print(f"  Elite={evo_info['elite_preserved']} LLM={evo_info['llm_bots_generated']} Genetic={evo_info['genetic_filled']}")

        # 回测新一代（在更长的窗口上）
        new_results = batch_backtest(window_df, next_bots, regime=window_regime)

        avg_ret = _avg_return(new_results)
        best_ret = _best_return(new_results)
        print(f"  结果: 平均={avg_ret:.2%}, 最好={best_ret:.2%}")

        # 输出到 gen_N
        gen_dir = os.path.join(abs_profile_dir, f"gen_{step}")
        generate_all_profiles(next_bots, new_results, output_dir=gen_dir)

        append_generation_log(
            evo_info,
            {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in new_results.items()},
            profile_dir=abs_profile_dir,
        )

        generation_summary.append({
            "gen": step,
            "end_bar": end_bar,
            "bots": len(next_bots),
            "avg_return": avg_ret,
            "best_return": best_ret,
        })

        # 准备下一轮
        current_bots = next_bots
        current_results_dict = {
            bid: r.to_dict() if hasattr(r, "to_dict") else r
            for bid, r in new_results.items()
        }

    # ---- 最终报告 ----
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 进化完成!")
    print(f"  总代数: {len(generation_summary)} (Gen 0 ~ Gen {len(generation_summary)-1})")
    print(f"{'=' * 60}")
    print(f"\n  {'Gen':>4} | {'End Bar':>8} | {'Bots':>4} | {'Avg Return':>11} | {'Best Return':>12}")
    print(f"  {'─'*4} | {'─'*8} | {'─'*4} | {'─'*11} | {'─'*12}")
    for g in generation_summary:
        print(f"  {g['gen']:>4} | {g['end_bar']:>8} | {g['bots']:>4} | {g['avg_return']:>+10.2%} | {g['best_return']:>+11.2%}")

    avg_trend = [g["avg_return"] for g in generation_summary]
    if len(avg_trend) >= 2:
        first_half = avg_trend[:len(avg_trend)//2]
        second_half = avg_trend[len(avg_trend)//2:]
        print(f"\n  前半段平均收益: {sum(first_half)/len(first_half):.2%}")
        print(f"  后半段平均收益: {sum(second_half)/len(second_half):.2%}")
        improvement = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
        print(f"  进化改善: {improvement:+.2%}")

    return generation_summary


def _avg_return(results) -> float:
    rets = []
    for r in results.values():
        rd = r.to_dict() if hasattr(r, "to_dict") else r
        rets.append(rd.get("total_return", 0))
    return sum(rets) / len(rets) if rets else 0


def _best_return(results) -> float:
    rets = []
    for r in results.values():
        rd = r.to_dict() if hasattr(r, "to_dict") else r
        rets.append(rd.get("total_return", 0))
    return max(rets) if rets else 0


def main():
    parser = argparse.ArgumentParser(description="策略发现工厂 v2.0")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT"],
                        help="交易对列表")
    parser.add_argument("--timeframe", default="1h", help="K线周期")
    parser.add_argument("--days", type=int, default=120, help="数据天数")
    parser.add_argument("--bots", type=int, default=100, help="目标Bot数量")
    parser.add_argument("--generate", type=int, default=150, help="候选Bot数量")
    parser.add_argument("--regime", default="v1", choices=["v1", "v2", "v3"],
                        help="Regime检测器版本")
    parser.add_argument("--output", default="profiles", help="输出目录")

    parser.add_argument("--evolve", action="store_true",
                        help="单次进化: 加载最新代profiles,LLM+遗传算法生成下一代")
    parser.add_argument("--walkforward", action="store_true",
                        help="Walk-forward进化: 历史数据按天切片,每天进化一次,一次跑完")
    parser.add_argument("--warmup-days", type=int, default=30,
                        help="Walk-forward预热天数(初始回测窗口)")
    parser.add_argument("--step-hours", type=int, default=24,
                        help="Walk-forward每步滑动小时数")
    parser.add_argument("--evolve-count", type=int, default=5,
                        help="LLM每次生成的新Bot数量")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4.6",
                        help="OpenRouter模型ID (如: google/gemini-2.0-flash)")
    parser.add_argument("--elite-pct", type=float, default=0.2,
                        help="进化时保留的Elite比例")
    parser.add_argument("--fitness", default="Sharpe",
                        choices=["TotalReturn", "Sharpe", "Calmar", "WinRate", "ProfitFactor"],
                        help="进化适应度指标")

    args = parser.parse_args()

    if args.walkforward:
        run_walkforward(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            target_bots=args.bots,
            generate_count=args.generate,
            llm_count=args.evolve_count,
            model=args.model,
            regime_version=args.regime,
            elite_pct=args.elite_pct,
            fitness_metric=args.fitness,
            profile_dir=args.output,
            warmup_days=args.warmup_days,
            step_hours=args.step_hours,
        )
    elif args.evolve:
        run_evolve(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            target_bots=args.bots,
            llm_count=args.evolve_count,
            model=args.model,
            regime_version=args.regime,
            elite_pct=args.elite_pct,
            fitness_metric=args.fitness,
            profile_dir=args.output,
        )
    else:
        run_pipeline(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            target_bots=args.bots,
            generate_count=args.generate,
            regime_version=args.regime,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
