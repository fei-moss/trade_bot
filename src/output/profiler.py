"""
Bot Profile输出模块

将策略配置 + 回测结果打包成标准化的Bot Profile文件。
这些文件就是最终交付物，可以直接被openclaw等管理平台消费。
"""

import os
import json
from datetime import datetime

from src.strategy.schema import BotStrategy
from src.backtest.engine import BacktestResult


def generate_profile(
    strategy: BotStrategy,
    backtest_result: BacktestResult,
    output_dir: str = "profiles",
) -> str:
    """
    生成单个bot的完整profile文件。

    Returns:
        输出文件路径
    """
    # 填充回测结果到策略对象
    strategy.backtest_summary = backtest_result.to_dict()

    # 确定best/worst regime
    regime_perf = backtest_result.regime_performance
    if regime_perf:
        sorted_regimes = sorted(regime_perf.items(), key=lambda x: x[1].get("return", 0))
        strategy.worst_regime = [sorted_regimes[0][0]] if sorted_regimes else []
        strategy.best_regime = [sorted_regimes[-1][0]] if sorted_regimes else []

    # 构建profile
    profile = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "strategy": strategy.to_dict(),
        "skill_summary": _generate_skill_summary(strategy, backtest_result),
    }

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{strategy.bot_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return path


def _generate_skill_summary(strategy: BotStrategy, result: BacktestResult) -> dict:
    """
    生成可被封装为skill的摘要信息。
    """
    return {
        "name": strategy.name,
        "personality": strategy.personality,
        "archetype": strategy.archetype,
        "aggressiveness": strategy.aggressiveness,
        "one_liner": (
            f"{strategy.name} - {strategy.personality}. "
            f"回测收益{result.total_return:.1%}, "
            f"胜率{result.win_rate:.0%}, "
            f"最大回撤{result.max_drawdown:.1%}."
        ),
        "best_for": strategy.best_regime,
        "worst_for": strategy.worst_regime,
        "risk_level": strategy.aggressiveness,
        "tags": strategy.tags,
    }


def generate_all_profiles(
    strategies: list[BotStrategy],
    results: dict[str, BacktestResult],
    output_dir: str = "profiles",
) -> list[str]:
    """
    批量生成所有bot的profile。
    """
    paths = []
    for strategy in strategies:
        if strategy.bot_id in results:
            path = generate_profile(strategy, results[strategy.bot_id], output_dir)
            paths.append(path)
            print(f"Generated: {path}")

    # 生成索引文件
    index = {
        "total_bots": len(paths),
        "generated_at": datetime.now().isoformat(),
        "bots": [],
    }
    for strategy in strategies:
        if strategy.bot_id in results:
            r = results[strategy.bot_id]
            index["bots"].append({
                "bot_id": strategy.bot_id,
                "name": strategy.name,
                "personality": strategy.personality,
                "archetype": strategy.archetype,
                "aggressiveness": strategy.aggressiveness,
                "total_return": round(r.total_return, 4),
                "sharpe": round(r.sharpe_ratio, 4),
                "max_drawdown": round(r.max_drawdown, 4),
                "win_rate": round(r.win_rate, 4),
                "total_trades": r.total_trades,
            })

    # 按收益排序
    index["bots"].sort(key=lambda x: x["total_return"], reverse=True)

    index_path = os.path.join(output_dir, "_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nIndex generated: {index_path}")
    print(f"Total profiles: {len(paths)}")

    return paths
