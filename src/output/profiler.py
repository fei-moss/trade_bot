"""
Bot Profile输出模块

将Bot配置(42维权重) + 回测结果打包成标准化的Profile文件。
输出格式兼容openclaw等管理平台消费。
"""

import os
import json
from datetime import datetime

from src.strategy.schema import BotConfig
from src.backtest.engine import BacktestResult


def generate_profile(
    bot: BotConfig,
    backtest_result: BacktestResult,
    output_dir: str = "profiles",
) -> str:
    """生成单个Bot的完整profile文件。"""
    bot.backtest_summary = backtest_result.to_dict()

    regime_perf = backtest_result.regime_performance
    if regime_perf:
        sorted_regimes = sorted(regime_perf.items(), key=lambda x: x[1].get("return", 0))
        bot.worst_regime = [sorted_regimes[0][0]] if sorted_regimes else []
        bot.best_regime = [sorted_regimes[-1][0]] if sorted_regimes else []

    profile = {
        "version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "bot": bot.to_dict(),
        "decision_weights": bot.weights.to_dict(),
        "skill_summary": _generate_skill_summary(bot, backtest_result),
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{bot.bot_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return path


def _generate_skill_summary(bot: BotConfig, result: BacktestResult) -> dict:
    """生成可被封装为skill的摘要信息。"""
    w = bot.weights
    return {
        "name": bot.name,
        "bot_id": bot.bot_id,
        "indicator_set": w.indicator_set,
        "entry_condition": w.entry_condition,
        "regime_focus": w.regime_focus,
        "base_leverage": w.base_leverage,
        "yolo_mode": w.yolo_mode,
        "one_liner": (
            f"{bot.name} | {w.indicator_set} | "
            f"Lev:{w.base_leverage}x | "
            f"Return:{result.total_return:.1%} | "
            f"WR:{result.win_rate:.0%} | "
            f"MDD:{result.max_drawdown:.1%}"
        ),
        "best_for": bot.best_regime,
        "worst_for": bot.worst_regime,
        "risk_level": _risk_level(w),
        "tags": bot.tags,
        "generation": bot.generation,
    }


def _risk_level(w) -> str:
    """根据权重推断风险等级"""
    score = 0
    if w.base_leverage >= 50:
        score += 3
    elif w.base_leverage >= 10:
        score += 2
    elif w.base_leverage >= 5:
        score += 1

    if w.yolo_mode in ("Medium", "High"):
        score += 2
    elif w.yolo_mode == "Low":
        score += 1

    if w.allow_blowup == "Yes":
        score += 1

    if w.sl_type == "None":
        score += 1

    if score >= 5:
        return "ultra_aggressive"
    elif score >= 3:
        return "aggressive"
    elif score >= 2:
        return "moderate"
    elif score >= 1:
        return "conservative"
    return "ultra_conservative"


def generate_all_profiles(
    bots: list[BotConfig],
    results: dict[str, BacktestResult],
    output_dir: str = "profiles",
) -> list[str]:
    """批量生成所有Bot的profile。"""
    paths = []
    for bot in bots:
        if bot.bot_id in results:
            path = generate_profile(bot, results[bot.bot_id], output_dir)
            paths.append(path)
            print(f"Generated: {path}")

    index = {
        "version": "2.0",
        "total_bots": len(paths),
        "generated_at": datetime.now().isoformat(),
        "bots": [],
    }
    for bot in bots:
        if bot.bot_id in results:
            r = results[bot.bot_id]
            w = bot.weights
            index["bots"].append({
                "bot_id": bot.bot_id,
                "name": bot.name,
                "indicator_set": w.indicator_set,
                "entry_condition": w.entry_condition,
                "base_leverage": w.base_leverage,
                "regime_focus": w.regime_focus,
                "total_return": round(r.total_return, 4),
                "sharpe": round(r.sharpe_ratio, 4),
                "max_drawdown": round(r.max_drawdown, 4),
                "win_rate": round(r.win_rate, 4),
                "total_trades": r.total_trades,
                "generation": bot.generation,
                "tags": bot.tags,
            })

    index["bots"].sort(key=lambda x: x["total_return"], reverse=True)

    index_path = os.path.join(output_dir, "_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nIndex generated: {index_path}")
    print(f"Total profiles: {len(paths)}")

    return paths
