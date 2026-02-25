"""
权重向量生成器

一键生成100个具有唯一42维权重向量的Bot。
支持:
- 随机生成（确保最大多样性）
- 唯一性校验（哈希去重）
- 遗传交叉（从Top Bot产生后代）
- 变异（随机扰动权重）
"""

import random
import hashlib
import json
from typing import Optional
from dataclasses import asdict

from src.strategy.schema import (
    WeightVector, BotConfig, PARAM_SPACE, NORMALIZE_GROUPS,
)


def generate_weight_vector(seed: Optional[int] = None) -> WeightVector:
    """随机生成一个42维权重向量"""
    rng = random.Random(seed)
    params = {}

    for param_name, spec in PARAM_SPACE.items():
        if spec["type"] == "discrete":
            params[param_name] = rng.choice(spec["options"])
        elif spec["type"] == "continuous":
            lo, hi = spec["range"]
            params[param_name] = round(rng.uniform(lo, hi), 4)

    _normalize_groups(params)
    _enforce_consistency(params)

    return WeightVector.from_dict(params)


def _normalize_groups(params: dict):
    """归一化需要总和=1的参数组"""
    for group_name, fields in NORMALIZE_GROUPS.items():
        total = sum(params.get(f, 0.0) for f in fields)
        if total > 0:
            for f in fields:
                params[f] = round(params.get(f, 0.0) / total, 4)
        else:
            equal = round(1.0 / len(fields), 4)
            for f in fields:
                params[f] = equal


def _enforce_consistency(params: dict):
    """强制参数一致性约束"""
    lev = params.get("base_leverage", 1)
    yolo = params.get("yolo_mode", "Off")

    if lev <= 2 and yolo in ("Medium", "High"):
        params["yolo_mode"] = "Off"

    if params.get("allow_blowup") == "No":
        params["max_dd_tolerance"] = min(params.get("max_dd_tolerance", 0.3), 0.5)

    if params.get("sl_type") == "None" and params.get("tp_type") == "None":
        params["sl_type"] = "Fixed"

    if params.get("regime_focus") in ("BULL_ONLY", "BEAR_ONLY"):
        if params.get("exit_on_regime_change") == "No":
            params["exit_on_regime_change"] = "OnlyIfLoss"


def generate_bot(
    bot_number: int,
    seed: Optional[int] = None,
) -> BotConfig:
    """生成单个Bot配置"""
    weights = generate_weight_vector(seed=seed)
    bot_id = f"bot_{bot_number:03d}"
    fp = weights.fingerprint()

    indicator = weights.indicator_set.replace("+", "_").lower()
    name = f"{bot_id}_{indicator}_{weights.entry_condition.lower()}"

    tags = _generate_tags(weights)

    return BotConfig(
        bot_id=bot_id,
        name=name,
        weights=weights,
        generation=0,
        tags=tags,
    )


def _generate_tags(w: WeightVector) -> list[str]:
    """根据权重特征自动生成标签"""
    tags = []

    if w.base_leverage >= 50:
        tags.append("high_leverage")
    elif w.base_leverage <= 5:
        tags.append("low_leverage")

    if w.yolo_mode in ("Medium", "High"):
        tags.append("yolo")

    if w.reverse_logic_prob > 0.1:
        tags.append("contrarian")

    if w.entry_condition == "MeanReversion":
        tags.append("mean_reversion")
    elif w.entry_condition == "MomentumBreak":
        tags.append("momentum")

    if w.regime_focus != "ALL_THREE":
        tags.append(f"regime_{w.regime_focus.lower()}")

    if w.sl_type == "None":
        tags.append("no_stoploss")

    return tags


def generate_batch(
    count: int = 100,
    max_attempts: int = 500,
) -> list[BotConfig]:
    """
    批量生成Bot，确保唯一性。

    Args:
        count: 目标Bot数量
        max_attempts: 最大尝试次数（防止无限循环）

    Returns:
        list of BotConfig
    """
    bots = []
    seen_fingerprints = set()
    attempts = 0
    bot_number = 1

    while len(bots) < count and attempts < max_attempts:
        bot = generate_bot(bot_number, seed=attempts * 7 + 13)
        fp = bot.weights.fingerprint()

        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            bots.append(bot)
            bot_number += 1

        attempts += 1

    print(f"生成 {len(bots)}/{count} 个唯一Bot (尝试 {attempts} 次)")
    return bots


# ============ 遗传算法支持 ============

def crossover(
    parent_a: WeightVector,
    parent_b: WeightVector,
    crossover_rate: float = 0.5,
    seed: Optional[int] = None,
) -> WeightVector:
    """
    两个权重向量的遗传交叉。
    每个参数有crossover_rate概率从parent_b继承，否则从parent_a。
    """
    rng = random.Random(seed)
    a_dict = asdict(parent_a)
    b_dict = asdict(parent_b)
    child = {}

    for key in a_dict:
        if rng.random() < crossover_rate:
            child[key] = b_dict[key]
        else:
            child[key] = a_dict[key]

    _normalize_groups(child)
    _enforce_consistency(child)
    return WeightVector.from_dict(child)


def mutate(
    weights: WeightVector,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
) -> WeightVector:
    """
    随机变异权重向量。
    每个参数有mutation_rate概率被随机重新赋值。
    """
    rng = random.Random(seed)
    params = asdict(weights)

    for param_name, spec in PARAM_SPACE.items():
        if rng.random() < mutation_rate:
            if spec["type"] == "discrete":
                params[param_name] = rng.choice(spec["options"])
            elif spec["type"] == "continuous":
                lo, hi = spec["range"]
                params[param_name] = round(rng.uniform(lo, hi), 4)

    _normalize_groups(params)
    _enforce_consistency(params)
    return WeightVector.from_dict(params)


def evolve_generation(
    bots: list[BotConfig],
    results: dict,
    target_count: int = 100,
    elite_pct: float = 0.2,
    mutation_rate: float = 0.2,
    fitness_metric: str = "Sharpe",
) -> list[BotConfig]:
    """
    进化一代Bot。

    1. 按fitness排序
    2. 保留elite
    3. 交叉产生新Bot
    4. 变异
    """
    scored = []
    for bot in bots:
        r = results.get(bot.bot_id)
        if r is None:
            continue
        if fitness_metric == "TotalReturn":
            score = r.total_return
        elif fitness_metric == "Sharpe":
            score = r.sharpe_ratio
        elif fitness_metric == "Calmar":
            score = r.total_return / r.max_drawdown if r.max_drawdown > 0 else 0
        elif fitness_metric == "WinRate":
            score = r.win_rate
        elif fitness_metric == "ProfitFactor":
            score = r.profit_factor if r.profit_factor != float("inf") else 10.0
        else:
            score = r.sharpe_ratio
        scored.append((bot, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    elite_count = max(int(target_count * elite_pct), 2)
    elites = [b for b, _ in scored[:elite_count]]

    new_bots = []
    gen = elites[0].generation + 1 if elites else 1

    for i, bot in enumerate(elites):
        new_bot = BotConfig(
            bot_id=f"bot_{i + 1:03d}",
            name=bot.name,
            weights=bot.weights,
            generation=gen,
            parent_ids=[bot.bot_id],
            tags=bot.tags,
        )
        new_bots.append(new_bot)

    rng = random.Random(gen * 42)
    bot_num = len(new_bots) + 1
    while len(new_bots) < target_count:
        pa = rng.choice(elites)
        pb = rng.choice(elites)
        child_weights = crossover(pa.weights, pb.weights, seed=bot_num)
        child_weights = mutate(child_weights, mutation_rate, seed=bot_num + 1000)

        child = BotConfig(
            bot_id=f"bot_{bot_num:03d}",
            name=f"bot_{bot_num:03d}_gen{gen}",
            weights=child_weights,
            generation=gen,
            parent_ids=[pa.bot_id, pb.bot_id],
            tags=_generate_tags(child_weights),
        )
        new_bots.append(child)
        bot_num += 1

    return new_bots[:target_count]
