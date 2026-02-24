"""
多样性评分与筛选

确保最终选出的100个bot之间差异足够大。
使用回测收益的相关性矩阵来度量策略间的相似度。
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Optional

from src.strategy.schema import BotStrategy
from src.backtest.engine import BacktestResult


def compute_return_correlation(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """
    计算各策略收益曲线之间的相关性矩阵。
    相关性越低 = 多样性越高。
    """
    equity_dict = {}
    max_len = 0
    for bot_id, result in results.items():
        eq = result.equity_curve.pct_change().dropna()
        equity_dict[bot_id] = eq
        max_len = max(max_len, len(eq))

    # 对齐长度
    aligned = {}
    for bot_id, eq in equity_dict.items():
        if len(eq) < max_len:
            eq = pd.concat([eq, pd.Series([0.0] * (max_len - len(eq)))]).reset_index(drop=True)
        else:
            eq = eq.iloc[:max_len].reset_index(drop=True)
        aligned[bot_id] = eq

    df = pd.DataFrame(aligned)
    return df.corr()


def diversity_score(corr_matrix: pd.DataFrame) -> dict[str, float]:
    """
    为每个策略计算多样性分数。
    分数 = 1 - 与其他策略的平均绝对相关性。
    分数越高 = 越独特。
    """
    scores = {}
    for bot_id in corr_matrix.columns:
        others = corr_matrix[bot_id].drop(bot_id)
        scores[bot_id] = 1 - others.abs().mean()
    return scores


def cluster_strategies(
    corr_matrix: pd.DataFrame,
    n_clusters: int = 20,
) -> dict[str, int]:
    """
    对策略做层次聚类，识别相似策略群组。

    Returns:
        dict[bot_id -> cluster_id]
    """
    # 将相关性转为距离
    dist = 1 - corr_matrix.abs()
    np.fill_diagonal(dist.values, 0)

    # 确保对称且非负
    dist = (dist + dist.T) / 2
    dist = dist.clip(lower=0)

    condensed = squareform(dist.values)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return {bot_id: int(label) for bot_id, label in zip(corr_matrix.columns, labels)}


def select_diverse_subset(
    strategies: list[BotStrategy],
    results: dict[str, BacktestResult],
    target_count: int = 100,
    min_trades: int = 5,
) -> list[BotStrategy]:
    """
    从候选策略中选出最多样化的子集。

    算法:
    1. 过滤掉交易次数过少的策略
    2. 计算收益相关性矩阵
    3. 聚类
    4. 从每个聚类中选表现最极端的（最好+最差）

    为什么也选最差的：因为目标是覆盖所有行情，
    在某行情中表现最差的策略，在另一行情中可能表现最好。
    """
    # 过滤
    valid = [s for s in strategies if s.bot_id in results and results[s.bot_id].total_trades >= min_trades]
    if len(valid) <= target_count:
        return valid

    valid_results = {s.bot_id: results[s.bot_id] for s in valid}

    # 计算相关性
    corr = compute_return_correlation(valid_results)

    # 聚类
    n_clusters = min(target_count // 2, len(valid) // 3)
    clusters = cluster_strategies(corr, n_clusters=max(n_clusters, 5))

    # 从每个聚类中选代表
    selected_ids = set()
    cluster_groups = {}
    for bot_id, cluster_id in clusters.items():
        cluster_groups.setdefault(cluster_id, []).append(bot_id)

    per_cluster = target_count // len(cluster_groups)

    for cluster_id, bot_ids in cluster_groups.items():
        # 按总收益排序
        sorted_bots = sorted(bot_ids, key=lambda x: valid_results[x].total_return)

        # 取最好和最差的
        take = max(per_cluster, 2)
        # 最好的一半
        best = sorted_bots[-(take // 2 + take % 2):]
        # 最差的一半
        worst = sorted_bots[:take // 2]
        selected_ids.update(best)
        selected_ids.update(worst)

    # 如果还不够，按多样性分数补齐
    if len(selected_ids) < target_count:
        scores = diversity_score(corr)
        remaining = [s for s in valid if s.bot_id not in selected_ids]
        remaining.sort(key=lambda s: scores.get(s.bot_id, 0), reverse=True)
        for s in remaining:
            if len(selected_ids) >= target_count:
                break
            selected_ids.add(s.bot_id)

    # 截断
    selected_ids = list(selected_ids)[:target_count]
    return [s for s in valid if s.bot_id in selected_ids]


def strategy_similarity_report(
    strategies: list[BotStrategy],
    results: dict[str, BacktestResult],
) -> dict:
    """
    生成策略多样性报告。
    """
    valid_results = {s.bot_id: results[s.bot_id] for s in strategies if s.bot_id in results}
    if len(valid_results) < 2:
        return {"error": "需要至少2个有效策略"}

    corr = compute_return_correlation(valid_results)
    scores = diversity_score(corr)
    clusters = cluster_strategies(corr, n_clusters=min(20, len(valid_results) // 2))

    # 统计
    returns = {bid: r.total_return for bid, r in valid_results.items()}
    sorted_by_return = sorted(returns.items(), key=lambda x: x[1])

    return {
        "total_strategies": len(valid_results),
        "avg_correlation": float(corr.values[np.triu_indices_from(corr.values, k=1)].mean()),
        "avg_diversity_score": float(np.mean(list(scores.values()))),
        "n_clusters": len(set(clusters.values())),
        "top_3_best": [(bid, round(ret, 4)) for bid, ret in sorted_by_return[-3:]],
        "top_3_worst": [(bid, round(ret, 4)) for bid, ret in sorted_by_return[:3]],
        "most_unique": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5],
        "most_similar_pair": _find_most_similar(corr),
    }


def _find_most_similar(corr: pd.DataFrame) -> tuple:
    """找到最相似的一对策略"""
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_masked = corr.where(mask)
    max_corr = corr_masked.max().max()
    for col in corr_masked.columns:
        for idx in corr_masked.index:
            if corr_masked.loc[idx, col] == max_corr:
                return (idx, col, round(float(max_corr), 4))
    return ("", "", 0.0)
