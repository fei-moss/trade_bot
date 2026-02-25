"""
多样性评分与筛选

双重多样性保障:
1. 权重向量空间的余弦相似度（确保DNA差异）
2. 回测收益曲线的相关性（确保行为差异）
"""

import numpy as np
import pandas as pd
from dataclasses import asdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, cosine

from src.strategy.schema import BotConfig, WeightVector, PARAM_SPACE
from src.backtest.engine import BacktestResult


def _weight_to_vector(w: WeightVector) -> np.ndarray:
    """将WeightVector转为数值向量（离散值用one-hot编码的索引）"""
    d = asdict(w)
    vec = []
    for param_name, spec in PARAM_SPACE.items():
        val = d.get(param_name)
        if spec["type"] == "continuous":
            lo, hi = spec["range"]
            normalized = (val - lo) / (hi - lo) if hi > lo else 0.5
            vec.append(normalized)
        elif spec["type"] == "discrete":
            options = spec["options"]
            idx = options.index(val) if val in options else 0
            normalized = idx / max(len(options) - 1, 1)
            vec.append(normalized)
    return np.array(vec)


def weight_similarity(a: WeightVector, b: WeightVector) -> float:
    """计算两个权重向量的余弦相似度 (0~1, 1=完全相同)"""
    va = _weight_to_vector(a)
    vb = _weight_to_vector(b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def compute_weight_distance_matrix(bots: list[BotConfig]) -> pd.DataFrame:
    """计算Bot间权重向量的距离矩阵"""
    vecs = {bot.bot_id: _weight_to_vector(bot.weights) for bot in bots}
    ids = list(vecs.keys())
    n = len(ids)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = cosine(vecs[ids[i]], vecs[ids[j]])
            dist[i, j] = d
            dist[j, i] = d

    return pd.DataFrame(dist, index=ids, columns=ids)


def compute_return_correlation(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """计算各策略收益曲线之间的相关性矩阵"""
    equity_dict = {}
    max_len = 0
    for bot_id, result in results.items():
        eq = result.equity_curve.pct_change().dropna()
        equity_dict[bot_id] = eq
        max_len = max(max_len, len(eq))

    aligned = {}
    for bot_id, eq in equity_dict.items():
        if len(eq) < max_len:
            eq = pd.concat([eq, pd.Series([0.0] * (max_len - len(eq)))]).reset_index(drop=True)
        else:
            eq = eq.iloc[:max_len].reset_index(drop=True)
        aligned[bot_id] = eq

    df = pd.DataFrame(aligned)
    return df.corr()


def diversity_score(bots: list[BotConfig], results: dict[str, BacktestResult]) -> dict[str, float]:
    """
    综合多样性分数 = 0.4 * 权重多样性 + 0.6 * 行为多样性
    分数越高 = 越独特
    """
    valid_ids = [b.bot_id for b in bots if b.bot_id in results]
    if len(valid_ids) < 2:
        return {bid: 1.0 for bid in valid_ids}

    valid_bots = [b for b in bots if b.bot_id in valid_ids]
    valid_results = {bid: results[bid] for bid in valid_ids}

    weight_dist = compute_weight_distance_matrix(valid_bots)
    return_corr = compute_return_correlation(valid_results)

    scores = {}
    for bot_id in valid_ids:
        if bot_id in weight_dist.index:
            w_score = weight_dist[bot_id].drop(bot_id, errors="ignore").mean()
        else:
            w_score = 0.5

        if bot_id in return_corr.columns:
            r_score = 1 - return_corr[bot_id].drop(bot_id, errors="ignore").abs().mean()
        else:
            r_score = 0.5

        scores[bot_id] = 0.4 * w_score + 0.6 * r_score

    return scores


def select_diverse_subset(
    bots: list[BotConfig],
    results: dict[str, BacktestResult],
    target_count: int = 100,
    min_trades: int = 5,
) -> list[BotConfig]:
    """
    从候选Bot中选出最多样化的子集。

    算法:
    1. 过滤交易次数过少的Bot
    2. 计算收益相关性 + 权重距离
    3. 聚类
    4. 每个聚类中选表现最好 + 最差的
    5. 按多样性分数补齐
    """
    valid = [b for b in bots if b.bot_id in results and results[b.bot_id].total_trades >= min_trades]
    if len(valid) <= target_count:
        return valid

    valid_results = {b.bot_id: results[b.bot_id] for b in valid}
    corr = compute_return_correlation(valid_results)

    n_clusters = min(target_count // 2, len(valid) // 3)
    n_clusters = max(n_clusters, 5)

    dist_arr = (1 - corr.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0)
    dist_arr = (dist_arr + dist_arr.T) / 2
    dist_arr = np.clip(dist_arr, 0, None)

    condensed = squareform(dist_arr)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    clusters = {bid: int(lbl) for bid, lbl in zip(corr.columns, labels)}

    selected_ids = set()
    cluster_groups = {}
    for bot_id, cluster_id in clusters.items():
        cluster_groups.setdefault(cluster_id, []).append(bot_id)

    per_cluster = max(target_count // len(cluster_groups), 2)

    for cluster_id, bot_ids in cluster_groups.items():
        sorted_bots = sorted(bot_ids, key=lambda x: valid_results[x].total_return)
        take = per_cluster
        best = sorted_bots[-(take // 2 + take % 2):]
        worst = sorted_bots[:take // 2]
        selected_ids.update(best)
        selected_ids.update(worst)

    if len(selected_ids) < target_count:
        scores = diversity_score(valid, results)
        remaining = [b for b in valid if b.bot_id not in selected_ids]
        remaining.sort(key=lambda b: scores.get(b.bot_id, 0), reverse=True)
        for b in remaining:
            if len(selected_ids) >= target_count:
                break
            selected_ids.add(b.bot_id)

    selected_ids = list(selected_ids)[:target_count]
    return [b for b in valid if b.bot_id in selected_ids]


def strategy_similarity_report(
    bots: list[BotConfig],
    results: dict[str, BacktestResult],
) -> dict:
    """生成策略多样性报告"""
    valid_results = {b.bot_id: results[b.bot_id] for b in bots if b.bot_id in results}
    if len(valid_results) < 2:
        return {"error": "需要至少2个有效策略"}

    corr = compute_return_correlation(valid_results)
    scores = diversity_score(bots, results)

    returns = {bid: r.total_return for bid, r in valid_results.items()}
    sorted_by_return = sorted(returns.items(), key=lambda x: x[1])

    triu_vals = corr.values[np.triu_indices_from(corr.values, k=1)]

    return {
        "total_bots": len(valid_results),
        "avg_correlation": float(triu_vals.mean()) if len(triu_vals) > 0 else 0,
        "avg_diversity_score": float(np.mean(list(scores.values()))),
        "n_clusters": len(set(_cluster_bots(corr).values())) if len(valid_results) >= 4 else 1,
        "top_3_best": [(bid, round(ret, 4)) for bid, ret in sorted_by_return[-3:]],
        "top_3_worst": [(bid, round(ret, 4)) for bid, ret in sorted_by_return[:3]],
        "most_unique": sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5],
    }


def _cluster_bots(corr: pd.DataFrame, n_clusters: int = 10) -> dict[str, int]:
    dist_arr = (1 - corr.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0)
    dist_arr = (dist_arr + dist_arr.T) / 2
    dist_arr = np.clip(dist_arr, 0, None)
    condensed = squareform(dist_arr)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return {bid: int(lbl) for bid, lbl in zip(corr.columns, labels)}
