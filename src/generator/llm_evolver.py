"""
LLM驱动的Bot进化引擎

通过OpenRouter调用LLM，分析当前一代Bot的回测表现，
理解成功/失败的模式，定向生成改进的下一代权重向量。

用法:
    evolver = LLMEvolver(model="anthropic/claude-sonnet-4-20250514")
    new_bots = evolver.evolve(bots, results, market_df, regime, count=5)
"""

import os
import re
import json
from datetime import datetime
from typing import Optional

from openai import OpenAI

from src.strategy.schema import (
    WeightVector, BotConfig, PARAM_SPACE, NORMALIZE_GROUPS,
)
from src.generator.weight_generator import (
    _normalize_groups, _enforce_consistency, _generate_tags,
    crossover, mutate,
)


DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _format_param_space() -> str:
    """将PARAM_SPACE格式化为LLM可读的参数说明。"""
    lines = []
    for name, spec in PARAM_SPACE.items():
        if spec["type"] == "discrete":
            opts = spec["options"]
            opts_str = ", ".join(str(o) for o in opts)
            lines.append(f"  {name}: discrete [{opts_str}]")
        else:
            lo, hi = spec["range"]
            lines.append(f"  {name}: continuous [{lo}, {hi}]")
    return "\n".join(lines)


def _format_bot_detail(bot: BotConfig, result_dict: dict) -> str:
    """格式化单个Bot的详情（权重+回测）供LLM阅读。"""
    w = bot.weights.to_dict()
    lines = [
        f"### {bot.bot_id} ({bot.name})",
        f"Generation: {bot.generation} | Tags: {bot.tags}",
        f"Backtest: return={result_dict['total_return']:.2%}, "
        f"sharpe={result_dict['sharpe_ratio']:.2f}, "
        f"MDD={result_dict['max_drawdown']:.2%}, "
        f"win_rate={result_dict['win_rate']:.0%}, "
        f"trades={result_dict['total_trades']}, "
        f"profit_factor={result_dict['profit_factor']:.2f}",
    ]

    regime_perf = result_dict.get("regime_performance", {})
    if regime_perf:
        parts = []
        for r, rp in regime_perf.items():
            parts.append(f"{r}: {rp.get('trades', 0)} trades, "
                         f"return={rp.get('return', 0):.2%}")
        lines.append(f"Regime表现: {' | '.join(parts)}")

    lines.append(f"Weights: {json.dumps(w, ensure_ascii=False)}")
    return "\n".join(lines)


SYSTEM_PROMPT = """你是一个量化交易策略进化引擎。你的任务是分析当前一代Bot的回测表现，找出成功和失败的模式，然后生成改进的下一代Bot权重向量。

关键规则:
1. 每个Bot由42维权重向量定义，每个参数有明确的合法值范围
2. discrete类型参数必须从给定选项中选择
3. continuous类型参数必须在给定范围内
4. 数据源权重(price_weight, volume_weight, funding_rate_weight, onchain_weight, meme_sentiment_weight)的总和必须=1
5. 生成的Bot要有多样性，不要全是同一风格

输出格式要求:
- 先输出一段分析（analysis字段）
- 然后输出一个JSON数组，每个元素包含:
  - "weights": 完整的42维权重字典
  - "reasoning": 这个Bot的设计理由（1-2句）

严格按以下格式输出，不要添加额外内容:
```json
{
  "analysis": "对当前一代的分析...",
  "bots": [
    {
      "weights": { ... 42个字段 ... },
      "reasoning": "设计理由"
    }
  ]
}
```"""


class LLMEvolver:
    """LLM驱动的进化引擎，通过OpenRouter调用任意模型。"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. "
                "Export it: export OPENROUTER_API_KEY=sk-or-..."
            )
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def evolve(
        self,
        bots: list[BotConfig],
        results: dict,
        market_df=None,
        regime=None,
        count: int = 5,
    ) -> tuple[list[BotConfig], str]:
        """
        LLM分析当前代Bot并生成下一代权重。

        Returns:
            (new_bots, analysis_text)
        """
        messages = self._build_prompt(bots, results, market_df, regime, count)

        print(f"  调用LLM ({self.model})...")
        raw_response = self._call_llm(messages)

        parsed = self._parse_response(raw_response)
        analysis = parsed.get("analysis", "")
        raw_bots = parsed.get("bots", [])

        if not raw_bots:
            print("  WARNING: LLM未返回有效Bot，使用遗传算法填充")
            return [], analysis

        current_gen = max((b.generation for b in bots), default=0)
        new_gen = current_gen + 1

        new_bots = []
        for i, raw in enumerate(raw_bots[:count]):
            weights_dict = raw.get("weights", {})
            reasoning = raw.get("reasoning", "")

            try:
                weights = self._validate_weights(weights_dict)
            except Exception as e:
                print(f"  WARNING: Bot {i+1} 权重校验失败: {e}, 跳过")
                continue

            bot_id = f"bot_llm_{new_gen}_{i+1:02d}"
            indicator = weights.indicator_set.replace("+", "_").lower()
            name = f"{bot_id}_{indicator}_{weights.entry_condition.lower()}"

            bot = BotConfig(
                bot_id=bot_id,
                name=name,
                weights=weights,
                generation=new_gen,
                parent_ids=[],
                tags=_generate_tags(weights) + ["llm_evolved"],
            )
            new_bots.append(bot)
            print(f"  LLM生成: {bot_id} | {reasoning[:60]}")

        print(f"  LLM成功生成 {len(new_bots)}/{count} 个Bot")
        return new_bots, analysis

    def _build_prompt(
        self,
        bots: list[BotConfig],
        results: dict,
        market_df,
        regime,
        count: int,
    ) -> list[dict]:
        """构建发给LLM的完整消息。"""
        scored = []
        for bot in bots:
            r = results.get(bot.bot_id)
            if r is None:
                continue
            result_dict = r if isinstance(r, dict) else r.to_dict()
            scored.append((bot, result_dict, result_dict.get("total_return", 0)))

        scored.sort(key=lambda x: x[2], reverse=True)

        returns = [s[2] for s in scored]
        avg_return = sum(returns) / len(returns) if returns else 0
        best_return = returns[0] if returns else 0
        worst_return = returns[-1] if returns else 0
        current_gen = max((b.generation for b in bots), default=0)

        # --- 行情环境 ---
        market_section = ""
        if market_df is not None:
            try:
                from src.data.formatter import enrich_dataframe, format_market_context
                enriched = enrich_dataframe(market_df)
                market_section = f"\n## 行情环境\n{format_market_context(enriched)}"
            except Exception:
                pass

        if regime is not None:
            try:
                from src.data.regime import regime_summary
                summary = regime_summary(market_df, regime)
                parts = []
                for r_name, s in summary.items():
                    parts.append(f"{r_name}: {s['pct']:.0%}")
                market_section += f"\nRegime分布: {' | '.join(parts)}"
            except Exception:
                pass

        # --- Top / Bottom Bots ---
        top_n = min(3, len(scored))
        bottom_n = min(3, len(scored))

        top_section = "\n## 表现最好的Bot (学习对象)\n"
        for bot, result_dict, _ in scored[:top_n]:
            top_section += _format_bot_detail(bot, result_dict) + "\n\n"

        bottom_section = "\n## 表现最差的Bot (避免重蹈覆辙)\n"
        for bot, result_dict, _ in scored[-bottom_n:]:
            bottom_section += _format_bot_detail(bot, result_dict) + "\n\n"

        user_msg = f"""## 当前代总览
- 代数: {current_gen}
- Bot数量: {len(scored)}
- 平均收益: {avg_return:.2%}
- 最高收益: {best_return:.2%}
- 最低收益: {worst_return:.2%}
{market_section}
{top_section}
{bottom_section}
## 42维参数空间定义
{_format_param_space()}

## 你的任务
生成 {count} 个新的42维权重向量。要求：
1. 至少2个继承Top Bot的优势基因，并针对弱点做改进
2. 至少1个专注于当前表现不好的regime
3. 至少1个实验性/逆向思维的Bot（探索新策略空间）
4. 保持多样性：不同的indicator_set、entry_condition、leverage等
5. 数据源权重(price_weight等5个)总和必须=1

请严格按系统消息中的JSON格式输出。"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _call_llm(self, messages: list[dict]) -> str:
        """调用OpenRouter API。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def _parse_response(self, raw: str) -> dict:
        """从LLM响应中提取JSON。支持markdown代码块和裸JSON。"""
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        json_str = code_block.group(1).strip() if code_block else raw.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        obj_match = re.search(r"\{[\s\S]*\}", json_str)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"  WARNING: 无法解析LLM响应为JSON")
        return {"analysis": raw[:500], "bots": []}

    def _validate_weights(self, raw: dict) -> WeightVector:
        """校验并修正LLM返回的权重字典，确保符合PARAM_SPACE。"""
        cleaned = {}

        for param_name, spec in PARAM_SPACE.items():
            val = raw.get(param_name)

            if spec["type"] == "discrete":
                options = spec["options"]
                if val in options:
                    cleaned[param_name] = val
                elif isinstance(val, str):
                    lower_map = {str(o).lower(): o for o in options}
                    cleaned[param_name] = lower_map.get(val.lower(), options[0])
                else:
                    cleaned[param_name] = options[0]

            elif spec["type"] == "continuous":
                lo, hi = spec["range"]
                try:
                    val = float(val)
                    cleaned[param_name] = round(max(lo, min(hi, val)), 4)
                except (TypeError, ValueError):
                    cleaned[param_name] = round((lo + hi) / 2, 4)

        _normalize_groups(cleaned)
        _enforce_consistency(cleaned)

        return WeightVector.from_dict(cleaned)


# ============ 混合进化 (Elite + LLM + Genetic) ============

def hybrid_evolve(
    bots: list[BotConfig],
    results: dict,
    market_df=None,
    regime=None,
    target_count: int = 10,
    elite_pct: float = 0.2,
    llm_count: int = 5,
    mutation_rate: float = 0.2,
    model: str = DEFAULT_MODEL,
    fitness_metric: str = "Sharpe",
) -> tuple[list[BotConfig], dict]:
    """
    混合进化：Elite保留 + LLM智能生成 + 遗传算法填充。

    Returns:
        (next_gen_bots, evolution_info)
    """
    import random

    scored = _score_bots(bots, results, fitness_metric)
    if not scored:
        raise ValueError("No bots with valid backtest results")

    current_gen = max((b.generation for b in bots), default=0)
    new_gen = current_gen + 1
    elite_count = max(int(target_count * elite_pct), 1)
    elites = [b for b, _ in scored[:elite_count]]

    next_gen = []
    for i, bot in enumerate(elites):
        elite_bot = BotConfig(
            bot_id=f"bot_{i+1:03d}",
            name=bot.name,
            weights=bot.weights,
            generation=new_gen,
            parent_ids=[bot.bot_id],
            tags=bot.tags + (["elite"] if "elite" not in bot.tags else []),
            best_regime=bot.best_regime,
            worst_regime=bot.worst_regime,
        )
        next_gen.append(elite_bot)

    llm_bots = []
    analysis = ""
    try:
        evolver = LLMEvolver(model=model)
        llm_bots, analysis = evolver.evolve(
            bots, results, market_df, regime, count=llm_count,
        )
    except EnvironmentError as e:
        print(f"  LLM不可用: {e}")
        print("  将用遗传算法替代LLM份额")
    except Exception as e:
        print(f"  LLM调用失败: {e}")
        print("  将用遗传算法替代LLM份额")

    for bot in llm_bots:
        if len(next_gen) >= target_count:
            break
        bot.bot_id = f"bot_{len(next_gen)+1:03d}"
        next_gen.append(bot)

    rng = random.Random(new_gen * 42)
    bot_num = len(next_gen) + 1
    while len(next_gen) < target_count:
        pa = rng.choice(elites)
        pb = rng.choice(elites)
        child_w = crossover(pa.weights, pb.weights, seed=bot_num)
        child_w = mutate(child_w, mutation_rate, seed=bot_num + 1000)

        child = BotConfig(
            bot_id=f"bot_{bot_num:03d}",
            name=f"bot_{bot_num:03d}_gen{new_gen}",
            weights=child_w,
            generation=new_gen,
            parent_ids=[pa.bot_id, pb.bot_id],
            tags=_generate_tags(child_w) + ["genetic"],
        )
        next_gen.append(child)
        bot_num += 1

    info = {
        "gen": new_gen,
        "parent_gen": current_gen,
        "timestamp": datetime.now().isoformat(),
        "model": model if llm_bots else None,
        "elite_preserved": len(elites),
        "llm_bots_generated": len(llm_bots),
        "genetic_filled": len(next_gen) - len(elites) - len(llm_bots),
        "total_bots": len(next_gen),
        "llm_analysis_summary": analysis[:500] if analysis else "",
        "fitness_metric": fitness_metric,
    }

    return next_gen[:target_count], info


def _score_bots(
    bots: list[BotConfig],
    results: dict,
    fitness_metric: str = "Sharpe",
) -> list[tuple[BotConfig, float]]:
    """按fitness排序Bot。"""
    scored = []
    for bot in bots:
        r = results.get(bot.bot_id)
        if r is None:
            continue

        rd = r if isinstance(r, dict) else r.to_dict()

        if fitness_metric == "TotalReturn":
            score = rd.get("total_return", 0)
        elif fitness_metric == "Sharpe":
            score = rd.get("sharpe_ratio", 0)
        elif fitness_metric == "Calmar":
            mdd = rd.get("max_drawdown", 0)
            score = rd.get("total_return", 0) / mdd if mdd > 0 else 0
        elif fitness_metric == "WinRate":
            score = rd.get("win_rate", 0)
        elif fitness_metric == "ProfitFactor":
            pf = rd.get("profit_factor", 0)
            score = min(pf, 10.0)
        else:
            score = rd.get("sharpe_ratio", 0)

        scored.append((bot, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ============ 进化日志 ============

def load_evolution_log(profile_dir: str = "profiles") -> dict:
    """加载进化日志。"""
    path = os.path.join(profile_dir, "evolution_log.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"generations": []}


def save_evolution_log(log: dict, profile_dir: str = "profiles"):
    """保存进化日志。"""
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, "evolution_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def append_generation_log(
    info: dict,
    results: dict,
    profile_dir: str = "profiles",
):
    """将本次进化的信息追加到日志。"""
    log = load_evolution_log(profile_dir)

    returns = []
    for r in results.values():
        rd = r if isinstance(r, dict) else r.to_dict()
        returns.append(rd.get("total_return", 0))

    info["best_return"] = max(returns) if returns else 0
    info["avg_return"] = sum(returns) / len(returns) if returns else 0
    info["worst_return"] = min(returns) if returns else 0

    log["generations"].append(info)
    save_evolution_log(log, profile_dir)
    return log


# ============ Profile 加载 ============

def find_latest_generation(base_dir: str = "profiles") -> tuple[str, int]:
    """
    扫描profiles目录，找到最新一代的路径。

    查找规则: gen_1/, gen_2/, ... 中编号最大的目录。
    如果没有gen_N目录，返回base_dir本身（即Gen 0）。

    Returns:
        (directory_path, generation_number)
    """
    if not os.path.isdir(base_dir):
        return base_dir, 0

    max_gen = -1
    max_dir = None

    for name in os.listdir(base_dir):
        if name.startswith("gen_") and os.path.isdir(os.path.join(base_dir, name)):
            try:
                gen_num = int(name.split("_")[1])
                if gen_num > max_gen:
                    index_check = os.path.join(base_dir, name, "_index.json")
                    if os.path.exists(index_check):
                        max_gen = gen_num
                        max_dir = os.path.join(base_dir, name)
            except (ValueError, IndexError):
                continue

    if max_dir:
        return max_dir, max_gen

    if os.path.exists(os.path.join(base_dir, "_index.json")):
        return base_dir, 0

    return base_dir, 0


def load_profiles(profile_dir: str = "profiles") -> tuple[list[BotConfig], dict]:
    """
    从profiles目录加载所有Bot配置和回测结果。

    Returns:
        (bots, results_dict) -- results_dict 的 value 是 dict 格式
    """
    bots = []
    results = {}

    index_path = os.path.join(profile_dir, "_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No _index.json found in {profile_dir}")

    for fname in sorted(os.listdir(profile_dir)):
        if not fname.startswith("bot_") or not fname.endswith(".json"):
            continue

        path = os.path.join(profile_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bot_data = data.get("bot", {})
        bot = BotConfig.from_dict(bot_data)
        bots.append(bot)

        if bot_data.get("backtest_summary"):
            results[bot.bot_id] = bot_data["backtest_summary"]

    print(f"从 {profile_dir} 加载了 {len(bots)} 个Bot, {len(results)} 个有回测结果")
    return bots, results
