"""
LLM策略生成器

让AI分析真实行情数据，生成多样化的交易策略配置。
通过预定义的"策略维度矩阵"确保生成的策略足够旗帜鲜明。
"""

import json
import random
from typing import Optional
from anthropic import Anthropic

from src.strategy.schema import (
    BotStrategy, Archetype, Aggressiveness, Direction, Timeframe,
    PositionSizing, IndicatorType, STRATEGY_DIMENSIONS,
    AGGRESSIVENESS_CONSTRAINTS,
)


# ============ 策略维度矩阵 ============
# 100个bot = 10种原型 × 5种激进度 × 2种方向偏好，再加额外变体

DIVERSITY_MATRIX = []
for archetype in Archetype:
    for aggr in Aggressiveness:
        for direction in [Direction.LONG_ONLY, Direction.BOTH]:
            DIVERSITY_MATRIX.append({
                "archetype": archetype.value,
                "aggressiveness": aggr.value,
                "direction": direction.value,
            })

# 从矩阵中选100个组合（打乱后取前100）
random.shuffle(DIVERSITY_MATRIX)
DIVERSITY_MATRIX = DIVERSITY_MATRIX[:100]


# ============ Prompt模板 ============

SYSTEM_PROMPT = """你是一个专业的量化交易策略设计师。你的任务是根据真实行情数据和指定的策略"性格"，
设计出一个旗帜鲜明的交易策略。

你必须输出严格的JSON格式策略配置，不要输出任何其他内容。

可用的技术指标:
- ema_cross: EMA交叉 (params: fast, slow)
- sma_cross: SMA交叉 (params: fast, slow)
- ema: 单EMA (params: period)
- sma: 单SMA (params: period)
- rsi: RSI (params: period)
- macd: MACD (params: fast, slow, signal)
- bollinger: 布林带 (params: period, std_dev)
- atr: ATR (params: period)
- adx: ADX (params: period)
- stochastic: 随机指标 (params: k_period, d_period)
- supertrend: 超级趋势 (params: period, multiplier)
- volume_spike: 成交量突增 (params: period)
- price_breakout: 价格突破 (params: period)
- vwap: 成交量加权均价
- obv: 能量潮
- ichimoku: 一目均衡表 (params: tenkan, kijun, senkou_b)
- keltner: 肯特纳通道 (params: ema_period, atr_period, multiplier)
- support_resistance: 支撑阻力 (params: period)
- candle_pattern: 蜡烛图形态 (params: pattern=hammer|shooting_star|doji|bullish_engulf|bearish_engulf)
- donchian: 唐奇安通道 (params: period)
- williams_r: 威廉指标 (params: period)
- cci: CCI (params: period)
- mfi: MFI (params: period)

条件类型: above, below, cross_above, cross_below, between, squeeze, expansion, increasing, decreasing

仓位管理方式: fixed, kelly, martingale, anti_martingale, vol_scaled
"""


def build_generation_prompt(
    archetype: str,
    aggressiveness: str,
    direction: str,
    market_summary: str,
    bot_number: int,
) -> str:
    """构建策略生成prompt"""

    constraints = AGGRESSIVENESS_CONSTRAINTS.get(aggressiveness, {})

    archetype_descriptions = {
        "trend_follower": "你是一个趋势追踪者。你相信'趋势是你的朋友'。在趋势形成时顺势进场，趋势消失时离场。你使用均线、ADX、Supertrend等趋势指标。",
        "mean_reverter": "你是一个均值回归者。你相信价格总会回到均值。在价格偏离均值过远时反向交易。你使用RSI、布林带、统计偏差等指标。",
        "breakout_hunter": "你是一个突破猎手。你等待价格突破关键位置（前高、前低、通道边界），然后果断进场。你使用唐奇安通道、布林带突破、价格突破等。",
        "scalper": "你是一个超短线刮头皮交易者。你追求快进快出，小利润高频率。你使用短周期指标，窄止盈止损，追求高胜率。",
        "swing_trader": "你是一个波段交易者。你在日线或4小时级别上寻找趋势中的回调入场点，持仓时间较长。你使用多时间周期确认。",
        "momentum_rider": "你是一个动量骑手。你追涨杀跌，在强势行情中加速买入。你使用MACD、RSI动量、成交量突增等。",
        "contrarian": "你是一个逆势交易者。你在大家恐慌时买入，在大家贪婪时卖出。你关注极端的RSI、价格与指标的背离、恐慌性成交量。",
        "volatility_player": "你是一个波动率玩家。你在布林带收窄(squeeze)时准备，扩张时进场。你用ATR、布林带宽度、Keltner通道来判断波动率状态。",
        "dca_accumulator": "你是一个定投积累者。你在价格下跌时分批买入，不追高。你使用均线偏离度、RSI超卖来决定加仓时机。",
        "grid_trader": "你是一个网格交易者。你在一定价格区间内设置多个买卖点，在震荡市中反复获利。你使用支撑阻力、布林带来确定网格范围。",
    }

    aggr_descriptions = {
        "ultra_conservative": "你极度保守。1倍杠杆，每次只用1-3%的资金，止损极窄。宁可错过也不冒险。",
        "conservative": "你比较保守。低杠杆(1-2倍)，小仓位(3-8%)，严格止损。追求稳定收益。",
        "moderate": "你风格适中。中等杠杆(3-5倍)，中等仓位(5-15%)，合理的止盈止损比。",
        "aggressive": "你很激进。高杠杆(10-20倍)，大仓位(15-30%)，宽止损追求大收益。愿意承受大回撤。可以考虑使用滚仓(rolling=true)来放大趋势中的收益。",
        "ultra_aggressive": "你极度激进。超高杠杆(20-50倍)，重仓(30-100%)。你的核心武器是滚仓：用浮盈当保证金继续加仓，在单边行情中实现指数级收益。你清楚这意味着一次回调就可能爆仓，但你追求的就是'要么暴富要么归零'。你必须使用rolling=true。",
    }

    return f"""请根据以下行情数据和策略性格，设计一个交易策略。

## 行情数据摘要
{market_summary}

## 策略性格要求

**原型**: {archetype}
{archetype_descriptions.get(archetype, "")}

**激进程度**: {aggressiveness}
{aggr_descriptions.get(aggressiveness, "")}

**方向**: {direction}

**参数约束**:
- 杠杆范围: {constraints.get("leverage", (1, 50))}
- 仓位范围: {constraints.get("max_position_pct", (0.01, 1.0))}
- 止损范围: {constraints.get("stop_loss_pct", (0.005, 0.5))}
- 止盈范围: {constraints.get("take_profit_pct", (0.005, 2.0))}
- 最大回撤: {constraints.get("max_drawdown_pct", (0.05, 0.95))}

## 要求
1. 策略必须旗帜鲜明，充分体现上述性格
2. 入场规则至少包含2个条件的组合
3. 出场规则必须完整（止盈+止损+至少一种其他退出方式）
4. 参数必须在约束范围内
5. 策略要能从上面的行情数据中获利（或在特定行情段获利）
6. 滚仓说明：rolling=true表示用浮盈当保证金继续开新仓，形成指数级收益/亏损。
   - 仅适合aggressive和ultra_aggressive级别
   - rolling_trigger_pct: 浮盈达到多少%时触发(如0.3=涨30%时滚)
   - rolling_reinvest_pct: 浮盈的多少%投入新仓(如0.8)
   - rolling_move_stop: 滚仓后老仓止损移到成本价
   - 保守/适中策略必须设rolling=false

请输出以下JSON格式（不要加```json标记，直接输出JSON）:
{{
    "bot_id": "bot_{bot_number:03d}_{archetype}_{aggressiveness}",
    "name": "策略中文名称",
    "personality": "一句话性格描述",
    "description": "详细描述这个策略的逻辑和适用场景",
    "archetype": "{archetype}",
    "aggressiveness": "{aggressiveness}",
    "market": {{
        "symbols": ["BTC/USDT"],
        "timeframe": "选择合适的timeframe",
        "secondary_timeframe": null
    }},
    "position": {{
        "direction": "{direction}",
        "leverage": 数字,
        "max_position_pct": 数字,
        "max_concurrent": 数字,
        "pyramiding": bool,
        "pyramiding_max": 数字,
        "position_sizing": "选择一种",
        "rolling": bool,
        "rolling_trigger_pct": 浮盈百分比达到多少触发滚仓(如0.3=30%浮盈),
        "rolling_reinvest_pct": 将多少比例浮盈投入新仓(如0.8=80%),
        "rolling_max_times": 最大滚仓次数(1-5),
        "rolling_move_stop": 滚仓时是否把老仓止损移到成本价(bool)
    }},
    "entry_rules": [
        {{
            "conditions": [
                {{
                    "indicator": "指标名",
                    "params": {{}},
                    "condition": "条件类型",
                    "value": null或数字,
                    "value2": null或数字
                }}
            ]
        }}
    ],
    "exit_rule": {{
        "stop_loss_pct": 数字,
        "take_profit_pct": 数字,
        "trailing_stop": bool,
        "trailing_stop_pct": 数字,
        "time_exit_bars": null或数字,
        "signal_exit": []
    }},
    "risk": {{
        "max_daily_loss_pct": 数字,
        "max_drawdown_pct": 数字,
        "cool_down_bars": 数字,
        "max_trades_per_day": 数字,
        "correlation_filter": false
    }},
    "tags": ["标签1", "标签2"]
}}"""


def summarize_market_data(df, regime=None) -> str:
    """
    将行情数据压缩成LLM可消化的摘要。
    """
    summary = []
    summary.append(f"交易对: 数据包含 {len(df)} 根K线")

    if "timestamp" in df.columns:
        summary.append(f"时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

    # 价格统计
    summary.append(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    summary.append(f"起始价: {df['close'].iloc[0]:.2f}, 结束价: {df['close'].iloc[-1]:.2f}")
    total_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    summary.append(f"总涨跌幅: {total_change:.2%}")

    # 波动率
    returns = df["close"].pct_change().dropna()
    summary.append(f"日均波动: {returns.std():.4f}")
    summary.append(f"最大单根涨幅: {returns.max():.2%}, 最大单根跌幅: {returns.min():.2%}")

    # 分段表现
    quarter = len(df) // 4
    for i, label in enumerate(["第1个月", "第2个月", "第3个月", "第4个月"]):
        start = i * quarter
        end = min((i + 1) * quarter, len(df) - 1)
        change = (df["close"].iloc[end] - df["close"].iloc[start]) / df["close"].iloc[start]
        vol = df["close"].iloc[start:end].pct_change().std()
        summary.append(f"{label}: 涨跌{change:.2%}, 波动率{vol:.4f}")

    # regime分布
    if regime is not None:
        for r in regime.unique():
            pct = (regime == r).mean()
            summary.append(f"行情类型 {r}: 占比{pct:.1%}")

    # 关键价格形态描述
    # 最近的几次大幅波动
    big_moves = returns[returns.abs() > returns.std() * 2]
    if len(big_moves) > 0:
        summary.append(f"大幅波动次数(>2σ): {len(big_moves)}次")

    return "\n".join(summary)


class StrategyGenerator:
    """LLM驱动的策略生成器"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_one(
        self,
        archetype: str,
        aggressiveness: str,
        direction: str,
        market_summary: str,
        bot_number: int,
    ) -> Optional[BotStrategy]:
        """生成单个策略"""
        prompt = build_generation_prompt(
            archetype, aggressiveness, direction, market_summary, bot_number
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()

            # 尝试解析JSON
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]

            data = json.loads(text)
            strategy = BotStrategy.from_dict(data)
            return strategy

        except json.JSONDecodeError as e:
            print(f"JSON解析失败 (bot_{bot_number}): {e}")
            return None
        except Exception as e:
            print(f"生成失败 (bot_{bot_number}): {e}")
            return None

    def generate_batch(
        self,
        market_data: dict,
        regime: Optional[dict] = None,
        count: int = 100,
    ) -> list[BotStrategy]:
        """
        批量生成策略。

        Args:
            market_data: {"symbol": DataFrame}
            regime: {"symbol": regime_series}
            count: 目标生成数量

        Returns:
            list of BotStrategy
        """
        # 生成行情摘要
        summaries = {}
        for symbol, df in market_data.items():
            r = regime.get(symbol) if regime else None
            summaries[symbol] = summarize_market_data(df, r)

        # 合并摘要
        combined_summary = "\n\n".join(
            f"### {symbol}\n{s}" for symbol, s in summaries.items()
        )

        # 确保多样性矩阵够用
        matrix = build_diversity_matrix(count)

        strategies = []
        for i, combo in enumerate(matrix):
            print(f"生成策略 {i + 1}/{count}: {combo['archetype']} / {combo['aggressiveness']}...")
            strategy = self.generate_one(
                archetype=combo["archetype"],
                aggressiveness=combo["aggressiveness"],
                direction=combo["direction"],
                market_summary=combined_summary,
                bot_number=i + 1,
            )
            if strategy:
                strategies.append(strategy)

        print(f"\n成功生成 {len(strategies)}/{count} 个策略")
        return strategies


def build_diversity_matrix(count: int = 100) -> list[dict]:
    """
    构建确保最大多样性的策略组合矩阵。

    策略 = 原型(10) × 激进度(5) × 方向(3) × 时间周期偏好
    从中均匀采样count个组合。
    """
    all_combos = []

    archetypes = [a.value for a in Archetype]
    aggressiveness_levels = [a.value for a in Aggressiveness]
    directions = [d.value for d in Direction]

    for arch in archetypes:
        for aggr in aggressiveness_levels:
            for dire in directions:
                all_combos.append({
                    "archetype": arch,
                    "aggressiveness": aggr,
                    "direction": dire,
                })

    # 打乱但保证每个原型至少有count/len(archetypes)个
    random.shuffle(all_combos)

    # 均匀采样：确保每个archetype至少出现 count // len(archetypes) 次
    selected = []
    per_archetype = count // len(archetypes)
    remainder = count % len(archetypes)

    for arch in archetypes:
        arch_combos = [c for c in all_combos if c["archetype"] == arch]
        take = per_archetype + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        # 确保激进度分布均匀
        selected_arch = []
        for aggr in aggressiveness_levels:
            aggr_combos = [c for c in arch_combos if c["aggressiveness"] == aggr]
            if aggr_combos:
                selected_arch.append(aggr_combos[0])
        # 补齐
        remaining = [c for c in arch_combos if c not in selected_arch]
        random.shuffle(remaining)
        selected_arch.extend(remaining[:max(0, take - len(selected_arch))])
        selected.extend(selected_arch[:take])

    return selected[:count]
