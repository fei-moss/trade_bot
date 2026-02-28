"""
策略设计器：用户自然语言 → DecisionParams + Bot人格卡

核心原则：不使用任何模板或原型，LLM 从第一性原理推导策略参数。
用户说什么就理解什么，生成完全独特的策略。

用法:
  designer = StrategyDesigner()
  bot = designer.design("帮我搞个稳一点的", market_context="...")
  # bot = {name, personality, params: DecisionParams, reasoning}
"""

import os
import json
from typing import Optional

from openai import OpenAI
from src.strategy.decision import DecisionParams

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

DESIGNER_PROMPT = """你是一个量化交易策略设计师。

用户会用**任意自然语言**描述他想要的交易风格——可能非常专业，也可能极其随意（如"帮我赚钱"、"稳一点"、"冲！"）。

你的任务是**理解用户真正想要什么**，然后从第一性原理推导出一组最适合的策略参数。

## 你的推理过程（必须在 reasoning 中体现）

1. **解读意图**：用户到底想要什么？保守？激进？什么方向？什么频率？
   - "稳" → 低杠杆、严止损、低频、小仓位
   - "冲/梭哈" → 高杠杆、大仓位、低阈值、滚仓
   - "赚钱就行" → 需要你结合行情判断最优策略
   - 提到人名（凉兮/利弗莫尔等）→ 理解其交易哲学本质，但不照搬
   - 提到具体指标（RSI/MACD）→ 调高对应权重

2. **结合行情**：如果提供了市场上下文，你必须考虑：
   - 下跌行情 → 偏空或谨慎做多
   - 上涨行情 → 偏多
   - 震荡行情 → 均值回归或降低频率
   - 用户没说方向时，你根据行情建议方向

3. **物理约束**（不可违反）：
   - 杠杆 × ATR% ≈ 单笔最大亏损%。高杠杆必须配宽止损：
     2-5x → 1-1.5 ATR, 10x → 2 ATR, 20x → 2.5-3 ATR, 50x+ → 3-5 ATR
   - 趋势策略开滚仓才能打破盈亏对称（否则50%胜率=不赚钱）
   - 极高入场阈值(>0.45)会导致几乎不开仓

4. **生成人格**：给 bot 一个独特的名字和性格描述（2-3句），让用户能直观理解这个 bot 的"性格"

## 决策函数参数（你要输出的东西）

交易信号 = Σ(weight_i × signal_i)，信号实际范围约 -0.5 ~ +0.5

### 信号权重（5个，自动归一化）
- trend_weight: 趋势跟踪（EMA交叉 + Supertrend）
- momentum_weight: 动量（RSI + MACD）
- mean_revert_weight: 均值回归（布林带）
- volume_weight: 量能（OBV + 量价配合）
- volatility_weight: 波动率（ATR突破/收缩）

### 核心参数
- entry_threshold (0.05~0.55): 0.15=激进, 0.25=中性, 0.40=保守
- long_bias (0~1): 0=只做空, 0.5=双向, 1=只做多

### 技术参数
- fast_ma_period (5~50), slow_ma_period (20~200)
- trend_strength_min (10~50): ADX阈值
- supertrend_mult (1~5)
- rsi_period (7~28), rsi_overbought (60~85), rsi_oversold (15~40)
- bb_period (10~50), bb_std (1.0~3.0)

### 杠杆 & 仓位
- base_leverage (1~200), max_leverage (1~200)
- risk_per_trade (0.01~1.0), max_position_pct (0.05~1.0)

### 止损 / 止盈
- sl_atr_mult (0.5~5.0), tp_rr_ratio (1.0~10.0)
- trailing_enabled, trailing_activation_pct (0.01~0.1), trailing_distance_atr (0.5~3.0)

### 滚仓
- rolling_enabled, rolling_trigger_pct (0.1~0.8), rolling_reinvest_pct (0.3~1.0)
- rolling_max_times (1~5), rolling_move_stop (true/false)

### Regime
- regime_sensitivity (0~1), exit_on_regime_change (true/false)

## 一些你可能不了解的币圈黑话/人物（不是模板，只是帮你理解用户在说什么）

用户可能提到币圈特有人物或黑话，帮你理解他们的*本质*：
- "凉兮"：币圈合约传奇，以极端高杠杆(100x+)闻名，从几万U滚到几千万U又归零多次，信奉重仓+滚仓+All-in
- "梭哈"：All-in，把全部资金投入一笔交易
- "滚仓"：用浮盈作为新仓保证金继续开仓，实现指数级收益
- "海龟"：海龟交易法，经典趋势跟踪系统，用ATR管理仓位和止损
- "马丁"：马丁格尔策略，亏损后加倍下注
- "网格"：在价格区间内等距挂单，赚取震荡差价
- "DCA/定投"：定期定额买入，不择时
- 用户说"赌狗"、"冲"、"梭"= 极端激进，100x+杠杆、大仓位
- 用户说"稳"、"保守"、"不要爆仓"= 低杠杆(2-5x)、小仓位(2-5%)

理解本质后，用你自己的推理设计参数，不要照搬任何固定模板。

## 输出格式

严格输出 JSON（注意：字符串中不要使用未转义的双引号，用单引号或省略）：
```json
{
  "name": "2-4字的bot名称",
  "personality": "2-3句话描述性格和交易哲学",
  "reasoning": "推理过程：解读意图→结合行情→参数决策",
  "params": {
    "trend_weight": 0.4,
    "momentum_weight": 0.3,
    ...全部你认为需要设置的参数...
  }
}
```"""


class StrategyDesigner:
    """从用户自然语言直接生成策略参数，无模板、无拟合。"""

    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise EnvironmentError("需要设置 OPENROUTER_API_KEY 环境变量")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def design(
        self,
        user_input: str,
        market_context: str = "",
    ) -> dict:
        """
        用户自然语言 → 完整策略。

        Returns:
            {
                "name": "bot名称",
                "personality": "性格描述",
                "reasoning": "推理过程",
                "params": DecisionParams,
            }
        """
        user_msg = f"## 用户说的话\n{user_input}\n"
        if market_context:
            user_msg += f"\n## 当前市场环境\n{market_context}\n"
        user_msg += "\n请理解用户意图，从第一性原理设计策略参数。"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DESIGNER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4096,
            temperature=0.5,
        )

        raw = response.choices[0].message.content
        parsed = self._parse(raw)

        param_dict = parsed.get("params", {})
        params = DecisionParams.from_dict(param_dict)
        params.normalize_weights()

        return {
            "name": parsed.get("name", "自定义Bot"),
            "personality": parsed.get("personality", ""),
            "reasoning": parsed.get("reasoning", ""),
            "params": params,
        }

    def design_batch(self, inputs: list[str], market_context: str = "") -> list[dict]:
        results = []
        for inp in inputs:
            try:
                r = self.design(inp, market_context)
                results.append(r)
                print(f"  ✓ {r['name']} ← \"{inp}\"")
            except Exception as e:
                print(f"  ✗ 设计失败: \"{inp}\" → {e}")
                results.append({
                    "name": inp[:4],
                    "personality": inp,
                    "reasoning": str(e),
                    "params": DecisionParams(),
                })
        return results

    def _parse(self, raw: str) -> dict:
        import re

        # 提取 JSON 块
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
        text = json_match.group(1) if json_match else raw

        # 尝试直接解析
        for candidate in [text, raw]:
            try:
                start = candidate.index("{")
                end = candidate.rindex("}") + 1
                return json.loads(candidate[start:end])
            except (ValueError, json.JSONDecodeError):
                pass

        # 修复常见 JSON 问题：中文引号、未转义引号
        for candidate in [text, raw]:
            try:
                start = candidate.index("{")
                end = candidate.rindex("}") + 1
                fixed = candidate[start:end]
                fixed = fixed.replace('\u201c', "'").replace('\u201d', "'")
                fixed = fixed.replace('\u2018', "'").replace('\u2019', "'")
                fixed = fixed.replace('\u300c', "'").replace('\u300d', "'")
                return json.loads(fixed)
            except (ValueError, json.JSONDecodeError):
                pass

        # 最后手段：逐字段正则提取
        result = {}
        for field in ["name", "personality", "reasoning"]:
            m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
            if m:
                result[field] = m.group(1)
        params_m = re.search(r'"params"\s*:\s*(\{[^}]+\})', raw, re.DOTALL)
        if params_m:
            try:
                result["params"] = json.loads(params_m.group(1))
            except json.JSONDecodeError:
                pass

        if result:
            return result

        print(f"  WARNING: 无法解析LLM响应: {raw[:200]}")
        return {"name": "自定义Bot", "personality": "", "reasoning": raw[:200], "params": {}}


# -- 保留旧名兼容 --
PromptExpander = StrategyDesigner


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    tests = [
        "保守一点帮我做多",
        "像凉兮一样干",
        "熊市做空赚钱",
        "我只会看RSI",
        "帮我赚钱就行",
        "冲！",
        "我想稳稳地每个月赚5%",
        "跌的时候做空涨的时候做多",
    ]
    if len(sys.argv) > 1:
        tests = [" ".join(sys.argv[1:])]

    designer = StrategyDesigner()
    for inp in tests:
        print(f"\n{'='*60}")
        print(f"  用户: \"{inp}\"")
        print(f"{'='*60}")
        bot = designer.design(inp)
        p = bot["params"]
        print(f"  名称: {bot['name']}")
        print(f"  性格: {bot['personality']}")
        print(f"  推理: {bot['reasoning'][:120]}...")
        dir_str = "做多" if p.long_bias > 0.7 else ("做空" if p.long_bias < 0.3 else "双向")
        print(f"  参数: {dir_str} | {p.base_leverage:.0f}x | "
              f"仓位{p.risk_per_trade*100:.0f}% | 阈值{p.entry_threshold:.2f} | "
              f"SL={p.sl_atr_mult:.1f}ATR | 滚仓={'开' if p.rolling_enabled else '关'}")
        print(f"  权重: T={p.trend_weight:.2f} M={p.momentum_weight:.2f} "
              f"R={p.mean_revert_weight:.2f} V={p.volume_weight:.2f} Vo={p.volatility_weight:.2f}")
