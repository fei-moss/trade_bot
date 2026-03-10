"""
LLM 参数调节层

慢脑: 理解用户意图 + 感知行情上下文 → 输出 DecisionParams
快脑: decision.py 的 compute_signals 每根K线自动执行

调用时机:
  1. 初始化: 用户描述策略风格 → 生成初始参数
  2. Regime 变化: 行情切换时 LLM 微调参数
  3. 定期: 每 N 根K线复盘一次
"""

import os
import json
from typing import Optional

from openai import OpenAI

from src.strategy.decision import DecisionParams


DEFAULT_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4.6")

SYSTEM_PROMPT = """你是一个专业的量化交易策略调参师。

用户会用自然语言描述他想要的交易风格，你需要输出一组连续参数来配置加权决策函数。

## 决策函数原理

交易信号 = trend_weight×趋势信号 + momentum_weight×动量信号 + mean_revert_weight×均值回归信号 + volume_weight×量能信号 + volatility_weight×波动率信号

当信号 > entry_threshold → 开多
当信号 < -entry_threshold → 开空

## 参数说明

### 信号权重（5个，会自动归一化到和=1）
- trend_weight: 趋势跟踪权重。高=重视EMA交叉和Supertrend方向
- momentum_weight: 动量权重。高=重视RSI和MACD信号
- mean_revert_weight: 均值回归权重。高=重视布林带回归
- volume_weight: 量能权重。高=重视OBV和量价配合
- volatility_weight: 波动率权重。高=重视ATR突破/收缩

### 交易阈值（重要：综合信号实际范围约 -0.5 ~ +0.5）
- entry_threshold (0.05~0.55): 越低越容易触发交易。实践中0.15=激进, 0.25=中性, 0.40=保守, >0.5几乎不会触发
- exit_threshold (0.03~0.30): 持仓时反向信号超过此值平仓

### 方向偏好
- long_bias (0~1): 0=只做空, 0.5=双向, 1=只做多

### 技术参数
- fast_ma_period (5~50): 快速均线周期
- slow_ma_period (20~200): 慢速均线周期
- trend_strength_min (10~50): ADX趋势强度阈值
- supertrend_mult (1~5): Supertrend 倍数
- rsi_period (7~28): RSI 周期
- rsi_overbought (60~85): RSI 超买线
- rsi_oversold (15~40): RSI 超卖线
- bb_period (10~50): 布林带周期
- bb_std (1.0~3.0): 布林带标准差倍数

### 杠杆 & 仓位
- base_leverage (1~200): 基础杠杆倍数
- max_leverage (1~200): 最大杠杆
- risk_per_trade (0.01~1.0): 每笔交易使用资金比例
- max_position_pct (0.05~1.0): 单笔最大资金占比

### 止损 / 止盈（重要：杠杆与止损的关系）
- sl_atr_mult (0.5~5.0): 止损距离 = X * ATR
  ⚠ 高杠杆必须配宽止损！杠杆×ATR百分比≈单笔最大亏损%
  例: 20x杠杆 + 1ATR(≈1.5%) → 单笔亏30%。建议: 5x配1ATR, 10x配2ATR, 20x配2.5-3ATR, 50x配3-5ATR
- tp_rr_ratio (1.0~10.0): 止盈/止损距离比 (风险回报比)
- trailing_enabled (true/false): 是否启用移动止损
- trailing_activation_pct (0.01~0.1): 浮盈X%后激活移动止损
- trailing_distance_atr (0.5~3.0): 移动止损距最高点 X * ATR

### 滚仓（趋势策略的利润放大器）
- rolling_enabled (true/false): 是否启用滚仓（用浮盈加仓）
  ⚠ 趋势策略强烈建议开启！没有滚仓时盈亏对称（赢25%/亏25%），50%胜率=不赚钱
  开启滚仓后，赢单可达+100%以上（滚仓放大），打破盈亏对称
- rolling_trigger_pct (0.1~0.8): 浮盈X%时触发滚仓
- rolling_reinvest_pct (0.3~1.0): 用浮盈的X%作为新仓保证金
- rolling_max_times (1~5): 最多滚仓次数
- rolling_move_stop (true/false): 滚仓后老仓止损移到成本价

### Regime 敏感度
- regime_sensitivity (0~1): 0=完全忽略行情阶段, 1=严格只在匹配行情交易
- exit_on_regime_change (true/false): 行情切换时是否立即平仓

## 输出格式

严格输出JSON，包含你调整的参数和理由:
```json
{
  "reasoning": "设计理由（1-3句）",
  "params": {
    "trend_weight": 0.5,
    "momentum_weight": 0.3,
    ...所有你要调整的参数...
  }
}
```

只需要输出你想要修改的参数，未提及的保持默认值。"""


DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")


class LLMTuner:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL):
        api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise EnvironmentError("需要设置 LLM_API_KEY 或 OPENROUTER_API_KEY 环境变量")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def tune(
        self,
        user_prompt: str,
        market_context: str = "",
        current_params: Optional[DecisionParams] = None,
        recent_performance: str = "",
    ) -> tuple[DecisionParams, str]:
        """
        LLM 根据用户意图 + 行情上下文调整参数。

        Returns:
            (adjusted_params, reasoning)
        """
        user_msg = f"## 用户需求\n{user_prompt}\n"

        if market_context:
            user_msg += f"\n## 当前行情上下文\n{market_context}\n"

        if current_params:
            user_msg += f"\n## 当前参数（可微调）\n```json\n{current_params.to_json()}\n```\n"

        if recent_performance:
            user_msg += f"\n## 近期表现\n{recent_performance}\n"

        user_msg += "\n请根据以上信息输出调整后的参数JSON。"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4096,
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        parsed = self._parse_response(raw)

        reasoning = parsed.get("reasoning", "")
        param_dict = parsed.get("params", {})

        if current_params:
            base = current_params.to_dict()
            base.update(param_dict)
            result = DecisionParams.from_dict(base)
        else:
            result = DecisionParams.from_dict(param_dict)

        result.normalize_weights()
        return result, reasoning

    def reflect(
        self,
        user_prompt: str,
        current_params: DecisionParams,
        trade_log: str,
        market_summary: str = "",
        generation: int = 0,
        cumulative_context: str = "",
        initial_params: Optional[DecisionParams] = None,
    ) -> tuple[DecisionParams, str]:
        """
        周期性反思进化：基于真实交易记录复盘调参。

        与 tune() 的区别：
        - tune 是"根据意图设计参数"
        - reflect 是"根据实战结果修正参数"

        改进 v2:
        - cumulative_context: 全局累计摘要（总收益/峰值/BTC总变化等）
        - initial_params: 初始参数快照，用于锚定核心性格、限制漂移
        """
        drift_warning = ""
        if initial_params:
            init = initial_params
            curr = current_params
            diffs = []
            if abs(curr.long_bias - init.long_bias) > 0.25:
                diffs.append(f"long_bias 从 {init.long_bias:.2f} 漂移到 {curr.long_bias:.2f}")
            if init.base_leverage > 0 and abs(curr.base_leverage / init.base_leverage - 1) > 0.5:
                diffs.append(f"base_leverage 从 {init.base_leverage:.0f}x 漂移到 {curr.base_leverage:.0f}x")
            if abs(curr.entry_threshold - init.entry_threshold) > 0.15:
                diffs.append(f"entry_threshold 从 {init.entry_threshold:.2f} 漂移到 {curr.entry_threshold:.2f}")
            if diffs:
                drift_warning = (
                    "\n⚠ **性格漂移警告**: 以下核心参数已大幅偏离初始设定:\n"
                    + "\n".join(f"  - {d}" for d in diffs)
                    + "\n请慎重考虑是否有必要继续偏离。短期回撤不应改变核心方向判断。\n"
                )

        msg = f"""## 反思任务（第 {generation} 轮进化）

你之前为用户设计了一套策略参数，现在要根据实战结果进行反思和改进。

## 用户的原始需求（这是策略的灵魂，不可背离）
{user_prompt}

## 当前参数
```json
{current_params.to_json()}
```

## 本周期的交易记录
{trade_log}

{f"## 本周期市场环境{chr(10)}{market_summary}" if market_summary else ""}

{f"## 全局累计表现（重要：不要只看本周期，要结合大局）{chr(10)}{cumulative_context}" if cumulative_context else ""}
{drift_warning}
## 反思原则

1. **先看大局再看细节** — 如果累计收益是正的，说明核心方向没错，本周期亏钱可能只是短期波动，不要过度反应
2. **分析哪些交易赚了、为什么** — 趋势判断对了？止损设得好？滚仓放大了？
3. **分析哪些交易亏了、为什么** — 止损太紧被扫？方向判错？入场阈值太低信号太多？
4. **找出参数的具体问题** — 不要泛泛而谈，要指出"sl_atr_mult=1.5太紧，应该加宽到2.0"
5. **微调而非重设** — 你是在优化，不是重新设计。单个参数调整幅度不超过 15%
6. **保持惯性** — 如果上一轮调参后效果还没充分体现，本轮应保持不变或仅微调

## ⚠ 禁止修改的参数（性格参数，已锁定）

以下参数定义了Bot的核心性格，你**不要输出**这些参数，即使你认为需要调整：
- long_bias（方向偏好）
- base_leverage / max_leverage（杠杆倍数）
- risk_per_trade / max_position_pct（仓位大小）
- rolling_*（滚仓配置）
- trend_weight / momentum_weight / mean_revert_weight / volume_weight / volatility_weight（信号权重）

## 允许调整的参数（战术参数）

只输出你要修改的战术参数：
- entry_threshold / exit_threshold（入场/出场阈值）
- sl_atr_mult（止损距离）
- tp_rr_ratio（止盈比）
- trailing_enabled / trailing_activation_pct / trailing_distance_atr（移动止损）
- regime_sensitivity / exit_on_regime_change（行情敏感度）
- fast_ma_period / slow_ma_period / rsi_period 等技术参数

输出调整后的参数和反思理由。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            max_tokens=4096,
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        parsed = self._parse_response(raw)

        reasoning = parsed.get("reasoning", "")
        param_dict = parsed.get("params", {})

        base = current_params.to_dict()
        base.update(param_dict)
        result = DecisionParams.from_dict(base)
        result.normalize_weights()
        return result, reasoning

    def _parse_response(self, raw: str) -> dict:
        """从 LLM 响应中提取 JSON。"""
        import re

        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

        print(f"  WARNING: 无法解析LLM响应: {raw[:200]}")
        return {}


def format_market_context(df, regime) -> str:
    """将行情数据格式化为LLM可读的上下文。"""
    from src.strategy.indicators import atr, adx, rsi, bollinger_bands
    from src.data.regime import regime_summary

    n = len(df)
    price = df["close"].iloc[-1]
    price_1d = df["close"].iloc[-24] if n > 24 else df["close"].iloc[0]
    price_7d = df["close"].iloc[-168] if n > 168 else df["close"].iloc[0]

    atr_val = atr(df, 14).iloc[-1]
    adx_val, _, _ = adx(df, 14)
    rsi_val = rsi(df, 14).iloc[-1]
    bb_u, bb_m, bb_l = bollinger_bands(df)

    summary = regime_summary(df, regime)
    regime_str = " | ".join(f"{r}: {s['pct']:.0%}" for r, s in summary.items())
    curr_regime = regime.iloc[-1]

    bb_pos = (price - bb_m.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1]) * 100 \
        if (bb_u.iloc[-1] - bb_l.iloc[-1]) > 0 else 0

    return (
        f"价格: ${price:,.0f} | 24h变化: {(price/price_1d-1)*100:+.1f}% | "
        f"7d变化: {(price/price_7d-1)*100:+.1f}%\n"
        f"ATR: ${atr_val:,.0f} ({atr_val/price*100:.2f}%) | "
        f"ADX: {adx_val.iloc[-1]:.0f} | RSI: {rsi_val:.0f}\n"
        f"布林带位置: {bb_pos:+.0f}% (0=中轨, ±100=触轨)\n"
        f"当前Regime: {curr_regime} | 分布: {regime_str}\n"
        f"数据量: {n} 根K线"
    )
