"""
策略配置Schema定义

定义了bot策略的所有可调维度，从50倍杠杆滚仓到1倍保守持仓。
每个bot的"性格"由这些参数唯一确定。
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json
import yaml


# ============ 枚举定义 ============

class Direction(str, Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"


class PositionSizing(str, Enum):
    FIXED = "fixed"                    # 固定仓位
    KELLY = "kelly"                    # 凯利公式
    MARTINGALE = "martingale"          # 亏损加倍
    ANTI_MARTINGALE = "anti_martingale"  # 盈利加仓
    VOLATILITY_SCALED = "vol_scaled"   # 波动率调整


class Archetype(str, Enum):
    """策略原型 - 定义bot的核心性格"""
    TREND_FOLLOWER = "trend_follower"       # 趋势追踪者
    MEAN_REVERTER = "mean_reverter"         # 均值回归者
    BREAKOUT_HUNTER = "breakout_hunter"     # 突破猎手
    SCALPER = "scalper"                     # 超短线刮头皮
    SWING_TRADER = "swing_trader"           # 波段交易者
    MOMENTUM_RIDER = "momentum_rider"       # 动量骑手
    CONTRARIAN = "contrarian"              # 逆势交易者
    VOLATILITY_PLAYER = "volatility_player" # 波动率玩家
    DCA_ACCUMULATOR = "dca_accumulator"     # 定投积累者
    GRID_TRADER = "grid_trader"            # 网格交易者


class Aggressiveness(str, Enum):
    """激进程度 - 从极端保守到极端激进"""
    ULTRA_CONSERVATIVE = "ultra_conservative"  # 1x杠杆, 1-2%仓位, 极窄止损
    CONSERVATIVE = "conservative"              # 1-2x, 5%仓位
    MODERATE = "moderate"                      # 3-5x, 10%仓位
    AGGRESSIVE = "aggressive"                  # 10-20x, 20%仓位
    ULTRA_AGGRESSIVE = "ultra_aggressive"      # 20-50x, 50%+仓位, 滚仓


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


# ============ 指标条件定义 ============

class IndicatorType(str, Enum):
    # 均线类
    EMA = "ema"
    SMA = "sma"
    EMA_CROSS = "ema_cross"
    SMA_CROSS = "sma_cross"
    # 震荡类
    RSI = "rsi"
    STOCHASTIC = "stochastic"
    MACD = "macd"
    # 波动类
    BOLLINGER = "bollinger"
    ATR = "atr"
    KELTNER = "keltner"
    # 趋势类
    ADX = "adx"
    SUPERTREND = "supertrend"
    ICHIMOKU = "ichimoku"
    # 量价类
    VOLUME_SPIKE = "volume_spike"
    VWAP = "vwap"
    OBV = "obv"
    # 价格行为
    PRICE_BREAKOUT = "price_breakout"
    SUPPORT_RESISTANCE = "support_resistance"
    CANDLE_PATTERN = "candle_pattern"


class ConditionOp(str, Enum):
    ABOVE = "above"
    BELOW = "below"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    BETWEEN = "between"
    SQUEEZE = "squeeze"         # 布林带收窄
    EXPANSION = "expansion"     # 布林带扩张
    INCREASING = "increasing"   # 值在上升
    DECREASING = "decreasing"   # 值在下降


# ============ 数据类定义 ============

@dataclass
class IndicatorCondition:
    """单个指标条件"""
    indicator: str          # IndicatorType value
    params: dict            # 指标参数 (period, fast, slow等)
    condition: str          # ConditionOp value
    value: Optional[float] = None  # 阈值 (如RSI > 70的70)
    value2: Optional[float] = None  # 第二阈值 (BETWEEN用)


@dataclass
class EntryRule:
    """入场规则 - conditions之间是AND关系"""
    conditions: list[IndicatorCondition] = field(default_factory=list)


@dataclass
class ExitRule:
    """出场规则"""
    stop_loss_pct: float = 0.02           # 止损百分比
    take_profit_pct: float = 0.06         # 止盈百分比
    trailing_stop: bool = False           # 移动止损
    trailing_stop_pct: float = 0.02       # 移动止损回撤
    time_exit_bars: Optional[int] = None  # N根K线后强制平仓
    signal_exit: list[IndicatorCondition] = field(default_factory=list)  # 信号平仓


@dataclass
class PositionConfig:
    """仓位配置"""
    direction: str = Direction.BOTH.value
    leverage: int = 1                     # 杠杆倍数 1-50
    max_position_pct: float = 0.10        # 单笔最大仓位占比
    max_concurrent: int = 1               # 最大同时持仓数
    pyramiding: bool = False              # 普通加仓（用原始资金）
    pyramiding_max: int = 3               # 最大加仓次数
    position_sizing: str = PositionSizing.FIXED.value

    # ---- 滚仓配置 ----
    rolling: bool = False                 # 是否启用滚仓（用浮盈当保证金开新仓）
    rolling_trigger_pct: float = 0.30     # 浮盈达到多少%触发滚仓（如30%浮盈时滚）
    rolling_reinvest_pct: float = 0.80    # 将多少比例的浮盈投入新仓（如80%浮盈）
    rolling_max_times: int = 3            # 最大滚仓次数（防止无限滚）
    rolling_move_stop: bool = True        # 滚仓时是否把老仓止损移到成本价（保本）


@dataclass
class RiskConfig:
    """风控配置"""
    max_daily_loss_pct: float = 0.10      # 日最大亏损
    max_drawdown_pct: float = 0.30        # 最大回撤
    cool_down_bars: int = 0               # 亏损后冷却K线数
    max_trades_per_day: int = 50          # 日最大交易次数
    correlation_filter: bool = False       # 相关性过滤(多币种时)


@dataclass
class MarketConfig:
    """市场配置"""
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = Timeframe.H1.value
    secondary_timeframe: Optional[str] = None  # 多时间周期确认


@dataclass
class BotStrategy:
    """完整的Bot策略配置 - 一个bot的全部"基因" """
    bot_id: str
    name: str
    personality: str                       # 一句话描述性格
    description: str                       # 详细描述
    archetype: str                         # Archetype value
    aggressiveness: str                    # Aggressiveness value

    market: MarketConfig = field(default_factory=MarketConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    entry_rules: list[EntryRule] = field(default_factory=list)  # OR关系
    exit_rule: ExitRule = field(default_factory=ExitRule)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # 元数据 (回测后填充)
    backtest_summary: Optional[dict] = None
    tags: list[str] = field(default_factory=list)
    best_regime: list[str] = field(default_factory=list)
    worst_regime: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为字典"""
        return asdict(self)

    def to_json(self, indent=2) -> str:
        """序列化为JSON"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """序列化为YAML"""
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)

    @classmethod
    def from_dict(cls, data: dict) -> "BotStrategy":
        """从字典构建"""
        # 重建嵌套对象
        market = MarketConfig(**data.get("market", {}))
        position = PositionConfig(**data.get("position", {}))

        entry_rules = []
        for er in data.get("entry_rules", []):
            conditions = [IndicatorCondition(**c) for c in er.get("conditions", [])]
            entry_rules.append(EntryRule(conditions=conditions))

        exit_data = data.get("exit_rule", {})
        signal_exit = [IndicatorCondition(**c) for c in exit_data.pop("signal_exit", [])]
        exit_rule = ExitRule(**exit_data, signal_exit=signal_exit)

        risk = RiskConfig(**data.get("risk", {}))

        return cls(
            bot_id=data["bot_id"],
            name=data["name"],
            personality=data.get("personality", ""),
            description=data.get("description", ""),
            archetype=data.get("archetype", ""),
            aggressiveness=data.get("aggressiveness", ""),
            market=market,
            position=position,
            entry_rules=entry_rules,
            exit_rule=exit_rule,
            risk=risk,
            backtest_summary=data.get("backtest_summary"),
            tags=data.get("tags", []),
            best_regime=data.get("best_regime", []),
            worst_regime=data.get("worst_regime", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BotStrategy":
        """从JSON构建"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_yaml_file(cls, path: str) -> "BotStrategy":
        """从YAML文件构建"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save(self, directory: str, fmt: str = "json"):
        """保存策略配置到文件"""
        import os
        os.makedirs(directory, exist_ok=True)
        if fmt == "json":
            path = os.path.join(directory, f"{self.bot_id}.json")
            with open(path, "w") as f:
                f.write(self.to_json())
        elif fmt == "yaml":
            path = os.path.join(directory, f"{self.bot_id}.yaml")
            with open(path, "w") as f:
                f.write(self.to_yaml())
        return path


# ============ 策略维度空间定义 ============
# 用于LLM生成时约束和引导参数范围

STRATEGY_DIMENSIONS = {
    "archetype": [a.value for a in Archetype],
    "aggressiveness": [a.value for a in Aggressiveness],
    "direction": [d.value for d in Direction],
    "timeframe": [t.value for t in Timeframe],
    "position_sizing": [p.value for p in PositionSizing],
    "leverage_range": {"min": 1, "max": 50},
    "position_pct_range": {"min": 0.01, "max": 1.0},
    "stop_loss_range": {"min": 0.005, "max": 0.50},
    "take_profit_range": {"min": 0.005, "max": 2.0},
    "indicators": [i.value for i in IndicatorType],
    "conditions": [c.value for c in ConditionOp],
}

# 激进程度对应的参数约束
AGGRESSIVENESS_CONSTRAINTS = {
    "ultra_conservative": {
        "leverage": (1, 1),
        "max_position_pct": (0.01, 0.03),
        "stop_loss_pct": (0.005, 0.015),
        "take_profit_pct": (0.01, 0.03),
        "max_drawdown_pct": (0.05, 0.10),
    },
    "conservative": {
        "leverage": (1, 2),
        "max_position_pct": (0.03, 0.08),
        "stop_loss_pct": (0.01, 0.03),
        "take_profit_pct": (0.02, 0.06),
        "max_drawdown_pct": (0.10, 0.20),
    },
    "moderate": {
        "leverage": (3, 5),
        "max_position_pct": (0.05, 0.15),
        "stop_loss_pct": (0.02, 0.05),
        "take_profit_pct": (0.04, 0.12),
        "max_drawdown_pct": (0.20, 0.35),
    },
    "aggressive": {
        "leverage": (10, 20),
        "max_position_pct": (0.15, 0.30),
        "stop_loss_pct": (0.03, 0.10),
        "take_profit_pct": (0.08, 0.30),
        "max_drawdown_pct": (0.35, 0.60),
    },
    "ultra_aggressive": {
        "leverage": (20, 50),
        "max_position_pct": (0.30, 1.0),
        "stop_loss_pct": (0.05, 0.50),
        "take_profit_pct": (0.15, 2.0),
        "max_drawdown_pct": (0.50, 0.95),
    },
}
