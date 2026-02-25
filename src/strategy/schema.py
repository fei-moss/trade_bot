"""
42维决策权重向量 Schema

每个Bot的完整DNA = 9大维度 × 42个参数。
权重向量决定Bot的"性格"，在回测中被翻译为具体的交易规则，
在实盘中作为LLM决策的上下文。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import hashlib


# ============ 参数空间定义 ============
# type: "discrete" = 从options中选一个, "continuous" = 从range中均匀采样

PARAM_SPACE = {
    # === 维度1: 体制感知 (3) ===
    "regime_focus": {
        "type": "discrete",
        "options": ["BULL_ONLY", "BEAR_ONLY", "SIDEWAYS_ONLY", "ALL_THREE"],
    },
    "regime_detector_version": {
        "type": "discrete",
        "options": ["v1", "v2", "v3"],
    },
    "regime_switch_freq": {
        "type": "discrete",
        "options": ["1h", "4h", "daily", "weekly"],
    },

    # === 维度2: 时间框架 (3) ===
    "primary_timeframe": {
        "type": "discrete",
        "options": ["1m", "5m", "15m", "1h", "4h", "1d"],
    },
    "secondary_timeframe": {
        "type": "discrete",
        "options": [None, "1h", "4h", "1d", "1w"],
    },
    "decision_freq": {
        "type": "discrete",
        "options": ["1min", "15min", "1h", "4h", "daily"],
    },

    # === 维度3: 技术指标组合 (5) ===
    "num_indicators": {
        "type": "discrete",
        "options": [2, 3, 4, 5],
    },
    "indicator_set": {
        "type": "discrete",
        "options": [
            "MA+RSI", "MACD+BB", "Supertrend+ADX", "Ichimoku+KDJ",
            "EMA+Volume", "Stochastic+CCI", "PurePriceAction", "Multi3",
        ],
    },
    "ma_period_range": {
        "type": "discrete",
        "options": ["5-20", "10-50", "20-100", "50-200", "100-500"],
    },
    "rsi_period": {
        "type": "discrete",
        "options": [7, 14, 21, 28],
    },
    "adx_period": {
        "type": "discrete",
        "options": [10, 14, 20, 28],
    },

    # === 维度4: 杠杆与仓位 (5) ===
    "base_leverage": {
        "type": "discrete",
        "options": [1, 5, 10, 20, 50, 100, 200],
    },
    "leverage_dynamic": {
        "type": "discrete",
        "options": ["Fixed", "RegimeScale", "VolatilityScale"],
    },
    "position_sizing": {
        "type": "discrete",
        "options": [
            "Fixed", "Kelly", "Martingale", "Anti-Martingale",
            "AllIn", "Volatility",
        ],
    },
    "max_position_per_trade": {
        "type": "discrete",
        "options": [0.05, 0.10, 0.20, 0.50, 1.00],
    },
    "risk_per_trade": {
        "type": "continuous",
        "range": (0.01, 0.10),
    },

    # === 维度5: 进出场规则 (5) ===
    "entry_condition": {
        "type": "discrete",
        "options": [
            "Strict", "Loose", "MomentumBreak",
            "MeanReversion", "RegimeConfirmed",
        ],
    },
    "sl_type": {
        "type": "discrete",
        "options": [
            "None", "Fixed", "ATR", "Trailing",
            "RegimeAdaptive", "MaxDD",
        ],
    },
    "tp_type": {
        "type": "discrete",
        "options": ["RR2", "RR3", "RR5", "Trailing", "None"],
    },
    "exit_on_regime_change": {
        "type": "discrete",
        "options": ["Yes", "No", "OnlyIfLoss"],
    },
    "entry_confirmation": {
        "type": "discrete",
        "options": ["SingleBar", "MultiBar", "VolumeSpike"],
    },

    # === 维度6: 进化与回测 (6) ===
    "backtest_years": {
        "type": "discrete",
        "options": [1, 2, 3, 5],
    },
    "backtest_granularity": {
        "type": "discrete",
        "options": ["1h", "4h", "1d", "Mixed"],
    },
    "evolution_style": {
        "type": "discrete",
        "options": [
            "Genetic", "LLM_SelfEdit", "RandomMutate",
            "RegimeSpecialist", "CommitteeVote",
        ],
    },
    "evolution_frequency": {
        "type": "discrete",
        "options": ["weekly", "every_3d", "daily"],
    },
    "fitness_metric": {
        "type": "discrete",
        "options": [
            "TotalReturn", "Sharpe", "Calmar",
            "WinRate", "ProfitFactor", "Custom",
        ],
    },
    "exploration_rate": {
        "type": "continuous",
        "range": (0.05, 0.30),
    },

    # === 维度7: 数据源权重 (5) — 归一化 ===
    "price_weight": {
        "type": "continuous",
        "range": (0.0, 1.0),
        "normalize_group": "data_source",
    },
    "volume_weight": {
        "type": "continuous",
        "range": (0.0, 1.0),
        "normalize_group": "data_source",
    },
    "funding_rate_weight": {
        "type": "continuous",
        "range": (0.0, 1.0),
        "normalize_group": "data_source",
    },
    "onchain_weight": {
        "type": "continuous",
        "range": (0.0, 1.0),
        "normalize_group": "data_source",
    },
    "meme_sentiment_weight": {
        "type": "continuous",
        "range": (0.0, 1.0),
        "normalize_group": "data_source",
    },

    # === 维度8: 决策风格 (5) ===
    "reasoning_depth": {
        "type": "discrete",
        "options": ["Short", "Medium", "Long", "Extreme"],
    },
    "temperature": {
        "type": "continuous",
        "range": (0.1, 1.3),
    },
    "creativity_bias": {
        "type": "discrete",
        "options": ["Conservative", "Balanced", "Chaotic", "YOLO"],
    },
    "sub_agent_count": {
        "type": "discrete",
        "options": [0, 1, 2, 3],
    },
    "bias_towards_action": {
        "type": "continuous",
        "range": (0.2, 0.8),
    },

    # === 维度9: 极端化开关 (5) ===
    "allow_blowup": {
        "type": "discrete",
        "options": ["Yes", "No", "OnlyInBull"],
    },
    "reverse_logic_prob": {
        "type": "continuous",
        "range": (0.0, 0.3),
    },
    "max_dd_tolerance": {
        "type": "continuous",
        "range": (0.1, 1.0),
    },
    "yolo_mode": {
        "type": "discrete",
        "options": ["Off", "Low", "Medium", "High"],
    },
    "regime_override_prob": {
        "type": "continuous",
        "range": (0.0, 0.2),
    },
}

NORMALIZE_GROUPS = {
    "data_source": [
        "price_weight", "volume_weight", "funding_rate_weight",
        "onchain_weight", "meme_sentiment_weight",
    ],
}

DIMENSION_NAMES = [
    "体制感知", "时间框架", "技术指标组合", "杠杆与仓位",
    "进出场规则", "进化与回测", "数据源权重", "决策风格", "极端化开关",
]


# ============ 工具函数 ============

def parse_ma_range(ma_period_range: str) -> tuple[int, int]:
    """解析MA周期范围字符串 → (fast_period, slow_period)"""
    parts = ma_period_range.split("-")
    return int(parts[0]), int(parts[1])


# ============ 权重向量 ============

@dataclass
class WeightVector:
    """42维决策权重向量 — Bot的DNA"""

    # 维度1: 体制感知 (3)
    regime_focus: str = "ALL_THREE"
    regime_detector_version: str = "v1"
    regime_switch_freq: str = "4h"

    # 维度2: 时间框架 (3)
    primary_timeframe: str = "1h"
    secondary_timeframe: Optional[str] = None
    decision_freq: str = "1h"

    # 维度3: 技术指标组合 (5)
    num_indicators: int = 3
    indicator_set: str = "MA+RSI"
    ma_period_range: str = "10-50"
    rsi_period: int = 14
    adx_period: int = 14

    # 维度4: 杠杆与仓位 (5)
    base_leverage: int = 5
    leverage_dynamic: str = "Fixed"
    position_sizing: str = "Fixed"
    max_position_per_trade: float = 0.10
    risk_per_trade: float = 0.02

    # 维度5: 进出场规则 (5)
    entry_condition: str = "Strict"
    sl_type: str = "Fixed"
    tp_type: str = "RR2"
    exit_on_regime_change: str = "No"
    entry_confirmation: str = "SingleBar"

    # 维度6: 进化与回测 (6)
    backtest_years: int = 1
    backtest_granularity: str = "1h"
    evolution_style: str = "Genetic"
    evolution_frequency: str = "weekly"
    fitness_metric: str = "Sharpe"
    exploration_rate: float = 0.10

    # 维度7: 数据源权重 (5)
    price_weight: float = 0.50
    volume_weight: float = 0.30
    funding_rate_weight: float = 0.10
    onchain_weight: float = 0.05
    meme_sentiment_weight: float = 0.05

    # 维度8: 决策风格 (5)
    reasoning_depth: str = "Medium"
    temperature: float = 0.7
    creativity_bias: str = "Balanced"
    sub_agent_count: int = 0
    bias_towards_action: float = 0.5

    # 维度9: 极端化开关 (5)
    allow_blowup: str = "No"
    reverse_logic_prob: float = 0.0
    max_dd_tolerance: float = 0.30
    yolo_mode: str = "Off"
    regime_override_prob: float = 0.0

    def fingerprint(self) -> str:
        """生成唯一指纹（用于去重）"""
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "WeightVector":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def get_effective_leverage(self, regime: str, volatility_rank: float) -> int:
        """根据动态策略计算实际杠杆"""
        lev = self.base_leverage
        yolo_mult = {"Off": 1.0, "Low": 1.5, "Medium": 2.5, "High": 5.0}
        lev = int(lev * yolo_mult.get(self.yolo_mode, 1.0))

        if self.leverage_dynamic == "RegimeScale":
            favorable = (
                (regime == "BULL" and self.regime_focus in ("BULL_ONLY", "ALL_THREE"))
                or (regime == "BEAR" and self.regime_focus in ("BEAR_ONLY", "ALL_THREE"))
                or (regime == "SIDEWAYS" and self.regime_focus in ("SIDEWAYS_ONLY", "ALL_THREE"))
            )
            lev = int(lev * (1.5 if favorable else 0.5))
        elif self.leverage_dynamic == "VolatilityScale":
            if volatility_rank < 0.3:
                lev = int(lev * 1.5)
            elif volatility_rank > 0.7:
                lev = int(lev * 0.5)

        return max(1, min(lev, 200))

    def get_sl_pct(self, atr_pct: float, regime: str) -> float:
        """根据SL类型计算止损百分比（价格方向）"""
        if self.sl_type == "None":
            return 0.999
        elif self.sl_type == "Fixed":
            return self.risk_per_trade
        elif self.sl_type == "ATR":
            return max(atr_pct * 2, 0.005)
        elif self.sl_type == "Trailing":
            return max(atr_pct * 1.5, 0.01)
        elif self.sl_type == "RegimeAdaptive":
            if regime == "SIDEWAYS":
                return 0.01
            elif regime in ("BULL", "BEAR"):
                return 0.03
            return 0.02
        elif self.sl_type == "MaxDD":
            return self.max_dd_tolerance
        return 0.02

    def get_tp_pct(self, sl_pct: float) -> Optional[float]:
        """根据TP类型计算止盈百分比"""
        if self.tp_type == "None":
            return None
        elif self.tp_type == "Trailing":
            return None
        elif self.tp_type == "RR2":
            return sl_pct * 2
        elif self.tp_type == "RR3":
            return sl_pct * 3
        elif self.tp_type == "RR5":
            return sl_pct * 5
        return sl_pct * 2

    def should_trade_regime(self, regime: str) -> bool:
        """检查当前regime是否在交易范围内"""
        if self.regime_focus == "ALL_THREE":
            return True
        mapping = {
            "BULL_ONLY": "BULL",
            "BEAR_ONLY": "BEAR",
            "SIDEWAYS_ONLY": "SIDEWAYS",
        }
        return regime == mapping.get(self.regime_focus, "")


@dataclass
class BotConfig:
    """Bot完整配置 = 权重向量 + 元数据"""

    bot_id: str
    name: str
    weights: WeightVector = field(default_factory=WeightVector)
    backtest_summary: Optional[dict] = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    best_regime: list[str] = field(default_factory=list)
    worst_regime: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "name": self.name,
            "weights": self.weights.to_dict(),
            "backtest_summary": self.backtest_summary,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "tags": self.tags,
            "best_regime": self.best_regime,
            "worst_regime": self.worst_regime,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "BotConfig":
        weights = WeightVector.from_dict(data.get("weights", {}))
        return cls(
            bot_id=data["bot_id"],
            name=data.get("name", ""),
            weights=weights,
            backtest_summary=data.get("backtest_summary"),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            tags=data.get("tags", []),
            best_regime=data.get("best_regime", []),
            worst_regime=data.get("worst_regime", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BotConfig":
        return cls.from_dict(json.loads(json_str))

    def save(self, directory: str):
        """保存Bot配置到文件"""
        import os
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.bot_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path
