from datetime import datetime
from pydantic import BaseModel, Field


# ── Upload Package ──

class DataFingerprint(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    exchange: str = "binance"
    start: str = Field(..., description="ISO 8601 起始时间")
    end: str = Field(..., description="ISO 8601 结束时间")
    bars: int
    first_close: float
    last_close: float
    checksum: str = Field("", description="sha256 of close prices, optional")


class BotCreate(BaseModel):
    name: str
    personality: str = ""
    params: dict
    evolution_config: dict | None = None


class SegmentResult(BaseModel):
    total_return: float
    total_trades: int
    win_rate: float


class EvolutionRound(BaseModel):
    round: int
    time_range: list[str] = Field(..., min_length=2, max_length=2, description="[start_iso, end_iso]")
    params_before: dict
    params_after: dict
    segment_result: SegmentResult


class BacktestResultSchema(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    blowup_count: int = 0


class TradeSubmit(BaseModel):
    entry_time: str
    exit_time: str | None = None
    direction: str
    entry_price: float
    exit_price: float | None = None
    pnl_pct: float
    leverage: float
    margin: float
    exit_reason: str = ""


class UploadPackage(BaseModel):
    version: str = "1.0"
    bot: BotCreate
    data_fingerprint: DataFingerprint
    backtest_result: BacktestResultSchema
    evolution_log: list[EvolutionRound] = []
    trades: list[TradeSubmit] = []


# ── Responses ──

class VerifyResponse(BaseModel):
    bot_id: str
    status: str
    match: bool
    verified_result: dict | None = None
    mismatch_details: dict | None = None


class BotSummary(BaseModel):
    id: str
    name: str
    personality: str
    status: str
    verified_return: float | None
    verified_sharpe: float | None
    verified_drawdown: float | None
    verified_trades: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class BotDetail(BotSummary):
    params: dict
    evolution_config: dict | None
    data_fingerprint: dict
    verified_blowup_count: int | None
    trades: list[TradeSubmit] = []

    model_config = {"from_attributes": True}


class LeaderboardEntry(BaseModel):
    rank: int
    id: str
    name: str
    personality: str
    verified_return: float
    verified_sharpe: float
    verified_drawdown: float
    verified_trades: int
    verified_blowup_count: int

    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list
