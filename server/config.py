import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://trade_bot:trade_bot_dev@localhost:5432/trade_bot"
    DATABASE_URL_SYNC: str = "postgresql+psycopg2://trade_bot:trade_bot_dev@localhost:5432/trade_bot"

    VERIFY_TOLERANCE_PCT: float = 0.01
    VERIFY_SEGMENT_TOLERANCE_PCT: float = 0.02

    DEFAULT_EXCHANGE: str = "binance"
    REGIME_VERSION: str = "v1"
    REGIME_MIN_DURATION: int = 192

    class Config:
        env_file = ".env"


settings = Settings()
