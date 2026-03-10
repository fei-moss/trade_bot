"""Trade Bot Verification Platform — FastAPI entry point."""

import sys
import os
import logging

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.database import engine, Base
from server.routers import verify, bots, leaderboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="Trade Bot Verification Platform",
    description="接收 OpenClaw Skill 上传的 Bot 回测结果，重跑验证，管理排行榜",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(verify.router)
app.include_router(bots.router)
app.include_router(leaderboard.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
