"""FastAPI 应用装配入口（骨架）。"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api import install_exception_handlers, router
from app.session import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("navsim")

app = FastAPI(
    title="Mobile Robot Navigation Simulation",
    version="0.1.0",
    description="本地移动机器人导航仿真：占用栅格 A* 规划 + 差分驱动纯追踪。",
)
app.state.session_store = SessionStore()
app.include_router(router, prefix="/api/v1")
install_exception_handlers(app)


@app.get("/")
async def read_root() -> dict:
    return {"service": "navsim", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "sessions": len(app.state.session_store.list_ids())}
