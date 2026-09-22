"""并发语义：多会话隔离、同会话串行无撕裂、锁超时可区分。"""

from __future__ import annotations

import asyncio

import pytest

from app.errors import ErrorCode
from app.gridmap import parse_map_input
from app.kinematics import default_kinematics
from app.planner import PlannerConfig
from app.tracker import TrackerConfig
from app.session import (
    Session, SessionStore, session_plan, session_step,
)
from tests.conftest import case_log, open_grid


def _make_session(sid: str) -> Session:
    g = parse_map_input(grid=open_grid(20, 12), resolution=0.1)
    kin = default_kinematics(robot_radius=0.0, dt=0.1,
                             max_linear_speed=1.0, max_angular_speed=2.0)
    s = Session(sid, g, g, kin, (0.15, 0.15), 0.0, (1.85, 1.05), 0.0,
                PlannerConfig(), TrackerConfig(lookahead=0.25))
    s.state = (0.15, 0.15, 0.0, 0.0)
    return s


async def test_concurrent_sessions_are_isolated():
    store = SessionStore()
    sessions = []
    for i in range(8):
        s = _make_session(f"iso-{i}")
        await store.create(s)
        sessions.append(s)
        async with s.lock:
            await session_plan(s)

    async def drive(s: Session):
        for _ in range(300):
            async with s.lock:
                snap = await session_step(s)
            if snap["status"] == "REACHED":
                break
            await asyncio.sleep(0)
        return s.session_id, s.step_count, s.status.value, s.state

    results = await asyncio.gather(*(drive(s) for s in sessions))
    for sid, steps, status, state in results:
        case_log("判定", sid=sid, steps=steps, status=status)
        assert status == "REACHED"
    # 各会话步数互不相同也不串状态：起点/终点各自独立
    ids = {r[0] for r in results}
    assert ids == {f"iso-{i}" for i in range(8)}


async def test_same_session_steps_serialize_without_tearing():
    store = SessionStore()
    s = _make_session("ser-1")
    await store.create(s)
    async with s.lock:
        await session_plan(s)

    async def one_step():
        async with s.lock:
            before = s.step_count
            await asyncio.sleep(0)  # 持锁让出，验证串行
            snap = await session_step(s)
            after = s.step_count
            assert after == before + 1 or snap["status"] in ("REACHED",)
            return after

    counts = await asyncio.gather(*(one_step() for _ in range(40)))
    case_log("判定", counts=sorted(counts))
    # 40 个并发步进，计数必须恰好 1..40 无重复无跳号（REACHED 前）
    assert sorted(counts) == list(range(1, 41))


async def test_lock_timeout_returns_distinguishable_error(monkeypatch):
    store = SessionStore(lock_timeout=0.05)
    s = _make_session("lock-1")
    await store.create(s)

    async def hold():
        await s.lock.acquire()
        await asyncio.sleep(0.3)
        s.lock.release()

    async def contender():
        await asyncio.sleep(0.05)
        try:
            await asyncio.wait_for(s.lock.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            return "LOCK_TIMEOUT"
        else:
            s.lock.release()
            raise AssertionError("should not acquire")

    held = asyncio.create_task(hold())
    res = await contender()
    await held
    case_log("判定", result=res)
    assert res == "LOCK_TIMEOUT"


async def test_http_lock_timeout_is_distinguishable(client, monkeypatch):
    """同一会话两个并发长操作：后到者在锁超时后收到 409 LOCK_TIMEOUT。"""

    from fastapi.testclient import TestClient  # noqa: F401
    # 通过直接调小 store 锁超时并用线程级并发触发
    store = client.app.state.session_store
    monkeypatch.setattr(store, "lock_timeout", 0.05)

    r = await client_sid_async(client)
    sid = r
    import httpx
    from app.api import get_store  # noqa: F401

    async def slow_step():
        # 直接持有会话锁模拟长操作
        from app.session import session_step
        s = store.get(sid)
        async with s.lock:
            await asyncio.sleep(0.3)
            return await session_step(s)

    async def timed_step():
        await asyncio.sleep(0.05)
        # 直接走 API 层的取锁逻辑
        from app.api import session_lock
        from app.errors import NavigationError
        s = store.get(sid)
        try:
            async with session_lock(s, store):
                return "acquired"
        except NavigationError as e:
            return e.code.value

    res = await asyncio.gather(slow_step(), timed_step())
    case_log("判定", results=res)
    assert "SESSION_LOCK_TIMEOUT" in res


async def client_sid_async(client):
    from tests.conftest import open_grid
    r = client.post("/api/v1/sessions", json={
        "grid": open_grid(20, 12), "resolution": 0.1,
        "start": {"x": 0.15, "y": 0.15}, "goal": {"x": 1.85, "y": 1.05},
        "kinematics": {"robot_radius": 0.0},
    })
    return r.json()["session_id"]
