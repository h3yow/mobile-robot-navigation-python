"""HTTP 层：请求模型、路由装配、统一错误处理、会话锁与幂等接线。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.errors import ErrorCode, NavigationError
from app.gridmap import parse_map_input
from app.kinematics import Kinematics, default_kinematics
from app.planner import PlannerConfig
from app.session import (
    Session,
    SessionStore,
    check_idempotency,
    session_plan,
    session_snapshot,
    session_step,
    session_update_obstacles,
    store_idempotency,
)
from app.tracker import TrackerConfig

logger = logging.getLogger("navsim.api")

router = APIRouter()


# ------------------------------------------------------------------------
# 请求模型
# ------------------------------------------------------------------------
class PoseModel(BaseModel):
    x: float = 0.0
    y: float = 0.0
    theta: float | None = None


class KinematicsModel(BaseModel):
    max_linear_speed: float | None = None
    max_angular_speed: float | None = None
    max_curvature: float | None = None
    robot_radius: float | None = None
    max_linear_accel: float | None = None
    dt: float | None = None


class CreateSessionRequest(BaseModel):
    session_id: str | None = None
    grid: list[str] | None = None
    occupancy: list[list[int]] | list[int] | None = None
    width: int | None = None
    height: int | None = None
    resolution: float = 0.1
    origin: list[float] | None = None
    start: PoseModel = Field(default_factory=PoseModel)
    goal: PoseModel = Field(default_factory=PoseModel)
    kinematics: KinematicsModel = Field(default_factory=KinematicsModel)
    planner_time_budget: float = 2.0
    max_expansions: int = 2_000_000
    lookahead: float | None = None
    goal_tolerance: float | None = None
    heading_tolerance: float | None = None
    path_horizon: float | None = None
    auto_plan: bool = True
    request_id: str | None = None


class PlanRequest(BaseModel):
    request_id: str | None = None


class StepRequest(BaseModel):
    request_id: str | None = None
    steps: int = 1


class ObstacleUpdate(BaseModel):
    cells: list[list[int]] | None = None
    reset: bool = False
    request_id: str | None = None


def get_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def build_kinematics(m: KinematicsModel) -> Kinematics:
    dt = m.dt if m.dt is not None else 0.1
    # 显式 max_curvature 时必须同时给出自洽的速度参数
    if m.max_curvature is not None:
        vmax = m.max_linear_speed if m.max_linear_speed is not None else 1.0
        wmax = m.max_angular_speed if m.max_angular_speed is not None else 1.0
        radius = m.robot_radius if m.robot_radius is not None else 0.15
        accel = m.max_linear_accel if m.max_linear_accel is not None else 1.0
        return Kinematics(
            max_linear_speed=vmax,
            max_angular_speed=wmax,
            max_curvature=m.max_curvature,
            robot_radius=radius,
            max_linear_accel=accel,
            dt=dt,
        )
    return default_kinematics(
        dt=dt,
        max_linear_speed=m.max_linear_speed if m.max_linear_speed is not None else 1.0,
        max_angular_speed=m.max_angular_speed if m.max_angular_speed is not None else 1.0,
        robot_radius=m.robot_radius if m.robot_radius is not None else 0.15,
        max_linear_accel=m.max_linear_accel if m.max_linear_accel is not None else 1.0,
    )


@asynccontextmanager
async def session_lock(session: Session, store: SessionStore):
    """取会话锁，超时返回可区分错误。"""

    try:
        await asyncio.wait_for(session.lock.acquire(), timeout=store.lock_timeout)
    except asyncio.TimeoutError:
        raise NavigationError(
            ErrorCode.SESSION_LOCK_TIMEOUT,
            f"会话 {session.session_id} 在 {store.lock_timeout}s 内未取得串行锁，"
            "可能存在长时间占用的请求",
            {"session_id": session.session_id,
             "lock_timeout": store.lock_timeout},
        )
    try:
        yield
    finally:
        session.lock.release()


@router.post("/sessions", status_code=201)
async def create_session(payload: CreateSessionRequest, request: Request) -> dict:
    store = get_store(request)
    sid = payload.session_id or uuid.uuid4().hex[:12]
    request_id = payload.request_id

    grid = parse_map_input(
        grid=payload.grid,
        occupancy=payload.occupancy,
        width=payload.width,
        height=payload.height,
        resolution=payload.resolution,
        origin=tuple(payload.origin) if payload.origin else None,
    )
    kin = build_kinematics(payload.kinematics)
    planner_cfg = PlannerConfig(
        time_budget=payload.planner_time_budget,
        max_expansions=payload.max_expansions,
    )
    tcfg_kwargs: dict[str, float] = {}
    if payload.lookahead is not None:
        tcfg_kwargs["lookahead"] = payload.lookahead
    if payload.goal_tolerance is not None:
        tcfg_kwargs["goal_tolerance"] = payload.goal_tolerance
    if payload.heading_tolerance is not None:
        tcfg_kwargs["heading_tolerance"] = payload.heading_tolerance
    if payload.path_horizon is not None:
        tcfg_kwargs["path_horizon"] = payload.path_horizon
    tracker_cfg = TrackerConfig(**tcfg_kwargs)

    start_theta = payload.start.theta if payload.start.theta is not None else 0.0
    goal_theta = payload.goal.theta
    session = Session(
        session_id=sid,
        grid=grid,
        base_grid=grid,
        kin=kin,
        start_xy=(payload.start.x, payload.start.y),
        start_theta=start_theta,
        goal_xy=(payload.goal.x, payload.goal.y),
        goal_theta=goal_theta,
        planner_cfg=planner_cfg,
        tracker_cfg=tracker_cfg,
        state=(payload.start.x, payload.start.y, start_theta, 0.0),
    )
    await store.create(session)
    logger.info(
        "session created id=%s map=%dx%d res=%g r=%g start=%s goal=%s",
        sid, grid.width, grid.height, grid.resolution, kin.robot_radius,
        session.start_xy, session.goal_xy,
    )

    body: dict[str, Any] = {"session_id": sid, "status": session.status.value,
                           "map": {"width": grid.width, "height": grid.height,
                                   "resolution": grid.resolution,
                                   "free_cells": grid.free_cell_count(),
                                   "notes": list(grid.notes)}}
    if payload.auto_plan:
        async with session_lock(session, store):
            if request_id:
                cached = check_idempotency(session, request_id, payload.model_dump())
                if cached is not None:
                    return cached
            plan_body = await session_plan(session)
            body.update(plan_body)
            body["status"] = session.status.value
            store_idempotency(session, request_id, body)
    return body


@router.get("/sessions")
async def list_sessions(request: Request) -> dict:
    store = get_store(request)
    return {"sessions": store.list_ids(), "count": len(store.list_ids())}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    store = get_store(request)
    session = store.get(session_id)
    async with session_lock(session, store):
        return session_snapshot(session)


@router.post("/sessions/{session_id}/plan")
async def plan_session(
    session_id: str, payload: PlanRequest | None, request: Request
) -> dict:
    store = get_store(request)
    session = store.get(session_id)
    rid = payload.request_id if payload else None
    fp = payload.model_dump() if payload else {}
    async with session_lock(session, store):
        if rid:
            cached = check_idempotency(session, rid, {"plan": fp})
            if cached is not None:
                return cached
        body = await session_plan(session)
        store_idempotency(session, rid, body)
        return body


@router.post("/sessions/{session_id}/step")
async def step_session(
    session_id: str, payload: StepRequest | None, request: Request
) -> dict:
    store = get_store(request)
    session = store.get(session_id)
    steps = payload.steps if payload else 1
    if not isinstance(steps, int) or steps <= 0 or steps > 10000:
        raise NavigationError(
            ErrorCode.INVALID_REQUEST,
            f"steps={steps!r} 非法：必须为 1..10000 的整数",
            {"steps": steps},
        )
    rid = payload.request_id if payload else None
    async with session_lock(session, store):
        if rid:
            # 步进幂等键含次数，避免多步请求被错误复用
            cached = check_idempotency(
                session, rid, {"step": {"steps": steps}}
            )
            if cached is not None:
                return cached
        last: dict = {}
        reached_or_blocked = False
        for _ in range(steps):
            last = await session_step(session)
            if last["status"] in ("REACHED", "BLOCKED", "FAILED"):
                reached_or_blocked = True
                break
        body = dict(last)
        body["stopped_early"] = reached_or_blocked
        store_idempotency(session, rid, body)
        return body


@router.post("/sessions/{session_id}/obstacles")
async def update_obstacles(
    session_id: str, payload: ObstacleUpdate, request: Request
) -> dict:
    store = get_store(request)
    session = store.get(session_id)
    cells: list[tuple[int, int, int]] = []
    for item in payload.cells or []:
        if len(item) != 3:
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                f"障碍项 {item!r} 必须为 [col,row,value] 三元组",
                {"item": item},
            )
        cells.append((int(item[0]), int(item[1]), int(item[2])))
    rid = payload.request_id
    async with session_lock(session, store):
        if rid:
            cached = check_idempotency(
                session, rid,
                {"obstacles": {"cells": payload.cells, "reset": payload.reset}},
            )
            if cached is not None:
                return cached
        body = await session_update_obstacles(
            session, cells=cells, reset=payload.reset
        )
        store_idempotency(session, rid, body)
        return body


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    store = get_store(request)
    removed = await store.delete(session_id)
    if not removed:
        raise NavigationError(
            ErrorCode.SESSION_NOT_FOUND,
            f"会话 {session_id} 不存在或已删除",
            {"session_id": session_id},
        )
    return {"deleted": session_id}


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NavigationError)
    async def _nav_error_handler(_: Request, exc: NavigationError) -> JSONResponse:
        logger.info("navigation error %s: %s", exc.code.value, exc.message)
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.exception_handler(Exception)
    async def _fallback_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        body = {
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": str(exc) or exc.__class__.__name__,
                "detail": {},
            }
        }
        return JSONResponse(status_code=500, content=body)
