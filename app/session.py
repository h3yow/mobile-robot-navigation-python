"""会话级仿真：状态机、并发串行化、幂等缓存与动态障碍重规划。

状态机
------
    CREATED ──plan 成功──▶ ACTIVE
    ACTIVE  ──step──▶ ACTIVE（途中检测到障碍并成功重规划：附带 replanned）
    ACTIVE  ──前方被堵、重规划失败──▶ BLOCKED
    BLOCKED ──更新障碍/重规划成功──▶ ACTIVE（失败后可继续执行）
    ACTIVE/BLOCKED ──到达目标──▶ REACHED（终态，拒绝继续步进）
    任何活动态 ──起点实体碰撞等不可恢复错误──▶ FAILED（可清理障碍后恢复）

并发语义
--------
- 每个会话持有一把 asyncio.Lock：同一 session 的 plan/step/obstacle/查询
  在 async 层串行执行，杜绝错序与状态撕裂；不同 session 使用不同锁，完全并行。
- 取锁有超时（默认 5s），超时返回 SESSION_LOCK_TIMEOUT（409）而非无限排队。
- 变更类请求支持 request_id 幂等：相同 request_id 重复提交返回首次结果；
  同 id 但请求体不同返回 IDEMPOTENCY_CONFLICT。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from app.errors import ErrorCode, NavigationError
from app.gridmap import OccupancyGrid
from app.kinematics import Kinematics
from app.planner import Plan, PlannerConfig, plan_global
from app.tracker import StepResult, TrackerConfig, pure_pursuit_step


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    REACHED = "REACHED"
    FAILED = "FAILED"


_TERMINAL = {SessionStatus.REACHED}


@dataclass
class StepEvent:
    step: int
    x: float
    y: float
    theta: float
    speed: float
    curvature: float
    angular_speed: float
    replanned: bool
    blocked: bool
    reached: bool
    replan_reason: str | None
    events: list[str] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    grid: OccupancyGrid                 # 当前（含动态障碍）地图
    base_grid: OccupancyGrid            # 建会话时的静态地图
    kin: Kinematics
    start_xy: tuple[float, float]
    start_theta: float
    goal_xy: tuple[float, float]
    goal_theta: float | None
    planner_cfg: PlannerConfig
    tracker_cfg: TrackerConfig
    status: SessionStatus = SessionStatus.CREATED
    plan: Plan | None = None
    state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    progress: int = 0
    step_count: int = 0
    replan_count: int = 0
    last_replan_step: int = -10
    replan_cooldown: int = 2
    history: list[StepEvent] = field(default_factory=list)
    dynamic_updates: int = 0
    blocked_cells: set[tuple[int, int]] = field(default_factory=set)
    last_note: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idempotency: dict[str, dict] = field(default_factory=dict)
    idempotency_hashes: dict[str, str] = field(default_factory=dict)

    def blocked_array(self) -> bytes:
        """静态占用 + 当前动态障碍叠加。"""

        buf = bytearray(self.base_grid.cells)
        for col, row in self.blocked_cells:
            if 0 <= col < self.grid.width and 0 <= row < self.grid.height:
                buf[row * self.grid.width + col] = 1
        return bytes(buf)

    def current_grid(self) -> OccupancyGrid:
        if not self.blocked_cells:
            return self.base_grid
        return OccupancyGrid(
            width=self.base_grid.width,
            height=self.base_grid.height,
            resolution=self.base_grid.resolution,
            origin_x=self.base_grid.origin_x,
            origin_y=self.base_grid.origin_y,
            cells=self.blocked_array(),
            notes=self.base_grid.notes,
        )


class SessionStore:
    """内存会话仓库。"""

    def __init__(self, lock_timeout: float = 5.0) -> None:
        self._sessions: dict[str, Session] = {}
        self._guard = asyncio.Lock()
        self.lock_timeout = lock_timeout

    async def create(self, session: Session) -> None:
        async with self._guard:
            if session.session_id in self._sessions:
                raise NavigationError(
                    ErrorCode.INVALID_REQUEST,
                    f"session_id={session.session_id} 已存在",
                    {"session_id": session.session_id},
                )
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Session:
        s = self._sessions.get(session_id)
        if s is None:
            raise NavigationError(
                ErrorCode.SESSION_NOT_FOUND,
                f"会话 {session_id} 不存在或已删除",
                {"session_id": session_id},
            )
        return s

    def list_ids(self) -> list[str]:
        return sorted(self._sessions.keys())

    async def delete(self, session_id: str) -> bool:
        async with self._guard:
            return self._sessions.pop(session_id, None) is not None


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False,
                   default=str).encode("utf-8")
    ).hexdigest()[:16]


def check_idempotency(session: Session, request_id: str | None, payload: dict) -> dict | None:
    """命中幂等缓存返回既有响应；同 id 不同体抛冲突；否则返回 None。"""

    if not request_id:
        return None
    fp = _fingerprint(payload)
    if request_id in session.idempotency:
        if session.idempotency_hashes.get(request_id) != fp:
            raise NavigationError(
                ErrorCode.IDEMPOTENCY_CONFLICT,
                f"request_id={request_id} 已用于不同的请求体",
                {"request_id": request_id},
            )
        cached = dict(session.idempotency[request_id])
        cached["idempotent_replay"] = True
        return cached
    session.idempotency_hashes[request_id] = fp
    return None


def store_idempotency(session: Session, request_id: str | None, response: dict) -> None:
    if request_id:
        session.idempotency[request_id] = response


def _do_plan(session: Session, *, reason: str) -> Plan:
    """以当前机器人位置为起点、原目标为终点重新规划。"""

    grid = session.current_grid()
    session.grid = grid
    plan = plan_global(
        grid,
        session.kin,
        (session.state[0], session.state[1]),
        session.goal_xy,
        cfg=session.planner_cfg,
        replan_count=session.replan_count,
    )
    session.plan = plan
    # 重规划后从新路径最近点开始跟踪
    session.progress = 0
    return plan


async def session_plan(session: Session) -> dict:
    """建立或重新建立全局路径（调用方须持会话锁）。"""

    if session.status in _TERMINAL:
        raise NavigationError(
            ErrorCode.SESSION_ALREADY_FINISHED,
            f"会话已处于终态 {session.status.value}，不能再规划",
            {"status": session.status.value},
        )
    plan = _do_plan(session, reason="explicit_plan")
    session.status = SessionStatus.ACTIVE
    session.last_note = "plan_ok"
    return {
        "status": session.status.value,
        "replan_count": session.replan_count,
        "path_length": len(plan.world_path),
        "path": [[round(x, 6), round(y, 6)] for x, y in plan.world_path],
        "expansions": plan.expansions,
        "elapsed_ms": round(plan.elapsed_ms, 3),
        "snapped_start": list(plan.snapped_start) if plan.snapped_start else None,
        "snapped_goal": list(plan.snapped_goal) if plan.snapped_goal else None,
        "notes": list(plan.notes),
    }


async def session_step(session: Session) -> dict:
    """推进一个仿真周期（调用方须持会话锁）。"""

    if session.status in _TERMINAL:
        raise NavigationError(
            ErrorCode.SESSION_ALREADY_FINISHED,
            f"会话已 {session.status.value}，步进被拒绝",
            {"status": session.status.value},
        )
    if session.plan is None or session.status == SessionStatus.CREATED:
        raise NavigationError(
            ErrorCode.INVALID_REQUEST,
            "会话尚无有效路径，请先调用 /plan",
            {},
            http_status=409,
        )

    grid = session.current_grid()
    session.grid = grid

    # 若当前位置已经撞入动态障碍：不移动，转 FAILED（可清除障碍后 plan 恢复）
    if grid.disk_collides(session.state[0], session.state[1], session.kin.robot_radius):
        session.status = SessionStatus.FAILED
        session.last_note = "current_pose_in_collision"
        raise NavigationError(
            ErrorCode.GOAL_IN_OBSTACLE,
            "机器人当前位置已与动态障碍重合；请移除相关障碍后重新 /plan",
            {"pose": [session.state[0], session.state[1]]},
            http_status=409,
        )

    sr: StepResult
    replanned_this_step = False
    replan_reason: str | None = None

    # 先尝试跟踪一步
    sr, new_state, new_progress = pure_pursuit_step(
        grid,
        session.kin,
        session.plan,
        session.state,
        cfg=session.tracker_cfg,
        progress=session.progress,
        goal_theta=session.goal_theta,
    )

    if sr.blocked and not sr.reached:
        # 前方（或执行弧段）被挡：尝试重规划。冷却期内不重复规划，
        # 仅保持停止并等待下周期按已有新路径行进。
        replan_reason = sr.replan_reason or "blocked"
        cooled_down = (
            session.step_count - session.last_replan_step >= session.replan_cooldown
        )
        if not cooled_down:
            session.status = SessionStatus.ACTIVE
            session.last_note = "replan_cooldown_hold"
            event = StepEvent(
                step=session.step_count + 1,
                x=session.state[0], y=session.state[1], theta=session.state[2],
                speed=0.0, curvature=0.0, angular_speed=0.0,
                replanned=False, blocked=True, reached=False,
                replan_reason=replan_reason,
                events=["cooldown_hold"],
            )
            session.step_count += 1
            session.history.append(event)
            return session_snapshot(session)
        try:
            session.replan_count += 1
            _do_plan(session, reason=replan_reason)
            session.last_replan_step = session.step_count
            replanned_this_step = True
            session.status = SessionStatus.ACTIVE
            session.last_note = f"replanned:{replan_reason}"
            # 重规划当周期不再移动（新路径下周期生效），记录一个零位移事件
            event = StepEvent(
                step=session.step_count + 1,
                x=session.state[0], y=session.state[1], theta=session.state[2],
                speed=0.0, curvature=0.0, angular_speed=0.0,
                replanned=True, blocked=False, reached=False,
                replan_reason=replan_reason,
                events=[f"replanned:{replan_reason}"],
            )
            session.step_count += 1
            session.history.append(event)
            return session_snapshot(session)
        except NavigationError as e:
            # 重规划失败：保持位置与状态，可继续（移除障碍后再步进/规划）
            session.replan_count -= 1
            session.status = SessionStatus.BLOCKED
            session.last_note = f"replan_failed:{e.code.value}"
            snap = session_snapshot(session)
            snap["replan_error"] = e.to_body()["error"]
            return snap

    # 提交状态
    session.state = new_state
    session.progress = new_progress
    session.step_count += 1
    event = StepEvent(
        step=session.step_count,
        x=sr.x, y=sr.y, theta=sr.theta,
        speed=sr.speed, curvature=sr.curvature, angular_speed=sr.angular_speed,
        replanned=replanned_this_step,
        blocked=sr.blocked,
        reached=sr.reached,
        replan_reason=replan_reason,
        events=list(sr.events),
    )
    session.history.append(event)
    if sr.reached:
        session.status = SessionStatus.REACHED
    elif sr.blocked:
        session.status = SessionStatus.BLOCKED
    else:
        session.status = SessionStatus.ACTIVE
    session.last_note = "reached" if sr.reached else ("blocked" if sr.blocked else "ok")
    return session_snapshot(session)


async def session_update_obstacles(
    session: Session,
    cells: list[tuple[int, int, int]] | None = None,
    reset: bool = False,
) -> dict:
    """更新动态障碍层（cells 为 [col,row,value]）。调用方须持锁。"""

    if reset:
        session.blocked_cells.clear()
    if cells:
        w, h = session.base_grid.width, session.base_grid.height
        for col, row, value in cells:
            if not (0 <= col < w and 0 <= row < h):
                raise NavigationError(
                    ErrorCode.COORDINATE_OUT_OF_BOUNDS,
                    f"动态障碍格 ({col},{row}) 越界（地图 {w}x{h}）",
                    {"cell": [col, row], "bounds": [w, h]},
                )
            if value not in (0, 1):
                raise NavigationError(
                    ErrorCode.MAP_INVALID_CELL_VALUE,
                    f"动态障碍格值 {value!r} 非法：仅允许 0/1",
                    {"cell": [col, row], "value": value},
                )
            if value == 1:
                session.blocked_cells.add((col, row))
            else:
                session.blocked_cells.discard((col, row))
    session.dynamic_updates += 1

    # 若机器人当前位置被新障碍覆盖：标记 FAILED，不移动；清除后可 plan 恢复
    grid = session.current_grid()
    session.grid = grid
    if grid.disk_collides(session.state[0], session.state[1], session.kin.robot_radius):
        session.status = SessionStatus.FAILED
        session.last_note = "dynamic_obstacle_overlaps_robot"
    elif session.status == SessionStatus.BLOCKED:
        # 障碍可能已被移除，回到 ACTIVE 等待下一次步进/重规划
        session.status = SessionStatus.ACTIVE
        session.last_note = "obstacles_updated_recoverable"
    else:
        session.last_note = "obstacles_updated"
    return session_snapshot(session)


def session_snapshot(session: Session) -> dict:
    """完整、可解释的状态快照。"""

    x, y, theta, speed = session.state
    plan_tail: list[list[float]] = []
    progress_now = session.progress
    if session.plan is not None:
        plan_tail = [
            [round(px, 6), round(py, 6)]
            for px, py in session.plan.world_path[progress_now:]
        ]
    last_events: list[str] = []
    if session.history:
        last_events = list(session.history[-1].events)
    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "pose": {
            "x": round(x, 6),
            "y": round(y, 6),
            "theta": round(theta, 6),
            "speed": round(speed, 6),
        },
        "goal": {"x": session.goal_xy[0], "y": session.goal_xy[1],
                 "theta": session.goal_theta},
        "start": {"x": session.start_xy[0], "y": session.start_xy[1],
                  "theta": session.start_theta},
        "step_count": session.step_count,
        "replan_count": session.replan_count,
        "progress_index": session.progress,
        "dynamic_updates": session.dynamic_updates,
        "dynamic_obstacle_cells": [list(c) for c in sorted(session.blocked_cells)],
        "remaining_path_length": len(plan_tail),
        "remaining_path": plan_tail,
        "last_note": session.last_note,
        "last_events": last_events,
        "last_step": _last_step_dict(session),
    }


def _last_step_dict(session: Session) -> dict | None:
    if not session.history:
        return None
    e = session.history[-1]
    return {
        "step": e.step,
        "x": round(e.x, 6), "y": round(e.y, 6), "theta": round(e.theta, 6),
        "speed": round(e.speed, 6), "curvature": round(e.curvature, 6),
        "angular_speed": round(e.angular_speed, 6),
        "replanned": e.replanned, "blocked": e.blocked, "reached": e.reached,
        "replan_reason": e.replan_reason, "events": e.events,
    }
