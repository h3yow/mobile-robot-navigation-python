"""会话级仿真状态机、并发保护与幂等缓存（骨架）。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from app.gridmap import OccupancyGrid
from app.kinematics import Kinematics
from app.planner import Plan, PlannerConfig
from app.tracker import StepResult, TrackerConfig


class SessionStatus(str, Enum):
    CREATED = "CREATED"          # 已建会话，尚无有效路径
    PLANNING = "PLANNING"        # 保留：规划瞬时状态
    ACTIVE = "ACTIVE"            # 有路径，可步进
    REPLANNING = "REPLANNING"    # 动态障碍触发的重新规划中
    BLOCKED = "BLOCKED"          # 当前指令无可行弧（可更新障碍后重试）
    REACHED = "REACHED"          # 到达目标
    FAILED = "FAILED"            # 重规划仍失败（可恢复）


@dataclass
class StepEvent:
    step: int
    x: float
    y: float
    theta: float
    speed: float
    curvature: float
    replanned: bool
    blocked: bool
    reached: bool
    replan_reason: str | None
    note: str = ""


@dataclass
class Session:
    session_id: str
    grid: OccupancyGrid
    kin: Kinematics
    start_xy: tuple[float, float]
    start_theta: float
    goal_xy: tuple[float, float]
    planner_cfg: PlannerConfig
    tracker_cfg: TrackerConfig
    status: SessionStatus = SessionStatus.CREATED
    plan: Plan | None = None
    state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x,y,theta,speed
    step_count: int = 0
    replan_count: int = 0
    created_replan_count: int = 0
    history: list[StepEvent] = field(default_factory=list)
    dynamic_updates: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idempotency: dict[str, dict] = field(default_factory=dict)


class SessionStore:
    """内存会话仓库：提供按会话串行化与全量查询。"""

    def __init__(self, lock_timeout: float = 5.0) -> None:
        self._sessions: dict[str, Session] = {}
        self.lock_timeout = lock_timeout

    def create(self, session: Session) -> None:
        raise NotImplementedError

    def get(self, session_id: str) -> Session:
        raise NotImplementedError

    def list_ids(self) -> list[str]:
        raise NotImplementedError

    def delete(self, session_id: str) -> bool:
        raise NotImplementedError


async def session_plan(session: Session, store: "SessionStore | None" = None) -> dict:
    """建立或重建立全局路径。骨架。"""
    raise NotImplementedError


async def session_step(session: Session) -> dict:
    """推进一个仿真周期（含动态障碍检测与重规划）。骨架。"""
    raise NotImplementedError


async def session_update_obstacles(session: Session, payload: dict) -> dict:
    """增量或全量更新动态障碍。骨架。"""
    raise NotImplementedError


def session_snapshot(session: Session) -> dict:
    """返回可解释的完整状态快照。骨架。"""
    raise NotImplementedError
