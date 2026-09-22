"""领域模型：会话状态、位姿、机器人约束、地图与规划结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SessionStatus(str, Enum):
    """会话状态机。

    状态迁移：
        CREATED --plan--> PLANNING --(成功)--> READY --step--> RUNNING
        RUNNING --step--> RUNNING / ARRIVED / BLOCKED
        BLOCKED --step(环境变化/重规划成功)--> RUNNING / ARRIVED
        READY/RUNNING/BLOCKED --replan--> READY
        任意 --reset--> READY（清轨迹）
    """

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    ARRIVED = "arrived"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Pose:
    """SE(2) 位姿，theta 为弧度。"""

    x: float
    y: float
    theta: float


@dataclass(frozen=True, slots=True)
class Twist:
    """体坐标系速度指令（差速/自行车模型：v 线速度，omega 角速度）。"""

    v: float
    omega: float


@dataclass(frozen=True, slots=True)
class RobotLimits:
    """机器人运动学约束。"""

    max_v: float
    min_v: float
    max_omega: float
    # 曲率上限 = 1 / 最小转弯半径（1/m）
    max_curvature: float
    max_acc: float
    footprint_radius: float

    def __post_init__(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PlannerParams:
    """全局规划参数。"""

    obstacle_inflation_cells: int
    allow_diagonal: bool
    max_expanded_nodes: int
    cost_diagonal: float


@dataclass(frozen=True, slots=True)
class ControllerParams:
    """局部轨迹跟踪参数。"""

    k_linear: float
    k_angular: float
    goal_tolerance: float
    lookahead_cells: int
    dt: float


@dataclass(slots=True)
class StepResult:
    """单次步进结果。"""

    pose: Pose
    cmd: Twist
    replanned: bool
    arrived: bool
    blocked: bool
    note: str


@dataclass(slots=True)
class SessionSnapshot:
    """会话对外可解释快照。"""

    session_id: str
    status: SessionStatus
    pose: Pose
    goal: Point | None
    path: list[Point]
    path_index: int
    step_count: int
    message: str
    last_note: str
    created_at: float
    updated_at: float
