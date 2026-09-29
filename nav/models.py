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
        from .errors import InvalidInputError

        if not (self.max_v > 0):
            raise InvalidInputError(
                "max_v 必须为正数", {"field": "max_v", "value": self.max_v}
            )
        if self.min_v > 0:
            raise InvalidInputError(
                "min_v 必须 <= 0（允许原地转向/倒车）",
                {"field": "min_v", "value": self.min_v},
            )
        if self.min_v < -self.max_v:
            raise InvalidInputError(
                "min_v 不得小于 -max_v（倒车速度受限）",
                {"field": "min_v", "value": self.min_v},
            )
        if not (self.max_omega > 0):
            raise InvalidInputError(
                "max_omega 必须为正数", {"field": "max_omega"}
            )
        if not (self.max_curvature > 0):
            raise InvalidInputError(
                "max_curvature 必须为正数", {"field": "max_curvature"}
            )
        if not (self.max_acc > 0):
            raise InvalidInputError("max_acc 必须为正数", {"field": "max_acc"})
        if not (self.footprint_radius > 0):
            raise InvalidInputError(
                "footprint_radius 必须为正数", {"field": "footprint_radius"}
            )

    def clamp_command(self, v: float, omega: float):
        """把 (v, omega) 裁剪到速度/角速度/曲率包络内，返回 (v, omega)。

        曲率约束：|omega / v| <= max_curvature（v 很小时放宽，交由角速度约束）。
        """
        # 线速度与角速度硬限幅
        v = max(self.min_v, min(self.max_v, v))
        omega = max(-self.max_omega, min(self.max_omega, omega))
        # 曲率限幅（仅在有明显平移速度时生效）
        eps = 1e-6
        if abs(v) > eps:
            max_omega_for_curvature = self.max_curvature * abs(v)
            if abs(omega) > max_omega_for_curvature:
                omega = max_omega_for_curvature if omega > 0 else -max_omega_for_curvature
        return v, omega


@dataclass(frozen=True, slots=True)
class PlannerParams:
    """全局规划参数。"""

    obstacle_inflation_cells: int = 1
    allow_diagonal: bool = True
    max_expanded_nodes: int = 200_000
    cost_diagonal: float = 1.4142135623730951

    def __post_init__(self) -> None:
        from .errors import InvalidInputError

        if self.obstacle_inflation_cells < 0:
            raise InvalidInputError(
                "obstacle_inflation_cells 不能为负",
                {"value": self.obstacle_inflation_cells},
            )
        if self.max_expanded_nodes <= 0:
            raise InvalidInputError("max_expanded_nodes 必须为正整数")
        if not (self.cost_diagonal >= 1.0):
            raise InvalidInputError("cost_diagonal 必须 >= 1（A* 可采纳性）")


@dataclass(frozen=True, slots=True)
class ControllerParams:
    """局部轨迹跟踪参数。"""

    k_linear: float = 1.0
    k_angular: float = 2.0
    goal_tolerance: float = 0.05
    # 纯追踪前视距离（米）；会话创建时会结合栅格分辨率做下限保护
    lookahead_cells: float = 0.6
    dt: float = 0.1

    def __post_init__(self) -> None:
        from .errors import InvalidInputError

        if self.k_linear <= 0 or self.k_angular <= 0:
            raise InvalidInputError("k_linear/k_angular 必须为正")
        if self.goal_tolerance <= 0:
            raise InvalidInputError("goal_tolerance 必须为正")
        if self.lookahead_cells < 0:
            raise InvalidInputError("lookahead_cells 不能为负")
        if self.dt <= 0:
            raise InvalidInputError("dt 必须为正")


@dataclass(slots=True)
class StepResult:
    """单次步进结果。"""

    pose: Pose
    cmd: Twist
    replanned: bool
    arrived: bool
    blocked: bool
    note: str
    # 本微步实际模拟的时长（秒）。完整微步等于 dt；部分推进时 < dt；
    # 未发生运动（到达/被挡零速）时为 0.0。默认 0.0 保持旧构造兼容。
    elapsed: float = 0.0


@dataclass(slots=True)
class TrajectoryRecord:
    """单个微步的执行轨迹记录。

    swept 为本微步实际扫过的稠密位姿采样：swept[0] 为运动前位姿，
    swept[-1] 为本步上报位姿（== pose）。把各记录首尾相接即可还原连续轨迹，
    且每个记录的 pose 与对外 StepResult 位姿序列、step_count 一一对应。
    """

    step: int
    pose: Pose
    cmd: Twist
    swept: list[Pose]
    dt: float


@dataclass(slots=True)
class AdvanceResult:
    """一次按时长推进的结果（可能由多个内部微步组成）。"""

    status: SessionStatus
    pose: Pose
    requested_duration: float
    simulated_duration: float
    steps_executed: int
    completed: bool
    arrived: bool
    blocked: bool
    reason: str
    results: list[StepResult] = field(default_factory=list)


@dataclass(slots=True)
class RehearsalReport:
    """只读预演结果：不改动任何会话状态。"""

    feasible: bool
    completed: bool
    duration: float
    steps: int
    min_clearance: float
    failure_index: int | None
    failure_step: int | None
    reason: str


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
