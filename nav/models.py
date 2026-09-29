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


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    """一个执行微步（tick）落盘后的轨迹样本。

    index 为会话内全局单调序号（等于记录该样本后的 step_count）；
    t 为会话内累计仿真时间（秒）。advance 一次推进产生多个连续样本。
    """

    index: int
    t: float
    pose: Pose
    v: float
    omega: float
    replanned: bool
    blocked: bool
    arrived: bool


@dataclass(slots=True)
class AdvanceResult:
    """一次按时长推进的结果（可能只走完一部分）。

    poses 为本次推进每个微步对外上报的位姿序列（含被挡住时的原地零速步），
    与 samples 中的 pose 一一对应；completed=False 时 blocked/arrived 说明提前停止原因。
    """

    poses: list[Pose]
    samples: list[TrajectorySample]
    steps_requested: int
    steps_executed: int
    requested_duration: float
    dt: float
    elapsed: float
    completed: bool
    arrived: bool
    blocked: bool
    replanned: bool
    note: str


@dataclass(slots=True)
class PreviewResult:
    """只读预演结果：给定路线与运行参数，判断能否安全走完全程。

    feasible=False 时 fail_index 为路线顶点下标（大概在第几个位置出问题），
    fail_reason 取 start_blocked/path_invalid/collision/blocked/step_budget；
    min_clearance 为全程（路线折线与实际扫掠轨迹）离障碍/边界的最近距离（米）。
    """

    feasible: bool
    ticks_simulated: int
    min_clearance: float
    min_clearance_point: Point | None
    clearance_capped: bool
    fail_index: int | None
    fail_reason: str | None
    fail_pose: Pose | None
    note: str


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
    sim_time: float = 0.0
    trajectory_length: int = 0
    trajectory_total: int = 0
    trajectory_dropped: int = 0
    trajectory_capacity: int = 0
    trajectory_truncated: bool = False
