"""局部轨迹跟踪：纯追踪式控制器，输出满足 v / omega / 曲率约束的指令。"""

from __future__ import annotations

from .models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    Twist,
)


class TrajectoryTracker:
    """差速运动学轨迹跟踪器。"""

    def __init__(self, limits: RobotLimits, params: ControllerParams) -> None:
        self.limits = limits
        self.params = params

    def find_aim_point(
        self, pose: Pose, path: list[Point], path_index: int
    ) -> tuple[Point, int]:
        """在路径上选择前视点，并推进路径下标。"""
        raise NotImplementedError

    def compute_command(self, pose: Pose, aim: Point) -> Twist:
        """根据位姿与前视点计算受运动学约束的速度指令。"""
        raise NotImplementedError

    def advance(self, pose: Pose, cmd: Twist) -> Pose:
        """按 dt 积分（差速模型，角度归一化到 [-pi, pi]）。"""
        raise NotImplementedError

    def at_goal(self, pose: Pose, goal: Point) -> bool:
        raise NotImplementedError
