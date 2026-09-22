"""仿真会话：持有地图/参数/位姿/路径，串行化状态迁移。"""

from __future__ import annotations

import threading
import uuid

from .controller import TrajectoryTracker
from .grid import OccupancyGrid
from .models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    SessionSnapshot,
    SessionStatus,
    StepResult,
)
from .planner import GridPlanner


class NavigationSession:
    """单个仿真会话的全部可变状态，自带可重入锁保证步进/重规划串行。"""

    def __init__(
        self,
        session_id: str,
        grid: OccupancyGrid,
        limits: RobotLimits,
        planner_params: PlannerParams,
        controller_params: ControllerParams,
        created_at: float,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        grid: OccupancyGrid,
        limits: RobotLimits,
        planner_params: PlannerParams,
        controller_params: ControllerParams,
        now: float,
    ) -> "NavigationSession":
        raise NotImplementedError

    # ---- 规划/重规划 ----
    def plan(self, start: Pose, goal: Point, now: float) -> list[Point]:
        raise NotImplementedError

    def replan(self, now: float) -> list[Point]:
        """以当前位姿为新起点重新规划到既有目标。"""
        raise NotImplementedError

    # ---- 动态障碍层 ----
    def update_dynamic_obstacles(self, rects, now: float) -> None:
        """全量更新动态障碍矩形列表（世界坐标），合并到动态障碍层。"""
        raise NotImplementedError

    # ---- 执行 ----
    def step(self, now: float) -> StepResult:
        """推进一步：必要时重规划，输出受约束速度指令并积分。"""
        raise NotImplementedError

    def reset(self, now: float) -> None:
        raise NotImplementedError

    def snapshot(self, now: float) -> SessionSnapshot:
        raise NotImplementedError
