"""全局路径规划：基于膨胀占用栅格的 A*（8 邻域）。"""

from __future__ import annotations

from .grid import OccupancyGrid
from .models import PlannerParams, Point


class GridPlanner:
    """占用栅格 A* 规划器。"""

    def __init__(self, grid: OccupancyGrid, params: PlannerParams) -> None:
        self.grid = grid
        self.params = params

    def plan(self, start: Point, goal: Point) -> list[Point]:
        """规划从 start 到 goal 的无碰撞世界坐标路径（含起终点）。

        拒绝情形（可区分错误）：
          - 坐标越界 -> CoordinateOutOfBoundsError
          - 起点/终点在障碍上 -> StartInObstacleError / GoalInObstacleError
          - 无可达路径 -> NoPathError
          - 超过节点预算 -> PlanningTimeoutError
        """
        raise NotImplementedError

    def is_path_clear(self, path: list[Point], footprint_radius: float) -> int:
        """返回路径上第一个被障碍阻挡的路径点下标；全部通畅返回 -1。"""
        raise NotImplementedError
