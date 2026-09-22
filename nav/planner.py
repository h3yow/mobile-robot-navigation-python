"""全局路径规划：基于膨胀占用栅格的 A*（8 邻域）。"""

from __future__ import annotations

import heapq
import math

from .errors import (
    GoalInObstacleError,
    NoPathError,
    PlanningTimeoutError,
    StartInObstacleError,
)
from .grid import OccupancyGrid
from .models import PlannerParams, Point


class GridPlanner:
    """占用栅格 A* 规划器。"""

    def __init__(self, grid: OccupancyGrid, params: PlannerParams) -> None:
        self.grid = grid
        self.params = params

    def plan(self, start: Point, goal: Point) -> list[Point]:
        """规划从 start 到 goal 的无碰撞世界坐标路径（含起终点）。

        规划在按 footprint 膨胀后的栅格上进行，保证路径与障碍保持安全间距。
        """
        g = self.grid
        p = self.params

        # 1) 起终点合法性（坐标越界错误由 world_to_cell 抛出）
        sx, sy = g.world_to_cell(start.x, start.y)
        gx, gy = g.world_to_cell(goal.x, goal.y)

        inflated = g.inflated(p.obstacle_inflation_cells)
        if inflated.is_occupied_cell(sx, sy):
            # 膨胀后不可通行：区分“本身在障碍上”与“距障碍过近”
            if g.is_occupied_cell(sx, sy):
                raise StartInObstacleError(
                    "起点位于障碍栅格内",
                    {"start": [start.x, start.y], "cell": [sx, sy]},
                )
            raise StartInObstacleError(
                "起点与障碍距离不足（机器人足迹+膨胀间距）",
                {"start": [start.x, start.y], "cell": [sx, sy]},
            )
        if inflated.is_occupied_cell(gx, gy):
            if g.is_occupied_cell(gx, gy):
                raise GoalInObstacleError(
                    "终点位于障碍栅格内",
                    {"goal": [goal.x, goal.y], "cell": [gx, gy]},
                )
            raise GoalInObstacleError(
                "终点与障碍距离不足（机器人足迹+膨胀间距）",
                {"goal": [goal.x, goal.y], "cell": [gx, gy]},
            )

        # 2) A* 搜索（膨胀栅格的单元中心坐标）
        cell_path = self._astar(inflated, (sx, sy), (gx, gy))

        # 3) 共线压缩后转世界坐标：首点用真实起点、末点用真实终点
        compressed = self._compress(cell_path)
        points: list[Point] = [Point(start.x, start.y)]
        for cx, cy in compressed[1:-1]:
            wx, wy = g.cell_center(cx, cy)
            points.append(Point(wx, wy))
        points.append(Point(goal.x, goal.y))

        # 4) 段级足迹碰撞复核（处理压缩后长直段跨角切割障碍的情况）
        points = self._verify_segments(points, g)
        # 5) 起终点同格时去掉重复端点
        if (
            len(points) == 2
            and abs(points[0].x - points[1].x) < 1e-12
            and abs(points[0].y - points[1].y) < 1e-12
        ):
            points = points[:1]
        return points

    # ------------------------------------------------------------------
    def _astar(
        self, grid: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        if start == goal:
            return [start]

        w, h = grid.width, grid.height
        diag = self.params.cost_diagonal
        moves = (
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, diag), (1, -1, diag), (-1, 1, diag), (-1, -1, diag),
        )

        def heuristic(cx: int, cy: int) -> float:
            dx, dy = abs(cx - goal[0]), abs(cy - goal[1])
            if self.params.allow_diagonal:
                return (dx + dy) + (diag - 2) * min(dx, dy)
            return float(dx + dy)

        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        counter = 0
        heapq.heappush(open_heap, (heuristic(*start), counter, start))
        closed: set[tuple[int, int]] = set()
        expanded = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)
            expanded += 1
            if expanded > self.params.max_expanded_nodes:
                raise PlanningTimeoutError(
                    "规划超过最大扩展节点数（地图过大或环境复杂）",
                    {"max_expanded_nodes": self.params.max_expanded_nodes},
                )

            cx, cy = current
            for dx, dy, step_cost in moves:
                if not self.params.allow_diagonal and dx != 0 and dy != 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if grid.data[ny * w + nx] == 1:
                    continue
                # 对角移动禁止“穿墙角”：两个正交邻格必须同时自由
                if dx != 0 and dy != 0:
                    if (
                        grid.data[cy * w + (cx + dx)] == 1
                        or grid.data[(cy + dy) * w + cx] == 1
                    ):
                        continue
                neighbor = (nx, ny)
                if neighbor in closed:
                    continue
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative + heuristic(nx, ny), counter, neighbor),
                    )

        raise NoPathError(
            "不存在从起点到终点的无碰撞可达路径",
            {"start_cell": list(start), "goal_cell": list(goal)},
        )

    @staticmethod
    def _compress(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """去掉路径中位于水平/竖直/45 度直线上的中间点。"""
        if len(path) <= 2:
            return path
        out = [path[0]]
        for i in range(1, len(path) - 1):
            ax, ay = path[i - 1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            dx1, dy1 = bx - ax, by - ay
            dx2, dy2 = cx - bx, cy - by
            if (dx1, dy1) != (dx2, dy2):
                out.append(path[i])
        out.append(path[-1])
        return out

    def _verify_segments(
        self, points: list[Point], base_grid: OccupancyGrid
    ) -> list[Point]:
        """逐段以分辨率步长做足迹碰撞复核。

        压缩只保留共线点，长对角段理论上不切角（A* 已防穿墙角），
        但起终点为任意世界坐标，故再以连续线段复核一次保证不穿障。
        """
        r = self.params.obstacle_inflation_cells
        # 这里复用膨胀格查询：临时构造检查网格
        inflated = base_grid.inflated(r)
        for a, b in zip(points, points[1:]):
            dist = math.hypot(b.x - a.x, b.y - a.y)
            steps = max(1, math.ceil(dist / (inflated.resolution * 0.5)))
            for i in range(steps + 1):
                t = i / steps
                x = a.x + (b.x - a.x) * t
                y = a.y + (b.y - a.y) * t
                if inflated.footprint_blocked(x, y, 0.0):
                    # 起终点已校验；若中间段命中，说明压缩跨角，放弃压缩重搜安全
                    raise NoPathError(
                        "路径安全复核失败（直段穿越膨胀障碍）",
                        {"segment": [[a.x, a.y], [b.x, b.y]]},
                    )
        return points

    def is_path_clear(self, path: list[Point], footprint_radius: float) -> int:
        """返回路径上第一个被障碍阻挡的路径点下标（按足迹精确检测）；全通返回 -1。"""
        if not path:
            return -1
        for i, pt in enumerate(path):
            if self.grid.footprint_blocked(pt.x, pt.y, footprint_radius):
                return i
        # 段间插值检查（障碍可能新出现在两段中间）
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            dist = math.hypot(b.x - a.x, b.y - a.y)
            steps = max(1, math.ceil(dist / (self.grid.resolution * 0.5)))
            for k in range(1, steps):
                t = k / steps
                x = a.x + (b.x - a.x) * t
                y = a.y + (b.y - a.y) * t
                if self.grid.footprint_blocked(x, y, footprint_radius):
                    return i + 1
        return -1
