"""全局规划器：可达性、可区分拒绝原因、窄通道与大图节点预算。"""

from __future__ import annotations

import logging
import time

import pytest

from nav.errors import (
    CoordinateOutOfBoundsError,
    GoalInObstacleError,
    NoPathError,
    PlanningTimeoutError,
    StartInObstacleError,
)
from nav.grid import OccupancyGrid
from nav.models import PlannerParams, Point
from nav.planner import GridPlanner

log = logging.getLogger("test_planner")


def _planner(rows, inflation=0, **kw):
    g = OccupancyGrid.from_raw(rows, resolution=1.0)
    return g, GridPlanner(g, PlannerParams(obstacle_inflation_cells=inflation, **kw))


def test_open_map_path_is_collision_free():
    rows = [[0] * 10 for _ in range(8)]
    g, pl = _planner(rows)
    start, goal = Point(0.5, 0.5), Point(9.5, 7.5)
    path = pl.plan(start, goal)
    log.info("输入 空地图 start=%s goal=%s -> 路径 %s",
             (start.x, start.y), (goal.x, goal.y),
             [(p.x, p.y) for p in path])
    assert path[0] == start and path[-1] == goal
    assert all(not g.footprint_blocked(p.x, p.y, 0.0) for p in path)


def test_path_goes_through_gap():
    rows = [[0] * 10 for _ in range(7)]
    for x in range(10):
        rows[3][x] = 1
    rows[3][7] = 0  # 唯一缺口
    g, pl = _planner(rows)
    path = pl.plan(Point(0.5, 0.5), Point(0.5, 6.5))
    # 缺口格 (7,3) 中心 (7.5,3.5)：路径（含段上采样）必须经过该列附近
    def _near_gap():
        for a, b in zip(path, path[1:]):
            for k in range(21):
                t = k / 20
                x = a.x + (b.x - a.x) * t
                y = a.y + (b.y - a.y) * t
                if abs(x - 7.5) < 0.6 and abs(y - 3.5) < 0.6:
                    return True
        return False

    assert _near_gap()
    log.info("带墙缺口场景 -> 路径穿过缺口 (7.5,3.5), 共 %d 点: %s",
             len(path), [(p.x, p.y) for p in path])
    assert pl.is_path_clear(path, 0.0) == -1


def test_sealed_map_rejected_as_unreachable():
    rows = [[0] * 6 for _ in range(6)]
    for x in range(6):
        rows[2][x] = 1  # 无缺口整墙
    g, pl = _planner(rows)
    with pytest.raises(NoPathError) as ei:
        pl.plan(Point(0.5, 0.5), Point(0.5, 4.5))
    log.info("关键判定：全封死 -> code=%s details=%s",
             ei.value.code, ei.value.details)
    assert ei.value.code == "goal_unreachable"


def test_all_obstacle_map_start_goal_distinguishable():
    rows = [[1] * 4 for _ in range(4)]
    g, pl = _planner(rows)
    with pytest.raises(StartInObstacleError) as ei:
        pl.plan(Point(0.5, 0.5), Point(3.5, 3.5))
    log.info("关键判定：全障碍地图起点 -> code=%s", ei.value.code)
    assert ei.value.code == "start_in_obstacle"

    rows2 = [[0] * 4 for _ in range(4)]
    rows2[3][3] = 1
    g2, pl2 = _planner(rows2)
    with pytest.raises(GoalInObstacleError) as ei:
        pl2.plan(Point(0.5, 0.5), Point(3.5, 3.5))
    log.info("关键判定：终点落障 -> code=%s", ei.value.code)
    assert ei.value.code == "goal_in_obstacle"


def test_out_of_bounds_start_goal():
    rows = [[0] * 4 for _ in range(4)]
    g, pl = _planner(rows)
    with pytest.raises(CoordinateOutOfBoundsError) as ei:
        pl.plan(Point(-0.5, 0.5), Point(3.5, 3.5))
    assert ei.value.code == "coordinate_out_of_bounds"
    with pytest.raises(CoordinateOutOfBoundsError) as ei:
        pl.plan(Point(0.5, 0.5), Point(4.5, 0.5))
    log.info("关键判定：起/终点越界 -> code=%s", ei.value.code)
    assert ei.value.code == "coordinate_out_of_bounds"


def test_inflation_makes_narrow_gap_unreachable():
    rows = [[0] * 7 for _ in range(5)]
    for x in range(7):
        rows[2][x] = 1
    rows[2][3] = 0  # 1 格宽缺口
    g, pl = _planner(rows, inflation=1)
    with pytest.raises(NoPathError) as ei:
        pl.plan(Point(0.5, 0.5), Point(0.5, 4.5))
    log.info("关键判定：1 格窄通道 + 1 格膨胀 -> code=%s（保守拒绝）",
             ei.value.code)
    assert ei.value.code == "goal_unreachable"
    # 不膨胀时同地图可达（结论可复现、可区分）
    g0, pl0 = _planner(rows, inflation=0)
    path = pl0.plan(Point(0.5, 0.5), Point(0.5, 4.5))
    assert len(path) >= 2


def test_node_budget_timeout_distinguishable():
    n = 80
    rows = [[0] * n for _ in range(n)]
    g, pl = _planner(rows, inflation=0, max_expanded_nodes=10)
    with pytest.raises(PlanningTimeoutError) as ei:
        pl.plan(Point(0.5, 0.5), Point(n - 0.5, n - 0.5))
    log.info("关键判定：节点预算 10 -> code=%s", ei.value.code)
    assert ei.value.code == "planning_timeout"


def test_large_open_map_plans_fast():
    n = 300
    rows = [[0] * n for _ in range(n)]
    g, pl = _planner(rows, inflation=0)
    t0 = time.perf_counter()
    path = pl.plan(Point(0.5, 0.5), Point(n - 0.5, n - 0.5))
    dt = time.perf_counter() - t0
    log.info("大图 %dx%d 规划耗时 %.3fs，路径点数 %d", n, n, dt, len(path))
    assert dt < 2.0
    assert path[0] == Point(0.5, 0.5)


def test_diagonal_corner_cutting_forbidden():
    # 对角两侧均为障碍时不得穿墙角：构造需要绕行的对角阻挡
    rows = [[0] * 4 for _ in range(4)]
    rows[1][1] = 1
    rows[2][2] = 1
    g, pl = _planner(rows, inflation=0)
    path = pl.plan(Point(0.5, 0.5), Point(3.5, 3.5))
    assert pl.is_path_clear(path, 0.0) == -1
    log.info("对角障碍不穿墙角，路径 %s", [(p.x, p.y) for p in path])


def test_single_cell_empty_map_start_equals_goal():
    # 1x1 空地图、零足迹近似（足迹必须放得下由 footprint 校验负责）
    g = OccupancyGrid.from_raw([[0]], resolution=1.0)
    from nav.models import Point as P
    from nav.planner import GridPlanner
    from nav.models import PlannerParams as PP

    path = GridPlanner(g, PP(obstacle_inflation_cells=0)).plan(
        P(0.5, 0.5), P(0.5, 0.5)
    )
    log.info("1x1 空地图起终点重合 -> 路径 %s", [(p.x, p.y) for p in path])
    assert len(path) == 1


def test_open_empty_room_straight_diagonal():
    # 完全空旷大房间
    rows = [[0] * 20 for _ in range(20)]
    g, pl = _planner(rows)
    path = pl.plan(Point(0.5, 0.5), Point(19.5, 19.5))
    log.info("空旷房间 -> 压缩为 %d 段直线路径", len(path) - 1)
    assert len(path) == 2
    assert pl.is_path_clear(path, 0.0) == -1


def test_on_map_border_coordinate_in_bounds():
    g = OccupancyGrid.from_raw([[0] * 3 for _ in range(3)], resolution=1.0)
    # 右/上边界坐标本身越界（半开区间 [origin, origin+n*res)）
    with pytest.raises(CoordinateOutOfBoundsError):
        g.world_to_cell(3.0, 0.0)
    # 紧贴边界内侧合法
    cx, cy = g.world_to_cell(2.999, 2.999)
    assert (cx, cy) == (2, 2)
    log.info("贴边坐标 (2.999,2.999) 合法 -> (2,2)；边界 (3.0,*) 越界")
