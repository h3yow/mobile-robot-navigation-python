"""全局规划：可达/不可达区分、窄通道、起终点非法与大地图性能。"""

from __future__ import annotations

import math
import random
import time

import pytest

from app.errors import NavigationError
from app.gridmap import parse_map_input
from app.kinematics import default_kinematics
from app.planner import (
    obstacle_rectangle_distance, plan_global, PlannerConfig,
)
from tests.conftest import box_with_gap, case_log


KIN0 = lambda: default_kinematics(robot_radius=0.0)


def _show_error(fn, code, label):
    with pytest.raises(NavigationError) as exc:
        fn()
    case_log("判定", case=label, code=exc.value.code.value, msg=exc.value.message[:60])
    assert exc.value.code.value == code


def test_path_avoids_obstacles():
    g = parse_map_input(grid=["....", ".##.", ".##.", "...."], resolution=0.1)
    p = plan_global(g, KIN0(), (0.05, 0.05), (0.35, 0.35))
    coll = sum(g.disk_collides(x, y, 0.0) for x, y in p.world_path)
    case_log("判定", path=p.cell_path, collisions=coll)
    assert coll == 0
    # 8 邻接路径相邻点距离为 0.1 或 0.1*sqrt2
    for (x1, y1), (x2, y2) in zip(p.world_path, p.world_path[1:]):
        assert abs(math.hypot(x2 - x1, y2 - y1) - 0.1) < 1e-9 or \
               abs(math.hypot(x2 - x1, y2 - y1) - 0.1 * math.sqrt(2)) < 1e-9


def test_invalid_start_goal_reasons_are_distinct():
    g = parse_map_input(grid=["....", ".##.", ".##.", "...."], resolution=0.1)
    _show_error(lambda: plan_global(g, KIN0(), (-0.1, 0.2), (0.35, 0.35)),
                "COORDINATE_OUT_OF_BOUNDS", "start oob")
    _show_error(lambda: plan_global(g, KIN0(), (0.5, 0.2), (0.35, 0.35)),
                "COORDINATE_OUT_OF_BOUNDS", "goal oob")
    _show_error(lambda: plan_global(g, KIN0(), (0.15, 0.25), (0.35, 0.35)),
                "START_IN_OBSTACLE", "start in wall")
    _show_error(lambda: plan_global(g, KIN0(), (0.05, 0.05), (0.15, 0.25)),
                "GOAL_IN_OBSTACLE", "goal in wall")


def test_unreachable_sealed_room():
    g = parse_map_input(grid=[
        "........", "..####..", "..#..#..", "..#..#..",
        "..#..#..", "..####..", "........",
    ], resolution=0.1)
    _show_error(lambda: plan_global(g, KIN0(), (0.05, 0.05), (0.35, 0.35)),
                "GOAL_UNREACHABLE", "sealed box")


def test_narrow_gap_radius_decision():
    rows = box_with_gap(w=20, h=13, gap_col=9)
    g = parse_map_input(grid=rows, resolution=0.1)
    p0 = plan_global(g, default_kinematics(robot_radius=0.0),
                     (0.5, 1.1), (1.5, 0.2))
    assert (9, 6) in p0.cell_path
    case_log("判定", point_robot_path=len(p0.cell_path))
    # 缝净宽 0.1m，r=0.06 过不去
    _show_error(
        lambda: plan_global(g, default_kinematics(robot_radius=0.06),
                            (0.5, 1.1), (1.5, 0.2)),
        "GOAL_UNREACHABLE", "fat robot")
    # r=0.049 恰好可过
    p49 = plan_global(g, default_kinematics(robot_radius=0.049),
                      (0.5, 1.1), (1.5, 0.2))
    case_log("判定", slim_robot_path=len(p49.cell_path))
    assert (9, 6) in p49.cell_path


def test_distance_transform_matches_brute_force():
    def brute(cells, w, h):
        out = []
        for r in range(h):
            for c in range(w):
                if cells[r * w + c] == 1:
                    out.append(0.0); continue
                best = float("inf")
                x, y = c + .5, h - 1 - r + .5
                for r2 in range(h):
                    for c2 in range(w):
                        if cells[r2 * w + c2] == 1:
                            x2, y2 = c2 + .5, h - 1 - r2 + .5
                            dx = max(abs(x - x2) - .5, 0)
                            dy = max(abs(y - y2) - .5, 0)
                            best = min(best, math.hypot(dx, dy))
                out.append(best)
        return out

    rng = random.Random(123)
    mismatches = 0
    for _ in range(60):
        w, h = rng.randint(1, 12), rng.randint(1, 12)
        cells = bytes(rng.randint(0, 1) for _ in range(w * h))
        if all(cells):
            continue
        got = obstacle_rectangle_distance(cells, w, h)
        want = brute(cells, w, h)
        mismatches += sum(1 for a, b in zip(got, want) if abs(a - b) > 1e-9)
    case_log("判定", brute_force_mismatches=mismatches)
    assert mismatches == 0


def test_large_map_plans_in_time_and_safely():
    rng = random.Random(0)
    rows = ["".join("." if rng.random() > 0.15 else "#" for _ in range(500))
            for _ in range(500)]
    t0 = time.time()
    g = parse_map_input(grid=rows, resolution=0.1)
    p = plan_global(g, KIN0(), (0.05, 0.05), (49.95, 49.95),
                    cfg=PlannerConfig(time_budget=3.0))
    elapsed = time.time() - t0
    coll = sum(g.disk_collides(x, y, 0.0) for x, y in p.world_path)
    case_log("判定", elapsed_s=round(elapsed, 3), path=len(p.cell_path),
             expansions=p.expansions, collisions=coll)
    assert elapsed < 3.0
    assert coll == 0
