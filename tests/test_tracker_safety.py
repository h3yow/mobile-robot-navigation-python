"""跟踪执行不变量：不穿障碍、不超速/曲率，动态障碍触发 blocked。"""

from __future__ import annotations

import math

from app.gridmap import parse_map_input
from app.kinematics import default_kinematics
from app.planner import plan_global
from app.tracker import pure_pursuit_step, TrackerConfig
from tests.conftest import case_log


def _run(grid, kin, plan, state, n=400, goal_theta=None, cfg=None):
    cfg = cfg or TrackerConfig(lookahead=0.25, goal_tolerance=0.06)
    prog = 0
    max_v = max_k = max_w = 0.0
    blocked_at = None
    for step in range(n):
        sr, state, prog = pure_pursuit_step(
            grid, kin, plan, state, cfg=cfg, progress=prog, goal_theta=goal_theta)
        max_v = max(max_v, abs(sr.speed))
        max_k = max(max_k, abs(sr.curvature))
        max_w = max(max_w, abs(sr.angular_speed))
        assert not grid.disk_collides(sr.x, sr.y, kin.robot_radius), step
        assert abs(sr.speed) <= kin.max_linear_speed + 1e-9
        assert abs(sr.curvature) <= kin.max_curvature + 1e-9
        assert abs(sr.angular_speed) <= kin.max_angular_speed + 1e-9
        if sr.blocked:
            blocked_at = (step, sr.replan_reason)
            break
        if sr.reached:
            return state, step + 1, (max_v, max_k, max_w), prog
    return state, blocked_at, (max_v, max_k, max_w), prog


def test_drive_straight_respects_limits_and_reaches():
    g = parse_map_input(grid=["." * 30] * 6, resolution=0.1)
    kin = default_kinematics(max_linear_speed=1.0, max_angular_speed=2.0,
                             robot_radius=0.0)
    plan = plan_global(g, kin, (0.15, 0.25), (2.85, 0.25))
    state, steps, peaks, _ = _run(g, kin, plan, (0.15, 0.25, 0.0, 0.0))
    case_log("判定", steps=steps, peaks=tuple(round(p, 3) for p in peaks),
             final=(round(state[0], 3), round(state[1], 3)))
    assert steps is not None
    assert peaks[0] <= 1.0 + 1e-9


def test_turn_in_open_space_reaches_with_alignment():
    g = parse_map_input(grid=["." * 30] * 30, resolution=0.1)
    kin = default_kinematics(max_linear_speed=0.8, max_angular_speed=1.5,
                             robot_radius=0.0)
    plan = plan_global(g, kin, (0.5, 0.5), (0.5, 1.5))
    state, steps, peaks, _ = _run(g, kin, plan, (0.5, 0.5, 0.0, 0.0),
                                  goal_theta=math.pi / 2)
    case_log("判定", steps=steps, theta=round(state[2], 3))
    assert steps is not None
    assert abs(state[2] - math.pi / 2) < 0.05


def test_dynamic_obstacle_on_path_triggers_blocked():
    g0 = parse_map_input(grid=["." * 20] * 12, resolution=0.1)
    kin = default_kinematics(max_linear_speed=1.0, max_angular_speed=2.0,
                             robot_radius=0.0)
    plan = plan_global(g0, kin, (0.15, 0.15), (1.85, 1.05))
    state = (0.15, 0.15, 0.0, 0.0)
    cfg = TrackerConfig(lookahead=0.25, goal_tolerance=0.06)
    prog = 0
    placed = False
    blocked = None
    for step in range(60):
        if step == 10 and not placed:
            px, py = plan.world_path[min(prog + 5, len(plan.world_path) - 1)]
            col, row = g0.world_to_cell(px, py)
            buf = bytearray(g0.cells); buf[row * 20 + col] = 1
            g = type(g0)(g0.width, g0.height, g0.resolution, g0.origin_x,
                         g0.origin_y, bytes(buf), g0.notes)
            placed = True
            case_log("输入", dynamic_obstacle_cell=(col, row), at_step=step)
        grid = g if placed else g0
        sr, state, prog = pure_pursuit_step(
            grid, kin, plan, state, cfg=cfg, progress=prog)
        assert not grid.disk_collides(sr.x, sr.y, 0.0)
        if sr.blocked:
            blocked = (step, sr.replan_reason, sr.events)
            break
    case_log("判定", blocked=blocked)
    assert blocked is not None
    assert blocked[1] == "path_blocked_ahead"
