"""长时推进（advance）：大跨度/高速度下不穿障、不越界、不超约束，部分推进与恢复。

日志按需求打印输入、关键判定与状态迁移依据。
"""

from __future__ import annotations

import logging
import math
import time

from nav.grid import OccupancyGrid
from nav.models import (
    ControllerParams,
    PlannerParams,
    Pose,
    Point,
    RobotLimits,
    SessionStatus,
)
from nav.session import NavigationSession

log = logging.getLogger("test_advance")


def _session(rows, resolution=1.0, origin=(0.0, 0.0), **lim):
    g = OccupancyGrid.from_raw(rows, resolution=resolution, origin=origin)
    limits = RobotLimits(
        max_v=lim.get("max_v", 3.0), min_v=0.0,
        max_omega=lim.get("max_omega", 2.0),
        max_curvature=lim.get("max_curvature", 3.0),
        max_acc=lim.get("max_acc", 4.0),
        footprint_radius=lim.get("footprint_radius", 0.3),
    )
    return NavigationSession(
        "adv", g, limits,
        PlannerParams(obstacle_inflation_cells=0),
        ControllerParams(dt=0.1), time.time(),
    )


def _assert_swept_collision_free(s, grid, radius):
    """把轨迹中每条记录的 swept 串起来，逐点做足迹碰撞复核。"""
    tr = s.get_trajectory()
    for rec in tr["records"]:
        for p in rec["swept"]:
            assert not grid.footprint_blocked(
                p["x"], p["y"], radius - 1e-7
            ), f"step {rec['step']} 扫掠点触障 {p}"


def _assert_limits(results, limits, dt_used=0.5):
    prev_v = 0.0
    for r in results:
        c = r.cmd
        assert limits.min_v - 1e-9 <= c.v <= limits.max_v + 1e-9
        assert abs(c.omega) <= limits.max_omega + 1e-9
        if abs(c.v) > 1e-7:
            assert abs(c.omega / c.v) <= limits.max_curvature + 1e-9
        # 加速度按本微步实际时长
        assert abs(c.v - prev_v) <= limits.max_acc * max(r.elapsed, 1e-6) + 1e-7
        prev_v = c.v


def test_large_duration_advance_along_corridor_reaches_goal():
    # 100m 长直走廊；一次推进 30 秒、dt=1s、速度上限 5m/s（单步位移 5m）
    rows = [[0] * 200 for _ in range(12)]
    g = OccupancyGrid.from_raw(rows, resolution=0.5)
    s = NavigationSession(
        "adv", g,
        RobotLimits(5.0, 0.0, 2.0, 3.0, 5.0, 0.3),
        PlannerParams(), ControllerParams(dt=0.1), time.time(),
    )
    s.plan(Pose(0.5, 2.5, 0.0), Point(99.0, 2.5), time.time())
    log.info("输入：duration=30 dt=1.0 max_v=5.0，走廊 100m，起点(0.5,2.5)")
    t0 = time.perf_counter()
    res = s.advance(30.0, time.time(), dt=1.0, max_v=5.0)
    wall = time.perf_counter() - t0
    log.info(
        "关键判定：arrived=%s steps=%d sim_t=%.2f wall=%.3fs 终态=%s",
        res.arrived, res.steps_executed, res.simulated_duration,
        wall, res.status.value,
    )
    assert res.arrived and res.completed
    assert res.steps_executed <= 25  # 大跨度只需少量微步
    assert wall < 2.0
    _assert_swept_collision_free(s, g, 0.3)
    _assert_limits(res.results, s.limits)
    log.info("状态迁移依据：100m 走廊以 %d 个微步走完，全程轨迹无碰撞、满足约束",
             res.steps_executed)


def test_big_step_never_tunnels_through_thin_dynamic_wall():
    # 5cm 厚动态墙 << 单步位移 5m：任何采样间隙都不得穿过
    rows = [[0] * 200 for _ in range(12)]
    g = OccupancyGrid.from_raw(rows, resolution=0.5)
    s = NavigationSession(
        "adv", g,
        RobotLimits(5.0, 0.0, 2.0, 3.0, 5.0, 0.3),
        PlannerParams(), ControllerParams(dt=0.1), time.time(),
    )
    s.plan(Pose(0.5, 2.5, 0.0), Point(99.0, 2.5), time.time())
    s.update_dynamic_obstacles([[50.0, 0.0, 50.05, 6.0]], time.time())
    log.info("输入：在 x=50.00..50.05 注入 5cm 厚墙，advance dt=1s v<=5")
    res = s.advance(30.0, time.time(), dt=1.0, max_v=5.0)
    log.info("关键判定：blocked=%s 停止位 x=%.3f（须在墙前 >=0.3m）steps=%d",
             res.blocked, s.pose.x, res.steps_executed)
    assert res.blocked and not res.arrived
    assert s.pose.x <= 50.0 - 0.3, s.pose.x
    _assert_swept_collision_free(s, s.effective_grid, 0.3)
    log.info("状态迁移：->BLOCKED，零穿障；%s", res.reason)


def test_narrow_gap_with_high_speed_is_safely_stopped_and_session_usable():
    # 带 3 格缺口的墙：激进参数（dt=0.5s、v<=3）在窄口前被安全拦住，
    # 不穿墙、不超约束；会话状态完整保留，可重新规划/继续。
    rows = [[0] * 30 for _ in range(10)]
    for x in range(30):
        if x not in (15, 16, 17):
            rows[4][x] = 1
    s = _session(rows)
    s.plan(Pose(0.5, 1.5, 0.0), Point(29.5, 7.5), time.time())
    r1 = s.advance(8.0, time.time(), dt=0.5, max_v=3.0)
    log.info("激进推进：blocked=%s 位置=(%.2f,%.2f) sim_t=%.2f（在窄口前安全停住）",
             r1.blocked, s.pose.x, s.pose.y, r1.simulated_duration)
    assert r1.blocked
    _assert_swept_collision_free(s, s.effective_grid, 0.3)
    _assert_limits(r1.results, s.limits)
    assert s.goal is not None and len(s.path) > 0  # 目标与路径保留

    # 会话未卡死：reset 清空状态/轨迹后，可在新环境重新规划并继续推进
    s.reset(time.time())
    s.base_grid = OccupancyGrid.from_raw(
        [[0] * 30 for _ in range(10)], resolution=1.0
    )
    s.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    r2 = s.advance(40.0, time.time(), dt=0.2, max_v=1.0)
    log.info("重置到开阔环境继续推进：arrived=%s steps=%d", r2.arrived, r2.steps_executed)
    assert r2.arrived
    assert s.status == SessionStatus.ARRIVED


def test_partial_arc_advances_to_wall_then_stops():
    """单元级验证：一个大 dt 微步只有前半段可通行时，二分停在墙前安全位置。

    构造：路径检查/到点判定都不拦截（目标在墙这侧），仅由“整段扫掠碰撞复核”
    触发部分推进。机器人距墙 ~0.85m，v=1，dt=2（整步位移 2m），
    因此只能走前 ~0.5 秒。
    """
    rows = [[0] * 20 for _ in range(20)]
    for x in range(20):
        rows[10][x] = 1  # y=10 薄墙
    g = OccupancyGrid.from_raw(rows, resolution=1.0)
    lim = RobotLimits(1.0, 0.0, 2.0, 3.0, 2.0, 0.3)
    s = NavigationSession(
        "arc", g, lim, PlannerParams(0), ControllerParams(dt=0.1), time.time()
    )
    s._rebuild_effective_grid()
    s.pose = Pose(2.0, 8.85, math.pi / 2)  # 距墙边界 1.15m
    s.goal = Point(2.0, 9.6)
    s.path = [Point(2.0, 8.85), Point(2.0, 9.6)]
    s.path_index = 0
    log.info("输入：墙在 y=10，机器人 y=8.85 朝墙，v=1，dt=2（位移2m）")
    r = s._microstep(2.0, None, time.time())
    log.info("关键判定：blocked=%s elapsed=%.3f 停在 y=%.3f（须 < 9.7）",
             r.blocked, r.elapsed, s.pose.y)
    assert r.blocked
    assert 0.0 < r.elapsed < 2.0
    assert s.pose.y < 10.0 - 0.3, s.pose.y
    _assert_swept_collision_free(s, g, 0.3)


def test_sealed_wall_advances_to_closest_safe_then_recovers():
    """端到端：封死动态墙 -> 尽量推进到墙前最后安全位置停住（不穿墙、状态保留、
    原因可解释）-> 清障后继续推进到达。"""
    rows = [[0] * 40 for _ in range(5)]
    s = _session(rows, max_v=1.0, max_omega=2.0, max_curvature=3.0,
                 max_acc=2.0, footprint_radius=0.3)
    s.plan(Pose(0.5, 2.5, 0.0), Point(39.5, 2.5), time.time())
    s.update_dynamic_obstacles([[3.0, 0.0, 4.0, 5.0]], time.time())
    log.info("输入：x=3..4 整面封死墙（不可绕行），起点 x=0.5，advance 10s")
    res = s.advance(10.0, time.time(), dt=2.0, max_v=1.0)
    log.info("关键判定：blocked=%s sim_t=%.2f 停在 x=%.3f（墙前 %.3f m）reason=%s",
             res.blocked, res.simulated_duration, s.pose.x,
             3.0 - s.pose.x, res.reason)
    # 没有一步都不走：从 0.5 推进到墙前最后安全位置（>2.0），
    # 且始终与墙保持至少一个足迹半径
    assert res.blocked and not res.arrived
    assert s.pose.x > 2.0, s.pose.x
    assert s.pose.x <= 3.0 - 0.3 + 1e-9, s.pose.x
    assert s.status == SessionStatus.BLOCKED
    assert res.simulated_duration < 10.0  # 没走满：被墙拦停
    assert "安全位置" in res.reason
    # 停住原因可解释：note 说明重规划失败且无法绕行
    assert "重规划失败" in s.last_note and "无法绕行" in s.last_note
    log.info("状态迁移：->BLOCKED，停住依据：%s", s.last_note)
    _assert_swept_collision_free(s, s.effective_grid, 0.3)
    _assert_limits(res.results, s.limits)

    # 会话未卡死：停在原地再推进仍是安全零速/微动，不会穿墙
    r_again = s.advance(2.0, time.time(), dt=2.0, max_v=1.0)
    assert r_again.blocked and not r_again.arrived
    assert s.pose.x <= 3.0 - 0.3 + 1e-9
    _assert_swept_collision_free(s, s.effective_grid, 0.3)

    # 状态保留、不卡死：清空障碍后继续推进到达
    s.update_dynamic_obstacles([], time.time())
    r2 = s.advance(60.0, time.time(), dt=0.2, max_v=1.0)
    assert r2.arrived
    _assert_swept_collision_free(s, s.effective_grid, 0.3)
    log.info("状态迁移：BLOCKED -> 清障 -> %s（会话可继续，未卡死）", s.status.value)


def test_partial_microstep_accel_computed_over_actual_elapsed():
    """部分推进微步：上报指令的加速度必须按本步实际执行时长核算。

    构造：max_v=0.5、max_acc=0.4、dt=2.0。按整段 dt 限幅时指令 v 可到 0.5，
    但墙前只能安全走 ~1.2s，按实际时长预算仅 0.48——若按整段时长上报，
    折算加速度 0.5/1.2≈0.42 > 0.4 超限（修复前的行为）。
    """
    rows = [[0] * 20 for _ in range(20)]
    for x in range(20):
        rows[10][x] = 1  # y=10 薄墙
    g = OccupancyGrid.from_raw(rows, resolution=1.0)
    lim = RobotLimits(0.5, 0.0, 2.0, 3.0, 0.4, 0.3)
    s = NavigationSession(
        "acc", g, lim, PlannerParams(0), ControllerParams(dt=0.1), time.time()
    )
    s._rebuild_effective_grid()
    s.pose = Pose(2.0, 9.0, math.pi / 2)
    s.goal = Point(2.0, 9.65)
    s.path = [Point(2.0, 9.0), Point(2.0, 9.65)]
    s.path_index = 0
    log.info("输入：墙 y=10，机器人 y=9.0 朝墙，max_v=0.5 max_acc=0.4 dt=2.0")
    r = s._microstep(2.0, None, time.time())
    log.info(
        "关键判定：blocked=%s elapsed=%.3f v=%.4f，加速度预算 %.4f，折算 %.4f",
        r.blocked, r.elapsed, r.cmd.v,
        lim.max_acc * r.elapsed, abs(r.cmd.v) / max(r.elapsed, 1e-9),
    )
    assert r.blocked and 0.0 < r.elapsed < 2.0
    # 核心断言：按本步实际执行时长折算的加速度不超上限
    assert abs(r.cmd.v - 0.0) <= lim.max_acc * r.elapsed + 1e-9
    # 确实推进了（不是零速），且停点安全（足迹不穿墙）
    assert s.pose.y > 9.0 and s.pose.y < 10.0 - 0.3 + 1e-9
    _assert_swept_collision_free(s, g, 0.3)
    # 停住原因可解释
    assert "加速度" in s.last_note or "触障" in s.last_note
    log.info("状态迁移：->BLOCKED；%s", s.last_note)


def test_advance_partial_steps_respect_accel_end_to_end():
    """端到端：advance 中的部分微步同样满足按实际时长核算的加速度约束。"""
    rows = [[0] * 40 for _ in range(5)]
    s = _session(rows, max_v=1.0, max_omega=2.0, max_curvature=3.0,
                 max_acc=0.2, footprint_radius=0.3)
    s.plan(Pose(0.5, 2.5, 0.0), Point(39.5, 2.5), time.time())
    s.update_dynamic_obstacles([[3.0, 0.0, 4.0, 5.0]], time.time())
    log.info("输入：封死墙 x=3..4，max_acc=0.2，advance dt=2s（加速爬坡中被拦）")
    res = s.advance(10.0, time.time(), dt=2.0, max_v=1.0)
    log.info("关键判定：blocked=%s 停在 x=%.3f sim_t=%.2f",
             res.blocked, s.pose.x, res.simulated_duration)
    assert res.blocked
    # 加速爬坡途中被墙拦住：存在部分微步，且每步（含部分步）都满足
    # 速度/曲率/按实际时长核算的加速度约束
    assert any(0.0 < r.elapsed < 2.0 for r in res.results), "应包含部分微步"
    _assert_limits(res.results, s.limits)
    _assert_swept_collision_free(s, s.effective_grid, 0.3)
    assert s.pose.x <= 3.0 - 0.3 + 1e-9
    # 清障后会话可继续
    s.update_dynamic_obstacles([], time.time())
    r2 = s.advance(60.0, time.time(), dt=0.2, max_v=1.0)
    assert r2.arrived
    log.info("状态迁移：BLOCKED -> 清障 -> %s", s.status.value)


def test_advance_validates_inputs_and_requires_plan():
    from nav.errors import InvalidInputError, SessionStateError

    rows = [[0] * 6 for _ in range(6)]
    s = _session(rows)
    for bad in (0.0, -1.0, float("nan"), 1000.0):
        try:
            s.advance(bad, time.time())
        except InvalidInputError:
            continue
        raise AssertionError(f"duration={bad} 应被拒绝")
    log.info("非法 duration（0/负/NaN/超上限）全部 invalid_input")
    with pytest_raises(SessionStateError):
        s.advance(1.0, time.time())
    # 非法 dt/max_v
    s.plan(Pose(0.5, 0.5, 0.0), Point(5.5, 5.5), time.time())
    with pytest_raises(InvalidInputError):
        s.advance(1.0, time.time(), dt=0.0)
    with pytest_raises(InvalidInputError):
        s.advance(1.0, time.time(), max_v=-2.0)
    log.info("未规划拒绝推进；非法 dt/max_v 拒绝")


def pytest_raises(exc):
    import pytest
    return pytest.raises(exc)


def test_advance_default_params_match_single_step_semantics():
    # 不传 dt/max_v 时行为与逐步 step 使用相同默认参数
    rows = [[0] * 20 for _ in range(12)]
    s1 = _session(rows, max_v=1.0, max_omega=2.0, max_curvature=3.0)
    s2 = _session(rows, max_v=1.0, max_omega=2.0, max_curvature=3.0)
    start = Pose(0.5, 5.5, 0.0)
    goal = Point(19.5, 5.5)
    s1.plan(start, goal, time.time())
    s2.plan(Pose(0.5, 5.5, 0.0), goal, time.time())
    # s1 连续 30 个 step；s2 一次 advance 3.0 秒（默认 dt=0.1）
    for _ in range(30):
        s1.step(time.time())
    r = s2.advance(3.0, time.time())
    log.info("兼容性：30×step 位姿(%.4f,%.4f) vs advance(3s) 位姿(%.4f,%.4f)",
             s1.pose.x, s1.pose.y, s2.pose.x, s2.pose.y)
    assert abs(s1.pose.x - s2.pose.x) < 1e-9
    assert abs(s1.pose.y - s2.pose.y) < 1e-9
    assert abs(s1.pose.theta - s2.pose.theta) < 1e-9
    assert s2.step_count == s1.step_count
