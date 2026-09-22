"""会话级仿真：动态障碍重规划、全程无碰撞/限速、BLOCKED 与恢复。"""

from __future__ import annotations

import logging
import math
import time

import pytest

from nav.grid import OccupancyGrid
from nav.models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    SessionStatus,
)
from nav.session import NavigationSession

log = logging.getLogger("test_session")


def _session(rows, resolution=1.0, origin=(0.0, 0.0), **lim_kw):
    g = OccupancyGrid.from_raw(rows, resolution=resolution, origin=origin)
    lim = RobotLimits(
        max_v=lim_kw.get("max_v", 1.0), min_v=0.0,
        max_omega=lim_kw.get("max_omega", 2.0),
        max_curvature=lim_kw.get("max_curvature", 3.0),
        max_acc=2.0, footprint_radius=lim_kw.get("footprint_radius", 0.3),
    )
    return NavigationSession(
        "test", g, lim,
        PlannerParams(obstacle_inflation_cells=0),
        ControllerParams(dt=0.1),
        time.time(),
    )


def _drive(s, max_steps=600, obstacle_at=None, clear_at=None, rects=None):
    """驱动仿真并收集每一步，返回 (arrived, history)。"""
    history = []
    replanned_at = []
    blocked_at = []
    for i in range(max_steps):
        if obstacle_at is not None and i == obstacle_at:
            s.update_dynamic_obstacles(rects, time.time())
            log.info("状态迁移依据：step=%d 注入动态障碍 %s", i, rects)
        if clear_at is not None and i == clear_at:
            s.update_dynamic_obstacles([], time.time())
            log.info("状态迁移依据：step=%d 动态障碍清空", i)
        r = s.step(time.time())
        history.append(r)
        if r.replanned:
            replanned_at.append(i)
        if r.blocked:
            blocked_at.append(i)
        if r.arrived:
            return True, history, replanned_at, blocked_at
    return False, history, replanned_at, blocked_at


def test_dynamic_obstacle_triggers_replan_and_avoidance():
    # 20x12 空场，沿 y=5.5 直行；step15 注入 x[6,7]、y[0,8] 的墙，留上方缺口
    rows = [[0] * 20 for _ in range(12)]
    s = _session(rows)
    s.plan(Pose(0.5, 5.5, 0.0), Point(19.5, 5.5), time.time())
    arrived, hist, replans, blocked = _drive(
        s, obstacle_at=15, rects=[[6.0, 0.0, 7.0, 8.0]]
    )
    log.info("关键判定：重规划发生于 steps=%s；阻塞步数=%d；到达=%s",
             replans, len(blocked), arrived)
    assert replans, "动态障碍必须触发至少一次重规划"
    assert arrived
    # 全程位姿都不得触障（动态+静态）
    for i, r in enumerate(hist):
        assert not s.effective_grid.footprint_blocked(
            r.pose.x, r.pose.y, s.limits.footprint_radius - 1e-6
        ), f"step {i} 位姿触障 {r.pose}"
    # 绕障期间机器人必须越过墙顶（y>8）
    assert max(r.pose.y for r in hist) > 7.5


def test_trajectory_never_violates_kinematic_limits():
    rows = [[0] * 12 for _ in range(9)]
    s = _session(rows)
    s.plan(Pose(0.5, 0.5, 0.3), Point(11.5, 8.5), time.time())
    arrived, hist, _, _ = _drive(s, max_steps=400)
    assert arrived
    prev_v = 0.0
    for i, r in enumerate(hist):
        c = r.cmd
        assert 0.0 - 1e-9 <= c.v <= s.limits.max_v + 1e-9
        assert abs(c.omega) <= s.limits.max_omega + 1e-9
        if abs(c.v) > 1e-6:
            assert abs(c.omega / c.v) <= s.limits.max_curvature + 1e-9, (i, c)
        assert abs(c.v - prev_v) <= s.limits.max_acc * 0.1 + 1e-9
        prev_v = c.v
    log.info("全程 %d 步均满足 v/omega/曲率/加速度约束", len(hist))


def test_sealed_dynamic_obstacle_blocks_then_recovers():
    rows = [[0] * 12 for _ in range(6)]
    s = _session(rows)
    s.plan(Pose(0.5, 2.5, 0.0), Point(11.5, 2.5), time.time())
    arrived, hist, replans, blocked = _drive(
        s, obstacle_at=20, clear_at=80, rects=[[3.0, 0.0, 4.0, 6.0]], max_steps=400
    )
    first_block = blocked[0]
    assert s.status == SessionStatus.ARRIVED and arrived
    # BLOCKED 期间必须零速且位姿冻结
    blk_steps = [hist[i] for i in blocked]
    assert all(r.cmd.v == 0.0 and r.cmd.omega == 0.0 for r in blk_steps)
    log.info("状态迁移：step=%d 进入 BLOCKED（零速），障碍清空后自动恢复并到达",
             first_block)


def test_step_before_plan_rejected():
    rows = [[0] * 4 for _ in range(4)]
    s = _session(rows)
    from nav.errors import SessionStateError

    with pytest.raises(SessionStateError) as ei:
        s.step(time.time())
    log.info("关键判定：未规划即步进 -> code=%s", ei.value.code)
    assert ei.value.code == "session_state_error"


def test_arrived_session_is_idempotent_on_extra_steps():
    rows = [[0] * 5 for _ in range(5)]
    s = _session(rows)
    s.plan(Pose(0.5, 0.5, 0.0), Point(4.5, 4.5), time.time())
    arrived, hist, _, _ = _drive(s, max_steps=300)
    assert arrived
    pose_after = s.pose
    for _ in range(3):
        r = s.step(time.time())  # 已到达后重复步进必须幂等：零速且停在原地
        assert r.arrived and r.cmd.v == 0.0
    assert math.hypot(s.pose.x - pose_after.x, s.pose.y - pose_after.y) < 1e-9
    log.info("到达后重复步进幂等：位姿保持 (%.3f,%.3f)", s.pose.x, s.pose.y)


def test_replan_after_static_change_manual():
    rows = [[0] * 10 for _ in range(8)]
    s = _session(rows)
    s.plan(Pose(0.5, 0.5, 0.0), Point(9.5, 7.5), time.time())
    # 手动更新障碍并显式重规划
    s.update_dynamic_obstacles([[4.0, 4.0, 6.0, 8.0]], time.time())
    path = s.replan(time.time())
    log.info("手动重规划 -> %d 点，状态=%s", len(path), s.status.value)
    assert s.status == SessionStatus.READY
    arrived, hist, _, blocked = _drive(s, max_steps=500)
    assert arrived
