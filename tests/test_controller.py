"""局部跟踪器：受约束指令生成、转向、到达与积分正确性。"""

from __future__ import annotations

import logging
import math

from nav.controller import TrajectoryTracker, _wrap_angle
from nav.models import ControllerParams, Point, Pose, RobotLimits, Twist

log = logging.getLogger("test_controller")


def _limits():
    return RobotLimits(
        max_v=1.0, min_v=-0.3, max_omega=1.5, max_curvature=2.0,
        max_acc=2.0, footprint_radius=0.2,
    )


def _run(pose, goal, steps=300):
    lim = _limits()
    tk = TrajectoryTracker(lim, ControllerParams(dt=0.1))
    path = [goal]
    idx = 0
    prev_v = 0.0
    history = []
    for i in range(steps):
        aim, idx = tk.find_aim_point(pose, path, idx)
        cmd = tk.compute_command(pose, aim, prev_v)
        # 全程约束断言
        assert lim.min_v - 1e-9 <= cmd.v <= lim.max_v + 1e-9
        assert abs(cmd.omega) <= lim.max_omega + 1e-9
        if abs(cmd.v) > 1e-6:
            assert abs(cmd.omega / cmd.v) <= lim.max_curvature + 1e-9
        assert abs(cmd.v - prev_v) <= lim.max_acc * 0.1 + 1e-9
        prev_v = cmd.v
        pose = tk.advance(pose, cmd)
        history.append((pose, cmd))
        if tk.at_goal(pose, goal):
            return pose, i + 1, history
    return pose, steps, history


def test_straight_line_arrives_within_constraints():
    pose, n, hist = _run(Pose(0, 0, 0), Point(3.0, 0))
    log.info("直行 3m：%d 步到达 (%.3f,%.3f)，最大 |omega|=%.3f",
             n, pose.x, pose.y, max(abs(c.omega) for _, c in hist))
    assert n < 300
    assert abs(pose.y) < 0.05


def test_turn_in_place_then_arrive():
    # 初始朝向背对行进方向，应先原地转向（v=0, omega!=0）再前进
    pose, n, hist = _run(Pose(0, 0, math.pi), Point(2.0, 0.5))
    spun = [c for _, c in hist[:20] if abs(c.v) < 1e-9 and abs(c.omega) > 1e-9]
    log.info("反向初始朝向：前 20 步含 %d 个原地转向指令，%d 步到达",
             len(spun), n)
    assert spun, "角度偏差大时必须允许原地转向"
    assert n < 300


def test_wrap_angle_normalized():
    # +/-pi 表示同一方向，数值上接受二者之一
    assert abs(abs(_wrap_angle(3 * math.pi)) - math.pi) < 1e-9
    assert abs(abs(_wrap_angle(-3 * math.pi)) - math.pi) < 1e-9
    assert abs(_wrap_angle(0.5)) < math.pi


def test_acceleration_ramp():
    lim = _limits()
    tk = TrajectoryTracker(lim, ControllerParams(dt=0.1))
    pose = Pose(0, 0, 0)
    cmd1 = tk.compute_command(pose, Point(10, 0), 0.0)
    cmd2 = tk.compute_command(pose, Point(10, 0), cmd1.v)
    log.info("加速约束：v0=0 -> v1=%.3f -> v2=%.3f（每步增量 <= %.2f）",
             cmd1.v, cmd2.v, lim.max_acc * 0.1)
    assert abs(cmd1.v) <= lim.max_acc * 0.1 + 1e-9
    assert abs(cmd2.v - cmd1.v) <= lim.max_acc * 0.1 + 1e-9
