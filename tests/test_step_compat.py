"""向后兼容：不传新增推进参数时，逐步执行结果与升级前逐位一致。

升级前的 step 语义（参考实现，见 _legacy_reference_step）：
  - 跟踪器使用**用户原始** controller 参数（前视距离不做下限保护）；
  - 欧拉积分：x += v·cos(θ)·dt，y += v·sin(θ)·dt；
  - 仅对积分后的预测位姿做端点足迹复核，整步或零速（无部分推进）。

回归背景：升级曾把 step 切到“前视下限保护 + 圆弧积分 + 扫掠复核 +
部分推进”的新微步实现，导致老调用每步位姿出现厘米级漂移
（本文件第一个用例的场景在第 38 步起可复现该漂移）。
"""

from __future__ import annotations

import logging
import math
import time

from nav.controller import TrajectoryTracker
from nav.grid import OccupancyGrid
from nav.models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    Twist,
)
from nav.session import NavigationSession

log = logging.getLogger("test_step_compat")


def _session(rows, resolution=1.0):
    g = OccupancyGrid.from_raw(rows, resolution=resolution)
    limits = RobotLimits(
        max_v=1.0, min_v=0.0, max_omega=2.0, max_curvature=3.0,
        max_acc=2.0, footprint_radius=0.3,
    )
    return NavigationSession(
        "compat", g, limits,
        PlannerParams(obstacle_inflation_cells=0),
        ControllerParams(dt=0.1),  # 默认前视 0.6（低于会话下限保护 1.5）
        time.time(),
    )


class _LegacyReference:
    """升级前 step 的独立参考实现（只读会话地图/路径，自行积分）。"""

    def __init__(self, session: NavigationSession) -> None:
        self.s = session
        # 关键：使用用户原始 controller 参数（无前视下限保护）
        self.tracker = TrajectoryTracker(
            session.limits, session.user_controller_params
        )
        self.pose = session.pose
        self.prev_v = 0.0
        self.path_index = 0

    def step(self) -> tuple[Pose, Twist, bool]:
        """返回 (新位姿, 指令, 是否被挡)。无动态障碍场景，路径不变。"""
        s = self.s
        tracker = self.tracker
        if s.effective_grid.footprint_blocked(
            self.pose.x, self.pose.y, s.limits.footprint_radius
        ):
            return self.pose, Twist(0.0, 0.0), True
        if tracker.at_goal(self.pose, s.goal):
            return self.pose, Twist(0.0, 0.0), False
        aim, self.path_index = tracker.find_aim_point(
            self.pose, s.path, self.path_index
        )
        cmd = tracker.compute_command(self.pose, aim, self.prev_v)
        nxt = tracker.advance(self.pose, cmd)  # 欧拉积分
        if s.effective_grid.footprint_blocked(
            nxt.x, nxt.y, s.limits.footprint_radius
        ):
            return self.pose, Twist(0.0, 0.0), True  # 整步拒绝，零速
        self.pose = nxt
        self.prev_v = cmd.v
        return nxt, cmd, False


def test_step_matches_pre_upgrade_reference_on_turning_run():
    # 12x9 空场，斜向目标 + 初始朝向偏置：全程 omega≠0，
    # 且默认前视 0.6 低于会话下限保护 1.5 —— 新旧语义在此场景会发散
    rows = [[0] * 12 for _ in range(9)]
    s = _session(rows)
    s.plan(Pose(0.5, 0.5, 0.3), Point(11.5, 8.5), time.time())
    log.info(
        "输入：起点(0.5,0.5,θ=0.3) 目标(11.5,8.5)，默认参数；"
        "用户前视=%.2f，会话下限保护后前视=%.2f",
        s.user_controller_params.lookahead_cells,
        s.controller_params.lookahead_cells,
    )
    # 前置断言：该场景确实能区分新旧语义（前视下限保护生效）
    assert s.user_controller_params.lookahead_cells < \
        s.controller_params.lookahead_cells

    ref = _LegacyReference(s)
    steps = 0
    for i in range(200):
        r = s.step(time.time())
        ref_pose, ref_cmd, ref_blocked = ref.step()
        # 逐位一致：位姿与指令都必须完全相等
        assert abs(r.pose.x - ref_pose.x) < 1e-12, (i, r.pose, ref_pose)
        assert abs(r.pose.y - ref_pose.y) < 1e-12, (i, r.pose, ref_pose)
        assert abs(r.pose.theta - ref_pose.theta) < 1e-12, (i, r.pose, ref_pose)
        assert abs(r.cmd.v - ref_cmd.v) < 1e-12, (i, r.cmd, ref_cmd)
        assert abs(r.cmd.omega - ref_cmd.omega) < 1e-12, (i, r.cmd, ref_cmd)
        assert r.blocked == ref_blocked
        steps += 1
        if r.arrived:
            break
    log.info(
        "关键判定：%d 步逐步执行与升级前参考实现逐位一致，"
        "终点 (%.4f,%.4f,%.4f)，arrived=%s",
        steps, s.pose.x, s.pose.y, s.pose.theta, s.status.value,
    )
    assert steps > 50  # 足够长的序列（旧回归在第 38 步即会发散）


def test_advance_without_new_params_matches_step_sequence():
    # 同一“会发散”的场景：advance(duration) 不传 dt/max_v
    # 必须与连续 step 逐位一致（默认参数 = 老调用方式）
    rows = [[0] * 12 for _ in range(9)]
    s1 = _session(rows)
    s2 = _session(rows)
    start = Pose(0.5, 0.5, 0.3)
    goal = Point(11.5, 8.5)
    s1.plan(start, goal, time.time())
    s2.plan(start, goal, time.time())
    n = 60
    for _ in range(n):
        s1.step(time.time())
    r = s2.advance(n * 0.1, time.time())
    log.info(
        "兼容性：%d×step 位姿(%.6f,%.6f,%.6f) vs advance(%.1fs 默认参数) "
        "位姿(%.6f,%.6f,%.6f)",
        n, s1.pose.x, s1.pose.y, s1.pose.theta,
        n * 0.1, s2.pose.x, s2.pose.y, s2.pose.theta,
    )
    assert r.steps_executed == n
    assert abs(s1.pose.x - s2.pose.x) < 1e-12
    assert abs(s1.pose.y - s2.pose.y) < 1e-12
    assert abs(s1.pose.theta - s2.pose.theta) < 1e-12
    assert s2.step_count == s1.step_count
    log.info("状态迁移依据：默认参数 advance 逐步等价于老 step 调用")


def test_step_response_schema_unchanged_with_default_elapsed():
    # 老调用的响应结构保持：StepResult 仅新增带默认值的 elapsed 字段
    rows = [[0] * 6 for _ in range(6)]
    s = _session(rows)
    s.plan(Pose(0.5, 0.5, 0.0), Point(5.5, 5.5), time.time())
    r = s.step(time.time())
    log.info("单步响应：pose=%s cmd=%s elapsed=%.2f arrived=%s blocked=%s",
             r.pose, r.cmd, r.elapsed, r.arrived, r.blocked)
    assert r.elapsed == s.user_controller_params.dt  # 完整微步 elapsed == dt
    assert not r.blocked and not r.arrived
