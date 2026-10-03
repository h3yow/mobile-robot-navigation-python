"""向后兼容：不传任何新增推进参数时，逐步 step 的结果与升级前完全一致。

参考实现 _legacy_reference 独立复刻升级前（长时推进能力引入前）的单步语义：
  - 一阶直线积分：x += v·cos(θ)·dt，y += v·sin(θ)·dt，θ += ω·dt；
  - 仅对积分后位姿做足迹复核（半径 footprint_radius）；
  - 预测位姿触障或重规划失败 -> 零速 BLOCKED（不做部分推进）。
测试逐位姿/逐指令（1e-12）比对 session.step() 与该参考实现，
任何运动学/复核语义的回归都会立即暴露。
"""

from __future__ import annotations

import logging
import math
import time

from nav.controller import TrajectoryTracker
from nav.grid import OccupancyGrid, rasterize_rect
from nav.models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    Twist,
)
from nav.planner import GridPlanner
from nav.session import NavigationSession

log = logging.getLogger("test_step_compat")

LIMITS = RobotLimits(
    max_v=1.0, min_v=0.0, max_omega=2.0, max_curvature=3.0,
    max_acc=2.0, footprint_radius=0.3,
)
CParams = ControllerParams(dt=0.1)


def _euler(pose: Pose, cmd: Twist, dt: float) -> Pose:
    """升级前的一阶直线积分（参考实现的唯一运动学假设）。"""
    th = pose.theta + cmd.omega * dt
    while th > math.pi:
        th -= 2 * math.pi
    while th < -math.pi:
        th += 2 * math.pi
    return Pose(
        x=pose.x + cmd.v * math.cos(pose.theta) * dt,
        y=pose.y + cmd.v * math.sin(pose.theta) * dt,
        theta=th,
    )


class _LegacyReference:
    """升级前逐步执行语义的独立参考实现（只复用未改变语义的组件）。"""

    def __init__(self, rows, start: Pose, goal: Point):
        self.grid = OccupancyGrid.from_raw(rows, resolution=1.0)
        # 会话层对规划膨胀的下限保护（升级中未改变）：用户膨胀 + 足迹折算格数
        footprint_cells = math.ceil((LIMITS.footprint_radius + 1e-9) / 1.0)
        self.planner_params = PlannerParams(
            obstacle_inflation_cells=footprint_cells,
        )
        # 会话的跟踪器使用用户原始 ControllerParams（升级中未改变）
        self.tracker = TrajectoryTracker(LIMITS, CParams)
        self.dt = CParams.dt
        self.pose = start
        self.goal = goal
        planner = GridPlanner(self.grid, self.planner_params)
        self.path = planner.plan(Point(start.x, start.y), goal)
        self.path_index = 0
        self.prev_v = 0.0
        self.step_count = 0

    def step(self) -> tuple[Pose, Twist, bool, bool]:
        """返回 (pose, cmd, arrived, blocked)，语义与升级前 step 一致。"""
        planner = GridPlanner(self.grid, self.planner_params)
        blocked_idx = planner.is_path_clear(self.path, LIMITS.footprint_radius)
        if blocked_idx >= 0:
            try:
                self.path = planner.plan(
                    Point(self.pose.x, self.pose.y), self.goal
                )
                self.path_index = 0
            except Exception:
                return self.pose, Twist(0.0, 0.0), False, True
        if self.tracker.at_goal(self.pose, self.goal):
            return self.pose, Twist(0.0, 0.0), True, False
        aim, self.path_index = self.tracker.find_aim_point(
            self.pose, self.path, self.path_index
        )
        cmd = self.tracker.compute_command(self.pose, aim, self.prev_v)
        next_pose = _euler(self.pose, cmd, self.dt)
        if self.grid.footprint_blocked(
            next_pose.x, next_pose.y, LIMITS.footprint_radius
        ):
            return self.pose, Twist(0.0, 0.0), False, True
        self.pose = next_pose
        self.prev_v = cmd.v
        self.step_count += 1
        arrived = self.tracker.at_goal(self.pose, self.goal)
        return self.pose, cmd, arrived, False


def _session(rows, start: Pose, goal: Point) -> NavigationSession:
    g = OccupancyGrid.from_raw(rows, resolution=1.0)
    s = NavigationSession(
        "compat", g, LIMITS, PlannerParams(obstacle_inflation_cells=0),
        ControllerParams(dt=0.1), time.time(),
    )
    s.plan(start, goal, time.time())
    return s


def _run_and_compare(name, rows, start, goal, n_steps=400,
                     obstacle_at=None, rects=None):
    s = _session(rows, start, goal)
    ref = _LegacyReference(rows, start, goal)
    max_omega = 0.0
    for i in range(n_steps):
        if obstacle_at is not None and i == obstacle_at:
            s.update_dynamic_obstacles(rects, time.time())
            for xmin, ymin, xmax, ymax in rects:
                mask = rasterize_rect(
                    ref.grid.width, ref.grid.height, ref.grid.resolution,
                    ref.grid.origin_x, ref.grid.origin_y,
                    xmin, ymin, xmax, ymax,
                )
                # 用公开合并接口重建参考实现的有效栅格
                ref.grid = ref.grid.merged(mask)
            log.info("状态迁移依据：step=%d 注入动态障碍 %s", i, rects)
        r = s.step(time.time())
        rp, rc, ra, rb = ref.step()
        max_omega = max(max_omega, abs(r.cmd.omega))
        assert abs(r.pose.x - rp.x) < 1e-12, (name, i, r.pose, rp)
        assert abs(r.pose.y - rp.y) < 1e-12, (name, i, r.pose, rp)
        assert abs(r.pose.theta - rp.theta) < 1e-12, (name, i)
        assert abs(r.cmd.v - rc.v) < 1e-12, (name, i)
        assert abs(r.cmd.omega - rc.omega) < 1e-12, (name, i)
        assert r.arrived == ra and r.blocked == rb, (name, i)
        if r.arrived:
            break
    log.info("关键判定：%s，%d 步逐位姿/指令与参考实现一致（max|omega|=%.3f）",
             name, s.step_count, max_omega)
    return s, max_omega


def test_step_matches_legacy_straight_corridor():
    rows = [[0] * 20 for _ in range(12)]
    s, _ = _run_and_compare(
        "直线走廊", rows, Pose(0.5, 5.5, 0.0), Point(19.5, 5.5)
    )
    assert s.status.value == "arrived"


def test_step_matches_legacy_with_turning():
    # 斜向目标 + 初始朝向偏差：保证存在明显角速度，
    # 从而能区分“一阶直线积分（老）”与“精确圆弧积分（新）”
    rows = [[0] * 12 for _ in range(9)]
    s, max_omega = _run_and_compare(
        "斜线转弯", rows, Pose(0.5, 0.5, 0.3), Point(11.5, 8.5)
    )
    assert max_omega > 0.01, "场景必须含明显转向，否则无法捕获积分语义回归"
    assert s.status.value == "arrived"


def test_step_matches_legacy_backward_start():
    rows = [[0] * 12 for _ in range(9)]
    s, max_omega = _run_and_compare(
        "背向起步", rows, Pose(2.0, 4.5, math.pi), Point(10.5, 4.5)
    )
    assert max_omega > 0.5  # 含原地转向段
    assert s.status.value == "arrived"


def test_step_matches_legacy_with_dynamic_replan():
    # 动态障碍触发自动重规划的路径也与老行为一致
    rows = [[0] * 20 for _ in range(12)]
    s, _ = _run_and_compare(
        "动态绕障", rows, Pose(0.5, 5.5, 0.0), Point(19.5, 5.5),
        obstacle_at=15, rects=[[6.0, 0.0, 7.0, 8.0]],
    )
    assert s.status.value == "arrived"
    log.info("状态迁移：动态障碍 -> 自动重规划 -> 到达，全程与参考实现一致")
