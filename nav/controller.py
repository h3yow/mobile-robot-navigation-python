"""局部轨迹跟踪：纯追踪式控制器，输出满足 v / omega / 曲率约束的指令。

运动学模型（差速驱动）：
    x'   = x + v cos(theta) dt
    y'   = y + v sin(theta) dt
    theta' = wrap(theta + omega dt)

速度包络（RobotLimits）：
    min_v <= v <= max_v, |omega| <= max_omega,
    |omega / v| <= max_curvature（平移时），|v - v_prev| <= max_acc dt。
"""

from __future__ import annotations

import math

from .models import (
    ControllerParams,
    Point,
    Pose,
    RobotLimits,
    Twist,
)


def _wrap_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class TrajectoryTracker:
    """差速运动学轨迹跟踪器。"""

    def __init__(self, limits: RobotLimits, params: ControllerParams) -> None:
        self.limits = limits
        self.params = params

    def find_aim_point(
        self, pose: Pose, path: list[Point], path_index: int
    ) -> tuple[Point, int]:
        """在路径上选择前视点，并推进路径下标。

        规则：从当前下标开始，丢弃已落在身后的路径点（点在机体坐标系 x<0），
        再向前取 lookahead_cells 个点作为前视点；到达路径末端时直接瞄准终点。
        """
        if not path:
            raise ValueError("find_aim_point: 空路径")
        idx = max(0, min(path_index, len(path) - 1))

        # 推进：当前点已非常接近，或落在机体后方
        while idx < len(path) - 1:
            pt = path[idx]
            dx, dy = pt.x - pose.x, pt.y - pose.y
            distance = math.hypot(dx, dy)
            if distance <= self.params.goal_tolerance:
                idx += 1
                continue
            # 机体坐标系下的纵向坐标
            local_x = dx * math.cos(-pose.theta) - dy * math.sin(-pose.theta)
            if local_x < -self.params.goal_tolerance:
                idx += 1
                continue
            break

        target_idx = min(len(path) - 1, idx + self.params.lookahead_cells)
        return path[target_idx], idx

    def compute_command(
        self, pose: Pose, aim: Point, previous_v: float = 0.0
    ) -> Twist:
        """根据位姿与前视点计算受运动学约束的速度指令。

        先按角度偏差决定角速度与降速系数（大角度差时原地转向），
        再按距离比例给线速度，最后依次施加加速度、速度、角速度、曲率限幅。
        """
        dx, dy = aim.x - pose.x, aim.y - pose.y
        distance = math.hypot(dx, dy)
        angle_err = _wrap_angle(math.atan2(dy, dx) - pose.theta)

        dt = self.params.dt
        # 1) 期望角速度：比例控制
        omega = self.params.k_angular * angle_err

        # 2) 期望线速度：正对目标时加速，背对目标时不前进（必要时倒车受限->原地转）
        alignment = max(0.0, math.cos(angle_err))
        desired_v = self.params.k_linear * distance * alignment
        # 角度偏差大时降速，避免超调
        desired_v *= alignment

        # 3) 加速度限幅（只限制前进方向的变化率）
        max_dv = self.limits.max_acc * dt
        desired_v = max(previous_v - max_dv, min(previous_v + max_dv, desired_v))

        # 4) 全包络限幅（速度/角速度/曲率）
        v, omega = self.limits.clamp_command(desired_v, omega)

        # 5) 数值稳定：极小速度归零，避免产生“原地漂移”
        if abs(v) < 1e-9:
            v = 0.0
        if abs(omega) < 1e-9:
            omega = 0.0
        return Twist(v=v, omega=omega)

    def advance(self, pose: Pose, cmd: Twist) -> Pose:
        """按 dt 做一阶积分（差速模型），角度归一化到 [-pi, pi]。"""
        dt = self.params.dt
        return Pose(
            x=pose.x + cmd.v * math.cos(pose.theta) * dt,
            y=pose.y + cmd.v * math.sin(pose.theta) * dt,
            theta=_wrap_angle(pose.theta + cmd.omega * dt),
        )

    def at_goal(self, pose: Pose, goal: Point) -> bool:
        return math.hypot(goal.x - pose.x, goal.y - pose.y) <= self.params.goal_tolerance
