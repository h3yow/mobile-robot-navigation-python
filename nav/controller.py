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

        规则：
          1. 从当前下标开始，若下一路径点比当前点更近（说明已越过当前点），
             则推进下标——但被越过点仍可能是最近点，不可盲目跳过；
          2. 前视点沿路径累计欧氏距离达到 lookahead 距离为止；
          3. 到达路径末端时直接瞄准终点。
        """
        if not path:
            raise ValueError("find_aim_point: 空路径")
        idx = max(0, min(path_index, len(path) - 1))

        # 推进到“剩余路径”上离机器人最近的顶点
        while idx < len(path) - 1:
            d_cur = math.hypot(path[idx].x - pose.x, path[idx].y - pose.y)
            d_next = math.hypot(path[idx + 1].x - pose.x, path[idx + 1].y - pose.y)
            if d_next <= d_cur:
                idx += 1
                continue
            break

        # 沿路径向前取累计距离达到前视距离的点
        lookahead = self.params.lookahead_cells  # 以米解释（默认参数与分辨率同量级）
        target_idx = idx
        accum = 0.0
        while target_idx < len(path) - 1:
            a, b = path[target_idx], path[target_idx + 1]
            seg = math.hypot(b.x - a.x, b.y - a.y)
            if accum + seg >= lookahead:
                target_idx += 1
                break
            accum += seg
            target_idx += 1
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

        # 1) 期望角速度
        #    小/中角度误差：纯追踪几何 kappa = 2*sin(alpha)/Ld，omega = v*kappa；
        #    大角度误差（背对目标）：先比例原地转向，不前进。
        if abs(angle_err) > math.pi / 2:
            omega = self.params.k_angular * angle_err
            desired_v = 0.0
        else:
            ld = max(distance, 1e-6)
            curvature = 2.0 * math.sin(angle_err) / ld
            # 期望线速度：距离比例 + 朝向对齐程度平滑降速
            desired_v = self.params.k_linear * distance * max(
                0.0, math.cos(angle_err)
            )
            omega = desired_v * curvature
            # 叠加小比例纠偏项（仅有平移速度时生效）
            omega += (
                0.5
                * self.params.k_angular
                * angle_err
                * min(1.0, desired_v)
            )
            # 曲率包络：超出 max_curvature 时等比收缩 omega，保持转弯方向
            if abs(omega) > 1e-12:
                scale = min(
                    1.0, self.limits.max_curvature * desired_v / abs(omega)
                )
                omega *= scale

        # 2) 加速度限幅
        max_dv = self.limits.max_acc * dt
        desired_v = max(previous_v - max_dv, min(previous_v + max_dv, desired_v))

        # 3) 全包络限幅（速度/角速度/曲率）
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
        return integrate_pose(pose, cmd, dt)

    def at_goal(self, pose: Pose, goal: Point) -> bool:
        return math.hypot(goal.x - pose.x, goal.y - pose.y) <= self.params.goal_tolerance


def integrate_pose(pose: Pose, cmd: Twist, dt: float) -> Pose:
    """差速模型积分（恒定 v/omega 的精确圆弧，直线为特例），角度归一化。"""
    theta_f = _wrap_angle(pose.theta + cmd.omega * dt)
    if abs(cmd.omega) < 1e-12:
        return Pose(
            x=pose.x + cmd.v * math.cos(pose.theta) * dt,
            y=pose.y + cmd.v * math.sin(pose.theta) * dt,
            theta=theta_f,
        )
    # 绕瞬时圆心 (cx, cy) 转动角度 omega*dt
    radius = cmd.v / cmd.omega
    cx = pose.x - radius * math.sin(pose.theta)
    cy = pose.y + radius * math.cos(pose.theta)
    return Pose(
        x=cx + radius * math.sin(theta_f),
        y=cy - radius * math.cos(theta_f),
        theta=theta_f,
    )


def integrate_pose_euler(pose: Pose, cmd: Twist, dt: float) -> Pose:
    """老版一阶积分（升级前 step 使用的运动学）：沿当前朝向直线平移 dt，
    角度在末端一次性更新并归一化。

    仅供 legacy 单步（step）路径使用，以保证不传新参数时位姿序列与
    升级前完全一致；advance/预演使用精确圆弧积分 integrate_pose。
    """
    return Pose(
        x=pose.x + cmd.v * math.cos(pose.theta) * dt,
        y=pose.y + cmd.v * math.sin(pose.theta) * dt,
        theta=_wrap_angle(pose.theta + cmd.omega * dt),
    )


def euler_poses(
    pose: Pose,
    cmd: Twist,
    dt: float,
    spacing: float,
    *,
    include_start: bool = True,
) -> list[Pose]:
    """按老版一阶积分模型采样 dt 内扫过的位姿序列（直线平移为特例）。

    与 arc_poses 同构，但运动模型为 integrate_pose_euler：中间点保持起始
    朝向做直线平移，终点朝向为归一化后的 theta + omega*dt。
    用于 legacy step 的轨迹记录，使 swept[-1] 与上报位姿严格一致。
    """
    travel = abs(cmd.v) * dt
    n = max(1, math.ceil(travel / spacing)) if travel > 0 else 1
    out: list[Pose] = []
    start_i = 0 if include_start else 1
    for i in range(start_i, n + 1):
        f = i / n
        out.append(
            pose if f <= 0.0 else integrate_pose_euler(pose, cmd, dt * f)
        )
    return out


def arc_poses(
    pose: Pose,
    cmd: Twist,
    dt: float,
    spacing: float,
    *,
    include_start: bool = True,
) -> list[Pose]:
    """采样一条指令在 dt 内扫过的位姿序列（恒 v/omega 的圆弧，直线为特例）。

    相邻采样点的机器人中心弦长不超过 spacing，因此用这些点做间隔 <= spacing
    的足迹复核即可保证整段运动不穿障（含大步长/薄墙场景）。
    include_start=False 时不返回起点，便于调用方把起点与上段终点去重。
    """
    travel = abs(cmd.v) * dt
    # 平移采样按弦长 spacing 控制；纯原地转向时位置不变，无需多点
    n = max(1, math.ceil(travel / spacing)) if travel > 0 else 1
    out: list[Pose] = []
    start_i = 0 if include_start else 1
    for i in range(start_i, n + 1):
        f = i / n
        out.append(
            pose if f <= 0.0 else integrate_pose(pose, cmd, dt * f)
        )
    return out
