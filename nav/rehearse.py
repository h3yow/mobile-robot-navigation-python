"""只读预演：按给定路线与运行参数模拟推进，判定可行性而不改动会话状态。

预演是纯函数式计算：
  - 不修改任何会话的位姿/路径/计数/轨迹（调用方只传入栅格与路线副本）；
  - 给定相同输入（栅格/路线/起点/参数）必然得到相同结果，便于批量调参筛选；
  - 逐步执行与真实 advance 相同的跟踪/约束/扫掠碰撞逻辑，
    因此“预演可行”与“实际推进安全”使用同一套判定；
  - 额外统计全程机器人净空（到障碍表面/边界距离减足迹半径）的最小值。
"""

from __future__ import annotations

import math

from .controller import TrajectoryTracker, arc_poses
from .errors import InvalidInputError
from .grid import OccupancyGrid
from .models import (
    ControllerParams,
    Point,
    Pose,
    RehearsalReport,
    RobotLimits,
)
from .planner import GridPlanner


def rehearse(
    grid: OccupancyGrid,
    route: list[Point],
    start_pose: Pose,
    limits: RobotLimits,
    planner_params,
    controller_params: ControllerParams,
    *,
    duration: float | None = None,
    dt: float | None = None,
    max_v: float | None = None,
    max_microsteps: int = 20000,
) -> RehearsalReport:
    """对给定路线做一次只读模拟预演。

    参数：
      grid：静态有效栅格（如需含动态障碍，由调用方先合并后传入）；
      route：世界坐标路线点（至少 1 个；单元素路线把该点当作目标）；
      start_pose：起始位姿；
      duration：最多模拟时长（秒），None 表示走到到达/受阻为止；
      dt：微步间隔，None 用 controller_params.dt；
      max_v：临时速度上限，None 用 limits.max_v；
      max_microsteps：微步预算（大图/长路线保护）。

    返回 RehearsalReport：
      feasible=能在预算内走完全程且不触障/越界/超约束；
      failure_index/failure_step：首个问题出现在第几个路线段/第几个微步；
      min_clearance：全程最小机器人净空（米，<0 表示发生碰撞风险）。
    """
    # ---- 输入校验（不改状态，非法即拒绝）----
    if not isinstance(route, (list, tuple)) or len(route) == 0:
        raise InvalidInputError("预演路线必须为非空点列表")
    for i, p in enumerate(route):
        if not isinstance(p, Point):
            raise InvalidInputError(
                "路线元素必须为 Point", {"index": i, "got": type(p).__name__}
            )
    if dt is not None and (not isinstance(dt, (int, float)) or dt <= 0):
        raise InvalidInputError("预演 dt 必须为正数", {"value": dt})
    if max_v is not None and (not isinstance(max_v, (int, float)) or max_v <= 0):
        raise InvalidInputError("预演 max_v 必须为正数", {"value": max_v})
    if duration is not None and (
        not isinstance(duration, (int, float)) or duration <= 0
    ):
        raise InvalidInputError("预演 duration 必须为正数", {"value": duration})
    if not isinstance(max_microsteps, int) or max_microsteps <= 0:
        raise InvalidInputError("max_microsteps 必须为正整数")

    step_dt = float(dt) if dt is not None else controller_params.dt
    eff_limits = limits
    if max_v is not None and float(max_v) != limits.max_v:
        eff_limits = RobotLimits(
            max_v=float(max_v),
            min_v=limits.min_v,
            max_omega=limits.max_omega,
            max_curvature=limits.max_curvature,
            max_acc=limits.max_acc,
            footprint_radius=limits.footprint_radius,
        )
    cparams = ControllerParams(
        k_linear=controller_params.k_linear,
        k_angular=controller_params.k_angular,
        goal_tolerance=controller_params.goal_tolerance,
        lookahead_cells=controller_params.lookahead_cells,
        dt=step_dt,
    )
    tracker = TrajectoryTracker(eff_limits, cparams)

    # 局部可变模拟状态（与任何会话隔离）
    path = list(route)
    path_index = 0
    pose = Pose(start_pose.x, start_pose.y, start_pose.theta)
    previous_v = 0.0
    sim_time = 0.0
    goal = path[-1]

    spacing = min(grid.resolution * 0.25, 0.2)
    r_safe = eff_limits.footprint_radius + spacing * 0.5

    def _clearance(p: Pose) -> float:
        # 信息性净空：到障碍表面/边界距离减去“真实”足迹半径（不做采样膨胀）
        return grid.clearance_at(p.x, p.y) - eff_limits.footprint_radius

    min_clearance = _clearance(pose)

    # 起点安全性
    if grid.footprint_blocked(
        pose.x, pose.y, eff_limits.footprint_radius
    ):
        return RehearsalReport(
            feasible=False, completed=False, duration=0.0, steps=0,
            min_clearance=min_clearance, failure_index=0, failure_step=0,
            reason="起点位于障碍内或足迹越界",
        )

    # 路线静态通畅性快检：给出“第几个位置出问题”的路线级定位
    planner = GridPlanner(grid, planner_params)
    blocked_idx = planner.is_path_clear(path, eff_limits.footprint_radius)

    steps = 0
    if tracker.at_goal(pose, goal):
        return RehearsalReport(
            feasible=True, completed=True, duration=0.0, steps=0,
            min_clearance=min_clearance, failure_index=None, failure_step=None,
            reason="起点已在目标容差内",
        )

    while steps < max_microsteps:
        if duration is not None and sim_time + 1e-12 >= duration:
            # 给定预算时长走完但未到终点：不算“可安全走完全程”
            return RehearsalReport(
                feasible=False, completed=False, duration=sim_time, steps=steps,
                min_clearance=min_clearance,
                failure_index=_current_index(path, pose),
                failure_step=steps,
                reason=(
                    f"模拟 {duration:g} 秒预算用尽仍未到达终点"
                    "（可增大 duration 或 max_v）"
                ),
            )

        aim, path_index = tracker.find_aim_point(pose, path, path_index)
        cmd = tracker.compute_command(pose, aim, previous_v)

        cur_dt = step_dt
        if duration is not None:
            cur_dt = min(step_dt, duration - sim_time)
        # 重算一次以正确 dt 应用加速度限幅
        cparams2 = ControllerParams(
            k_linear=cparams.k_linear, k_angular=cparams.k_angular,
            goal_tolerance=cparams.goal_tolerance,
            lookahead_cells=cparams.lookahead_cells, dt=cur_dt,
        )
        tracker2 = TrajectoryTracker(eff_limits, cparams2)
        cmd = tracker2.compute_command(pose, aim, previous_v)

        # 运动学约束复核（compute_command 已限幅，这里做断言式记录）
        limit_violation = _command_violates(cmd, eff_limits, previous_v, cur_dt)
        if limit_violation is not None:
            return RehearsalReport(
                feasible=False, completed=False, duration=sim_time, steps=steps,
                min_clearance=min_clearance,
                failure_index=_current_index(path, pose),
                failure_step=steps, reason=limit_violation,
            )

        # 扫掠碰撞复核（与执行同一套细采样 + 半径膨胀）
        swept = arc_poses(pose, cmd, cur_dt, spacing)
        collide = False
        last_free_p = pose
        for p in swept:
            if grid.footprint_blocked(p.x, p.y, r_safe):
                collide = True
                # 碰撞点不计入净空，但用上一个自由点（紧贴安全边界）反映
                # “差一点撞上”的最小净空，使报告净空能落到接近 0 的区间
                min_clearance = min(min_clearance, _clearance(last_free_p))
                break
            min_clearance = min(min_clearance, _clearance(p))
            last_free_p = p

        if collide:
            reason = "沿该路线推进时轨迹触障/越界"
            if blocked_idx >= 0:
                reason = (
                    f"路线第 {blocked_idx} 段被障碍阻挡"
                    "（预演不做自动重规划，请先重规划或调整路线）"
                )
            return RehearsalReport(
                feasible=False, completed=False, duration=sim_time, steps=steps,
                min_clearance=min_clearance,
                failure_index=blocked_idx if blocked_idx >= 0
                else _current_index(path, pose),
                failure_step=steps, reason=reason,
            )

        # 提交到局部模拟状态
        from .controller import integrate_pose
        pose = integrate_pose(pose, cmd, cur_dt)
        previous_v = cmd.v
        sim_time += cur_dt
        steps += 1

        if tracker2.at_goal(pose, goal):
            return RehearsalReport(
                feasible=True, completed=True, duration=sim_time, steps=steps,
                min_clearance=min_clearance, failure_index=None,
                failure_step=None,
                reason=f"可安全走完全程，预计 {sim_time:.2f} 秒 / {steps} 个微步",
            )

    return RehearsalReport(
        feasible=False, completed=False, duration=sim_time, steps=steps,
        min_clearance=min_clearance,
        failure_index=_current_index(path, pose), failure_step=steps,
        reason=f"达到微步预算 {max_microsteps} 仍未走完（路线过长或步长过小）",
    )


def _current_index(path: list[Point], pose: Pose) -> int:
    """返回离当前位姿最近的路线点下标（用于问题定位）。"""
    best_i, best_d = 0, math.inf
    for i, p in enumerate(path):
        d = (p.x - pose.x) ** 2 + (p.y - pose.y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _command_violates(cmd, limits: RobotLimits, previous_v: float, dt: float):
    if cmd.v < limits.min_v - 1e-9 or cmd.v > limits.max_v + 1e-9:
        return f"线速度 {cmd.v:.3f} 超出 [{limits.min_v},{limits.max_v}]"
    if abs(cmd.omega) > limits.max_omega + 1e-9:
        return f"角速度 {cmd.omega:.3f} 超过上限 {limits.max_omega}"
    if abs(cmd.v) > 1e-9 and abs(cmd.omega / cmd.v) > limits.max_curvature + 1e-9:
        return (
            f"曲率 {abs(cmd.omega / cmd.v):.3f} 超过上限 {limits.max_curvature}"
        )
    if abs(cmd.v - previous_v) > limits.max_acc * dt + 1e-9:
        return (
            f"速度变化 {abs(cmd.v - previous_v):.3f} 超过加速度预算 "
            f"{limits.max_acc * dt:.3f}"
        )
    return None
