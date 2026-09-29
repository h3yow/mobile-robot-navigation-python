"""只读预演：给定路线与运行参数，在不触碰任何会话状态的前提下模拟跟踪全过程。

纯函数、确定性：相同（地图/约束/路线/参数）输入必得相同输出，不读取时钟、
不使用随机数、不修改调用方对象；调用方（service/session）在会话锁外执行，
因此预演不会阻塞同一会话或其它会话的正常推进与查询。
"""

from __future__ import annotations

import dataclasses
import math

from .controller import TrajectoryTracker
from .grid import OccupancyGrid
from .models import (
    ControllerParams,
    Point,
    Pose,
    PreviewResult,
    RobotLimits,
)

# 间隙搜索的默认/最大外扩栅格数：搜索半径内无障碍即按该距离封顶并标记 capped。
DEFAULT_CLEARANCE_CELLS = 8.0


def preview_path(
    grid: OccupancyGrid,
    limits: RobotLimits,
    start: Pose,
    path: list[Point],
    *,
    dt: float,
    max_v: float,
    max_ticks: int,
    clearance_cells: float = DEFAULT_CLEARANCE_CELLS,
    controller_params: ControllerParams | None = None,
) -> PreviewResult:
    """在给定栅格上预演沿 path 的纯跟踪过程。

    返回 PreviewResult：feasible 表示能否在预算 tick 内安全到达；
    不可行时给出 fail_reason 与大致 fail_index（路线顶点下标）；
    min_clearance 为机器人外接圆边缘到障碍/边界的全程最近距离（米）。

    fail_reason 取值：
      start_blocked 起点足迹本身触障/越界；
      path_invalid   路线某顶点足迹触障/越界（路线几何不可执行）；
      collision      实际扫掠轨迹触障（按这套 dt/max_v 跟踪时穿障）；
      blocked        连续多 tick 零平移且未到达（被几何/曲率困在原地）；
      step_budget    tick 预算用尽仍未到达（参数太慢或路线太长）。
    """
    radius = limits.footprint_radius
    cap_centre = clearance_cells * grid.resolution

    # 1) 参数自检（调用方通常已校验，这里防御性兜底）
    if dt <= 0.0 or max_v <= 0.0 or max_ticks <= 0:
        raise ValueError("preview: dt/max_v/max_ticks 必须为正数")
    if not path:
        return PreviewResult(
            feasible=False, ticks_simulated=0, min_clearance=0.0,
            min_clearance_point=None, clearance_capped=False,
            fail_index=None, fail_reason="path_invalid", fail_pose=None,
            note="预演路线为空",
        )

    # 2) 起点安全性
    if grid.footprint_blocked(start.x, start.y, radius):
        c0 = grid.point_clearance(start.x, start.y, clearance_cells)
        return PreviewResult(
            feasible=False, ticks_simulated=0,
            min_clearance=max(0.0, c0 - radius),
            min_clearance_point=Point(start.x, start.y),
            clearance_capped=c0 >= cap_centre - 1e-12,
            fail_index=None, fail_reason="start_blocked",
            fail_pose=start,
            note="起点足迹触障或越出地图",
        )

    # 3) 路线顶点安全性（路线本身几何是否可执行）
    for i, pt in enumerate(path):
        if grid.footprint_blocked(pt.x, pt.y, radius):
            c = grid.point_clearance(pt.x, pt.y, clearance_cells)
            return PreviewResult(
                feasible=False, ticks_simulated=0,
                min_clearance=max(0.0, c - radius),
                min_clearance_point=Point(pt.x, pt.y),
                clearance_capped=c >= cap_centre - 1e-12,
                fail_index=i, fail_reason="path_invalid",
                fail_pose=Pose(pt.x, pt.y, 0.0),
                note=f"路线第 {i} 个顶点足迹触障或越界",
            )

    # 4) 用与执行完全一致的跟踪器/约束做确定性仿真
    sim_limits = dataclasses.replace(limits, max_v=max_v)
    if controller_params is not None:
        params = dataclasses.replace(controller_params, dt=dt)
    else:
        params = ControllerParams(dt=dt)
    tracker = TrajectoryTracker(sim_limits, params)

    # 初始间隙（起点）
    min_centre = grid.point_clearance(start.x, start.y, clearance_cells)
    min_pt = Point(start.x, start.y)

    # 路线折线间隙（细于分辨率采样，覆盖长直段中间）
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg_len = math.hypot(b.x - a.x, b.y - a.y)
        steps = max(1, math.ceil(seg_len / max(grid.resolution * 0.5, 1e-9)))
        for k in range(steps + 1):
            t = k / steps
            x = a.x + (b.x - a.x) * t
            y = a.y + (b.y - a.y) * t
            c = grid.point_clearance(x, y, clearance_cells)
            if c < min_centre:
                min_centre, min_pt = c, Point(x, y)

    pose = start
    idx = 0
    nearest = 0
    previous_v = 0.0
    stuck_ticks = 0
    ticks = 0

    def _advance_nearest(p: Pose, cur: int) -> int:
        """沿路径推进到离当前位姿最近的顶点（用于报告“大概在第几个位置”）。"""
        j = cur
        while j < len(path) - 1:
            da = math.hypot(path[j].x - p.x, path[j].y - p.y)
            db = math.hypot(path[j + 1].x - p.x, path[j + 1].y - p.y)
            if db <= da:
                j += 1
                continue
            break
        return j

    for tick in range(1, max_ticks + 1):
        aim, idx = tracker.find_aim_point(pose, path, idx)
        cmd = tracker.compute_command(pose, aim, previous_v)
        nxt = tracker.advance(pose, cmd)
        ticks = tick
        nearest = _advance_nearest(nxt, nearest)

        # 微步胶囊扫掠复核：哪怕该 tick 位移很大也不得穿障/越界
        if grid.segment_blocked(pose.x, pose.y, nxt.x, nxt.y, radius):
            c = grid.segment_clearance(
                pose.x, pose.y, nxt.x, nxt.y, clearance_cells
            )
            if c < min_centre:
                min_centre, min_pt = c, Point(
                    (pose.x + nxt.x) / 2, (pose.y + nxt.y) / 2
                )
            return PreviewResult(
                feasible=False, ticks_simulated=tick,
                min_clearance=max(0.0, min_centre - radius),
                min_clearance_point=min_pt,
                clearance_capped=min_centre >= cap_centre - 1e-12,
                fail_index=nearest, fail_reason="collision",
                fail_pose=nxt,
                note=f"第 {tick} 个微步扫掠轨迹触障（瞄准路线顶点 {idx}）",
            )

        # 间隙跟踪（沿微步采样）
        step_len = math.hypot(nxt.x - pose.x, nxt.y - pose.y)
        n_samples = max(1, math.ceil(step_len / max(grid.resolution * 0.5, 1e-9)))
        for k in range(n_samples + 1):
            t = k / n_samples
            x = pose.x + (nxt.x - pose.x) * t
            y = pose.y + (nxt.y - pose.y) * t
            c = grid.point_clearance(x, y, clearance_cells)
            if c < min_centre:
                min_centre, min_pt = c, Point(x, y)

        pose, previous_v = nxt, cmd.v

        if tracker.at_goal(pose, path[-1]):
            return PreviewResult(
                feasible=True, ticks_simulated=tick,
                min_clearance=max(0.0, min_centre - radius),
                min_clearance_point=min_pt,
                clearance_capped=min_centre >= cap_centre - 1e-12,
                fail_index=None, fail_reason=None, fail_pose=None,
                note=f"预演通过：{tick} 个微步安全到达，"
                     f"全程最近间隙 {max(0.0, min_centre - radius):.3f}m",
            )

        # 原地卡住检测（几乎零平移且离目标还远）
        if step_len < 1e-7:
            stuck_ticks += 1
        else:
            stuck_ticks = 0
        if stuck_ticks >= max(20, math.ceil(2.0 / dt)):
            return PreviewResult(
                feasible=False, ticks_simulated=tick,
                min_clearance=max(0.0, min_centre - radius),
                min_clearance_point=min_pt,
                clearance_capped=min_centre >= cap_centre - 1e-12,
                fail_index=nearest, fail_reason="blocked",
                fail_pose=pose,
                note=f"连续 {stuck_ticks} 个微步零平移，跟踪被几何/曲率约束困在原地",
            )

    return PreviewResult(
        feasible=False, ticks_simulated=ticks,
        min_clearance=max(0.0, min_centre - radius),
        min_clearance_point=min_pt,
        clearance_capped=min_centre >= cap_centre - 1e-12,
        fail_index=nearest, fail_reason="step_budget",
        fail_pose=pose,
        note=f"微步预算 {max_ticks} 用尽仍未到达（dt={dt}, max_v={max_v}）",
    )
