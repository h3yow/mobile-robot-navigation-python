"""仿真会话：持有地图/参数/位姿/路径，串行化状态迁移。

动态障碍以世界坐标矩形列表表示，每次步进时全量栅格化叠加到静态地图上，
形成“当前有效占用栅格”；规划与路径通畅性检查都基于该栅格，
从而保证任何时刻轨迹都不穿越已上报的动态障碍。
"""

from __future__ import annotations

import math
import threading
from collections import deque

from .controller import (
    TrajectoryTracker,
    arc_poses,
    euler_poses,
    integrate_pose,
    integrate_pose_euler,
)
from .errors import (
    InvalidInputError,
    SessionStateError,
)
from .grid import OccupancyGrid, rasterize_rect
from .models import (
    AdvanceResult,
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    SessionSnapshot,
    SessionStatus,
    StepResult,
    TrajectoryRecord,
    Twist,
)
from .planner import GridPlanner

# ---- 长时推进/轨迹相关常量 ----
# 单条轨迹记录中扫掠位姿的采样数上限（采样再密也只保留这么多点做回查）
MAX_SWIPT_POINTS_PER_RECORD = 256
# advance 单次允许模拟的最大时长（秒）与最大微步数（防误用导致长时间占用）
MAX_ADVANCE_DURATION = 60.0
MAX_MICROSTEPS = 20000
# 轨迹记录默认保留条数；超出后丢弃最旧记录（FIFO 环形缓冲）
DEFAULT_TRAJECTORY_LIMIT = 2000
# 部分推进时“缩短时长 -> 按新时长重算指令 -> 复核”的迭代上限
# （每轮 move_dt 几何级缩小，正常 2~3 轮即收敛；超限则安全地零速停住）
MAX_PARTIAL_REFINEMENTS = 16


class AdvanceParams:
    """一次 advance 的运行参数（仅本次调用生效，不改会话默认配置）。"""

    __slots__ = ("duration", "dt", "max_v")

    def __init__(
        self,
        duration: float,
        dt: float | None = None,
        max_v: float | None = None,
    ) -> None:
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise InvalidInputError("duration 必须为数值", {"value": repr(duration)})
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise InvalidInputError("duration 必须为有限正数（秒）", {"value": duration})
        if duration > MAX_ADVANCE_DURATION:
            raise InvalidInputError(
                f"duration 不能超过 {MAX_ADVANCE_DURATION:g} 秒",
                {"value": duration, "max": MAX_ADVANCE_DURATION},
            )
        if dt is not None:
            if not isinstance(dt, (int, float)) or isinstance(dt, bool):
                raise InvalidInputError("dt 必须为数值", {"value": repr(dt)})
            dt = float(dt)
            if not math.isfinite(dt) or dt <= 0:
                raise InvalidInputError("dt 必须为有限正数（秒）", {"value": dt})
        if max_v is not None:
            if not isinstance(max_v, (int, float)) or isinstance(max_v, bool):
                raise InvalidInputError("max_v 必须为数值", {"value": repr(max_v)})
            max_v = float(max_v)
            if not math.isfinite(max_v) or max_v <= 0:
                raise InvalidInputError("max_v 必须为有限正数", {"value": max_v})
        self.duration = duration
        self.dt = dt
        self.max_v = max_v


class NavigationSession:
    """单个仿真会话的全部可变状态，自带可重入锁保证步进/重规划串行。"""

    def __init__(
        self,
        session_id: str,
        grid: OccupancyGrid,
        limits: RobotLimits,
        planner_params: PlannerParams,
        controller_params: ControllerParams,
        created_at: float,
        trajectory_limit: int = DEFAULT_TRAJECTORY_LIMIT,
    ) -> None:
        if not isinstance(trajectory_limit, int) or trajectory_limit < 1:
            raise InvalidInputError(
                "trajectory_limit 必须为 >=1 的整数", {"value": trajectory_limit}
            )
        self.id = session_id
        self.base_grid = grid
        self.limits = limits
        self.trajectory_limit = trajectory_limit
        # 规划安全余量：用户显式膨胀格数 + 足迹半径折算格数（向上取整）。
        # 规划路径据此与障碍保持至少一个外接半径的间距；
        # 跟踪阶段的切角安全由 max_curvature 速度包络 + 逐步足迹复核共同保证。
        footprint_cells = math.ceil(
            (limits.footprint_radius + 1e-9) / grid.resolution
        )
        effective_inflation = (
            planner_params.obstacle_inflation_cells + footprint_cells
        )
        self.planner_params = PlannerParams(
            obstacle_inflation_cells=effective_inflation,
            allow_diagonal=planner_params.allow_diagonal,
            max_expanded_nodes=planner_params.max_expanded_nodes,
            cost_diagonal=planner_params.cost_diagonal,
        )
        self.user_planner_params = planner_params
        # 前视距离下限：同时不小于 1.5 个栅格与纯追踪几何所需的 2/max_curvature，
        # 保证粗分辨率与大曲率约束下跟踪都稳定且不超曲率包络。
        lookahead = max(
            float(controller_params.lookahead_cells),
            1.5 * grid.resolution,
            2.0 / limits.max_curvature,
        )
        self.controller_params = ControllerParams(
            k_linear=controller_params.k_linear,
            k_angular=controller_params.k_angular,
            goal_tolerance=controller_params.goal_tolerance,
            lookahead_cells=lookahead,
            dt=controller_params.dt,
        )
        self.tracker = TrajectoryTracker(limits, controller_params)

        self.lock = threading.RLock()
        self.status = SessionStatus.CREATED
        self.pose: Pose | None = None
        self.goal: Point | None = None
        self.path: list[Point] = []
        self.path_index = 0
        self.step_count = 0
        self.previous_v = 0.0
        self.dynamic_rects: list[tuple[float, float, float, float]] = []
        self.effective_grid: OccupancyGrid = grid
        # 执行轨迹环形缓冲：只保留最近 trajectory_limit 条微步记录，
        # 超出后丢弃最旧记录；不同会话各持一份，互不影响。
        self.trajectory: deque[TrajectoryRecord] = deque(
            maxlen=trajectory_limit
        )
        self.trajectory_dropped = 0
        self.last_note = "会话已创建，等待规划"
        self.created_at = created_at
        self.updated_at = created_at

    @classmethod
    def create(
        cls,
        grid: OccupancyGrid,
        limits: RobotLimits,
        planner_params: PlannerParams,
        controller_params: ControllerParams,
        now: float,
        trajectory_limit: int = DEFAULT_TRAJECTORY_LIMIT,
    ) -> "NavigationSession":
        return cls(
            session_id="",
            grid=grid,
            limits=limits,
            planner_params=planner_params,
            controller_params=controller_params,
            created_at=now,
            trajectory_limit=trajectory_limit,
        )

    # ------------------------------------------------------------------
    def _touch(self, now: float) -> None:
        self.updated_at = now

    def _rebuild_effective_grid(self) -> None:
        """把动态障碍矩形全量栅格化并叠加到静态地图。"""
        if not self.dynamic_rects:
            self.effective_grid = self.base_grid
            return
        mask = [0] * (self.base_grid.width * self.base_grid.height)
        g = self.base_grid
        for xmin, ymin, xmax, ymax in self.dynamic_rects:
            rect = rasterize_rect(
                g.width, g.height, g.resolution,
                g.origin_x, g.origin_y, xmin, ymin, xmax, ymax,
            )
            mask = [a | b for a, b in zip(mask, rect)]
        self.effective_grid = g.merged(mask)

    def update_dynamic_obstacles(self, rects, now: float) -> None:
        """全量更新动态障碍矩形列表（世界坐标），随后重建有效栅格。"""
        with self.lock:
            parsed: list[tuple[float, float, float, float]] = []
            for r in rects or []:
                if len(r) != 4:
                    from .errors import InvalidInputError

                    raise InvalidInputError(
                        "动态障碍矩形必须为 [xmin,ymin,xmax,ymax]", {"rect": r}
                    )
                xmin, ymin, xmax, ymax = (float(v) for v in r)
                parsed.append((xmin, ymin, xmax, ymax))
            self.dynamic_rects = parsed
            self._rebuild_effective_grid()
            self._touch(now)
            self.last_note = f"动态障碍更新：{len(parsed)} 个矩形"

    # ------------------------------------------------------------------
    def plan(self, start: Pose, goal: Point, now: float) -> list[Point]:
        """在当前有效栅格上规划，成功后进入 READY。"""
        with self.lock:
            self._rebuild_effective_grid()
            planner = GridPlanner(self.effective_grid, self.planner_params)
            path = planner.plan(Point(start.x, start.y), goal)
            self.pose = start
            self.goal = goal
            self.path = path
            self.path_index = 0
            self.step_count = 0
            self.previous_v = 0.0
            self._clear_trajectory()
            self.status = SessionStatus.READY
            self.last_note = f"规划成功：{len(path)} 个路径点"
            self._touch(now)
            return list(path)

    def replan(self, now: float) -> list[Point]:
        """以当前位姿为新起点重新规划到既有目标。"""
        with self.lock:
            if self.goal is None or self.pose is None:
                raise SessionStateError("尚未设置起点/目标，无法重规划")
            self._rebuild_effective_grid()
            planner = GridPlanner(self.effective_grid, self.planner_params)
            path = planner.plan(
                Point(self.pose.x, self.pose.y), self.goal
            )
            self.path = path
            self.path_index = 0
            self.previous_v = 0.0
            self.status = SessionStatus.READY
            self.last_note = f"重规划成功：{len(path)} 个路径点"
            self._touch(now)
            return list(path)

    # ------------------------------------------------------------------
    def step(self, now: float) -> StepResult:
        """推进一步（老调用方式，行为与长时推进能力引入前完全一致）。

        流程（整步在会话锁内完成，杜绝错序与状态撕裂）：
          1. 重建有效栅格（纳入最新动态障碍）；
          2. 若当前位姿已被障碍覆盖（机器人被困在障碍内）-> BLOCKED，零速；
          3. 检查剩余路径通畅性；被挡则尝试重规划；
             重规划失败（不可达/超时）-> BLOCKED，零速，状态可继续（环境变化后再 step 即恢复）；
          4. 跟踪器选前视点、算受约束指令、按老版一阶模型积分；
          5. 积分后位姿必须无碰撞，否则拒绝该运动（零速，BLOCKED）；
          6. 到达目标 -> ARRIVED。

        兼容性约定：本路径刻意保持升级前语义——一阶直线积分、仅对积分后
        位姿做足迹复核、整步走不下则零速停住（不做部分推进），因此
        不传任何新增推进参数时，逐步执行的位姿序列与升级前逐步一致。
        新增的部分推进/扫掠复核/尽量前进语义只属于 advance 微步路径。
        """
        with self.lock:
            if self.pose is None or self.goal is None or not self.path:
                raise SessionStateError("会话尚未成功规划，无法步进")

            self._rebuild_effective_grid()
            note_parts: list[str] = []

            # 1) 当前位姿安全性
            if self.effective_grid.footprint_blocked(
                self.pose.x, self.pose.y, self.limits.footprint_radius
            ):
                self.status = SessionStatus.BLOCKED
                self.last_note = "当前位姿已被障碍占据，无法运动"
                self._touch(now)
                return StepResult(
                    pose=self.pose, cmd=_zero_twist(), replanned=False,
                    arrived=False, blocked=True, note=self.last_note,
                )

            # 2) 路径通畅性检查 + 自动重规划
            replanned = False
            planner = GridPlanner(self.effective_grid, self.planner_params)
            blocked_idx = planner.is_path_clear(
                self.path, self.limits.footprint_radius
            )
            if blocked_idx >= 0:
                note_parts.append(f"路径在第 {blocked_idx} 段被动态障碍阻挡")
                try:
                    new_path = planner.plan(
                        Point(self.pose.x, self.pose.y), self.goal
                    )
                    self.path = new_path
                    self.path_index = 0
                    replanned = True
                    note_parts.append(f"自动重规划成功（{len(new_path)} 点）")
                except Exception as exc:  # 不可达 / 超时 / 起终点非法
                    self.status = SessionStatus.BLOCKED
                    self.last_note = (
                        f"{'; '.join(note_parts)}；重规划失败：{exc.message if hasattr(exc, 'message') else exc}"
                    )
                    self._touch(now)
                    return StepResult(
                        pose=self.pose, cmd=_zero_twist(), replanned=False,
                        arrived=False, blocked=True, note=self.last_note,
                    )

            # 3) 到达判定（在执行运动之前先看是否已经在终点容差内）
            if self.tracker.at_goal(self.pose, self.goal):
                self.status = SessionStatus.ARRIVED
                self.last_note = "已到达目标"
                self._touch(now)
                return StepResult(
                    pose=self.pose, cmd=_zero_twist(), replanned=replanned,
                    arrived=True, blocked=False, note=self.last_note,
                )

            # 4) 跟踪控制（老版一阶直线积分，与升级前一致）
            aim, new_idx = self.tracker.find_aim_point(
                self.pose, self.path, self.path_index
            )
            self.path_index = new_idx
            cmd = self.tracker.compute_command(self.pose, aim, self.previous_v)
            dt = self.controller_params.dt
            next_pose = integrate_pose_euler(self.pose, cmd, dt)

            # 5) 运动后碰撞复核：任何时刻不得穿越障碍
            if self.effective_grid.footprint_blocked(
                next_pose.x, next_pose.y, self.limits.footprint_radius
            ):
                self.status = SessionStatus.BLOCKED
                self.last_note = "下一步预测位姿触障，拒绝执行该速度指令"
                self._touch(now)
                return StepResult(
                    pose=self.pose, cmd=_zero_twist(), replanned=replanned,
                    arrived=False, blocked=True, note=self.last_note,
                )

            pose_before = self.pose
            self.pose = next_pose
            self.previous_v = cmd.v
            self.step_count += 1
            arrived = self.tracker.at_goal(next_pose, self.goal)
            self.status = (
                SessionStatus.ARRIVED if arrived else SessionStatus.RUNNING
            )
            # 轨迹记录（老版运动模型采样，swept[-1] 与上报位姿严格一致）
            self._record_motion(
                pose_before, next_pose, cmd, dt,
                self._collision_spacing(), sampler=euler_poses,
            )
            note_parts.append(
                f"步进 {self.step_count}：v={cmd.v:.3f} omega={cmd.omega:.3f} "
                f"瞄准路径点[{self.path_index}]"
            )
            if arrived:
                note_parts.append("已到达目标")
            self.last_note = "；".join(note_parts)
            self._touch(now)
            return StepResult(
                pose=next_pose, cmd=cmd, replanned=replanned,
                arrived=arrived, blocked=False, note=self.last_note,
                elapsed=dt,
            )

    def _collision_spacing(self) -> float:
        """扫掠碰撞复核的采样间距：1/4 栅格（上限 0.2m，避免细分辨率下过密）。"""
        return min(self.base_grid.resolution * 0.25, 0.2)

    def _arc_collides(
        self, pose: Pose, cmd: Twist, duration: float, spacing: float
    ) -> bool:
        """检查一条恒定指令圆弧在 duration 内是否触障/越界。

        以间距 <= spacing 采样扫掠位姿，并对每个采样点用
        ``footprint_radius + spacing/2`` 的膨胀半径复核，
        因此相邻采样之间的“隧穿”缝隙也被覆盖，薄墙/窄通道不会漏检。
        """
        r_safe = self.limits.footprint_radius + spacing * 0.5
        poses = arc_poses(pose, cmd, duration, spacing)
        for p in poses:
            if self.effective_grid.footprint_blocked(p.x, p.y, r_safe):
                return True
        return False

    def _microstep(
        self,
        dt: float,
        max_v_override: float | None,
        now: float,
    ) -> StepResult:
        """执行一个微步（调用方须已重建有效栅格并持有会话锁）。

        与单步相同的安全流程，但允许指定 dt 与临时速度上限：
          1. 当前位姿安全性；
          2. 路径通畅性 + 自动重规划；
          3. 到点判定；
          4. 跟踪器算受约束指令（含 dt/速度上限）；
          5. 整段圆弧扫掠碰撞复核；若全步走不下则二分取最大可行部分，
             并按本步实际执行时长重算指令（加速度按该时长限幅）、
             复核后回延到该指令的最大安全时长；
             任何部分都走不下则零速 BLOCKED（原地停住、状态保留）；
          6. 到达目标 -> ARRIVED；成功运动写入轨迹缓冲。
        """
        note_parts: list[str] = []

        # 本次微步生效的参数/约束（不写回会话默认配置）
        limits = self.limits
        if max_v_override is not None and max_v_override != self.limits.max_v:
            limits = RobotLimits(
                max_v=max_v_override,
                min_v=self.limits.min_v,
                max_omega=self.limits.max_omega,
                max_curvature=self.limits.max_curvature,
                max_acc=self.limits.max_acc,
                footprint_radius=self.limits.footprint_radius,
            )

        def _tracker_for(step_dt: float) -> TrajectoryTracker:
            """构造一个以 step_dt 为步长的跟踪器（加速度限幅按该时长计算）。"""
            return TrajectoryTracker(
                limits,
                ControllerParams(
                    k_linear=self.controller_params.k_linear,
                    k_angular=self.controller_params.k_angular,
                    goal_tolerance=self.controller_params.goal_tolerance,
                    lookahead_cells=self.controller_params.lookahead_cells,
                    dt=step_dt,
                ),
            )

        tracker = _tracker_for(dt)

        # 1) 当前位姿安全性
        if self.effective_grid.footprint_blocked(
            self.pose.x, self.pose.y, self.limits.footprint_radius
        ):
            self.status = SessionStatus.BLOCKED
            self.last_note = "当前位姿已被障碍占据，无法运动"
            self._touch(now)
            return StepResult(
                pose=self.pose, cmd=_zero_twist(), replanned=False,
                arrived=False, blocked=True, note=self.last_note,
            )

        # 2) 路径通畅性检查 + 自动重规划
        replanned = False
        planner = GridPlanner(self.effective_grid, self.planner_params)
        blocked_idx = planner.is_path_clear(
            self.path, self.limits.footprint_radius
        )
        if blocked_idx >= 0:
            note_parts.append(f"路径在第 {blocked_idx} 段被动态障碍阻挡")
            try:
                new_path = planner.plan(
                    Point(self.pose.x, self.pose.y), self.goal
                )
                self.path = new_path
                self.path_index = 0
                replanned = True
                note_parts.append(f"自动重规划成功（{len(new_path)} 点）")
            except Exception as exc:  # 不可达 / 超时 / 起终点非法
                # 前路被封死且绕不过去：不再一步都不走就原地停住，
                # 而是带着当前（被挡的）路径继续走下面的跟踪流程——
                # 扫掠复核 + 二分会把机器人停在障碍前最后一个安全位置，
                # 即“尽量推进到最靠近的安全位置”；状态保留，清障/换环境/
                # 重规划后可直接继续。
                note_parts.append(
                    f"重规划失败（{exc.message if hasattr(exc, 'message') else exc}），"
                    "无法绕行；将沿当前路径尽量推进到最靠近阻挡点的安全位置"
                )

        # 3) 到达判定
        if tracker.at_goal(self.pose, self.goal):
            self.status = SessionStatus.ARRIVED
            self.last_note = "已到达目标"
            self._touch(now)
            return StepResult(
                pose=self.pose, cmd=_zero_twist(), replanned=replanned,
                arrived=True, blocked=False, note=self.last_note,
            )

        # 4) 跟踪控制
        aim, new_idx = tracker.find_aim_point(
            self.pose, self.path, self.path_index
        )
        self.path_index = new_idx

        # 5) 整段扫掠碰撞复核（不只是终点）；走不满则“能走多少走多少”，
        #    且部分推进时按本步真正执行的时长重算速度指令，
        #    保证上报指令满足 |v - v_prev| <= max_acc * 实际执行时长。
        spacing = self._collision_spacing()
        move_dt = dt
        cmd = tracker.compute_command(self.pose, aim, self.previous_v)
        partial = False
        accepted = False
        for _ in range(MAX_PARTIAL_REFINEMENTS):
            if not self._arc_collides(self.pose, cmd, move_dt, spacing):
                accepted = True
                break
            # 当前指令走不满 move_dt：二分求最大可行时长
            frac = self._max_safe_fraction(self.pose, cmd, move_dt, spacing)
            if frac <= 0.0:
                break  # 任何微小运动都触障 -> 零速停住（下面统一处理）
            move_dt *= frac
            partial = True
            # 关键：时长缩短后必须按新的实际时长重算指令，
            # 否则按原 dt 限幅的速度变化折算到 move_dt 上会超出加速度上限；
            # 重算后指令的圆弧需重新复核（下一轮循环），直到
            # “指令按 move_dt 限幅”且“该指令在 move_dt 内全程无碰撞”同时成立。
            cmd = _tracker_for(move_dt).compute_command(
                self.pose, aim, self.previous_v
            )

        if not accepted:
            self.status = SessionStatus.BLOCKED
            self.last_note = (
                "预测运动轨迹触障/越界（含完整扫掠段复核），拒绝该速度指令，原地停住"
            )
            if note_parts:
                self.last_note = "；".join(note_parts + [self.last_note])
            self._touch(now)
            return StepResult(
                pose=self.pose, cmd=_zero_twist(), replanned=replanned,
                arrived=False, blocked=True, note=self.last_note,
            )

        # 5b) 时长回延：若上面的重算把指令放得更保守（更慢），同一指令往往
        #     能安全执行更久——在 [move_dt, dt] 上二分该指令的最大安全时长。
        #     指令不变而执行时长变长，加速度预算更宽松，约束仍然满足；
        #     这样部分推进能停到“最靠近”的安全位置，而不是提前很远停下。
        if partial and move_dt < dt:
            if not self._arc_collides(self.pose, cmd, dt, spacing):
                move_dt = dt
                partial = False
            else:
                lo, hi = move_dt, dt
                for _ in range(24):
                    mid = (lo + hi) * 0.5
                    if self._arc_collides(self.pose, cmd, mid, spacing):
                        hi = mid
                    else:
                        lo = mid
                move_dt = lo

        pose_before = self.pose
        next_pose = integrate_pose(self.pose, cmd, move_dt)

        self.pose = next_pose
        self.previous_v = cmd.v
        self.step_count += 1
        arrived = tracker.at_goal(next_pose, self.goal)
        if arrived:
            self.status = SessionStatus.ARRIVED
        elif partial:
            self.status = SessionStatus.BLOCKED
        else:
            self.status = SessionStatus.RUNNING

        # 轨迹记录（只有真实发生了运动才产生一条，与 step_count 对齐）
        self._record_motion(pose_before, next_pose, cmd, move_dt, spacing)

        note_parts.append(
            f"步进 {self.step_count}：v={cmd.v:.3f} omega={cmd.omega:.3f} "
            f"dt={move_dt:.3f}s 瞄准路径点[{self.path_index}]"
        )
        if partial:
            note_parts.append(
                f"整段 {dt:.3f}s 中仅前 {move_dt / dt * 100:.1f}%（{move_dt:.3f}s）"
                "可安全通过，随后轨迹触障/越界，已在该位置停住；"
                "速度指令已按实际执行时长重新限幅（满足加速度上限）"
            )
        if arrived:
            note_parts.append("已到达目标")
        self.last_note = "；".join(note_parts)
        self._touch(now)
        return StepResult(
            pose=next_pose, cmd=cmd, replanned=replanned,
            arrived=arrived, blocked=partial, note=self.last_note,
            elapsed=move_dt,
        )

    def _max_safe_fraction(
        self, pose: Pose, cmd: Twist, dt: float, spacing: float
    ) -> float:
        """二分求 [0,1] 内使整段圆弧无碰撞的最大时长比例。

        若任一无穷小运动都触障（f->0 仍碰撞），返回 0.0。
        """
        # 先看“趋近于 0”的一小步是否安全
        tiny = 1e-3
        if self._arc_collides(pose, cmd, dt * tiny, spacing):
            return 0.0
        lo, hi = tiny, 1.0
        for _ in range(24):
            mid = (lo + hi) * 0.5
            if self._arc_collides(pose, cmd, dt * mid, spacing):
                hi = mid
            else:
                lo = mid
        return lo

    def _record_motion(
        self,
        start_pose: Pose,
        end_pose: Pose,
        cmd: Twist,
        move_dt: float,
        spacing: float,
        *,
        sampler=arc_poses,
    ) -> None:
        """把一次已执行的微步写入轨迹环形缓冲。

        sampler 为扫掠位姿采样函数（与执行该步所用的运动模型一致：
        advance 微步用精确圆弧 arc_poses，legacy step 用 euler_poses）。
        """
        travel = abs(cmd.v) * move_dt
        if travel <= 0:
            swept = [start_pose, end_pose]
        else:
            # 存储采样数封顶：碰撞复核用细 spacing，记录点可更粗，
            # 但始终包含起点、终点与中间扫过位置
            store_spacing = max(
                spacing, travel / (MAX_SWIPT_POINTS_PER_RECORD - 1)
            )
            swept = sampler(
                start_pose, cmd, move_dt, store_spacing
            )
        record = TrajectoryRecord(
            step=self.step_count,
            pose=end_pose,
            cmd=cmd,
            swept=swept,
            dt=move_dt,
        )
        if len(self.trajectory) >= self.trajectory_limit:
            self.trajectory_dropped += 1
        self.trajectory.append(record)

    def _clear_trajectory(self) -> None:
        self.trajectory.clear()
        self.trajectory_dropped = 0

    def advance(
        self,
        duration: float,
        now: float,
        *,
        dt: float | None = None,
        max_v: float | None = None,
    ) -> AdvanceResult:
        """在一次调用内模拟最多 duration 秒的连续运动。

        - dt：本次推进使用的微步间隔（秒），缺省用会话默认 controller.dt；
        - max_v：本次推进临时速度上限，缺省用会话 limits.max_v；
          两者只影响本次调用，不改动会话配置。
        - 每个微步都做完整的扫掠碰撞/越界/约束复核：
          能走多少走多少（部分微步也会停在安全位置），走不动就原地停住。
        - 被挡住/到达后会话状态完整保留，可继续 advance / step / replan。
        """
        params = AdvanceParams(duration, dt=dt, max_v=max_v)
        with self.lock:
            if self.pose is None or self.goal is None or not self.path:
                raise SessionStateError("会话尚未成功规划，无法推进")

            step_dt = params.dt or self.controller_params.dt
            remaining = params.duration
            results: list[StepResult] = []
            executed = 0.0
            n_micro = 0
            reason = ""
            start_pose = self.pose
            stop_blocked = False

            while remaining > 1e-12 and n_micro < MAX_MICROSTEPS:
                self._rebuild_effective_grid()
                cur_dt = min(step_dt, remaining)
                r = self._microstep(cur_dt, params.max_v, now)
                results.append(r)
                n_micro += 1
                if r.arrived:
                    executed += r.elapsed
                    reason = "已到达目标"
                    remaining = 0.0
                    break
                if r.blocked:
                    executed += r.elapsed
                    stop_blocked = True
                    if r.elapsed <= 0.0 and _same_pose(r.pose, start_pose):
                        reason = (
                            "起始方向即被阻挡（预测轨迹触障/越界或路径不可达），"
                            "原地停住；清除障碍、更换环境或重新规划后可继续"
                        )
                    elif r.elapsed <= 0.0:
                        reason = (
                            "前方路线被障碍/边界阻挡，已尽量推进到最靠近的"
                            "安全位置后停住（继续运动的任何微小位移都会触障/越界）；"
                            "清除障碍、更换环境或重新规划后可继续"
                        )
                    else:
                        reason = (
                            "前方路线被障碍/边界阻挡，已尽量推进到最靠近的"
                            "安全位置后停住（最后一个微步仅前半段可安全通行）；"
                            "清除障碍、更换环境或重新规划后可继续"
                        )
                    remaining = 0.0
                    break
                executed += cur_dt
                remaining -= cur_dt

            if n_micro >= MAX_MICROSTEPS and remaining > 1e-12:
                reason = (
                    f"达到单次推进最大微步数 {MAX_MICROSTEPS}，"
                    "已安全模拟部分时长（可继续调用推进剩余时长）"
                )
            if not reason:
                reason = f"已完整安全模拟 {params.duration:g} 秒"
            arrived_now = bool(results and results[-1].arrived)
            return AdvanceResult(
                status=self.status,
                pose=self.pose,
                requested_duration=params.duration,
                simulated_duration=executed,
                steps_executed=n_micro,
                # 到达终点即视为完成（即使时长未满）；或完整模拟完时长且未被挡
                completed=arrived_now
                or ((not stop_blocked) and executed + 1e-9 >= params.duration),
                arrived=arrived_now,
                blocked=stop_blocked,
                reason=reason,
                results=results,
            )

    def get_trajectory(self) -> dict:
        """返回当前保留的执行轨迹（只读视图，调用方修改不影响会话）。

        返回字段：
          - records：每条微步一条，含 step / pose / cmd / dt / swept
            （swept 为该微步扫过的稠密位姿，首尾相接可还原连续轨迹，
             且 record.step 与历史 StepResult 的 step_count 一一对应）；
          - limit：保留上限；dropped：因超限被丢弃的最旧记录条数；
          - retained / total：当前保留/累计记录条数。
        """
        with self.lock:
            records = [
                {
                    "step": rec.step,
                    "pose": {
                        "x": rec.pose.x, "y": rec.pose.y, "theta": rec.pose.theta
                    },
                    "cmd": {"v": rec.cmd.v, "omega": rec.cmd.omega},
                    "dt": rec.dt,
                    "swept": [
                        {"x": p.x, "y": p.y, "theta": p.theta}
                        for p in rec.swept
                    ],
                }
                for rec in self.trajectory
            ]
            return {
                "records": records,
                "limit": self.trajectory_limit,
                "dropped": self.trajectory_dropped,
                "retained": len(records),
                "total": self.trajectory_dropped + len(records),
            }

    def reset(self, now: float) -> None:
        """清空目标与轨迹，回到 CREATED（地图与参数保留）。"""
        with self.lock:
            self.goal = None
            self.pose = None
            self.path = []
            self.path_index = 0
            self.step_count = 0
            self.previous_v = 0.0
            self.status = SessionStatus.CREATED
            self.dynamic_rects = []
            self.effective_grid = self.base_grid
            self._clear_trajectory()
            self.last_note = "会话已重置"
            self._touch(now)

    def snapshot(self, now: float) -> SessionSnapshot:
        with self.lock:
            self._touch(now)
            return SessionSnapshot(
                session_id=self.id,
                status=self.status,
                pose=self.pose if self.pose is not None else Pose(0.0, 0.0, 0.0),
                goal=self.goal,
                path=list(self.path),
                path_index=self.path_index,
                step_count=self.step_count,
                message=f"会话 {self.id} 当前状态 {self.status.value}",
                last_note=self.last_note,
                created_at=self.created_at,
                updated_at=self.updated_at,
            )


def _zero_twist() -> Twist:
    return Twist(v=0.0, omega=0.0)


def _same_pose(a: Pose, b: Pose) -> bool:
    return abs(a.x - b.x) < 1e-12 and abs(a.y - b.y) < 1e-12
