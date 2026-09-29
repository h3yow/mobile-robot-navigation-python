"""仿真会话：持有地图/参数/位姿/路径/执行轨迹，串行化状态迁移。

动态障碍以世界坐标矩形列表表示，每次步进时全量栅格化叠加到静态地图上，
形成“当前有效占用栅格”；规划与路径通畅性检查都基于该栅格，
从而保证任何时刻轨迹都不穿越已上报的动态障碍。

长时仿真能力：
  * step：单微步（兼容老接口）；
  * advance(duration, dt, max_v)：一次按时长推进，内部切微步逐帧限幅，
    相邻上报位姿之间用“线段扫掠胶囊”做精确碰撞复核（大跨度不穿障），
    局部制动视距内被挡则自动重规划，失败原地零速停住且可恢复；
  * get_trajectory：查询 FIFO 环形缓冲（默认 2000 条，超限丢最旧）；
  * preview：取快照后在锁外调用 nav.preview 做只读、确定性预演。
"""

from __future__ import annotations

import dataclasses
import math
import threading
from collections import deque

from .controller import TrajectoryTracker
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
    PreviewResult,
    RobotLimits,
    SessionSnapshot,
    SessionStatus,
    StepResult,
    TrajectorySample,
)
from .planner import GridPlanner
from .preview import preview_path

# 每次 advance 允许的内部微步总数上限（配合 service 层的时长上限双重保护）
MAX_ADVANCE_TICKS = 20_000


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
        trajectory_capacity: int = 2000,
    ) -> None:
        self.id = session_id
        self.base_grid = grid
        self.limits = limits
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
        # 跟踪器必须使用做过前视下限保护的参数（self.controller_params），
        # 否则大速度/大步长下纯追踪会因前视不足而切角穿障。
        self.tracker = TrajectoryTracker(limits, self.controller_params)

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
        self.last_note = "会话已创建，等待规划"
        self.created_at = created_at
        self.updated_at = created_at

        # 执行轨迹：会话内 FIFO 环形缓冲，超容量丢弃最旧样本（不丢会话、不报错）。
        if trajectory_capacity < 1:
            raise InvalidInputError(
                "trajectory_capacity 必须为正整数",
                {"value": trajectory_capacity},
            )
        self.trajectory_capacity = int(trajectory_capacity)
        self.trajectory: deque[TrajectorySample] = deque(
            maxlen=self.trajectory_capacity
        )
        self.trajectory_total = 0
        self.trajectory_dropped = 0
        self.sim_time = 0.0

    @classmethod
    def create(
        cls,
        grid: OccupancyGrid,
        limits: RobotLimits,
        planner_params: PlannerParams,
        controller_params: ControllerParams,
        now: float,
        trajectory_capacity: int = 2000,
    ) -> "NavigationSession":
        return cls(
            session_id="",
            grid=grid,
            limits=limits,
            planner_params=planner_params,
            controller_params=controller_params,
            created_at=now,
            trajectory_capacity=trajectory_capacity,
        )

    # ------------------------------------------------------------------
    def _touch(self, now: float) -> None:
        self.updated_at = now

    def _reset_trajectory(self) -> None:
        """规划/重置时清空轨迹缓冲与仿真时钟（保留容量配置）。"""
        self.trajectory.clear()
        self.trajectory_total = 0
        self.trajectory_dropped = 0
        self.sim_time = 0.0

    def _safe_aim_within_horizon(self, horizon: float) -> Point:
        """瞄准点不安全时，沿路径在当前位姿前方 horizon 米内找最远的安全点。"""
        start_i = max(0, min(self.path_index, len(self.path) - 1))
        radius = self.limits.footprint_radius
        remaining = horizon
        px, py = self.pose.x, self.pose.y
        safe = Point(px, py)
        step = max(self.effective_grid.resolution * 0.25, 1e-9)
        i = start_i
        while i < len(self.path):
            nx, ny = self.path[i].x, self.path[i].y
            seg = math.hypot(nx - px, ny - py)
            n = max(1, math.ceil(seg / step))
            blocked_here = False
            for k in range(n + 1):
                t = k / n
                x = px + (nx - px) * t
                y = py + (ny - py) * t
                if math.hypot(x - self.pose.x, y - self.pose.y) > horizon:
                    return safe
                if self.effective_grid.footprint_blocked(x, y, radius):
                    blocked_here = True
                    break
                safe = Point(x, y)
            if blocked_here:
                return safe
            remaining -= seg
            px, py = nx, ny
            i += 1
            if remaining <= 0:
                return safe
        return safe

    def _local_blocked_index(
        self, grid: OccupancyGrid, horizon: float, radius: float
    ) -> int:
        """检查当前位姿前方沿路径 horizon 米弧长内的折线是否被挡。

        从当前位姿出发，沿“位姿 -> path[path_index..]”折线以细于分辨率的
        步长采样（覆盖长直段中间），累计弧长超过 horizon 即停止；
        范围内任一采样点足迹触障即返回其将要到达的顶点下标，否则 -1。
        不做“跳到更近顶点”的推断，避免越过路径中间的阻挡。
        """
        if not self.path or self.pose is None:
            return -1
        # 沿“位姿前方剩余路径”采样。起始段：从 path_index 起找到第一段，
        # 使当前位姿位于该段起点之前（投影参数 t>=0），避免先回头查身后点。
        start_i = max(0, min(self.path_index, len(self.path) - 1))
        start_point = Point(self.pose.x, self.pose.y)
        while start_i < len(self.path) - 1:
            a, b = self.path[start_i], self.path[start_i + 1]
            vx, vy = b.x - a.x, b.y - a.y
            l2 = vx * vx + vy * vy
            t_proj = (
                ((self.pose.x - a.x) * vx + (self.pose.y - a.y) * vy) / l2
                if l2 > 1e-18 else 0.0
            )
            if t_proj >= 0.0:
                # 位姿已在该段覆盖区间之后（t>1）则进入下一段；
                # 否则从当前位姿开始检查本段
                if t_proj <= 1.0:
                    break
                start_i += 1
                continue
            start_i += 1

        accum = 0.0
        prev_x, prev_y = start_point.x, start_point.y
        step = max(grid.resolution * 0.5, 1e-9)
        for i in range(start_i + 1, len(self.path)):
            pt = self.path[i]
            seg = math.hypot(pt.x - prev_x, pt.y - prev_y)
            n = max(1, math.ceil(seg / step))
            for k in range(n + 1):
                t = k / n
                x = prev_x + (pt.x - prev_x) * t
                y = prev_y + (pt.y - prev_y) * t
                arc = accum + math.hypot(x - prev_x, y - prev_y)
                if arc > horizon:
                    return -1  # 视距外：本 tick 不关心
                if grid.footprint_blocked(x, y, radius):
                    return i
            accum += seg
            prev_x, prev_y = pt.x, pt.y
        return -1

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
            self._reset_trajectory()
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
            self._reset_trajectory()
            self.status = SessionStatus.READY
            self.last_note = f"重规划成功：{len(path)} 个路径点"
            self._touch(now)
            return list(path)

    # ------------------------------------------------------------------
    def step(self, now: float, dt: float | None = None,
             max_v: float | None = None) -> StepResult:
        """推进一步（兼容老调用：默认一个 controller dt、会话速度上限）。"""
        with self.lock:
            already_arrived = self.status == SessionStatus.ARRIVED
            result = self._tick_once(dt=dt, max_v=max_v)
            # 已到达之后的重复步进是幂等空步（零速、位姿冻结），不产生新样本
            if not (already_arrived and result.arrived):
                self._record_sample(
                    result, dt if dt is not None else self.controller_params.dt
                )
            self._touch(now)
            return result

    def advance(
        self,
        now: float,
        duration: float,
        *,
        dt: float | None = None,
        max_v: float | None = None,
    ) -> AdvanceResult:
        """在一次调用内安全推进最多 duration 秒。

        内部按 dt（默认 controller.dt）切成多个微步，逐微步执行与
        “位姿间线段扫掠胶囊”碰撞复核；任一步走不了即提前返回，
        已完成的运动全部保留，会话状态可继续使用（换环境/重规划/再推进）。
        """
        with self.lock:
            if self.pose is None or self.goal is None or not self.path:
                raise SessionStateError("会话尚未成功规划，无法推进")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                raise InvalidInputError("duration 必须为数值", {"value": duration})
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise InvalidInputError(
                    "duration 必须为有限正数（秒）", {"value": duration}
                )
            tick_dt = self.controller_params.dt if dt is None else float(dt)
            if tick_dt <= 0:
                raise InvalidInputError("推进步长 dt 必须为正数", {"dt": tick_dt})

            steps_requested = max(1, int(math.floor(duration / tick_dt + 1e-9)))
            steps_requested = min(steps_requested, MAX_ADVANCE_TICKS)
            poses: list[Pose] = []
            samples: list[TrajectorySample] = []
            replanned_any = False
            completed = False
            last = None
            for i in range(steps_requested):
                result = self._tick_once(dt=tick_dt, max_v=max_v)
                self._record_sample(result, tick_dt)
                samples.append(self.trajectory[-1])
                poses.append(result.pose)
                last = result
                replanned_any = replanned_any or result.replanned
                if result.arrived:
                    completed = True
                    break
                if result.blocked:
                    break
            else:
                completed = True

            steps_executed = len(poses)
            elapsed = steps_executed * tick_dt
            if last is None:
                last = StepResult(
                    pose=self.pose, cmd=_zero_twist(), replanned=False,
                    arrived=False, blocked=False, note="",
                )
            if completed and last.arrived:
                note = f"按时长推进完成：{steps_executed} 个微步后到达目标"
            elif completed:
                note = (
                    f"按时长推进完成：执行 {steps_executed}/{steps_requested} "
                    f"个微步（{elapsed:.3f}s），未到终点也未被挡"
                )
            else:
                note = (
                    f"仅推进 {steps_executed}/{steps_requested} 个微步"
                    f"（{elapsed:.3f}s）后提前停止：{last.note}"
                )
            self.last_note = note
            self._touch(now)
            return AdvanceResult(
                poses=poses,
                samples=samples,
                steps_requested=steps_requested,
                steps_executed=steps_executed,
                requested_duration=duration,
                dt=tick_dt,
                elapsed=elapsed,
                completed=completed,
                arrived=bool(last.arrived),
                blocked=bool(last.blocked),
                replanned=replanned_any,
                note=note,
            )

    def _record_sample(self, result: StepResult, dt: float) -> None:
        """把一个实际发生的微步落盘到 FIFO 轨迹缓冲并推进仿真时钟。

        只记录执行尝试（含被挡时的原地零速步与到达那一步）；
        ARRIVED 后幂等的纯查询式步进不再产生新样本，避免长时间空转占内存。
        """
        sample = TrajectorySample(
            index=self.step_count,
            t=self.sim_time + dt,
            pose=result.pose,
            v=result.cmd.v,
            omega=result.cmd.omega,
            replanned=result.replanned,
            blocked=result.blocked,
            arrived=result.arrived,
        )
        if len(self.trajectory) == self.trajectory_capacity:
            self.trajectory_dropped += 1
        self.trajectory.append(sample)
        self.trajectory_total += 1
        self.sim_time += dt

    def get_trajectory(self, now: float) -> dict:
        """返回当前保留的轨迹样本及容量/丢弃统计（不改变状态）。

        超出容量后 FIFO 丢弃最旧样本：length 为当前保留条数，
        total 为会话累计产生条数，dropped=total-length 为因超限丢弃条数，
        truncated 表示当前序列不是从头开始（最早的已丢失）。
        """
        with self.lock:
            self._touch(now)
            return {
                "capacity": self.trajectory_capacity,
                "length": len(self.trajectory),
                "total": self.trajectory_total,
                "dropped": self.trajectory_dropped,
                "truncated": self.trajectory_dropped > 0,
                "sim_time": self.sim_time,
                "samples": [
                    {
                        "index": s.index,
                        "t": s.t,
                        "x": s.pose.x,
                        "y": s.pose.y,
                        "theta": s.pose.theta,
                        "v": s.v,
                        "omega": s.omega,
                        "replanned": s.replanned,
                        "blocked": s.blocked,
                        "arrived": s.arrived,
                    }
                    for s in self.trajectory
                ],
            }

    def preview(
        self,
        path: list[Point] | None,
        *,
        dt: float,
        max_v: float,
        max_ticks: int,
        start: Pose | None = None,
    ) -> PreviewResult:
        """只读预演：默认预演当前路径，不修改会话任何状态。

        仅在取快照时短暂持会话锁；仿真本身在锁外执行，
        因此不阻塞同会话的推进/查询，也不影响其它会话。
        """
        with self.lock:
            grid = self.effective_grid
            limits = self.limits
            ctrl_params = self.controller_params
            if path is None:
                use_path = list(self.path)
                use_start = self.pose if start is None else start
            else:
                use_path = list(path)
                if start is None:
                    raise InvalidInputError(
                        "预演自定义路线时必须提供 start 位姿"
                    )
                use_start = start
        if use_start is None:
            raise SessionStateError("会话尚未规划，且未提供 start/路线")
        return preview_path(
            grid, limits, use_start, use_path,
            dt=dt, max_v=max_v, max_ticks=max_ticks,
            controller_params=ctrl_params,
        )

    def _tick_once(self, *, dt: float | None = None,
                   max_v: float | None = None) -> StepResult:
        """执行一个微步（调用方持锁）。

        流程（整步在会话锁内完成，杜绝错序与状态撕裂）：
          1. 重建有效栅格（纳入最新动态障碍）；
          2. 若当前位姿已被障碍覆盖（机器人被困在障碍内）-> BLOCKED，零速；
          3. 检查剩余路径通畅性；被挡则尝试重规划；
             重规划失败（不可达/超时）-> BLOCKED，零速，状态可继续（环境变化后再 step 即恢复）；
          4. 跟踪器选前视点、算受约束指令、积分（可用本次调用的 dt/速度上限覆盖）；
          5. 积分后“位姿间线段扫掠胶囊”必须无碰撞，否则拒绝该运动（零速，BLOCKED）；
          6. 到达目标 -> ARRIVED。
        """
        if self.pose is None or self.goal is None or not self.path:
            raise SessionStateError("会话尚未成功规划，无法步进")

        tick_dt = self.controller_params.dt if dt is None else dt
        if tick_dt <= 0:
            raise InvalidInputError("推进步长 dt 必须为正数", {"dt": tick_dt})
        sim_limits = self.limits
        if max_v is not None:
            if max_v <= 0:
                raise InvalidInputError(
                    "max_v 必须为正数", {"max_v": max_v}
                )
            sim_limits = dataclasses.replace(self.limits, max_v=float(max_v))

        self._rebuild_effective_grid()
        note_parts: list[str] = []

        # 1) 当前位姿安全性
        if self.effective_grid.footprint_blocked(
            self.pose.x, self.pose.y, self.limits.footprint_radius
        ):
            self.status = SessionStatus.BLOCKED
            self.last_note = "当前位姿已被障碍占据，无法运动"
            return StepResult(
                pose=self.pose, cmd=_zero_twist(), replanned=False,
                arrived=False, blocked=True, note=self.last_note,
            )

        # 2) 局部路径通畅性检查：只检查当前位姿前方一个“安全视距”内的折线。
        #    视距 = 前视距离 + 当前速度的制动距离 + 本 tick 最大位移；
        #    即使跟踪器瞄准路径末端（短路径），也不会去检查视距之外的阻挡——
        #    机器人先朝目标前进，逼近到视距内才触发重规划/停车（能走多少走多少）。
        replanned = False
        planner = GridPlanner(self.effective_grid, self.planner_params)
        # 高速覆盖时提前量必须按本次实际速度上限给足（用 max 保证不低估）
        v_head = max(self.previous_v, sim_limits.max_v)
        horizon = (
            self.controller_params.lookahead_cells
            + v_head * v_head / (2.0 * self.limits.max_acc + 1e-12)
            + sim_limits.max_v * tick_dt
        )
        local_blocked_idx = self._local_blocked_index(
            self.effective_grid, horizon, sim_limits.footprint_radius
        )
        if local_blocked_idx >= 0:
            note_parts.append(
                f"前方 {horizon:.2f}m 视距（沿路径弧长）内被阻挡（局部点 {local_blocked_idx}）"
            )
            try:
                new_path = planner.plan(
                    Point(self.pose.x, self.pose.y), self.goal
                )
                self.path = new_path
                self.path_index = 0
                replanned = True
                note_parts.append(f"自动重规划成功（{len(new_path)} 点）")
            except Exception as exc:  # 不可达 / 超时 / 起终点非法
                # 重规划失败：制动视距内确认无路，本 tick 零速停住。
                # 下一 tick 会用最新栅格再次尝试（撤障/换环境后自动恢复）。
                self.previous_v = 0.0
                self.status = SessionStatus.BLOCKED
                self.last_note = (
                    f"{'; '.join(note_parts)}；重规划失败：{exc.message if hasattr(exc, 'message') else exc}"
                )
                return StepResult(
                    pose=self.pose, cmd=_zero_twist(), replanned=False,
                    arrived=False, blocked=True, note=self.last_note,
                )

        # 3) 到达判定（在执行运动之前先看是否已经在终点容差内）
        if self.tracker.at_goal(self.pose, self.goal):
            self.status = SessionStatus.ARRIVED
            self.last_note = "已到达目标"
            return StepResult(
                pose=self.pose, cmd=_zero_twist(), replanned=replanned,
                arrived=True, blocked=False, note=self.last_note,
            )

        # 4) 跟踪控制（本次调用可覆盖 dt 与速度上限）
        tracker = self.tracker
        if dt is not None or max_v is not None:
            overridden_params = (
                self.controller_params if dt is None
                else ControllerParams(
                    k_linear=self.controller_params.k_linear,
                    k_angular=self.controller_params.k_angular,
                    goal_tolerance=self.controller_params.goal_tolerance,
                    lookahead_cells=self.controller_params.lookahead_cells,
                    dt=tick_dt,
                )
            )
            tracker = TrajectoryTracker(sim_limits, overridden_params)

        aim, new_idx = tracker.find_aim_point(
            self.pose, self.path, self.path_index
        )
        # 若瞄准点本身落在障碍上（典型：短路径的终点在封死墙后），
        # 沿剩余路径回退到视距内第一个安全点作为瞄准，避免朝墙直冲；
        # 瞄准点安全时保持原纯追踪行为（不因视距截断而低速爬行）。
        if self.effective_grid.footprint_blocked(
            aim.x, aim.y, sim_limits.footprint_radius
        ):
            aim = self._safe_aim_within_horizon(horizon)
        self.path_index = new_idx
        cmd = tracker.compute_command(self.pose, aim, self.previous_v)
        next_pose = tracker.advance(self.pose, cmd)

        # 5) 运动后碰撞复核：端点 + 位姿间线段扫掠胶囊，大跨度微步也不得穿障
        seg_blocked = self.effective_grid.segment_blocked(
            self.pose.x, self.pose.y,
            next_pose.x, next_pose.y,
            self.limits.footprint_radius,
        )
        if seg_blocked or self.effective_grid.footprint_blocked(
            next_pose.x, next_pose.y, self.limits.footprint_radius
        ):
            self.status = SessionStatus.BLOCKED
            self.last_note = "下一步预测扫掠轨迹触障或越界，拒绝执行该速度指令"
            return StepResult(
                pose=self.pose, cmd=_zero_twist(), replanned=replanned,
                arrived=False, blocked=True, note=self.last_note,
            )

        self.pose = next_pose
        self.previous_v = cmd.v
        self.step_count += 1
        arrived = self.tracker.at_goal(next_pose, self.goal)
        self.status = (
            SessionStatus.ARRIVED if arrived else SessionStatus.RUNNING
        )
        note_parts.append(
            f"步进 {self.step_count}：v={cmd.v:.3f} omega={cmd.omega:.3f} "
            f"瞄准路径点[{self.path_index}]"
        )
        if arrived:
            note_parts.append("已到达目标")
        self.last_note = "；".join(note_parts)
        return StepResult(
            pose=next_pose, cmd=cmd, replanned=replanned,
            arrived=arrived, blocked=False, note=self.last_note,
        )

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
            self._reset_trajectory()
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
                sim_time=self.sim_time,
                trajectory_length=len(self.trajectory),
                trajectory_total=self.trajectory_total,
                trajectory_dropped=self.trajectory_dropped,
                trajectory_capacity=self.trajectory_capacity,
                trajectory_truncated=self.trajectory_dropped > 0,
            )


def _zero_twist():
    from .models import Twist

    return Twist(v=0.0, omega=0.0)
