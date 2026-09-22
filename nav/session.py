"""仿真会话：持有地图/参数/位姿/路径，串行化状态迁移。

动态障碍以世界坐标矩形列表表示，每次步进时全量栅格化叠加到静态地图上，
形成“当前有效占用栅格”；规划与路径通畅性检查都基于该栅格，
从而保证任何时刻轨迹都不穿越已上报的动态障碍。
"""

from __future__ import annotations

import math
import threading
import time

from .controller import TrajectoryTracker
from .errors import (
    SessionStateError,
    StartInObstacleError,
)
from .grid import OccupancyGrid, rasterize_rect
from .models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    SessionSnapshot,
    SessionStatus,
    StepResult,
)
from .planner import GridPlanner


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
    ) -> None:
        self.id = session_id
        self.base_grid = grid
        self.limits = limits
        # 规划安全余量：用户显式配置的膨胀格数 + 足迹半径折算格数（向上取整），
        # 保证规划出的路径与障碍的间距足以容纳连续足迹圆 + 跟踪侧向偏差。
        footprint_cells = math.ceil(
            limits.footprint_radius / grid.resolution
        )
        # 额外 1 格跟踪缓冲，吸收转弯时的侧向控制偏差
        effective_inflation = (
            planner_params.obstacle_inflation_cells + footprint_cells + 1
        )
        self.planner_params = PlannerParams(
            obstacle_inflation_cells=effective_inflation,
            allow_diagonal=planner_params.allow_diagonal,
            max_expanded_nodes=planner_params.max_expanded_nodes,
            cost_diagonal=planner_params.cost_diagonal,
        )
        self.user_planner_params = planner_params
        # 前视距离下限：1.5 个栅格，保证粗分辨率下跟踪稳定
        lookahead = max(
            float(controller_params.lookahead_cells),
            1.5 * grid.resolution,
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
    ) -> "NavigationSession":
        return cls(
            session_id="",
            grid=grid,
            limits=limits,
            planner_params=planner_params,
            controller_params=controller_params,
            created_at=now,
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
        """推进一步。

        流程（整步在会话锁内完成，杜绝错序与状态撕裂）：
          1. 重建有效栅格（纳入最新动态障碍）；
          2. 若当前位姿已被障碍覆盖（机器人被困在障碍内）-> BLOCKED，零速；
          3. 检查剩余路径通畅性；被挡则尝试重规划；
             重规划失败（不可达/超时）-> BLOCKED，零速，状态可继续（环境变化后再 step 即恢复）；
          4. 跟踪器选前视点、算受约束指令、积分；
          5. 积分后位姿必须无碰撞，否则拒绝该运动（零速，BLOCKED）；
          6. 到达目标 -> ARRIVED。
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

            # 4) 跟踪控制
            aim, new_idx = self.tracker.find_aim_point(
                self.pose, self.path, self.path_index
            )
            self.path_index = new_idx
            cmd = self.tracker.compute_command(self.pose, aim, self.previous_v)
            next_pose = self.tracker.advance(self.pose, cmd)

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
            self._touch(now)
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


def _zero_twist():
    from .models import Twist

    return Twist(v=0.0, omega=0.0)
