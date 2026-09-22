"""局部轨迹跟踪：纯追踪（Pure Pursuit）+ 运动学钳制 + 弧段碰撞复核。

控制模型
--------
- 全局路径为栅格中心折线；跟踪器维护“最近路径点索引 progress”，
  在前方 lookahead 弧长处选取追踪目标点 g。
- 纯追踪曲率（自行车等效）：
      kappa = 2 * sin(alpha) / Ld,  alpha = atan2(g_y-p_y, g_x-p_x) - theta
  随后由 :class:`Kinematics` 统一钳制到速度/曲率/角速度包络。
- 速度按曲率减速：v = v_max * max(k_turn, 1 - |kappa|/kappa_max * (1-k_turn))，
  并受最大加速度约束（相对上一速度）。

安全保证
--------
每个控制周期对实际执行弧段做**密集等距采样**（间距 <= collision_spacing，
默认取 min(分辨率/2, robot_radius)），逐点用当前（含动态障碍）地图做圆盘
碰撞检测；若首选指令碰撞，依次尝试降速、原地转向对准、零速保持，均不通过
则本周期不移动并返回 blocked=True，由会话层触发重规划或进入 BLOCKED。
任何被实际执行的状态都不可能穿越障碍或超出运动学包络。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.errors import ErrorCode, NavigationError
from app.gridmap import OccupancyGrid
from app.kinematics import Kinematics
from app.planner import Plan


@dataclass
class StepResult:
    """单步执行结果（全部为执行**之后**的真实量）。"""

    x: float
    y: float
    theta: float
    speed: float
    curvature: float
    angular_speed: float
    reached: bool
    blocked: bool
    obstructed: bool
    replanned: bool
    lookahead_index: int
    progress_index: int
    distance_to_goal: float
    collisions_checked: int
    replan_reason: str | None = None
    candidates_tried: int = 0
    events: list[str] = field(default_factory=list)


@dataclass
class TrackerConfig:
    lookahead: float = 0.3
    goal_tolerance: float = 0.05
    heading_tolerance: float = 0.08           # rad，到达后对准目标朝向的容差
    pre_align_threshold: float = 0.35        # rad，超过先原地自转对准
    path_horizon: float = 0.6                 # m，路径前瞻预警距离
    collision_spacing: float | None = None  # None -> min(res/2, robot_radius)
    turn_speed_fraction: float = 0.25        # 原地/大曲率时保留的速度比例
    min_speed_fraction: float = 0.12         # 爬行最低速度比例

    def __post_init__(self) -> None:
        if self.lookahead <= 0.0:
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                f"lookahead={self.lookahead!r} 非法：必须为正数",
                {"lookahead": self.lookahead},
            )
        if self.goal_tolerance <= 0.0:
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                f"goal_tolerance={self.goal_tolerance!r} 非法：必须为正数",
                {"goal_tolerance": self.goal_tolerance},
            )
        if not (0.0 < self.turn_speed_fraction <= 1.0):
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                "turn_speed_fraction 必须在 (0,1] 内",
                {"turn_speed_fraction": self.turn_speed_fraction},
            )
        if not (0.0 < self.min_speed_fraction <= 1.0):
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                "min_speed_fraction 必须在 (0,1] 内",
                {"min_speed_fraction": self.min_speed_fraction},
            )


def _path_nearest_index(
    path: list[tuple[float, float]],
    x: float,
    y: float,
    start_from: int,
    look_behind: int = 4,
) -> int:
    """从 start_from 附近（允许回看 look_behind 个点）找最近路径点。"""

    lo = max(0, start_from - look_behind)
    hi = min(len(path), start_from + look_behind + 1)
    best = start_from
    best_d = math.inf
    for i in range(lo, hi):
        px, py = path[i]
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_d = d
            best = i
    return best


def _lookahead_point(
    path: list[tuple[float, float]], nearest: int, lookahead: float
) -> tuple[tuple[float, float], int]:
    """沿折线累计弧长，取距 nearest >= lookahead 的插值点。"""

    x, y = path[nearest]
    acc = 0.0
    i = nearest
    while i + 1 < len(path):
        ax, ay = path[i]
        bx, by = path[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if acc + seg >= lookahead:
            t = 0.0 if seg == 0.0 else (lookahead - acc) / seg
            return (ax + (bx - ax) * t, ay + (by - ay) * t), i + 1
        acc += seg
        i += 1
    return path[-1], len(path) - 1


def _arc_safe(
    grid: OccupancyGrid,
    kin: Kinematics,
    x: float,
    y: float,
    theta: float,
    speed: float,
    curvature: float,
    dt: float,
    spacing: float,
    omega: float | None = None,
) -> tuple[bool, int]:
    """对执行弧段密集采样做圆盘碰撞检测。返回 (安全, 检查点数)。"""

    points = kin.arc_points(x, y, theta, speed, curvature, dt, spacing, omega)
    radius = kin.robot_radius
    for px, py in points:
        if not math.isfinite(px) or not math.isfinite(py) or grid.disk_collides(
            px, py, radius
        ):
            return False, len(points)
    return True, len(points)


def _rotate_in_place(
    grid: OccupancyGrid,
    kin: Kinematics,
    x: float,
    y: float,
    theta: float,
    target_theta: float,
    spacing: float,
    cfg: TrackerConfig,
) -> tuple[StepResult | None, tuple[float, float, float, float]]:
    """朝 target_theta 原地自转一个周期；已对准则返回 None。"""

    yaw_err = math.atan2(
        math.sin(target_theta - theta), math.cos(target_theta - theta)
    )
    if abs(yaw_err) <= cfg.heading_tolerance:
        return None, (x, y, theta, 0.0)
    omega = max(-kin.max_angular_speed, min(kin.max_angular_speed, yaw_err / kin.dt))
    safe, n = _arc_safe(
        grid, kin, x, y, theta, 0.0, 0.0, kin.dt, spacing, omega=omega
    )
    if not safe:
        return None, (x, y, theta, 0.0)
    xn, yn, thn = kin.step_arc(x, y, theta, 0.0, 0.0, kin.dt, omega=omega)
    return (
        StepResult(
            x=xn, y=yn, theta=thn, speed=0.0, curvature=0.0, angular_speed=omega,
            reached=False, blocked=False, obstructed=False, replanned=False,
            lookahead_index=0, progress_index=0,
            distance_to_goal=0.0, collisions_checked=n,
            events=["rotating_in_place"],
        ),
        (xn, yn, thn, 0.0),
    )


def pure_pursuit_step(
    grid: OccupancyGrid,
    kin: Kinematics,
    plan: Plan,
    state: tuple[float, float, float, float],
    *,
    cfg: TrackerConfig | None = None,
    progress: int = 0,
    goal_theta: float | None = None,
) -> tuple[StepResult, tuple[float, float, float, float], int]:
    """执行一个控制周期。

    返回 (步结果, 新状态 (x,y,theta,v), 新 progress)。
    本函数不修改 plan；是否重规划由会话层决定。blocked=True 表示本周期
    所有候选指令均不安全，机器人保持原地不动。
    goal_theta 给定时，进入容差圆后会原地转向对准该朝向再宣告 reached。
    """

    cfg = cfg or TrackerConfig()
    x, y, theta, prev_speed = state
    path = plan.world_path
    if not path:
        raise NavigationError(ErrorCode.INTERNAL_ERROR, "路径为空，无法跟踪", {})

    spacing = cfg.collision_spacing
    if spacing is None:
        spacing = max(1e-4, min(grid.resolution * 0.5,
                                kin.robot_radius if kin.robot_radius > 0
                                else grid.resolution * 0.5))

    events: list[str] = []
    nearest = _path_nearest_index(path, x, y, progress)
    goal_x, goal_y = path[-1]
    dist_goal = math.hypot(goal_x - x, goal_y - y)

    if dist_goal <= cfg.goal_tolerance:
        # 进入目标容差圆：刹停；若指定目标朝向则原地自转对准
        target_theta = goal_theta
        if target_theta is None and len(path) >= 2:
            ax, ay = path[-2]
            target_theta = math.atan2(goal_y - ay, goal_x - ax)
        if target_theta is not None:
            rot, new_state = _rotate_in_place(
                grid, kin, x, y, theta, target_theta, spacing, cfg
            )
            if rot is not None:
                rot.lookahead_index = len(path) - 1
                rot.progress_index = nearest
                rot.distance_to_goal = math.hypot(goal_x - rot.x, goal_y - rot.y)
                return rot, new_state, nearest
        sr = StepResult(
            x=x, y=y, theta=theta, speed=0.0, curvature=0.0, angular_speed=0.0,
            reached=True, blocked=False, obstructed=False, replanned=False,
            lookahead_index=len(path) - 1, progress_index=nearest,
            distance_to_goal=dist_goal, collisions_checked=0,
            events=events + ["goal_tolerance_reached"],
        )
        return sr, (x, y, theta, 0.0), nearest

    target, look_idx = _lookahead_point(path, nearest, cfg.lookahead)
    bearing = math.atan2(target[1] - y, target[0] - x)
    heading_error = bearing - theta
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    # 路径前瞻预警：沿剩余路径走 path_horizon 米，逐折线点检查圆盘碰撞。
    # 看到前方路径被新障碍占据即返回 blocked（不移动），交给会话层重规划，
    # 这样动态障碍不必等到机器人鼻尖撞上才触发。
    radius = kin.robot_radius
    acc = 0.0
    horizon_idx = nearest
    for i in range(nearest, len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        steps_on_seg = max(1, math.ceil(seg / spacing))
        for t in range(steps_on_seg + 1):
            px = ax + (bx - ax) * t / steps_on_seg
            py = ay + (by - ay) * t / steps_on_seg
            if grid.disk_collides(px, py, radius):
                sr = StepResult(
                    x=x, y=y, theta=theta, speed=0.0, curvature=0.0,
                    angular_speed=0.0,
                    reached=False, blocked=True, obstructed=False,
                    replanned=False,
                    lookahead_index=i, progress_index=nearest,
                    distance_to_goal=dist_goal, collisions_checked=0,
                    replan_reason="path_blocked_ahead",
                    events=events + [f"blocked:path_blocked_ahead@{i}"],
                )
                return sr, (x, y, theta, 0.0), nearest
        acc += seg
        horizon_idx = i + 1
        if acc >= cfg.path_horizon:
            break

    # 起步/重规划后朝向偏差过大：先原地自转对准前方路径方向（不位移）
    if abs(heading_error) > cfg.pre_align_threshold:
        rot, new_state = _rotate_in_place(
            grid, kin, x, y, theta, bearing, spacing, cfg
        )
        if rot is not None:
            rot.lookahead_index = look_idx
            rot.progress_index = nearest
            rot.distance_to_goal = dist_goal
            return rot, new_state, nearest

    # 纯追踪曲率，Ld 取实际到目标点距离（避免近点时曲率虚高）
    ld = max(math.hypot(target[0] - x, target[1] - y), 1e-9)
    kappa_cmd = 2.0 * math.sin(heading_error) / ld
    v_max = kin.max_linear_speed

    # 速度曲线：大转角减速；朝向偏差大时进一步降速
    k_frac = min(1.0, abs(kappa_cmd) / kin.max_curvature)
    h_frac = min(1.0, abs(heading_error) / (math.pi / 2.0))
    desired_speed = v_max * max(
        cfg.min_speed_fraction,
        cfg.turn_speed_fraction
        + (1.0 - cfg.turn_speed_fraction) * (1.0 - max(k_frac, h_frac)),
    )
    # 接近目标减速
    desired_speed = min(desired_speed, max(dist_goal / kin.dt, v_max * cfg.min_speed_fraction))
    # 加速度限制
    dv_max = kin.max_linear_accel * kin.dt
    desired_speed = max(
        prev_speed - dv_max, min(prev_speed + dv_max, desired_speed)
    )

    candidates: list[tuple[float, float, str]] = []
    v_cmd, k_cmd, _ = kin.clamp_command(desired_speed, kappa_cmd)
    candidates.append((v_cmd, k_cmd, "pursuit"))
    # 候选 2：半速纯追踪曲率
    v2, k2, _ = kin.clamp_command(v_cmd * 0.5, kappa_cmd)
    candidates.append((v2, k2, "slow"))
    # 候选 3：爬行速度
    v3, k3, _ = kin.clamp_command(v_max * cfg.min_speed_fraction, kappa_cmd)
    candidates.append((v3, k3, "creep"))
    # 候选 4：完全停止（零位移，起点无碰撞时恒安全）
    candidates.append((0.0, 0.0, "stop"))

    checked_total = 0
    moving: list[tuple[float, float, str]] = []
    for v_c, k_c, tag in candidates:
        v_c, k_c, _ = kin.clamp_command(v_c, k_c)
        safe, n = _arc_safe(grid, kin, x, y, theta, v_c, k_c, kin.dt, spacing)
        checked_total += n
        if safe and tag != "stop":
            moving.append((v_c, k_c, tag))
            break
    stop_safe, n_stop = _arc_safe(grid, kin, x, y, theta, 0.0, 0.0, kin.dt, spacing)
    checked_total += n_stop

    if not stop_safe:
        # 机器人当前位置已处于碰撞态：不产生任何移动，交由上层失败恢复
        sr = StepResult(
            x=x, y=y, theta=theta, speed=0.0, curvature=0.0, angular_speed=0.0,
            reached=False, blocked=True, obstructed=True, replanned=False,
            lookahead_index=look_idx, progress_index=nearest,
            distance_to_goal=dist_goal, collisions_checked=checked_total,
            replan_reason="current_pose_in_collision",
            candidates_tried=len(candidates),
            events=events + ["blocked:current_pose_in_collision"],
        )
        return sr, (x, y, theta, 0.0), nearest

    if not moving:
        # 所有移动指令都不安全，但原地安全：刹停并请求重规划
        sr = StepResult(
            x=x, y=y, theta=theta, speed=0.0, curvature=0.0, angular_speed=0.0,
            reached=False, blocked=True, obstructed=False, replanned=False,
            lookahead_index=look_idx, progress_index=nearest,
            distance_to_goal=dist_goal, collisions_checked=checked_total,
            replan_reason="path_obstructed_ahead",
            candidates_tried=len(candidates),
            events=events + ["blocked:path_obstructed_ahead"],
        )
        return sr, (x, y, theta, 0.0), nearest

    v_c, k_c, tag = moving[0]
    if tag != "pursuit":
        events.append(f"speed_degraded:{tag}")
    xn, yn, thn = kin.step_arc(x, y, theta, v_c, k_c, kin.dt)
    omega = v_c * k_c

    sr = StepResult(
        x=xn, y=yn, theta=thn, speed=v_c, curvature=k_c, angular_speed=omega,
        reached=False, blocked=False, obstructed=False, replanned=False,
        lookahead_index=look_idx, progress_index=nearest,
        distance_to_goal=math.hypot(goal_x - xn, goal_y - yn),
        collisions_checked=checked_total,
        candidates_tried=1,
        events=events,
    )
    new_nearest = _path_nearest_index(path, xn, yn, nearest)
    return sr, (xn, yn, thn, v_c), new_nearest
