"""差分驱动机器人运动学模型与约束。

模型约定
--------
状态量为 (x, y, theta, v)，控制量为 (v, kappa)：
- v       线速度（m/s，允许负值倒车），|v| <= max_linear_speed
- kappa   瞬时曲率（1/m），|kappa| <= max_curvature
- 角速度 omega = v * kappa，执行前钳制到 |omega| <= max_angular_speed

单步采用**精确圆弧积分**，避免欧拉积分在大曲率下的数值漂移：
    dtheta = v*kappa*dt
    x' = x + (sin(theta+dtheta) - sin(theta)) / kappa
    y' = y + (cos(theta) - cos(theta+dtheta)) / kappa
小曲率时退化为直线积分（数值稳定）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.errors import ErrorCode, NavigationError

_TURN_IN_PLACE_EPS = 1e-9


def _wrap_angle(theta: float) -> float:
    """归一化到 (-pi, pi]。"""

    wrapped = (theta + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi:
        return math.pi
    return wrapped


@dataclass(frozen=True)
class Kinematics:
    """差分驱动运动学参数（全部为正量约束）。"""

    max_linear_speed: float       # m/s
    max_angular_speed: float      # rad/s
    max_curvature: float          # 1/m，= 1/最小转弯半径
    robot_radius: float           # m，圆盘 footprint
    max_linear_accel: float       # m/s^2
    dt: float                     # s，默认步进周期

    def __post_init__(self) -> None:
        def bad(name: str, value: object) -> NavigationError:
            return NavigationError(
                ErrorCode.INVALID_KINEMATICS,
                f"运动学参数 {name}={value!r} 非法：必须为有限正数（robot_radius 允许为 0）",
                {"parameter": name, "value": value},
            )

        finite_pos = (
            ("max_linear_speed", self.max_linear_speed),
            ("max_angular_speed", self.max_angular_speed),
            ("max_curvature", self.max_curvature),
            ("max_linear_accel", self.max_linear_accel),
            ("dt", self.dt),
        )
        for name, value in finite_pos:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise bad(name, value)
        radius = self.robot_radius
        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0.0:
            raise bad("robot_radius", radius)
        # 物理一致性：满线速度下达到曲率上限所需角速度不得超过角速度上限。
        if self.max_linear_speed * self.max_curvature > self.max_angular_speed + 1e-9:
            raise NavigationError(
                ErrorCode.INVALID_KINEMATICS,
                "运动学参数不自洽：max_linear_speed*max_curvature "
                f"= {self.max_linear_speed * self.max_curvature:.6g} 超过 "
                f"max_angular_speed={self.max_angular_speed:.6g}",
                {
                    "max_linear_speed": self.max_linear_speed,
                    "max_curvature": self.max_curvature,
                    "required_angular_speed": self.max_linear_speed * self.max_curvature,
                    "max_angular_speed": self.max_angular_speed,
                },
            )

    def min_turning_radius(self) -> float:
        return 1.0 / self.max_curvature

    def feasible_speed(self, speed: float) -> bool:
        return math.isfinite(speed) and abs(speed) <= self.max_linear_speed + 1e-12

    def feasible_curvature(self, curvature: float) -> bool:
        return math.isfinite(curvature) and abs(curvature) <= self.max_curvature + 1e-9

    def clamp_command(
        self, speed: float, curvature: float, omega: float | None = None
    ) -> tuple[float, float, float]:
        """将指令钳制到速度/曲率/角速度包络内。

        - 常规行驶指令 (speed, curvature)：
              |speed'| <= max_linear_speed
              |curvature'| <= max_curvature
              |omega'| = |speed'*curvature'| <= max_angular_speed
        - 原地自转：speed=0 且显式给定 omega（差分底盘两轮反转），
          此时曲率无定义，返回 curvature=0，角速度被钳到 omega_max。
        """

        if speed == 0.0 and omega is not None:
            w = max(-self.max_angular_speed, min(self.max_angular_speed, omega))
            return 0.0, 0.0, w
        v = max(-self.max_linear_speed, min(self.max_linear_speed, speed))
        k = max(-self.max_curvature, min(self.max_curvature, curvature))
        w = v * k
        if abs(w) > self.max_angular_speed:
            # v != 0（omega != 0 蕴含 v != 0）：降低曲率以满足角速度上限
            k = math.copysign(self.max_angular_speed / abs(v), k)
            w = v * k
        return v, k, w

    def step_arc(
        self,
        x: float,
        y: float,
        theta: float,
        speed: float,
        curvature: float,
        dt: float | None = None,
        omega: float | None = None,
    ) -> tuple[float, float, float]:
        """精确圆弧积分：由行驶指令推进一个 dt；speed=0+omega 时原地自转。"""

        step_dt = self.dt if dt is None else dt
        if not math.isfinite(step_dt) or step_dt < 0.0:
            raise NavigationError(
                ErrorCode.INVALID_KINEMATICS,
                f"步进时长 dt={step_dt!r} 非法：必须为非负有限数",
                {"dt": step_dt},
            )
        v, k, w = self.clamp_command(speed, curvature, omega)
        if v == 0.0 and omega is not None:
            return x, y, _wrap_angle(theta + w * step_dt)
        dtheta = v * k * step_dt
        if abs(k) < _TURN_IN_PLACE_EPS:
            # 直线（含原地/静止）： x += v*cos*t, y += v*sin*t
            xn = x + v * step_dt * math.cos(theta)
            yn = y + v * step_dt * math.sin(theta)
        else:
            xn = x + (math.sin(theta + dtheta) - math.sin(theta)) / k
            yn = y + (math.cos(theta) - math.cos(theta + dtheta)) / k
        return xn, yn, _wrap_angle(theta + dtheta)

    def arc_points(
        self,
        x: float,
        y: float,
        theta: float,
        speed: float,
        curvature: float,
        dt: float,
        max_spacing: float,
        omega: float | None = None,
    ) -> list[tuple[float, float]]:
        """沿本步圆弧按弧长等距采样（含起点与终点），供碰撞复核。

        采样间距 <= max_spacing（米），保证最窄通道场景下不会漏检。
        """

        v, k, w = self.clamp_command(speed, curvature, omega)
        if v == 0.0 and omega is not None:
            # 原地自转：位置不变，仅采样当前点（碰撞只需检测 footprint 当前位置）
            return [(x, y)]
        arc_len = abs(v) * dt
        n = 1
        if max_spacing > 0.0 and arc_len > 0.0:
            n = max(1, math.ceil(arc_len / max_spacing))
        points: list[tuple[float, float]] = []
        for i in range(n + 1):
            xi, yi, _ = self.step_arc(
                x, y, theta, v, k, dt * i / n, omega=None
            )
            points.append((xi, yi))
        return points


def default_kinematics(
    dt: float = 0.1,
    max_linear_speed: float = 1.0,
    max_angular_speed: float = 1.0,
    robot_radius: float = 0.15,
    max_linear_accel: float = 1.0,
) -> Kinematics:
    """由线速度/角速度上限推导曲率上限 kappa_max = omega_max / v_max。

    差分底盘在低速下可原地转向，曲率上限取“满线速度时仍可执行”的保守值，
    保证任何被钳制后的指令都在两轮差速可行域内。
    """

    return Kinematics(
        max_linear_speed=max_linear_speed,
        max_angular_speed=max_angular_speed,
        max_curvature=max_angular_speed / max_linear_speed,
        robot_radius=robot_radius,
        max_linear_accel=max_linear_accel,
        dt=dt,
    )
