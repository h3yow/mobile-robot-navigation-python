"""导航服务错误类型。

每个错误都带稳定的 ``code`` 字符串，供 API 与测试做可区分判定；
``http_status`` 给出对应 HTTP 状态码。
"""

from __future__ import annotations


class NavError(Exception):
    """所有导航领域错误的基类。"""

    code: str = "nav_error"
    http_status: int = 400

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class InvalidInputError(NavError):
    """请求参数结构或取值非法（分辨率/尺寸/参数越界等）。"""

    code = "invalid_input"
    http_status = 422


class MapShapeError(NavError):
    """地图数据形状/维度不一致。"""

    code = "map_shape_error"
    http_status = 422


class CoordinateOutOfBoundsError(NavError):
    """坐标落在地图覆盖范围之外。"""

    code = "coordinate_out_of_bounds"
    http_status = 422


class StartInObstacleError(NavError):
    """起点被障碍占据。"""

    code = "start_in_obstacle"
    http_status = 422


class GoalInObstacleError(NavError):
    """终点被障碍占据。"""

    code = "goal_in_obstacle"
    http_status = 422


class NoPathError(NavError):
    """起点终点合法但不存在无碰撞可达路径。"""

    code = "goal_unreachable"
    http_status = 422


class PlanningTimeoutError(NavError):
    """规划在给定节点预算内未能完成（大图保护）。"""

    code = "planning_timeout"
    http_status = 504


class SessionNotFoundError(NavError):
    """会话不存在。"""

    code = "session_not_found"
    http_status = 404


class SessionExpiredError(NavError):
    """会话超过空闲 TTL，已被回收。"""

    code = "session_expired"
    http_status = 404


class SessionStateError(NavError):
    """会话当前状态不允许该操作（如未规划就步进）。"""

    code = "session_state_error"
    http_status = 409


class IdempotencyConflictError(NavError):
    """同一幂等键被用于不同的请求体。"""

    code = "idempotency_conflict"
    http_status = 409


class TrajectoryBlockedError(NavError):
    """执行期动态障碍封死所有可行运动（内部信号，通常映射为重规划失败）。"""

    code = "trajectory_blocked"
    http_status = 409
