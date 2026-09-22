"""导航服务统一错误码与异常类型（骨架：仅类型定义）。"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """所有可对外区分的错误原因。"""

    # 地图与参数输入
    MAP_EMPTY = "MAP_EMPTY"
    MAP_SHAPE_INVALID = "MAP_SHAPE_INVALID"
    MAP_DIMENSION_MISMATCH = "MAP_DIMENSION_MISMATCH"
    MAP_INVALID_CELL_VALUE = "MAP_INVALID_CELL_VALUE"
    MAP_INVALID_RESOLUTION = "MAP_INVALID_RESOLUTION"
    MAP_INVALID_ORIGIN = "MAP_INVALID_ORIGIN"
    MAP_NO_FREE_SPACE = "MAP_NO_FREE_SPACE"
    INVALID_KINEMATICS = "INVALID_KINEMATICS"
    INVALID_REQUEST = "INVALID_REQUEST"

    # 坐标与规划
    COORDINATE_OUT_OF_BOUNDS = "COORDINATE_OUT_OF_BOUNDS"
    START_IN_OBSTACLE = "START_IN_OBSTACLE"
    GOAL_IN_OBSTACLE = "GOAL_IN_OBSTACLE"
    GOAL_UNREACHABLE = "GOAL_UNREACHABLE"
    PLAN_TIME_BUDGET_EXCEEDED = "PLAN_TIME_BUDGET_EXCEEDED"

    # 会话与并发
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_ALREADY_FINISHED = "SESSION_ALREADY_FINISHED"
    SESSION_LOCK_TIMEOUT = "SESSION_LOCK_TIMEOUT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"

    INTERNAL_ERROR = "INTERNAL_ERROR"


# 各错误码默认 HTTP 状态码
DEFAULT_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.MAP_EMPTY: 422,
    ErrorCode.MAP_SHAPE_INVALID: 422,
    ErrorCode.MAP_DIMENSION_MISMATCH: 422,
    ErrorCode.MAP_INVALID_CELL_VALUE: 422,
    ErrorCode.MAP_INVALID_RESOLUTION: 422,
    ErrorCode.MAP_INVALID_ORIGIN: 422,
    ErrorCode.MAP_NO_FREE_SPACE: 422,
    ErrorCode.INVALID_KINEMATICS: 422,
    ErrorCode.INVALID_REQUEST: 422,
    ErrorCode.COORDINATE_OUT_OF_BOUNDS: 422,
    ErrorCode.START_IN_OBSTACLE: 422,
    ErrorCode.GOAL_IN_OBSTACLE: 422,
    ErrorCode.GOAL_UNREACHABLE: 422,
    ErrorCode.PLAN_TIME_BUDGET_EXCEEDED: 408,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.SESSION_ALREADY_FINISHED: 409,
    ErrorCode.SESSION_LOCK_TIMEOUT: 409,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.INTERNAL_ERROR: 500,
}


class NavigationError(Exception):
    """业务异常：携带可区分错误码与结构化细节。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        detail: dict | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail: dict = detail or {}
        self.http_status = http_status or DEFAULT_HTTP_STATUS[code]

    def to_body(self) -> dict:
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "detail": self.detail,
            }
        }
