"""会话级服务：创建/查询/步进/重规划的并发安全入口、TTL 与幂等缓存。"""

from __future__ import annotations

import threading
import time

from .models import SessionSnapshot, StepResult


class NavigationService:
    """进程内多会话管理器。"""

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        # 最小骨架：仅初始化容器，方法逻辑后续填充
        self.ttl_seconds = float(ttl_seconds)
        self._sessions: dict = {}
        self._idempotent: dict = {}
        self._lock = threading.RLock()

    def create_session(self, payload: dict, idempotency_key: str | None) -> str:
        raise NotImplementedError

    def plan_session(self, session_id: str, payload: dict) -> SessionSnapshot:
        raise NotImplementedError

    def step_session(
        self, session_id: str, payload: dict | None, idempotency_key: str | None
    ) -> StepResult:
        raise NotImplementedError

    def replan_session(self, session_id: str) -> SessionSnapshot:
        raise NotImplementedError

    def reset_session(self, session_id: str) -> SessionSnapshot:
        raise NotImplementedError

    def get_session(self, session_id: str) -> SessionSnapshot:
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    def purge_expired(self, now: float | None = None) -> int:
        """回收超时空闲会话，返回回收数量。"""
        raise NotImplementedError
