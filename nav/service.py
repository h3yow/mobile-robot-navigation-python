"""会话级服务：创建/查询/步进/重规划的并发安全入口、TTL 与幂等缓存。

并发模型：
  - 服务级 RLock 保护会话字典与幂等缓存；
  - 每个会话自带 RLock，step/plan/replan 等状态迁移在会话锁内原子完成，
    因此同一会话的并发请求不会错序或读到撕裂状态（请求按获取锁的顺序串行化）；
  - 不同会话持有不同锁，互不阻塞。
幂等：
  - create/step 支持 Idempotency-Key；同键同体重放返回首次结果，
    同键不同体返回 idempotency_conflict（409）。
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from .errors import (
    IdempotencyConflictError,
    InvalidInputError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionStateError,
)
from .grid import OccupancyGrid
from .models import (
    AdvanceResult,
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    SessionSnapshot,
    StepResult,
)
from .rehearse import rehearse
from .session import DEFAULT_TRAJECTORY_LIMIT, NavigationSession

_DEFAULT_LIMITS = dict(
    max_v=1.0, min_v=0.0, max_omega=1.5, max_curvature=2.0,
    max_acc=2.0, footprint_radius=0.3,
)
_DEFAULT_PLANNER = dict(
    obstacle_inflation_cells=0, allow_diagonal=True,
    max_expanded_nodes=200_000,
)
_DEFAULT_CONTROLLER = dict(
    k_linear=1.0, k_angular=2.0, goal_tolerance=0.05,
    lookahead_cells=0.6, dt=0.1,
)


def _require_object(payload: object, name: str) -> dict:
    if not isinstance(payload, dict):
        raise InvalidInputError(f"{name} 必须为 JSON 对象", {"got": type(payload).__name__})
    return payload


def _build_grid(body: dict) -> OccupancyGrid:
    if "map" not in body:
        raise InvalidInputError("缺少 map 字段")
    m = _require_object(body["map"], "map")
    if "cells" not in m:
        raise InvalidInputError("map 缺少 cells 字段")
    resolution = m.get("resolution", 1.0)
    origin = m.get("origin", [0.0, 0.0])
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise InvalidInputError("map.origin 必须为长度 2 的数组 [x,y]")
    return OccupancyGrid.from_raw(
        m["cells"],
        resolution=resolution,
        origin=(float(origin[0]), float(origin[1])),
        width=m.get("width"),
        height=m.get("height"),
    )


def _build_limits(body: dict) -> RobotLimits:
    raw = _DEFAULT_LIMITS.copy()
    if "limits" in body and body["limits"] is not None:
        raw.update(_require_object(body["limits"], "limits"))
    return RobotLimits(
        max_v=float(raw["max_v"]),
        min_v=float(raw["min_v"]),
        max_omega=float(raw["max_omega"]),
        max_curvature=float(raw["max_curvature"]),
        max_acc=float(raw["max_acc"]),
        footprint_radius=float(raw["footprint_radius"]),
    )


def _build_planner(body: dict) -> PlannerParams:
    raw = _DEFAULT_PLANNER.copy()
    if "planner" in body and body["planner"] is not None:
        raw.update(_require_object(body["planner"], "planner"))
    return PlannerParams(
        obstacle_inflation_cells=int(raw["obstacle_inflation_cells"]),
        allow_diagonal=bool(raw["allow_diagonal"]),
        max_expanded_nodes=int(raw["max_expanded_nodes"]),
    )


def _build_controller(body: dict) -> ControllerParams:
    raw = _DEFAULT_CONTROLLER.copy()
    if "controller" in body and body["controller"] is not None:
        raw.update(_require_object(body["controller"], "controller"))
    return ControllerParams(
        k_linear=float(raw["k_linear"]),
        k_angular=float(raw["k_angular"]),
        goal_tolerance=float(raw["goal_tolerance"]),
        lookahead_cells=float(raw["lookahead_cells"]),
        dt=float(raw["dt"]),
    )


def _build_trajectory_limit(body: dict) -> int:
    raw = body.get("trajectory_limit", DEFAULT_TRAJECTORY_LIMIT)
    if isinstance(raw, bool) or not isinstance(raw, int):
        try:
            raw = int(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                "trajectory_limit 必须为整数", {"value": body.get("trajectory_limit")}
            ) from exc
    if raw < 1:
        raise InvalidInputError(
            "trajectory_limit 必须 >= 1", {"value": raw}
        )
    return raw


def _parse_pose(d: object, name: str) -> Pose:
    d = _require_object(d, name)
    try:
        return Pose(float(d["x"]), float(d["y"]), float(d.get("theta", 0.0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} 需要数值字段 x,y[,theta]") from exc


def _parse_point(d: object, name: str) -> Point:
    d = _require_object(d, name)
    try:
        return Point(float(d["x"]), float(d["y"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} 需要数值字段 x,y") from exc


def _stable_body(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class NavigationService:
    """进程内多会话管理器。"""

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._sessions: dict[str, NavigationSession] = {}
        # idem-key -> (body 指纹, 已缓存响应 dict)
        self._idempotent: dict[str, tuple[str, dict]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _get_live(self, session_id: str, now: float) -> NavigationSession:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                # 区分“从未存在/已删除”与“曾存在但过期”
                raise SessionNotFoundError(
                    "会话不存在", {"session_id": session_id}
                )
            if now - sess.updated_at > self.ttl_seconds:
                self._sessions.pop(session_id, None)
                raise SessionExpiredError(
                    "会话已超过空闲 TTL 被回收",
                    {"session_id": session_id, "ttl_seconds": self.ttl_seconds},
                )
            return sess

    # ------------------------------------------------------------------
    def create_session(
        self, payload: dict, idempotency_key: str | None = None
    ) -> dict:
        body = _require_object(payload, "请求体")
        fingerprint = _stable_body(body)
        with self._lock:
            if idempotency_key is not None:
                hit = self._idempotent.get(idempotency_key)
                if hit is not None:
                    fp, cached = hit
                    if fp != fingerprint:
                        raise IdempotencyConflictError(
                            "同一 Idempotency-Key 对应不同请求体",
                            {"idempotency_key": idempotency_key},
                        )
                    return cached

            grid = _build_grid(body)
            limits = _build_limits(body)
            planner_params = _build_planner(body)
            controller_params = _build_controller(body)
            trajectory_limit = _build_trajectory_limit(body)
            now = time.time()
            sid = uuid.uuid4().hex
            sess = NavigationSession(
                sid, grid, limits, planner_params, controller_params, now,
                trajectory_limit=trajectory_limit,
            )
            self._sessions[sid] = sess
            snap = sess.snapshot(now)
            result = {"session_id": sid, "session": _snapshot_dict(snap)}
            if idempotency_key is not None:
                self._idempotent[idempotency_key] = (fingerprint, result)
            return result

    # ------------------------------------------------------------------
    def plan_session(self, session_id: str, payload: dict) -> dict:
        body = _require_object(payload, "请求体")
        now = time.time()
        sess = self._get_live(session_id, now)
        if "start" not in body or "goal" not in body:
            raise InvalidInputError("plan 需要 start 与 goal 字段")
        start = _parse_pose(body["start"], "start")
        goal = _parse_point(body["goal"], "goal")
        with sess.lock:
            path = sess.plan(start, goal, now)
            snap = sess.snapshot(now)
        return {
            "session": _snapshot_dict(snap),
            "path": [[p.x, p.y] for p in path],
        }

    def step_session(
        self,
        session_id: str,
        payload: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        body = payload or {}
        body = _require_object(body, "请求体")
        fingerprint = f"step:{session_id}:{_stable_body(body)}"
        now = time.time()
        sess = self._get_live(session_id, now)

        # 幂等：同键必须返回同一步进结果（step 是状态迁移操作）
        with self._lock:
            if idempotency_key is not None:
                hit = self._idempotent.get(idempotency_key)
                if hit is not None:
                    fp, cached = hit
                    if fp != fingerprint:
                        raise IdempotencyConflictError(
                            "同一 Idempotency-Key 对应不同请求体",
                            {"idempotency_key": idempotency_key},
                        )
                    return cached

        with sess.lock:
            rects = body.get("dynamic_obstacles")
            if rects is not None:
                sess.update_dynamic_obstacles(rects, now)
            result = sess.step(now)
            snap = sess.snapshot(now)
        out = {
            "session": _snapshot_dict(snap),
            "result": _step_dict(result),
        }
        if idempotency_key is not None:
            with self._lock:
                self._idempotent[idempotency_key] = (fingerprint, out)
        return out

    def advance_session(
        self,
        session_id: str,
        payload: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """一次调用内按指定时长连续推进（可带动态障碍与临时速度上限）。"""
        body = payload or {}
        body = _require_object(body, "请求体")
        fingerprint = f"advance:{session_id}:{_stable_body(body)}"
        now = time.time()
        sess = self._get_live(session_id, now)

        with self._lock:
            if idempotency_key is not None:
                hit = self._idempotent.get(idempotency_key)
                if hit is not None:
                    fp, cached = hit
                    if fp != fingerprint:
                        raise IdempotencyConflictError(
                            "同一 Idempotency-Key 对应不同请求体",
                            {"idempotency_key": idempotency_key},
                        )
                    return cached

        if "duration" not in body:
            raise InvalidInputError("advance 需要 duration 字段（模拟秒数）")
        try:
            duration = float(body["duration"])
        except (TypeError, ValueError) as exc:
            raise InvalidInputError("duration 必须为数值（秒）") from exc
        dt = _optional_positive_float(body, "dt")
        max_v = _optional_positive_float(body, "max_v")

        with sess.lock:
            rects = body.get("dynamic_obstacles")
            if rects is not None:
                sess.update_dynamic_obstacles(rects, now)
            result = sess.advance(duration, now, dt=dt, max_v=max_v)
            snap = sess.snapshot(now)
        out = {
            "session": _snapshot_dict(snap),
            "advance": _advance_dict(result),
        }
        if idempotency_key is not None:
            with self._lock:
                self._idempotent[idempotency_key] = (fingerprint, out)
        return out

    def get_trajectory(self, session_id: str) -> dict:
        """查询会话当前保留的执行轨迹（只读，不改变会话状态）。"""
        now = time.time()
        sess = self._get_live(session_id, now)
        with sess.lock:
            data = sess.get_trajectory()
            snap = sess.snapshot(now)
        data["session"] = _snapshot_dict(snap)
        return data

    def rehearse_session(self, session_id: str, payload: dict | None = None) -> dict:
        """只读预演：不改会话状态，不与会话推进互斥。

        route 缺省用会话当前规划路线；start_pose 缺省用当前位姿；
        map 缺省用会话当前有效栅格（含最近一次动态障碍）。
        """
        body = _require_object(payload or {}, "请求体")
        now = time.time()
        sess = self._get_live(session_id, now)
        # 只在会话锁内取一份不可变/独立副本，随后释放锁做计算，
        # 因此预演不阻塞同一会话或其它会话的推进与查询。
        with sess.lock:
            if sess.pose is None or sess.goal is None:
                raise SessionStateError(
                    "会话尚未成功规划，无法预演（请先 plan 或显式提供 route/start）"
                )
            grid = sess.effective_grid
            current_path = list(sess.path)
            current_pose = sess.pose
            limits = sess.limits
            planner_params = sess.planner_params
            controller_params = sess.controller_params
            before = sess.snapshot(now)

        route_raw = body.get("route")
        if route_raw is None:
            route = current_path
        else:
            if not isinstance(route_raw, (list, tuple)) or not route_raw:
                raise InvalidInputError("route 必须为非空数组")
            route = [_parse_point(p, "route") for p in route_raw]

        start_raw = body.get("start")
        start_pose = (
            _parse_pose(start_raw, "start") if start_raw is not None
            else Pose(current_pose.x, current_pose.y, current_pose.theta)
        )
        duration = _optional_positive_float(body, "duration")
        dt = _optional_positive_float(body, "dt")
        max_v = _optional_positive_float(body, "max_v")
        max_microsteps = body.get("max_microsteps", 20000)
        if isinstance(max_microsteps, bool) or not isinstance(max_microsteps, int):
            raise InvalidInputError("max_microsteps 必须为正整数")

        report = rehearse(
            grid, route, start_pose, limits, planner_params, controller_params,
            duration=duration, dt=dt, max_v=max_v,
            max_microsteps=max_microsteps,
        )
        return {
            "report": _report_dict(report),
            "session_observed": _snapshot_dict(before),
            "note": "预演为只读计算，未修改会话状态；session_observed 为预演开始时快照",
        }

    def replan_session(self, session_id: str) -> dict:
        now = time.time()
        sess = self._get_live(session_id, now)
        with sess.lock:
            path = sess.replan(now)
            snap = sess.snapshot(now)
        return {
            "session": _snapshot_dict(snap),
            "path": [[p.x, p.y] for p in path],
        }

    def reset_session(self, session_id: str) -> dict:
        now = time.time()
        sess = self._get_live(session_id, now)
        with sess.lock:
            sess.reset(now)
            snap = sess.snapshot(now)
        return {"session": _snapshot_dict(snap)}

    def get_session(self, session_id: str) -> dict:
        now = time.time()
        sess = self._get_live(session_id, now)
        with sess.lock:
            snap = sess.snapshot(now)
        return {"session": _snapshot_dict(snap)}

    def delete_session(self, session_id: str) -> dict:
        now = time.time()
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess is None:
                raise SessionNotFoundError(
                    "会话不存在", {"session_id": session_id}
                )
        return {"deleted": session_id}

    def purge_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        removed = 0
        with self._lock:
            dead = [
                sid for sid, sess in self._sessions.items()
                if now - sess.updated_at > self.ttl_seconds
            ]
            for sid in dead:
                self._sessions.pop(sid, None)
                removed += 1
        return removed


# ----------------------------------------------------------------------
def _snapshot_dict(snap: SessionSnapshot) -> dict:
    return {
        "session_id": snap.session_id,
        "status": snap.status.value,
        "pose": {"x": snap.pose.x, "y": snap.pose.y, "theta": snap.pose.theta},
        "goal": None if snap.goal is None else {"x": snap.goal.x, "y": snap.goal.y},
        "path": [[p.x, p.y] for p in snap.path],
        "path_index": snap.path_index,
        "step_count": snap.step_count,
        "message": snap.message,
        "last_note": snap.last_note,
        "created_at": snap.created_at,
        "updated_at": snap.updated_at,
    }


def _step_dict(r: StepResult) -> dict:
    return {
        "pose": {"x": r.pose.x, "y": r.pose.y, "theta": r.pose.theta},
        "cmd": {"v": r.cmd.v, "omega": r.cmd.omega},
        "replanned": r.replanned,
        "arrived": r.arrived,
        "blocked": r.blocked,
        "note": r.note,
        "elapsed": r.elapsed,
    }


def _advance_dict(r: AdvanceResult) -> dict:
    return {
        "status": r.status.value,
        "pose": {"x": r.pose.x, "y": r.pose.y, "theta": r.pose.theta},
        "requested_duration": r.requested_duration,
        "simulated_duration": r.simulated_duration,
        "steps_executed": r.steps_executed,
        "completed": r.completed,
        "arrived": r.arrived,
        "blocked": r.blocked,
        "reason": r.reason,
        "results": [_step_dict(x) for x in r.results],
    }


def _report_dict(rep) -> dict:
    return {
        "feasible": rep.feasible,
        "completed": rep.completed,
        "duration": rep.duration,
        "steps": rep.steps,
        "min_clearance": rep.min_clearance,
        "failure_index": rep.failure_index,
        "failure_step": rep.failure_step,
        "reason": rep.reason,
    }


def _optional_positive_float(body: dict, key: str) -> float | None:
    if key not in body or body[key] is None:
        return None
    raw = body[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise InvalidInputError(f"{key} 必须为数值", {"value": raw})
    val = float(raw)
    if not math.isfinite(val) or val <= 0:
        raise InvalidInputError(f"{key} 必须为有限正数", {"value": val})
    return val
