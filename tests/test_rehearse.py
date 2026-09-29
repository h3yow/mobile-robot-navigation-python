"""只读预演：可复现、不改会话状态、不阻塞其它会话、问题定位与净空、大图性能。"""

from __future__ import annotations

import logging
import threading
import time

from nav.service import NavigationService

log = logging.getLogger("test_rehearse")

MAP = {"cells": [[0] * 40 for _ in range(6)], "resolution": 1.0}
WALL_CELLS = [[1 if y == 2 else 0 for _ in range(40)] for y in range(6)]


def test_rehearse_feasible_route_is_reproducible():
    svc = NavigationService()
    sid = svc.create_session({"map": MAP})["session_id"]
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}}
    )
    rep1 = svc.rehearse_session(sid, {})["report"]
    rep2 = svc.rehearse_session(sid, {})["report"]
    log.info("可行路线：feasible=%s t=%.2f steps=%d min_clearance=%.3f",
             rep1["feasible"], rep1["duration"], rep1["steps"],
             rep1["min_clearance"])
    assert rep1["feasible"] and rep1["completed"]
    assert rep1 == rep2
    # 起点 (0.5,2.5) 距左边界 0.5m，减足迹 0.3 -> 全程最小机器人净空 0.2m
    # （地图边界按墙处理；这是全路线最紧的地方，侧墙中线净空为 2.2m）
    assert abs(rep1["min_clearance"] - 0.2) < 0.05, rep1["min_clearance"]
    log.info("两次预演完全一致；全程最小净空 %.3fm（起点贴左边界所致）",
             rep1["min_clearance"])


def test_rehearse_does_not_modify_session_state():
    svc = NavigationService()
    sid = svc.create_session({"map": MAP})["session_id"]
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}}
    )
    before = svc.get_session(sid)["session"]
    for _ in range(3):
        svc.rehearse_session(sid, {"duration": 5.0, "dt": 0.1, "max_v": 2.0})
    after = svc.get_session(sid)["session"]
    log.info("预演前后 step=%s/%s status=%s/%s",
             before["step_count"], after["step_count"],
             before["status"], after["status"])
    assert before["step_count"] == after["step_count"] == 0
    assert before["status"] == after["status"] == "ready"
    assert before["pose"] == after["pose"]
    assert svc.get_trajectory(sid)["retained"] == 0
    log.info("多次预演后位姿/步数/状态/轨迹均未改变")


def test_rehearse_detects_blocked_route_with_location():
    svc = NavigationService()
    sid = svc.create_session(
        {"map": {"cells": WALL_CELLS, "resolution": 1.0}}
    )["session_id"]
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 0.5}, "goal": {"x": 0.5, "y": 0.5}}
    )
    cross_route = [{"x": 0.5, "y": 0.5}, {"x": 39.5, "y": 5.5}]
    rep = svc.rehearse_session(sid, {"route": cross_route})["report"]
    log.info("跨墙路线：feasible=%s failure_index=%s failure_step=%s "
             "min_clearance=%.3f reason=%s",
             rep["feasible"], rep["failure_index"], rep["failure_step"],
             rep["min_clearance"], rep["reason"])
    assert not rep["feasible"]
    assert rep["failure_index"] is not None
    assert rep["failure_step"] is not None
    # 在墙前安全边界停住：最小净空接近 0（足迹已非常贴近障碍），明显小于开阔值
    assert rep["min_clearance"] < 0.2, rep["min_clearance"]
    assert svc.get_session(sid)["session"]["step_count"] == 0


def test_rehearse_reports_timeout_when_duration_too_short():
    svc = NavigationService()
    sid = svc.create_session({"map": MAP})["session_id"]
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}}
    )
    rep = svc.rehearse_session(sid, {"duration": 2.0})["report"]
    log.info("2 秒预算：feasible=%s reason=%s", rep["feasible"], rep["reason"])
    assert not rep["feasible"] and not rep["completed"]
    assert "预算用尽" in rep["reason"]
    rep_ok = svc.rehearse_session(sid, {"duration": 60.0})["report"]
    assert rep_ok["feasible"]


def test_rehearse_speed_tuning_verdicts_are_stable():
    svc = NavigationService()
    sid = svc.create_session({"map": MAP})["session_id"]
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}}
    )
    verdicts = {}
    for mv in (0.5, 1.0, 2.0):
        r1 = svc.rehearse_session(sid, {"max_v": mv})["report"]
        r2 = svc.rehearse_session(sid, {"max_v": mv})["report"]
        assert (r1["feasible"], r1["steps"], round(r1["duration"], 6)) == (
            r2["feasible"], r2["steps"], round(r2["duration"], 6)
        )
        verdicts[mv] = (r1["feasible"], r1["duration"])
    log.info("速度调参结论：%s", verdicts)
    assert verdicts[2.0][1] < verdicts[1.0][1] < verdicts[0.5][1]


def test_rehearse_does_not_block_other_sessions():
    svc = NavigationService()
    big_n = 400
    big_map = {"cells": [[0] * big_n for _ in range(20)], "resolution": 0.5}
    sid_big = svc.create_session({"map": big_map})["session_id"]
    svc.plan_session(
        sid_big, {"start": {"x": 0.5, "y": 5.0},
                  "goal": {"x": big_n * 0.5 - 0.5, "y": 5.0}}
    )
    sid_small = svc.create_session(
        {"map": MAP, "limits": {"max_v": 5.0, "max_acc": 6.0}}
    )["session_id"]
    svc.plan_session(
        sid_small, {"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}}
    )
    errors = []

    def rehearse_worker():
        try:
            for _ in range(3):
                svc.rehearse_session(sid_big, {"max_microsteps": 2000})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def step_worker():
        for _ in range(400):
            out = svc.step_session(sid_small, {})
            if out["result"]["arrived"]:
                break

    t0 = time.perf_counter()
    t1 = threading.Thread(target=rehearse_worker)
    t2 = threading.Thread(target=step_worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    wall = time.perf_counter() - t0
    final = svc.get_session(sid_small)["session"]
    log.info("并发预演期间小会话 status=%s steps=%d 耗时 %.2fs 错误=%d",
             final["status"], final["step_count"], wall, len(errors))
    assert not errors
    assert final["status"] == "arrived"
    assert svc.get_session(sid_big)["session"]["step_count"] == 0


def test_rehearse_large_map_returns_quickly():
    big_n = 800
    big_map = {"cells": [[0] * big_n for _ in range(60)], "resolution": 0.5}
    svc = NavigationService()
    sid = svc.create_session({"map": big_map})["session_id"]
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 15.0},
              "goal": {"x": big_n * 0.5 - 0.5, "y": 15.0}}
    )
    t0 = time.perf_counter()
    rep = svc.rehearse_session(
        sid, {"dt": 0.5, "max_v": 2.0, "max_microsteps": 800}
    )["report"]
    dt = time.perf_counter() - t0
    log.info("800x60 地图预演：feasible=%s steps=%d 耗时 %.3fs",
             rep["feasible"], rep["steps"], dt)
    assert dt < 10.0


def test_rehearse_invalid_inputs():
    import pytest
    from nav.errors import InvalidInputError, SessionStateError

    svc = NavigationService()
    sid = svc.create_session({"map": MAP})["session_id"]
    with pytest.raises(SessionStateError):
        svc.rehearse_session(sid, {})
    svc.plan_session(
        sid, {"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}}
    )
    for bad in ({"dt": 0}, {"max_v": -1}, {"duration": 0},
                {"route": []}, {"max_microsteps": 0}):
        with pytest.raises(InvalidInputError):
            svc.rehearse_session(sid, bad)
    log.info("预演非法输入均拒绝")
