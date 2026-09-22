"""服务与 HTTP：并发隔离、串行化、幂等、TTL 与可区分错误码。"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from nav.errors import (
    CoordinateOutOfBoundsError,
    GoalInObstacleError,
    IdempotencyConflictError,
    InvalidInputError,
    SessionExpiredError,
    SessionNotFoundError,
    SessionStateError,
    StartInObstacleError,
    NoPathError,
)
from nav.service import NavigationService

log = logging.getLogger("test_service_api")

MAP_8x6 = {"cells": [[0] * 8 for _ in range(6)], "resolution": 1.0}
MAP_20 = {"cells": [[0] * 20 for _ in range(12)], "resolution": 1.0}


def _plan(svc, sid, gx, gy, sx=0.5, sy=0.5):
    return svc.plan_session(
        sid,
        {"start": {"x": sx, "y": sy, "theta": 0.0},
         "goal": {"x": gx, "y": gy}},
    )


# ---------------------------------------------------------------------
# 并发会话隔离
# ---------------------------------------------------------------------
def test_concurrent_sessions_are_isolated(client):
    ids = []
    barrier = threading.Barrier(6)

    def worker(i):
        barrier.wait()
        r = client.post("/sessions", json={"map": MAP_20})
        sid = r.json()["session_id"]
        client.post(
            f"/sessions/{sid}/plan",
            json={"start": {"x": 0.5, "y": 0.5 + i * 0.0},
                  "goal": {"x": 19.5, "y": float(1 + i)}},
        )
        for _ in range(400):
            rr = client.post(f"/sessions/{sid}/step", json={})
            if rr.json()["result"]["arrived"]:
                break
        final = client.get(f"/sessions/{sid}").json()["session"]
        ids.append((sid, final["goal"]["y"], final["pose"]["y"],
                    final["step_count"]))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({sid for sid, *_ in ids}) == 6, "会话 id 必须唯一"
    for sid, gy, py, n in ids:
        log.info("会话 %s goal_y=%.1f 终态 y=%.2f steps=%d", sid[:8], gy, py, n)
        assert abs(py - gy) < 0.1, "各会话必须独立到达各自目标"


def test_concurrent_steps_same_session_serialized(service):
    sid = service.create_session({"map": MAP_20})["session_id"]
    _plan(service, sid, 19.5, 11.5)
    errors = []
    counts = []

    def worker():
        try:
            for _ in range(60):
                out = service.step_session(sid, {})
                counts.append(out["session"]["step_count"])
                if out["result"]["arrived"]:
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
    # 所有观测到的 step_count 必须构成 1..N 的无重复序列（无错序/撕裂/丢步）
    assert sorted(counts) == list(range(1, len(counts) + 1))
    final = service.get_session(sid)["session"]
    log.info("4 线程并发步进同一会话：总步数=%d，状态=%s（无重复无撕裂）",
             final["step_count"], final["status"])


def test_session_step_is_under_lock_against_replan(service):
    sid = service.create_session({"map": MAP_20})["session_id"]
    _plan(service, sid, 19.5, 11.5)
    ok = {"step": 0, "replan": 0}

    def stepper():
        for _ in range(200):
            r = service.step_session(sid, {})
            ok["step"] += 1
            if r["result"]["arrived"]:
                break

    def replanner():
        for _ in range(20):
            time.sleep(0.001)
            try:
                service.replan_session(sid)
                ok["replan"] += 1
            except Exception:  # noqa: BLE001
                pass

    ts = [threading.Thread(target=stepper), threading.Thread(target=replanner)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    snap = service.get_session(sid)["session"]
    log.info("步进与重规划并发：step=%d replan=%d 终态=%s",
             ok["step"], ok["replan"], snap["status"])
    # 并发交错后状态必须合法且自洽：ready（最后一次是 replan）/running/arrived
    assert snap["status"] in ("ready", "running", "arrived")
    assert snap["step_count"] == ok["step"], "步数统计不得因并发丢失或重复"


# ---------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------
def test_idempotent_create_and_step(client):
    body = {"map": MAP_8x6}
    r1 = client.post("/sessions", json=body, headers={"Idempotency-Key": "k1"})
    r2 = client.post("/sessions", json=body, headers={"Idempotency-Key": "k1"})
    assert r1.json()["session_id"] == r2.json()["session_id"]

    sid = r1.json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 0.5}, "goal": {"x": 7.5, "y": 5.5}})
    s1 = client.post(f"/sessions/{sid}/step", json={},
                     headers={"Idempotency-Key": "s1"})
    s2 = client.post(f"/sessions/{sid}/step", json={},
                     headers={"Idempotency-Key": "s1"})
    assert s1.json()["result"] == s2.json()["result"]
    assert s2.json()["session"]["step_count"] == 1
    log.info("幂等：重复创建同 sid，重复步进 step_count 保持 1")

    # 同键不同体 -> 409 可区分
    r3 = client.post("/sessions", json={"map": MAP_20},
                     headers={"Idempotency-Key": "k1"})
    assert r3.status_code == 409
    assert r3.json()["error"]["code"] == "idempotency_conflict"


# ---------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------
def test_expired_session_distinguishable():
    svc = NavigationService(ttl_seconds=0.05)
    sid = svc.create_session({"map": MAP_8x6})["session_id"]
    time.sleep(0.08)
    with pytest.raises(SessionExpiredError) as ei:
        svc.get_session(sid)
    log.info("关键判定：TTL 过期 -> code=%s", ei.value.code)
    assert ei.value.code == "session_expired"
    assert svc.purge_expired() == 0  # 访问时已被回收


def test_delete_then_not_found():
    svc = NavigationService()
    sid = svc.create_session({"map": MAP_8x6})["session_id"]
    svc.delete_session(sid)
    with pytest.raises(SessionNotFoundError) as ei:
        svc.get_session(sid)
    assert ei.value.code == "session_not_found"


# ---------------------------------------------------------------------
# 非法 / 退化输入通过 HTTP 的可区分错误码
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload,expected_code",
    [
        ({"map": {"cells": [[0, 1], [0]], "resolution": 1.0}}, "map_shape_error"),
        ({"map": {"cells": [[0, 1, 2]], "resolution": 1.0}}, "invalid_input"),
        ({"map": {"cells": [[0]], "resolution": -1.0}}, "invalid_input"),
        ({"map": {"cells": [[0]], "resolution": 1.0},
          "limits": {"max_v": -2.0}}, "invalid_input"),
        ({}, "invalid_input"),
    ],
)
def test_invalid_create_payloads(client, payload, expected_code):
    r = client.post("/sessions", json=payload)
    log.info("输入 %s -> HTTP %d code=%s", payload, r.status_code,
             r.json()["error"]["code"])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == expected_code


def test_plan_error_codes_over_http(client):
    sid = client.post("/sessions", json={"map": MAP_8x6}).json()["session_id"]

    def _post_plan(start, goal):
        return client.post(f"/sessions/{sid}/plan",
                           json={"start": start, "goal": {"x": goal[0], "y": goal[1]}})

    r = _post_plan({"x": 100, "y": 0}, (1, 1))
    assert r.json()["error"]["code"] == "coordinate_out_of_bounds"

    r = _post_plan({"x": 0.5, "y": 0.5}, (100, 1))
    assert r.json()["error"]["code"] == "coordinate_out_of_bounds"
    log.info("起终点越界 -> %s", r.json()["error"]["code"])

    # 未规划步进
    sid2 = client.post("/sessions", json={"map": MAP_8x6}).json()["session_id"]
    r = client.post(f"/sessions/{sid2}/step", json={})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "session_state_error"


def test_obstacle_start_goal_codes_via_service():
    svc = NavigationService()
    # 全障碍起点
    sid = svc.create_session(
        {"map": {"cells": [[1] * 4 for _ in range(4)], "resolution": 1.0}}
    )["session_id"]
    with pytest.raises(StartInObstacleError) as ei:
        _plan(svc, sid, 3.5, 3.5)
    assert ei.value.code == "start_in_obstacle"

    # 无路径
    cells = [[0] * 6 for _ in range(6)]
    for x in range(6):
        cells[2][x] = 1
    sid2 = svc.create_session(
        {"map": {"cells": cells, "resolution": 1.0}}
    )["session_id"]
    with pytest.raises(NoPathError) as ei:
        _plan(svc, sid2, 0.5, 4.5, sx=0.5, sy=0.5)
    assert ei.value.code == "goal_unreachable"
    log.info("起障碍=%s 不可达=%s 均可区分",
             "start_in_obstacle", ei.value.code)


def test_origin_and_resolution_variants(client):
    # origin 非零 + 细分辨率：服务端稳定建图并规划
    cells = [[0] * 10 for _ in range(8)]
    r = client.post("/sessions", json={
        "map": {"cells": cells, "resolution": 0.25, "origin": [-1.0, 2.0]}
    })
    assert r.status_code == 200
    sid = r.json()["session_id"]
    # 世界范围 x[-1,1.5) y[2,4)
    r = client.post(f"/sessions/{sid}/plan", json={
        "start": {"x": -0.875, "y": 2.125}, "goal": {"x": 1.375, "y": 3.875}})
    log.info("非零原点/细分辨率规划 -> HTTP %d, 路径点 %d",
             r.status_code, len(r.json().get("path", [])))
    assert r.status_code == 200
