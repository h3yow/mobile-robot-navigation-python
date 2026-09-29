"""新增 HTTP 能力：advance / trajectory / rehearse 的端点接线、错误码与只读语义。"""

from __future__ import annotations

import logging

log = logging.getLogger("test_advance_api")

MAP = {"cells": [[0] * 40 for _ in range(6)], "resolution": 1.0}


def test_advance_and_trajectory_endpoints(client):
    r = client.post("/sessions", json={"map": MAP})
    sid = r.json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}})
    r = client.post(f"/sessions/{sid}/advance",
                    json={"duration": 25.0, "dt": 0.5, "max_v": 2.0})
    assert r.status_code == 200
    adv = r.json()["advance"]
    log.info("POST /advance -> arrived=%s steps=%d sim_t=%.2f",
             adv["arrived"], adv["steps_executed"], adv["simulated_duration"])
    assert adv["arrived"] and "reason" in adv and "results" in adv
    # 轨迹端点
    t = client.get(f"/sessions/{sid}/trajectory").json()
    assert t["retained"] == adv["steps_executed"]
    assert t["limit"] == 2000
    assert len(t["records"][0]["swept"]) >= 2
    log.info("GET /trajectory -> retained=%d 首条 swept=%d 点",
             t["retained"], len(t["records"][0]["swept"]))


def test_advance_idempotent_with_key(client):
    sid = client.post("/sessions", json={"map": MAP}).json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}})
    body = {"duration": 1.0, "dt": 0.5}
    r1 = client.post(f"/sessions/{sid}/advance", json=body,
                     headers={"Idempotency-Key": "a1"})
    n1 = r1.json()["session"]["step_count"]
    r2 = client.post(f"/sessions/{sid}/advance", json=body,
                     headers={"Idempotency-Key": "a1"})
    n2 = r2.json()["session"]["step_count"]
    assert n1 == n2
    assert r1.json()["advance"] == r2.json()["advance"]
    log.info("advance 幂等：重复请求 step_count 保持 %d", n2)
    # 同键不同体 -> 409
    r3 = client.post(f"/sessions/{sid}/advance", json={"duration": 2.0},
                     headers={"Idempotency-Key": "a1"})
    assert r3.status_code == 409
    assert r3.json()["error"]["code"] == "idempotency_conflict"


def test_advance_validation_http(client):
    sid = client.post("/sessions", json={"map": MAP}).json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}})
    for bad in ({}, {"duration": -1}, {"duration": 100}, {"duration": "x"},
                {"duration": 1, "dt": 0}, {"duration": 1, "max_v": 0}):
        r = client.post(f"/sessions/{sid}/advance", json=bad)
        assert r.status_code == 422, (bad, r.status_code)
        assert r.json()["error"]["code"] == "invalid_input"
    log.info("advance 非法输入（缺 duration/负/超上限/非数值/非法 dt/max_v）-> 422")


def test_advance_before_plan_is_state_error(client):
    sid = client.post("/sessions", json={"map": MAP}).json()["session_id"]
    r = client.post(f"/sessions/{sid}/advance", json={"duration": 1.0})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "session_state_error"


def test_rehearse_endpoint_readonly_and_404(client):
    sid = client.post("/sessions", json={"map": MAP}).json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}})
    rep = client.post(f"/sessions/{sid}/rehearse", json={}).json()["report"]
    assert rep["feasible"]
    # 会话状态未被预演改动
    snap = client.get(f"/sessions/{sid}").json()["session"]
    assert snap["step_count"] == 0 and snap["status"] == "ready"
    log.info("POST /rehearse -> feasible=%s；会话 step_count 仍为 0（只读）",
             rep["feasible"])
    # 未规划会话预演 -> 409
    sid2 = client.post("/sessions", json={"map": MAP}).json()["session_id"]
    assert client.post(f"/sessions/{sid2}/rehearse", json={}).status_code == 409
    # 不存在会话 -> 404
    assert client.post("/nope/x/rehearse", json={}).status_code in (404, 422)
    r = client.get("/sessions/does-not-exist/trajectory")
    assert r.status_code == 404


def test_trajectory_limit_create_option(client):
    sid = client.post(
        "/sessions",
        json={"map": MAP, "trajectory_limit": 3},
    ).json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 2.5}, "goal": {"x": 39.5, "y": 2.5}})
    client.post(f"/sessions/{sid}/advance", json={"duration": 3.0})
    t = client.get(f"/sessions/{sid}/trajectory").json()
    log.info("trajectory_limit=3：retained=%d dropped=%d total=%d",
             t["retained"], t["dropped"], t["total"])
    assert t["retained"] == 3 and t["limit"] == 3
    assert t["dropped"] > 0 and t["total"] == t["dropped"] + 3
    # 非法保留上限
    r = client.post("/sessions", json={"map": MAP, "trajectory_limit": 0})
    assert r.status_code == 422
