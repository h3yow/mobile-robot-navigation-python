"""HTTP 幂等：重复 request_id 返回同一结果；不同请求体冲突。"""

from __future__ import annotations

from tests.conftest import case_log, open_grid


def _sid(client):
    r = client.post("/api/v1/sessions", json={
        "grid": open_grid(20, 12), "resolution": 0.1,
        "start": {"x": 0.15, "y": 0.15}, "goal": {"x": 1.85, "y": 1.05},
        "kinematics": {"robot_radius": 0.0}, "lookahead": 0.25,
    })
    assert r.status_code == 201
    return r.json()["session_id"]


def test_step_request_id_replays_without_double_step(client):
    sid = _sid(client)
    a = client.post(f"/api/v1/sessions/{sid}/step",
                    json={"steps": 5, "request_id": "r-1"}).json()
    b = client.post(f"/api/v1/sessions/{sid}/step",
                    json={"steps": 5, "request_id": "r-1"}).json()
    c = client.post(f"/api/v1/sessions/{sid}/step",
                    json={"steps": 5, "request_id": "r-2"}).json()
    case_log("判定", a=a["step_count"], b=b["step_count"],
             replay=b.get("idempotent_replay"), c=c["step_count"])
    assert a["step_count"] == 5
    assert b["step_count"] == 5 and b["idempotent_replay"] is True
    assert c["step_count"] == 10


def test_same_id_different_body_conflicts(client):
    sid = _sid(client)
    client.post(f"/api/v1/sessions/{sid}/obstacles",
                json={"cells": [[1, 1, 1]], "request_id": "o-1"})
    r = client.post(f"/api/v1/sessions/{sid}/obstacles",
                    json={"cells": [[2, 2, 1]], "request_id": "o-1"})
    case_log("判定", http=r.status_code, code=r.json()["error"]["code"])
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    # 冲突不改变状态：第一次障碍仍在，第二次未写入
    snap = client.get(f"/api/v1/sessions/{sid}").json()
    assert [1, 1] in snap["dynamic_obstacle_cells"]
    assert [2, 2] not in snap["dynamic_obstacle_cells"]


def test_obstacles_toggle_is_idempotent(client):
    sid = _sid(client)
    body = {"cells": [[3, 3, 1], [4, 4, 0]], "reset": False, "request_id": "o-2"}
    a = client.post(f"/api/v1/sessions/{sid}/obstacles", json=body).json()
    b = client.post(f"/api/v1/sessions/{sid}/obstacles", json=body).json()
    case_log("判定", updates_a=a["dynamic_updates"],
             updates_b=b["dynamic_updates"], replay=b.get("idempotent_replay"))
    assert a["dynamic_updates"] == 1
    assert b["dynamic_updates"] == 1 and b["idempotent_replay"] is True
