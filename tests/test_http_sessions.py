"""HTTP 端到端：动态障碍重规划、错误原因区分、失败恢复、大地图。"""

from __future__ import annotations

import time

from tests.conftest import case_log, open_grid


def _create(client, **over):
    body = {
        "grid": open_grid(20, 12),
        "resolution": 0.1,
        "start": {"x": 0.15, "y": 0.15, "theta": 0.0},
        "goal": {"x": 1.85, "y": 1.05, "theta": 0.0},
        "kinematics": {"robot_radius": 0.0, "dt": 0.1,
                       "max_linear_speed": 1.0, "max_angular_speed": 2.0},
        "lookahead": 0.25, "goal_tolerance": 0.06,
    }
    body.update(over)
    r = client.post("/api/v1/sessions", json=body)
    assert r.status_code == 201, r.text
    return r.json()["session_id"], r.json()


def _run_until(client, sid, target=("REACHED", "BLOCKED", "FAILED"), limit=300):
    last = {}
    for _ in range(limit):
        r = client.post(f"/api/v1/sessions/{sid}/step", json={"steps": 1})
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in target:
            break
    return last


def test_create_plan_and_reach(client):
    sid, d = _create(client)
    case_log("判定", sid=sid, status=d["status"], path=d["path_length"])
    assert d["status"] == "ACTIVE"
    last = _run_until(client, sid)
    case_log("状态迁移", final=last["status"], steps=last["step_count"],
             pose=last["pose"])
    assert last["status"] == "REACHED"
    assert last["step_count"] > 0


def test_dynamic_obstacle_forces_single_replan(client):
    sid, _ = _create(client)
    client.post(f"/api/v1/sessions/{sid}/step", json={"steps": 10})
    snap = client.get(f"/api/v1/sessions/{sid}").json()
    px, py = snap["remaining_path"][5]
    col = int((px - 0.0) / 0.1)
    row = 12 - 1 - int((py - 0.0) / 0.1)
    case_log("输入", obstacle_cell=[col, row], at_step=snap["step_count"])
    r = client.post(f"/api/v1/sessions/{sid}/obstacles",
                    json={"cells": [[col, row, 1]]})
    assert r.status_code == 200
    replans_seen = 0
    blocked_seen = 0
    last = None
    for _ in range(200):
        last = client.post(f"/api/v1/sessions/{sid}/step",
                           json={"steps": 1}).json()
        ls = last["last_step"] or {}
        if ls.get("replanned"):
            replans_seen += 1
            case_log("状态迁移", step=last["step_count"],
                     reason=ls.get("replan_reason"),
                     event=ls.get("events"))
        if last["status"] == "BLOCKED":
            blocked_seen += 1
        if last["status"] == "REACHED":
            break
    case_log("判定", replans=replans_seen, blocked=blocked_seen,
             final=last["status"])
    assert last["status"] == "REACHED"
    assert replans_seen == 1  # 一次障碍恰好触发一次重规划


def test_sealed_goal_rejected_with_reason(client):
    grid = [
        "........", "..####..", "..#..#..", "..#..#..",
        "..#..#..", "..####..", "........",
    ]
    r = client.post("/api/v1/sessions", json={
        "grid": grid, "resolution": 0.1,
        "start": {"x": 0.05, "y": 0.05},
        "goal": {"x": 0.35, "y": 0.35},
        "kinematics": {"robot_radius": 0.0},
    })
    case_log("判定", status=r.status_code, code=r.json()["error"]["code"],
             message=r.json()["error"]["message"])
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "GOAL_UNREACHABLE"


def test_start_goal_errors_are_distinct(client):
    grid = ["....", ".##.", ".##.", "...."]
    base = {"grid": grid, "resolution": 0.1,
            "kinematics": {"robot_radius": 0.0}}
    cases = [
        ({"start": {"x": -0.1, "y": 0.2}, "goal": {"x": 0.35, "y": 0.35}},
         422, "COORDINATE_OUT_OF_BOUNDS"),
        ({"start": {"x": 0.15, "y": 0.25}, "goal": {"x": 0.35, "y": 0.35}},
         422, "START_IN_OBSTACLE"),
        ({"start": {"x": 0.05, "y": 0.05}, "goal": {"x": 0.15, "y": 0.25}},
         422, "GOAL_IN_OBSTACLE"),
    ]
    for extra, http_code, code in cases:
        r = client.post("/api/v1/sessions", json={**base, **extra})
        case_log("判定", expect=code, got=r.json()["error"]["code"],
                 http=r.status_code)
        assert r.status_code == http_code
        assert r.json()["error"]["code"] == code


def test_degenerate_maps_return_distinct_codes(client):
    cases = [
        ({"grid": []}, "MAP_EMPTY"),
        ({"grid": ["###", "###"]}, "MAP_NO_FREE_SPACE"),
        ({"occupancy": [[0, 0], [0]]}, "MAP_DIMENSION_MISMATCH"),
        ({"occupancy": [[0, 2]]}, "MAP_INVALID_CELL_VALUE"),
        ({"grid": [".."], "resolution": -1}, "MAP_INVALID_RESOLUTION"),
    ]
    for body, code in cases:
        r = client.post("/api/v1/sessions", json=body)
        case_log("判定", input=body, got=r.json()["error"]["code"])
        assert r.json()["error"]["code"] == code


def test_step_after_finish_is_rejected(client):
    sid, _ = _create(client)
    last = _run_until(client, sid)
    assert last["status"] == "REACHED"
    r = client.post(f"/api/v1/sessions/{sid}/step", json={"steps": 1})
    case_log("判定", http=r.status_code, code=r.json()["error"]["code"])
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "SESSION_ALREADY_FINISHED"


def test_missing_session_404(client):
    r = client.get("/api/v1/sessions/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_blocked_then_clear_obstacle_recovers(client):
    sid, _ = _create(client)
    # 构造不可绕行：在机器人正前方近距离放一道横贯墙，先让它 BLOCKED
    client.post(f"/api/v1/sessions/{sid}/step", json={"steps": 4})
    snap = client.get(f"/api/v1/sessions/{sid}").json()
    # 在路径前方远处横放一排（留出起点不碰撞）
    cells = []
    px, py = snap["remaining_path"][8]
    row = 12 - 1 - int(py / 0.1)
    for col in range(20):
        cells.append([col, row, 1])
    client.post(f"/api/v1/sessions/{sid}/obstacles", json={"cells": cells})
    last = _run_until(client, sid, target=("BLOCKED", "FAILED"), limit=40)
    case_log("状态迁移", after_wall=last["status"],
             replan_error=last.get("replan_error", {}).get("code"))
    assert last["status"] == "BLOCKED"
    assert last["replan_error"]["code"] in ("GOAL_UNREACHABLE",
                                             "PLAN_TIME_BUDGET_EXCEEDED")
    # 清除障碍 -> 可继续
    client.post(f"/api/v1/sessions/{sid}/obstacles", json={"reset": True})
    last = _run_until(client, sid)
    case_log("状态迁移", after_clear=last["status"],
             steps=last["step_count"])
    assert last["status"] == "REACHED"


def test_large_map_single_step_is_fast(client):
    grid = ["." * 300 for _ in range(300)]
    t0 = time.time()
    r = client.post("/api/v1/sessions", json={
        "grid": grid, "resolution": 0.1,
        "start": {"x": 0.15, "y": 0.15},
        "goal": {"x": 29.85, "y": 29.85},
        "kinematics": {"robot_radius": 0.0},
        "planner_time_budget": 5.0,
    })
    create_ms = (time.time() - t0) * 1000
    assert r.status_code == 201, r.text
    sid = r.json()["session_id"]
    t0 = time.time()
    rs = client.post(f"/api/v1/sessions/{sid}/step", json={"steps": 1})
    step_ms = (time.time() - t0) * 1000
    case_log("判定", create_ms=round(create_ms), step_ms=round(step_ms),
             status=rs.json()["status"])
    assert rs.status_code == 200
    assert step_ms < 200


def test_occupancy_array_forms_and_origin_resolution(client):
    """二维数组、一维数组 + width/height、非零 origin 都可建会话并规划。"""
    grid = [[0] * 8 for _ in range(6)]
    r1 = client.post("/api/v1/sessions", json={
        "occupancy": grid, "resolution": 0.25, "origin": [1.0, -0.5],
        "start": {"x": 1.125, "y": -0.375},
        "goal": {"x": 2.625, "y": 0.625},
        "kinematics": {"robot_radius": 0.0},
    })
    case_log("判定", form="2d+origin", status=r1.status_code)
    assert r1.status_code == 201 and r1.json()["status"] == "ACTIVE"

    flat = [v for row in grid for v in row]
    r2 = client.post("/api/v1/sessions", json={
        "occupancy": flat, "width": 8, "height": 6, "resolution": 0.25,
        "start": {"x": 0.125, "y": 0.125},
        "goal": {"x": 1.625, "y": 1.125},
        "kinematics": {"robot_radius": 0.0},
    })
    case_log("判定", form="flat+width/height", status=r2.status_code,
             path=r2.json().get("path_length"))
    assert r2.status_code == 201 and r2.json()["path_length"] >= 7


def test_replan_after_obstacle_makes_way_when_cleared_midblock(client):
    """障碍先阻挡（封墙）→ BLOCKED；移除后再步进可恢复到达。"""
    sid, _ = _create(client)
    client.post(f"/api/v1/sessions/{sid}/step", json={"steps": 3})
    snap = client.get(f"/api/v1/sessions/{sid}").json()
    px, py = snap["remaining_path"][8]
    row = 12 - 1 - int(py / 0.1)
    client.post(f"/api/v1/sessions/{sid}/obstacles",
                json={"cells": [[c, row, 1] for c in range(20)]})
    blocked = _run_until(client, sid, target=("BLOCKED", "FAILED"), limit=30)
    assert blocked["status"] == "BLOCKED"
    client.post(f"/api/v1/sessions/{sid}/obstacles", json={"reset": True})
    last = _run_until(client, sid)
    case_log("状态迁移", final=last["status"], steps=last["step_count"])
    assert last["status"] == "REACHED"
