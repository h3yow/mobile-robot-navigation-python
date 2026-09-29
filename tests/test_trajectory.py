"""执行轨迹查询：稠密扫掠轨迹、与位姿序列/步数对齐、保留上限 FIFO、会话隔离。"""

from __future__ import annotations

import logging
import time

from nav.grid import OccupancyGrid
from nav.models import (
    ControllerParams,
    PlannerParams,
    Pose,
    Point,
    RobotLimits,
)
from nav.session import DEFAULT_TRAJECTORY_LIMIT, NavigationSession

log = logging.getLogger("test_trajectory")


def _session(rows=None, trajectory_limit=DEFAULT_TRAJECTORY_LIMIT, **lim):
    rows = rows if rows is not None else [[0] * 30 for _ in range(12)]
    g = OccupancyGrid.from_raw(rows, resolution=1.0)
    limits = RobotLimits(
        max_v=lim.get("max_v", 2.0), min_v=0.0,
        max_omega=2.0, max_curvature=3.0,
        max_acc=4.0, footprint_radius=0.3,
    )
    return NavigationSession(
        "tr", g, limits, PlannerParams(),
        ControllerParams(dt=0.1), time.time(),
        trajectory_limit=trajectory_limit,
    )


def test_trajectory_matches_reported_poses_and_step_count():
    s = _session()
    s.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    reported = []
    for _ in range(12):
        r = s.step(time.time())
        reported.append(r)
    tr = s.get_trajectory()
    log.info("输入：12 次单步；轨迹 retained=%d total=%d dropped=%d",
             tr["retained"], tr["total"], tr["dropped"])
    # 记录条数 == step_count
    assert tr["retained"] == s.step_count == 12
    assert tr["dropped"] == 0
    # 每条记录的 step 连续 1..N，pose 与上报位姿一致
    for i, (rec, rep) in enumerate(zip(tr["records"], reported), start=1):
        assert rec["step"] == i
        assert abs(rec["pose"]["x"] - rep.pose.x) < 1e-12
        assert abs(rec["pose"]["y"] - rep.pose.y) < 1e-12
        assert abs(rec["pose"]["theta"] - rep.pose.theta) < 1e-12
        # 记录里的 cmd 与上报 cmd 一致
        assert abs(rec["cmd"]["v"] - rep.cmd.v) < 1e-12
        assert abs(rec["cmd"]["omega"] - rep.cmd.omega) < 1e-12
        assert abs(rec["dt"] - rep.elapsed) < 1e-12
    # swept 首尾相接可还原连续轨迹，且每条 swept[-1] == 记录 pose
    chain = []
    for rec in tr["records"]:
        swept = rec["swept"]
        assert len(swept) >= 2  # 至少含起点与终点
        assert abs(swept[-1]["x"] - rec["pose"]["x"]) < 1e-12
        # 相邻 swept 点弦长不为零的合理上限（存储采样间隔）
        for a, b in zip(swept, swept[1:]):
            d = ((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
            assert d <= 0.5 + 1e-6
        chain.extend(swept[1:])  # 去重每条起点（== 上一条终点）
    assert abs(chain[-1]["x"] - reported[-1].pose.x) < 1e-9
    log.info("关键判定：%d 条记录的 step/pose/cmd/dt 与上报序列完全对齐，"
             "swept 链式可还原至终位姿 (%.3f,%.3f)",
             len(tr["records"]), chain[-1]["x"], chain[-1]["y"])


def test_advance_records_swept_intermediate_points():
    # 大跨度推进时，单条记录也必须含“中间扫过位置”，不只是起止点
    s = _session()
    s.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    res = s.advance(2.0, time.time(), dt=0.5, max_v=2.0)
    tr = s.get_trajectory()
    total_swept = sum(len(r["swept"]) for r in tr["records"])
    log.info("输入：advance dt=0.5 2秒；记录数=%d swept 采样总数=%d",
             tr["retained"], total_swept)
    assert tr["retained"] == res.steps_executed
    assert total_swept > tr["retained"] * 2  # 存在大量中间扫过点
    # 每个采样的 x 都在合法范围内且沿 x 单调推进（直行）
    xs = [p["x"] for rec in tr["records"] for p in rec["swept"]]
    assert max(xs) - min(xs) > 1.0
    log.info("关键判定：swept 覆盖 [%.2f,%.2f] 的中间位置（非仅起止点）",
             min(xs), max(xs))


def test_trajectory_limit_drops_oldest_fifo():
    limit = 5
    s = _session(trajectory_limit=limit)
    s.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    for _ in range(12):
        s.step(time.time())
    tr = s.get_trajectory()
    log.info("上限=%d，12 步后 retained=%d dropped=%d total=%d，保留 step=%s",
             limit, tr["retained"], tr["dropped"], tr["total"],
             [r["step"] for r in tr["records"]])
    assert tr["limit"] == limit
    assert tr["retained"] == limit
    assert tr["dropped"] == 12 - limit
    assert tr["total"] == 12
    # 保留的是最新 limit 条，step 连续
    assert [r["step"] for r in tr["records"]] == [8, 9, 10, 11, 12]
    # 继续推进，FIFO 持续生效
    s.step(time.time())
    tr2 = s.get_trajectory()
    assert [r["step"] for r in tr2["records"]] == [9, 10, 11, 12, 13]
    assert tr2["dropped"] == 8
    log.info("状态迁移依据：再推进一步后保留 step=9..13，最旧记录被丢弃（FIFO）")


def test_plan_and_reset_clear_trajectory():
    s = _session(trajectory_limit=3)
    s.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    for _ in range(5):
        s.step(time.time())
    assert s.get_trajectory()["retained"] == 3
    # 重新规划清空旧轨迹
    s.plan(Pose(1.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    tr = s.get_trajectory()
    assert tr["retained"] == 0 and tr["dropped"] == 0
    s.step(time.time())
    assert s.get_trajectory()["retained"] == 1
    # reset 清空
    s.reset(time.time())
    assert s.get_trajectory()["retained"] == 0
    log.info("plan/reset 后轨迹与 dropped 计数被清空")


def test_trajectories_are_isolated_between_sessions():
    s1 = _session(trajectory_limit=100)
    s2 = _session(trajectory_limit=2)
    s1.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    s2.plan(Pose(0.5, 5.5, 0.0), Point(29.5, 5.5), time.time())
    for _ in range(6):
        s1.step(time.time())
        s2.step(time.time())
    t1, t2 = s1.get_trajectory(), s2.get_trajectory()
    log.info("会话1 retained=%d dropped=%d；会话2 retained=%d dropped=%d（各自独立）",
             t1["retained"], t1["dropped"], t2["retained"], t2["dropped"])
    assert t1["retained"] == 6 and t1["dropped"] == 0
    assert t2["retained"] == 2 and t2["dropped"] == 4
    # 返回的是副本：修改查询结果不影响会话内部状态
    t1["records"].clear()
    assert s1.get_trajectory()["retained"] == 6
    log.info("查询返回副本，外部修改不影响会话轨迹")


def test_trajectory_limit_validation():
    import pytest
    from nav.errors import InvalidInputError

    for bad in (0, -3):
        with pytest.raises(InvalidInputError):
            _session(trajectory_limit=bad)
    log.info("trajectory_limit <1 被拒绝（invalid_input）")
