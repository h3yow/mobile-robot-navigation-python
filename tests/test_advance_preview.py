"""按时长推进 / 执行轨迹查询 / 只读预演的行为测试。

覆盖：
  * 一次大跨度推进在窄通道、1 格薄隔墙下不穿障、不越界、不超约束；
  * 部分推进（中途被挡能走多少走多少，原地停住，会话可继续）；
  * 执行轨迹含中间扫过位姿，与上报位姿序列/步数一一对应；
  * 超容量 FIFO 丢弃语义；
  * 预演可复现、不改会话状态、能预判失败位置与最近间隙，且不阻塞并发推进；
  * 非法输入与老调用方式兼容。
"""

from __future__ import annotations

import logging
import math
import threading
import time

import pytest

from nav.errors import InvalidInputError, SessionStateError
from nav.grid import OccupancyGrid
from nav.models import (
    ControllerParams,
    PlannerParams,
    Point,
    Pose,
    RobotLimits,
    SessionStatus,
)
from nav.service import NavigationService
from nav.session import NavigationSession

log = logging.getLogger("test_advance_preview")

MAP20 = {"cells": [[0] * 20 for _ in range(12)], "resolution": 1.0}


def _session(rows, resolution=1.0, origin=(0.0, 0.0), capacity=2000, **lim_kw):
    g = OccupancyGrid.from_raw(rows, resolution=resolution, origin=origin)
    lim = RobotLimits(
        max_v=lim_kw.get("max_v", 1.0), min_v=0.0,
        max_omega=lim_kw.get("max_omega", 2.0),
        max_curvature=lim_kw.get("max_curvature", 3.0),
        max_acc=lim_kw.get("max_acc", 2.0),
        footprint_radius=lim_kw.get("footprint_radius", 0.2),
    )
    return NavigationSession(
        "adv", g, lim,
        PlannerParams(obstacle_inflation_cells=0),
        ControllerParams(dt=0.1),
        time.time(),
        trajectory_capacity=capacity,
    )


def _assert_segment_safe(s, p1, p2, label):
    """相邻上报位姿之间的胶囊扫掠不得触障/越界。"""
    assert not s.effective_grid.segment_blocked(
        p1.x, p1.y, p2.x, p2.y, s.limits.footprint_radius
    ), f"{label}: {p1} -> {p2} 扫掠触障/越界"


# ---------------------------------------------------------------------
# 1) 大跨度推进：窄通道
# ---------------------------------------------------------------------
def test_advance_long_span_through_narrow_corridor_no_tunneling():
    # 15x15 S 形窄通道：墙仅 1 格厚，缺口 3 格（足迹膨胀 1 格后净宽 1 格）。
    # 横墙 y=9 挡 x<9（缺口在右上），横墙 y=5 挡 x>=6（缺口在左下），
    # 规划路径必须两次穿过 1 格薄壁上的窄缺口。
    n = 15
    rows = [[0] * n for _ in range(n)]
    for x in range(0, 9):
        rows[9][x] = 1
    for x in range(6, n):
        rows[5][x] = 1
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(2.5, 2.5, 0.0), Point(12.5, 12.5), time.time())
    log.info("窄通道路径顶点：%s",
             [(round(p.x, 1), round(p.y, 1)) for p in s.path])

    # 一次请求 30 秒、速度上限 5 m/s：整段位移远超墙厚（1m）与缺口宽度
    r = s.advance(time.time(), 30.0, max_v=5.0)
    log.info(
        "窄通道大跨度推进输入：duration=30s max_v=5；执行=%d/%d elapsed=%.2f "
        "arrived=%s completed=%s",
        r.steps_executed, r.steps_requested, r.elapsed, r.arrived, r.completed,
    )
    assert r.arrived and r.completed, r.note
    poses = r.poses
    # 所有相邻上报位姿（含跨步长距离）之间的扫掠都不穿障、不越界
    for i in range(1, len(poses)):
        _assert_segment_safe(s, poses[i - 1], poses[i], f"corridor tick {i}")
        seg_len = math.hypot(
            poses[i].x - poses[i - 1].x, poses[i].y - poses[i - 1].y
        )
        assert seg_len <= 5.0 * r.dt + 1e-9
    # 穿越第一道墙（y=5 段，x<6 缺口）时必须待在缺口一侧，不得压到 x>=6 的墙
    crossing1 = [p for p in poses if 4.6 < p.y < 5.4]
    for p in crossing1:
        assert p.x < 5.8, f"穿 y=5 缺口时压墙 x={p.x}"
    # 穿越第二道墙（y=9 段，x>=9 缺口）
    crossing2 = [p for p in poses if 8.6 < p.y < 9.4]
    for p in crossing2:
        assert p.x > 8.2, f"穿 y=9 缺口时压墙 x={p.x}"
    log.info("关键判定：%d 个位姿穿过两处 1 格薄壁缺口，扫掠零碰撞", len(poses))


# ---------------------------------------------------------------------
# 2) 薄隔墙：单次位移比墙厚还大也不得穿墙
# ---------------------------------------------------------------------
def test_advance_cannot_jump_over_thin_wall():
    # 10 列 x 6 行：x=4 为仅 1 格厚的整墙，机器人在墙左侧
    rows = [[0] * 10 for _ in range(6)]
    for y in range(6):
        rows[y][4] = 1
    s = _session(rows, footprint_radius=0.2)
    # 目标在墙右侧 -> 不可达，但 planner 会先拒绝；改为手动设置“穿过墙”的路径：
    # 通过预演验证纯跟踪若沿穿墙路线走会被 collision 拦下。
    s.plan(Pose(0.5, 2.5, 0.0), Point(0.5, 2.5), time.time())  # 同点占位规划
    ghost_path = [Point(0.5, 2.5), Point(9.5, 2.5)]
    pv = s.preview(ghost_path, dt=0.5, max_v=5.0, max_ticks=100,
                   start=Pose(0.5, 2.5, 0.0))
    log.info("薄隔墙预演输入 dt=0.5 max_v=5（单步位移 2.5m >> 墙厚 1m）")
    assert not pv.feasible
    assert pv.fail_reason == "collision", pv.note
    log.info("关键判定：预演在路线第 %s 点附近报 collision，最近间隙 %.3f，note=%s",
             pv.fail_index, pv.min_clearance, pv.note)
    # 会话状态未被预演改动
    assert s.status == SessionStatus.READY
    assert s.pose.x == 0.5


def test_advance_blocked_by_dynamic_thin_wall_partial_then_stops():
    rows = [[0] * 20 for _ in range(8)]
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(0.5, 2.5, 0.0), Point(19.5, 2.5), time.time())
    # x=12 处 1 格厚竖墙封死整个通道高度（世界 y∈[0,8]），
    # 机器人可一路推进到墙前，但无法通过、无法绕行：能走多少走多少。
    s.update_dynamic_obstacles([[12.0, 0.0, 13.0, 8.0]], time.time())
    r = s.advance(time.time(), 10.0, max_v=5.0)
    log.info("封死后推进输入 duration=10s：执行=%d elapsed=%.2f blocked=%s arrived=%s "
             "最远 x=%.3f",
             r.steps_executed, r.elapsed, r.blocked, r.arrived, s.pose.x)
    assert r.blocked and not r.arrived
    # 能走多少走多少：必须实际推进过一段，但最远位姿不得贴进/穿过 1m 薄墙
    assert r.steps_executed > 1
    assert s.pose.x > 0.6
    assert s.pose.x < 12.0 - s.limits.footprint_radius
    # 已走轨迹（含最后一个零速停住步）全部安全
    for i in range(1, len(r.poses)):
        _assert_segment_safe(s, r.poses[i - 1], r.poses[i], f"blocked tick {i}")
    progress_x = s.pose.x
    # 停住后再来一次推进仍然原地 blocked，会话不卡死
    r2 = s.advance(time.time(), 2.0, max_v=5.0)
    assert r2.blocked
    assert abs(s.pose.x - progress_x) < 1e-9
    log.info("状态迁移依据：走到 x=%.3f 后 RUNNING->BLOCKED，重复推进原地停住",
             progress_x)
    # 环境恢复：撤掉墙后自动重规划，能继续走并到达
    s.update_dynamic_obstacles([], time.time())
    r3 = s.advance(time.time(), 20.0, max_v=5.0)
    assert r3.arrived, r3.note
    log.info("撤障后继续推进到达：%s", r3.note)


# ---------------------------------------------------------------------
# 3) 轨迹查询：含中间位姿、与上报序列/步数一致
# ---------------------------------------------------------------------
def test_trajectory_records_swept_poses_and_matches_reports():
    rows = [[0] * 16 for _ in range(10)]
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(0.5, 0.5, 0.3), Point(15.5, 9.5), time.time())
    r1 = s.advance(time.time(), 5.0, max_v=2.0)
    tr1 = s.get_trajectory(time.time())
    assert tr1["total"] == len(r1.samples) == r1.steps_executed
    # 轨迹样本与本次 advance 上报位姿逐一对齐
    for smp, pose, sample in zip(tr1["samples"], r1.poses, r1.samples):
        assert smp["index"] == sample.index
        assert (smp["x"], smp["y"]) == (pose.x, pose.y)
        assert smp["t"] == pytest.approx(sample.t)
    # 步号连续、仿真时间单调
    idxs = [smp["index"] for smp in tr1["samples"]]
    assert idxs == list(range(1, len(idxs) + 1))
    ts = [smp["t"] for smp in tr1["samples"]]
    assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))
    log.info("第一段推进 %d 步，轨迹索引 1..%d、仿真时间 %.1f..%.1f 对齐",
             len(idxs), idxs[-1], ts[0], ts[-1])

    # 继续推进：轨迹接续；再混一次老式单步
    r2 = s.advance(time.time(), 5.0, max_v=2.0)
    one = s.step(time.time())
    tr2 = s.get_trajectory(time.time())
    assert tr2["total"] == r1.steps_executed + r2.steps_executed + 1
    assert tr2["samples"][-1]["x"] == pytest.approx(one.pose.x)
    assert tr2["length"] == tr2["total"]
    # 全程轨迹本身也无穿障
    for i in range(1, len(tr2["samples"])):
        a, b = tr2["samples"][i - 1], tr2["samples"][i]
        assert not s.effective_grid.segment_blocked(
            a["x"], a["y"], b["x"], b["y"], s.limits.footprint_radius
        )
    log.info("关键判定：advance+step 混跑后轨迹共 %d 条，全部与上报位姿一致",
             tr2["total"])


# ---------------------------------------------------------------------
# 4) 保留上限：FIFO 丢弃与统计
# ---------------------------------------------------------------------
def test_trajectory_capacity_fifo_eviction():
    rows = [[0] * 40 for _ in range(4)]
    s = _session(rows, capacity=50, footprint_radius=0.2)
    s.plan(Pose(0.5, 1.5, 0.0), Point(39.5, 1.5), time.time())
    r = s.advance(time.time(), 30.0, max_v=2.0)
    assert r.steps_executed > 50, "必须产生超过容量的步数"
    tr = s.get_trajectory(time.time())
    log.info("容量输入 capacity=50，实际产生 total=%d，保留 length=%d，dropped=%d",
             tr["total"], tr["length"], tr["dropped"])
    assert tr["length"] == 50
    assert tr["dropped"] == tr["total"] - 50
    assert tr["truncated"] is True
    # 保留的是最近 50 条：索引连续且等于最后产生的一段
    idxs = [smp["index"] for smp in tr["samples"]]
    assert idxs == list(range(tr["total"] - 49, tr["total"] + 1))
    assert tr["sim_time"] == pytest.approx(tr["total"] * 0.1)
    # 快照里也能看到相同统计
    snap = s.snapshot(time.time())
    assert snap.trajectory_dropped == tr["dropped"]
    assert snap.trajectory_truncated is True
    log.info("状态迁移依据：超限后 oldest-first 丢弃，窗口为索引 %d..%d",
             idxs[0], idxs[-1])


def test_trajectory_cleared_on_replan_and_reset():
    rows = [[0] * 8 for _ in range(6)]
    s = _session(rows)
    s.plan(Pose(0.5, 0.5, 0.0), Point(7.5, 5.5), time.time())
    s.advance(time.time(), 2.0)
    assert s.get_trajectory(time.time())["total"] > 0
    s.reset(time.time())
    tr = s.get_trajectory(time.time())
    assert tr["total"] == 0 and tr["length"] == 0 and tr["sim_time"] == 0.0
    log.info("reset 后轨迹与 sim_time 清空（容量保持 %d）", tr["capacity"])


# ---------------------------------------------------------------------
# 5) 预演：可复现 / 不改状态 / 失败定位与间隙
# ---------------------------------------------------------------------
def test_preview_reproducible_and_state_invariant():
    svc = NavigationService()
    sid = svc.create_session({"map": MAP20})["session_id"]
    svc.plan_session(sid, {"start": {"x": 0.5, "y": 0.5, "theta": 0.0},
                           "goal": {"x": 19.5, "y": 11.5}})
    before = svc.get_session(sid)["session"]
    outs = []
    for _ in range(3):
        out = svc.preview_session(sid, {"dt": 0.1, "max_v": 1.5})["preview"]
        outs.append(out)
    a, b, c = outs
    assert a["feasible"] == b["feasible"] == c["feasible"]
    assert a["ticks_simulated"] == b["ticks_simulated"] == c["ticks_simulated"]
    assert a["min_clearance"] == b["min_clearance"] == c["min_clearance"]
    assert a["fail_reason"] == b["fail_reason"] == c["fail_reason"]
    log.info("预演可复现：feasible=%s ticks=%d 最近间隙=%.3f reason=%s",
             a["feasible"], a["ticks_simulated"], a["min_clearance"], a["fail_reason"])
    after = svc.get_session(sid)["session"]
    # updated_at 会被查询本身 touch（既有行为），其余字段必须逐字段不变
    assert after["updated_at"] >= before["updated_at"]
    assert {k: v for k, v in before.items() if k != "updated_at"} == {
        k: v for k, v in after.items() if k != "updated_at"
    }
    assert after["step_count"] == 0 and after["trajectory"]["total"] == 0
    log.info("关键判定：三次预演后 step_count/trajectory/位姿均无变化")


def test_preview_detects_blocking_and_reports_index_and_clearance():
    # 墙封死：x=6 整墙，目标在右侧，预演必须给出 collision 之外的失败原因/位置
    rows = [[0] * 14 for _ in range(8)]
    for y in range(8):
        rows[y][6] = 1
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(0.5, 3.5, 0.0), Point(0.5, 3.5), time.time())
    pv = s.preview([Point(0.5, 3.5), Point(13.5, 3.5)],
                   dt=0.2, max_v=3.0, max_ticks=500,
                   start=Pose(0.5, 3.5, 0.0))
    log.info("预演输入 路线穿 x=6 整墙：feasible=%s reason=%s fail_index=%s "
             "最近间隙=%.3f ticks=%d", pv.feasible, pv.fail_reason,
             pv.fail_index, pv.min_clearance, pv.ticks_simulated)
    assert not pv.feasible
    assert pv.fail_reason == "collision"
    # fail_pose 是“被拒绝的预测位姿”：它已经落到墙体/越墙一侧，
    # 说明扫掠胶囊在该微步检出穿障（执行期同一位移会被拒绝，机器人不会真的过去）
    assert pv.fail_pose.x >= 5.0
    assert pv.min_clearance == pytest.approx(0.0, abs=0.21)
    log.info("关键判定：第 %d 微步预测位姿 x=%.2f 已越墙，被扫掠复核拒绝",
             pv.ticks_simulated, pv.fail_pose.x)

    # 同样参数对安全路线应 feasible，且给出有限最近间隙
    rows2 = [[0] * 14 for _ in range(8)]
    s2 = _session(rows2, footprint_radius=0.2)
    s2.plan(Pose(0.5, 3.5, 0.0), Point(0.5, 3.5), time.time())
    pv2 = s2.preview([Point(0.5, 3.5), Point(13.5, 3.5)],
                     dt=0.2, max_v=3.0, max_ticks=500,
                     start=Pose(0.5, 3.5, 0.0))
    assert pv2.feasible
    assert pv2.min_clearance > 0.0
    log.info("安全路线预演通过：最近间隙 %.3f（机器人外缘到墙/边界）",
             pv2.min_clearance)


def test_preview_step_budget_flags_slow_params():
    rows = [[0] * 30 for _ in range(4)]
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(0.5, 1.5, 0.0), Point(0.5, 1.5), time.time())
    pv = s.preview([Point(0.5, 1.5), Point(29.5, 1.5)],
                   dt=0.1, max_v=0.5, max_ticks=100,
                   start=Pose(0.5, 1.5, 0.0))
    assert not pv.feasible and pv.fail_reason == "step_budget"
    log.info("参数过慢/预算过小 -> reason=%s ticks=%d", pv.fail_reason,
             pv.ticks_simulated)


def test_preview_does_not_block_other_sessions_advance():
    svc = NavigationService()
    sid_a = svc.create_session({"map": MAP20})["session_id"]
    svc.plan_session(sid_a, {"start": {"x": 0.5, "y": 0.5, "theta": 0.0},
                             "goal": {"x": 19.5, "y": 11.5}})
    sid_b = svc.create_session({"map": MAP20})["session_id"]
    svc.plan_session(sid_b, {"start": {"x": 0.5, "y": 11.5, "theta": 0.0},
                             "goal": {"x": 19.5, "y": 0.5}})
    errors = []
    stop = {"go": True}

    def previewer():
        # 大图预算的预演反复跑（锁外仿真），期间另一个会话必须持续推进
        try:
            while stop["go"]:
                svc.preview_session(sid_a, {"dt": 0.1, "max_v": 1.0,
                                            "max_ticks": 2000})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=previewer)
    t.start()
    try:
        r = svc.advance_session(sid_b, {"duration": 30.0, "max_v": 3.0})
    finally:
        stop["go"] = False
        t.join()
    assert not errors, errors
    assert r["advance"]["arrived"]
    log.info("预演线程并发期间另一会话 advance 正常到达 steps=%d",
             r["advance"]["steps_executed"])


# ---------------------------------------------------------------------
# 8) 大 dt / 高速度：跨 tick 约束与穿障安全
# ---------------------------------------------------------------------
def test_advance_large_dt_and_high_speed_respects_constraints():
    n = 15
    rows = [[0] * n for _ in range(n)]
    for x in range(0, 9):
        rows[9][x] = 1
    for x in range(6, n):
        rows[5][x] = 1
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(2.5, 2.5, 0.0), Point(12.5, 12.5), time.time())
    # 大微步 dt=0.5 + 高速 5：单步位移可达 2.5m（>1m 墙厚）
    r = s.advance(time.time(), 30.0, dt=0.5, max_v=5.0)
    log.info("大 dt 推进：dt=0.5 max_v=5 -> ticks=%d arrived=%s",
             r.steps_executed, r.arrived)
    assert r.arrived
    prev_v = 0.0
    for smp in r.samples:
        assert 0.0 - 1e-9 <= smp.v <= 5.0 + 1e-9
        assert abs(smp.omega) <= s.limits.max_omega + 1e-9
        if abs(smp.v) > 1e-6:
            assert abs(smp.omega / smp.v) <= s.limits.max_curvature + 1e-9
        assert abs(smp.v - prev_v) <= s.limits.max_acc * r.dt + 1e-9
        prev_v = smp.v
    # 所有相邻上报位姿（大跨度）之间扫掠零穿障
    for i in range(1, len(r.poses)):
        _assert_segment_safe(s, r.poses[i - 1], r.poses[i], f"bigdt {i}")
    log.info("关键判定：%d 个大跨度微步全程满足 v/ω/曲率/加速度且不穿障",
             r.steps_executed)


def test_high_speed_does_not_overshoot_goal_near_boundary():
    # 终点贴东边界（19.5 + 足迹 0.3 = 19.8，仅 0.2m 余量）：
    # 高速推进必须末端制动，在终点容差内停住，不得冲过终点越界被 BLOCKED。
    rows = [[0] * 20 for _ in range(8)]
    s = _session(rows, footprint_radius=0.3)
    s.plan(Pose(0.5, 2.5, 0.0), Point(19.5, 2.5), time.time())
    r = s.advance(time.time(), 20.0, max_v=5.0)
    log.info("高速末端制动：arrived=%s 终点 x=%.3f ticks=%d",
             r.arrived, s.pose.x, r.steps_executed)
    assert r.arrived and not r.blocked
    assert s.pose.x <= 19.5 + 0.05


def test_preview_matches_execution_tick_for_tick():
    rows = [[0] * 20 for _ in range(12)]
    s = _session(rows, footprint_radius=0.3)
    s.plan(Pose(0.5, 0.5, 0.0), Point(19.5, 11.5), time.time())
    pv = s.preview(None, dt=0.1, max_v=2.0, max_ticks=5000)
    r = s.advance(time.time(), 60.0, max_v=2.0)
    log.info("预演 ticks=%d feasible=%s；执行 ticks=%d arrived=%s",
             pv.ticks_simulated, pv.feasible, r.steps_executed, r.arrived)
    assert pv.feasible == r.arrived
    assert pv.ticks_simulated == r.steps_executed


def test_trajectory_contains_dense_intermediate_swept_poses():
    # 长距离直行：轨迹必须包含比“起/止”多得多的中间扫过位姿
    rows = [[0] * 30 for _ in range(4)]
    s = _session(rows, footprint_radius=0.2)
    s.plan(Pose(0.5, 1.5, 0.0), Point(29.5, 1.5), time.time())
    r = s.advance(time.time(), 30.0, max_v=2.0)
    assert r.arrived and r.steps_executed > 50
    tr = s.get_trajectory(time.time())
    xs = [smp["x"] for smp in tr["samples"]]
    assert xs[0] < 1.0 and xs[-1] > 29.0
    # 相邻样本间距都不超过一个微步的最大位移（中间位姿确实被记录，不是抽样首尾）
    gaps = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    assert max(gaps) <= 2.0 * 0.1 + 1e-9
    assert len(set(round(x, 6) for x in xs)) == len(xs)  # 无重复填充
    log.info("中间扫过位姿 %d 个，x 从 %.2f 连续到 %.2f，最大间距 %.3f",
             len(xs), xs[0], xs[-1], max(gaps))


# ---------------------------------------------------------------------
# 6) 非法输入
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"duration": 0},
        {"duration": -1.0},
        {"duration": "3"},
        {"duration": 1000.0},
        {"duration": 1.0, "dt": 0},
        {"duration": 1.0, "max_v": -2},
        {"duration": float("nan")},
    ],
)
def test_advance_invalid_inputs(service, payload):
    sid = service.create_session({"map": MAP20})["session_id"]
    service.plan_session(sid, {"start": {"x": 0.5, "y": 5.5},
                               "goal": {"x": 19.5, "y": 5.5}})
    with pytest.raises(InvalidInputError) as ei:
        service.advance_session(sid, payload)
    log.info("输入 %s -> code=%s", payload, ei.value.code)
    assert ei.value.code == "invalid_input"


def test_advance_before_plan_rejected(service):
    sid = service.create_session({"map": MAP20})["session_id"]
    with pytest.raises(SessionStateError) as ei:
        service.advance_session(sid, {"duration": 1.0})
    assert ei.value.code == "session_state_error"
    log.info("未规划即 advance -> %s", ei.value.code)


def test_preview_invalid_args(service):
    sid = service.create_session({"map": MAP20})["session_id"]
    service.plan_session(sid, {"start": {"x": 0.5, "y": 5.5},
                               "goal": {"x": 19.5, "y": 5.5}})
    with pytest.raises(InvalidInputError):
        service.preview_session(sid, {"max_ticks": 0})
    with pytest.raises(InvalidInputError):
        service.preview_session(sid, {"path": []})
    with pytest.raises(InvalidInputError):
        service.preview_session(sid, {"path": [{"x": 1, "y": 1}]})  # 缺 start
    log.info("预演非法 max_ticks/空 path/缺 start 均被 invalid_input 拒绝")


def test_trajectory_capacity_creation_bounds(client):
    r = client.post("/sessions", json={"map": MAP20, "trajectory_capacity": 0})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
    r = client.post("/sessions", json={"map": MAP20, "trajectory_capacity": 10 ** 9})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_input"
    log.info("trajectory_capacity 越界（0 / 1e9）均返回 invalid_input")


# ---------------------------------------------------------------------
# 7) 兼容：不传新参数时老行为不变；advance 默认参数等价于一串 step
# ---------------------------------------------------------------------
def test_legacy_step_unchanged(service):
    sid = service.create_session({"map": MAP20})["session_id"]
    service.plan_session(sid, {"start": {"x": 0.5, "y": 5.5, "theta": 0.0},
                               "goal": {"x": 19.5, "y": 5.5}})
    out = service.step_session(sid, {})
    # 老响应字段完整
    assert set(out["result"].keys()) == {
        "pose", "cmd", "replanned", "arrived", "blocked", "note"
    }
    assert out["session"]["step_count"] == 1
    # 轨迹查询是纯增量能力，老快照仍带全部旧字段
    snap = service.get_session(sid)["session"]
    for key in ("status", "pose", "goal", "path", "path_index", "step_count",
                "message", "last_note", "created_at", "updated_at"):
        assert key in snap
    log.info("老调用方式（POST /step 空体）行为与响应字段保持不变")


def test_advance_defaults_match_repeated_steps():
    # 默认 dt=0.1、默认速度上限：advance(1.0) 的轨迹应与连续 10 次 step 完全一致
    rows = [[0] * 14 for _ in range(8)]

    def fresh():
        s = _session(rows, footprint_radius=0.2)
        s.plan(Pose(0.5, 0.5, 0.2), Point(13.5, 7.5), 0.0)
        return s

    s1 = fresh()
    s2 = fresh()
    ra = s1.advance(0.0, 1.0)  # now 不参与运动学
    poses_step = []
    for _ in range(10):
        r = s2.step(0.0)
        poses_step.append(r.pose)
    assert ra.steps_executed == 10
    for pa, pb in zip(ra.poses, poses_step):
        assert (pa.x, pa.y, pa.theta) == pytest.approx((pb.x, pb.y, pb.theta), abs=1e-12)
    log.info("兼容性：advance(1.0) 默认参数与 10 次 step 位姿逐点一致")


def test_advance_http_roundtrip_and_idempotent(client):
    r = client.post("/sessions", json={"map": MAP20})
    sid = r.json()["session_id"]
    client.post(f"/sessions/{sid}/plan",
                json={"start": {"x": 0.5, "y": 5.5, "theta": 0},
                      "goal": {"x": 19.5, "y": 5.5}})
    body = {"duration": 3.0, "max_v": 2.0}
    a = client.post(f"/sessions/{sid}/advance", json=body,
                    headers={"Idempotency-Key": "adv1"}).json()
    steps_after_first = a["session"]["step_count"]
    b = client.post(f"/sessions/{sid}/advance", json=body,
                    headers={"Idempotency-Key": "adv1"}).json()
    assert a["advance"] == b["advance"]
    assert b["session"]["step_count"] == steps_after_first
    tr = client.get(f"/sessions/{sid}/trajectory").json()
    assert tr["length"] == steps_after_first
    log.info("advance 幂等：同键重放步数保持 %d，轨迹长度一致", steps_after_first)
