"""全局路径规划。

代价模型
--------
- 8 邻接 A*：直线代价 1，对角代价 sqrt(2)；启发式为 octile 距离（可采纳、一致）。
- 障碍按机器人半径做**精确欧氏膨胀**（Felzenszwalb-Huttenlocher EDT，O(W*H)），
  膨胀后规划等价于点机器人在配置空间中搜索，保证路径中心线两侧 >= robot_radius
  （另由跟踪层在实时地图上逐弧段复核，双重保险）。
- 边界同样按半径封闭，路径不可能穿出地图。
- 起/终点若仅落在膨胀层（贴墙但实体不碰），在吸附容差内吸附到最近可行栅格；
  实体碰撞、越界、不可达分别给出可区分错误。
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

from app.errors import ErrorCode, NavigationError
from app.gridmap import OccupancyGrid
from app.kinematics import Kinematics

_SQRT2 = math.sqrt(2.0)
_INF = float("inf")


@dataclass
class Plan:
    """一次全局规划结果。"""

    world_path: list[tuple[float, float]]
    cell_path: list[tuple[int, int]]
    inflated: bool
    replan_count: int = 0
    expansions: int = 0
    elapsed_ms: float = 0.0
    snapped_start: tuple[float, float] | None = None
    snapped_goal: tuple[float, float] | None = None
    notes: tuple[str, ...] = field(default_factory=())


@dataclass
class PlannerConfig:
    time_budget: float = 2.0          # 秒
    max_expansions: int = 2_000_000
    inflation: bool = True
    snap_tolerance_cells: int = 3     # 起终点膨胀层吸附最大距离（格）

    def __post_init__(self) -> None:
        if not isinstance(self.time_budget, (int, float)) or self.time_budget <= 0:
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                f"planner time_budget={self.time_budget!r} 非法：必须为正数",
                {"time_budget": self.time_budget},
            )
        if not isinstance(self.max_expansions, int) or self.max_expansions <= 0:
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                f"planner max_expansions={self.max_expansions!r} 非法：必须为正整数",
                {"max_expansions": self.max_expansions},
            )


# ------------------------------------------------------------------------
# 精确欧氏距离：每个空闲格**中心**到障碍格矩形并集的距离（单位：格）。
# 与 OccupancyGrid.disk_collides 的“点到障碍矩形”几何严格一致。
#
# 第 1 步（行）：左右两次线性扫描求 h[r][c] = 同行最近障碍矩形边缘的
#   水平间隙平方；同行无障碍时 h = +inf。
# 第 2 步（列）：上下两个方向各自做一遍带半格截断的 Felzenszwalb 抛物线
#   包络扫描：处理查询行 i 时只把严格位于该侧（j<i / j>i）的源点入栈，
#   边缘顶点到查询点距离非负，栈顶即该侧受限最优，总计 O(W*H)。
# ------------------------------------------------------------------------
def obstacle_rectangle_sq_distance(
    cells: bytes, width: int, height: int
) -> list[float]:
    """每个空闲格中心到障碍矩形并集的**平方**距离（单位：格²）。"""

    INF = 1e18
    total = width * height
    h = [INF] * total

    # ---- 第 1 步：行内水平间隙平方（左右扫描） ----------------------------
    for r_i in range(height):
        base = r_i * width
        left_col = -1
        for c_i in range(width):
            idx = base + c_i
            if cells[idx] == 1:
                left_col = c_i
                h[idx] = 0.0
            elif left_col >= 0:
                d = c_i - left_col - 0.5
                h[idx] = d * d
        right_col = -1
        for c_i in range(width - 1, -1, -1):
            idx = base + c_i
            if cells[idx] == 1:
                right_col = c_i
                h[idx] = 0.0
            elif right_col >= 0:
                d = right_col - c_i - 0.5
                dd = d * d
                if dd < h[idx]:
                    h[idx] = dd

    # ---- 第 2 步：列方向带半格截断的 Felzenszwalb 包络（上下两遍） --------
    # 直接在扁平 h 上原地写入列最优；预分配的包络数组跨列复用。
    sv = [0.0] * height
    sf = [0.0] * height
    sz = [0.0] * (height + 1)
    col = [0.0] * height

    for c_i in range(width):
        for r_i in range(height):
            col[r_i] = h[r_i * width + c_i]

        for direction in (1, -1):
            top = -1
            k = 0
            rows = range(height) if direction == 1 else range(height - 1, -1, -1)
            for i in rows:
                j = i - direction
                if 0 <= j < height:
                    fj = col[j]
                    if fj < INF / 2:
                        v = j + 0.5 if direction == 1 else -(j - 0.5)
                        while top >= 0:
                            vk, fk = sv[top], sf[top]
                            cross = ((fj + v * v) - (fk + vk * vk)) / (
                                2.0 * (v - vk)
                            )
                            if top == 0 or cross > sz[top]:
                                break
                            top -= 1
                        if top < 0:
                            top = 0
                            k = 0
                            sv[0], sf[0], sz[0] = v, fj, -INF
                            sz[1] = INF
                        else:
                            vk, fk = sv[top], sf[top]
                            cross = ((fj + v * v) - (fk + vk * vk)) / (
                                2.0 * (v - vk)
                            )
                            top += 1
                            sv[top], sf[top] = v, fj
                            sz[top] = cross
                            sz[top + 1] = INF
                            if k > top:
                                k = top
                if top < 0:
                    continue
                q = float(i) if direction == 1 else -float(i)
                while k < top and sz[k + 1] <= q:
                    k += 1
                d = q - sv[k]
                val = d * d + sf[k]
                out_idx = i * width + c_i
                if val < h[out_idx]:
                    h[out_idx] = val
    return h


def obstacle_rectangle_distance(
    cells: bytes, width: int, height: int
) -> list[float]:
    """每个空闲格中心到障碍矩形并集的欧氏距离（单位：格）。"""

    return [math.sqrt(v) if v < 1e17 else _INF for v in
            obstacle_rectangle_sq_distance(cells, width, height)]


def inflate_grid(grid: OccupancyGrid, robot_radius: float) -> bytes:
    """返回按机器人半径膨胀 + 边界封闭后的一维占用数组。"""

    width, height = grid.width, grid.height
    sq = obstacle_rectangle_sq_distance(grid.cells, width, height)
    radius_cells = robot_radius / grid.resolution
    radius_sq = radius_cells * radius_cells
    eps = 1e-9
    buf = bytearray(width * height)
    for r_i in range(height):
        rb = height - 1 - r_i
        base = r_i * width
        for c_i in range(width):
            idx = base + c_i
            hit_obstacle = sq[idx] <= radius_sq + eps
            edge_cells = min(c_i + 0.5, width - 0.5 - c_i, rb + 0.5,
                             height - 0.5 - rb)
            if hit_obstacle or edge_cells <= radius_cells + eps:
                buf[idx] = 1
    return bytes(buf)


# ------------------------------------------------------------------------
def _nearest_free_cell(
    blocked: bytes, width: int, height: int, col: int, row: int, max_ring: int
) -> tuple[int, int] | None:
    """以 (col,row) 为中心按切比雪夫环由近及远寻找最近的未膨胀格。"""

    for ring in range(1, max_ring + 1):
        for dc in range(-ring, ring + 1):
            for dr in range(-ring, ring + 1):
                if max(abs(dc), abs(dr)) != ring:
                    continue
                cc, rr = col + dc, row + dr
                if 0 <= cc < width and 0 <= rr < height and blocked[rr * width + cc] == 0:
                    return cc, rr
    return None


def _astar(
    blocked: bytes,
    width: int,
    height: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    time_budget: float,
    max_expansions: int,
    started_at: float,
) -> tuple[list[tuple[int, int]], int]:
    """8 邻接 A*。对角移动仅在两个正交邻格均空闲时允许（禁止切角）。"""

    n = width * height
    s_idx = start[1] * width + start[0]
    g_idx = goal[1] * width + goal[0]
    if s_idx == g_idx:
        return [start], 0

    gx, gy = goal

    def h(col: int, row: int) -> float:
        dx = abs(col - gx)
        dy = abs(row - gy)
        return (dx + dy) + (_SQRT2 - 2.0) * min(dx, dy)

    inf = math.inf
    gscore = [inf] * n
    parent = [-1] * n
    closed = bytearray(n)
    gscore[s_idx] = 0.0
    pq: list[tuple[float, int, int]] = [(h(*start), 0, s_idx)]
    counter = 0
    expansions = 0

    while pq:
        _, _, cur = heapq.heappop(pq)
        if closed[cur]:
            continue
        if cur == g_idx:
            path: list[tuple[int, int]] = []
            node = cur
            while node != -1:
                path.append((node % width, node // width))
                node = parent[node]
            path.reverse()
            return path, expansions
        closed[cur] = 1
        expansions += 1
        if expansions > max_expansions:
            raise NavigationError(
                ErrorCode.PLAN_TIME_BUDGET_EXCEEDED,
                f"规划扩展节点数超过上限 {max_expansions}",
                {"max_expansions": max_expansions},
            )
        if (expansions & 1023) == 0 and time.perf_counter() - started_at > time_budget:
            raise NavigationError(
                ErrorCode.PLAN_TIME_BUDGET_EXCEEDED,
                f"规划耗时超过时间预算 {time_budget:.3f}s（已扩展 {expansions} 节点）",
                {"time_budget": time_budget, "expansions": expansions},
            )

        col = cur % width
        row = cur // width
        base_g = gscore[cur]
        for dc, dr, step_cost in (
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, _SQRT2), (1, -1, _SQRT2), (-1, 1, _SQRT2), (-1, -1, _SQRT2),
        ):
            nc, nr = col + dc, row + dr
            if not (0 <= nc < width and 0 <= nr < height):
                continue
            ni = nr * width + nc
            if closed[ni] or blocked[ni] == 1:
                continue
            if dc != 0 and dr != 0:
                # 对角必须两个正交邻格都可通行，禁止从两个障碍夹角切过去
                if blocked[row * width + nc] == 1 or blocked[nr * width + col] == 1:
                    continue
            tentative = base_g + step_cost
            if tentative < gscore[ni]:
                gscore[ni] = tentative
                parent[ni] = cur
                counter += 1
                heapq.heappush(pq, (tentative + h(nc, nr), counter, ni))

    raise NavigationError(
        ErrorCode.GOAL_UNREACHABLE,
        "目标不可达：A* 穷尽可通行区域仍无法到达目标（膨胀后连通域不包含目标）",
        {"expansions": expansions, "free_cells_explored": expansions},
    )


def plan_global(
    grid: OccupancyGrid,
    kin: Kinematics,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    cfg: PlannerConfig | None = None,
    replan_count: int = 0,
) -> Plan:
    """在栅格上执行全局规划。"""

    cfg = cfg or PlannerConfig()
    t0 = time.perf_counter()
    notes: list[str] = list(grid.notes)

    for name, pt in (("start", start_xy), ("goal", goal_xy)):
        if (
            not isinstance(pt, tuple)
            or len(pt) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in pt)
        ):
            raise NavigationError(
                ErrorCode.INVALID_REQUEST,
                f"{name} 坐标 {pt!r} 非法：必须为两个有限数 [x, y]",
                {name: pt},
            )

    # 1) 越界 / 实体碰撞判定（先于膨胀，原因可区分）
    for name, (x, y), oob_code, occ_code in (
        ("start", start_xy, ErrorCode.COORDINATE_OUT_OF_BOUNDS, ErrorCode.START_IN_OBSTACLE),
        ("goal", goal_xy, ErrorCode.COORDINATE_OUT_OF_BOUNDS, ErrorCode.GOAL_IN_OBSTACLE),
    ):
        if not grid.world_in_bounds(x, y, 0.0):
            xmin, ymin, xmax, ymax = grid.world_bounds()
            raise NavigationError(
                oob_code,
                f"{name} 坐标 ({x:.4g}, {y:.4g}) 越界，地图范围 "
                f"x∈[{xmin:.4g},{xmax:.4g}], y∈[{ymin:.4g},{ymax:.4g}]",
                {"point": [x, y], "bounds": [xmin, ymin, xmax, ymax]},
            )
        if grid.disk_collides(x, y, kin.robot_radius):
            col, row = grid.world_to_cell(x, y)
            raise NavigationError(
                occ_code,
                f"{name} 坐标 ({x:.4g}, {y:.4g}) 位于障碍内或与障碍相交 "
                f"（cell=({col},{row}), robot_radius={kin.robot_radius:g}）",
                {"point": [x, y], "cell": [col, row], "robot_radius": kin.robot_radius},
            )

    # 2) 膨胀
    blocked = inflate_grid(grid, kin.robot_radius) if cfg.inflation else grid.cells
    if cfg.inflation:
        notes.append("inflated_by_robot_radius")

    # 3) 起终点格 + 膨胀层吸附
    snapped: dict[str, tuple[float, float]] = {}

    def resolve_cell(name: str, xy: tuple[float, float]) -> tuple[int, int]:
        col, row = grid.world_to_cell(*xy)
        if blocked[row * grid.width + col] == 0:
            return col, row
        found = _nearest_free_cell(
            blocked, grid.width, grid.height, col, row, cfg.snap_tolerance_cells
        )
        if found is None:
            raise NavigationError(
                ErrorCode.GOAL_UNREACHABLE if name == "goal" else ErrorCode.START_IN_OBSTACLE,
                f"{name} 落在膨胀封闭区内，且在 {cfg.snap_tolerance_cells} 格内无可吸附位置",
                {"point": list(xy), "cell": [col, row],
                 "snap_tolerance_cells": cfg.snap_tolerance_cells},
            )
        snapped[name] = grid.cell_center_world(*found)
        notes.append(f"{name}_snapped")
        return found

    start_cell = resolve_cell("start", start_xy)
    goal_cell = resolve_cell("goal", goal_xy)

    # 4) A* 搜索
    cell_path, expansions = _astar(
        blocked,
        grid.width,
        grid.height,
        start_cell,
        goal_cell,
        cfg.time_budget,
        cfg.max_expansions,
        t0,
    )
    world_path = [grid.cell_center_world(c, r) for c, r in cell_path]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return Plan(
        world_path=world_path,
        cell_path=cell_path,
        inflated=cfg.inflation,
        replan_count=replan_count,
        expansions=expansions,
        elapsed_ms=elapsed_ms,
        snapped_start=snapped.get("start"),
        snapped_goal=snapped.get("goal"),
        notes=tuple(notes),
    )
