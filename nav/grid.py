"""占用栅格：构建/校验、世界坐标与栅格索引互转、障碍查询与膨胀。"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

from .errors import (
    CoordinateOutOfBoundsError,
    InvalidInputError,
    MapShapeError,
)


def _normalize_value(v: object) -> int:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        if v in (0, 1):
            return v
    if isinstance(v, float) and v in (0.0, 1.0):
        return int(v)
    raise InvalidInputError(
        "栅格取值只能为 0（自由）或 1（障碍）", {"value": repr(v)}
    )


def rasterize_rect(
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> list[int]:
    """把世界坐标矩形 [xmin,xmax]x[ymin,ymax] 栅格化为同形状的 0/1 扁平列表。

    与栅格无交叠时返回全 0（不报错），由调用方决定如何解释。
    """
    if xmax < xmin or ymax < ymin:
        raise InvalidInputError(
            "动态障碍矩形必须满足 xmin<=xmax 且 ymin<=ymax",
            {"rect": [xmin, ymin, xmax, ymax]},
        )
    cx0 = max(0, math.floor((xmin - origin_x) / resolution))
    cy0 = max(0, math.floor((ymin - origin_y) / resolution))
    # 上界：覆盖所有与矩形相交的单元
    cx1 = min(width - 1, math.ceil((xmax - origin_x) / resolution) - 1)
    cy1 = min(height - 1, math.ceil((ymax - origin_y) / resolution) - 1)
    mask = [0] * (width * height)
    if cx0 > cx1 or cy0 > cy1:
        return mask
    for cy in range(cy0, cy1 + 1):
        base = cy * width
        for cx in range(cx0, cx1 + 1):
            mask[base + cx] = 1
    return mask


class OccupancyGrid:
    """二维占用栅格。

    data 为定长 list[int]（0 自由 / 1 障碍），索引规则 data[y*width + x]。
    origin 为栅格单元 (0,0) 左下角对应的世界坐标；resolution 为米/格。
    """

    __slots__ = (
        "width", "height", "resolution", "origin_x", "origin_y", "data",
        "_proximity",
    )

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        data: list[int],
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.data = data
        self._proximity: "ProximityIndex | None" = None

    # ---- 构建与校验 ----
    @classmethod
    def from_raw(
        cls,
        cells: Sequence,
        *,
        resolution: float,
        origin: tuple[float, float] = (0.0, 0.0),
        width: int | None = None,
        height: int | None = None,
    ) -> "OccupancyGrid":
        """从二维行序列（[[0,1,...],...]）或一维扁平序列（需给 width/height）构建。

        - resolution 必须为有限正数，origin 必须为有限数；
        - 行长度不一致 / 一维长度与 width*height 不符 -> MapShapeError；
        - 出现 0/1 以外取值 -> InvalidInputError。
        """
        if not isinstance(resolution, (int, float)) or isinstance(resolution, bool):
            raise InvalidInputError("resolution 必须为数值", {"value": repr(resolution)})
        if not math.isfinite(float(resolution)) or float(resolution) <= 0:
            raise InvalidInputError(
                "resolution 必须为有限正数", {"value": resolution}
            )
        ox, oy = origin
        if not (math.isfinite(float(ox)) and math.isfinite(float(oy))):
            raise InvalidInputError("origin 必须为有限数值", {"origin": [ox, oy]})

        rows: Sequence
        try:
            n = len(cells)
        except TypeError as exc:
            raise InvalidInputError("cells 必须为二维或一维序列") from exc

        # 判定二维：第一个元素本身是序列且不是字符串/字节
        is_2d = (
            n > 0
            and isinstance(cells[0], Sequence)
            and not isinstance(cells[0], (str, bytes))
        )

        if is_2d:
            w = len(cells[0])
            if w == 0:
                raise MapShapeError("地图行数/列数不得为 0")
            flat: list[int] = []
            for row in cells:
                if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                    raise MapShapeError("二维地图的每一行必须为等长序列")
                if len(row) != w:
                    raise MapShapeError(
                        "地图行长度不一致",
                        {"expected": w, "actual": len(row)},
                    )
                flat.extend(_normalize_value(v) for v in row)
            h = n
        else:
            if width is None or height is None:
                raise InvalidInputError(
                    "一维 cells 必须同时提供 width 与 height"
                )
            if not isinstance(width, int) or not isinstance(height, int):
                raise InvalidInputError("width/height 必须为整数")
            if width <= 0 or height <= 0:
                raise MapShapeError("width/height 必须为正整数")
            if width * height != n:
                raise MapShapeError(
                    "一维 cells 长度与 width*height 不符",
                    {"cells": n, "width": width, "height": height},
                )
            w, h = width, height
            flat = [_normalize_value(v) for v in cells]

        if (width is not None and width != w) or (height is not None and height != h):
            raise MapShapeError(
                "显式 width/height 与数据实际尺寸不一致",
                {"given": [width, height], "actual": [w, h]},
            )
        return cls(w, h, float(resolution), float(ox), float(oy), flat)

    # ---- 坐标变换 ----
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """世界坐标 -> 栅格索引（floor），越界抛 CoordinateOutOfBoundsError。"""
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            raise CoordinateOutOfBoundsError(
                "坐标必须为有限数值", {"x": x, "y": y}
            )
        cx = math.floor((float(x) - self.origin_x) / self.resolution)
        cy = math.floor((float(y) - self.origin_y) / self.resolution)
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            raise CoordinateOutOfBoundsError(
                "坐标超出地图覆盖范围",
                {
                    "x": x,
                    "y": y,
                    "map_x_range": [
                        self.origin_x,
                        self.origin_x + self.width * self.resolution,
                    ],
                    "map_y_range": [
                        self.origin_y,
                        self.origin_y + self.height * self.resolution,
                    ],
                },
            )
        return cx, cy

    def cell_center(self, cx: int, cy: int) -> tuple[float, float]:
        """栅格索引 -> 单元中心世界坐标。"""
        return (
            self.origin_x + (cx + 0.5) * self.resolution,
            self.origin_y + (cy + 0.5) * self.resolution,
        )

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    # ---- 障碍查询 ----
    def is_occupied_cell(self, cx: int, cy: int) -> bool:
        """越界一律按障碍（墙）处理，保证任何轨迹不会穿出地图边界。"""
        if not self.in_bounds(cx, cy):
            return True
        return self.data[cy * self.width + cx] == 1

    def is_occupied_world(self, x: float, y: float) -> bool:
        cx, cy = self.world_to_cell(x, y)
        return self.is_occupied_cell(cx, cy)

    def footprint_blocked(self, x: float, y: float, radius: float) -> bool:
        """检查以 (x,y) 为圆心、radius 为半径的机器人外接圆是否触障或越界。

        1) 圆与地图世界边界求交，越出覆盖范围即冲突；
        2) 扫描与圆外接方形相交的全部栅格，做圆-矩形最近点精确判定。
        """
        x_min = self.origin_x
        y_min = self.origin_y
        x_max = self.origin_x + self.width * self.resolution
        y_max = self.origin_y + self.height * self.resolution
        if x - radius < x_min or x + radius > x_max:
            return True
        if y - radius < y_min or y + radius > y_max:
            return True

        cx = math.floor((x - self.origin_x) / self.resolution)
        cy = math.floor((y - self.origin_y) / self.resolution)
        r_cells = math.ceil(radius / self.resolution) + 1
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                gx, gy = cx + dx, cy + dy
                if not self.in_bounds(gx, gy):
                    continue
                ccx, ccy = self.cell_center(gx, gy)
                half = self.resolution / 2
                closest_x = min(max(x, ccx - half), ccx + half)
                closest_y = min(max(y, ccy - half), ccy + half)
                if (closest_x - x) ** 2 + (closest_y - y) ** 2 <= radius**2:
                    if self.data[gy * self.width + gx] == 1:
                        return True
        return False

    def merged(self, other_mask: Sequence[int]) -> "OccupancyGrid":
        """与同形状 0/1 掩码做逐格 OR，返回新栅格（动态障碍层叠加）。"""
        if len(other_mask) != len(self.data):
            raise MapShapeError(
                "动态障碍掩码与地图尺寸不一致",
                {"mask": len(other_mask), "map": len(self.data)},
            )
        merged_data = [
            1 if (a == 1 or b == 1) else 0 for a, b in zip(self.data, other_mask)
        ]
        return OccupancyGrid(
            self.width, self.height, self.resolution,
            self.origin_x, self.origin_y, merged_data,
        )

    # ---- 连续运动安全 ----
    def segment_footprint_blocked(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        radius: float,
        spacing: float | None = None,
    ) -> bool:
        """检查从 (x0,y0) 到 (x1,y1) 的直线扫掠过程中外接圆是否触障/越界。

        采样步长不超过 spacing（缺省为 1/4 栅格），并且每个采样点都以
        ``radius + spacing/2`` 做半径膨胀复核，因此任意两个相邻采样之间
        机器人中心移动不超过 spacing，真实外接圆与所检圆之间的间隙被覆盖，
        从机制上杜绝“大步长跨过薄墙/窄通道”的漏检。
        """
        dist = math.hypot(x1 - x0, y1 - y0)
        step = spacing if spacing is not None else self.resolution * 0.25
        if step <= 0:
            step = self.resolution * 0.25
        check_r = radius + step * 0.5
        n = max(1, math.ceil(dist / step))
        for i in range(n + 1):
            t = i / n
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if self.footprint_blocked(x, y, check_r):
                return True
        return False

    def clearance_at(self, x: float, y: float) -> float:
        """返回世界点 (x,y) 到最近障碍表面（或地图世界边界）的距离（米）。

        点位于障碍内时返回 0.0；地图外返回 0.0。结果基于栅格分辨率离散，
        精度约为一个栅格量级（预演报告用途，碰撞判定仍以 footprint_blocked
        的精确圆-矩形检测为准）。
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0
        cx = math.floor((x - self.origin_x) / self.resolution)
        cy = math.floor((y - self.origin_y) / self.resolution)
        if not self.in_bounds(cx, cy):
            return 0.0
        if self._proximity is None:
            self._proximity = ProximityIndex.build(self)
        return self._proximity.surface_clearance(self, cx, cy, x, y)

    def inflated(self, radius_cells: int) -> "OccupancyGrid":
        """按 Chebyshev 距离 <= radius_cells 做障碍膨胀（保守膨胀，含对角方向）。"""
        if radius_cells <= 0:
            return OccupancyGrid(
                self.width, self.height, self.resolution,
                self.origin_x, self.origin_y, list(self.data),
            )
        w, h = self.width, self.height
        # 多源 BFS：从所有障碍格出发，距离 <= radius_cells 的格子全部置 1
        dist = [-1] * (w * h)
        q: deque[int] = deque()
        for i, v in enumerate(self.data):
            if v == 1:
                dist[i] = 0
                q.append(i)
        while q:
            idx = q.popleft()
            d = dist[idx]
            if d >= radius_cells:
                continue
            cx, cy = idx % w, idx // w
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        ni = ny * w + nx
                        if dist[ni] == -1:
                            dist[ni] = d + 1
                            q.append(ni)
        out = [1 if d >= 0 else 0 for d in dist]
        return OccupancyGrid(
            w, h, self.resolution, self.origin_x, self.origin_y, out
        )

    def clone_data(self) -> list[int]:
        return list(self.data)


class ProximityIndex:
    """障碍邻近索引：基于 Felzenszwalb 精确欧氏距离变换（栅格中心坐标）。

    每个自由格记录最近障碍格的栅格坐标；查询世界点时取该格及其 8 邻域
    障碍格，对点-障碍矩形做精确最近点距离，并与到地图世界边界的距离取 min。
    建索引为 O(W·H)，单次查询为 O(1)，适合预演在大地图上反复查询。
    """

    __slots__ = ("nearest_x", "nearest_y", "resolution")

    def __init__(self, nearest_x: list[int], nearest_y: list[int],
                 resolution: float) -> None:
        self.nearest_x = nearest_x
        self.nearest_y = nearest_y
        self.resolution = resolution

    @classmethod
    def build(cls, grid: OccupancyGrid) -> "ProximityIndex":
        w, h = grid.width, grid.height
        inf = float("inf")
        d2: list[float] = [0.0] * (w * h)
        nx: list[int] = [-1] * (w * h)
        ny: list[int] = [-1] * (w * h)

        # 初始代价：障碍格 0，自由格 +inf
        f = [0.0 if v == 1 else inf for v in grid.data]

        # --- 沿 x 方向的 1D 变换（逐行，站点代价 0/inf）---
        for row in range(h):
            base = row * w
            dist, winner = _edt1d(f[base:base + w])
            for col in range(w):
                idx = base + col
                d2[idx] = dist[col]
                if winner[col] >= 0:
                    nx[idx] = winner[col]
                    ny[idx] = row

        # --- 沿 y 方向的 1D 变换（逐列，偏移为 x 变换结果）---
        for col in range(w):
            offsets = [d2[row * w + col] for row in range(h)]
            dist, winner = _edt1d(offsets)
            for row in range(h):
                idx = row * w + col
                d2[idx] = dist[row]
                src_row = winner[row]
                if src_row >= 0:
                    src_idx = src_row * w + col
                    nx[idx] = nx[src_idx]
                    ny[idx] = ny[src_idx]

        return cls(nx, ny, grid.resolution)

    def surface_clearance(
        self, grid: OccupancyGrid, cx: int, cy: int, x: float, y: float
    ) -> float:
        # 到地图世界边界的距离（边界按“墙”处理）
        boundary = min(
            x - grid.origin_x,
            grid.origin_x + grid.width * grid.resolution - x,
            y - grid.origin_y,
            grid.origin_y + grid.height * grid.resolution - y,
        )
        boundary = max(0.0, boundary)
        if grid.data[cy * grid.width + cx] == 1:
            return 0.0
        sx = self.nearest_x[cy * grid.width + cx]
        sy = self.nearest_y[cy * grid.width + cx]
        if sx < 0:
            # 全自由图：没有障碍格，只受边界约束
            return boundary
        half = grid.resolution / 2
        r = self.resolution
        diag_half = r * 0.70710678  # 格心到格角的距离

        def _rect_dist(gx: int, gy: int) -> float:
            ccx = grid.origin_x + (gx + 0.5) * r
            ccy = grid.origin_y + (gy + 0.5) * r
            qx = min(max(x, ccx - half), ccx + half)
            qy = min(max(y, ccy - half), ccy + half)
            return math.hypot(qx - x, qy - y)

        # 最近障碍格给出表面距离上界（其格心距离 - 对角半宽也是下界参考）
        best = _rect_dist(sx, sy)
        # 以查询点为中心圈定候选窗口：表面距离能 < best 的障碍格，其格心到
        # 查询点的距离必 < best + 格对角半宽。近墙时窗口只有 1~3 格，
        # 空旷处按最近障碍距离封顶，单次查询为亚线性小常数。
        radius_m = best + diag_half + r
        R = math.ceil(radius_m / r) + 1
        x0 = max(0, cx - R)
        x1 = min(grid.width - 1, cx + R)
        y0 = max(0, cy - R)
        y1 = min(grid.height - 1, cy + R)
        for gy0 in range(y0, y1 + 1):
            base = gy0 * grid.width
            for gx0 in range(x0, x1 + 1):
                if gx0 == sx and gy0 == sy:
                    continue
                if grid.data[base + gx0] != 1:
                    continue
                ccx = grid.origin_x + (gx0 + 0.5) * r
                ccy = grid.origin_y + (gy0 + 0.5) * r
                # 格心到查询点距离减对角半宽 = 表面距离的严格下界
                if math.hypot(ccx - x, ccy - y) - diag_half >= best:
                    continue
                best = min(best, _rect_dist(gx0, gy0))

        return max(0.0, min(best, boundary))


def _edt1d(f: list[float]) -> tuple[list[float], list[int]]:
    """一维 Felzenszwalb 平方距离变换。

    输入 f[i] 为站点偏移代价（站点处为 0，其余 +inf），输出：
      dist[i]   = min_q (i - q)^2 + f[q]
      winner[i] = 取得最小值的站点下标（无站点时为 -1）
    抛物线包络算法，复杂度 O(n)。
    """
    n = len(f)
    inf = float("inf")
    v = [0] * n  # 包络上各抛物线对应的站点
    z = [0.0] * (n + 1)  # 相邻抛物线交点

    # 用第一个有限站点（障碍格）初始化包络，+inf 站点不参与构造
    first = next((i for i, val in enumerate(f) if math.isfinite(val)), -1)
    if first < 0:
        return [inf] * n, [-1] * n

    k = 0
    v[0] = first
    z[0] = -inf
    z[1] = inf
    for q in range(first + 1, n):
        if not math.isfinite(f[q]):
            continue
        while True:
            denom = 2 * q - 2 * v[k]
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / denom
            if s <= z[k] and k > 0:
                k -= 1
                continue
            break
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = inf

    dist = [0.0] * n
    winner = [-1] * n
    k = 0
    for i in range(n):
        while z[k + 1] < i:
            k += 1
        q = v[k]
        dist[i] = (i - q) ** 2 + f[q]
        winner[i] = q
    return dist, winner
