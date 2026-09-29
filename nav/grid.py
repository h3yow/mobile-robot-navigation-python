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

    __slots__ = ("width", "height", "resolution", "origin_x", "origin_y", "data")

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

    # ---- 线段扫掠（大跨度位移安全复核）----
    def segment_blocked(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
    ) -> bool:
        """圆盘沿 (p1)->(p2) 线段扫掠形成的胶囊是否触障或越界。

        精确判定：边界先做胶囊-世界矩形相交检查；障碍侧扫描线段外接矩形
        （外扩 radius）覆盖的栅格，对每个障碍格做线段-矩形最近点距离比较，
        因此即使单次位移大于隔墙厚度/通道宽度也不会“穿障漏检”。
        """
        res = self.resolution
        x_min = self.origin_x
        y_min = self.origin_y
        x_max = x_min + self.width * res
        y_max = y_min + self.height * res
        # 胶囊越出世界边界（含端点与线段上离边界最近处）
        lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
        lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
        if lo_x - radius < x_min or hi_x + radius > x_max:
            return True
        if lo_y - radius < y_min or hi_y + radius > y_max:
            return True

        cx0 = max(0, math.floor((lo_x - radius - x_min) / res))
        cx1 = min(self.width - 1, math.floor((hi_x + radius - x_min) / res))
        cy0 = max(0, math.floor((lo_y - radius - y_min) / res))
        cy1 = min(self.height - 1, math.floor((hi_y + radius - y_min) / res))
        r2 = radius * radius
        for gy in range(cy0, cy1 + 1):
            base = gy * self.width
            for gx in range(cx0, cx1 + 1):
                if self.data[base + gx] != 1:
                    continue
                ccx = x_min + (gx + 0.5) * res
                ccy = y_min + (gy + 0.5) * res
                half = res / 2
                if _seg_rect_dist2(x1, y1, x2, y2,
                                   ccx - half, ccy - half,
                                   ccx + half, ccy + half) <= r2:
                    return True
        return False

    def point_clearance(
        self, x: float, y: float, max_search_cells: float
    ) -> float:
        """点到障碍单元或世界边界的最近中心距离（米）；无候选时按搜索半径封顶。"""
        res = self.resolution
        x_min = self.origin_x
        y_min = self.origin_y
        x_max = x_min + self.width * res
        y_max = y_min + self.height * res
        best = min(x - x_min, x_max - x, y - y_min, y_max - y)
        cap = max_search_cells * res
        if best >= cap:
            return cap
        cx = math.floor((x - x_min) / res)
        cy = math.floor((y - y_min) / res)
        r = math.ceil(max_search_cells) + 1
        half = res / 2
        for gy in range(max(0, cy - r), min(self.height, cy + r + 1)):
            base = gy * self.width
            ccy = y_min + (gy + 0.5) * res
            for gx in range(max(0, cx - r), min(self.width, cx + r + 1)):
                if self.data[base + gx] != 1:
                    continue
                ccx = x_min + (gx + 0.5) * res
                qx = max(abs(x - ccx) - half, 0.0)
                qy = max(abs(y - ccy) - half, 0.0)
                d = math.hypot(qx, qy)
                if d < best:
                    best = d
                    if best <= 0.0:
                        return 0.0
        return best

    def segment_clearance(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        max_search_cells: float,
    ) -> float:
        """线段到障碍单元或世界边界的最近中心距离（米）。

        大地图友好：只扫描线段外接矩形外扩 max_search_cells 个栅格内的障碍；
        若搜索半径内没有任何障碍，返回 max_search_cells*resolution（封顶值）。
        """
        res = self.resolution
        x_min = self.origin_x
        y_min = self.origin_y
        x_max = x_min + self.width * res
        y_max = y_min + self.height * res
        cap = max_search_cells * res
        # 边界：线段上离某条边最近的距离（端点夹住时由端点距离给出）
        best = min(
            min(x1, x2) - x_min,
            x_max - max(x1, x2),
            min(y1, y2) - y_min,
            y_max - max(y1, y2),
        )
        if best >= cap:
            return cap
        lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
        lo_y, hi_y = (y1, y2) if y1 <= y2 else (y2, y1)
        cx0 = max(0, math.floor((lo_x - cap - x_min) / res))
        cx1 = min(self.width - 1, math.floor((hi_x + cap - x_min) / res))
        cy0 = max(0, math.floor((lo_y - cap - y_min) / res))
        cy1 = min(self.height - 1, math.floor((hi_y + cap - y_min) / res))
        half = res / 2
        for gy in range(cy0, cy1 + 1):
            base = gy * self.width
            ccy = y_min + (gy + 0.5) * res
            for gx in range(cx0, cx1 + 1):
                if self.data[base + gx] != 1:
                    continue
                ccx = x_min + (gx + 0.5) * res
                d2 = _seg_rect_dist2(
                    x1, y1, x2, y2,
                    ccx - half, ccy - half, ccx + half, ccy + half,
                )
                d = math.sqrt(d2)
                if d < best:
                    best = d
                    if best <= 0.0:
                        return 0.0
        return best

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


# ----------------------------------------------------------------------
# 线段扫掠碰撞/间隙所需的几何辅助（线段 vs 轴对齐矩形的精确距离平方）
# ----------------------------------------------------------------------
def _seg_intersects_aabb(
    ax: float, ay: float, bx: float, by: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> bool:
    """Liang–Barsky 裁剪：线段与 AABB 是否相交（含端点落在边界上）。"""
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - xmin), (dx, xmax - ax),
                 (-dy, ay - ymin), (dy, ymax - ay)):
        if abs(p) < 1e-15:
            if q < 0.0:
                return False  # 线段平行于该对边界且位于外侧
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return True


def _point_seg_dist2(
    px: float, py: float,
    ax: float, ay: float, bx: float, by: float,
) -> float:
    vx, vy = bx - ax, by - ay
    l2 = vx * vx + vy * vy
    if l2 <= 1e-30:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * vx + (py - ay) * vy) / l2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = ax + t * vx, ay + t * vy
    return (px - qx) ** 2 + (py - qy) ** 2


def _seg_seg_dist2(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> float:
    """两条线段之间的最近距离平方。

    线段间最近点对要么包含端点，要么（平行且投影区间重叠时）垂直距离——
    后者也等于某个端点到另一线段的垂足距离。因此四个“点到另一线段”距离
    的最小值即精确结果。
    """
    best = float("inf")
    for px0, py0 in ((ax, ay), (bx, by)):
        d2 = _point_seg_dist2(px0, py0, cx, cy, dx, dy)
        if d2 < best:
            best = d2
    for qx0, qy0 in ((cx, cy), (dx, dy)):
        d2 = _point_seg_dist2(qx0, qy0, ax, ay, bx, by)
        if d2 < best:
            best = d2
    return best


def _seg_rect_dist2(
    x1: float, y1: float, x2: float, y2: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> float:
    """线段到轴对齐矩形的最近距离平方（相交/包含时为 0）。"""
    if _seg_intersects_aabb(
        x1, y1, x2, y2, xmin, ymin, xmax, ymax
    ):
        return 0.0
    best = float("inf")
    edges = (
        (xmin, ymin, xmax, ymin),
        (xmax, ymin, xmax, ymax),
        (xmax, ymax, xmin, ymax),
        (xmin, ymax, xmin, ymin),
    )
    for cx0, cy0, cx1, cy1 in edges:
        d2 = _seg_seg_dist2(
            x1, y1, x2, y2, cx0, cy0, cx1, cy1
        )
        if d2 < best:
            best = d2
    return best
