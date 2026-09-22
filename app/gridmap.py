"""占用栅格地图：多格式输入解析、坐标变换与碰撞查询。

坐标约定
--------
- 世界坐标单位为米；原点 ``origin=(ox, oy)`` 为地图**左下角**。
- 存储为行主序一维字节数组，row 0 对应世界 y 最大的一行（北向），
  与常见 ASCII 地图自上而下的阅读顺序一致。
- 格值：0=空闲，1=占用；地图边界之外一律视为占用（防止穿出地图）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.errors import ErrorCode, NavigationError

FREE_CHARS = frozenset({".", "0", " "})
OCCUPIED_CHARS = frozenset({"#", "1", "x", "X"})


@dataclass(frozen=True)
class OccupancyGrid:
    """占用栅格。"""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    cells: bytes
    notes: tuple[str, ...] = field(default=())

    # ---- 基本索引 -------------------------------------------------------
    def index(self, col: int, row: int) -> int:
        return row * self.width + col

    def in_bounds_cell(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def cell_occupied(self, col: int, row: int) -> bool:
        """越界一律按占用处理。"""

        if not self.in_bounds_cell(col, row):
            return True
        return self.cells[self.index(col, row)] == 1

    def free_cell_count(self) -> int:
        return self.width * self.height - sum(self.cells)

    # ---- 坐标变换 -------------------------------------------------------
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """世界坐标 -> (col, stored_row)；不做越界裁剪。"""

        col = math.floor((x - self.origin_x) / self.resolution)
        row_bottom = math.floor((y - self.origin_y) / self.resolution)
        return col, self.height - 1 - row_bottom

    def cell_center_world(self, col: int, row: int) -> tuple[float, float]:
        row_bottom = self.height - 1 - row
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row_bottom + 0.5) * self.resolution
        return x, y

    def world_bounds(
        self,
    ) -> tuple[float, float, float, float]:
        """返回 (xmin, ymin, xmax, ymax)。"""

        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width * self.resolution,
            self.origin_y + self.height * self.resolution,
        )

    def world_in_bounds(self, x: float, y: float, margin: float = 0.0) -> bool:
        xmin, ymin, xmax, ymax = self.world_bounds()
        return (
            xmin + margin <= x <= xmax - margin
            and ymin + margin <= y <= ymax - margin
        )

    # ---- 碰撞 -----------------------------------------------------------
    def disk_collides(self, x: float, y: float, radius: float) -> bool:
        """圆盘 footprint 是否与障碍或地图外区域碰撞。

        对圆盘外接框覆盖的每个栅格做“点到格方形”精确距离判定，
        采样粒度只取决于半径/分辨率，不受机器人瞬时位置影响，无漏检方向。
        """

        if radius < 0.0 or not math.isfinite(x) or not math.isfinite(y):
            return True
        # 地图边界外 = 碰撞
        if not self.world_in_bounds(x, y, radius):
            return True
        r = self.resolution
        c0 = math.floor((x - self.origin_x - radius) / r)
        c1 = math.floor((x - self.origin_x + radius) / r)
        rb0 = math.floor((y - self.origin_y - radius) / r)
        rb1 = math.floor((y - self.origin_y + radius) / r)
        for col in range(c0, c1 + 1):
            for rb in range(rb0, rb1 + 1):
                if col < 0 or rb < 0 or col >= self.width or rb >= self.height:
                    # 外接框触及地图外
                    return True
                if self.cells[(self.height - 1 - rb) * self.width + col] != 1:
                    continue
                gx0 = self.origin_x + col * r
                gy0 = self.origin_y + rb * r
                closest_x = min(max(x, gx0), gx0 + r)
                closest_y = min(max(y, gy0), gy0 + r)
                dx = x - closest_x
                dy = y - closest_y
                if dx * dx + dy * dy <= radius * radius + 1e-12:
                    return True
        return False


def _validate_resolution_origin(
    resolution: object, origin: object
) -> tuple[float, float, float]:
    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool):
        raise NavigationError(
            ErrorCode.MAP_INVALID_RESOLUTION,
            f"地图分辨率 resolution={resolution!r} 非法：必须为正数（米/格）",
            {"resolution": resolution},
        )
    res_f = float(resolution)
    if not math.isfinite(res_f) or res_f <= 0.0:
        raise NavigationError(
            ErrorCode.MAP_INVALID_RESOLUTION,
            f"地图分辨率 resolution={resolution!r} 非法：必须为正有限数",
            {"resolution": resolution},
        )
    if origin is None:
        return res_f, 0.0, 0.0
    if (
        not isinstance(origin, (list, tuple))
        or len(origin) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in origin)
        or not all(math.isfinite(float(v)) for v in origin)
    ):
        raise NavigationError(
            ErrorCode.MAP_INVALID_ORIGIN,
            f"地图原点 origin={origin!r} 非法：必须为两个有限数 [x, y]",
            {"origin": list(origin) if isinstance(origin, (list, tuple)) else origin},
        )
    return res_f, float(origin[0]), float(origin[1])


def parse_map_input(
    grid: list[str] | None = None,
    occupancy: list[list[int]] | list[int] | None = None,
    width: int | None = None,
    height: int | None = None,
    resolution: float = 0.1,
    origin: list[float] | tuple[float, ...] | None = None,
) -> OccupancyGrid:
    """将多种地图输入归一化为 :class:`OccupancyGrid`。

    支持形式：
      1. ``grid``：ASCII 行列表，``.``/``0``/空格=空闲，``#``/``1``/``x``=占用；
         行宽不齐时右侧以空闲补齐（记录在 notes 的 ``ascii_rows_padded``）。
      2. ``occupancy``：二维 0/1 数组（行序同 ASCII，自上而下），行宽必须一致。
      3. ``occupancy`` 为一维 0/1 数组时必须同时给出匹配的 width/height。
      4. 显式 width/height 与数据尺寸不一致时返回 MAP_DIMENSION_MISMATCH。
    """

    res_f, ox, oy = _validate_resolution_origin(resolution, origin)
    notes: list[str] = []

    if grid is not None and occupancy is not None:
        raise NavigationError(
            ErrorCode.INVALID_REQUEST,
            "grid 与 occupancy 两种地图形式不可同时提供",
            {},
        )

    rows: list[list[int]] = []

    if grid is not None:
        if not isinstance(grid, list) or not all(isinstance(line, str) for line in grid):
            raise NavigationError(
                ErrorCode.MAP_SHAPE_INVALID,
                "grid 必须为字符串行列表 list[str]",
                {},
            )
        if len(grid) == 0:
            raise NavigationError(ErrorCode.MAP_EMPTY, "grid 为空：至少需要一行", {})
        w = max(len(line) for line in grid)
        if w == 0:
            raise NavigationError(ErrorCode.MAP_EMPTY, "grid 为空：存在零长度行且无有效列", {})
        for line in grid:
            row: list[int] = []
            for ch in line:
                if ch in FREE_CHARS:
                    row.append(0)
                elif ch in OCCUPIED_CHARS:
                    row.append(1)
                else:
                    raise NavigationError(
                        ErrorCode.MAP_INVALID_CELL_VALUE,
                        f"grid 含非法字符 {ch!r}：允许 '.', '0', ' ', '#', '1', 'x'",
                        {"char": ch, "ascii": ord(ch)},
                    )
            if len(row) < w:
                row.extend([0] * (w - len(row)))
                notes.append("ascii_rows_padded")
            rows.append(row)

    elif occupancy is not None:
        if not isinstance(occupancy, list) or len(occupancy) == 0:
            raise NavigationError(ErrorCode.MAP_EMPTY, "occupancy 为空", {})
        if isinstance(occupancy[0], (list, tuple)):
            for row in occupancy:
                if not isinstance(row, (list, tuple)) or len(row) == 0:
                    raise NavigationError(
                        ErrorCode.MAP_SHAPE_INVALID,
                        "occupancy 必须为非空二维 0/1 数组",
                        {},
                    )
                parsed_row: list[int] = []
                for v in row:
                    if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1):
                        raise NavigationError(
                            ErrorCode.MAP_INVALID_CELL_VALUE,
                            f"occupancy 含非法格值 {v!r}：仅允许 0 或 1",
                            {"value": v},
                        )
                    parsed_row.append(v)
                rows.append(parsed_row)
            widths = {len(r) for r in rows}
            if len(widths) != 1:
                raise NavigationError(
                    ErrorCode.MAP_DIMENSION_MISMATCH,
                    f"occupancy 各行宽度不一致：{sorted(widths)}",
                    {"row_widths": sorted(widths)},
                )
        else:
            # 一维形式必须给出 width/height
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                raise NavigationError(
                    ErrorCode.MAP_DIMENSION_MISMATCH,
                    "一维 occupancy 必须同时提供正整数 width 与 height",
                    {"width": width, "height": height},
                )
            if len(occupancy) != width * height:
                raise NavigationError(
                    ErrorCode.MAP_DIMENSION_MISMATCH,
                    f"一维 occupancy 长度 {len(occupancy)} != width*height={width * height}",
                    {"length": len(occupancy), "width": width, "height": height},
                )
            for v in occupancy:
                if isinstance(v, bool) or not isinstance(v, int) or v not in (0, 1):
                    raise NavigationError(
                        ErrorCode.MAP_INVALID_CELL_VALUE,
                        f"occupancy 含非法格值 {v!r}：仅允许 0 或 1",
                        {"value": v},
                    )
            flat = [int(v) for v in occupancy]
            rows = [flat[i * width : (i + 1) * width] for i in range(height)]
    else:
        raise NavigationError(
            ErrorCode.MAP_EMPTY,
            "未提供地图：grid 与 occupancy 至少需要其一",
            {},
        )

    h = len(rows)
    w = len(rows[0])

    if width is not None and width != w:
        raise NavigationError(
            ErrorCode.MAP_DIMENSION_MISMATCH,
            f"显式 width={width} 与数据宽度 {w} 不一致",
            {"provided_width": width, "data_width": w},
        )
    if height is not None and height != h:
        raise NavigationError(
            ErrorCode.MAP_DIMENSION_MISMATCH,
            f"显式 height={height} 与数据高度 {h} 不一致",
            {"provided_height": height, "data_height": h},
        )
    if w <= 0 or h <= 0:
        raise NavigationError(
            ErrorCode.MAP_SHAPE_INVALID,
            f"地图尺寸非法：width={w}, height={h}",
            {"width": w, "height": h},
        )

    buf = bytearray(w * h)
    for r_i, row in enumerate(rows):
        for c_i, v in enumerate(row):
            buf[r_i * w + c_i] = v

    if sum(buf) == len(buf):
        raise NavigationError(
            ErrorCode.MAP_NO_FREE_SPACE,
            f"地图全部被占用（{w}x{h}），不存在任何空闲栅格",
            {"width": w, "height": h},
        )

    return OccupancyGrid(
        width=w,
        height=h,
        resolution=res_f,
        origin_x=ox,
        origin_y=oy,
        cells=bytes(buf),
        notes=tuple(notes),
    )
