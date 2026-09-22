"""占用栅格：构建/校验、世界坐标与栅格索引互转、障碍查询与膨胀。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OccupancyGrid:
    """二维占用栅格。

    data 为定长 list[int]（0 自由 / 1 障碍），索引规则 data[y*width + x]。
    origin 为栅格单元 (0,0) 左下角对应的世界坐标；resolution 为米/格。
    """

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: list[int]

    # ---- 构建与校验 ----
    @classmethod
    def from_raw(
        cls,
        cells,
        *,
        resolution: float,
        origin=(0.0, 0.0),
        width: int | None = None,
        height: int | None = None,
    ) -> "OccupancyGrid":
        """从多种输入（二维序列/一维序列+宽高）构建栅格并做一致性校验。"""
        raise NotImplementedError

    # ---- 坐标变换 ----
    def world_to_cell(self, x: float, y: float):
        """世界坐标 -> 栅格索引（floor），越界抛 CoordinateOutOfBoundsError。"""
        raise NotImplementedError

    def cell_center(self, cx: int, cy: int):
        """栅格索引 -> 单元中心世界坐标。"""
        raise NotImplementedError

    def in_bounds(self, cx: int, cy: int) -> bool:
        raise NotImplementedError

    # ---- 障碍查询 ----
    def is_occupied_cell(self, cx: int, cy: int) -> bool:
        raise NotImplementedError

    def is_occupied_world(self, x: float, y: float) -> bool:
        raise NotImplementedError

    def footprint_blocked(self, x: float, y: float, radius: float) -> bool:
        """以机器人外接半径检查世界坐标点是否与障碍（或边界）冲突。"""
        raise NotImplementedError

    def inflated(self, radius_cells: int) -> "OccupancyGrid":
        """返回按给定格数做障碍膨胀后的新栅格（用于规划）。"""
        raise NotImplementedError
