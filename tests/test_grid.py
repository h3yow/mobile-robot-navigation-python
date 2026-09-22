"""栅格构建/校验/坐标变换/碰撞/膨胀的退化与非法输入测试。"""

from __future__ import annotations

import logging

import pytest

from nav.errors import (
    CoordinateOutOfBoundsError,
    InvalidInputError,
    MapShapeError,
)
from nav.grid import OccupancyGrid, rasterize_rect

log = logging.getLogger("test_grid")


def test_2d_and_1d_inputs_equivalent():
    cells2d = [[0, 1, 0], [1, 0, 0]]
    g2 = OccupancyGrid.from_raw(cells2d, resolution=0.5, origin=(1.0, -2.0))
    flat = [0, 1, 0, 1, 0, 0]
    g1 = OccupancyGrid.from_raw(flat, resolution=0.5, origin=(1.0, -2.0),
                                width=3, height=2)
    log.info("输入二维=%s 一维=%s -> 尺寸=%dx%d, data=%s",
             cells2d, flat, g2.width, g2.height, g2.data)
    assert g2.data == flat == g1.data
    assert (g2.width, g2.height) == (3, 2)


def test_ragged_rows_rejected():
    with pytest.raises(MapShapeError) as ei:
        OccupancyGrid.from_raw([[0, 1], [0]], resolution=1.0)
    log.info("关键判定：行长度不一致 -> code=%s", ei.value.code)
    assert ei.value.code == "map_shape_error"


def test_flat_length_mismatch_rejected():
    with pytest.raises(MapShapeError) as ei:
        OccupancyGrid.from_raw([0, 1, 0], resolution=1.0, width=2, height=2)
    log.info("关键判定：一维长度与 w*h 不符 -> code=%s, details=%s",
             ei.value.code, ei.value.details)
    assert ei.value.code == "map_shape_error"


def test_explicit_dimension_conflict_rejected():
    with pytest.raises(MapShapeError) as ei:
        OccupancyGrid.from_raw([[0, 1], [0, 1]], resolution=1.0, width=3, height=2)
    assert ei.value.code == "map_shape_error"


@pytest.mark.parametrize("bad_res", [0, -1.0, float("nan"), float("inf")])
def test_bad_resolution_rejected(bad_res):
    with pytest.raises(InvalidInputError) as ei:
        OccupancyGrid.from_raw([[0]], resolution=bad_res)
    log.info("关键判定：resolution=%s -> code=%s", bad_res, ei.value.code)
    assert ei.value.code == "invalid_input"


def test_bad_cell_value_rejected():
    with pytest.raises(InvalidInputError) as ei:
        OccupancyGrid.from_raw([[0, 2]], resolution=1.0)
    assert ei.value.code == "invalid_input"


def test_coordinate_transform_roundtrip():
    g = OccupancyGrid.from_raw([[0] * 4 for _ in range(3)],
                               resolution=0.5, origin=(-1.0, 2.0))
    cx, cy = g.world_to_cell(-0.75, 2.25)
    wx, wy = g.cell_center(cx, cy)
    log.info("坐标变换 (-0.75,2.25) -> cell(%d,%d) -> center(%.2f,%.2f)",
             cx, cy, wx, wy)
    assert (cx, cy) == (0, 0)
    assert (wx, wy) == (-0.75, 2.25)


@pytest.mark.parametrize("pt", [(10.0, 0.0), (-1.0001, 0.0), (0.0, 1.6)])
def test_out_of_bounds_distinguishable(pt):
    g = OccupancyGrid.from_raw([[0] * 2 for _ in range(3)],
                               resolution=0.5)  # 覆盖 x[0,1) y[0,1.5)
    with pytest.raises(CoordinateOutOfBoundsError) as ei:
        g.world_to_cell(*pt)
    log.info("关键判定：越界点 %s -> code=%s", pt, ei.value.code)
    assert ei.value.code == "coordinate_out_of_bounds"


def test_footprint_free_vs_obstacle_vs_border():
    # 障碍格 (2,0)，世界中心 (2.5,0.5)
    g = OccupancyGrid.from_raw([[0, 0, 1], [0, 0, 0]], resolution=1.0)
    assert g.footprint_blocked(0.5, 0.5, 0.3) is False
    assert g.footprint_blocked(2.5, 0.5, 0.3) is True
    assert g.footprint_blocked(0.2, 0.5, 0.3) is True   # 圆越过左边界
    log.info("足迹判定：自由=False 障碍=True 越界=True 均符合预期")


def test_inflation_seals_one_cell_gap():
    # y=1 整行障碍，只在 x=2 留 1 格缺口
    rows = [[0] * 5 for _ in range(4)]
    for x in range(5):
        rows[1][x] = 1
    rows[1][2] = 0
    g = OccupancyGrid.from_raw(rows, resolution=1.0)
    gi = g.inflated(1)
    log.info("膨胀前缺口格(2,1)=%d；膨胀 1 格后=%d（缺口被保守封死）",
             g.data[1 * 5 + 2], gi.data[1 * 5 + 2])
    assert g.data[1 * 5 + 2] == 0
    assert gi.data[1 * 5 + 2] == 1


def test_rasterize_rect_clipped_to_map():
    m = rasterize_rect(3, 2, 1.0, 0.0, 0.0, -5.0, -5.0, 0.5, 0.5)
    assert sum(m) == 1
    log.info("部分越界矩形栅格化后仅落 1 格（裁剪而非报错）：%s", m)
