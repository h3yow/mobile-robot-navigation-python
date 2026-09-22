"""地图与参数输入形式、退化场景的稳定可区分判定。"""

from __future__ import annotations

import pytest

from app.errors import NavigationError
from app.gridmap import parse_map_input
from tests.conftest import case_log


def expect_error(fn, code, label):
    with pytest.raises(NavigationError) as exc:
        fn()
    case_log("判定", case=label, expect=code, got=exc.value.code.value,
             message=exc.value.message)
    assert exc.value.code == code, (label, exc.value.code)


def test_ascii_and_2d_and_flat_equivalent():
    g1 = parse_map_input(grid=["#.", ".#"], resolution=0.5, origin=[1.0, 2.0])
    g2 = parse_map_input(occupancy=[[1, 0], [0, 1]], resolution=0.5, origin=[1.0, 2.0])
    g3 = parse_map_input(occupancy=[1, 0, 0, 1], width=2, height=2,
                         resolution=0.5, origin=[1.0, 2.0])
    case_log("输入", forms=["ascii", "2d", "1d"], cells=[list(g1.cells),
             list(g2.cells), list(g3.cells)])
    assert g1.cells == g2.cells == g3.cells
    # 坐标往返
    col, row = g1.world_to_cell(1.25, 2.25)
    assert (col, row) == (0, 1)
    assert g1.cell_center_world(col, row) == (1.25, 2.25)


def test_ragged_ascii_rows_padded_with_note():
    g = parse_map_input(grid=["###", "#."])
    case_log("输入", ascii=["###", "#."], width=g.width, notes=g.notes)
    assert g.width == 3 and g.height == 2
    assert "ascii_rows_padded" in g.notes
    assert g.cells == bytes([1, 1, 1, 1, 0, 0])


def test_degenerate_empty_and_full_maps():
    expect_error(lambda: parse_map_input(), "MAP_EMPTY", "missing map")
    expect_error(lambda: parse_map_input(grid=[]), "MAP_EMPTY", "empty grid")
    expect_error(lambda: parse_map_input(grid=[""]), "MAP_EMPTY", "blank row")
    expect_error(lambda: parse_map_input(grid=["###", "###"]),
                 "MAP_NO_FREE_SPACE", "fully occupied")
    expect_error(lambda: parse_map_input(occupancy=[]), "MAP_EMPTY", "empty occ")


def test_dimension_and_value_mismatches():
    expect_error(
        lambda: parse_map_input(occupancy=[[0, 0], [0]], width=None),
        "MAP_DIMENSION_MISMATCH", "ragged 2d")
    expect_error(
        lambda: parse_map_input(grid=["##", "##"], width=3),
        "MAP_DIMENSION_MISMATCH", "explicit width mismatch")
    expect_error(
        lambda: parse_map_input(occupancy=[0, 1], width=3, height=1),
        "MAP_DIMENSION_MISMATCH", "flat length mismatch")
    expect_error(
        lambda: parse_map_input(occupancy=[[0, 2]]),
        "MAP_INVALID_CELL_VALUE", "bad cell value")
    expect_error(
        lambda: parse_map_input(grid=["a."]),
        "MAP_INVALID_CELL_VALUE", "bad ascii char")


def test_resolution_and_origin_validation():
    expect_error(lambda: parse_map_input(occupancy=[[0]], resolution=0),
                 "MAP_INVALID_RESOLUTION", "zero resolution")
    expect_error(lambda: parse_map_input(occupancy=[[0]], resolution=-0.1),
                 "MAP_INVALID_RESOLUTION", "negative resolution")
    expect_error(lambda: parse_map_input(occupancy=[[0]], origin=[1.0]),
                 "MAP_INVALID_ORIGIN", "origin length 1")
    expect_error(lambda: parse_map_input(occupancy=[[0]], origin=[1.0, "x"]),
                 "MAP_INVALID_ORIGIN", "origin non numeric")


def test_disk_collision_geometry():
    g = parse_map_input(grid=["###", "#.#", "###"], resolution=0.1)
    # 中心通道格 (1,1) 世界 (0.15,0.15)；通道宽 0.1
    case_log("输入", point=(0.15, 0.15), corridor_width=0.1)
    assert g.disk_collides(0.15, 0.15, 0.0) is False
    assert g.disk_collides(0.15, 0.15, 0.04) is False
    assert g.disk_collides(0.15, 0.15, 0.06) is True
    assert g.disk_collides(-0.01, 0.15, 0.0) is True  # 越界即碰撞
    # 闭区间：边界点在内，边界外即越界；加 margin 后边界点也判越界
    assert g.world_in_bounds(0.0, 0.15) is True
    assert g.world_in_bounds(-0.01, 0.15) is False
    assert g.world_in_bounds(0.0, 0.15, margin=0.05) is False


def test_out_of_bounds_world_coordinates_distinguishable():
    g = parse_map_input(grid=["..", ".."], resolution=1.0)
    case_log("输入", bounds=g.world_bounds())
    assert not g.world_in_bounds(-0.01, 1.0)
    assert g.world_in_bounds(0.5, 0.5)
