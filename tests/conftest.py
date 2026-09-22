"""pytest 公共夹具与日志配置。

要求“单测日志中打印输入、关键判定与状态迁移依据”：
所有测试通过 logging 记录关键信息，pytest -s 时直接可见。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from nav.api import create_app
from nav.service import NavigationService

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@pytest.fixture()
def service() -> NavigationService:
    return NavigationService(ttl_seconds=60.0)


@pytest.fixture()
def client(service):
    import warnings

    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient

    app = create_app(service)
    return TestClient(app)


def make_map(w: int, h: int, resolution: float = 1.0, origin=(0.0, 0.0),
             walls: list[tuple[int, int]] | None = None):
    """便捷构造占用栅格：walls 为障碍格 (x,y) 列表。"""
    from nav.grid import OccupancyGrid

    cells = [[0] * w for _ in range(h)]
    for x, y in walls or []:
        cells[y][x] = 1
    return OccupancyGrid.from_raw(
        cells, resolution=resolution, origin=origin
    )


@pytest.fixture()
def builder():
    return make_map
