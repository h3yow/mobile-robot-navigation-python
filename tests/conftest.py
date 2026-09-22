"""测试公共夹具与日志工具。

所有测试通过 ``case_log`` 打印【输入】【判定】【状态迁移】，
便于复现与排查错误原因区分。
"""

from __future__ import annotations

import logging
import sys

import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _case_header(request):
    logger = logging.getLogger("navsim.test")
    logger.info("=" * 70)
    logger.info("CASE %s", request.node.name)
    yield
    logger.info("=" * 70)


def case_log(stage: str, **fields) -> None:
    """结构化打印测试输入/判定/迁移依据。"""

    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    logging.getLogger("navsim.test").info("[%s] %s", stage, parts)


def open_grid(w: int, h: int, resolution: float = 0.1) -> list[str]:
    return ["." * w for _ in range(h)]


def box_with_gap(w: int = 20, h: int = 13, gap_col: int = 9) -> list[str]:
    """两个大房间 + 一堵带单格缺口的中墙。"""

    rows = ["#" * w]
    rows += ["#" + "." * (w - 2) + "#"] * ((h - 3) // 2)
    rows.append("#" * gap_col + "." + "#" * (w - gap_col - 1))
    rows += ["#" + "." * (w - 2) + "#"] * (h - 3 - (h - 3) // 2)
    rows.append("#" * w)
    assert len(rows) == h
    return rows
