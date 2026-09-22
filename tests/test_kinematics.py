"""运动学约束、圆弧积分与原地自转。"""

from __future__ import annotations

import math

import pytest

from app.errors import NavigationError
from app.kinematics import Kinematics, default_kinematics
from tests.conftest import case_log


def test_default_curvature_consistency():
    kin = default_kinematics(max_linear_speed=1.0, max_angular_speed=2.0)
    case_log("判定", kappa=kin.max_curvature, R=kin.min_turning_radius())
    assert kin.max_curvature == 2.0


def test_reject_non_positive_and_inconsistent_params():
    cases = [
        dict(max_linear_speed=0, max_angular_speed=1, max_curvature=1,
             robot_radius=.1, max_linear_accel=1, dt=.1),
        dict(max_linear_speed=1, max_angular_speed=1, max_curvature=-2,
             robot_radius=.1, max_linear_accel=1, dt=.1),
        dict(max_linear_speed=1, max_angular_speed=1, max_curvature=2,
             robot_radius=.1, max_linear_accel=1, dt=.1),  # v*k > w
    ]
    for kw in cases:
        with pytest.raises(NavigationError) as exc:
            Kinematics(**kw)
        case_log("判定", params=kw, code=exc.value.code.value)
        assert exc.value.code.value == "INVALID_KINEMATICS"


def test_arc_integration_full_circle_and_line():
    kin = default_kinematics(max_linear_speed=1.0, max_angular_speed=1.0)
    x, y, th = 0.0, 0.0, 0.0
    for _ in range(63):
        x, y, th = kin.step_arc(x, y, th, 1.0, 1.0, 0.1)
    case_log("判定", after_circle=(round(x, 3), round(y, 3)), theta=round(th, 3))
    assert math.hypot(x, y) < 0.05
    xn, yn, thn = kin.step_arc(0, 0, 0, 1, 0, 0.1)
    assert (xn, yn) == (0.1, 0.0)


def test_clamp_respects_all_envelopes():
    kin = default_kinematics(max_linear_speed=1.0, max_angular_speed=1.0,
                             max_linear_speed_=None) if False else \
        Kinematics(1.0, 1.0, 1.0, 0.1, 1.0, 0.1)
    v, k, w = kin.clamp_command(2.0, 5.0)
    case_log("判定", cmd=(v, k, w))
    assert v == 1.0 and abs(k) <= 1.0 and abs(w) <= 1.0
    v, k, w = kin.clamp_command(1.0, 5.0)
    assert abs(k - 1.0) < 1e-12 and abs(w - 1.0) < 1e-12


def test_in_place_rotation_changes_heading_only():
    kin = Kinematics(1.0, 2.0, 2.0, 0.0, 1.0, 0.1)
    x, y, th = kin.step_arc(1.0, 2.0, 0.0, 0.0, 0.0, 0.1, omega=1.5)
    case_log("判定", pose=(x, y, th))
    assert (x, y) == (1.0, 2.0)
    assert abs(th - 0.15) < 1e-12
    # omega 超限被钳制
    _, _, w = kin.clamp_command(0.0, 0.0, omega=9.0)
    assert w == 2.0
