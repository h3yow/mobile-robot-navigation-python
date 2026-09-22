"""HTTP 层：请求模型、路由、统一错误处理（骨架）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.errors import ErrorCode, NavigationError
from app.session import SessionStore


class PoseModel(BaseModel):
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class KinematicsModel(BaseModel):
    max_linear_speed: float | None = None
    max_angular_speed: float | None = None
    max_curvature: float | None = None
    robot_radius: float | None = None
    max_linear_accel: float | None = None
    dt: float | None = None


class CreateSessionRequest(BaseModel):
    session_id: str | None = None
    grid: list[str] | None = None
    occupancy: list[list[int]] | None = None
    width: int | None = None
    height: int | None = None
    resolution: float = 0.1
    origin: list[float] | None = None
    start: PoseModel = Field(default_factory=PoseModel)
    goal: PoseModel = Field(default_factory=PoseModel)
    kinematics: KinematicsModel = Field(default_factory=KinematicsModel)
    planner_time_budget: float = 2.0
    lookahead: float | None = None
    goal_tolerance: float | None = None
    auto_plan: bool = True
    request_id: str | None = None


class StepRequest(BaseModel):
    request_id: str | None = None
    steps: int = 1


class ObstacleUpdate(BaseModel):
    cells: list[list[int]] | None = None       # [[col,row,value], ...]
    rectangles: list[dict[str, Any]] | None = None
    reset: bool = False
    request_id: str | None = None


router = APIRouter()


def get_store(request: Request) -> SessionStore:
    raise NotImplementedError


@router.post("/sessions", status_code=201)
async def create_session(payload: CreateSessionRequest, request: Request) -> dict:
    raise NotImplementedError


@router.get("/sessions")
async def list_sessions(request: Request) -> dict:
    raise NotImplementedError


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/plan")
async def plan_session(session_id: str, payload: dict | None, request: Request) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/step")
async def step_session(session_id: str, payload: StepRequest | None, request: Request) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/obstacles")
async def update_obstacles(session_id: str, payload: ObstacleUpdate, request: Request) -> dict:
    raise NotImplementedError


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    raise NotImplementedError


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NavigationError)
    async def _nav_error_handler(_: Request, exc: NavigationError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_body())

    @app.exception_handler(Exception)
    async def _fallback_handler(_: Request, exc: Exception) -> JSONResponse:
        body = {
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": str(exc) or exc.__class__.__name__,
                "detail": {},
            }
        }
        return JSONResponse(status_code=500, content=body)
