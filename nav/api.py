"""HTTP 层：FastAPI 路由、请求解析与统一错误处理。

端点：
  GET    /health
  POST   /sessions                       创建会话（支持 Idempotency-Key）
  GET    /sessions/{id}                  查询状态快照
  POST   /sessions/{id}/plan             设置起终点并全局规划
  POST   /sessions/{id}/step             单步执行（可携带 dynamic_obstacles）
  POST   /sessions/{id}/advance          按指定时长连续推进（dt/max_v 可调）
  GET    /sessions/{id}/trajectory       查询保留的执行轨迹（含扫掠位姿）
  POST   /sessions/{id}/rehearse         只读预演（不改会话状态）
  POST   /sessions/{id}/replan           手动触发重规划
  POST   /sessions/{id}/reset            重置会话
  DELETE /sessions/{id}                  删除会话
"""

from __future__ import annotations

import json

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import InvalidInputError, NavError
from .service import NavigationService

router = APIRouter()


def get_service(request: Request) -> NavigationService:
    return request.app.state.nav_service


async def _read_json(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidInputError(
            "请求体不是合法 JSON", {"detail": str(exc)}
        ) from exc
    if not isinstance(data, dict):
        raise InvalidInputError("请求体必须为 JSON 对象")
    return data


def _idem_key(request: Request) -> str | None:
    return request.headers.get("idempotency-key")


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/sessions")
async def create_session(request: Request) -> dict:
    body = await _read_json(request)
    return get_service(request).create_session(body, _idem_key(request))


@router.get("/sessions/{session_id}")
def get_session(request: Request, session_id: str) -> dict:
    return get_service(request).get_session(session_id)


@router.post("/sessions/{session_id}/plan")
async def plan_session(request: Request, session_id: str) -> dict:
    body = await _read_json(request)
    return get_service(request).plan_session(session_id, body)


@router.post("/sessions/{session_id}/step")
async def step_session(request: Request, session_id: str) -> dict:
    body = await _read_json(request)
    return get_service(request).step_session(
        session_id, body, _idem_key(request)
    )


@router.post("/sessions/{session_id}/advance")
async def advance_session(request: Request, session_id: str) -> dict:
    body = await _read_json(request)
    return get_service(request).advance_session(
        session_id, body, _idem_key(request)
    )


@router.get("/sessions/{session_id}/trajectory")
def get_trajectory(request: Request, session_id: str) -> dict:
    return get_service(request).get_trajectory(session_id)


@router.post("/sessions/{session_id}/rehearse")
async def rehearse_session(request: Request, session_id: str) -> dict:
    body = await _read_json(request)
    return get_service(request).rehearse_session(session_id, body)


@router.post("/sessions/{session_id}/replan")
def replan_session(request: Request, session_id: str) -> dict:
    return get_service(request).replan_session(session_id)
    return get_service(request).replan_session(session_id)


@router.post("/sessions/{session_id}/reset")
def reset_session(request: Request, session_id: str) -> dict:
    return get_service(request).reset_session(session_id)


@router.delete("/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    return get_service(request).delete_session(session_id)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NavError)
    async def _nav_error_handler(_request: Request, exc: NavError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": exc.to_dict()},
        )


def create_app(service: NavigationService | None = None) -> FastAPI:
    """构造 FastAPI 应用，注入服务并挂载路由。"""
    app = FastAPI(title="mobile-robot-navigation")
    app.state.nav_service = service if service is not None else NavigationService()
    app.include_router(router)
    register_exception_handlers(app)
    return app
