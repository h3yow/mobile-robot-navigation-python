"""HTTP 层：FastAPI 路由、请求解析与统一错误处理。"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import NavError
from .service import NavigationService

router = APIRouter()


def get_service(request: Request) -> NavigationService:
    return request.app.state.nav_service


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/sessions")
def create_session(request: Request) -> dict:
    raise NotImplementedError


@router.get("/sessions/{session_id}")
def get_session(request: Request, session_id: str) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/plan")
def plan_session(request: Request, session_id: str) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/step")
def step_session(request: Request, session_id: str) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/replan")
def replan_session(request: Request, session_id: str) -> dict:
    raise NotImplementedError


@router.post("/sessions/{session_id}/reset")
def reset_session(request: Request, session_id: str) -> dict:
    raise NotImplementedError


@router.delete("/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    raise NotImplementedError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NavError)
    async def _nav_error_handler(_request: Request, exc: NavError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"error": exc.to_dict()})


def create_app(service: NavigationService | None = None) -> FastAPI:
    """构造 FastAPI 应用，注入服务并挂载路由。"""
    app = FastAPI(title="mobile-robot-navigation")
    app.state.nav_service = service if service is not None else NavigationService()
    app.include_router(router)
    register_exception_handlers(app)
    return app
