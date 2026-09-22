"""兼容入口：`fastapi run main.py` / `uvicorn main:app` 均可。"""

from app.main import app  # noqa: F401
