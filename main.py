from fastapi import FastAPI

from nav.api import create_app

app = create_app()


@app.get("/")
def read_root():
    return {"service": "mobile-robot-navigation", "docs": "/docs", "health": "/health"}
