"""FastAPI entrypoint for ZeroErr-SQL-3B."""

from __future__ import annotations

from fastapi import FastAPI

from zeroerr.api.deps import get_settings
from zeroerr.api.routes import router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Execution-guided Text-to-SQL SLM with a self-correction guardrail.",
)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("zeroerr.api.main:app", host="0.0.0.0", port=8000, reload=True)