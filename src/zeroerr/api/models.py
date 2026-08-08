"""Pydantic request / response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Text2SQLRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural-language question.")
    database_id: str = Field(..., description="Registered sandbox database identifier.")
    max_repair_rounds: int = Field(default=3, ge=1, le=5)
    temperature: float = Field(default=0.1, ge=0.0, le=1.5)


class RepairAttemptOut(BaseModel):
    round: int
    sql: str
    error: str | None = None


class Text2SQLResponse(BaseModel):
    status: str
    sql: str | None = None
    rounds: int
    attempts: list[RepairAttemptOut] = []
    error: str | None = None


class HealthOut(BaseModel):
    status: str
    engine_backend: str
    sandbox_engine: str
    databases: list[str]