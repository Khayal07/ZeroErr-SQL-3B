"""HTTP routes for the text-to-SQL service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from zeroerr.api.deps import get_engine, get_registry, get_settings
from zeroerr.api.models import HealthOut, Text2SQLRequest, Text2SQLResponse
from zeroerr.config import Settings
from zeroerr.data.schemas import render_ddl
from zeroerr.engine.base import LLMEngine
from zeroerr.guardrail.orchestrator import GuardrailLoop
from zeroerr.guardrail.sandbox import SandboxRegistry

router = APIRouter()


def _schema_text_for(registry: SandboxRegistry, database_id: str) -> str:
    from zeroerr.guardrail.schema_extractor import schema_from_sandbox

    sandbox = registry.sandbox_for(database_id)
    schema = schema_from_sandbox(sandbox)
    schema.db_id = database_id
    return render_ddl(schema)


@router.get("/health", response_model=HealthOut)
def health(
    registry: SandboxRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> HealthOut:
    return HealthOut(
        status="ok",
        engine_backend=settings.engine_backend,
        sandbox_engine=settings.sandbox_engine,
        databases=list(registry.discover()),
    )


@router.get("/dbs")
def list_databases(registry: SandboxRegistry = Depends(get_registry)) -> JSONResponse:
    return JSONResponse(content={"databases": list(registry.discover())})


@router.post("/v1/text2sql", response_model=Text2SQLResponse)
def text2sql(
    payload: Text2SQLRequest,
    engine: LLMEngine = Depends(get_engine),
    registry: SandboxRegistry = Depends(get_registry),
) -> Text2SQLResponse:
    try:
        sandbox = registry.sandbox_for(payload.database_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    schema_text = _schema_text_for(registry, payload.database_id)
    loop = GuardrailLoop(
        sandbox=sandbox,
        engine=engine,
        max_rounds=payload.max_repair_rounds,
        temperature=payload.temperature,
    )
    result = loop.run(schema_text=schema_text, question=payload.question)
    resp = result.to_dict()
    if not result.ok and result.error == "all retry rounds exhausted":
        return JSONResponse(status_code=422, content=resp)
    return resp