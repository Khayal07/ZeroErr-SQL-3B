"""Execution-guided self-correction loop: generate -> execute -> repair (max N)."""

from __future__ import annotations

from dataclasses import dataclass, field

from zeroerr.data.chatml import build_generation_prompt, build_repair_prompt, extract_sql
from zeroerr.engine.base import LLMEngine
from zeroerr.guardrail.error_hints import hint_for_error
from zeroerr.guardrail.sandbox import SQLiteSandbox


@dataclass
class RepairAttempt:
    round: int
    sql: str
    error: str | None


@dataclass
class GuardrailResult:
    ok: bool
    sql: str | None
    rounds: int
    attempts: list[RepairAttempt] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": "resolved" if self.ok else "unresolved",
            "sql": self.sql,
            "rounds": self.rounds,
            "attempts": [{"round": a.round, "sql": a.sql, "error": a.error} for a in self.attempts],
            "error": self.error,
        }


class GuardrailLoop:
    """Generate SQL, execute it in the sandbox, feed errors back to the SLM, retry."""

    def __init__(
        self,
        sandbox: SQLiteSandbox,
        engine: LLMEngine,
        max_rounds: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ):
        self.sandbox = sandbox
        self.engine = engine
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _candidate_columns(self) -> list[str]:
        try:
            from zeroerr.guardrail.schema_extractor import schema_from_sandbox

            schema = schema_from_sandbox(self.sandbox)
            return [c for cols in schema.tables.values() for c in cols]
        except Exception:
            return []

    def run(self, schema_text: str, question: str, initial_sql: str | None = None) -> GuardrailResult:
        attempts: list[RepairAttempt] = []
        sql = initial_sql
        if not sql:
            raw = self.engine.generate(
                build_generation_prompt(schema_text, question),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            sql = extract_sql(raw)

        columns = self._candidate_columns()

        for round_no in range(1, self.max_rounds + 1):
            if not sql:
                return GuardrailResult(ok=False, sql=None, rounds=round_no, attempts=attempts, error="empty generation")
            result = self.sandbox.execute(sql)
            attempts.append(RepairAttempt(round=round_no, sql=sql, error=None if result.ok else result.error))
            if result.ok:
                return GuardrailResult(ok=True, sql=sql, rounds=round_no, attempts=attempts)
            if round_no == self.max_rounds:
                return GuardrailResult(ok=False, sql=sql, rounds=round_no, attempts=attempts, error="all retry rounds exhausted")

            hint = hint_for_error(result.error or "", columns)
            raw = self.engine.generate(
                build_repair_prompt(schema_text, question, sql, hint),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            sql = extract_sql(raw)