"""Tests for the self-correction guardrail loop using a scripted fake engine."""

from __future__ import annotations

from zeroerr.engine.base import LLMEngine
from zeroerr.guardrail.orchestrator import GuardrailLoop

GOOD = "SELECT name FROM department ORDER BY department_id"
BAD_TABLE = "SELECT name FROM nope_table"


class ScriptedEngine(LLMEngine):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_resolves_after_one_repair(dept_sandbox):
    engine = ScriptedEngine([BAD_TABLE, GOOD])
    loop = GuardrailLoop(dept_sandbox, engine, max_rounds=3)
    result = loop.run(schema_text="fake schema", question="list departments")
    assert result.ok
    assert result.sql == GOOD
    assert result.rounds == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].error and "no such table" in result.attempts[0].error


def test_repair_prompt_contains_error_hint(dept_sandbox):
    engine = ScriptedEngine([BAD_TABLE, GOOD])
    loop = GuardrailLoop(dept_sandbox, engine, max_rounds=3)
    loop.run(schema_text="schema", question="q")
    repair_prompt = engine.prompts[1]
    assert "nope_table" in repair_prompt
    assert "no such table" in repair_prompt or "does not exist" in repair_prompt


def test_succeeds_first_attempt(dept_sandbox):
    engine = ScriptedEngine([GOOD])
    loop = GuardrailLoop(dept_sandbox, engine, max_rounds=3)
    result = loop.run(schema_text="s", question="q")
    assert result.ok and result.rounds == 1
    assert len(engine.prompts) == 1


def test_exhausts_retries(dept_sandbox):
    engine = ScriptedEngine([BAD_TABLE, BAD_TABLE, BAD_TABLE])
    loop = GuardrailLoop(dept_sandbox, engine, max_rounds=3)
    result = loop.run(schema_text="s", question="q")
    assert not result.ok
    assert result.rounds == 3
    assert result.error == "all retry rounds exhausted"


def test_captures_to_dict(dept_sandbox):
    engine = ScriptedEngine([GOOD])
    loop = GuardrailLoop(dept_sandbox, engine)
    data = loop.run(schema_text="s", question="q").to_dict()
    assert data["status"] == "resolved"
    assert data["sql"] == GOOD