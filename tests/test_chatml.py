"""Tests for ChatML rendering and SQL extraction."""

from __future__ import annotations

from zeroerr.data.chatml import (
    build_generation_prompt,
    extract_sql,
    render_chatml,
)


def test_render_full_turn():
    out = render_chatml(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    )
    assert out.startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nq<|im_end|>" in out
    assert "<|im_start|>assistant\na<|im_end|>" in out


def test_generation_prompt_ends_with_assistant_marker():
    out = build_generation_prompt("SCHEMA", "what is the count?")
    assert out.endswith("<|im_start|>assistant\n")
    assert "SCHEMA" in out


def test_extract_sql_strips_fences():
    raw = "```sql\nSELECT 1;\n```"
    assert extract_sql(raw) == "SELECT 1"


def test_extract_sql_strips_copy():
    raw = 'Here is your query:\nSELECT * FROM t;'
    assert extract_sql(raw) == "SELECT * FROM t"


def test_extract_sql_with_leading_with():
    raw = "WITH x AS (SELECT 1) SELECT * FROM x;"
    assert extract_sql(raw) == "WITH x AS (SELECT 1) SELECT * FROM x"


def test_extract_sql_empty():
    assert extract_sql("   ") == ""