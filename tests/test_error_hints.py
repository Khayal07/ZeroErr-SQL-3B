"""Tests for error-message -> repair-hint translation."""

from __future__ import annotations

from zeroerr.guardrail.error_hints import hint_for_error

COLS = ["name", "salary", "department_id"]


def test_no_such_column_with_suggestion():
    hint = hint_for_error('no such column: salaray', COLS)
    assert "salaray" in hint
    assert "salary" in hint


def test_no_such_column_without_candidates():
    hint = hint_for_error("no such column: xyz", None)
    assert "xyz" in hint
    assert "did you mean" not in hint


def test_no_such_table():
    assert "missing_t" in hint_for_error("no such table: missing_t", COLS)


def test_ambiguous_column():
    assert "qualify" in hint_for_error("ambiguous column name: id", COLS)


def test_syntax_error():
    hint = hint_for_error("syntax error near \"SELECT\"", COLS)
    assert "syntax error" in hint


def test_incomplete_input():
    assert "incomplete" in hint_for_error("incomplete input", COLS)


def test_unknown_function():
    hint = hint_for_error('no such function: bogusfn', COLS)
    assert "bogusfn" in hint


def test_fallback_returns_truncated_message():
    assert hint_for_error("something totally unexpected happened here", COLS).startswith("something")