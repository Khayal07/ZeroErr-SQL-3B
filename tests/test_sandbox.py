"""Tests for the read-only SQLite sandbox."""

from __future__ import annotations

import pytest

from zeroerr.guardrail.sandbox import SQLiteSandbox


def test_allow_select(dept_sandbox):
    res = dept_sandbox.execute("SELECT name FROM department ORDER BY department_id")
    assert res.ok
    assert res.columns == ["name"]
    assert res.rows == [("Engineering",), ("Sales",), ("Marketing",)]


def test_blocks_writes(dept_sandbox):
    res = dept_sandbox.execute("INSERT INTO employee VALUES (99, 'X', 1, NULL, 2026)")
    assert not res.ok
    assert "only SELECT/WITH/EXPLAIN allowed" in res.error


def test_missing_column_reports_error(dept_sandbox):
    res = dept_sandbox.execute("SELECT does_not_exist FROM employee")
    assert not res.ok
    assert "no such column" in res.error


def test_read_only_mode_blocks_writes_even_when_allowed_prefix_slippery(dept_sandbox):
    res = dept_sandbox.execute("WITH x AS (SELECT 1) UPDATE employee SET name = 'y'")
    assert not res.ok


def test_timeout_fires_for_slow_query(dept_sandbox, monkeypatch):
    sandbox = SQLiteSandbox(dept_sandbox.database, timeout_seconds=0.2)

    def slow_run(conn, sql):
        import time

        time.sleep(5)
        return None

    monkeypatch.setattr(sandbox, "_run", slow_run)
    res = sandbox.execute("SELECT 1")
    assert not res.ok
    assert "timed out" in res.error


def test_row_cap(dept_sandbox):
    res = dept_sandbox.execute("SELECT employee_id FROM employee")
    assert res.row_count == 5


def test_readonly_path_does_not_create_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SQLiteSandbox(tmp_path / "missing.sqlite")