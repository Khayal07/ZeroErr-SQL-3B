"""Shared pytest fixtures: build sqlite sandbox databases in a temp dir."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zeroerr.guardrail.sandbox import SQLiteSandbox

from eval.fixtures.build_fixtures import _DATABASES


@pytest.fixture(scope="session")
def fixture_dbs(tmp_path_factory):
    base = tmp_path_factory.mktemp("dbs")
    paths = {}
    for name, statements in _DATABASES.items():
        import sqlite3

        path = base / f"{name}.sqlite"
        conn = sqlite3.connect(path)
        for sql in statements:
            conn.execute(sql)
        conn.commit()
        conn.close()
        paths[name] = path
    return paths


@pytest.fixture()
def dept_sandbox(fixture_dbs) -> SQLiteSandbox:
    return SQLiteSandbox(fixture_dbs["department_emp"])