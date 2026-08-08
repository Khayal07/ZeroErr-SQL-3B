"""Tests for the dataset prep CLI helpers."""

from __future__ import annotations

import json

from zeroerr.data.prep import schema_text_for, split_by_db, to_chatml_row


def test_split_by_db_is_leakage_free():
    rows = [
        {"db_id": f"d{i % 3}", "question": f"q{i}", "query": "SELECT 1", "text": f"t{i}"}
        for i in range(30)
    ]
    train, val = split_by_db(rows, val_fraction=0.2, seed=0)
    train_ids = {r["db_id"] for r in train}
    val_ids = {r["db_id"] for r in val}
    assert train_ids.isdisjoint(val_ids)
    assert sum(1 for r in rows if r["db_id"] in val_ids) == len(val)


def test_schema_injected_from_fixture(fixture_dbs):
    text = schema_text_for("department_emp", str(fixture_dbs["department_emp"].parent))
    assert "CREATE TABLE employee" in text
    assert "department_id" in text


def test_chatml_row_embeds_schema(schema_dir_fixture):
    row = to_chatml_row({"db_id": "department_emp", "question": "q", "query": "SELECT 1"}, schema_dir_fixture)
    assert row["text"].startswith("<|im_start|>system\n")
    assert "CREATE TABLE" in row["text"]


def test_repair_row_uses_broken_sql():
    row = to_chatml_row(
        {"db_id": "d1", "question": "q", "query": "SELECT 1", "broken_sql": "SELEC 1", "repair_target": True}
    )
    assert "SELEC 1" in row["text"]


def test_schema_text_missing_db_returns_empty(fixture_dbs):
    assert schema_text_for("nope", str(fixture_dbs["department_emp"].parent)) == ""