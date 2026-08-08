"""Tests for Spider schema normalization and DDL rendering."""

from __future__ import annotations

from zeroerr.data.schemas import Column, Schema, from_spider_json, render_ddl

MODERN = {
    "db_id": "mydb",
    "tables": [
        {"name": "users", "columns": [{"name": "id", "type": "INTEGER"}, {"name": "email", "type": "TEXT"}]}
    ],
    "foreign_keys": [{"table": "orders", "column": "user_id", "ref_table": "users", "ref_column": "id"}],
    "primary_keys": [{"table": "users", "column": "id"}],
}

CLASSIC = {
    "db_id": "classic",
    "table_names_original": ["users"],
    "column_names_original": [[0, "id"], [0, "email"]],
    "column_types": ["integer", "text"],
    "foreign_keys": [],
    "primary_keys": [[0, 0]],
}


def test_from_modern():
    schema = from_spider_json(MODERN)
    assert schema.db_id == "mydb"
    assert schema.column_names("users") == ["id", "email"]
    assert ("orders", "user_id", "users", "id") in schema.foreign_keys


def test_from_classic():
    schema = from_spider_json(CLASSIC)
    assert schema.column_names("users") == ["id", "email"]
    assert schema.primary_keys == [("users", "id")]


def test_render_ddl_contains_tables_and_fks():
    text = render_ddl(from_spider_json(MODERN))
    assert "CREATE TABLE users" in text
    assert "REFERENCES" in text or "orders.user_id -> users.id" in text