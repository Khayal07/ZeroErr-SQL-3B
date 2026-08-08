"""Tests for dataset filtering and difficulty bucketing."""

from __future__ import annotations

from zeroerr.data.filter import (
    attach_repair_rows,
    balance_by_difficulty,
    corrupt_sql,
    dedupe,
    difficulty_bucket,
    filter_dataset,
    is_rejected,
    is_too_long,
)

EASY = "SELECT name FROM department"
HARD = "SELECT d.name, COUNT(*) FROM department d JOIN employee e ON d.department_id = e.dept_id WHERE e.salary > 50000 GROUP BY d.department_id HAVING COUNT(*) > 1 ORDER BY COUNT(*) DESC"


def test_difficulty_buckets():
    assert difficulty_bucket(EASY) == "easy"
    assert difficulty_bucket(HARD) == "hard"
    assert difficulty_bucket("SELECT name FROM department WHERE salary > 10") == "medium"


def test_is_rejected_blocks_writes():
    assert is_rejected("DELETE FROM employee")
    assert is_rejected("INSERT INTO employee VALUES (1)")
    assert is_rejected("drop table employee")
    assert not is_rejected(EASY)
    assert not is_rejected("with x as (select 1) select * from x")


def test_too_long():
    assert not is_too_long("q", EASY)
    assert is_too_long("a" * 500, EASY)
    assert is_too_long("q", "select " + "a" * 3000)


def test_dedupe():
    rows = [{"db_id": "d1", "question": "same?", "query": EASY}] * 3
    assert len(dedupe(rows)) == 1


def test_balance_caps_easy():
    rows = [{"db_id": f"d{i}", "question": f"q{i}", "query": EASY} for i in range(50)]
    out = balance_by_difficulty(rows, per_bucket=10)
    assert len(out) == 10


def test_filter_dataset_end_to_end():
    rows = [
        {"db_id": "d1", "question": "list deps", "query": EASY},
        {"db_id": "d2", "question": "bad", "query": "DELETE FROM x"},
        {"db_id": "d1", "question": "list deps", "query": EASY},
        {"db_id": "d3", "question": "agg", "query": HARD},
    ]
    out = filter_dataset(rows, per_bucket=10, seed=1)
    assert len(out) == 2
    assert all(not is_rejected(r["query"]) for r in out)


def test_corrupt_sql_is_broken():
    import random

    rng = random.Random(0)
    assert corrupt_sql(EASY, rng) != EASY


def test_attach_repair_rows():
    rows = [{"db_id": "d1", "question": "q", "query": EASY}]
    out = attach_repair_rows(rows, fraction=1.0, seed=0)
    assert len(out) == 2
    assert any("repair_target" in r for r in out)