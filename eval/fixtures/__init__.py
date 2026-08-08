"""Fixtures for tests and the smoke benchmark (no network needed)."""

from __future__ import annotations

from pathlib import Path

from eval.fixtures.build_fixtures import FIXTURE_DIR, build_all

QUESTIONS = [
    (
        "department_emp",
        "List each department along with the average salary of its employees.",
        "SELECT d.name, AVG(e.salary) FROM department d LEFT JOIN employee e ON d.department_id = e.dept_id GROUP BY d.department_id",
    ),
    (
        "department_emp",
        "Count employees hired in or after 2020.",
        "SELECT COUNT(*) FROM employee WHERE hire_year >= 2020",
    ),
    (
        "department_emp",
        "What is the highest salary across all departments?",
        "SELECT MAX(salary) FROM employee",
    ),
]


def fixture_db_dir() -> Path:
    return FIXTURE_DIR