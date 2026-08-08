"""Build small SQLite fixture databases used by tests and demo evaluation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / ".." / "fixtures" / "db"

_DATABASES: dict[str, list[str]] = {
    "department_emp": [
        "CREATE TABLE department (department_id INTEGER PRIMARY KEY, name TEXT, location TEXT)",
        "CREATE TABLE employee (employee_id INTEGER PRIMARY KEY, name TEXT, salary REAL, dept_id INTEGER REFERENCES department(department_id), hire_year INTEGER)",
        "INSERT INTO department VALUES (1, 'Engineering', 'New York'), (2, 'Sales', 'Chicago'), (3, 'Marketing', 'New York')",
        "INSERT INTO employee VALUES (1, 'Alice', 90000, 1, 2020), (2, 'Bob', 75000, 2, 2019), (3, 'Carol', 85000, 1, 2021), (4, 'Dave', 70000, 3, 2018), (5, 'Eve', 95000, NULL, 2022)",
    ]
}


def build_all(target_dir: str | Path = FIXTURE_DIR) -> list[Path]:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for name, statements in _DATABASES.items():
        path = target / f"{name}.sqlite"
        conn = sqlite3.connect(path)
        try:
            for sql in statements:
                conn.execute(sql)
            conn.commit()
        finally:
            conn.close()
        created.append(path)
    return created


if __name__ == "__main__":
    for p in build_all():
        print(f"created {p}")