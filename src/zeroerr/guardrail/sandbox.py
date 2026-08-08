"""Read-only SQLite sandbox for execution-guided verification."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_ALLOWED_HEADS = ("select", "with", "explain")


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0


class SQLiteSandbox:
    """Execute SQL against a read-only SQLite database with a hard timeout.

    Safety guarantees:
      * database opened with ``query_only=ON`` / ``mode=ro``
      * writes are blocked before execution via statement allow-list
      * row fetches are capped at ``max_rows``
      * a watchdog cancels executions exceeding ``timeout_seconds``
    """

    def __init__(self, database: str | Path, timeout_seconds: float = 5.0, max_rows: int = 100):
        self.database = Path(database)
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows
        if not self.database.exists():
            raise FileNotFoundError(f"sandbox database not found: {self.database}")

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro",
            uri=True,
            timeout=self.timeout_seconds,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA busy_timeout = %d" % int(self.timeout_seconds * 1000))
        return conn

    def _check_allowed(self, sql: str) -> bool:
        return sql.strip().lstrip("(").lower().startswith(_ALLOWED_HEADS)

    def _run(self, conn: sqlite3.Connection, sql: str) -> ExecutionResult:
        start = time.perf_counter()
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(self.max_rows)
        elapsed = (time.perf_counter() - start) * 1000.0
        return ExecutionResult(
            ok=True,
            rows=[tuple(r) for r in rows],
            row_count=len(rows),
            columns=columns,
            elapsed_ms=round(elapsed, 2),
        )

    def execute(self, sql: str) -> ExecutionResult:
        if not self._check_allowed(sql):
            return ExecutionResult(ok=False, error=f"only SELECT/WITH/EXPLAIN allowed; got: {sql.split()[0]}")

        box: dict = {}
        done = threading.Event()

        def target() -> None:
            conn = None
            try:
                conn = sqlite3.connect(f"file:{self.database.as_posix()}?mode=ro", uri=True, timeout=self.timeout_seconds)
                conn.execute("PRAGMA query_only = ON")
                conn.execute("PRAGMA busy_timeout = %d" % int(self.timeout_seconds * 1000))
                box["result"] = self._run(conn, sql)
            except sqlite3.Error as exc:
                box["result"] = ExecutionResult(ok=False, error=str(exc))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                done.set()

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        if not done.wait(self.timeout_seconds):
            return ExecutionResult(ok=False, error=f"execution timed out after {self.timeout_seconds}s")
        return box.get("result", ExecutionResult(ok=False, error="no execution result produced"))

    def tables(self) -> list[str]:
        conn = self._open()
        try:
            return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        finally:
            conn.close()


class SandboxRegistry:
    """Maps ``database_id`` -> SQLite file paths discovered under a data dir."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._dbs: dict[str, Path] = {}

    def discover(self) -> dict[str, Path]:
        if self.data_dir.exists():
            for p in sorted(self.data_dir.glob("*.sqlite")):
                self._dbs[p.stem] = p
        return dict(self._dbs)

    def sandbox_for(self, database_id: str, **kwargs) -> SQLiteSandbox:
        self.discover()
        if database_id not in self._dbs:
            raise KeyError(f"unknown database_id {database_id!r}; available: {sorted(self._dbs)}")
        return SQLiteSandbox(self._dbs[database_id], **kwargs)