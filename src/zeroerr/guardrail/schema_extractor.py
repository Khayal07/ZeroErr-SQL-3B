"""Schema extraction from a live SQLite file."""

from __future__ import annotations

from sqlite3 import Connection

from zeroerr.data.schemas import Column, Schema


def extract_sqlite_schema(conn: Connection) -> Schema:
    """Introspect a SQLite connection into a normalized :class:`Schema`."""
    db_id = "sqlite"
    tables: dict[str, list[Column]] = {}

    for name in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        table = name[0]
        cols = []
        for pr in conn.execute(f'PRAGMA table_info("{table}")'):
            cols.append(Column(name=pr[1], data_type=(pr[2] or "TEXT")))
        tables[table] = cols

    foreign_keys: list[tuple[str, str, str, str]] = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            foreign_keys.append((table, fk[3], fk[2], fk[4]))

    return Schema(db_id=db_id, tables=tables, foreign_keys=foreign_keys)


def schema_from_sandbox(sqlite_sandbox) -> Schema:
    """Convenience wrapper around :func:`extract_sqlite_schema` for a sandbox object."""
    conn = sqlite_sandbox._open()
    try:
        return extract_sqlite_schema(conn)
    finally:
        conn.close()