"""Schema data model and serialization into compact prompt-friendly text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Column:
    name: str
    data_type: str = "TEXT"


@dataclass
class Schema:
    db_id: str
    tables: dict[str, list[Column]] = field(default_factory=dict)
    foreign_keys: list[tuple[str, str, str, str]] = field(default_factory=list)
    primary_keys: list[tuple[str, str]] = field(default_factory=list)

    def column_names(self, table: str) -> list[str]:
        return [c.name for c in self.tables.get(table, [])]


def from_spider_json(raw: dict[str, Any]) -> Schema:
    """Load a Spider-style schema dict into a normalized :class:`Schema`.

    Supports both the modern ``tables``/``columns`` layout and the classic
    ``table_names_original`` + ``column_names_original`` index layout.
    """
    db_id = raw.get("db_id") or (raw.get("db_names") or [""])[0]

    if "tables" in raw:
        tables: dict[str, list[Column]] = {}
        for entry in raw["tables"]:
            tname = entry["name"]
            tables[tname] = [Column(c["name"], c.get("type", "TEXT")) for c in entry.get("columns", [])]
        fks = [(fk["table"], fk["column"], fk["ref_table"], fk["ref_column"]) for fk in raw.get("foreign_keys", [])]
        pks = [(pk.get("table", ""), pk["column"]) for pk in raw.get("primary_keys", [])]
        return Schema(db_id=db_id, tables=tables, foreign_keys=fks, primary_keys=pks)

    table_names = raw.get("table_names_original") or raw.get("table_names") or []
    columns = raw.get("column_names_original") or raw.get("column_names") or []
    column_types = raw.get("column_types") or [""] * len(columns)

    tables = {tname: [] for tname in table_names}
    for (tidx, cname), ctype in zip(columns, column_types):
        if tidx is None or tidx > len(table_names) - 1:
            continue
        tables[table_names[tidx]].append(Column(name=cname, data_type=ctype))

    fks: list[tuple[str, str, str, str]] = []
    for ft, fc, rt, rc in raw.get("foreign_keys", []):
        fks.append((table_names[ft], columns[fc][1], table_names[rt], columns[rc][1]))

    pks: list[tuple[str, str]] = []
    for table_id, col_id in raw.get("primary_keys", []):
        pks.append((table_names[table_id], columns[col_id][1]))

    return Schema(db_id=db_id, tables=tables, foreign_keys=fks, primary_keys=pks)


def render_ddl(schema: Schema) -> str:
    """Render a compact, model-friendly DDL string."""
    lines = [f"DB: {schema.db_id}"]
    for table, cols in schema.tables.items():
        col_repr = ", ".join(f"{c.name} {c.data_type}" for c in cols)
        lines.append(f"CREATE TABLE {table} ( {col_repr} )")
    if schema.foreign_keys:
        fk_lines = []
        for lt, lc, rt, rc in schema.foreign_keys:
            fk_lines.append(f"{lt}.{lc} -> {rt}.{rc}")
        lines.append("REFERENCES: " + "; ".join(fk_lines))
    if schema.primary_keys:
        pk = ", ".join(f"{t}.{c}" for t, c in schema.primary_keys)
        lines.append(f"PRIMARY KEY: {pk}")
    return "\n".join(lines)


def normalize_table_name(name: str) -> str:
    """Lowercase + strip non-word chars, used for FK/column sanity matching."""
    return re.sub(r"[^a-z0-9_]", "", name.lower())