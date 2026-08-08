"""Turn raw SQLite error strings into concise, actionable repair hints."""

from __future__ import annotations

import re
from difflib import get_close_matches

_KNOWN_COLUMNS: list[str] = []


def _set_known_columns(columns: list[str]) -> None:
    global _KNOWN_COLUMNS
    _KNOWN_COLUMNS = columns


def _nearest(missing: str, candidates: list[str]) -> str | None:
    matches = get_close_matches(missing, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def hint_for_error(error: str, schema_columns: list[str] | None = None) -> str:
    """Map a SQLite error string to a compact human-readable repair hint."""
    candidates = schema_columns or _KNOWN_COLUMNS
    err = error or ""

    m = re.search(r"no such column:\s*(\S+)", err)
    if m:
        missing = m.group(1).strip()
        hint = f"column '{missing}' does not exist"
        if candidates:
            near = _nearest(missing, candidates)
            if near:
                hint += f"; did you mean '{near}'?"
        return hint

    m = re.search(r"no such table:\s*(\S+)", err)
    if m:
        return f"table '{m.group(1).strip()}' does not exist in the schema"

    if "ambiguous column name" in err:
        return "column is ambiguous; qualify it with its table name (e.g. t1.col)"

    m = re.search(r"syntax error near\s+[\"']?([^\"']*)", err)
    if m or "syntax error" in err:
        near = m.group(1).strip() if m and m.group(1) else "the offending token"
        return f"SQL syntax error near {near!r}; check keywords, commas and parentheses"

    if "incomplete input" in err:
        return "statement is incomplete; finish the SELECT/WITH clause"

    if "no such function" in err:
        m = re.search(r"no such function:\s*(\S+)", err)
        fn = m.group(1) if m else "function"
        return f"unknown SQL function '{fn}'; use a SQLite-supported function"

    if "table" in err and "already exists" in err:
        return "avoid CREATE/DROP statements; only SELECT/WITH is allowed"

    if "unable to open database" in err:
        return "sandbox database could not be opened"

    return err[:280]