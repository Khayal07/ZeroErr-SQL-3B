"""CLI entrypoint: raw Spider/BIRD JSONL -> filtered ChatML training JSONL.

Usage:
    python -m zeroerr.data.prep --input data/raw/spider_train.jsonl --output data/chatml/train.jsonl --schemas-dir data/databases --per-bucket 2000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from zeroerr.data.chatml import ChatExample, render_chatml
from zeroerr.data.filter import filter_dataset
from zeroerr.data.schemas import render_ddl
from zeroerr.guardrail.schema_extractor import extract_sqlite_schema


def schema_text_for(db_id: str, schemas_dir: str | None) -> str:
    if not schemas_dir:
        return ""
    path = Path(schemas_dir) / f"{db_id}.sqlite"
    if not path.exists():
        return ""
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        schema = extract_sqlite_schema(conn)
        schema.db_id = db_id
        return render_ddl(schema)
    finally:
        conn.close()


def to_chatml_row(row: dict, schemas_dir: str | None = None) -> dict:
    schema_text = row.get("schema_text") or schema_text_for(row.get("db_id", ""), schemas_dir)
    if "broken_sql" in row:
        system = (
            "You are an expert SQL debugger. Repair the broken SQL below using the "
            "error message. Return only a corrected SQLite statement.\n"
            f"Broken SQL:\n{row['broken_sql']}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["query"]},
        ]
    else:
        messages = ChatExample(schema_text=schema_text, question=row["question"], answer=row["query"]).to_messages()
    return {
        "db_id": row.get("db_id", ""),
        "question": row["question"],
        "query": row["query"],
        "text": render_chatml(messages),
        "plain": {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter and format a raw Text-to-SQL dataset into ChatML.")
    parser.add_argument("-i", "--input", required=True, help="raw jsonl: {db_id, question, query}")
    parser.add_argument("-o", "--out", required=True, help="output ChatML jsonl path")
    parser.add_argument("--schemas-dir", default=None, help="dir of <db_id>.sqlite files used to inject schemas")
    parser.add_argument("--per-bucket", type=int, default=2000)
    parser.add_argument("--with-repairs", action="store_true")
    args = parser.parse_args()

    raw = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            raw.append(json.loads(line))

    filtered = filter_dataset(raw, per_bucket=args.per_bucket, with_repairs=args.with_repairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in filtered:
            fh.write(json.dumps(to_chatml_row(row, args.schemas_dir), ensure_ascii=False) + "\n")

    print(f"wrote {len(filtered)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())