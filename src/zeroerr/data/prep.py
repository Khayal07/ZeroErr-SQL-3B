"""CLI entrypoint: raw Spider/BIRD JSONL -> filtered ChatML training JSONL.

Usage:
    python -m zeroerr.data.prep --input data/raw/spider_train.jsonl --output data/chatml/train.jsonl --schemas-dir data/databases --per-bucket 2000
"""

from __future__ import annotations

import argparse
import json
import random
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


def split_by_db(rows: list[dict], val_fraction: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split by db_id so no database appears in both splits (no leakage)."""
    db_ids = sorted({row["db_id"] for row in rows})
    rng = random.Random(seed)
    n_val = max(1, round(len(db_ids) * val_fraction))
    val_ids = set(rng.sample(db_ids, n_val))
    train = [row for row in rows if row["db_id"] not in val_ids]
    val = [row for row in rows if row["db_id"] in val_ids]
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter and format a raw Text-to-SQL dataset into ChatML.")
    parser.add_argument("-i", "--input", required=True, help="raw jsonl: {db_id, question, query}")
    parser.add_argument("-o", "--out", required=True, help="output ChatML jsonl path")
    parser.add_argument("--schemas-dir", default=None, help="dir of <db_id>.sqlite files used to inject schemas")
    parser.add_argument("--per-bucket", type=int, default=2000)
    parser.add_argument("--with-repairs", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.0, help="0..1 fraction of db_ids held out as validation")
    args = parser.parse_args()

    raw = []
    for line in Path(args.input).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            raw.append(json.loads(line))

    filtered = filter_dataset(raw, per_bucket=args.per_bucket, with_repairs=args.with_repairs)
    formatted = [to_chatml_row(row, args.schemas_dir) for row in filtered]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.val_fraction > 0:
        train_rows, val_rows = split_by_db(formatted, args.val_fraction)
        val_path = out_path.with_name(out_path.stem + ".val" + out_path.suffix)
        _write_jsonl(train_rows, out_path)
        _write_jsonl(val_rows, val_path)
        print(f"wrote {len(train_rows)} train rows -> {out_path}")
        print(f"wrote {len(val_rows)} val rows -> {val_path}")
    else:
        _write_jsonl(formatted, out_path)
        print(f"wrote {len(formatted)} rows -> {out_path}")
    return 0


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())