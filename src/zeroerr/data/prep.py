"""CLI entrypoint: raw Spider/BIRD JSONL -> filtered ChatML training JSONL.

Usage:
    python -m zeroerr.data.prep --input data/spider_dev.jsonl --output data/chatml/dev.jsonl --per-bucket 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zeroerr.data.chatml import ChatExample, render_chatml
from zeroerr.data.filter import filter_dataset


def to_chatml_row(row: dict) -> dict:
    schema_text = row.get("schema_text", "")
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
    parser.add_argument("-i", "--input", required=True, help="raw jsonl: {db_id, question, query, schema?}")
    parser.add_argument("-o", "--out", required=True, help="output ChatML jsonl path")
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
            fh.write(json.dumps(to_chatml_row(row), ensure_ascii=False) + "\n")

    print(f"wrote {len(filtered)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())