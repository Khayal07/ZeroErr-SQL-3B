"""Fetch Spider train/dev Q&A pairs with their DDL schemas into data/raw.

Sources:
    - philikai/SQL_Spider_DDL   (full 8.6k train split, includes DDL schema per db)
    - xlangai/spider            (validation split for dev questions)

Usage:
    .venv/bin/python scripts/download_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from datasets import load_dataset

    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    ddl = load_dataset("philikai/SQL_Spider_DDL", split="train")
    schema_map: dict[str, str] = {}
    train_rows: list[dict] = []
    for r in ddl:
        schema_map.setdefault(r["db_id"], r["DDL_schema"])
        train_rows.append(
            {"db_id": r["db_id"], "question": r["question"], "query": r["query"], "schema_text": r["DDL_schema"]}
        )
    _write_jsonl(out / "spider_train.jsonl", train_rows)

    try:
        dev = load_dataset("xlangai/spider", split="validation")
    except FileNotFoundError:
        dev = None
    if dev is not None:
        dev_rows = [
            {
                "db_id": r["db_id"],
                "question": r["question"],
                "query": r["query"],
                "schema_text": schema_map.get(r["db_id"], ""),
            }
            for r in dev
        ]
        _write_jsonl(out / "spider_dev.jsonl", dev_rows)

    print(f"train: {len(train_rows)} rows -> data/raw/spider_train.jsonl")
    print(f"dev: {0 if dev is None else len(dev_rows)} rows -> data/raw/spider_dev.jsonl")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())