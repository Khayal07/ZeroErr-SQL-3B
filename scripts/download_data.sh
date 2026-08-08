#!/usr/bin/env bash
# Download Spider (+ optionally BIRD-SQL dev) into data/raw.
set -euo pipefail

RAW_DATA="data/raw"
mkdir -p "$RAW_DATA"

echo "[zeroerr] downloading Spider train via the Hugging Face hub..."

python - "$RAW_DATA" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

from datasets import load_dataset

ds = load_dataset("xlangai/spider", split="train")
with (out / "spider_train.jsonl").open("w", encoding="utf-8") as fh:
    for r in ds:
        fh.write(json.dumps({"db_id": r["db_id"], "question": r["question"], "query": r["query"]}) + "\n")
print(f"spider: wrote {(out / 'spider_train.jsonl')}")
PY

cat <<'EOF'
BIRD-SQL tip: the official dev set (questions + gold.sql + database/.sqlite)
must be downloaded from https://bird-bench.github.io/ after accepting the terms.
Drop the raw dev jsons into data/raw/bird/ and re-run the prep CLI.
EOF