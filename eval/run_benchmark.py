"""CLI benchmark: Base vs Fine-Tuned model, Execution Accuracy + VES.

Examples:
    python eval/run_benchmark.py --smoke
    python eval/fixtures/build_fixtures.py
    python eval/run_benchmark.py --data data/chatml/dev.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from zeroerr.data.schemas import render_ddl
from zeroerr.engine.ollama import OllamaEngine
from zeroerr.guardrail.orchestrator import GuardrailLoop
from zeroerr.guardrail.sandbox import SQLiteSandbox, SandboxRegistry
from zeroerr.guardrail.schema_extractor import schema_from_sandbox

from eval.exec_accuracy import ves_score as _ves


@dataclass
class BenchRow:
    question: str
    db_id: str
    gold: str | None
    pred: str | None = None
    correct: bool = False
    rounds: int = 0
    elapsed_ms: float = 0.0
    ves: float = 0.0
    error: str | None = None


def load_dataset(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def local_run(
    engine,
    questions: list[tuple[str, str, str | None]],
    sandbox_dir: str | Path,
    max_rounds: int = 3,
) -> list[BenchRow]:
    registry = SandboxRegistry(sandbox_dir)
    available = registry.discover()
    out: list[BenchRow] = []
    for db_id, question, gold in questions:
        if db_id not in available:
            continue
        sandbox = available[db_id]
        sb = SQLiteSandbox(sandbox)
        schema = schema_from_sandbox(sb)
        schema.db_id = db_id
        loop = GuardrailLoop(sb, engine, max_rounds=max_rounds)
        t0 = time.perf_counter()
        result = loop.run(render_ddl(schema), question)
        elapsed = (time.perf_counter() - t0) * 1000.0

        correct = False
        ves = 0.0
        if result.ok:
            pred_res = sb.execute(result.sql)
            if gold:
                gold_res = sb.execute(gold)
                correct = pred_res.ok and gold_res.ok and sorted(pred_res.rows) == sorted(gold_res.rows)
                if correct:
                    ves = _ves(pred_res, gold_res)
            else:
                correct = pred_res.ok
        out.append(
            BenchRow(
                db_id=db_id,
                question=question,
                gold=gold,
                pred=result.sql,
                correct=correct,
                rounds=result.rounds,
                elapsed_ms=round(elapsed, 1),
                ves=round(ves, 3),
                error=result.error,
            )
        )
    return out

def write_csv(tag: str, rows: list[BenchRow], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["db_id", "question", "gold", "pred", "correct", "rounds", "elapsed_ms", "ves"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)
    return path


def summarize(tag: str, rows: list[BenchRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"tag": tag, "n": 0}
    correct = sum(r.correct for r in rows)
    return {
        "tag": tag,
        "n": n,
        "exec_accuracy": round(correct / n, 4),
        "avg_rounds": round(sum(r.rounds for r in rows) / n, 2),
        "avg_ms": round(sum(r.elapsed_ms for r in rows) / n, 1),
        "avg_ves": round(sum(r.ves for r in rows) / n, 4),
        "unresolved": sum(1 for r in rows if r.error),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark base vs fine-tuned model.")
    parser.add_argument("--smoke", action="store_true", help="run against bundled sqlite fixtures")
    parser.add_argument("--data", default=None, help="path to ChatML/Spider jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--model-base", default="qwen2.5:3b-instruct")
    parser.add_argument("--model-ft", default="zeroerr:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--sandbox-dir", default="data/databases")
    parser.add_argument("--no-base", action="store_true")
    args = parser.parse_args()

    questions: list[dict] = []
    sandbox_dir = Path(args.sandbox_dir)
    if args.smoke:
        from eval.fixtures import QUESTIONS, build_all

        build_all(sandbox_dir)
        questions = [
            {"db_id": db_id, "question": question, "query": gold}
            for db_id, question, gold in QUESTIONS
        ][: args.limit]
    elif args.data:
        questions = load_dataset(args.data)[: args.limit]
    else:
        parser.error("provide --smoke or --data")

    triples = []
    for item in questions:
        triples.append((item["db_id"], item["question"], item.get("query")))

    engines = {}
    if not args.no_base:
        engines["base"] = OllamaEngine(model=args.model_base, base_url=args.base_url)
    engines["ft"] = OllamaEngine(model=args.model_ft, base_url=args.base_url)

    summaries = []
    for tag, engine in engines.items():
        rows = local_run(engine, triples, sandbox_dir, max_rounds=args.max_rounds)
        path = write_csv(tag, rows, args.out)
        s = summarize(tag, rows)
        s["csv"] = str(path)
        summaries.append(s)
        print(f"{tag}: EX={s['exec_accuracy']} n={s['n']} avg_rounds={s['avg_rounds']} avg_ms={s['avg_ms']} VES={s['avg_ves']}")

    if not args.no_base and len(summaries) == 2:
        ft, base = summaries
        print(f"delta exec-accuracy: {(ft['exec_accuracy'] - base['exec_accuracy']):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())