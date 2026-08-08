"""Execution accuracy (EX) and valid efficiency score (VES) metrics."""

from __future__ import annotations

from zeroerr.guardrail.sandbox import ExecutionResult, SQLiteSandbox


def normalize_result(result: ExecutionResult) -> list[tuple]:
    """Sort rows for result-set comparison, ignoring row order."""
    return sorted(result.rows)


def correct(pred_sql: str, gold_sql: str, sandbox: SQLiteSandbox) -> bool:
    """Execution accuracy for a single pair: both run and produce equal result sets."""
    pred = sandbox.execute(pred_sql)
    gold = sandbox.execute(gold_sql)
    if not pred.ok or not gold.ok:
        return False
    return normalize_result(pred) == normalize_result(gold)


def ex_accuracy(pairs: list[tuple[str, str]], sandbox: SQLiteSandbox) -> float:
    """Fraction of (pred, gold) pairs yielding identical result sets."""
    if not pairs:
        return 0.0
    return sum(1 for p, g in pairs if correct(p, g, sandbox)) / len(pairs)


def ves_score(pred: ExecutionResult, gold: ExecutionResult) -> float:
    """BIRD valid-efficiency score: correctness x (gold_time / pred_time), capped at 1."""
    if pred.elapsed_ms <= 1.0 or gold.elapsed_ms <= 1.0:
        return 0.0
    if normalize_result(pred) != normalize_result(gold):
        return 0.0
    return min(1.0, gold.elapsed_ms / pred.elapsed_ms)