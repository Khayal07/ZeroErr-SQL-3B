"""Dataset filtering, deduplication, and difficulty balancing for Spider/BIRD."""

from __future__ import annotations

import random
import re

MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 320
MAX_SQL_CHARS = 900

_DIFFICULTY_KEYWORDS = [
    "where",
    "join",
    "group by",
    "having",
    "order by",
    "limit",
    "union",
    "intersect",
    "except",
    "distinct",
    "exists",
    "case when",
]

_DIFFICULTY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _DIFFICULTY_KEYWORDS) + r")\b", re.IGNORECASE
)


def difficulty_bucket(sql: str) -> str:
    """Classify a query as easy / medium / hard based on structural complexity.

    easy:     no clauses beyond a bare SELECT
    medium:   a few filtering/clause keywords
    hard:     joins + several clauses or nested subqueries
    """
    lower = sql.lower()
    clauses = {kw for kw in _DIFFICULTY_KEYWORDS if kw in lower}
    has_join = "join" in lower
    subqueries = max(len(re.findall(r"\bselect\b", lower)) - 1, 0)

    score = 0
    if has_join:
        score += 2
    if subqueries:
        score += 2 * subqueries
    score += min(len(clauses), 3)

    if score == 0:
        return "easy"
    if score < 4 and subqueries < 2:
        return "medium"
    return "hard"


def is_rejected(sql: str) -> bool:
    """Reject anything that is not a read-only SELECT/WITH statement."""
    head = sql.strip().lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        return True
    return bool(re.search(r"\b(insert\b|update\b|delete\b|drop\b|alter\b|create\b|attach\b|pragma\b)", sql.lower()))


def is_too_long(question: str, sql: str) -> bool:
    return len(question) > MAX_QUESTION_CHARS or len(sql) > MAX_SQL_CHARS


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in rows:
        key = (row.get("db_id", ""), row.get("question", "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def balance_by_difficulty(rows: list[dict], per_bucket: int = 2000, seed: int = 42) -> list[dict]:
    """Cap each difficulty bucket (easy/medium/hard) to ``per_bucket`` samples."""
    buckets: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for row in rows:
        buckets[difficulty_bucket(row.get("query", ""))].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for bucket in ("easy", "medium", "hard"):
        items = buckets[bucket]
        if len(items) > per_bucket:
            items = rng.sample(items, per_bucket)
        selected.extend(items)
    return selected


def corrupt_sql(sql: str, rng: random.Random) -> str:
    """Inject a single realistic error so repair-pair training data is exercisable."""
    mutations = [
        re.sub(r"\bFROM\b", "FRM", sql, count=1, flags=re.IGNORECASE),
        re.sub(r"\bWHERE\b", "WHRE", sql, count=1, flags=re.IGNORECASE),
        sql.replace("ORDER BY", "ORDRE BY", 1),
        sql.replace("GROUP BY", "GROUP", 1),
    ]
    applied = [m for m in mutations if m != sql] or mutations
    return rng.choice(applied)


def attach_repair_rows(rows: list[dict], fraction: float = 0.1, seed: int = 42) -> list[dict]:
    """Duplicate up to ``fraction`` of rows into repair-pair examples that teach
    the model to fix broken SQL given an execution error."""
    rng = random.Random(seed)
    repairs: list[dict] = []
    for row in rows:
        if rng.random() > fraction:
            continue
        repairs.append(
            {
                "db_id": row.get("db_id", ""),
                "question": row["question"],
                "query": row["query"],
                "broken_sql": corrupt_sql(row["query"], rng),
                "repair_target": True,
            }
        )
    return rows + repairs


def filter_dataset(
    rows: list[dict],
    per_bucket: int = 2000,
    with_repairs: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    """End-to-end pipeline: reject unsafe rows, cap length, dedupe, balance buckets,
    optionally add repair pairs, shuffle."""
    keep = [
        row
        for row in rows
        if row.get("question") and row.get("query")
        and not is_rejected(row["query"])
        and not is_too_long(row.get("question", ""), row["query"])
    ]
    keep = dedupe(keep)
    keep = balance_by_difficulty(keep, per_bucket=per_bucket, seed=seed)
    if with_repairs:
        keep = attach_repair_rows(keep, seed=seed)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(keep)
    for row in keep:
        row.setdefault("db_id", "unknown")
    return keep