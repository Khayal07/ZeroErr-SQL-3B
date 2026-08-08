"""ChatML message construction and prompt templating for Qwen instruct models."""

from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "You are an expert SQL engineer. Generate valid {dialect} SQL that answers the "
    "question using ONLY the schema provided. Return a single SQL statement. "
    "Do not add explanations or markdown fences."
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class ChatExample:
    schema_text: str
    question: str
    answer: str

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": f"{SYSTEM_PROMPT.format(dialect='SQLite')}\n\nSchema:\n{self.schema_text}"},
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": self.answer},
        ]


def render_chatml(messages: list[dict[str, str]], with_assistant: bool = True) -> str:
    """Render a list of role/content dicts as a Qwen ChatML prompt string.

    When ``with_assistant`` is False the assistant start marker is appended so
    the model can begin generating its answer.
    """
    parts = []
    for msg in messages:
        parts.append(f"{IM_START}{msg['role']}\n{msg['content']}{IM_END}\n")
    if not with_assistant:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def build_generation_prompt(schema_text: str, question: str) -> str:
    """Prompt used at inference time (no assistant turn yet)."""
    return render_chatml(
        [
            {"role": "system", "content": f"{SYSTEM_PROMPT.format(dialect='SQLite')}\n\nSchema:\n{schema_text}"},
            {"role": "user", "content": question},
        ],
        with_assistant=False,
    )


def build_repair_prompt(schema_text: str, question: str, bad_sql: str, error_hint: str) -> str:
    """Prompt used by the guardrail loop when a generated query failed to execute."""
    system = (
        f"{SYSTEM_PROMPT.format(dialect='SQLite')}\n\nSchema:\n{schema_text}\n\n"
        "The SQL below failed to execute. Fix ONLY the error and return a complete "
        "correct SQL statement.\n\n"
        f"Said SQL:\n{bad_sql}\n\n"
        f"Error:\n{error_hint}"
    )
    return render_chatml(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        with_assistant=False,
    )


def extract_sql(raw: str) -> str:
    """Extract the final SQL statement from a model response.\n\nStrips markdown fences, leading hand-off copy and trailing semicolons."""
    text = raw.strip()
    for opener in ("```sql", "```"):
        if opener in text:
            text = text.split(opener, 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
            break
    first = text.lower().find("select")
    second = text.lower().find("with ")
    start = min(x for x in (first, second) if x >= 0) if (first >= 0 or second >= 0) else 0
    if start > 0 and any(ch in text[:start] for ch in ":="):
        text = text[start:]
    text = text.strip().rstrip(";")
    return text.strip()