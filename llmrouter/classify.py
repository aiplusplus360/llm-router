"""Classify a request into a quality floor and its hard constraints.

Two things are being computed here and they are not the same:

  **hard constraints**  what the model *must* support — context length, tools,
                        vision, JSON mode. Violating one produces an outright
                        failure, so these are filters, not scores.

  **quality floor**     how much model the task *deserves*. Missing this does
                        not error; it silently produces worse output, which is
                        the failure mode nobody notices until a customer does.

Everything is heuristic and deterministic. A router that calls a model to
decide which model to call is a router that doubles your latency and cost to
save money. These rules can be inspected, argued with, and unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple

NANO, SMALL, MEDIUM, LARGE = "nano", "small", "medium", "large"

# ~4 characters per token is the standard English approximation and is close
# enough for a routing decision. This is deliberately not a tokenizer: pulling
# one in would make the install heavyweight to improve a number we then
# multiply by a price estimate that is itself approximate.
CHARS_PER_TOKEN = 4


@dataclass
class Request:
    prompt: str
    # Optional explicit signals — a caller usually knows more than a heuristic.
    requires: Set[str] = field(default_factory=set)
    task: str = ""
    expected_output_tokens: int = 400

    @property
    def tokens_in(self) -> int:
        return max(1, len(self.prompt) // CHARS_PER_TOKEN)


@dataclass
class Classification:
    task: str
    min_tier: str
    capabilities: Set[str]
    reasons: List[str]
    complexity: int

    def explain(self) -> str:
        return f"{self.task} -> {self.min_tier} ({'; '.join(self.reasons)})"


# Signals that a task needs reasoning rather than pattern-matching. Ordered
# strongest-first so the reason string names the most specific match.
_COMPLEXITY_SIGNALS: List[Tuple[str, int, "re.Pattern[str]"]] = [
    (
        "multi-step reasoning",
        3,
        re.compile(
            r"\b(step[- ]by[- ]step|chain of thought|prove|derive|"
            r"first .* then .* (finally|then)|reason(ing)? about)\b",
            re.I,
        ),
    ),
    (
        "code synthesis",
        3,
        re.compile(
            r"\b(implement|refactor|optimi[sz]e|debug|fix the bug|write (a|an) "
            r"(function|class|module|script|test)|algorithm)\b",
            re.I,
        ),
    ),
    (
        "mathematical work",
        2,
        re.compile(r"\b(calculat|comput|solve|equation|probability|integral|matrix)\w*", re.I),
    ),
    (
        "structured extraction",
        1,
        re.compile(r"\b(extract|parse|classify|categor|normalis|normaliz)\w*", re.I),
    ),
    (
        "summarisation",
        1,
        re.compile(r"\b(summari[sz]e|tl;?dr|condense|shorten)\b", re.I),
    ),
]

_TASK_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    (
        "code",
        re.compile(
            r"```|\bdef |\bclass |\bfunction\b|\bimport \w+|\bSELECT\b|"
            # A request to *write* code contains no code — matching only code
            # syntax classified "implement a quicksort" as chat.
            r"\b(implement|refactor|debug|write (a|an|the) "
            r"(function|class|module|script|test|program))\b",
            re.I,
        ),
    ),
    ("reasoning", re.compile(r"\b(why|explain|analyse|analyze|compare|trade-?offs?)\b", re.I)),
    ("extraction", re.compile(r"\b(extract|parse|return (the )?json|as json)\b", re.I)),
    ("summarisation", re.compile(r"\b(summari[sz]e|tl;?dr|condense)\b", re.I)),
    ("chat", re.compile(r".", re.S)),
]

_LONG_INPUT_TOKENS = 24_000


def classify(request: Request) -> Classification:
    prompt = request.prompt
    reasons: List[str] = []
    capabilities: Set[str] = set(request.requires)

    # ---- task -------------------------------------------------------------
    task = request.task
    if not task:
        for candidate, pattern in _TASK_PATTERNS:
            if pattern.search(prompt):
                task = candidate
                break
    if not task:
        task = "chat"
    reasons.append(f"task={task}")

    # Hard capability implications of the task itself.
    if task == "code":
        capabilities.add("chat")
    if task == "extraction":
        capabilities.add("json")
        reasons.append("extraction needs JSON mode")

    # ---- complexity -------------------------------------------------------
    complexity = 0
    for label, weight, pattern in _COMPLEXITY_SIGNALS:
        if pattern.search(prompt):
            complexity = max(complexity, weight)
            reasons.append(f"{label} (+{weight})")

    # Length is its own signal: a long context needs a model that can hold
    # enough of it in attention to reason across it, not just fit it.
    tokens = request.tokens_in
    if tokens > _LONG_INPUT_TOKENS:
        complexity = max(complexity, 2)
        capabilities.add("long-context")
        reasons.append(f"long input ~{tokens:,} tokens")
    elif tokens > 8_000:
        complexity += 1
        reasons.append(f"large input ~{tokens:,} tokens")

    # ---- quality floor ----------------------------------------------------
    if complexity >= 3:
        min_tier = LARGE
    elif complexity == 2:
        min_tier = MEDIUM
    elif complexity == 1:
        min_tier = SMALL
    else:
        min_tier = NANO

    # An explicit capability requirement can raise the floor on its own.
    if "vision" in capabilities and min_tier == NANO:
        min_tier = MEDIUM
        reasons.append("vision requires medium+")
    if "tools" in capabilities and min_tier == NANO:
        min_tier = SMALL
        reasons.append("tool use requires small+")

    reasons.append(f"complexity={complexity}")

    return Classification(
        task=task,
        min_tier=min_tier,
        capabilities=capabilities,
        reasons=reasons,
        complexity=complexity,
    )
