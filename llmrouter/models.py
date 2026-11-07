"""Model catalogue and cost arithmetic.

Routing decisions are only as good as the cost model behind them, so cost is
computed from a single place: `ModelSpec.estimate`. Every number the router
reports comes from here, which means a pricing change is one edit and not a
search across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Tiers, cheapest first. A router's job is to pick the lowest tier that can
# actually do the work — so the ordering is load-bearing, not cosmetic.
TIERS = ("nano", "small", "medium", "large")


def tier_rank(tier: str) -> int:
    try:
        return TIERS.index(tier)
    except ValueError:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}") from None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: str
    # USD per 1,000 tokens. Split in/out because output is typically 3-4x the
    # input price and dominates the bill on generation-heavy workloads.
    input_per_1k: float
    output_per_1k: float
    context_window: int
    capabilities: Set[str] = field(default_factory=set)
    # Rough p50 latency for a 500-token completion, in seconds. Used only to
    # break ties between models of equal cost.
    latency_s: float = 1.0

    def __post_init__(self) -> None:
        tier_rank(self.tier)  # validate eagerly
        if self.input_per_1k < 0 or self.output_per_1k < 0:
            raise ValueError(f"{self.name}: price cannot be negative")
        if self.context_window <= 0:
            raise ValueError(f"{self.name}: context_window must be positive")

    def estimate(self, tokens_in: int, tokens_out: int) -> float:
        """Cost in USD for one call."""
        return (
            tokens_in / 1000 * self.input_per_1k
            + tokens_out / 1000 * self.output_per_1k
        )

    def supports(self, required: Set[str]) -> bool:
        return required <= self.capabilities

    @property
    def rank(self) -> int:
        return tier_rank(self.tier)


# A plausible 2026 catalogue. Prices are illustrative but internally consistent:
# each tier is roughly 4x the previous one on both axes, which is the shape real
# price ladders have.
DEFAULT_CATALOG: List[ModelSpec] = [
    ModelSpec(
        name="nano-1",
        tier="nano",
        input_per_1k=0.00005,
        output_per_1k=0.0002,
        context_window=32_000,
        capabilities={"chat", "json"},
        latency_s=0.35,
    ),
    ModelSpec(
        name="small-1",
        tier="small",
        input_per_1k=0.00020,
        output_per_1k=0.0008,
        context_window=128_000,
        capabilities={"chat", "json", "tools"},
        latency_s=0.8,
    ),
    ModelSpec(
        name="medium-1",
        tier="medium",
        input_per_1k=0.00080,
        output_per_1k=0.0032,
        context_window=200_000,
        capabilities={"chat", "json", "tools", "vision"},
        latency_s=1.6,
    ),
    ModelSpec(
        name="large-1",
        tier="large",
        input_per_1k=0.00300,
        output_per_1k=0.0120,
        context_window=1_000_000,
        capabilities={"chat", "json", "tools", "vision", "long-context"},
        latency_s=3.0,
    ),
]


def catalog_by_name(catalog: Optional[List[ModelSpec]] = None) -> Dict[str, ModelSpec]:
    return {spec.name: spec for spec in (catalog or DEFAULT_CATALOG)}


def cheapest(
    catalog: List[ModelSpec],
    *,
    min_tier: str = "nano",
    capabilities: Optional[Set[str]] = None,
    min_context: int = 0,
) -> Optional[ModelSpec]:
    """Cheapest model meeting every hard constraint, or None.

    Tie-break on latency: two models at the same price are not equivalent to a
    user waiting on them.
    """
    required = capabilities or set()
    floor = tier_rank(min_tier)
    eligible = [
        spec
        for spec in catalog
        if spec.rank >= floor
        and spec.supports(required)
        and spec.context_window >= min_context
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda s: (s.input_per_1k + s.output_per_1k, s.latency_s))
