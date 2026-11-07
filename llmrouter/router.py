"""The router: choose a model, respect a budget, and explain the choice.

Three guarantees this module makes:

1. **Never exceed the budget.** If the chosen model does not fit, the router
   either degrades to something that does — and records that it did, so the
   downgrade is visible in `Decision.notes` — or raises if nothing is
   affordable. Pass `on_budget_pressure="raise"` to forbid degradation
   entirely.
2. **Every decision is explainable.** "Why did this go to the cheap model?" is
   the first question asked when quality drops, and it must have an answer
   better than "the heuristic said so".
3. **Every decision has a fallback chain.** A provider outage should not take
   the feature down; it should move one tier up and keep going.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .classify import Classification, Request, classify
from .models import DEFAULT_CATALOG, ModelSpec, cheapest, catalog_by_name, tier_rank


class BudgetExceeded(Exception):
    """The remaining budget cannot cover any model that meets the constraints."""

    def __init__(self, message: str, *, remaining: float, needed: float) -> None:
        super().__init__(message)
        self.remaining = remaining
        self.needed = needed


class NoModelFound(Exception):
    """No model in the catalogue satisfies the hard constraints."""


@dataclass
class Budget:
    """A spending cap, in USD.

    Use one per user request, per job, or per day — the router does not care,
    it only tracks a remaining balance. Thread-safety is the caller's problem,
    which is the honest contract for a library that does not own your scheduler.
    """

    limit: float
    spent: float = 0.0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("budget limit cannot be negative")

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent)

    def check(self, amount: float) -> bool:
        return amount <= self.remaining + 1e-12

    def charge(self, amount: float) -> None:
        self.spent += max(0.0, amount)

    def __str__(self) -> str:
        return f"${self.spent:.6f} / ${self.limit:.6f}"


@dataclass
class Decision:
    model: ModelSpec
    classification: Classification
    estimated_cost: float
    tokens_in: int
    tokens_out: int
    fallbacks: List[ModelSpec] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"model      {self.model.name} ({self.model.tier})",
            f"task       {self.classification.task}",
            f"cost       ${self.estimated_cost:.6f} "
            f"({self.tokens_in} in / {self.tokens_out} out)",
            f"reasons    {'; '.join(self.classification.reasons)}",
        ]
        if self.fallbacks:
            lines.append(f"fallbacks  {', '.join(m.name for m in self.fallbacks)}")
        for note in self.notes:
            lines.append(f"note       {note}")
        return "\n".join(lines)


class Router:
    def __init__(
        self,
        catalog: Optional[Sequence[ModelSpec]] = None,
        *,
        budget: Optional[Budget] = None,
        max_cost_per_call: Optional[float] = None,
        registry: Optional[Dict[str, ModelSpec]] = None,
        on_budget_pressure: str = "downgrade",
    ) -> None:
        if on_budget_pressure not in {"downgrade", "raise"}:
            raise ValueError("on_budget_pressure must be 'downgrade' or 'raise'")
        self.catalog: List[ModelSpec] = list(catalog or DEFAULT_CATALOG)
        if not self.catalog:
            raise ValueError("catalog cannot be empty")
        self.budget = budget
        self.max_cost_per_call = max_cost_per_call
        self.on_budget_pressure = on_budget_pressure
        # Optional per-name override, used when a caller pins a model.
        self._by_name = registry or catalog_by_name(self.catalog)

    # ------------------------------------------------------------- routing --

    def route(self, request: Request, *, force: Optional[str] = None) -> Decision:
        """Choose a model for `request`.

        `force` pins a specific catalogue entry — useful for A/B tests and for
        callers that already know what they want.
        """
        classification = classify(request)
        tokens_out = request.expected_output_tokens
        notes: List[str] = []

        if force:
            model = self._by_name.get(force)
            if model is None:
                raise NoModelFound(f"forced model {force!r} is not in the catalogue")
            notes.append("pinned by caller; quality floor not applied")
        else:
            model = self._select(classification, request, tokens_out, notes)

        cost = model.estimate(request.tokens_in, tokens_out)

        # Second gate: the budget may still be too small for a model that is
        # otherwise eligible.
        if self.budget is not None and not self.budget.check(cost):
            if self.on_budget_pressure == "raise":
                cheapest_cost = self._cheapest_eligible_cost(
                    classification, request, tokens_out
                )
                raise BudgetExceeded(
                    f"remaining budget ${self.budget.remaining:.6f} cannot cover "
                    f"{model.name} (${cost:.6f}); cheapest eligible costs "
                    f"${cheapest_cost:.6f}",
                    remaining=self.budget.remaining,
                    needed=cheapest_cost,
                )

            downgraded, downgrade_cost, cheapest_cost = self._affordable(
                classification, request, tokens_out
            )
            if downgraded is None:
                raise BudgetExceeded(
                    f"remaining budget ${self.budget.remaining:.6f} cannot cover "
                    f"any model meeting constraints (cheapest eligible costs "
                    f"${cheapest_cost:.6f})",
                    remaining=self.budget.remaining,
                    needed=cheapest_cost,
                )
            notes.append(
                f"downgraded from {model.name} (${cost:.6f}) to {downgraded.name} "
                f"(${downgrade_cost:.6f}) to stay within budget"
            )
            model, cost = downgraded, downgrade_cost

        if self.max_cost_per_call is not None and cost > self.max_cost_per_call:
            raise BudgetExceeded(
                f"estimated ${cost:.6f} exceeds per-call cap "
                f"${self.max_cost_per_call:.6f}",
                remaining=self.max_cost_per_call,
                needed=cost,
            )

        return Decision(
            model=model,
            classification=classification,
            estimated_cost=cost,
            tokens_in=request.tokens_in,
            tokens_out=tokens_out,
            fallbacks=self._fallbacks(model, classification, request, tokens_out),
            notes=notes,
        )

    def _select(
        self,
        classification: Classification,
        request: Request,
        tokens_out: int,
        notes: List[str],
    ) -> ModelSpec:
        model = cheapest(
            self.catalog,
            min_tier=classification.min_tier,
            capabilities=classification.capabilities,
            min_context=0,
        )
        if model is not None and model.context_window < request.tokens_in:
            # The classifier's floor did not imply a big enough window; retry
            # with the window as a hard constraint.
            model = cheapest(
                self.catalog,
                min_tier=classification.min_tier,
                capabilities=classification.capabilities,
                min_context=request.tokens_in,
            )
            if model is not None:
                notes.append(
                    f"raised context requirement to {request.tokens_in} tokens"
                )
        if model is None:
            raise NoModelFound(
                "no model satisfies "
                f"tier>={classification.min_tier}, "
                f"capabilities={sorted(classification.capabilities)}, "
                f"context>={request.tokens_in}"
            )
        return model

    def _meets_hard_constraints(
        self, classification: Classification, request: Request
    ) -> List[ModelSpec]:
        """Models satisfying capabilities and context — the survival filter."""
        return [
            spec
            for spec in self.catalog
            if spec.supports(classification.capabilities)
            and spec.context_window >= request.tokens_in
        ]

    def _eligible(
        self, classification: Classification, request: Request
    ) -> List[ModelSpec]:
        """Hard constraints *and* the quality floor.

        "Eligible" must mean eligible: a model below the quality floor is not a
        valid answer for this request, so reporting its price as the cheapest
        available would be misleading in an error message.
        """
        floor = tier_rank(classification.min_tier)
        return [
            spec
            for spec in self._meets_hard_constraints(classification, request)
            if spec.rank >= floor
        ]

    def _cheapest_eligible_cost(
        self, classification: Classification, request: Request, tokens_out: int
    ) -> float:
        candidates = self._eligible(classification, request)
        if not candidates:
            return 0.0
        return min(spec.estimate(request.tokens_in, tokens_out) for spec in candidates)

    def _affordable(
        self,
        classification: Classification,
        request: Request,
        tokens_out: int,
    ) -> tuple[Optional[ModelSpec], float, float]:
        """Cheapest model that fits the remaining budget.

        Deliberately drops the quality floor: graceful degradation to a cheaper
        tier is the point of a budget. The downgrade is recorded in
        `Decision.notes` so it is never silent.

        Returns `(model, cost, cheapest_eligible_cost)` so an error can report
        the price of the cheapest model that actually meets the request.
        """
        if self.budget is None:
            return None, 0.0, 0.0
        candidates = sorted(
            self._meets_hard_constraints(classification, request),
            key=lambda s: s.estimate(request.tokens_in, tokens_out),
        )
        if not candidates:
            return None, 0.0, 0.0
        floor_cost = self._cheapest_eligible_cost(classification, request, tokens_out)
        for spec in candidates:
            cost = spec.estimate(request.tokens_in, tokens_out)
            if self.budget.check(cost):
                return spec, cost, floor_cost
        return None, 0.0, floor_cost

    def _fallbacks(
        self,
        primary: ModelSpec,
        classification: Classification,
        request: Request,
        tokens_out: int,
    ) -> List[ModelSpec]:
        """Ordered alternates: same tier first, then up, then down.

        Up before down is deliberate. A silent downgrade on retry is how a
        transient outage becomes a quality incident nobody traces back.
        """
        eligible = [
            spec
            for spec in self.catalog
            if spec.name != primary.name
            and spec.supports(classification.capabilities)
            and spec.context_window >= request.tokens_in
        ]
        eligible.sort(
            key=lambda s: (
                abs(s.rank - primary.rank),
                s.rank < primary.rank,
                s.estimate(request.tokens_in, tokens_out),
            )
        )
        return eligible[:3]

    # ---------------------------------------------------------- accounting --

    def record(self, decision: Decision, *, tokens_in: Optional[int] = None,
               tokens_out: Optional[int] = None) -> float:
        """Charge actual usage. Returns the amount charged.

        Call this with the real token counts after the call completes. Recording
        the *estimate* instead is how budgets drift: output length varies far
        more than input length does.
        """
        actual_in = tokens_in if tokens_in is not None else decision.tokens_in
        actual_out = tokens_out if tokens_out is not None else decision.tokens_out
        amount = decision.model.estimate(actual_in, actual_out)
        if self.budget is not None:
            self.budget.charge(amount)
        return amount
