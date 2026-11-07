"""Cost-aware model routing."""

__version__ = "0.1.0"

from .classify import Classification, Request, classify
from .models import DEFAULT_CATALOG, ModelSpec, cheapest, tier_rank
from .router import Budget, BudgetExceeded, Decision, NoModelFound, Router

__all__ = [
    "Budget",
    "BudgetExceeded",
    "Classification",
    "Decision",
    "DEFAULT_CATALOG",
    "ModelSpec",
    "NoModelFound",
    "Request",
    "Router",
    "cheapest",
    "classify",
    "tier_rank",
]
