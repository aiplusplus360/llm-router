"""CLI: choose a model for a prompt, and show the reasoning.

    llm-router route "summarise this" --budget 0.005
    llm-router compare "implement quicksort"
    llm-router classify "extract emails from this text"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .classify import Request, classify
from .models import DEFAULT_CATALOG
from .router import Budget, BudgetExceeded, NoModelFound, Router


def cmd_route(args: argparse.Namespace) -> int:
    budget = Budget(args.budget) if args.budget is not None else None
    router = Router(
        budget=budget,
        max_cost_per_call=args.max_cost,
        on_budget_pressure=args.on_budget_pressure,
    )
    request = Request(
        prompt=args.prompt,
        requires=set(args.require or ()),
        task=args.task or "",
        expected_output_tokens=args.output_tokens,
    )

    try:
        decision = router.route(request, force=args.force)
    except (BudgetExceeded, NoModelFound) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "model": decision.model.name,
                    "tier": decision.model.tier,
                    "task": decision.classification.task,
                    "estimated_cost": round(decision.estimated_cost, 8),
                    "tokens_in": decision.tokens_in,
                    "tokens_out": decision.tokens_out,
                    "fallbacks": [m.name for m in decision.fallbacks],
                    "reasons": decision.classification.reasons,
                    "notes": decision.notes,
                },
                indent=2,
            )
        )
    else:
        print(decision.explain())
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    result = classify(Request(prompt=args.prompt, requires=set(args.require or ())))
    print(result.explain())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Show what every model would cost for this prompt."""
    request = Request(
        prompt=args.prompt,
        requires=set(args.require or ()),
        expected_output_tokens=args.output_tokens,
    )
    result = classify(request)
    print(f"task {result.task}  floor {result.min_tier}  "
          f"({request.tokens_in} tokens in, {args.output_tokens} out)\n")
    print(f"{'model':<12} {'tier':<8} {'eligible':<9} {'cost':>14}")
    for spec in DEFAULT_CATALOG:
        from .models import tier_rank

        eligible = spec.rank >= tier_rank(result.min_tier) and spec.supports(
            result.capabilities
        )
        cost = spec.estimate(request.tokens_in, args.output_tokens)
        print(f"{spec.name:<12} {spec.tier:<8} {'yes' if eligible else 'no':<9} ${cost:>13.6f}")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    for spec in DEFAULT_CATALOG:
        caps = ",".join(sorted(spec.capabilities))
        print(
            f"{spec.name:<12} {spec.tier:<8} "
            f"in ${spec.input_per_1k:.5f}/1k  out ${spec.output_per_1k:.5f}/1k  "
            f"ctx {spec.context_window:>9,}  [{caps}]"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-router",
        description="Route prompts to the cheapest model that can do the job.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    route = sub.add_parser("route", help="choose a model")
    route.add_argument("prompt")
    route.add_argument("--budget", type=float, help="total budget in USD")
    route.add_argument("--max-cost", type=float, help="cap per call in USD")
    route.add_argument("--require", action="append", help="required capability")
    route.add_argument("--task", help="override detected task")
    route.add_argument("--force", help="pin a model by name")
    route.add_argument("--output-tokens", type=int, default=400)
    route.add_argument(
        "--on-budget-pressure",
        choices=["downgrade", "raise"],
        default="downgrade",
        help="what to do when the preferred model does not fit the budget",
    )
    route.add_argument("--json", action="store_true")
    route.set_defaults(func=cmd_route)

    cls = sub.add_parser("classify", help="show classification only")
    cls.add_argument("prompt")
    cls.add_argument("--require", action="append")
    cls.set_defaults(func=cmd_classify)

    compare = sub.add_parser("compare", help="cost every model for this prompt")
    compare.add_argument("prompt")
    compare.add_argument("--require", action="append")
    compare.add_argument("--output-tokens", type=int, default=400)
    compare.set_defaults(func=cmd_compare)

    catalog = sub.add_parser("catalog", help="list the model catalogue")
    catalog.set_defaults(func=cmd_catalog)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
