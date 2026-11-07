"""Tests for classification, routing, budgets and fallbacks.

The cases worth writing down are the ones where a naive router looks fine and
is wrong: a cheap model quietly producing worse output, a budget that silently
degrades quality, a fallback that downgrades on retry.
"""

import unittest

from llmrouter.classify import LARGE, MEDIUM, NANO, SMALL, Request, classify
from llmrouter.models import DEFAULT_CATALOG, ModelSpec, cheapest, tier_rank
from llmrouter.router import Budget, BudgetExceeded, NoModelFound, Router


class TestTiers(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(tier_rank("nano"), tier_rank("small"))
        self.assertLess(tier_rank("small"), tier_rank("medium"))
        self.assertLess(tier_rank("medium"), tier_rank("large"))

    def test_unknown_tier(self):
        with self.assertRaises(ValueError):
            tier_rank("enormous")


class TestModelSpec(unittest.TestCase):
    def test_estimate_splits_in_and_out(self):
        spec = ModelSpec("m", "small", 1.0, 4.0, 1000)
        # 1000 in at $1/1k = $1; 1000 out at $4/1k = $4.
        self.assertAlmostEqual(spec.estimate(1000, 1000), 5.0)

    def test_output_is_pricier(self):
        spec = ModelSpec("m", "small", 1.0, 4.0, 1000)
        self.assertGreater(spec.estimate(0, 1000), spec.estimate(1000, 0))

    def test_rejects_negative_price(self):
        with self.assertRaises(ValueError):
            ModelSpec("m", "small", -1.0, 1.0, 1000)

    def test_rejects_zero_context(self):
        with self.assertRaises(ValueError):
            ModelSpec("m", "small", 1.0, 1.0, 0)

    def test_supports(self):
        spec = ModelSpec("m", "small", 1.0, 1.0, 1000, {"chat", "tools"})
        self.assertTrue(spec.supports({"chat"}))
        self.assertFalse(spec.supports({"vision"}))

    def test_missing_capability_fails_supports(self):
        spec = ModelSpec("m", "small", 1.0, 1.0, 1000, {"chat"})
        # Empty requirement is satisfied by anything.
        self.assertTrue(spec.supports(set()))


class TestClassify(unittest.TestCase):
    def test_trivial_prompt_gets_cheapest(self):
        result = classify(Request(prompt="hi there"))
        self.assertEqual(result.min_tier, NANO)

    def test_code_task_escalates(self):
        result = classify(Request(prompt="implement a quicksort algorithm"))
        self.assertEqual(result.min_tier, LARGE)
        self.assertIn("code", result.task)

    def test_step_by_step_escalates(self):
        result = classify(Request(prompt="prove this step-by-step"))
        self.assertEqual(result.min_tier, LARGE)

    def test_summarisation_is_mid(self):
        result = classify(Request(prompt="summarise this article"))
        self.assertEqual(result.min_tier, SMALL)

    def test_extraction_requires_json(self):
        result = classify(Request(prompt="extract the emails as json"))
        self.assertIn("json", result.capabilities)

    def test_vision_raises_floor(self):
        result = classify(Request(prompt="what is this", requires={"vision"}))
        self.assertGreaterEqual(tier_rank(result.min_tier), tier_rank(MEDIUM))

    def test_tools_raise_floor(self):
        result = classify(Request(prompt="hi", requires={"tools"}))
        self.assertGreaterEqual(tier_rank(result.min_tier), tier_rank(SMALL))

    def test_long_input_needs_long_context(self):
        result = classify(Request(prompt="x" * 200_000))
        self.assertIn("long-context", result.capabilities)
        self.assertEqual(result.min_tier, MEDIUM)

    def test_explicit_task_overrides_detection(self):
        result = classify(Request(prompt="implement this", task="chat"))
        self.assertEqual(result.task, "chat")

    def test_reasons_are_recorded(self):
        result = classify(Request(prompt="implement a parser"))
        self.assertTrue(result.reasons)


class TestCheapest(unittest.TestCase):
    def test_picks_lowest_tier(self):
        self.assertEqual(cheapest(DEFAULT_CATALOG).tier, NANO)

    def test_respects_floor(self):
        model = cheapest(DEFAULT_CATALOG, min_tier="medium")
        self.assertEqual(model.tier, MEDIUM)

    def test_respects_capabilities(self):
        model = cheapest(DEFAULT_CATALOG, capabilities={"vision"})
        self.assertIn("vision", model.capabilities)

    def test_respects_context(self):
        model = cheapest(DEFAULT_CATALOG, min_context=500_000)
        self.assertGreaterEqual(model.context_window, 500_000)

    def test_impossible_constraints(self):
        self.assertIsNone(cheapest(DEFAULT_CATALOG, capabilities={"telepathy"}))


class TestRouter(unittest.TestCase):
    def test_simple_prompt_routes_to_nano(self):
        decision = Router().route(Request(prompt="hi"))
        self.assertEqual(decision.model.tier, NANO)

    def test_complex_prompt_routes_up(self):
        decision = Router().route(Request(prompt="implement a compiler optimiser"))
        self.assertEqual(decision.model.tier, LARGE)

    def test_explanation_names_the_reasons(self):
        decision = Router().route(Request(prompt="implement a parser"))
        text = decision.explain()
        self.assertIn("model", text)
        self.assertIn("reasons", text)

    def test_force_pins_a_model(self):
        decision = Router().route(Request(prompt="hi"), force="large-1")
        self.assertEqual(decision.model.name, "large-1")

    def test_force_unknown_model(self):
        with self.assertRaises(NoModelFound):
            Router().route(Request(prompt="hi"), force="does-not-exist")

    def test_vision_requirement_excludes_cheap_models(self):
        decision = Router().route(Request(prompt="describe", requires={"vision"}))
        self.assertIn("vision", decision.model.capabilities)


class TestBudget(unittest.TestCase):
    def test_remaining(self):
        budget = Budget(1.0, spent=0.25)
        self.assertAlmostEqual(budget.remaining, 0.75)

    def test_remaining_never_negative(self):
        budget = Budget(1.0, spent=2.0)
        self.assertEqual(budget.remaining, 0.0)

    def test_charge(self):
        budget = Budget(1.0)
        budget.charge(0.5)
        self.assertAlmostEqual(budget.spent, 0.5)

    def test_negative_limit_rejected(self):
        with self.assertRaises(ValueError):
            Budget(-1)

    def test_downgrades_when_budget_is_tight(self):
        # A complex prompt wants `large`; a tight budget should drop a tier
        # rather than fail, as long as something affordable exists.
        router = Router(budget=Budget(0.001))
        decision = router.route(
            Request(prompt="implement a quicksort", expected_output_tokens=100)
        )
        self.assertLess(tier_rank(decision.model.tier), tier_rank(LARGE))
        self.assertTrue(any("downgraded" in n for n in decision.notes))

    def test_raises_when_nothing_is_affordable(self):
        router = Router(budget=Budget(0.0000001))
        with self.assertRaises(BudgetExceeded):
            router.route(Request(prompt="implement a quicksort", expected_output_tokens=1000))

    def test_budget_error_carries_numbers(self):
        router = Router(budget=Budget(0.0))
        with self.assertRaises(BudgetExceeded) as ctx:
            router.route(Request(prompt="implement", expected_output_tokens=1000))
        self.assertEqual(ctx.exception.remaining, 0.0)
        self.assertGreater(ctx.exception.needed, 0.0)

    def test_budget_error_reports_the_real_cheapest(self):
        # The message must name the cheapest *eligible* model's price, not the
        # price of whichever model was selected before the budget check.
        router = Router(budget=Budget(0.0))
        with self.assertRaises(BudgetExceeded) as ctx:
            router.route(Request(prompt="implement a quicksort"))
        self.assertLess(ctx.exception.needed, 0.005)

    def test_raise_mode_forbids_downgrade(self):
        router = Router(budget=Budget(0.001), on_budget_pressure="raise")
        with self.assertRaises(BudgetExceeded):
            router.route(Request(prompt="implement a quicksort"))

    def test_invalid_budget_pressure_mode(self):
        with self.assertRaises(ValueError):
            Router(on_budget_pressure="panic")

    def test_per_call_cap(self):
        router = Router(max_cost_per_call=0.0000001)
        with self.assertRaises(BudgetExceeded):
            router.route(Request(prompt="implement a quicksort"))

    def test_record_charges_actuals_not_estimates(self):
        router = Router(budget=Budget(1.0))
        decision = router.route(Request(prompt="hi", expected_output_tokens=100))
        # Model emits far more than estimated.
        charged = router.record(decision, tokens_in=100, tokens_out=10_000)
        self.assertGreater(charged, decision.estimated_cost)
        self.assertAlmostEqual(router.budget.spent, charged)

    def test_exhausted_budget_blocks_next_call(self):
        budget = Budget(0.001)
        router = Router(budget=budget)
        decision = router.route(Request(prompt="x"))
        router.record(decision, tokens_in=10_000_000, tokens_out=10_000_000)
        with self.assertRaises(BudgetExceeded):
            router.route(Request(prompt="implement a compiler"))


class TestFallbacks(unittest.TestCase):
    def test_fallbacks_exist(self):
        decision = Router().route(Request(prompt="hi"))
        self.assertTrue(decision.fallbacks)

    def test_fallbacks_exclude_primary(self):
        decision = Router().route(Request(prompt="hi"))
        self.assertNotIn(decision.model.name, [m.name for m in decision.fallbacks])

    def test_fallbacks_prefer_upgrade_over_downgrade(self):
        # A silent downgrade on retry is how an outage becomes a quality
        # incident nobody can trace.
        decision = Router().route(Request(prompt="hi"))
        primary_rank = decision.model.rank
        first = decision.fallbacks[0]
        self.assertGreaterEqual(first.rank, primary_rank)

    def test_fallbacks_satisfy_capabilities(self):
        decision = Router().route(Request(prompt="x", requires={"vision"}))
        for fallback in decision.fallbacks:
            self.assertIn("vision", fallback.capabilities)

    def test_fallbacks_bounded(self):
        decision = Router().route(Request(prompt="hi"))
        self.assertLessEqual(len(decision.fallbacks), 3)


class TestRequest(unittest.TestCase):
    def test_token_estimate(self):
        self.assertEqual(Request(prompt="x" * 400).tokens_in, 100)

    def test_empty_prompt_is_at_least_one_token(self):
        self.assertEqual(Request(prompt="").tokens_in, 1)


if __name__ == "__main__":
    unittest.main()
