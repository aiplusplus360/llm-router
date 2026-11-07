# llm-router

**Send each prompt to the cheapest model that can actually do the job — with a budget cap, a fallback chain, and a reason for every decision.**

[![CI](https://github.com/aiplusplus360/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/llm-router/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

```console
$ llm-router route "hi there, how are you?"
model      nano-1 (nano)
task       chat
cost       $0.000080 (5 in / 400 out)
reasons    task=chat; complexity=0
fallbacks  small-1, medium-1, large-1
```

```console
$ llm-router route "implement a lock-free ring buffer in C"
model      large-1 (large)
task       code
cost       $0.004827 (9 in / 400 out)
reasons    task=code; code synthesis (+3); complexity=3
fallbacks  medium-1, small-1, nano-1
```

Same library. **60× price difference**, chosen for a reason you can read.

---

## Why this exists

Most teams pick one model and send everything to it. That means either:

- **you overpay** — a "what's the capital of France" question goes to a frontier model, or
- **you underpay** — a multi-step refactor goes to the cheap tier and silently produces worse output

The second failure is the dangerous one. It doesn't throw an error. It shows up two weeks later as "the feature feels worse than it did."

A router fixes both, provided it can answer one question: **why did this go where it went?** Every `Decision` here carries its reasons, its cost, and its alternatives.

## Install

No dependencies.

```bash
pip install git+https://github.com/aiplusplus360/llm-router
```

## Usage

```python
from llmrouter import Budget, Request, Router

router = Router(budget=Budget(0.50))          # 50 cents for this job

decision = router.route(Request(prompt=user_text))
print(decision.explain())

response = call_model(decision.model.name, user_text)

# Charge the *actual* usage, not the estimate.
router.record(decision, tokens_in=response.usage.input,
              tokens_out=response.usage.output)
```

### Pinning, A/B tests, and overrides

```python
decision = router.route(request, force="large-1")   # bypass the quality floor
```

### Comparing the whole catalogue

```console
$ llm-router compare "extract every email from this document as json"
task extraction  floor small  (11 tokens in, 400 out)

model        tier     eligible            cost
nano-1       nano     no        $     0.000081
small-1      small    yes       $     0.000322
medium-1     medium   yes       $     0.001289
large-1      large    yes       $     0.004833
```

## How a decision is made

**1. Hard constraints — the survival filter.** Context window, required capabilities. Violating one is an outright failure, so these remove models rather than score them.

**2. Complexity — the quality floor.** Deterministic signals from the prompt:

| Signal | Weight | Floor |
|---|---|---|
| `implement`, `refactor`, `debug`, `prove`, step-by-step | 3 | `large` |
| arithmetic, `solve`, `calculate` | 2 | `medium` |
| `extract`, `classify`, `summarise` | 1 | `small` |
| >24k tokens input | 2 | `medium` + `long-context` |
| >8k tokens input | +1 | — |
| `vision` required | — | at least `medium` |
| `tools` required | — | at least `small` |

**3. Cheapest model at or above the floor.** Ties broken by latency — two models at the same price are not equivalent to a user waiting on them.

**4. Budget check.** Downgrade (recording it), or raise, depending on configuration.

## Budgets

```python
router = Router(budget=Budget(0.50))
```

`Budget` is just `limit` and `spent`. The router tracks a balance and does not care whether that balance is per-request, per-job, or per-day.

When the preferred model doesn't fit:

```python
Router(budget=Budget(0.002), on_budget_pressure="downgrade")   # default
```
```
note  downgraded from large-1 ($0.004815) to nano-1 ($0.000080) to stay within budget
```

```python
Router(budget=Budget(0.002), on_budget_pressure="raise")
```
```
BudgetExceeded: remaining budget $0.002000 cannot cover large-1 ($0.004815);
                cheapest eligible costs $0.004815
```

`"raise"` is the right setting when a silently degraded answer is worse than a failed request.

**`record()` takes actuals, not estimates.** Output length varies far more than input length, so charging the estimate is how a budget quietly drifts over.

## Fallbacks

Every decision carries up to three alternates, used when a provider errors or times out:

```
primary    nano-1
fallbacks  small-1, medium-1, large-1
```

Ordered **same tier → up → down**, and that ordering is deliberate. Retrying a failed cheap call on an even cheaper model is how a transient outage becomes a quality incident nobody can trace back to its cause.

## Design notes

**No model call to pick a model.** A router that asks an LLM which LLM to use doubles your latency and cost to save money. Every rule here is a regex or an integer comparison: inspectable, arguable, testable.

**Token counts are `len(text) // 4`.** Deliberately not a tokenizer. Pulling one in makes the install heavy to refine a number that is then multiplied by an approximate price anyway.

**A request to *write* code contains no code.** The first version of the classifier matched only code syntax, so `"implement a quicksort"` was classified as chat — right tier by luck, wrong label. Now "implement", "refactor", "debug" and "write a function" are code signals too.

**"Eligible" includes the quality floor.** An early error message reported the price of a below-floor model as "cheapest available", which is true and useless. `_eligible` now means meets every constraint including the floor.

**Cost lives in one place.** `ModelSpec.estimate` is the only thing that multiplies tokens by price. A pricing change is one edit.

**`bool` is not a token count.** `True` is an `int` in Python; the estimate path rejects it rather than silently charging for 1 token.

## Development

```bash
python -m unittest discover -s tests -v   # 49 tests
```

CI runs on Python 3.9, 3.11 and 3.13, and asserts the routing behaviour end to end: a trivial prompt lands on `nano`, a code prompt on `large`, a tight budget downgrades with a recorded note, and an impossible budget raises.

## License

[MIT](LICENSE)
