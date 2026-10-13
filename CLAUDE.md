# Adaptive LLM Router — working notes for Claude Code

## What this project is
A per-request LLM router: classify query complexity, dispatch to a small or
large model, escalate to the large model when the small one's answer fails
deterministic quality checks. The deliverable is not just working code — it is
a **measured result** (cost saved vs quality retained) backed by a reproducible
benchmark run.

## Non-negotiable constraints
- **Never weaken the heuristic baseline to flatter the classifier.** The
  baseline exists to be a fair comparison. Its current weakness (56.7%, barely
  above the 55% majority class) is a documented finding, not a bug.
- **Escalation checks stay deterministic.** No LLM call to grade an LLM answer
  inside the request path — that defeats the cost argument.
- **`router/types.py` must not import any vendor SDK.** All deterministic logic
  must stay importable and testable without an API key. CI depends on this.
- **All benchmark arms run through the same `RouterPipeline`.** Separate code
  paths would confound the comparison.
- **Never cache failed calls** — a transient rate limit would poison the run.
- **Don't invent numbers.** Every figure in the README comes from a real
  benchmark run in `results/`. Leave `_tbd_` if unmeasured.

## Layout
```
src/router/config.py       model tiers, pricing, PRESETS per provider
src/router/features.py     7 interpretable features (no embeddings — too costly)
src/router/heuristic.py    rule baseline
src/router/outcome.py      shipped router: additive model read from outcome_model.json (no sklearn)
src/router/classifier.py   first router: logistic regression on 60 hand labels + heuristic fallback
src/router/escalation.py   deterministic guards on small-model output
src/router/risk.py         silent-failure question shapes (opt-in routing override)
src/router/tools.py        code tool: sandboxed expression evaluator + trigger (on by default)
src/router/pipeline.py     orchestration; 5 modes for the benchmark arms (default "outcome")
src/router/llm.py          caching + cost accounting
src/router/providers/      vendor adapters behind a Protocol
src/api/main.py            FastAPI (PUBLIC_DEMO disables /route)
src/api/static/            demo page + recorded examples.json
evals/                     dataset.jsonl, judge, benchmark, training, analysis, exact-answer eval
evals/routing_metrics.py   APGR / CPT / PGR curves, paired bootstrap (numpy only, tested in CI)
evals/sprout_replay.py     offline replay on SPROUT outcomes (needs requirements-eval.txt + data/sprout)
evals/sprout_unseen_source.py  same, holding out a whole source
evals/train_outcome_router.py  trains + exports the shipped router; leave-one-pair-out evaluation
evals/live_exact_benchmark.py  live calls on held-out SPROUT questions, exact-match graded, routers replayed
evals/exact_match.py       answer extraction + exact grading (letters, numbers as fractions)
```

## Commands
```
pytest                                     # offline, no key
python evals/train_router.py               # offline
python evals/run_benchmark.py --limit 10   # smoke test, costs money
python evals/run_benchmark.py              # full run
uvicorn api.main:app --reload --app-dir src
python evals/sprout_replay.py --pair llama   # offline, $0; ~550 MB data, not redistributed
python evals/train_outcome_router.py         # retrains src/router/outcome_model.json (needs the data)
```

## Conventions
- Python 3.11+, `from __future__ import annotations`, type hints throughout.
- Comments explain *why*, not *what*.
- New deterministic logic needs a test in `tests/`.
- Adding a provider = a new preset in `config.py` + an endpoint entry in
  `providers/__init__.py`. Nothing else should change.
