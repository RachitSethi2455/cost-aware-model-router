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
src/router/classifier.py   logistic regression + heuristic fallback
src/router/escalation.py   deterministic guards on small-model output
src/router/pipeline.py     orchestration; 4 modes for the benchmark arms
src/router/llm.py          caching + cost accounting
src/router/providers/      vendor adapters behind a Protocol
src/api/main.py            FastAPI (PUBLIC_DEMO disables /route)
src/api/static/            demo page + recorded examples.json
evals/                     dataset.jsonl, judge, benchmark, training
```

## Commands
```
pytest                                     # offline, no key
python evals/train_router.py               # offline
python evals/run_benchmark.py --limit 10   # smoke test, costs money
python evals/run_benchmark.py              # full run
uvicorn api.main:app --reload --app-dir src
```

## Conventions
- Python 3.11+, `from __future__ import annotations`, type hints throughout.
- Comments explain *why*, not *what*.
- New deterministic logic needs a test in `tests/`.
- Adding a provider = a new preset in `config.py` + an endpoint entry in
  `providers/__init__.py`. Nothing else should change.
