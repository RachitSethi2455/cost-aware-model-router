# Adaptive LLM Router

Per-request model selection for LLM applications. Classifies incoming query
complexity, dispatches to a small or large model accordingly, and escalates
automatically when the small model's answer fails a set of deterministic
quality checks.

Built to answer a question most LLM projects skip: **how much of your inference
bill is spent on requests that never needed the expensive model?**

```
                      ┌──────────────────┐
   query ────────────▶│ feature extract  │  6 interpretable features
                      └────────┬─────────┘
                               ▼
                      ┌──────────────────┐
                      │     router       │  heuristic │ logistic regression
                      └────┬────────┬────┘
                  P<0.5    │        │   P≥0.5
                           ▼        ▼
              ┌────────────────┐  ┌──────────────────┐
              │  Haiku 4.5     │  │     Opus 5       │
              │  $1 / $5 MTok  │  │  $5 / $25 MTok   │
              └───────┬────────┘  └────────┬─────────┘
                      ▼                    │
             ┌──────────────────┐          │
             │ escalation check │          │
             │ (deterministic)  │          │
             └───┬──────────┬───┘          │
              ok │          │ fail         │
                 │          └──────────────┤ retry on large
                 ▼                         ▼
                    response + cost metadata
```

## Design decisions

**Escalation checks are deterministic, not a second LLM call.** Asking a model
to grade its own answer costs roughly what calling the large model costs, which
defeats the purpose, and self-assessment is unreliable. Instead the pipeline
checks structural signals: empty output, refusal `stop_reason`, truncation at
the token ceiling, hedging language, and "code was requested but no code came
back." These catch most real failures for free.

**Routing wrong is survivable if you can detect it.** The escalation path means
a misroute costs latency and one extra call, not a wrong answer. This is why
the router can be aggressive about sending traffic to the small model.

**Six interpretable features, no embeddings.** An embedding call would cost as
much as the routing decision saves. Logistic regression over named features
also means the coefficients are readable — see below.

**Types are separated from the SDK** (`router/types.py`), so all deterministic
logic is importable and testable without the `anthropic` package or an API key.
CI runs the full offline suite on every push.

## Results

Run `python evals/run_benchmark.py` to populate. Four arms over the labeled
eval set, all through identical pipeline code so the comparison isn't
confounded:

| Arm | Quality | $/100 req | p50 latency | % routed small |
|---|---|---|---|---|
| All small (Haiku 4.5) | _tbd_ | _tbd_ | _tbd_ | 100% |
| All large (Opus 5) | 1.000 | _tbd_ | _tbd_ | 0% |
| Heuristic router | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Classifier router | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Router accuracy (offline, no API calls):**

| Router | Accuracy vs labels |
|---|---|
| Majority class baseline | 55.0% |
| Heuristic rules | 56.7% |
| Logistic regression (5-fold CV) | 80.0% ± 6.7% |

**Learned feature weights** (positive pushes toward the large model):

| Feature | Coefficient |
|---|---|
| `reasoning_marker_count` | +1.146 |
| `simple_marker_count` | −0.969 |
| `n_constraints` | −0.719 |
| `n_sentences` | +0.425 |
| `n_words` | +0.385 |
| `has_code` | +0.377 |

## Known limitations

Written up deliberately — these are the interesting parts.

1. **The heuristic baseline is weak (56.7%, barely above majority class).** It
   systematically under-routes: long complex queries that lack literal marker
   words like "explain" or "compare" slip through as simple. The classifier's
   advantage is partly that it learns length and sentence structure matter, not
   just keywords. A stronger hand-tuned baseline would narrow the gap.

2. **`n_constraints` has a negative coefficient, which is counterintuitive.**
   Bullet-point structure was expected to signal difficulty; instead the model
   learned it correlates with well-specified, mechanically answerable requests.
   Small-sample artifacts are also plausible at n=60.

3. **Quality is measured relative to the large model, not absolute truth.** The
   judge grades candidates against the large model's own answer, so the
   all-large arm scores 1.00 by construction. This measures *quality retention
   under routing*, not correctness. A human-labeled subset would be needed for
   the latter.

4. **The eval set is small (60 queries) and hand-written by one person.** CV
   standard deviation is wide. Numbers are directional, not precise.

5. **Adversarial "simple-looking" queries are the dominant failure mode.**
   Arithmetic, letter-counting, and one-word ambiguous prompts look trivial by
   every surface feature but need the large model. Surface features cannot fix
   this; it needs either a semantic signal or an accepted escalation cost.

## Using a different provider

The routing logic is provider-agnostic — only the transport differs. Vendor
SDKs live behind `router/providers/`, so switching is a `.env` change:

```bash
PROVIDER_PRESET=groq
GROQ_API_KEY=gsk_...
```

| Preset | Small | Large | Cost ratio |
|---|---|---|---|
| `anthropic` | Haiku 4.5 | Opus 5 | 5.0x |
| `openai` | gpt-4o-mini | gpt-4o | 16.7x |
| `gemini` | gemini-2.0-flash | gemini-2.5-pro | 12.5x |
| `groq` | llama-3.1-8b | llama-3.3-70b | 11.8x |
| `deepseek` | deepseek-chat | deepseek-reasoner | 2.0x |
| `openrouter` | llama-3.1-8b | claude-sonnet-4.5 | 150x |
| `local` | llama3.2:3b (Ollama) | llama3.1:8b | free |

**Verify the per-MTok rates in `config.py` against your provider's pricing page
before benchmarking.** Every cost figure in this project derives from them; a
stale rate silently invalidates the headline result.

Adding a provider: one preset in `config.py`, one endpoint entry in
`providers/__init__.py`. Any endpoint exposing `/v1/chat/completions` works
through the existing OpenAI-compatible adapter.

## Setup

```bash
git clone https://github.com/<you>/llm-router.git
cd llm-router
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # add your ANTHROPIC_API_KEY
```

## Usage

```bash
# Offline tests — no API key required
pytest

# Train the router (offline, uses dataset priors)
python evals/train_router.py

# Smoke-test the benchmark on 10 queries before spending real money
python evals/run_benchmark.py --limit 10

# Full benchmark, all four arms
python evals/run_benchmark.py

# Ablation: same run with escalation disabled
python evals/run_benchmark.py --no-escalation --tag no_escalation

# Relabel from observed outcomes and retrain
python evals/train_router.py --from-benchmark results/benchmark_main.json

# Chart for the README
python scripts/make_chart.py results/benchmark_main.json

# Serve
uvicorn api.main:app --reload --app-dir src
```

### API

```bash
# Free — routing decision only, no model call
curl -X POST localhost:8000/explain \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the capital of Japan?"}'

# Full route
curl -X POST localhost:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"query": "Explain why CAP theorem is often oversimplified."}'

curl localhost:8000/stats
```

Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
docker build -t llm-router .
docker run -p 8000:8000 --env-file .env llm-router
```

## Cost control

Responses are cached in SQLite keyed by `(model_id, prompt)`. Re-running the
benchmark while debugging scoring code costs nothing after the first pass.
Delete `data/response_cache.sqlite` to force fresh calls.

## Project layout

```
src/router/     config, features, heuristic, classifier, escalation, pipeline, llm, cache, types
src/api/        FastAPI service
evals/          dataset.jsonl, judge, benchmark runner, router training
tests/          offline unit tests
scripts/        chart generation
```
