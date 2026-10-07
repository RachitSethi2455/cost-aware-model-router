# Adaptive LLM Router

[![tests](https://github.com/RachitSethi2455/cost-aware-model-router/actions/workflows/ci.yml/badge.svg)](https://github.com/RachitSethi2455/cost-aware-model-router/actions/workflows/ci.yml)

**At a glance**

- **Live result:** on real Gemini models (14 questions), routing cut cost
  **36–76%** against always using the large model while keeping **93–96%** of
  its answer quality, and cut median latency from 12.1 s to 5–8 s. The two
  routers sit at different points on that trade-off; see [Results](#results).
- **Generalises to new kinds of questions:** 78% routing accuracy on 60
  labelled queries, and **73%** on question categories never seen in training,
  where a bag-of-words classifier drops to 60%.
- **Safety net:** a deterministic check retries a cheap answer on the large
  model when it is empty, refused, cut off, hedging or missing requested code.
  It cannot catch an answer that is confidently wrong, and in the live runs
  routing, not escalation, did all the quality protection.
- **Drop-in for existing apps:** an OpenAI-compatible endpoint, so any
  OpenAI client can use the router by changing its base URL and sending
  `model="auto"`.
- **Engineering:** FastAPI service with a free interactive demo page, 7
  provider presets (Anthropic, OpenAI, Gemini, Groq, DeepSeek, OpenRouter,
  local), response cache, cost accounting, Docker / Render / Vercel configs,
  and an offline test suite plus lint in CI.

Per-request model selection for LLM applications. Classifies incoming query
complexity, dispatches to a small or large model accordingly, and escalates
automatically when the small model's answer fails a set of deterministic
quality checks.

Built to answer a question most LLM projects skip: **how much of your inference
bill is spent on requests that never needed the expensive model?**

```
                      ┌──────────────────┐
   query ────────────▶│ feature extract  │  7 interpretable features
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

**Seven interpretable features, no embeddings.** An embedding call would cost as
much as the routing decision saves. Logistic regression over named features
also means the coefficients are readable — see below.

**Types are separated from the SDK** (`router/types.py`), so all deterministic
logic is importable and testable without the `anthropic` package or an API key.
CI runs the full offline suite on every push.

## Results

Four arms, all through identical pipeline code so the comparison isn't
confounded. **Live run on 14 questions**, `gemini-free` preset (small
`gemini-3.1-flash-lite`, large `gemini-3.6-flash`, judge `gemini-3.5-flash`),
6–7 Oct 2026. The questions are 4 from each category of the eval set
(lookup/format, reasoning/code, complex, adversarial); 2 of the planned 16
were dropped because the free tier's overload errors never let the large
model produce a reference answer. The full 60-question run needs a paid key.
Raw results: [`results/benchmark_gemini_free16.json`](results/benchmark_gemini_free16.json).

| Arm | Quality | $/100 req | p50 latency | % routed small |
|---|---|---|---|---|
| All small | 0.893 | $0.086 | 4.5 s | 100% |
| All large | 1.000 (reference) | $0.843 | 12.1 s | 0% |
| Heuristic router | 0.929 | $0.199 | 5.0 s | 86% |
| Classifier router | **0.964** | $0.543 | 8.1 s | 50% |

Against always-large, the two routers land at different points on the
cost/quality curve: the heuristic cut cost **76%** keeping **93%** of quality;
the classifier cut cost **36%** keeping **96%**. Costs are at paid-tier
rates; latency is each model call as originally measured.

What the 14 questions show:

- **Where the quality goes.** The small model fell short on 3 of 14: a
  reasoning explanation (m01), a subtle bug fix (m04) and an exhaustive list
  (a03). The classifier sent 2 of those 3 to the large model, the heuristic
  only 1; that is the whole quality difference between them.
- **What the classifier's caution costs.** It also sent 4 questions to the
  large model that the small model answered fully. Its labels say how hard a
  question *looks*, which overestimates what this small model needs.
  Relabelling from observed outcomes (`train_router.py --from-benchmark`)
  is the fix, once the full 60-question run exists.
- **Routing pays even with a small price gap.** The large model's list price is
  about 3x the small one's, but it cost about 10x more per request because it
  writes far more (hidden reasoning) tokens. Output volume drives cost more
  than the price list does.
- **Escalation never fired.** All the quality protection came from routing.
  On "List the countries that border exactly three other countries" the small
  model returned a confident but incomplete list; structural checks (empty,
  refused, truncated, hedging) cannot see a wrong answer that looks complete.
  This is limitation 5, observed live.

A first run on 9 of these questions
([`results/benchmark_gemini_free_n9.json`](results/benchmark_gemini_free_n9.json))
had the heuristic ahead; the larger sample separated the two routers.

**Router accuracy (offline, no API calls; `python evals/analyze_router.py`):**

| Model | Accuracy, random splits | Accuracy, unseen category | AUC | Brier (lower is better) |
|---|---|---|---|---|
| Majority class | 55.0% | 26.7% | 0.500 (no ranking) | 0.248 |
| Heuristic rules | 56.7% | 56.7% | 0.811 | n/a (scores, not probabilities) |
| TF-IDF + logistic regression (bag of words) | 79.8% ± 3.7 | 60.0% | 0.870 ± 0.018 | 0.219 ± 0.002 |
| **Router: 7 features + logistic regression** | **78.2% ± 3.8** | **73.3%** | 0.833 ± 0.012 | **0.152 ± 0.011** |

*Random splits* are out-of-fold predictions from 20 reshuffled stratified
5-fold splits (± is the spread across the 20 repeats). *Unseen category*
trains on 12 of the 13 question categories and tests on the one left out.

**Why hand-made features rather than bag-of-words.** On random splits the
two are tied: similar phrasings land in both train and test, and TF-IDF
ranks slightly better. Hold out a whole category and TF-IDF drops to 60%: it
had learned each category's topic words rather than difficulty, and on unseen
easy categories it collapses (lookups 1/9 correct, formatting 2/6,
classification 1/4, against the router's 5/9, 6/6 and 4/4). The router's
probabilities are also better calibrated (Brier 0.152 vs 0.219), which
matters because it acts on a probability threshold. Both fail on the
adversarial categories; see limitation 5.

The rule-based heuristic ranks queries reasonably (AUC 0.81); its 56.7%
accuracy comes from a badly placed cut-off rather than bad signals.

**Threshold trade-off** (router, out-of-fold, mean over repeats):

| Threshold | Routed to small | Hard queries sent to small | Easy queries sent to large |
|---|---|---|---|
| 0.3 | 33% | 13% | 42% |
| 0.4 | 41% | 22% | 35% |
| **0.5 (default)** | 51% | 26% | 17% |
| 0.6 | 57% | 29% | 8% |
| 0.7 | 61% | 32% | 4% |

A hard query sent to the small model costs an escalation round trip if the
quality checks catch the failure, and a weaker answer if they don't. An easy
query sent to the large model only costs money. Which side to favour depends
on how good escalation is in practice, which is what the live benchmark
measures.

An earlier version of this README quoted 80.0% from a single 5-fold split;
with 60 examples one split swings by several points depending on the seed,
so the repeated-CV mean is the honest figure. The 6-feature version of the
router scored 75.7% on the same protocol.

The seventh feature, `asks_for_code`, was added after live testing showed
"Write a Python function to merge two sorted lists" routed to the small model:
`has_code` only detects code pasted *into* a query, not a request *for* code.
The +2.5 point gain is within one standard deviation, and the feature was
designed after seeing the eval set, so treat it as directional.

**Learned feature weights** (positive pushes toward the large model):

| Feature | Coefficient |
|---|---|
| `reasoning_marker_count` | +1.149 |
| `simple_marker_count` | −0.909 |
| `asks_for_code` | +0.711 |
| `n_constraints` | −0.664 |
| `has_code` | +0.460 |
| `n_words` | +0.316 |
| `n_sentences` | +0.309 |

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

## Lessons from live testing

The offline test suite passed from day one. Running against real Gemini
models still found four bugs, each now fixed and covered by a test:

1. **Thinking models eat the output budget.** With `max_tokens=1024`,
   `gemini-3.8-flash` spent ~980 tokens on hidden reasoning and returned a
   40-token fragment (`finish_reason=length`). Claude Opus 5 also thinks by
   default. `MAX_TOKENS` is now 8192, and the judge's budget went from 200 to
   2048 for the same reason.
2. **Hidden reasoning tokens were not billed.** Gemini reports them only in
   `total_tokens`, so the large model looked ~25x cheaper than it was, which
   would have inflated the headline saving. They are now counted as output.
3. **Failures looked like successes.** If both tiers failed, `/route` returned
   200 with an empty answer. It now returns 502 with the provider error, and a
   missing API key returns 503 instead of 500.
4. **`.env` was never loaded,** and a stale system-wide `GEMINI_API_KEY`
   silently overrode it (confusing 401s). `.env` is now loaded on startup,
   with a warning when a system variable overrides it.

The first benchmark smoke run on the free tier also hit quota limits (429s)
and "high demand" errors (503s). The benchmark now retries with backoff,
drops questions whose reference answer could not be produced, and reports
call and judge failures per arm instead of silently scoring them as 0.

Two more found later:

5. **Stacked retries burned the quota.** The vendor SDKs retry twice on their
   own, and the benchmark's backoff retried each of those three times, so one
   overloaded query sent up to 12 requests against a 20-requests/day free
   quota. SDK retries are now configurable (`PROVIDER_MAX_RETRIES`) and the
   benchmark sets them to 0, keeping one visible retry policy.
6. **A per-request threshold leaked into later requests.** `/route` stored
   the caller's threshold on a pipeline shared by every request. It is now
   passed per call.

`gemini-3.8-flash` stayed unusable on the free tier (overloaded and
quota-exhausted for a week), so the `gemini-free` preset uses
`gemini-3.6-flash` as the large model and `gemini-3.5-flash` as the judge.

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
| `gemini` | gemini-3.1-flash-lite | gemini-3.8-flash | 3.0x |
| `gemini-free` | gemini-3.1-flash-lite | gemini-3.6-flash | 3.0x |
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
git clone https://github.com/RachitSethi2455/cost-aware-model-router.git
cd cost-aware-model-router
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # runtime deps + pytest, matplotlib

cp .env.example .env        # pick PROVIDER_PRESET and add that provider's key
```

To try it for free, use `PROVIDER_PRESET=gemini` with a key from Google AI
Studio; both Gemini tiers have a free tier (with daily quotas, and free-tier
prompts may be used by Google to improve its products). If a key in `.env`
seems ignored, check for the same variable set system-wide: system
environment variables take precedence, and the router prints a warning.

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

### OpenAI-compatible endpoint

Existing apps can use the router by changing one line: point any OpenAI
client at the router and send `model="auto"`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
r = client.chat.completions.create(
    model="auto",  # or "small" / "large" to force a tier
    messages=[{"role": "user", "content": "What is the capital of Japan?"}],
)
print(r.choices[0].message.content)   # The capital of Japan is Tokyo.
print(r.model)                        # gemini-3.1-flash-lite (the model that answered)
print(r.model_extra["router"])        # tier chosen/served, escalated, confidence, cost
```

- Routing reads the latest user message; both tiers receive the whole
  conversation, including system messages.
- `usage` counts every billed call, including an escalation retry.
- Other OpenAI parameters (`temperature`, ...) are accepted and ignored.
  Streaming and non-text content return a 400 for now.
- Errors use OpenAI's `{"error": {...}}` shape, so the SDK raises its usual
  exceptions (`BadRequestError`, ...). `GET /v1/models` lists `auto`,
  `small` and `large`.

Verified with the official `openai` Python SDK against live Gemini models,
including a multi-turn conversation.

### Demo page

`http://localhost:8000/` serves an interactive page: type a question and see
which model the router picks, the probability against the threshold, and a
per-feature breakdown of *why* (each feature's push in log-odds, which sums
exactly to the decision). It only calls `/explain`, so it never spends a token.
It also shows real answers recorded from live Gemini runs; refresh them with
`python scripts/export_demo_examples.py` after running the queries live.

### Deploying the public demo

Set `PUBLIC_DEMO=1` on any public deployment. It disables `/route` (403) so
strangers cannot spend your API key, and `/explain` is rate-limited per client
(`EXPLAIN_RATE_LIMIT`, default 60/min). No API key is needed for the demo.

The Docker image trains the router during the build and listens on `$PORT`:

- **Vercel:** import the repo as a new project; no settings needed. `app.py`
  is the entrypoint: it turns `PUBLIC_DEMO` on by default and, because the
  model file is not in git and the filesystem is read-only, trains the router
  in memory on cold start (60 examples, deterministic, under a second).
- **Render:** `render.yaml` is a Blueprint for the free plan with
  `PUBLIC_DEMO=1` set. In the Render dashboard choose New → Blueprint and
  pick this repo. Render sets `PORT` itself.
- **Hugging Face Spaces:** create a Docker Space, push this repo to it, and add
  `sdk: docker` and `app_port: 8000` to the Space's README front matter; set
  `PUBLIC_DEMO=1` under Settings → Variables.

`/route` status codes:

| Code | Meaning |
|---|---|
| 200 | Answered. `tier_served` and `escalated` say which model produced it; `truncated` is true if the answer hit the token ceiling. |
| 502 | The provider failed (bad model id, rate limit, overload) and escalation could not rescue it. `detail.error` has the provider's message. |
| 503 | Not configured (missing API key or SDK). Nothing was sent to any model. |
| 403 | `PUBLIC_DEMO=1`: model calls are disabled on a public deployment. |

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
src/api/        FastAPI service + demo page (static/)
evals/          dataset.jsonl, judge, benchmark runner, router training
tests/          offline unit tests
scripts/        chart generation
```
