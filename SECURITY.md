# Security

## Threat model

The router has two ways for untrusted input to cause harm.

1. **The code tool's evaluator.** The small model writes a Python expression
   from the user's question, and `src/router/tools.py` evaluates it. Because a
   user can steer that expression through prompt injection, **every expression
   is treated as attacker-controlled**. Plain symbolic arithmetic in a question
   reaches the evaluator directly, with no model in between.
2. **Spending.** Every routed request costs money, and a query can be phrased
   to push it toward the expensive model (OWASP LLM10:2025, "Unbounded
   Consumption").

Out of scope: the security of the model providers themselves, and API keys,
which are read from the environment and never logged or returned.

## The evaluator

It is an **allowlist interpreter, not `eval`**. It parses the expression and
walks the syntax tree itself, executing only the node types it knows:
literals, arithmetic, comparisons, `and`/`or`/`not`, conditional
expressions, lists and tuples, slicing, simple comprehensions, calls to a
fixed set of functions (`len`, `sum`, `abs`, `min`, `max`, `round`, `int`,
`str`, `sorted`, `reversed`, `any`, `all`, `range`, `is_prime`) by name, and
six string methods (`count`, `lower`, `upper`, `replace`, `split`, `join`)
on values whose exact type is `str`.

That excludes the three ingredients every published 2026 escape needed:
attribute access that reaches dunder or frame attributes, reflective builtins
that turn strings into capabilities (`getattr`, `vars`, `type`, `eval`), and
ways to define types or import modules. `str.format` and printf-style `%`
formatting are also excluded, because they perform lookups and allocations
inside the format string, where a syntax-tree check cannot see them.

**Resource limits**, checked before or while values are built:

| Limit | Value |
|---|---|
| Expression length | 500 characters |
| Exponent / power result | exponent ≤ 10,000; result ≤ 100,000 bits |
| `range()` size | 1,000,000 |
| Interpreted comprehension steps (whole expression) | 100,000 |
| Any string | 100,000 characters |
| Total characters/elements produced (whole expression) | 2,000,000 |
| `is_prime` argument | ≤ 10^12 |

**Process isolation.** On Linux, where the service deploys, each evaluation
runs in a child process (`forkserver`) with a 2-second wall-clock timeout and
`RLIMIT_AS` (512 MB) and `RLIMIT_CPU` limits, so a pathological expression
cannot block or crash the web worker. `SANDBOX_ISOLATION` selects `process`,
`inline` or `auto` (the default: process where `forkserver` exists).

**Regression tests** (`tests/test_sandbox.py`) cover:

- the escape chains from PraisonAI CVE-2026-47392 (`print.__self__` →
  `vars()`), Crawl4AI CVE-2026-53753 (generator `gi_frame` → `f_back` →
  `f_builtins`) and CrewAI CVE-2026-37008 (`ctypes` import), plus classic
  subclass walks and format-string lookups;
- resource bombs, each of which must be rejected in under 2 seconds;
- a fuzz test over random expressions, which must either return a plain
  value or raise `UnsafeExpression`;
- a check that the evaluator never calls `eval`, `exec` or `compile`;
- a check that the public demo cannot reach the evaluator at all.

**If you extend it:** adding `getattr`, `type`, `vars`, `str.format`,
f-strings, lambdas or multi-statement code reopens documented attacks. For
multi-statement programs, use a real isolation boundary (gVisor or a
Firecracker microVM), not an in-process interpreter.

## Spending

- `PUBLIC_DEMO=1` disables every endpoint that calls a model (`/route`,
  `/v1/chat/completions`), so the public demo cannot spend an API key.
  `/explain` is rate-limited per client.
- Known gap: routing uses surface features, so a client can raise its own
  share of large-model calls by reshaping a question; for the current router,
  splitting it into more sentences or adding a code block does that. A
  per-client cap on large-model calls or a daily spend limit would bound this.

## Reporting

Please open a GitHub issue, or for anything sensitive contact the maintainer
through the LinkedIn link on the profile before disclosing details.
