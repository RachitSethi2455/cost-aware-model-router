"""Program-aided answers: the small model writes an expression, code computes it.

For exact-answer questions (arithmetic, counting letters, primality, string
reversal, digit sums) a small model answering directly is wrong more than half
the time, and confidently so. A larger model only makes a right answer more
likely. Here the small model instead translates the question into one Python
expression, and `safe_eval` computes it, so the arithmetic itself is exact.

`safe_eval` is not Python's eval. It walks the expression's AST and allows only
a whitelist: literals, arithmetic, comparisons, a handful of builtins, a few
string methods, slicing, and simple comprehensions. There are no imports, no
attribute access beyond the listed methods, no names except comprehension
variables and the listed functions, and size limits on powers, ranges and
string repetition, so a hostile expression can neither reach the system nor
hang the server. Anything outside the whitelist raises UnsafeExpression and
the caller falls back to answering normally.
"""

from __future__ import annotations

import ast
import math
import multiprocessing
import operator
import re
import sys
from dataclasses import dataclass

from .config import SANDBOX_ISOLATION
from .types import CallResult

MAX_RANGE = 1_000_000          # size of a range(); built-ins iterate it at C speed
MAX_STEPS = 100_000            # comprehension steps this evaluator interprets (~0.2 s)
MAX_POW_BITS = 100_000         # result size of a ** b, in bits
MAX_STR = 100_000              # characters a string result may reach
# Total characters/elements across every value one expression produces. Each
# operation has its own limit, but [[0] * 99999 for i in range(99999)] passes
# all of them and still asks for ten billion elements.
MAX_TOTAL_ALLOC = 2_000_000

CODE_PROMPT = """Turn the question into ONE Python expression that computes the exact answer.

Allowed: numbers, strings, + - * / // % **, comparisons, and/or/not,
len, sum, abs, min, max, round, int, str, sorted, reversed, any, all, range, is_prime(n),
string methods .count .lower .upper .replace .split .join, slicing like s[::-1],
and simple comprehensions like sum(int(d) for d in str(n)).
For a yes/no question, write an expression that evaluates to True or False.
Do not work out the answer yourself: write the expression that computes it.
A bare number or string is not accepted, even if the question says "calculate".
If the question has no exact answer that code can compute, reply with NONE.

Reply with only the expression or NONE: no explanation, no code fences.

Question: {question}"""


class UnsafeExpression(ValueError):
    pass


def _is_prime(n: int) -> bool:
    if not isinstance(n, int) or n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n > 10 ** 12:
        raise UnsafeExpression("is_prime argument too large")
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def _range(*args):
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise UnsafeExpression("range too large")
    return r


def _join(sep, items):
    # Check the size while building the parts, not after joining them.
    parts, total = [], 0
    for item in items:
        part = str(item)
        total += len(part) + len(sep)
        if total > MAX_STR:
            raise UnsafeExpression("join result too large")
        parts.append(part)
    return sep.join(parts)


def _reversed(x):
    # reversed('abc') is an iterator in Python; a string in is a string out.
    return x[::-1] if isinstance(x, str) else list(reversed(x))


FUNCTIONS = {
    "len": len, "sum": sum, "abs": abs, "min": min, "max": max, "round": round,
    "int": int, "str": str, "sorted": sorted, "reversed": _reversed,
    "any": any, "all": all, "range": _range, "is_prime": _is_prime,
}

# Node types that do actual work. An expression with none of them, such as
# '1174' or "'LIFO' == 'LIFO'", is the model stating an answer, not computing
# one, and gets no more trust than answering directly.
_COMPUTING_NODES = (ast.BinOp, ast.Call, ast.Subscript, ast.GeneratorExp, ast.ListComp)
STRING_METHODS = {"count", "lower", "upper", "replace", "split", "join"}

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _pow(base, exp):
    if not isinstance(exp, (int, float)) or abs(exp) > 10_000:
        raise UnsafeExpression("exponent too large")
    if isinstance(base, (int, float)) and abs(base) > 1 and exp > 0:
        if exp * math.log2(abs(base)) > MAX_POW_BITS:
            raise UnsafeExpression("power too large")
    return operator.pow(base, exp)


def _mul(a, b):
    for s, n in ((a, b), (b, a)):
        if isinstance(s, (str, list)) and isinstance(n, int) and len(s) * n > MAX_STR:
            raise UnsafeExpression("repetition too large")
    return operator.mul(a, b)


class _Evaluator:
    def __init__(self):
        self.scopes: list[dict] = []
        # Shared across nested comprehensions: two nested loops of a million
        # each must not mean a trillion steps.
        self.steps = 0
        self.allocated = 0

    def run(self, node):
        method = getattr(self, f"_{type(node).__name__}", None)
        if method is None:
            raise UnsafeExpression(f"{type(node).__name__} is not allowed")
        return self._account(method(node))

    def _account(self, value):
        """Size limits on every value produced, per value and in total."""
        if isinstance(value, (str, list, tuple)):
            if isinstance(value, str) and len(value) > MAX_STR:
                raise UnsafeExpression("string too long")
            # Values are counted each time a node returns them, so this
            # over-counts; legitimate expressions stay far below the budget.
            self.allocated += len(value)
            if self.allocated > MAX_TOTAL_ALLOC:
                raise UnsafeExpression("expression allocates too much")
        return value

    def _Expression(self, n):
        return self.run(n.body)

    def _Constant(self, n):
        if not isinstance(n.value, (int, float, str, bool)):
            raise UnsafeExpression("constant type not allowed")
        return n.value

    def _BinOp(self, n):
        op = type(n.op)
        if op not in _BINOPS:
            raise UnsafeExpression("operator not allowed")
        left, right = self.run(n.left), self.run(n.right)
        if op is ast.Mod and isinstance(left, str):
            # printf-style formatting: '%099999999d' % 1 allocates ~100 MB.
            raise UnsafeExpression("string formatting not allowed")
        if op is ast.Pow:
            return _pow(left, right)
        if op is ast.Mult:
            return _mul(left, right)
        return _BINOPS[op](left, right)

    def _UnaryOp(self, n):
        if type(n.op) not in _UNARY:
            raise UnsafeExpression("operator not allowed")
        return _UNARY[type(n.op)](self.run(n.operand))

    def _BoolOp(self, n):
        values = [self.run(v) for v in n.values]
        return all(values) if isinstance(n.op, ast.And) else any(values)

    def _Compare(self, n):
        left = self.run(n.left)
        for op, comp in zip(n.ops, n.comparators, strict=True):
            if type(op) not in _COMPARE:
                raise UnsafeExpression("comparison not allowed")
            right = self.run(comp)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True

    def _IfExp(self, n):
        return self.run(n.body) if self.run(n.test) else self.run(n.orelse)

    def _List(self, n):
        return [self.run(e) for e in n.elts]

    def _Tuple(self, n):
        return tuple(self.run(e) for e in n.elts)

    def _Subscript(self, n):
        return self.run(n.value)[self.run(n.slice)]

    def _Slice(self, n):
        part = lambda x: None if x is None else self.run(x)  # noqa: E731
        return slice(part(n.lower), part(n.upper), part(n.step))

    def _Name(self, n):
        for scope in reversed(self.scopes):
            if n.id in scope:
                return scope[n.id]
        if n.id in FUNCTIONS:
            # Only callable by name (see _Call); never a value to pass around.
            raise UnsafeExpression(f"{n.id} can only be called")
        if n.id in ("True", "False"):
            return n.id == "True"
        raise UnsafeExpression(f"name {n.id!r} is not allowed")

    def _Call(self, n):
        if n.keywords:
            raise UnsafeExpression("keyword arguments not allowed")
        args = [self.run(a) for a in n.args]
        if isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS:
            return FUNCTIONS[n.func.id](*args)
        if isinstance(n.func, ast.Attribute) and n.func.attr in STRING_METHODS:
            target = self.run(n.func.value)
            # Exact type, not isinstance: a str subclass could override methods.
            if type(target) is not str:
                raise UnsafeExpression("methods are only allowed on strings")
            if n.func.attr == "join":
                return _join(target, *args)
            if n.func.attr == "replace" and len(args) >= 2:
                old, new = str(args[0]), str(args[1])
                if len(target) + target.count(old or " ") * len(new) > MAX_STR:
                    raise UnsafeExpression("replace result too large")
            return getattr(target, n.func.attr)(*args)
        raise UnsafeExpression("call not allowed")

    def _comprehension(self, generators, element, collect):
        out = []

        def loop(i, scope):
            if i == len(generators):
                self.scopes.append(scope)
                try:
                    out.append(self.run(element))
                finally:
                    self.scopes.pop()
                return
            gen = generators[i]
            if not isinstance(gen.target, ast.Name) or gen.is_async:
                raise UnsafeExpression("only simple comprehension targets allowed")
            self.scopes.append(scope)
            try:
                iterable = self.run(gen.iter)
            finally:
                self.scopes.pop()
            for value in iterable:
                self.steps += 1
                if self.steps > MAX_STEPS:
                    raise UnsafeExpression("comprehension too long")
                inner = {**scope, gen.target.id: value}
                self.scopes.append(inner)
                try:
                    keep = all(self.run(cond) for cond in gen.ifs)
                finally:
                    self.scopes.pop()
                if keep:
                    loop(i + 1, inner)

        loop(0, {})
        return collect(out)

    def _GeneratorExp(self, n):
        return self._comprehension(n.generators, n.elt, list)

    def _ListComp(self, n):
        return self._comprehension(n.generators, n.elt, list)


SANDBOX_TIMEOUT_S = 2.0
SANDBOX_MEMORY_BYTES = 512 * 1024 * 1024


def _evaluate_in_child(expression: str, conn) -> None:
    """Child-process entry point: apply OS limits, evaluate, report back."""
    if sys.platform.startswith("linux"):
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (SANDBOX_MEMORY_BYTES, SANDBOX_MEMORY_BYTES))
        cpu = math.ceil(SANDBOX_TIMEOUT_S)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    try:
        conn.send(("ok", safe_eval(expression)))
    except UnsafeExpression as exc:
        conn.send(("unsafe", str(exc)))
    except MemoryError:
        conn.send(("unsafe", "memory limit reached"))
    except Exception as exc:  # noqa: BLE001 - anything unexpected is a rejection
        conn.send(("unsafe", type(exc).__name__))
    finally:
        conn.close()


def safe_eval_isolated(expression: str, timeout_s: float = SANDBOX_TIMEOUT_S):
    """safe_eval in a separate process with a wall-clock timeout and, on
    Linux, address-space and CPU limits.

    In-process evaluation shares the web worker: a slow expression blocks it
    and Python threads cannot be killed. A child process can be.
    """
    methods = multiprocessing.get_all_start_methods()
    # forkserver forks from a clean single-threaded helper, which is safe
    # inside a threaded web server; spawn is the portable fallback.
    if "forkserver" in methods:
        ctx = multiprocessing.get_context("forkserver")
        # Keep the helper lean, so the memory limit applies to a small process
        # rather than to a copy of whatever the web server had imported.
        ctx.set_forkserver_preload(["router.tools"])
    else:
        ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_evaluate_in_child, args=(expression, sender), daemon=True)
    proc.start()
    sender.close()
    try:
        if not receiver.poll(timeout_s):
            raise UnsafeExpression("evaluation timed out")
        try:
            status, payload = receiver.recv()
        except EOFError as exc:  # child died: killed by a limit, or crashed
            raise UnsafeExpression("evaluation process ended without a result") from exc
    finally:
        receiver.close()
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=1)
    if status != "ok":
        raise UnsafeExpression(payload)
    return payload


def evaluate(expression: str, isolation: str | None = None):
    """Evaluate with the configured isolation: "process", "inline", or
    "auto" (process where forkserver exists, i.e. Linux, where this deploys;
    inline elsewhere, where starting a process costs a second)."""
    mode = (isolation or SANDBOX_ISOLATION).lower()
    if mode == "process":
        return safe_eval_isolated(expression)
    if mode == "auto" and "forkserver" in multiprocessing.get_all_start_methods():
        try:
            return safe_eval_isolated(expression)
        except OSError:
            # Some serverless hosts cannot start processes. "auto" keeps the
            # request working; set SANDBOX_ISOLATION=process to fail instead.
            pass
    return safe_eval(expression)


def safe_eval(expression: str):
    """Evaluate a whitelisted Python expression. Raises UnsafeExpression."""
    expression = expression.strip()
    if len(expression) > 500:
        raise UnsafeExpression("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"not a valid expression: {exc.msg}") from exc
    try:
        return _Evaluator().run(tree)
    except UnsafeExpression:
        raise
    except (ArithmeticError, TypeError, ValueError, IndexError, KeyError, RecursionError) as exc:
        raise UnsafeExpression(f"{type(exc).__name__}: {exc}") from exc


def extract_expression(reply: str) -> str | None:
    """The expression from a model reply, or None if it declined (NONE)."""
    text = reply.strip()
    fenced = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = text.strip("`").strip()
    if not text or text.upper().startswith("NONE"):
        return None
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[0].strip() if len(lines) == 1 else None


# When to try the code path. Broader than the risk rules on purpose: it should
# also catch rephrasings ("Multiply 407 by 62, ..."), and a false trigger only
# costs one cheap call, because a declined or non-computing reply falls back.
_COMPUTE_WORDS = re.compile(
    r"\b(?:multiply|multiplied|times|divide|divided|dividing|add|adding|plus|subtract|"
    r"subtracting|minus|sum|total|product|digits?|prime|divisors?|divisible|factors?|"
    r"remainder|modulo|squared?|cubed?|power|average|mean)\b",
    re.IGNORECASE,
)
_CHAR_WORDS = re.compile(
    r"\b(?:count|occurs?|occurrences?|times|letters?|characters?|vowels?|consonants?|"
    r"reversed?|reversing|backwards?|palindrome)\b",
    re.IGNORECASE,
)
_QUOTED_WORD = re.compile(r"'[A-Za-z]{2,}'|\"[A-Za-z]{2,}\"")
_INLINE_ARITHMETIC = re.compile(r"\d\s*(?:[+*/×÷^]|\s-\s)\s*\d")


def looks_computable(query: str) -> bool:
    """Whether a question looks like it has an exact, computable answer."""
    from .risk import silent_failure_risks  # local: risk does not depend on tools

    # An open-world list ("countries that border...") is a silent-failure risk
    # but not computable; trying code first only adds a wasted round trip
    # (11 s on the live run) before the normal answer.
    computable_risks = [r for r in silent_failure_risks(query) if r != "exhaustive_list"]
    if computable_risks or _INLINE_ARITHMETIC.search(query):
        return True
    if re.search(r"\d{2,}", query) and _COMPUTE_WORDS.search(query):
        return True
    return bool(_QUOTED_WORD.search(query) and _CHAR_WORDS.search(query))


def computes(expression: str) -> bool:
    """Whether the expression does any work, rather than state a literal."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(node, _COMPUTING_NODES) for node in ast.walk(tree))


def format_result(value) -> str:
    """Render a computed value the way the question expects."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_result(v) for v in value)
    return str(value)


@dataclass
class CodeAnswer:
    """Outcome of asking a model for an expression and computing it.

    status: "answered" (value computed), "declined" (model said NONE, or
    wrote a literal that computes nothing), "unsafe" (expression rejected or
    failed to evaluate), "failed" (the model call itself failed). Only
    "answered" carries an answer; every other status means the caller should
    fall back to answering normally.
    """

    status: str
    answer: str | None
    expression: str | None
    call: CallResult | None
    detail: str | None = None


# A question that is only an arithmetic expression in symbols, wrapped in a
# "what is / calculate" frame, needs no model at all: parse it and compute it.
# Seed 17 showed why: asked to "Calculate the result of 557 * 65 - ...", the
# small model did the arithmetic in its head and wrote a wrong number.
_EXPRESSION_SPAN = re.compile(
    r"\(*\s*\d+(?:\.\d+)?\s*\)*(?:\s*[-+*/×÷^]\s*\(*\s*\d+(?:\.\d+)?\s*\)*)+"
)
_UNSPACED_HYPHEN = re.compile(r"\d-\d")  # dates (2026-10-07) and ranges (3-5)
_QUESTION_FRAME = re.compile(
    r"^(?:(?:hey|hi|yo|so|ok|okay|quick question)[,!]?\s+)?"
    r"(?:what\s+is|what's|whats|calculate|compute|evaluate|work\s+out|solve|find)"
    r"(?:\s+the\s+(?:result|value|answer))?(?:\s+of)?\s*$",
    re.IGNORECASE,
)


def direct_arithmetic(question: str) -> str | None:
    """The expression to compute if the question is nothing but symbolic
    arithmetic in a question frame; None otherwise ("Is 12 * 3 > 40?" is not)."""
    body = re.split(r"\s+(?:Reply|Answer|Respond|Give)\b", question, maxsplit=1)[0]
    match = _EXPRESSION_SPAN.search(body)
    if not match or _UNSPACED_HYPHEN.search(match.group(0)):
        return None
    frame = (body[:match.start()] + body[match.end():]).strip().rstrip("?.!:= ").strip()
    if not _QUESTION_FRAME.match(frame):
        return None
    return match.group(0).strip().replace("^", "**").replace("×", "*").replace("÷", "/")


def solve_with_code(client, spec, question: str, cached_only: bool = False) -> CodeAnswer:
    """Compute `question`'s answer: parse it directly when it is plain symbolic
    arithmetic, otherwise ask `spec` for an expression and compute that."""
    parsed = direct_arithmetic(question)
    if parsed is not None:
        try:
            value = evaluate(parsed)
        except UnsafeExpression:
            pass  # e.g. unbalanced parentheses: let the model try instead
        else:
            return CodeAnswer("answered", format_result(value), parsed, None,
                              "parsed from the question, no model call")
    prompt = CODE_PROMPT.format(question=question)
    call = client.cached(spec, prompt) if cached_only else client.complete(spec, prompt)
    if call is None or call.error:
        return CodeAnswer("failed", None, None, call, call.error if call else "not cached")
    expression = extract_expression(call.text)
    if expression is None:
        return CodeAnswer("declined", None, None, call)
    if not computes(expression):
        return CodeAnswer("declined", None, expression, call, "no computation")
    try:
        value = evaluate(expression)
    except UnsafeExpression as exc:
        return CodeAnswer("unsafe", None, expression, call, str(exc))
    return CodeAnswer("answered", format_result(value), expression, call)
