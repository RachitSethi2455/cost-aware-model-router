"""Security regression suite for the code tool's expression evaluator.

The payloads come from real 2026 escapes of in-process Python sandboxes
(PraisonAI CVE-2026-47392, Crawl4AI CVE-2026-53753, CrewAI CVE-2026-37008)
and from classic eval-sandbox breakouts, plus resource-exhaustion inputs.
Each must be rejected with UnsafeExpression, and quickly.
"""

import ast
import random
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router.tools import UnsafeExpression, evaluate, safe_eval, safe_eval_isolated

ESCAPES = [
    # PraisonAI: builtins reached through a bound method's __self__, then vars().
    "print.__self__",
    "vars(print.__self__)['__import__']('os')",
    # Crawl4AI: generator frame walk to the real builtins.
    "(x for x in []).gi_frame.f_back.f_builtins",
    "[x for x in (y for y in [1])][0]",            # allowed shape, harmless value
    # Classic subclass walk to reach os / subprocess.
    "().__class__.__base__.__subclasses__()",
    "''.__class__.__mro__[1].__subclasses__()",
    "[].__class__",
    "(1).__class__",
    # Reflective builtins that turn strings into capabilities.
    "getattr('', '__class__')",
    "type('')",
    "vars()",
    "globals()",
    "locals()",
    "eval('1+1')",
    "exec('x=1')",
    "compile('1', 'f', 'eval')",
    "__import__('os')",
    "__builtins__",
    # CrewAI: ctypes through an import.
    "__import__('ctypes').CDLL(None)",
    # str.format does attribute lookups inside the format string itself.
    "'{0.__class__}'.format(1)",
    "'{0.__class__}'.format_map({})",
    # Functions as values, so they can be called indirectly.
    "[len][0]('abc')",
    "len",
    "(len)",
    "str.join('', 'ab')",
    # Methods on non-strings, or methods outside the six allowed.
    "[1, 2].count(1)",
    "'abc'.__add__('d')",
    "'abc'.encode()",
    "'abc'.count.__self__",
    # Syntax the evaluator never accepts.
    "lambda: 1",
    "(x := 1)",
    "f'{1}'",
    "{'a': 1}",
    "{1, 2}",
    "1; 2",
    "await x",
    "x",
]


@pytest.mark.parametrize("expression", ESCAPES)
def test_known_escape_payloads_are_rejected(expression):
    if expression == "[x for x in (y for y in [1])][0]":
        assert safe_eval(expression) == 1  # control: nested comprehensions still work
        return
    with pytest.raises(UnsafeExpression):
        safe_eval(expression)


RESOURCE_BOMBS = [
    "9 ** 9 ** 9 ** 9",
    "10 ** 10 ** 10",
    "'a' * 10 ** 9",
    "[0] * 10 ** 9",
    "('x' * 10 ** 4).replace('x', 'x' * 10 ** 4)",
    "'-'.join(str(i) for i in range(10 ** 6))",
    "''.join('x' * 99999 for i in range(99999))",
    "[[0] * 99999 for i in range(99999)]",
    "['a' * 99999 for i in range(99999)]",
    "sorted(range(10 ** 6)) + sorted(range(10 ** 6)) + sorted(range(10 ** 6))",
    "sum(1 for i in range(10 ** 12))",
    "sum(1 for i in range(10 ** 6) for j in range(10 ** 6))",
    "is_prime(2 ** 127 - 1)",
    "'%099999999d' % 1",
    "'%s' % 'x'",
    "int('9' * 100000)",
    "str(10 ** 10000)",
    "1 / 0",
]


@pytest.mark.parametrize("expression", RESOURCE_BOMBS)
def test_resource_bombs_fail_fast(expression):
    start = time.perf_counter()
    with pytest.raises(UnsafeExpression):
        safe_eval(expression)
    assert time.perf_counter() - start < 2.0


def test_number_formatting_with_percent_still_works():
    """Blocking str % must not block integer modulo."""
    assert safe_eval("17 % 5") == 2


def test_evaluator_never_calls_eval_exec_or_compile():
    """The evaluator walks the AST itself; a validate-then-eval design turns
    any gap in node coverage into real execution (the Crawl4AI lesson)."""
    tree = ast.parse((ROOT / "src" / "router" / "tools.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert not {c.func.id for c in calls} & {"eval", "exec", "compile", "__import__", "setattr"}
    # getattr appears exactly twice: dispatching to the evaluator's own node
    # handlers, and fetching a str method already checked against
    # STRING_METHODS. A new use must be reviewed, so pin the count.
    getattr_targets = [ast.unparse(c.args[0]) for c in calls if c.func.id == "getattr"]
    assert sorted(getattr_targets) == ["self", "target"]


# Pieces for random expressions: allowed operations mixed with forbidden ones.
_ATOMS = ["1", "7", "10 ** 5", "'ab'", "'x' * 1000", "True", "range(50)", "[1, 2, 3]",
          "len", "print", "__import__", "x", "''.__class__", "(i for i in range(3))"]
_UNARY = ["-{}", "not {}", "abs({})", "str({})", "sorted({})", "reversed({})", "{}[0]",
          "{}[::-1]", "{}.upper()", "{}.__class__", "{}.gi_frame", "type({})", "vars({})",
          "sum(int(d) for d in str({}))", "[c for c in {}]", "len({})", "is_prime({})"]
_BINARY = ["{} + {}", "{} * {}", "{} ** {}", "{} % {}", "{} // {}", "{} == {}",
           "{} and {}", "'-'.join([{}, {}])", "max({}, {})", "{} if {} else 0"]


def _random_expression(rng: random.Random, depth: int = 0) -> str:
    if depth > 3 or rng.random() < 0.3:
        return rng.choice(_ATOMS)
    if rng.random() < 0.5:
        return rng.choice(_UNARY).format(_random_expression(rng, depth + 1))
    return rng.choice(_BINARY).format(_random_expression(rng, depth + 1),
                                      _random_expression(rng, depth + 1))


def test_random_expressions_only_return_plain_values_or_reject():
    """Fuzz: whatever the shape, the evaluator returns a plain value or raises
    UnsafeExpression, never another exception, never a function or a frame."""
    rng = random.Random(1234)
    plain = (int, float, str, bool, list, tuple, range)
    for _ in range(600):
        expression = _random_expression(rng)
        start = time.perf_counter()
        try:
            value = safe_eval(expression)
        except UnsafeExpression:
            pass
        else:
            assert isinstance(value, plain), (expression, type(value))
            if isinstance(value, (list, tuple)):
                assert all(isinstance(v, plain) for v in value), expression
        assert time.perf_counter() - start < 2.0, expression


# Process isolation. On Windows these spawn a fresh interpreter, so they are
# slower; the timeout is generous where the result must come back.

def test_isolated_evaluation_returns_the_value():
    assert safe_eval_isolated("17 * 24 + 3 ** 7 - 1024 // 8", timeout_s=60) == 2467


def test_isolated_evaluation_propagates_rejections():
    with pytest.raises(UnsafeExpression, match="not allowed|only be called"):
        safe_eval_isolated("__import__('os')", timeout_s=60)


def test_isolated_evaluation_times_out():
    with pytest.raises(UnsafeExpression, match="timed out"):
        safe_eval_isolated("sum(i * i for i in range(99999))", timeout_s=0.001)


def test_inline_mode_skips_the_process():
    assert evaluate("2 + 2", isolation="inline") == 4


def test_public_demo_never_reaches_the_evaluator(monkeypatch):
    """On the public deployment the evaluator must be unreachable."""
    from fastapi.testclient import TestClient

    from api import main
    from router import tools

    def explode(*args, **kwargs):
        raise AssertionError("evaluator called in public demo mode")

    monkeypatch.setattr(main, "PUBLIC_DEMO", True)
    monkeypatch.setattr(tools, "safe_eval", explode)
    monkeypatch.setattr(tools, "safe_eval_isolated", explode)
    c = TestClient(main.app)
    q = "What is 17 * 24 + 3^7 - 1024 / 8?"
    assert c.post("/route", json={"query": q}).status_code == 403
    assert c.post("/v1/chat/completions",
                  json={"messages": [{"role": "user", "content": q}]}).status_code == 403
    assert c.post("/explain", json={"query": q}).status_code == 200
