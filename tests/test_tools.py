"""Program-aided answers: the sandboxed evaluator must be correct and closed."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from router.tools import UnsafeExpression, extract_expression, format_result, safe_eval


@pytest.mark.parametrize("expr, expected", [
    ("486 * 83 - 4267 // 17 + 449", 40536),
    ("'nevertheless'.count('e')", 4),
    ("'algorithm'[::-1]", "mhtirogla"),
    ("sum(int(d) for d in str(1120410304605))", 27),
    ("is_prime(7919)", True),
    ("not is_prime(1729)", True),
    ("len([c for c in 'onomatopoeia' if c in 'aeiou'])", 8),
    ("''.join(reversed('abc'))", "cba"),
    ("max(3, 9, 4) - min(3, 9, 4)", 6),
    ("2 ** 10", 1024),
])
def test_computes_exact_answers(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",     # no imports, no dunder names
    "open('/etc/passwd').read()",             # no file access
    "().__class__.__bases__[0].__subclasses__()",  # classic sandbox escape
    "(lambda: 1)()",                          # no lambdas
    "eval('1+1')",                            # no eval
    "len.__self__",                           # no attribute access
    "[len][0]('abc')",                        # functions only called by name
    "str.join('', 'ab')",                     # methods only on string values
    "x",                                      # unknown names
    "1; 2",                                   # statements are not expressions
    "{'a': 1}['a']",                          # no dicts
])
def test_rejects_anything_outside_the_whitelist(expr):
    with pytest.raises(UnsafeExpression):
        safe_eval(expr)


@pytest.mark.parametrize("expr", [
    "10 ** 10 ** 10",
    "9 ** 999999",
    "'a' * 10 ** 9",
    "sum(range(10 ** 9))",
    "sum(1 for i in range(10 ** 6) for j in range(10 ** 6))",
    "('a' * 1000).replace('a', 'b' * 100000)",
    "int('9' * 100000)",
    "is_prime(10 ** 15 + 37)",
    "1 / 0",
])
def test_resource_limits_fail_fast(expr):
    start = time.perf_counter()
    with pytest.raises(UnsafeExpression):
        safe_eval(expr)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("reply, expected", [
    ("486 * 83", "486 * 83"),
    ("```python\n'abc'[::-1]\n```", "'abc'[::-1]"),
    ("`is_prime(97)`", "is_prime(97)"),
    ("NONE", None),
    ("none - this has no exact answer", None),
    ("x = 5\nprint(x)", None),  # multi-line code is not an expression
])
def test_extracts_the_expression(reply, expected):
    assert extract_expression(reply) == expected


def test_formats_results_for_the_question():
    assert format_result(True) == "yes" and format_result(False) == "no"
    assert format_result(12.0) == "12" and format_result(0.1 + 0.2) == "0.3"
    assert format_result(["Monday", "Tuesday"]) == "Monday, Tuesday"


class ScriptedClient:
    """Returns a fixed reply text (or error) for any prompt."""

    def __init__(self, text="", error=None):
        self.text, self.error, self.prompts = text, error, []

    def complete(self, spec, prompt, **kwargs):
        from router.types import CallResult
        self.prompts.append(prompt)
        return CallResult(text=self.text, model_id="m", tier="small", input_tokens=1,
                          output_tokens=1, cost_usd=0.0, latency_s=0.0, error=self.error)

    def cached(self, spec, prompt, **kwargs):
        return None


@pytest.mark.parametrize("reply, error, status, answer", [
    ("486 * 83 - 4267 // 17 + 449", None, "answered", "40536"),
    ("not is_prime(1729)", None, "answered", "yes"),
    ("NONE", None, "declined", None),
    ("'Chinua Achebe'", None, "declined", None),      # a stated answer, not a computation
    ("'LIFO' == 'LIFO'", None, "declined", None),     # a tautology, not a computation
    ("1174", None, "declined", None),
    ("__import__('os').getcwd()", None, "unsafe", None),
    ("", "RateLimitError: 429", "failed", None),
])
def test_solve_with_code_outcomes(reply, error, status, answer):
    from router.config import SMALL
    from router.tools import solve_with_code
    client = ScriptedClient(reply, error)
    # Worded, so it needs the model; symbolic arithmetic would be parsed directly.
    result = solve_with_code(client, SMALL, "Multiply 486 by 83, subtract 4267 divided by 17, then add 449.")
    assert (result.status, result.answer) == (status, answer)
    assert "Question: Multiply 486 by 83" in client.prompts[0]


def test_solve_with_code_cached_only_makes_no_call():
    from router.config import SMALL
    from router.tools import solve_with_code
    client = ScriptedClient("1 + 1")
    result = solve_with_code(client, SMALL, "Add 1 and 1 together.", cached_only=True)
    assert result.status == "failed" and client.prompts == []


@pytest.mark.parametrize("query, expected", [
    ("What is 486 * 83 - 4267 / 17 + 449?", True),
    ("Multiply 407 by 62, subtract the result of dividing 605 by 5, then add 225.", True),
    ("Count how often 'r' occurs in 'strawberry'.", True),
    ("Write 'possession' backwards.", True),
    ("Does 6923 have any divisors other than 1 and itself?", True),
    ("Add up all the digits in 6650632026309.", True),
    ("What is the capital of Australia?", False),
    ("Convert 45 degrees Celsius to Fahrenheit.", False),
    ("Spell out the number 4728 in English words.", False),
    ("What is the plural of 'mouse'?", False),
    ("How many continents are there?", False),
])
def test_trigger(query, expected):
    from router.tools import looks_computable
    assert looks_computable(query) is expected


def test_reversed_keeps_strings_as_strings():
    assert safe_eval("reversed('abc')") == "cba"
    assert safe_eval("reversed([1, 2, 3])") == [3, 2, 1]


class PipelineLLM:
    """Answers the code prompt with a fixed reply, anything else normally."""

    def __init__(self, code_reply):
        self.code_reply, self.prompts = code_reply, []

    def complete(self, spec, prompt, **kwargs):
        from router.types import CallResult
        self.prompts.append(prompt)
        is_code = prompt.startswith("Turn the question into ONE Python expression")
        return CallResult(text=self.code_reply if is_code else "A direct answer of some length.",
                          model_id=spec.model_id, tier=spec.name, input_tokens=1,
                          output_tokens=1, cost_usd=0.001, latency_s=0.0, stop_reason="end_turn")


def _pipe(code_reply, enabled=True):
    from router.pipeline import RouterPipeline
    llm = PipelineLLM(code_reply)
    return RouterPipeline(mode="heuristic", client=llm, enable_code_tool=enabled), llm


def test_pipeline_answers_by_code_when_triggered():
    pipe, llm = _pipe("486 * 83 - 4267 // 17 + 449")
    result = pipe.run("Multiply 486 by 83, subtract 4267 divided by 17, then add 449.")
    assert result.answer == "40536" and result.tier_served == "small"
    assert result.tool == {"expression": "486 * 83 - 4267 // 17 + 449", "answer": "40536"}
    assert len(result.calls) == 1 and len(llm.prompts) == 1


def test_pipeline_falls_back_and_counts_the_extra_call():
    pipe, llm = _pipe("NONE")
    result = pipe.run("Multiply 486 by 83, subtract 4267 divided by 17, then add 449.")
    assert result.tool is None and result.answer == "A direct answer of some length."
    assert len(result.calls) == 2 and result.total_cost_usd == 0.002


@pytest.mark.parametrize("query, kwargs, enabled", [
    ("What is the capital of Australia?", {}, True),                     # not computable
    ("What is 12 * 34?", {"history": [{"role": "user", "content": "hi"}]}, True),  # multi-turn
    ("What is 12 * 34?", {}, False),                                    # switched off
])
def test_pipeline_skips_the_tool(query, kwargs, enabled):
    pipe, llm = _pipe("12 * 34", enabled=enabled)
    result = pipe.run(query, **kwargs)
    assert result.tool is None and len(llm.prompts) == 1


def test_open_world_lists_do_not_trigger_the_tool():
    from router.risk import silent_failure_risks
    from router.tools import looks_computable
    q = "List the countries that border exactly three other countries."
    assert silent_failure_risks(q) == ["exhaustive_list"] and not looks_computable(q)


@pytest.mark.parametrize("question, expression", [
    ("What is 557 * 65 - 2484 / 12 + 424? Reply with just the number.", "557 * 65 - 2484 / 12 + 424"),
    ("Calculate the result of 557 * 65 - 2484 / 12 + 424.", "557 * 65 - 2484 / 12 + 424"),
    ("hey, what's 10 - 3?", "10 - 3"),
    ("What is 17 * 24 + 3^7 - 1024 / 8?", "17 * 24 + 3**7 - 1024 / 8"),
    ("What is (12 + 3) * 4?", "(12 + 3) * 4"),
    ("Is 12 * 3 greater than 40?", None),            # not a bare calculation
    ("What is 2026-10-07?", None),                    # a date, not subtraction
    ("I have 3 + 4 apples, how many is that?", None),
    ("Multiply 407 by 62, then subtract 5.", None),   # words: the model handles it
])
def test_direct_arithmetic_parses_only_bare_calculations(question, expression):
    from router.tools import direct_arithmetic
    assert direct_arithmetic(question) == expression


def test_bare_calculations_need_no_model_call():
    from router.config import SMALL
    from router.tools import solve_with_code
    client = ScriptedClient("this would be wrong")
    result = solve_with_code(client, SMALL, "Calculate the result of 557 * 65 - 2484 / 12 + 424.")
    assert (result.status, result.answer, result.call) == ("answered", "36422", None)
    assert client.prompts == []


def test_api_handles_an_answer_with_no_model_call(monkeypatch):
    """/route and /v1 must not assume at least one model call."""
    from fastapi.testclient import TestClient

    from api import main
    from router.pipeline import RouterPipeline

    pipe = RouterPipeline(mode="heuristic", client=PipelineLLM("unused"), enable_code_tool=True)
    monkeypatch.setattr(main, "get_pipeline", lambda mode: pipe)
    c = TestClient(main.app)
    q = "What is 17 * 24 + 3^7 - 1024 / 8?"

    r = c.post("/route", json={"query": q}).json()
    assert r["answer"] == "2467" and r["n_calls"] == 0 and r["cost_usd"] == 0
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": q}]}).json()
    assert r["choices"][0]["message"]["content"] == "2467" and r["model"] == "calculator"
