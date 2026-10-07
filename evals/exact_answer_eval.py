"""Held-out test of silent failures, with answers checked by code, not a judge.

The risk rules in router/risk.py were written after seeing four adversarial
questions in the eval set. This script measures them on questions they were
not written against: generated from a fixed seed, in question shapes where
Python can compute the right answer, so no judge model (and no judge bias) is
involved.

For each question it records:
  - whether the risk rules flag it (so it would skip the small model),
  - whether the small and large models answer it correctly.

Five question types, each asked in a canonical wording and in a paraphrase
the rules were not written for. `digit_sum` is a shape no rule targets, kept
as an honest coverage check. A list of harmless look-alike questions measures
false alarms alongside the 27 small-prior queries of the eval set.

Usage:
    python evals/exact_answer_eval.py                  # small and large
    python evals/exact_answer_eval.py --tiers small     # small model only
    python evals/exact_answer_eval.py --cached-only     # re-score from cache
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "src")]

# Importing run_benchmark sets PROVIDER_MAX_RETRIES=0 before router.config
# loads: RetryingClient owns retries here too.
from run_benchmark import DailyQuotaExhausted, RetryingClient

from router.config import LARGE, PROVIDER_PRESET, RESULTS_DIR, SMALL
from router.risk import silent_failure_risks

DATASET = Path(__file__).parent / "dataset.jsonl"
WORDS = ["nevertheless", "mississippi", "bookkeeper", "onomatopoeia", "parallelogram",
         "abracadabra", "committee", "possession", "strawberry", "embarrassment",
         "accommodate", "millennium", "occurrence", "assessment", "independence"]

# Harmless questions that share surface features with the risky shapes.
# Written before any model was run; a flag on one of these is a false alarm.
LOOKALIKES = [
    "How many continents are there?",
    "How many days are in a leap year?",
    "What year did World War II end?",
    "List three primary colours.",
    "List the days of the week.",
    "Name a famous painting by Leonardo da Vinci.",
    "What is the boiling point of water in Celsius?",
    "Is Python a compiled language?",
    "What does 'prime' mean in 'prime minister'?",
    "Convert 5 kilometres to metres.",
    "What is the plural of 'mouse'?",
    "Spell the word 'necessary'.",
    "Give me a synonym for 'quick'.",
    "What is 10% of 200?",
    "Who wrote 'Pride and Prejudice'?",
    "What is the opposite of 'reverse' gear in a car?",
]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def generate(seed: int, per_wording: int) -> list[dict]:
    """Questions with computed answers. kind: number | yesno | text."""
    rng = random.Random(seed)
    qs: list[dict] = []

    def add(qtype, wording, query, expected, kind):
        qs.append({"id": f"{qtype}-{wording[0]}{sum(q['type'] == qtype for q in qs) + 1}",
                   "type": qtype, "wording": wording, "query": query,
                   "expected": expected, "kind": kind})

    for i in range(2 * per_wording):
        wording = "canonical" if i < per_wording else "paraphrase"

        a, b, e = rng.randint(23, 989), rng.randint(12, 97), rng.randint(11, 499)
        d = rng.randint(3, 19)
        c = d * rng.randint(20, 300)
        if wording == "canonical":
            add("arithmetic", wording,
                f"What is {a} * {b} - {c} / {d} + {e}? Reply with just the number.",
                a * b - c // d + e, "number")
        else:
            # Same expression as the canonical form, in words: the paraphrase
            # must differ in phrasing only, not in difficulty.
            add("arithmetic", wording,
                f"Multiply {a} by {b}, subtract the result of dividing {c} by {d}, then add {e}. "
                "What do you get? Reply with just the number.",
                a * b - c // d + e, "number")

        word = rng.choice(WORDS)
        letter = rng.choice(sorted({ch for ch in word if word.count(ch) >= 2}))
        query = (f"How many times does the letter '{letter}' appear in the word '{word}'? "
                 if wording == "canonical" else
                 f"Count how often '{letter}' occurs in '{word}'. ")
        add("letter_count", wording, query + "Reply with just the number.",
            word.count(letter), "number")

        # Odd, not divisible by 3 or 5, so the easy tells are gone.
        want_prime = i % 2 == 0
        while True:
            n = rng.randint(1001, 19999)
            if n % 2 and n % 3 and n % 5 and _is_prime(n) == want_prime:
                break
        if wording == "canonical":
            add("primality", wording, f"Is {n} a prime number? Reply with just yes or no.",
                "yes" if want_prime else "no", "yesno")
        else:
            add("primality", wording,
                f"Does {n} have any divisors other than 1 and itself? Reply with just yes or no.",
                "no" if want_prime else "yes", "yesno")

        n = rng.randint(10 ** 11, 10 ** 13)
        query = (f"What is the sum of the digits of {n}? " if wording == "canonical" else
                 f"Add up all the digits in {n}. ")
        add("digit_sum", wording, query + "Reply with just the number.",
            sum(int(ch) for ch in str(n)), "number")

        word = rng.choice(WORDS)
        query = (f"Reverse the letters of the word '{word}'. Reply with just the reversed word."
                 if wording == "canonical" else
                 f"Write '{word}' backwards. Reply with just the result.")
        add("reversal", wording, query, word[::-1], "text")
    return qs


def is_correct(reply: str, expected, kind: str) -> bool:
    text = reply.strip()
    if kind == "number":
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return bool(nums) and float(nums[-1].replace(",", "")) == float(expected)
    if kind == "yesno":
        m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
        return bool(m) and m.group(1).lower() == expected
    tokens = re.findall(r"[a-z]+", text.lower())
    return expected in tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["small", "large"], choices=["small", "large"])
    # Seed 7 was used while developing this script (and exposed a paraphrase
    # that was easier than its canonical form); reported numbers use seed 11.
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--per-wording", type=int, default=4)
    ap.add_argument("--cached-only", action="store_true")
    ap.add_argument("--large-only-flagged", action="store_true",
                    help="ask the large model only the questions the rules flag: the ones "
                         "routing would send it, which is what decides whether the rules help")
    args = ap.parse_args()

    questions = generate(args.seed, args.per_wording)
    client = RetryingClient()
    specs = {"small": SMALL, "large": LARGE}
    print(f"{len(questions)} generated questions (seed {args.seed}), preset {PROVIDER_PRESET}: "
          f"small {SMALL.model_id}, large {LARGE.model_id}\n")

    for q in questions:
        q["flagged_by"] = silent_failure_risks(q["query"])
        for tier in args.tiers:
            spec = specs[tier]
            if tier == "large" and args.large_only_flagged and not q["flagged_by"]:
                q[tier] = None
                continue
            if args.cached_only:
                res = client.cached(spec, q["query"])
            else:
                try:
                    res = client.complete(spec, q["query"])
                except DailyQuotaExhausted as exc:
                    print(f"  ! {exc}")
                    res = None
            if res is None or res.error:
                q[tier] = None
                continue
            q[tier] = is_correct(res.text, q["expected"], q["kind"])
            q[f"{tier}_reply"] = res.text.strip()[:200]

    # ---- report -----------------------------------------------------------
    def rate(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs)}/{len(xs)}" if xs else "-"

    print(f"{'type':<14}{'wording':<12}{'flagged':>9}{'small ok':>10}{'large ok':>10}")
    groups = defaultdict(list)
    for q in questions:
        groups[(q["type"], q["wording"])].append(q)
    for (qtype, wording), qs in groups.items():
        print(f"{qtype:<14}{wording:<12}{rate([bool(q['flagged_by']) for q in qs]):>9}"
              f"{rate([q.get('small') for q in qs]):>10}{rate([q.get('large') for q in qs]):>10}")

    wrong_small = [q for q in questions if q.get("small") is False]
    caught = [q for q in wrong_small if q["flagged_by"]]
    print(f"\nSmall model wrong (silent failures): {len(wrong_small)}/"
          f"{sum(q.get('small') is not None for q in questions)}")
    print(f"  of those, flagged by the rules: {len(caught)}/{len(wrong_small)}")
    flagged = [q for q in questions if q["flagged_by"] and q.get("large") is not None
               and q.get("small") is not None]
    if flagged:
        print(f"On questions the rules flag (n={len(flagged)}): small model "
              f"{rate([q['small'] for q in flagged])}, large model {rate([q['large'] for q in flagged])}")
    both = [q for q in questions if q.get("small") is not None and q.get("large") is not None]
    if both:
        routed = [q["large"] if q["flagged_by"] else q["small"] for q in both]
        print(f"Accuracy on questions both tiers answered (n={len(both)}): "
              f"small only {rate([q['small'] for q in both])}, "
              f"large only {rate([q['large'] for q in both])}, "
              f"small + risk rules {sum(routed)}/{len(both)}")

    small_prior = [json.loads(line)["query"] for line in DATASET.read_text(encoding="utf-8").splitlines()
                   if line.strip() and json.loads(line)["prior"] == "small"]
    for name, items in (("eval-set small-prior queries", small_prior), ("look-alike questions", LOOKALIKES)):
        flagged = [(x, silent_failure_risks(x)) for x in items if silent_failure_risks(x)]
        print(f"False alarms on {name}: {len(flagged)}/{len(items)}"
              + "".join(f"\n    {rules} {x}" for x, rules in flagged))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"exact_answer_eval_seed{args.seed}.json"
    out.write_text(json.dumps({"preset": PROVIDER_PRESET, "small": SMALL.model_id,
                               "large": LARGE.model_id, "seed": args.seed,
                               "questions": questions}, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
