"""Grade logged own-traffic answer pairs offline and write training rows.

Each line of data/shadow_log.jsonl (router/shadow.py) holds a question and
both models' answers. This grades the small model's answer against the large
model's with the project's LLM judge (evals/judge.py), outside the request
path, and writes data/shadow_labels.jsonl: the question's v2 features and the
gain, the large model's grade minus the small model's on a 0-1 scale (the
large answer is the reference, so it scores 1). Then:

    python evals/train_outcome_router.py --own-traffic data/shadow_labels.jsonl

Judge calls are cached, so re-running only grades new pairs. A failed judge
call is skipped, not counted as a loss for the small model.

Usage:
    python evals/label_shadow.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

from router.config import DATA_DIR  # noqa: E402


def label_entries(entries: list[dict], grade) -> tuple[list[dict], int]:
    """(training rows, judge failures). grade(query, reference, candidate)
    returns the judge's {"score": 0|1|2|None, ...}."""
    rows, failed = [], 0
    for entry in entries:
        verdict = grade(entry["query"], entry["large"]["answer"], entry["small"]["answer"])
        if verdict["score"] is None:
            failed += 1
            continue
        rows.append({"features_v2": entry["features_v2"], "gain": 1 - verdict["score"] / 2,
                     "source": entry["source"], "small_model": entry["small"]["model"],
                     "large_model": entry["large"]["model"], "ts": entry["ts"]})
    return rows, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DATA_DIR / "shadow_log.jsonl")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "shadow_labels.jsonl")
    args = ap.parse_args()
    if not args.log.exists():
        raise SystemExit(f"No shadow log at {args.log}. Set SHADOW_RATE (e.g. 0.05) and serve "
                         "some traffic first.")

    from judge import grade

    from router.llm import LLMClient
    client = LLMClient()
    entries = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line]
    rows, failed = label_entries(entries, lambda q, ref, cand: grade(client, q, ref, cand))
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    helped = sum(r["gain"] > 0 for r in rows)
    print(f"Graded {len(rows)} pairs ({failed} judge failures skipped); the large model did "
          f"better on {helped}. Wrote {args.out}")


if __name__ == "__main__":
    main()
