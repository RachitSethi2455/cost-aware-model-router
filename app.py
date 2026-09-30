"""Vercel entrypoint: the public demo as a single serverless function.

Vercel looks for a FastAPI instance named `app` in ./app.py. Two things
differ from a normal deployment:

- PUBLIC_DEMO defaults to on, so a deployment made without any settings can
  never spend an API key. Set PUBLIC_DEMO=0 in the Vercel project to opt out.
- The trained model is not in git and the filesystem is read-only, so the
  router is trained in memory on cold start from evals/dataset.jsonl. It is
  60 examples and deterministic, so every instance gets the same model.

Locally, run `uvicorn api.main:app --app-dir src` as usual.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PUBLIC_DEMO", "1")

from api import main  # noqa: E402
from router.classifier import ClassifierRouter, train  # noqa: E402

if not main.get_classifier().trained:
    rows = [json.loads(line) for line in
            (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    main._classifier = ClassifierRouter(
        pipe=train([r["query"] for r in rows], [int(r["prior"] == "large") for r in rows])
    )

app = main.app
