"""Test-wide settings, applied before the router's config is imported."""

import os

# Tests must not append to the developer's real decision or shadow logs;
# tests that check logging point them at a temporary file.
os.environ["DECISION_LOG"] = ""
os.environ["SHADOW_RATE"] = "0"
