# Collect allam-2-7b and gpt-oss-120b answers on the 900 "own traffic"
# questions (evals/own_traffic_questions.json, disjoint from the live test set).
# Stops at the daily token budget (175K per model: Groq allows 200K, leaving
# room for 2-3 requests from other tools on the same key) and resumes from the cache
# when run again. Paced at 5,000 tokens a minute per model so other tools on
# the same Groq key keep headroom.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OPENBLAS_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
& ".venv\Scripts\python.exe" -u -W ignore evals\live_exact_benchmark.py --preset groq-allam --question-set own --max-tokens-small 1024 --tpm-small 5000 --tpm-large 5000 --daily-token-budget 175000
Write-Host "`n=== Finished for now. If it says 'Daily limit reached', run this script again tomorrow. ===`n"
