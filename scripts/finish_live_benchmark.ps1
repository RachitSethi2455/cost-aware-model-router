# Finish both live-benchmark runs, one after the other, in a visible terminal.
# Paced at 5,000 tokens a minute per model so other tools on the same Groq key
# keep per-minute headroom. Completed calls come from the cache.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OPENBLAS_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$python = ".venv\Scripts\python.exe"

Write-Host "`n=== Pair 1 of 2: allam-2-7b -> gpt-oss-120b ===`n"
& $python -u -W ignore evals\live_exact_benchmark.py --preset groq-allam --max-tokens-small 1024 --tpm-small 5000 --tpm-large 5000
Write-Host "`n=== Pair 2 of 2: gpt-oss-20b -> gpt-oss-120b ===`n"
& $python -u -W ignore evals\live_exact_benchmark.py --preset groq-oss --tpm-small 5000 --tpm-large 5000
Write-Host "`n=== Done ===`n"
