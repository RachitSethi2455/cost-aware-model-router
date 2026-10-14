# Train and evaluate the outcome router on both feature sets, one after the
# other: v1 (evaluation only, for a like-for-like baseline), then v2, which
# is exported as the shipped model if its export checks out.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OPENBLAS_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$python = ".venv\Scripts\python.exe"

Write-Host "`n=== Step 1 of 2 (0%): v1 features, evaluation only ===`n"
& $python -u -W ignore evals\train_outcome_router.py --features v1 --no-export
Write-Host "`n=== Step 2 of 2 (50%): v2 features, train and export ===`n"
& $python -u -W ignore evals\train_outcome_router.py --features v2
Write-Host "`n=== Done (100%) ===`n"
