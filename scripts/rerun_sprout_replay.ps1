# Re-run the offline SPROUT replay for both pairs, then redraw the chart.
# One after the other: each run needs most of this machine's free memory.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OPENBLAS_NUM_THREADS = "1"
$env:PYTHONIOENCODING = "utf-8"
$python = ".venv\Scripts\python.exe"

Write-Host "`n=== Step 1 of 3 (0%): Llama 3.1 8B -> 70B ===`n"
& $python -u -W ignore evals\sprout_replay.py --pair llama
Write-Host "`n=== Step 2 of 3 (33%): GPT-4o-mini -> GPT-4o ===`n"
& $python -u -W ignore evals\sprout_replay.py --pair gpt4o
Write-Host "`n=== Step 3 of 3 (67%): chart ===`n"
& $python -u -W ignore scripts\sprout_chart.py
Write-Host "`n=== Done (100%) ===`n"
