# NOSH End-to-End DeepEval Runner
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:DEEPEVAL_TELEMETRY_OPT_OUT = "YES"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         RUNNING NOSH END-TO-END DEEPEVAL EVALUATION SUITE" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

& "$PSScriptRoot\venv\Scripts\python.exe" -u "$PSScriptRoot\evaluations\run_evaluation.py"
