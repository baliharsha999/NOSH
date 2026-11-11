# NOSH Conversational Recipe Intelligence Launcher
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         STARTING NOSH CONVERSATIONAL RECIPE ASSISTANT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Yellow
Start-Process -FilePath "$PSScriptRoot\venv\Scripts\python.exe" -ArgumentList "-m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000" -WindowStyle Hidden

Start-Sleep -Seconds 2

Write-Host "[2/2] Launching Streamlit Web UI on http://localhost:8501..." -ForegroundColor Green
& "$PSScriptRoot\venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\frontend\streamlit_app.py" --server.port 8501
