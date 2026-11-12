# NOSH Conversational Recipe Intelligence Launcher
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "         STARTING NOSH CONVERSATIONAL RECIPE ASSISTANT" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Check and release port 8501 if previously occupied
$port8501 = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($port8501) {
    Write-Host "Re-initializing Streamlit process on port 8501..." -ForegroundColor Yellow
    foreach ($conn in $port8501) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# Check if FastAPI backend is already running on port 8000
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if (-not $port8000) {
    Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Yellow
    Start-Process -FilePath "$PSScriptRoot\venv\Scripts\python.exe" -ArgumentList "-m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000" -WindowStyle Hidden
    Start-Sleep -Seconds 2
} else {
    Write-Host "[1/2] FastAPI Backend is already active on http://localhost:8000." -ForegroundColor Green
}

Write-Host "[2/2] Launching Streamlit Web UI on http://localhost:8501..." -ForegroundColor Green
& "$PSScriptRoot\venv\Scripts\python.exe" -m streamlit run "$PSScriptRoot\frontend\streamlit_app.py" --server.port 8501
