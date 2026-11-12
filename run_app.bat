@echo off
title NOSH Recipe Intelligence Runner
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ======================================================================
echo           STARTING NOSH CONVERSATIONAL RECIPE ASSISTANT
echo ======================================================================
echo.
:: Check if FastAPI backend is already running on port 8000
netstat -ano | findstr :8000 >nul 2>&1
if errorlevel 1 (
    echo [1/2] Launching FastAPI Backend on http://localhost:8000 ...
    start "NOSH FastAPI Backend" /min "%~dp0venv\Scripts\python.exe" -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
    timeout /t 2 /nobreak >nul
) else (
    echo [1/2] FastAPI Backend is already active on http://localhost:8000.
)

echo [2/2] Launching Streamlit Web UI on http://localhost:8501 ...
"%~dp0venv\Scripts\python.exe" -m streamlit run frontend/streamlit_app.py --server.port 8501
