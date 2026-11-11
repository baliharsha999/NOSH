@echo off
title NOSH Recipe Intelligence Runner
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ======================================================================
echo           STARTING NOSH CONVERSATIONAL RECIPE ASSISTANT
echo ======================================================================
echo.
echo [1/2] Launching FastAPI Backend on http://localhost:8000 ...
start "NOSH FastAPI Backend" /min "%~dp0venv\Scripts\python.exe" -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000

timeout /t 2 /nobreak >nul

echo [2/2] Launching Streamlit Web UI on http://localhost:8501 ...
"%~dp0venv\Scripts\python.exe" -m streamlit run frontend/streamlit_app.py --server.port 8501
