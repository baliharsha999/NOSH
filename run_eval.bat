@echo off
title NOSH DeepEval Suite Runner
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set DEEPEVAL_TELEMETRY_OPT_OUT=YES

echo ======================================================================
echo           RUNNING NOSH END-TO-END DEEPEVAL EVALUATION SUITE
echo ======================================================================
echo.
"%~dp0venv\Scripts\python.exe" -u "%~dp0evaluations\run_evaluation.py"
pause
