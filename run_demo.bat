@echo off
title RF-DETR Live Human Detection Demo
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo ===================================================
echo   Starting RF-DETR Live Webcam Human Detection Demo
echo ===================================================
echo [1/2] Launching Python engine...
echo.
python -u demo.py
if %ERRORLEVEL% equ 0 (
    echo.
    echo Demo closed successfully!
) else (
    echo.
    echo Demo failed with error level %ERRORLEVEL%
)
pause

