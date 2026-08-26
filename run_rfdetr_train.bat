@echo off
title Train RF-DETR Model
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

echo ===================================================
echo   Starting RF-DETR Model Training
echo ===================================================
echo.
python rfdetr_train.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo Training complete!
) else (
    echo.
    echo Training failed with error level %ERRORLEVEL%
)
echo.
pause
