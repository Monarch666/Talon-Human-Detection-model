@echo off
title Convert Dataset to YOLO Format
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

echo ===================================================
echo   Converting Dataset Labels to YOLO Format
echo ===================================================
echo.
echo Launching convert_to_yolo.py...
echo.

python convert_to_yolo.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ===================================================
    echo   Dataset conversion completed successfully!
    echo ===================================================
) else (
    echo.
    echo ===================================================
    echo   Conversion failed with Error Code: %ERRORLEVEL%
    echo ===================================================
)

echo.
pause
