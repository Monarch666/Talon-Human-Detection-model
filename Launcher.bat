@echo off
title Human Detection Pipeline Launcher
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8

:MENU
cls
echo ===================================================
echo       Human Detection Pipeline Launcher
echo ===================================================
echo.
echo  [1] Convert Dataset to YOLO Format (convert_to_yolo.py)
echo  [2] Reorganize Dataset for RF-DETR (reorganize_dataset.py)
echo  [3] Train RF-DETR Model (rfdetr_train.py)
echo  [4] Train RT-DETR Model (train.py)
echo  [5] Evaluate Model & Confusion Matrix (eval_confusion_matrix.py)
echo  [6] Run Live Webcam Demo (demo.py)
echo  [7] Exit
echo.
echo ===================================================
set /p choice="Select an option (1-7): "

if "%choice%"=="1" goto CONVERT
if "%choice%"=="2" goto REORGANIZE
if "%choice%"=="3" goto TRAIN_RFDETR
if "%choice%"=="4" goto TRAIN_RTDETR
if "%choice%"=="5" goto EVAL
if "%choice%"=="6" goto DEMO
if "%choice%"=="7" goto END

echo.
echo Invalid selection. Please choose a number from 1 to 7.
echo.
pause
goto MENU

:CONVERT
echo.
echo ===================================================
echo   Running Dataset Conversion (convert_to_yolo.py)...
echo ===================================================
python convert_to_yolo.py
echo.
pause
goto MENU

:REORGANIZE
echo.
echo ===================================================
echo   Reorganizing Dataset (reorganize_dataset.py)...
echo ===================================================
python reorganize_dataset.py
echo.
pause
goto MENU

:TRAIN_RFDETR
echo.
echo ===================================================
echo   Starting RF-DETR Training (rfdetr_train.py)...
echo ===================================================
python rfdetr_train.py
echo.
pause
goto MENU

:TRAIN_RTDETR
echo.
echo ===================================================
echo   Starting RT-DETR Training (train.py)...
echo ===================================================
python train.py
echo.
pause
goto MENU

:EVAL
echo.
echo ===================================================
echo   Evaluating Model (eval_confusion_matrix.py)...
echo ===================================================
python eval_confusion_matrix.py
echo.
pause
goto MENU

:DEMO
echo.
echo ===================================================
echo   Starting Live Webcam Demo (demo.py)...
echo ===================================================
python demo.py
echo.
pause
goto MENU

:END
exit
