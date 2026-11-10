@echo off
chcp 65001 >nul
title CinematicSRT v2.0 Pro Edition
cd /d "%~dp0"

:: 1. Search for Python environment in order of priority:
:: - Local venv
:: - Desktop cinematic venv
:: - System Python
set PYTHON_EXE=""

if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON_EXE="%~dp0venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\Desktop\cinematic\cinematic-srt\venv\Scripts\python.exe" (
    set PYTHON_EXE="%USERPROFILE%\Desktop\cinematic\cinematic-srt\venv\Scripts\python.exe"
) else (
    python --version >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set PYTHON_EXE=python
    )
)

if %PYTHON_EXE%=="" (
    echo [ERROR] لم يتم العثور على بيئة بايثون جاهزة!
    echo يرجى تشغيل ملف setup.bat أولاً ليقوم بتجهيز كل شيء تلقائياً.
    pause
    exit /b 1
)

echo ==========================================================
echo    🎬 Starting CinematicSRT v2.0 Pro Edition
echo ==========================================================
echo  - Python Interpreter: %PYTHON_EXE%
echo  - Server Address:     http://127.0.0.1:8001
echo ==========================================================
echo.

:: Open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8001"

%PYTHON_EXE% -m uvicorn app:app --host 127.0.0.1 --port 8001

pause
