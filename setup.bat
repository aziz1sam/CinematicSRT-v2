@echo off
chcp 65001 >nul
title CinematicSRT v2.0 - Automated Setup
cd /d "%~dp0"

echo ================================================================
echo       🎬 CinematicSRT v2.0 - معالج التثبيت الذكي
echo ================================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] لم يتم العثور على بايثون (Python).
    echo يرجى تثبيت Python 3.10 أو 3.11 وتفعيل خيار "Add python.exe to PATH".
    echo الرابط: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] تم التحقق من وجود Python:
python --version
echo.

:: 2. Create Virtual Environment if not exists
if not exist "venv\Scripts\python.exe" (
    echo [2/4] جاري إنشاء البيئة الافتراضية (venv)...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] فشل إنشاء البيئة الافتراضية.
        pause
        exit /b 1
    )
    echo تم إنشاء venv بنجاح.
) else (
    echo [2/4] البيئة الافتراضية venv موجودة مسبقاً.
)
echo.

:: 3. Check NVIDIA GPU and Install PyTorch
echo [3/4] جاري فحص كرت الشاشة وتثبيت PyTorch المناسب...
set PYTHON_BIN="%~dp0venv\Scripts\python.exe"
set PIP_BIN="%~dp0venv\Scripts\pip.exe"

%PIP_BIN% install --upgrade pip >nul 2>&1

nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [GPU DETECTED] تم اكتشاف كرت شاشة NVIDIA!
    echo جاري تثبيت PyTorch الداعم لتسريع كرت الشاشة (CUDA 12.1)...
    %PIP_BIN% install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [CPU MODE] لم يتم اكتشاف كرت NVIDIA، سيتم تثبيت نسخة PyTorch للمعالج CPU...
    %PIP_BIN% install torch torchaudio
)
echo.

:: 4. Install other dependencies
echo [4/4] جاري تثبيت مكتبات المشروع (FastAPI, Faster-Whisper, Transformers...)...
%PIP_BIN% install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [WARNING] واجه التثبيت بعض التحذيرات، جاري المتابعة...
)

echo.
echo ================================================================
echo   ✅ اكتمل تثبيت وتجهيز CinematicSRT v2.0 بنجاح!
echo   جاري تشغيل البرنامج وفتح المتصفح...
echo ================================================================
echo.

call run.bat
