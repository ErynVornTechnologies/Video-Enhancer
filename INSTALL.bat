@echo off
:: ============================================================
::  BORING STUFF #08 - Video Enhancer - Installer (Windows 10/11)
::  Installs: Python 3.12, FFmpeg, a local .venv with CUDA PyTorch,
::  spandrel + GUI packages, and downloads the model weights.
::  Your global Python packages are never touched.
::  Erynvorn Technologies | https://erynvorntechnologies.com
:: ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Video Enhancer - Installer

echo.
echo ============================================================
echo   VIDEO ENHANCER - automatic installer
echo   Needs: NVIDIA GPU, internet, about 6 GB of free disk space
echo ============================================================
echo.

echo [1/6] Checking Python 3.12...
py -3.12 -c "import sys" >nul 2>&1
if %errorlevel% neq 0 (
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo       [ERROR] Python 3.12 and winget are both missing.
        echo       Install Python 3.12 from python.org, then run this file again.
        pause & exit /b 1
    )
    echo       Installing Python 3.12 with winget...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo       [!] Close this window and run INSTALL.bat again.
    pause & exit /b 0
)
py -3.12 --version

echo.
echo [2/6] Checking FFmpeg...
set "HAVE_FF=0"
if exist "C:\ffmpeg\bin\ffmpeg.exe" set "HAVE_FF=1"
ffmpeg -version >nul 2>&1 && set "HAVE_FF=1"
if "%HAVE_FF%"=="1" (
    echo       FFmpeg found.
) else (
    echo       Installing FFmpeg with winget...
    winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    echo       [!] FFmpeg is added to PATH for NEW windows only.
)

echo.
echo [3/6] Checking NVIDIA GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
if %errorlevel% neq 0 (
    echo       [WARNING] nvidia-smi not found. A CUDA-capable NVIDIA GPU and
    echo       a current driver are required; CPU mode would be unusably slow.
)

echo.
echo [4/6] Creating local environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 ( echo       [ERROR] venv creation failed. & pause & exit /b 1 )
)
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m pip install --quiet --upgrade pip

echo.
echo [5/6] Installing packages (PyTorch is about 3 GB, be patient)...
:: ORDER MATTERS: CUDA torch first from the PyTorch index. If spandrel is
:: installed first, pip pulls a CPU-only torch from PyPI instead.
"%PY%" -m pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 ( echo       [ERROR] PyTorch install failed. & pause & exit /b 1 )
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo       [ERROR] package install failed. & pause & exit /b 1 )

echo.
echo [6/6] Downloading model weights and verifying...
"%PY%" -c "from model import ENGINES, ensure_weights; [print('  model:', ensure_weights(k)) for k in ENGINES]"
"%PY%" -c "import torch, spandrel, customtkinter; print('  torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

echo.
echo [audio 1/2] DeepFilterNet voice denoiser (standalone, 27 MB, CPU)...
if not exist "tools\deep-filter.exe" (
    if not exist "tools" mkdir tools
    curl.exe -L --fail -o "tools\deep-filter.exe" "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-pc-windows-msvc.exe"
    if errorlevel 1 echo       [WARNING] download failed: audio mode "voice" will be unavailable.
) else (
    echo       already installed.
)

echo.
echo [audio 2/2] ClearerVoice HQ voice model (optional, ~4 GB, own environment)
echo       Needed only for audio mode "voice-hq". Skipped automatically in 20 s.
choice /C YN /T 20 /D N /M "      Install ClearerVoice now"
if errorlevel 2 goto :skip_cv
if not exist ".venv_audio\Scripts\python.exe" py -3.12 -m venv .venv_audio
set "PYA=%~dp0.venv_audio\Scripts\python.exe"
"%PYA%" -m pip install --quiet --upgrade pip
:: same rule as above: CUDA torch first, or clearvoice pulls a CPU build
"%PYA%" -m pip install "torch==2.11.0" "torchvision==0.26.0" torchaudio --index-url https://download.pytorch.org/whl/cu128
"%PYA%" -m pip install clearvoice
:: fetch + verify the weights now (cv_worker refuses to run on random weights)
"%PYA%" -c "import sys; sys.argv=['x']; import cv_worker as w, os; os.makedirs(w.MODELS, exist_ok=True); os.chdir(w.MODELS); w.ensure_weights('MossFormer2_SE_48K'); print('      ClearerVoice weights OK')"
:skip_cv

echo.
echo ============================================================
echo   INSTALLATION FINISHED
echo   Start the app:   run.bat
echo   Self-test:       .venv\Scripts\python.exe test_enhancer.py
echo.
echo   Boring Stuff by Erynvorn Technologies
echo   https://erynvorntechnologies.com
echo   https://github.com/ErynVornTechnologies
echo ============================================================
echo.
pause
endlocal
