@echo off
REM DeepSeek OCR Setup Script for Windows with NVIDIA GPU
REM This script sets up the application for GPU-accelerated OCR

echo ========================================
echo DeepSeek OCR - Windows GPU Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

echo [1/6] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/6] Upgrading pip...
python -m pip install --upgrade pip

echo [4/6] Installing PyTorch with CUDA 11.8...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch with CUDA
    pause
    exit /b 1
)

echo [5/6] Installing GPU dependencies...
pip install -r requirements-gpu.txt
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies failed to install
    echo You may need to install flash-attn manually
)

echo [6/6] Setting up configuration...
if not exist .env (
    copy .env.example .env
    echo Configuration file created: .env
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo IMPORTANT: Edit .env file and set:
echo   DEVICE_MODE=gpu
echo   USE_FLASH_ATTENTION=true  (if flash-attn installed successfully)
echo.
echo To download the model (recommended):
echo   python download_model.py
echo.
echo To start the application:
echo   python app.py
echo.
echo The app will run at http://localhost:5001
echo.
pause
