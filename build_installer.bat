@echo off
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   FACEBLUR  --  Build Installer
echo ==================================================
echo.

set ENV_NAME=faceblur
set SCRIPT=face_blur.py
set FFMPEG_EXE=ffmpeg.exe

:: Find Inno Setup ISCC.exe (checks v6 and v7, 32-bit and 64-bit paths)
set ISCC_PATH=
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe
if exist "C:\Program Files (x86)\Inno Setup 7\ISCC.exe" set ISCC_PATH=C:\Program Files (x86)\Inno Setup 7\ISCC.exe
if exist "C:\Program Files\Inno Setup 7\ISCC.exe" set ISCC_PATH=C:\Program Files\Inno Setup 7\ISCC.exe

:: Find conda
set CONDA_ROOT=%USERPROFILE%\anaconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set CONDA_ROOT=%USERPROFILE%\miniconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set CONDA_ROOT=C:\ProgramData\anaconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set CONDA_ROOT=C:\ProgramData\miniconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [ERROR] Cannot find conda.
    pause
    exit /b 1
)
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Cannot activate %ENV_NAME%.
    pause
    exit /b 1
)
echo       Using env: %CONDA_PREFIX%

cd /d "%~dp0"

:: Check ffmpeg
if not exist "%~dp0%FFMPEG_EXE%" (
    if not exist "%~dp0download_ffmpeg.ps1" (
        echo [ERROR] download_ffmpeg.ps1 not found.
        pause
        exit /b 1
    )
    echo       Downloading ffmpeg...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_ffmpeg.ps1"
    if errorlevel 1 (
        echo [ERROR] ffmpeg download failed.
        pause
        exit /b 1
    )
)
echo       ffmpeg OK.

:: Step 0: Install common packages
echo.
echo [0/4] Installing common packages...
"%CONDA_PREFIX%\Scripts\pip.exe" install ultralytics opencv-python numpy pyinstaller dill win10toast pillow

:: Step 1: Build CPU version
echo.
echo [1/4] Building CPU version...
"%CONDA_PREFIX%\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --upgrade -q
if errorlevel 1 (
    echo [ERROR] CPU torch install failed.
    pause
    exit /b 1
)
taskkill /f /im FACEBLUR_CPU.exe >nul 2>&1
if exist "%~dp0dist\FACEBLUR_CPU.exe" del /f /q "%~dp0dist\FACEBLUR_CPU.exe"
"%CONDA_PREFIX%\Scripts\pyinstaller.exe" --onefile --noconsole --clean --collect-data ultralytics --hidden-import ultralytics --add-data "ffmpeg.exe;." --name FACEBLUR_CPU %SCRIPT%
if errorlevel 1 (
    echo [ERROR] CPU build failed.
    pause
    exit /b 1
)
echo       CPU exe ready.

:: Step 2: Build GPU version
echo.
echo [2/4] Building GPU version...
if not exist "%~dp0install_torch.ps1" (
    echo [ERROR] install_torch.ps1 not found.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_torch.ps1" "%CONDA_PREFIX%\Scripts\pip.exe"
if errorlevel 1 (
    echo [ERROR] GPU torch install failed.
    pause
    exit /b 1
)
taskkill /f /im FACEBLUR_GPU.exe >nul 2>&1
if exist "%~dp0dist\FACEBLUR_GPU.exe" del /f /q "%~dp0dist\FACEBLUR_GPU.exe"
"%CONDA_PREFIX%\Scripts\pyinstaller.exe" --onefile --noconsole --clean --collect-data ultralytics --hidden-import ultralytics --add-data "ffmpeg.exe;." --name FACEBLUR_GPU %SCRIPT%
if errorlevel 1 (
    echo [ERROR] GPU build failed.
    pause
    exit /b 1
)
echo       GPU exe ready.

:: Step 3: Check Inno Setup
echo.
echo [3/4] Checking Inno Setup...
if "%ISCC_PATH%"=="" (
    echo [ERROR] Inno Setup not found.
    echo         Download from: https://jrsoftware.org/isdl.php
    echo         Install with default settings then run this bat again.
    pause
    exit /b 1
)
echo       Found: %ISCC_PATH%

:: Step 4: Compile installer
echo.
echo [4/4] Compiling installer...
if not exist "%~dp0installer_output" mkdir "%~dp0installer_output"
"%ISCC_PATH%" "%~dp0installer.iss"
if errorlevel 1 (
    echo [ERROR] Inno Setup compile failed.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   Done!
echo   Distribute: %~dp0installer_output\FACEBLUR_Setup.exe
echo ==================================================
echo.
pause
