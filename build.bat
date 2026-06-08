@echo off
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   FACEBLUR  --  Dev Build
echo ==================================================
echo.

set ENV_NAME=faceblur
set PYTHON_VER=3.10
set SCRIPT=face_blur.py
set FFMPEG_EXE=ffmpeg.exe

:: DEV MODE: use --console so startup errors are visible
:: Change to --noconsole for release
set BUILD_MODE=--console

echo [1/4] Initializing conda...
set CONDA_ROOT=%USERPROFILE%\anaconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set CONDA_ROOT=%USERPROFILE%\miniconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set CONDA_ROOT=C:\ProgramData\anaconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" set CONDA_ROOT=C:\ProgramData\miniconda3
if not exist "%CONDA_ROOT%\Scripts\activate.bat" (
    echo [ERROR] Cannot find conda.
    pause & exit /b 1
)
echo       Found: %CONDA_ROOT%
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"

echo.
echo [2/4] Activating "%ENV_NAME%"...
call conda env list | findstr /C:"%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    echo       Creating new env...
    call conda create -n %ENV_NAME% python=%PYTHON_VER% -y
    if errorlevel 1 ( echo [ERROR] Create failed. & pause & exit /b 1 )
)
call conda activate %ENV_NAME%
if errorlevel 1 ( echo [ERROR] Activate failed. & pause & exit /b 1 )
echo       Activated: %CONDA_PREFIX%

echo.
echo [3/4] Checking packages (install only if missing)...
"%CONDA_PREFIX%\Scripts\python.exe" -c "import ultralytics, cv2, numpy, PIL, win10toast" >nul 2>&1
if errorlevel 1 (
    echo       Some packages missing, installing...
    if exist "%~dp0install_torch.ps1" (
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_torch.ps1" "%CONDA_PREFIX%\Scripts\pip.exe"
    )
    "%CONDA_PREFIX%\Scripts\pip.exe" install ultralytics opencv-python numpy pyinstaller dill win10toast pillow
    if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
) else (
    echo       All packages present, skipping install.
)

:: Check pyinstaller separately
"%CONDA_PREFIX%\Scripts\pyinstaller.exe" --version >nul 2>&1
if errorlevel 1 (
    echo       Installing pyinstaller...
    "%CONDA_PREFIX%\Scripts\pip.exe" install pyinstaller
)

echo.
echo [4/4] Building exe...
cd /d "%~dp0"

:: Check ffmpeg
if not exist "%FFMPEG_EXE%" (
    if exist "%~dp0download_ffmpeg.ps1" (
        echo       Downloading ffmpeg...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_ffmpeg.ps1"
    ) else (
        echo [WARN] ffmpeg.exe not found - audio merging will be skipped
    )
)

taskkill /f /im FACEBLUR.exe >nul 2>&1
if exist "%~dp0dist\FACEBLUR.exe" del /f /q "%~dp0dist\FACEBLUR.exe"

"%CONDA_PREFIX%\Scripts\pyinstaller.exe" --onefile %BUILD_MODE% --clean ^
    --collect-data ultralytics ^
    --hidden-import ultralytics ^
    --hidden-import PIL ^
    --hidden-import win10toast ^
    --add-data "ffmpeg.exe;." ^
    --name FACEBLUR %SCRIPT%

if errorlevel 1 (
    echo [ERROR] Build failed.
    pause & exit /b 1
)

echo.
echo ==================================================
echo   Done!  %~dp0dist\FACEBLUR.exe
echo   Running in CONSOLE mode - errors will be visible
echo   Change BUILD_MODE to --noconsole for release
echo ==================================================
echo.

:: Auto-run the exe for quick testing
set /p RUN="Run FACEBLUR.exe now? (y/n): "
if /i "%RUN%"=="y" start "" "%~dp0dist\FACEBLUR.exe"

pause
