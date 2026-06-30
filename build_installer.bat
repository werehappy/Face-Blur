@echo off
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo   FACEBLUR v1.3  --  Build Installer (small / no torch bundled)
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

:: ffmpeg
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

:: head models (size-selectable) - required, the installer ships them next to
:: the exe. Fail early with a clear message rather than a cryptic Inno error.
for %%M in (head_n.pt head_s.pt head_m.pt) do (
    if not exist "%~dp0%%M" (
        echo [ERROR] %%M not found next to this script.
        echo         Place head_n.pt, head_s.pt and head_m.pt in: %~dp0
        echo         ^(train them with train_all.py^)
        pause
        exit /b 1
    )
)
echo       head models OK.

:: Step 0: Build-env packages.
:: NOTE: we install CPU torch in the BUILD env only so PyInstaller can analyze
:: ultralytics. torch is then EXCLUDED from the exe, so it is never bundled.
echo.
echo [0/4] Installing build packages...
:: Hold numpy<2 across EVERY pip install in this build (incl. torch and the
:: GPU install_torch.ps1). opencv/torch wheels here are built for numpy 1.x;
:: letting numpy 2.x in crashes them and makes PyInstaller drop cv2. A pip
:: constraints file keeps the resolver from ever picking numpy 2.x, so numpy
:: is not deleted and re-downloaded on each build.
:: Relative path: pip splits PIP_CONSTRAINT on spaces, and the project path
:: may contain spaces. CWD is already this folder (cd above), so relative works.
>constraints.txt echo numpy^<2
set PIP_CONSTRAINT=constraints.txt

:: One clean opencv (CONTRIB build, for cv2.legacy.TrackerCSRT).
"%CONDA_PREFIX%\Scripts\pip.exe" uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless >nul 2>&1
"%CONDA_PREFIX%\Scripts\pip.exe" install ultralytics pyinstaller dill win10toast pillow
if errorlevel 1 ( echo [ERROR] package install failed. & pause & exit /b 1 )
:: CPU torch in the BUILD env only (so PyInstaller can analyze ultralytics).
:: torch is EXCLUDED from the exe.
"%CONDA_PREFIX%\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -q
:: opencv-contrib LAST (force) so its cv2 - including cv2.legacy - overrides
:: the opencv-python that ultralytics pulls in as a dependency.
:: ultralytics pulls in opencv-python; remove it so only the CONTRIB build is
:: registered. Two opencv distros over the same cv2 folder confuses
:: PyInstaller and makes it drop cv2 from the exe.
"%CONDA_PREFIX%\Scripts\pip.exe" uninstall -y opencv-python opencv-python-headless >nul 2>&1
"%CONDA_PREFIX%\Scripts\pip.exe" install --force-reinstall --no-deps opencv-contrib-python

:: Verify cv2 imports BEFORE building. If it cannot, PyInstaller silently
:: drops it and the exe dies at runtime with "No module named cv2".
echo       Verifying opencv...
"%CONDA_PREFIX%\python.exe" -c "import numpy,cv2; cv2.legacy.TrackerCSRT_create; print('cv2',cv2.__version__,'numpy',numpy.__version__)"
if errorlevel 1 (
    echo [ERROR] cv2 failed to import/verify in the build env.
    echo         Run:  pip install "numpy^<2" opencv-contrib-python   then retry.
    pause
    exit /b 1
)

:: Step 1: Build the single small exe (torch excluded)
echo.
echo [1/4] Building FACEBLUR.exe (torch excluded)...
taskkill /f /im FACEBLUR.exe >nul 2>&1
if exist "%~dp0dist\FACEBLUR.exe" del /f /q "%~dp0dist\FACEBLUR.exe"
"%CONDA_PREFIX%\Scripts\pyinstaller.exe" --onefile --noconsole --clean ^
    --collect-data ultralytics ^
    --hidden-import ultralytics ^
    --hidden-import PIL ^
    --hidden-import win10toast ^
    --hidden-import pickletools ^
    --hidden-import html.parser ^
    --collect-all cv2 ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --add-data "ffmpeg.exe;." ^
    --name FACEBLUR %SCRIPT%
if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)
echo       FACEBLUR.exe ready.

:: Step 2: Prepare bundled embeddable Python (pip-capable, no torch)
echo.
echo [2/4] Preparing bundled Python (faceblur_env)...
if not exist "%~dp0download_embed_python.ps1" (
    echo [ERROR] download_embed_python.ps1 not found.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_embed_python.ps1"
if errorlevel 1 (
    echo [ERROR] Embeddable Python prep failed.
    pause
    exit /b 1
)
echo       faceblur_env ready.

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
echo   (torch downloads on the user's first launch)
echo ==================================================
echo.
pause
