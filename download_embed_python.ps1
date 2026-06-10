# Prepares a bundled, pip-capable embeddable Python in .\faceblur_env
# This folder is shipped by the installer. On first run, FACEBLUR uses it to
# pip-install the correct torch build for the user's GPU.
#
# IMPORTANT: the Python version here MUST match the version used to build the
# exe (build_installer.bat uses python 3.10), so torch's cp310 wheels match
# the frozen app's ABI.
#
# Usage:  powershell -ExecutionPolicy Bypass -File download_embed_python.ps1

$ErrorActionPreference = "Stop"

$PyVersion = "3.10.11"
$EmbedUrl  = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"

$Root    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EnvDir  = Join-Path $Root "faceblur_env"
$Zip     = Join-Path $Root "python-embed.zip"
$GetPip  = Join-Path $EnvDir "get-pip.py"
$PthFile = Join-Path $EnvDir "python310._pth"
$PyExe   = Join-Path $EnvDir "python.exe"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Preparing bundled Python (embeddable $PyVersion)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Skip if already prepared (pip present)
if (Test-Path (Join-Path $EnvDir "Scripts\pip.exe")) {
    Write-Host "  faceblur_env already prepared. Skipping." -ForegroundColor Green
    exit 0
}

# Clean any partial previous attempt
if (Test-Path $EnvDir) {
    Write-Host "  Removing incomplete faceblur_env..."
    Remove-Item -Recurse -Force $EnvDir
}
New-Item -ItemType Directory -Force -Path $EnvDir | Out-Null

# 1. Download + extract embeddable python
Write-Host "  Downloading embeddable Python..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $EmbedUrl -OutFile $Zip
Write-Host "  Extracting..."
Expand-Archive -Path $Zip -DestinationPath $EnvDir -Force
Remove-Item $Zip -Force

# 2. Enable site-packages so pip works and installs land in Lib\site-packages
#    The embeddable distro ships python310._pth with "import site" commented out.
Write-Host "  Configuring python310._pth..."
if (-not (Test-Path $PthFile)) {
    throw "python310._pth not found - embeddable layout changed; check $PyVersion"
}
@"
python310.zip
.
Lib\site-packages

# Enable site so pip and added paths work
import site
"@ | Set-Content -Encoding ASCII $PthFile

New-Item -ItemType Directory -Force -Path (Join-Path $EnvDir "Lib\site-packages") | Out-Null

# 3. Bootstrap pip
Write-Host "  Downloading get-pip.py..."
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPip
Write-Host "  Installing pip into faceblur_env..."
& $PyExe $GetPip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed (exit $LASTEXITCODE)" }
Remove-Item $GetPip -Force

# 4. Sanity check
& $PyExe -m pip --version
if ($LASTEXITCODE -ne 0) { throw "pip not working in faceblur_env" }

Write-Host ""
Write-Host "  Done. faceblur_env is ready (python + pip, no torch yet)." -ForegroundColor Green
Write-Host "  torch will be installed on the user's machine at first run." -ForegroundColor Green
