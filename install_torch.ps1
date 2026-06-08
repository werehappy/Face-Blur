# Detects CUDA version and installs the correct torch build
# Usage: install_torch.ps1 [pip_path]
# If pip_path not provided, uses 'pip' from PATH

param([string]$PipPath = "pip")

Write-Host "  Detecting GPU / CUDA..."

$cudaVersion = $null
$nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue

if ($nvidiaSmi) {
    try {
        $output = & nvidia-smi 2>&1 | Out-String
        if ($output -match "CUDA Version:\s*(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            $cudaVersion = "$major.$minor"
            Write-Host "  Found CUDA $cudaVersion"
        }
    } catch {}
}

if ($cudaVersion -eq $null) {
    Write-Host "  No CUDA GPU detected. Installing CPU-only torch..."
    & $PipPath install torch torchvision --index-url https://download.pytorch.org/whl/cpu --upgrade --force-reinstall -q
} elseif ([double]$cudaVersion -ge 12.1) {
    Write-Host "  Installing torch for CUDA 12.1+ (your CUDA: $cudaVersion)..."
    & $PipPath install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall -q
} elseif ([double]$cudaVersion -ge 11.8) {
    Write-Host "  Installing torch for CUDA 11.8 (your CUDA: $cudaVersion)..."
    & $PipPath install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall -q
} else {
    Write-Host "  CUDA $cudaVersion is older than 11.8. Installing CPU-only torch..."
    & $PipPath install torch torchvision --index-url https://download.pytorch.org/whl/cpu --upgrade --force-reinstall -q
}

Write-Host "  Torch install complete."
