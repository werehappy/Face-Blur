$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$zip = Join-Path $PSScriptRoot "ffmpeg_dl.zip"
$exe = Join-Path $PSScriptRoot "ffmpeg.exe"

Write-Host "  Downloading ffmpeg..."
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing

Write-Host "  Extracting ffmpeg.exe..."
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z = [System.IO.Compression.ZipFile]::OpenRead($zip)
$entry = $z.Entries | Where-Object { $_.Name -eq "ffmpeg.exe" } | Select-Object -First 1
[System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $exe, $true)
$z.Dispose()

Remove-Item $zip
Write-Host "  ffmpeg.exe ready."
