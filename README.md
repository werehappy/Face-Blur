# FACEBLUR v1.0
**Automated face censoring application**
Made by werehappy

---

## Overview

FACEBLUR is a desktop application that automatically detects and censors faces in video files using YOLOv11 face detection. It supports multiple censor styles, GPU acceleration, batch processing, and audio preservation.

---

## Features

### Detection
- **YOLOv11 face detection** — nano / medium / large model options
- **Edge strip detection** — additional YOLO pass on frame borders to catch partially out-of-frame faces
- **Face tracking (CSRT)** — smooth box interpolation between detection frames, eliminates flickering
- **Confidence heatmap** — blur intensity scales with detection confidence
- **Downscale detection** — runs detection on reduced resolution for speed, scales boxes back up
- **Frame skipping** — runs YOLO every N frames, uses tracker between detections

### Censor Modes
- **Blur** — Gaussian blur (kernel size adjustable)
- **Pixelate** — classic pixel-block censor
- **Black Bar** — solid black rectangle

### Processing
- **Audio preservation** — ffmpeg merges original audio into output (XVID → H.264)
- **Batch processing** — queue multiple files, process sequentially
- **Processing queue** — per-file status icons: `[ ]` waiting → `[>>]` processing → `[OK]` done → `[!!]` failed
- **Resume on cancel** — partial temp file kept when cancelled
- **Export report** — saves `faceblur_report.json` with per-file stats after processing
- **FP16 inference** — half precision on GPU for ~2x faster inference

### UI
- **Splash screen** — shown immediately on launch while loading in background
- **GPU/CPU indicator** — top-right shows active device (green = GPU, orange = CPU)
- **Collapsible sections** — PARAMETERS / OPTIONS / PERFORMANCE collapse to save space
- **Video thumbnail preview** — shows frame from selected file
- **First frame preview** — runs detection on frame 1 before processing starts
- **Drag and drop** — drag video files directly onto the file list
- **Settings persistence** — all settings saved on close, restored on next launch
- **Window position memory** — remembers size and position between sessions
- **Keyboard shortcuts** — `Ctrl+O` add files, `Ctrl+Enter` process, `Escape` cancel, `Delete` remove file
- **ETA display** — estimated time remaining in H:MM:SS format
- **FPS display** — live frames per second during processing
- **Windows toast notification** — system notification when processing completes
- **Tips popup** — `? TIPS & SHORTCUTS` button opens tips window
- **Reset to Defaults** — one-click reset of all parameters

### Output
- **Auto output suffix** — `_blurred` / `_pixelated` / `_blackbar` based on selected mode
- **H.264 encoding** — ffmpeg re-encodes final output for maximum compatibility
- **Custom output folder** — optional, defaults to same folder as source

### Build & Distribution
- `build.bat` — fast dev build with console mode for testing
- `build_installer.bat` — builds CPU + GPU exes and compiles Inno Setup installer
- `installer.iss` — Inno Setup script with GPU auto-detection
- `install_torch.ps1` — auto-detects CUDA version, installs correct torch build
- `download_ffmpeg.ps1` — downloads and extracts ffmpeg.exe

---

## Recommended Settings

| Parameter | Value | Notes |
|---|---|---|
| Confidence | 0.40 | Good balance of detection vs false positives |
| Padding | 0.25 | Covers hair, chin, and ear edges |
| Blur kernel | 51 | Strong enough to be unrecognizable |
| Pixel size | 15 | Clear pixelation effect |
| Frame skip | 2 | 2x faster, barely noticeable |
| Detect scale | 0.50 | 2x faster detection, minimal accuracy loss |
| Edge strip | ON | Catches partial/out-of-frame faces |
| Debug boxes | OFF | Clean output for production |

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl + O` | Add files |
| `Ctrl + Enter` | Start processing |
| `Escape` | Cancel processing |
| `Delete` | Remove selected file from list |

---

## File Structure

```
project/
├── face_blur.py              # Main application
├── build.bat                 # Dev build script
├── build_installer.bat       # Installer build script
├── installer.iss             # Inno Setup script
├── install_torch.ps1         # CUDA auto-detection
├── download_ffmpeg.ps1       # ffmpeg downloader
├── ffmpeg.exe                # Bundled after first build
└── dist/
    ├── FACEBLUR_CPU.exe      # CPU-only build
    ├── FACEBLUR_GPU.exe      # CUDA GPU build
    └── (installer_output/)
        └── FACEBLUR_Setup.exe  # Final installer for distribution
```

---

## Build Instructions

### Development Build
```bash
# In Anaconda Prompt
conda activate faceblur
cd path\to\project
build.bat
```
Produces `dist\FACEBLUR.exe` in console mode — errors visible in terminal.

### Distribution Installer
```bash
build_installer.bat
```
Produces `installer_output\FACEBLUR_Setup.exe` with both CPU and GPU versions.
Requires Inno Setup 6 or 7 installed (`https://jrsoftware.org/isdl.php`).

### Environment Setup (first time)
```bash
conda create -n faceblur python=3.10 -y
conda activate faceblur
pip install ultralytics opencv-python numpy pyinstaller dill win10toast pillow
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv11 face detection |
| `opencv-python` | Video I/O, frame processing, CSRT tracking |
| `torch` | Neural network inference (CPU or CUDA) |
| `numpy` | Array operations |
| `pillow` | Thumbnail preview images |
| `ffmpeg` | Audio merging, H.264 encoding |
| `dill` | Multiprocessing serialization |
| `win10toast` | Windows toast notifications (optional) |
| `pyinstaller` | Build executable |

---

## GPU Support

The installer automatically detects the user's Nvidia GPU during installation and installs the appropriate version:

| User's CUDA | Installed Version |
|---|---|
| None / No Nvidia GPU | CPU version |
| CUDA 12.1+ | GPU version (cu121) |
| CUDA 11.8 | GPU version (cu118) |
| CUDA < 11.8 | CPU version (driver too old) |

If the GPU version is installed but CUDA is unavailable at runtime, the app displays a driver update prompt in the log.

---

## Known Issues / Pending

- [ ] Slider value display (visual refresh issue on some systems)
- [ ] Toast notification reliability (uses PowerShell fallback)
- [ ] Multi-threaded parallel file processing (sequential only currently)
- [ ] Named preset save/load profiles
- [ ] Per-file individual progress bars for batch processing

---

## Version History

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026 | Initial release |

---

*FACEBLUR v1.0 — made by werehappy*
