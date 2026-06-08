# FACEBLUR v1.0 — Release Notes

**Released by werehappy**

---

## What is FACEBLUR?

FACEBLUR is a desktop application that automatically detects and censors faces in video files. It uses YOLOv11 face detection with GPU acceleration support, preserves original audio, and outputs standard MP4 files.

---

## Highlights

- **One-click installer** — automatically detects your GPU and installs the correct version (CPU or CUDA)
- **Three censor styles** — Blur, Pixelate, or Black Bar
- **Audio preserved** — output video keeps the original audio track
- **Batch processing** — process multiple videos in one session
- **Fast processing** — frame skipping + downscale detection + GPU FP16 support

---

## What's Included

| File | Description |
|---|---|
| `FACEBLUR_Setup.exe` | Windows installer (CPU + GPU versions bundled) |

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 10/11 64-bit |
| RAM | 4 GB | 8 GB+ |
| GPU | None (CPU mode) | Nvidia GTX 1060+ with CUDA 11.8+ |
| Storage | 2 GB free | 4 GB free |
| Internet | Required (first run only) | — |

---

## Installation

1. Download `FACEBLUR_Setup.exe`
2. Run the installer
3. The installer automatically detects your GPU and installs the correct version
4. Launch from the desktop shortcut

**First launch:** The app will download the YOLOv11 face model (~6MB) on first use. Internet connection required.

---

## Features in v1.0

### Detection
- YOLOv11 face detection (nano / medium / large models)
- Edge strip detection for partially out-of-frame faces
- CSRT face tracking between detection frames
- Confidence-weighted blur intensity
- Downscale detection for faster processing

### Processing
- Frame skipping (detect every N frames)
- FP16 GPU inference
- Audio preservation via ffmpeg
- H.264 output encoding
- Batch file queue with status tracking
- Export processing report as JSON

### UI
- Splash screen
- GPU/CPU status indicator
- Collapsible settings sections
- Video thumbnail preview
- First frame detection preview
- Drag and drop file support
- ETA and FPS display during processing
- Windows toast notification on completion
- Settings saved between sessions
- Keyboard shortcuts
- Tips & shortcuts popup

---

## Known Issues

- Slider value display may not update visually on some systems (values are correct internally)
- Toast notifications require Windows 10/11
- Processing is sequential — multiple files processed one at a time

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl + O` | Add files |
| `Ctrl + Enter` | Start processing |
| `Escape` | Cancel |
| `Delete` | Remove selected file |

---

## Feedback & Bug Reports

Report issues at: **https://github.com/werehappy/Face-Blur/issues**

---

*FACEBLUR v1.0 — made by werehappy*
