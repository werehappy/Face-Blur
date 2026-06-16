# FACEBLUR v1.1 — Release Notes

**Released by werehappy**

---

## What is FACEBLUR?

FACEBLUR is a desktop application that automatically detects and censors faces in video files. It uses YOLOv11 face detection with GPU acceleration support, preserves original audio, and outputs standard MP4 files. Optional whole-head detection extends coverage to the backs and sides of heads.

---

## Highlights

- **One-click installer** — automatically detects your GPU and installs the correct version (CPU or CUDA)
- **Three censor styles** — Blur, Pixelate, or Black Bar
- **Whole-head detection (optional)** — censors backs/sides of heads, not just faces, using robust person→head-region detection plus an optional dedicated `head.pt`
- **Motion-aware anti-flicker** — boxes follow camera motion and hold through missed detections, so coverage stays glued to the head on dynamic footage
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

**First launch:** The app downloads the YOLOv11 face model (~6 MB) and the matching torch build for your hardware. Internet connection required for first run only.

---

## Features

### Detection
- YOLOv11 face detection (nano / medium / large models)
- **Whole-head detection (optional):**
  - Person detection → head region (top-center, sized by shoulder width); robust to motion blur, helmets, side/cut-off faces — anyone with some torso in frame
  - Optional dedicated `head.pt` next to the app, used *in addition* (union) to catch heads with no body visible
  - Face, person→head, and head.pt boxes merged as a union — adding `head.pt` only adds coverage
  - **No double-masking:** a face already covered by a head/person box is not censored a second time, so each head gets a single clean censor region
- Edge-strip detection for partially out-of-frame faces
- CSRT face tracking between detection frames
- Motion-aware anti-flicker box smoothing (camera-motion compensation, per-track velocity coasting, snap-on-large-motion, evidence-graduated hold)
- Per-source debug overlay (cyan = face, yellow = person→head, red = head.pt) with per-source counts in the log
- Confidence-weighted blur intensity
- Downscale detection for faster processing

### Processing
- Frame skipping (detect every N frames)
- FP16 GPU inference
- Audio preservation via ffmpeg
- H.264 output encoding
- Batch file queue with status tracking
- Resume on cancel (partial temp file kept)
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
- Settings and window position saved between sessions
- Keyboard shortcuts
- Tips & shortcuts popup
- Reset to Defaults

---

## Recommended Settings for Whole-Head Mode

For dynamic, blurred, or partial footage (e.g. body-cam / CQB):

| Setting | Value |
|---|---|
| Frame skip | 1 |
| Detect scale | 1.00 |
| Confidence | ~0.25 or lower |
| Padding | 0.30+ |
| Show debug boxes | ON (first run, to confirm coverage) |

No public head model is trained on helmeted/blurred/tactical heads — for reliable
coverage on that kind of footage, fine-tune YOLOv8 on a few hundred labeled frames
from your own clips and drop the result in as `head.pt`.

---

## Known Issues

- Slider value display may not update visually on some systems (values are correct internally)
- Toast notifications require Windows 10/11 (PowerShell fallback)
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

*FACEBLUR v1.1 — made by werehappy*
