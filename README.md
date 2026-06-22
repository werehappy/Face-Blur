# FACEBLUR v1.2
**Automated face censoring application**
Made by werehappy

---

## Overview

FACEBLUR is a desktop application that automatically detects and censors faces in video files using YOLOv11 face detection. It supports optional whole-head detection (catches the back and sides of the head, not just the face), multiple censor styles, GPU acceleration, batch processing, and audio preservation.

---

## Features

### Detection
- **YOLOv11 face detection** — nano / medium / large model options
- **Whole-head detection** — optional pass that also censors heads the face model misses (backs/sides, partial heads). By default it uses **person detection** and censors the head *region* of each detected person, which is far more robust on hard footage (motion blur, helmets, partial bodies) than a head model. If you drop a dedicated head model in as `head.pt`, it is used *in addition* to the person method. All results are merged with the face boxes (union), so it only adds coverage. Toggle via the **Detect whole head** checkbox in OPTIONS. See *Whole-Head Detection Setup* below.
- **No double-masking** — when whole-head mode is on, a face that's already covered by a head/person box is not censored a second time. The face box is only kept (and grown to head size) for heads the head/person passes genuinely missed, so each head gets exactly one censor region instead of an overlapping pair.
- **Edge strip detection** — additional YOLO pass on frame borders to catch partially out-of-frame faces
- **Face tracking (CSRT)** — smooth box interpolation between detection frames, eliminates flickering
- **Motion-aware box smoothing** — detections become tracks that follow **camera motion** (global frame-to-frame shift estimated by phase correlation, ~4 ms/frame) and their own velocity, so held boxes stay glued to the head during fast pans instead of drifting onto walls. Position jitter is damped, but large real movement snaps instantly (no lag). Hold time is **graduated by evidence**: a 1-frame false positive disappears after ~3 frames, while a repeatedly-detected head earns up to 8 frames of blind coverage — detections every 2nd–3rd frame produce continuous, flicker-free cover. Toggle via **Smooth boxes (anti-flicker)** in OPTIONS (default ON)
- **Per-source debug** — with **Show debug boxes** + whole-head mode on, thin outlines show which detector found each head: cyan = face model, yellow = person→head region, red = `head.pt`; the log prints per-source counts per file
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
| Detect whole head | OFF | Turn ON to also censor backs/sides of heads (slower) |
| Debug boxes | OFF | Clean output for production |
| Smooth boxes | ON | Anti-flicker: smooths box motion and holds boxes through missed detections |

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
pip install ultralytics opencv-contrib-python "numpy<2" pyinstaller dill win10toast pillow
```
Notes:
- Use **opencv-contrib-python** (not `opencv-python`) — the face tracker uses
  `cv2.legacy.TrackerCSRT_create`, which only exists in the contrib build.
  Don't install both; they conflict.
- Pin **numpy<2**. opencv/torch binaries in this stack are built against numpy 1.x,
  and a numpy 2.x in the same environment causes native crashes (and makes
  PyInstaller silently drop `cv2`).
- **torch is not installed here.** The app downloads the matching CPU/CUDA build
  on first run; `build_installer.bat` installs CPU torch into the build env only
  (for analysis) and excludes it from the exe.

---

## Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv11 face detection; COCO `yolo11n` person detector for whole-head mode (or a dedicated `head.pt` if provided) |
| `opencv-contrib-python` | Video I/O, frame processing, CSRT tracking (contrib build required for `cv2.legacy` trackers) |
| `torch` | Neural network inference (CPU or CUDA) — downloaded on first run, not bundled |
| `numpy` | Array operations — **pin `numpy<2`** for binary compatibility with opencv/torch |
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

## Whole-Head Detection Setup

The **Detect whole head** option (OPTIONS section) runs a second, dedicated
head-detection model alongside the face model and merges the results. Unlike a
face model, a head model fires on the head itself — so it catches the back and
sides of a head, and works even when only the head/shoulders are in frame.

**How whole-head detection works (both methods together, union):**

1. **Person detection → head region (always on).** The COCO person detector
   (`yolo11n`, auto-downloaded) finds each person and censors the head region
   (top-center, sized by shoulder width). Robust to motion blur, helmets,
   fully-side faces and cut-off faces — anyone with some torso in frame.
2. **A dedicated `head.pt` next to the app (used in addition, if present).**
   A real head model fires on the head itself, so it catches heads with **no
   body visible** — the one case the person method can't reach. This pass runs
   at full resolution with edge strips, regardless of the Detect scale slider.
   Put a **CrowdHuman-trained** model here (see `test_models.py` to pick one for
   your footage), or later a model **fine-tuned on your own clips**.

The results of both passes are merged with the face boxes (union), so adding a
`head.pt` can only **add** coverage. A face box that already falls inside a
head/person box is dropped rather than censored twice, so heads get a single
clean censor region. If neither head/person model loads, whole-head mode behaves
like normal face-only detection.

**Tuning for hard footage (dynamic camera, motion blur, cut-off heads).** Such
footage (e.g. CQB/body-cam) benefits from:

| Setting | Value | Why |
|---|---|---|
| Frame skip | 1 | Detect every frame; tracking is unreliable under fast camera motion |
| Detect scale | 1.00 | Full resolution; blurry/partial heads need every pixel |
| Confidence | ~0.25 or lower | For privacy, over-cover; recall matters more than precision |
| Padding | 0.30+ | The head region is an estimate — extra padding guarantees coverage |

The head/person pass also runs at a lower confidence floor than the face pass
automatically (`HEAD_CONF_DROP`, default 0.15).

**Verifying it works.** With **Detect whole head** on, the log shows the method
(`person -> head region` or `dedicated head model (head.pt)`) and, on the first
frame, `Head model alone: N head(s)`. Turn on **Show debug boxes** to see what's
being added. Two standalone diagnostics ship alongside the app:

- `test_detect.py path\to\clip.mp4` — compares person vs head detection on your
  footage (blue = person, yellow = head region, green = head model).
- `test_head.py path\to\clip.mp4` — runs just a head model.

**Best results on out-of-distribution footage: fine-tune.** No public model is
trained on helmeted/blurred/tactical heads. For reliable detection, label a few
hundred frames from your own clips and fine-tune YOLOv8, then save the result as
`head.pt`. ultralytics makes the training a few lines; the effort is the labeling.

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
| v1.0.1 | 2026 | Reduced the size of installer |
| v1.1 | 2026 | Whole-head detection (optional): person→head-region unioned with an optional `head.pt`, with no double-masking; motion-aware anti-flicker box smoothing (camera-motion compensation, velocity coasting, evidence-graduated hold); per-source debug overlay and counts; CSRT tracking fixes. |
| v1.2 | 2026 | **Bundled a fine-tuned `head.pt`.** A dedicated head detector was trained on custom footage and now ships with the app, so whole-head mode catches backs/sides/partial heads out of the box instead of relying on the person→head-region estimate alone. The model was produced by **labeling a few hundred frames** from real target clips (the helmeted / motion-blurred / cut-off heads no public model covers) and **fine-tuning YOLO** on them — labeling is the bulk of the work; the training itself is a few lines (see *Whole-Head Detection Setup → Best results on out-of-distribution footage: fine-tune*). Build scripts (`build.bat`, `build_installer.bat`, `installer.iss`) now install `head.pt` next to the exe, where a user-supplied `head.pt` still overrides it. Recommended for hard footage: Frame skip 1, Detect scale 1.00, Confidence ~0.25, Padding 0.30+. |

---

*FACEBLUR v1.2 — made by werehappy*
