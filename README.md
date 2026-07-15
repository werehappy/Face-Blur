# FACEBLUR v1.4
**Automated head & face censoring application**
Made by werehappy

---

## Overview

FACEBLUR is a desktop application that automatically detects and censors people's heads in video files. As of v1.4 the primary detector is a **fine-tuned head model** that fires on the head itself — front, back, side, helmeted, or motion-blurred — so coverage no longer depends on a face being visible. Three size-selectable head models ship with the app. Two optional aids (a person→head *rescue* prior and edge-strip inference) and an optional face safety net can each be switched on independently. It supports multiple censor styles, GPU acceleration, batch processing, and audio preservation.

---

## Features

### Detection
- **Head detection (primary)** — a **fine-tuned head model** is the main and, by default, only detector. It fires on the head itself, so it catches backs, sides, partial heads, helmets, and motion-blurred heads, and works even when no face or no body is visible. Three sizes ship with the app and are selectable in OPTIONS (see *Head model sizes*). On by default.
- **Head model sizes** — a **Head model size** selector (OPTIONS) chooses which bundled detector runs: `head_n.pt` (nano, fastest), `head_s.pt` (small, balanced — default), or `head_m.pt` (medium, most accurate, slower). All three are fine-tuned on the same data; pick by the speed-vs-accuracy trade for your hardware (medium is heavy on CPU). The choice persists across sessions.
- **Person→head aid (rescue)** — *optional, off by default.* A COCO person detector estimates a head *region* from each body and uses those regions as a **soft spatial prior** that rescues weak head detections: the head model is run at a low candidate floor, and a low-confidence head box that overlaps a person-derived region is boosted past the operating point, while an isolated low-confidence box is not. Crucially, the person→head regions are **not** added as boxes themselves, so a mislocalized estimate (e.g. on forward-held gear) can no longer paint a censor on its own. In testing this raised recall by up to ~3 points at a small precision cost. Toggle via **Person→head aid (rescue)** in OPTIONS.
- **Edge strip detection** — *optional, off by default.* Runs the head model over the four frame borders at full resolution to recover heads cut off at the edge of frame. Toggle via **Edge strip detection** in OPTIONS.
- **Face safety net** — *optional, off by default.* Runs a YOLOv11 face model and keeps a face box (grown to head size) only where no head box already covers it, so it fills genuine head-model misses without double-masking. Toggle via **Face safety net** in OPTIONS.
- **No double-masking** — a face or region that's already covered by a head box is not censored a second time; each head gets exactly one censor region.
- **Matched inference size** — the head model runs at a fixed inference size (`HEAD_INFER_IMGSZ`, default **960**) that must equal the size it was trained at. A train/inference size mismatch was the cause of scale-dependent false positives; keeping them matched fixes it.
- **Face tracking (CSRT)** — smooth box interpolation between detection frames, eliminates flickering
- **Motion-aware box smoothing** — detections become tracks that follow **camera motion** (global frame-to-frame shift estimated by phase correlation, ~4 ms/frame) and their own velocity, so held boxes stay glued to the head during fast pans instead of drifting onto walls. Position jitter is damped, but large real movement snaps instantly (no lag). Hold time is **graduated by evidence**: a 1-frame false positive disappears after ~3 frames, while a repeatedly-detected head earns up to 8 frames of blind coverage. Toggle via **Smooth boxes (anti-flicker)** in OPTIONS (default ON)
- **Per-source debug** — with **Show debug boxes** on, thin outlines show which source produced each box: red = head model, yellow = person→head rescue region (when the aid is on), cyan = face safety net (when on); the log prints per-source counts per file
- **Confidence heatmap** — blur intensity scales with detection confidence
- **Downscale detection** — the face safety net can run on reduced resolution for speed; the head pass always runs at full resolution
- **Frame skipping** — runs detection every N frames, uses tracker between detections

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
- **Splash screen** — shown immediately on launch while loading in background. During the one-time first-run torch download it shows a moving busy bar with an elapsed clock (e.g. `Downloading GPU libraries (one-time, ~2.5GB)  (3:42)`) so it is clearly alive, not frozen.
- **GPU/CPU indicator** — top-right shows active device (green = GPU, orange = CPU)
- **Collapsible sections** — PARAMETERS / OPTIONS / PERFORMANCE collapse to save space
- **Video thumbnail preview** — shows frame from selected file
- **First frame preview** — runs detection on frame 1 before processing starts
- **Head model size selector** — nano / small / medium dropdown in OPTIONS for the head detector
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
- `build.bat` — fast dev build with console mode for testing (copies `head_n/s/m.pt` next to the exe)
- `build_installer.bat` — builds the exe and compiles the Inno Setup installer (requires `head_n/s/m.pt` present)
- `installer.iss` — Inno Setup script with GPU auto-detection; bundles the three head models
- `install_torch.ps1` — auto-detects CUDA version, installs correct torch build
- `download_ffmpeg.ps1` — downloads and extracts ffmpeg.exe

### Head-model training pipeline (for fine-tuning your own head detector)
- `sample_frames.py` — pull diverse frames from clips (accepts a folder, with `-r` for subfolders)
- `bootstrap_labels.py` — pre-label frames so you correct instead of draw
- `split_dataset.py` — arrange labels into a clip-level train/val split + `data.yaml`
- `check_split.py` — verify the split is clip-level with no leakage; reports clips per split
- `fix_split.py` — repair a leaked split by moving whole clips (images + labels) between splits
- `train_head.py` — two-phase (hyperparameter search → long final run) training for one model
- `train_all.py` — train `head_n.pt`, `head_s.pt`, `head_m.pt` back-to-back, unattended, with a comparison summary
- `diagnose_heads.py` — run a model over a clip/image at a chosen `imgsz` to catch train/inference mismatch
- `HEAD_MODEL_PIPELINE.pdf` / `.md` — the full end-to-end walkthrough

---

## Recommended Settings

| Parameter | Value | Notes |
|---|---|---|
| Confidence | 0.40 | Good balance of detection vs false positives |
| Padding | 0.25 | Covers hair, chin, and ear edges |
| Blur kernel | 51 | Strong enough to be unrecognizable |
| Pixel size | 15 | Clear pixelation effect |
| Frame skip | 2 | 2x faster, barely noticeable |
| Detect scale | 0.50 | Speeds up the optional face net; head pass is always full-res |
| Head detection | ON | Primary detector (default) |
| Head model size | small | nano = fastest, medium = most accurate/slowest |
| Person→head aid | OFF | Turn ON to rescue weak head detections (recall boost, small precision cost) |
| Edge strip | OFF | Turn ON to catch heads cut off at frame borders |
| Face safety net | OFF | Turn ON to also cover heads the head model misses via a face model |
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
├── head_n.pt                 # Fine-tuned head model (nano)
├── head_s.pt                 # Fine-tuned head model (small, default)
├── head_m.pt                 # Fine-tuned head model (medium)
├── sample_frames.py          # Pipeline: sample frames to label
├── bootstrap_labels.py       # Pipeline: pre-label frames
├── split_dataset.py          # Pipeline: clip-level train/val split
├── check_split.py            # Pipeline: verify split (no leakage)
├── fix_split.py              # Pipeline: repair a leaked split
├── train_head.py             # Pipeline: train one model (tune + final)
├── train_all.py              # Pipeline: train n/s/m back-to-back
├── diagnose_heads.py         # Pipeline: test a model at a given imgsz
└── dist/
    ├── FACEBLUR.exe          # Single exe (torch downloaded on first run)
    ├── head_n.pt / head_s.pt / head_m.pt   # Copied next to the exe
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
| `ultralytics` | YOLOv11 head detection (bundled fine-tuned `head_n/s/m.pt`, primary); COCO `yolo11n` person detector for the optional person→head rescue aid; YOLOv11 face model for the optional face safety net |
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

## Head Detection Setup

Head detection is the app's **primary method** and is on by default. A dedicated
head model fires on the head itself — so it catches the back and sides of a
head, and works even when only the head/shoulders (or nothing but the head) are
in frame. Three optional aids can each be switched on independently in OPTIONS.

**How detection works (v1.4):**

1. **Fine-tuned head model (primary).** FACEBLUR ships three fine-tuned head
   detectors and you pick one with the **Head model size** selector:
   `head_n.pt` (nano), `head_s.pt` (small, default), `head_m.pt` (medium). The
   chosen model runs at full resolution at a fixed inference size
   (`HEAD_INFER_IMGSZ`, default 960) that matches how the models were trained.
   This is the only detector that runs unless you enable an aid.
2. **Person→head aid — rescue (optional).** A COCO person detector (`yolo11n`,
   auto-downloaded) estimates a head region from each body (top-center, sized by
   shoulder width). These regions are used as a **soft prior**: the head model
   is run at a low candidate floor, and a weak head detection that overlaps a
   region is boosted past the operating point (`conf × (1 + overlap)`), while an
   isolated weak detection is discarded. The regions are **never added as boxes
   themselves**, so a region that lands on forward-held gear (e.g. a weapon
   illuminator) cannot create a censor on its own — it can only reinforce a real
   but low-confidence head detection. This recovers hard heads (blurred,
   helmeted, partial) at a modest precision cost.
3. **Edge strips (optional).** The head model additionally runs over the four
   frame borders at full resolution, recovering heads truncated at the edge.
4. **Face safety net (optional).** A face model runs and its boxes are kept
   (grown to head size) only where no head box already covers them, filling
   genuine head-model misses. A face already inside a head box is dropped rather
   than censored twice.

> **Important: inference size must match training size.** The head models are
> trained at 960 and the app runs them at `HEAD_INFER_IMGSZ = 960`. If you
> retrain at a different size (e.g. 1280 for better small/distant-head recall),
> you **must** set `HEAD_INFER_IMGSZ` to the same value, or scale-mismatch false
> positives return. A larger size also makes the head pass slower — noticeably on
> CPU. Use `diagnose_heads.py` to test a model at a given size before shipping.

**Tuning for hard footage (dynamic camera, motion blur, cut-off heads).** Such
footage (e.g. CQB/body-cam, or dense crowds) benefits from:

| Setting | Value | Why |
|---|---|---|
| Frame skip | 1 | Detect every frame; tracking is unreliable under fast camera motion |
| Detect scale | 1.00 | Full resolution; blurry/partial heads need every pixel |
| Confidence | ~0.25 or lower | For privacy, over-cover; recall matters more than precision |
| Padding | 0.30+ | Extra padding guarantees coverage around the detected head |
| Person→head aid | ON | Rescues weak head detections on blurred/partial heads |
| Edge strip | ON | If heads are frequently cut off at frame borders |
| Head model size | small or medium | Larger = better on crowded/occluded heads, slower |

The head pass also runs at a lower confidence floor than the face net
automatically (`HEAD_CONF_DROP`, default 0.15); when the person→head aid is on,
candidates are collected down to a rescue floor (`PRIOR_RESCUE_FLOOR`, 0.10).

**Verifying it works.** The log names the head model in use (e.g. `Head model:
using head_s.pt`) and reports which aids are active. Turn on **Show debug boxes**
to see what each source adds (red = head model, yellow = person→head rescue
region when the aid is on, cyan = face safety net when on).

**Training your own head models.** No public model is trained on helmeted /
blurred / tactical or dense-crowd heads — the only reliable fix is fine-tuning on
frames from your own footage. The full procedure is in **`HEAD_MODEL_PIPELINE.pdf`
/ `.md`**; in short:

1. `sample_frames.py footage_folder -r --out dataset_raw` — pull diverse frames.
2. `bootstrap_labels.py dataset_raw` then correct in Yolo_Label (one class: head).
3. `split_dataset.py dataset_raw` → `check_split.py` (verify clip-level, no leakage;
   `fix_split.py` repairs leaks).
4. `train_all.py --data head_dataset/data.yaml` — trains `head_n/s/m.pt`
   back-to-back (or `train_head.py` for one model with a hyperparameter search).
5. Drop the resulting `head_n/s/m.pt` in the project root and rebuild.

Train at imgsz 960 to match the app (or change `HEAD_INFER_IMGSZ` to match). If you
have multiple footage domains (e.g. helmet-cam and crowds), balance them by head
count, keep a held-out test clip per domain, and measure recall per domain — a
pooled score hides one domain underperforming behind another.

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
| v1.2.1 | 2026 | **Fixed flaky / slow first launch.** The splash screen could intermittently hang on startup — requiring several reopens, or taking so long the user gave up — because the background loader updated the splash from a worker thread. Tkinter is single-threaded, so the cross-thread widget calls (and `update()`) raced the main `mainloop()` and occasionally deadlocked the window or left the Tcl interpreter in a bad state, most often during the one-time torch download on first run. `set_status` now marshals all splash updates onto the Tk thread via `after()` instead of touching widgets directly. The genuine first-run torch download still takes a few minutes, but the splash stays responsive and launches are now reliable. |
| v1.3 | 2026 | **Size-selectable fine-tuned head models + smarter whole-head detection.** Ships three fine-tuned head detectors (`head_n.pt` / `head_s.pt` / `head_m.pt`), selectable by size in OPTIONS (nano/small/medium — speed vs accuracy). When a fine-tuned head model is loaded it is now the **primary** head detector and the person→head-region geometry pass is **disabled** — that estimate was firing on forward-held gear (e.g. weapon illuminators), so demoting it removes those false positives; the person→head pass remains only as a fallback when no head model is present (tunable via `PERSON_HEAD_MODE`). The head model now runs at a fixed inference size (`HEAD_INFER_IMGSZ`, 960) that **must match its training size**, fixing scale-mismatch false positives. New training pipeline and helper scripts (`sample_frames.py` folder support, `check_split.py`, `fix_split.py`, `diagnose_heads.py`, `train_head.py`, `train_all.py`) plus an updated `HEAD_MODEL_PIPELINE` doc cover labeling, clip-level splitting, per-domain balance, and training all three sizes at once. The first-run splash now shows a moving busy bar with an elapsed clock during the one-time torch download (the GPU build is ~2.5GB) so it is clearly alive. Build scripts and the installer now bundle all three head models next to the exe. |
| v1.4 | 2026 | **Head detection is now the primary method, with independent opt-in aids.** The fine-tuned head model is the main and, by default, only detector; the face model is no longer always on. Three aids can each be toggled independently in OPTIONS, all **off by default**: (1) a **person→head rescue aid** that uses person-derived head regions as a *soft prior* to boost weak-but-overlapping head detections past the operating point — regions are never added as standalone boxes, so misplaced estimates on forward-held gear can no longer create a censor by themselves (this replaces the old all-or-nothing person→head union/disable policy and, in measured evaluation, recovered most of the recall of the raw union while giving back most of its precision cost); (2) **edge-strip inference** over the four frame borders; and (3) a **face safety net** that fills genuine head-model misses without double-masking. The old "Detect whole head" checkbox and the `PERSON_HEAD_MODE` auto-policy are retired in favor of the explicit toggles. All settings persist across sessions. |

---

*FACEBLUR v1.4 — made by werehappy*
