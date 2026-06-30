# FACEBLUR v1.3 — Release Notes

*Made by werehappy*

This release makes whole-head detection accurate enough to rely on — driven by
bundled, size-selectable fine-tuned head models — and removes the false
positives that the older person→head geometry could produce. It also adds a full
training pipeline for building your own head models, and makes first launch
clearly show progress.

---

## Highlights

- **Size-selectable head models.** Three fine-tuned head detectors now ship with
  the app — nano, small, and medium — chosen from a new **Head model size**
  selector in OPTIONS. Trade speed for accuracy to suit your hardware.
- **Head model is now the primary whole-head detector.** It replaces the old
  person→head-region estimate as the main source of head boxes.
- **Weapon/gear false positives removed.** The person→head geometry pass — which
  could land on forward-held gear like a weapon illuminator — is now disabled
  whenever a head model is loaded.
- **Scale-mismatch false positives fixed.** The head model now runs at a fixed
  inference size that matches its training size.
- **Visible first-launch progress.** The splash now shows a moving bar and an
  elapsed clock during the one-time torch download.
- **Full head-model training pipeline** added for fine-tuning on your own footage.

---

## New features

### Size-selectable head models
- The app bundles `head_n.pt` (nano, fastest), `head_s.pt` (small, balanced —
  default), and `head_m.pt` (medium, most accurate, slowest).
- A **Head model size** dropdown in OPTIONS picks which one runs when *Detect
  whole head* is on. The choice is saved across sessions.
- Note: **medium** runs a heavier network on every frame and is slow on CPU
  machines — pick nano or small if processing speed matters.

### First-launch progress indicator
- During the one-time torch download, the splash shows a moving busy bar with an
  elapsed clock (e.g. `Downloading GPU libraries (one-time, ~2.5GB)  (3:42)`).
- On a machine with an Nvidia GPU the app downloads the CUDA build of torch
  (~2.5GB), which legitimately takes several minutes on first run. The clock
  makes it clear the app is working, not frozen.

### Head-model training pipeline
New and updated helper scripts for fine-tuning your own head detector:
- `sample_frames.py` — now accepts a **folder** of clips (with `-r` for
  subfolders), not just individual filenames.
- `check_split.py` — verifies the train/val split is clip-level with no frame
  leakage (the #1 cause of good metrics but poor real-world results).
- `fix_split.py` — repairs a leaked split by moving whole clips (images +
  labels) between train and val.
- `diagnose_heads.py` — runs a model over a clip/image at a chosen inference
  size, to catch train/inference size mismatches.
- `train_head.py` — two-phase training (hyperparameter search, then a long final
  run) for a single model.
- `train_all.py` — trains `head_n.pt`, `head_s.pt`, and `head_m.pt`
  back-to-back, unattended, and writes a comparison summary.
- `HEAD_MODEL_PIPELINE.pdf` / `.md` — the full end-to-end walkthrough, updated
  for all of the above.

---

## Changes to detection behavior

- **Head model primary, person→head fallback only.** When a fine-tuned head
  model is loaded, it is the primary head detector and the person→head geometry
  pass is **disabled**. The person→head pass now runs only when no head model is
  present (e.g. a partial install). This is controlled by the new
  `PERSON_HEAD_MODE` setting in `face_blur.py` (`"user_off"` default; also
  `"any_off"`, `"always"`, `"never"`).
- **The face model still runs with whole-head detection on.** Face and head
  passes are unioned; the face pass is a cheap frontal-face backstop. With a
  good head model, leaving the face model on **nano** is the recommended default.
- **Matched inference size.** The head model runs at `HEAD_INFER_IMGSZ`
  (default 960), which must equal the size the model was trained at. Mismatched
  sizes were the cause of objects (e.g. weapon lights) being misread as heads.
- **Debug overlay colors:** red = head model, cyan = face model, yellow =
  person→head fallback (only shown when no head model is loaded).

---

## Bug fixes

- **Weapon-illuminator / gear false positives.** Caused by the person→head
  geometry estimate landing on forward-held equipment; fixed by demoting that
  pass when a head model is loaded.
- **Scale-dependent false positives.** Caused by running the head model at a
  different size than it was trained at; fixed by matching `HEAD_INFER_IMGSZ` to
  the training size.
- **First-launch "looks frozen."** The splash now animates with an elapsed clock
  during the torch download (builds on the v1.2.1 thread-safety fix).

---

## Build & distribution

- The installer (`installer.iss`) and both build scripts (`build.bat`,
  `build_installer.bat`) now bundle all three head models (`head_n/s/m.pt`) next
  to the exe. The installer build fails early with a clear message if any are
  missing.
- Installer version is now **1.3**.

---

## Upgrade notes

- **Your saved settings are preserved.** This release does not reset settings;
  the new *Head model size* option simply defaults to **small**. (Internally,
  `SETTINGS_VERSION` is intentionally unchanged so your tuned confidence,
  padding, etc. carry over.)
- **First launch after install downloads torch again** (the install ships
  without torch). On a GPU machine this is the ~2.5GB CUDA build — give it a few
  minutes; the new elapsed clock shows it progressing.
- **If you retrain head models at a different size than 960**, set
  `HEAD_INFER_IMGSZ` in `face_blur.py` to match, or scale-mismatch false
  positives will return. Larger sizes also make the head pass slower, especially
  on CPU.

---

*FACEBLUR v1.3 — made by werehappy*
