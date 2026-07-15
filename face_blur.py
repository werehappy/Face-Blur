"""
face_blur.py — Face Detection & Censoring GUI Application
Uses YOLOv11-face + OpenCV + Tkinter
"""

import os
import sys
import json
import time
import threading
import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox

# Version
VERSION = "1.4"

# Settings persistence
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".faceblur_settings.json")

SETTINGS_VERSION = "1.2.1"

def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        # Reset settings if version changed to avoid stale values
        if data.get("settings_version") != SETTINGS_VERSION:
            return {}
        return data
    except Exception:
        return {}

def save_settings(data):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# cv2 and numpy imported lazily after splash screen
cv2 = None
np  = None

def _lazy_imports():
    """Import slow modules in background after splash is shown."""
    global cv2, np
    import cv2 as _cv2
    import numpy as _np
    cv2 = _cv2
    np  = _np


# ══════════════════════════════════════════════════════════
#  GPU DETECTION & TORCH INSTALLER
# ══════════════════════════════════════════════════════════

def detect_gpu():
    """
    Returns dict with keys:
      has_nvidia  - bool, nvidia-smi found
      cuda_ver    - str like "12.1" or None
      torch_cuda  - bool, current torch has CUDA
      gpu_name    - str or None
    """
    import subprocess, shutil
    result = {"has_nvidia": False, "cuda_ver": None,
              "torch_cuda": False, "gpu_name": None}
    try:
        import torch
        result["torch_cuda"] = torch.cuda.is_available()
        if result["torch_cuda"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Check nvidia-smi even if torch says no CUDA
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output([smi], text=True, timeout=5)
            result["has_nvidia"] = True
            import re
            m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
            if m:
                result["cuda_ver"] = m.group(1)
        except Exception:
            pass
    return result

def get_python_executable():
    """
    Get the real Python executable path.
    When running as a PyInstaller bundle, sys.executable is the .exe itself.
    We need the actual python.exe to run pip.
    """
    import sys
    # If frozen (PyInstaller exe), find python.exe next to the exe
    # or in a standard relative path
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        venv = _get_venv_path()
        # Prefer the bundled embeddable python inside faceblur_env.
        # This is the python the installer ships; pip-installing with it
        # places torch into faceblur_env\Lib\site-packages, which
        # _add_venv_to_path() then injects into sys.path.
        candidates = [
            os.path.join(venv, "python.exe"),
            os.path.join(venv, "Scripts", "python.exe"),
            os.path.join(exe_dir, "python.exe"),
            os.path.join(exe_dir, "_internal", "python.exe"),
        ]
        # Also check CONDA_PREFIX if set (useful when running from source)
        conda = os.environ.get("CONDA_PREFIX", "")
        if conda:
            candidates.insert(0, os.path.join(conda, "python.exe"))
        for c in candidates:
            if os.path.exists(c):
                return c
        # Last resort: find python.exe on PATH
        import shutil
        found = shutil.which("python")
        if found:
            return found
        return None
    else:
        return sys.executable

def _pick_torch_index(cuda_ver, has_nvidia):
    """
    Choose the right torch wheel index for the user's hardware.
    Returns (index_url, human_label).
    Key fix: an Nvidia GPU whose CUDA version we could not parse should still
    get a CUDA build (assume a modern driver) instead of silently falling back
    to CPU, which is what left users on 'GPU available - not enabled'.
    """
    ver = float(cuda_ver) if cuda_ver else 0
    if ver >= 12.1:
        return "https://download.pytorch.org/whl/cu121", "CUDA 12.1+"
    if ver >= 11.8:
        return "https://download.pytorch.org/whl/cu118", "CUDA 11.8"
    if has_nvidia and ver == 0:
        # GPU present but CUDA version unknown -> assume a modern driver.
        return "https://download.pytorch.org/whl/cu121", "CUDA (version unknown, assuming 12.x)"
    if has_nvidia:
        # GPU present but driver/CUDA genuinely too old for current torch.
        return "https://download.pytorch.org/whl/cpu", "CPU (CUDA too old)"
    return "https://download.pytorch.org/whl/cpu", "CPU"

def install_torch_for_cuda(cuda_ver, log_fn=None, has_nvidia=False):
    """Install correct torch build using the real Python executable."""
    import subprocess, sys

    index, label = _pick_torch_index(cuda_ver, has_nvidia)

    if log_fn:
        log_fn("Installing torch for {}...\n".format(label), "accent")

    python_exe = get_python_executable()
    if python_exe is None:
        if log_fn:
            log_fn("[ERROR] Could not find Python executable to run pip.\n", "error")
        return False

    if log_fn:
        log_fn("  Using Python: {}\n".format(python_exe), "dim")

    # Pin numpy to the exact version baked into the exe. torch otherwise pulls
    # whatever numpy it likes (often a newer major version), and having two
    # binary-incompatible numpys reachable crashes the app natively at import
    # (the console just vanishes). Matching versions removes that conflict.
    np_pin = None
    try:
        import numpy as _np_v
        np_pin = _np_v.__version__
    except Exception:
        pass

    cmd = [python_exe, "-m", "pip", "install",
           "torch", "torchvision"]
    if np_pin:
        cmd.append("numpy=={}".format(np_pin))
    cmd += ["--index-url", index,
            # IMPORTANT: do NOT add --extra-index-url pypi here. With both
            # indexes, pip pulls torch from PyPI (the Windows wheel there is
            # CPU-only) instead of the +cuXXX build from the PyTorch index.
            # The PyTorch index is self-contained for torch and its deps.
            # force a clean swap (e.g. CPU build -> CUDA build).
            "--force-reinstall", "--no-cache-dir"]

    # CREATE_NO_WINDOW prevents a new console window on Windows
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **kwargs
        )
        for line in proc.stdout:
            if log_fn and line.strip():
                log_fn("  " + line.strip() + "\n", "dim")
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        if log_fn:
            log_fn("[ERROR] {}\n".format(e), "error")
        return False


# ══════════════════════════════════════════════════════════
#  YOLO
# ══════════════════════════════════════════════════════════

# Auto suffix based on censor mode
MODE_SUFFIX = {
    "Blur":      "_blurred",
    "Pixelate":  "_pixelated",
    "Black Bar": "_blackbar",
}

YOLO_MODELS = {
    "nano  (fastest)":   ("yolov11n-face.pt", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"),
    "medium (accurate)": ("yolov11m-face.pt", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11m-face.pt"),
    "large (best)":      ("yolov11l-face.pt", "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11l-face.pt"),
}
_detector_cache = {}

# ── Whole-head detection ──
# Face models only fire on the facial region, so they miss backs/sides of heads.
# To also cover the whole head we run a second detector and merge the results.
#
# DEFAULT METHOD = PERSON DETECTION -> HEAD REGION.
# On hard footage (motion blur, helmets, partial/cut-off heads, e.g. CQB headcam)
# dedicated head models fail, but the COCO "person" detector is very robust to
# blur/occlusion/partial bodies. So by default we detect the person and censor
# the head REGION (top-of-body, sized by shoulder width). This needs some torso
# in frame, which is usually the case.
#
# The two methods are COMBINED (union), not either/or, because they cover
# complementary failures:
#   - person -> head region: robust to blur/helmets/side views/cut-off faces,
#     but needs some torso in frame.
#   - a real head model (head.pt, e.g. CrowdHuman-trained or fine-tuned on your
#     footage): fires on the head itself, so it can catch a head with NO body
#     visible -- the one case the person method cannot reach.
# Whatever loads is used; if nothing loads, whole-head mode quietly behaves
# like face-only.
HEAD_MODEL_FILE = "head.pt"        # user-provided / fine-tuned head model (takes priority)

# Selectable fine-tuned head models by size (produced by train_all.py). The
# OPTIONS selector picks one; all are treated as user models, so the person->head
# geometry pass is disabled when any of them is loaded (see PERSON_HEAD_MODE).
# A legacy single head.pt is still honored as a fallback. IMPORTANT: all three
# must be trained at HEAD_INFER_IMGSZ (960) so inference matches training.
HEAD_MODELS = {
    "nano (fastest)":   "head_n.pt",
    "small (balanced)": "head_s.pt",
    "medium (best)":    "head_m.pt",
}
HEAD_MODEL_DEFAULT_KEY = "small (balanced)"

# COCO person detector -> head region. Loaded via an EXPLICIT URL (the same robust
# urllib path the face models use) instead of relying on ultralytics' internal
# auto-downloader. The internal downloader frequently fails inside the frozen
# PyInstaller exe (settings dir / SSL / offline first-run), which silently left
# whole-head mode with no person model -> output was face-only. An explicit URL
# matches how the face models already download successfully.
HEAD_PERSON_MODEL = "yolo11n.pt"
HEAD_PERSON_URL   = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
HEAD_PERSON_CLASS = 0              # COCO "person"

# Whole-head COVERAGE geometry. These are the values validated on real footage
# in the session-2 testing (test_detect.py uses the same 0.45/0.55). The padding
# slider provides the extra margin on top, so keep these modest -- padding does
# the work of covering ears/crown, and it is now applied correctly.
HEAD_REGION_W_FRAC = 0.45   # person->head width  as a fraction of shoulder width
HEAD_REGION_H_FRAC = 0.55   # person->head height as a fraction of shoulder width
HEAD_REGION_TOP    = 0.0    # extend the region ABOVE the person-box top (frac of head height)
# Face boxes are grown only SLIGHTLY toward head size in whole-head mode (faces
# cover the lower-front of the head). Padding handles the rest; keep this mild so
# coverage stays reasonable rather than ballooning when padding is also high.
FACE_TO_HEAD_W     = 1.15   # grow a face box's WIDTH  by this (a little, for ears)
FACE_TO_HEAD_H     = 1.25   # grow a face box's HEIGHT by this (a little, for crown/jaw)
FACE_TO_HEAD_UP    = 0.10   # shift the grown box UP by this frac of its new height

# Default head model, auto-downloaded when the user has NOT supplied their own
# head.pt. A real head detector fires on the head itself, so it adds the one case
# the person->region method cannot reach: a head with NO body in frame (peeking
# side/back of head). Saved to a SEPARATE file so a user-supplied head.pt always
# wins and is never overwritten.
#   NOTE: public head models (SCUT-HEAD here) are trained on civilian, frontal,
#   unoccluded heads. They help general footage but are weak on helmeted/blurred/
#   tactical (CQB) footage -- for that, drop in a fine-tuned head.pt.
#   Set HEAD_DEFAULT_URL = None to disable this auto-download entirely.
HEAD_DEFAULT_FILE = "head_default.pt"
HEAD_DEFAULT_URL  = "https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/nano.pt"

# Heads (esp. backs/partials) score LOWER than faces, so run the head/person pass
# at a lower confidence floor than the face pass (face conf minus this, min 0.15).
HEAD_CONF_DROP = 0.15

# Inference size for the dedicated head model (head.pt / head_default.pt).
# This MUST match the imgsz the head model was TRAINED at. ultralytics defaults
# to 640 when no imgsz is given; running a 960-trained head model at any other
# size shifts how small bright objects (e.g. weapon illuminators) are scaled and
# can make them get misread as heads at high confidence. Keep this equal to your
# head.pt training imgsz. Face/person passes are left at the default (640) since
# those models were trained at 640.
HEAD_INFER_IMGSZ = 960

# --- Person->head geometry pass: demotion policy -----------------------------
# The person->head pass ESTIMATES a head box from a COCO person box (sized by
# shoulder width). It is robust to blur/helmets/cut-off bodies, but it is pure
# GEOMETRY, not detection: on gear-forward poses the person box widens to include
# an extended weapon, and the estimated head region can land on the muzzle (e.g.
# a weapon illuminator), painting a censor where there is no head. With a
# fine-tuned head.pt now doing primary detection, this pass is demoted.
#   "user_off" : person->head runs UNLESS your own head.pt is loaded (default).
#                Users with only the auto-downloaded default keep it as a fallback.
#   "any_off"  : person->head runs only when NO head model is loaded at all.
#   "always"   : legacy behavior -- person->head always unions in.
#   "never"    : person->head fully disabled.
PERSON_HEAD_MODE = "user_off"

# Whole-head mode: a face box is treated as ALREADY covered (and dropped, to
# avoid double-masking the same head) when at least this fraction of the RAW
# face box falls inside some head/person box. Lower = more aggressive
# suppression of redundant face censors; raise it toward 0.6 if you start
# seeing real faces left uncovered next to a head box.
FACE_COVERED_FRAC = 0.5

def get_detector_by_file(filename, url=None, log_fn=None):
    """Load (and cache) a YOLO model by file.
    url=None -> let ultralytics auto-fetch known weights (e.g. yolo11n.pt).
    url set   -> download to `filename` if missing."""
    if filename not in _detector_cache:
        from ultralytics import YOLO
        downloaded_now = False
        if url and not os.path.exists(filename):
            if log_fn:
                log_fn("  Downloading {} (first time only)...\n".format(filename), "warning")
            import urllib.request
            urllib.request.urlretrieve(url, filename)
            downloaded_now = True
            if log_fn:
                log_fn("  Download complete.\n", "success")
        try:
            model = YOLO(filename)
        except Exception:
            # A corrupt/partial download must not poison future runs.
            if downloaded_now:
                try: os.remove(filename)
                except Exception: pass
            raise
        # Enable FP16 (half precision) on GPU for ~2x faster inference
        try:
            import torch
            if torch.cuda.is_available():
                model.model.half()
        except Exception:
            pass
        _detector_cache[filename] = model
    return _detector_cache[filename]

def get_detector(model_key, log_fn=None):
    filename, url = YOLO_MODELS[model_key]
    return get_detector_by_file(filename, url, log_fn)

def get_head_detector(head_size=None, log_fn=None):
    """Load the head detector and the auxiliary person detector.
      head_model   -- the fine-tuned head model (head_n/s/m.pt) or a legacy
                      head.pt; auto-downloaded default if none present. This is
                      the primary detector.
      person_model -- COCO person detector, used only when the person->head aid
                      is enabled (as a rescue prior, not a box source).
    Returns (head_model, person_model, head_is_user).
    """
    head_model = None
    head_is_user = False
    # 1) The selected fine-tuned size model (head_n/s/m.pt) first, then a legacy
    #    single head.pt as a fallback/manual override. Both count as "user"
    #    models, which demotes the person->head pass (see PERSON_HEAD_MODE).
    candidates = []
    if head_size and head_size in HEAD_MODELS:
        candidates.append(HEAD_MODELS[head_size])
    if HEAD_MODEL_FILE not in candidates:
        candidates.append(HEAD_MODEL_FILE)
    for cand in candidates:
        if os.path.exists(cand):
            try:
                head_model = get_detector_by_file(cand, None, log_fn)
                head_is_user = True
                if log_fn:
                    log_fn("  Head model: using {}.\n".format(cand), "success")
                break
            except Exception as e:
                if log_fn:
                    log_fn("  [WARN] {} failed to load ({}).\n".format(cand, e), "warning")
    # 2) Otherwise auto-download a default head model (to a SEPARATE file, so a
    #    user head.pt added later still wins and this is never overwritten).
    if head_model is None and HEAD_DEFAULT_URL:
        try:
            head_model = get_detector_by_file(HEAD_DEFAULT_FILE, HEAD_DEFAULT_URL, log_fn)
            if log_fn:
                log_fn("  Head model: using default {} (fine-tune a head.pt for "
                       "tactical/CQB footage).\n".format(HEAD_DEFAULT_FILE), "dim")
        except Exception as e:
            if log_fn:
                log_fn("  [WARN] default head model unavailable ({}).\n".format(e), "warning")
    # 3) Person detection -> head region (always attempted; complements the head
    #    model and is robust to blur/helmets/partial bodies).
    person_model = None
    try:
        person_model = get_detector_by_file(HEAD_PERSON_MODEL, HEAD_PERSON_URL, log_fn)
    except Exception as e:
        if log_fn:
            log_fn("  [WARN] person model failed to load ({}).\n".format(e), "warning")
    return head_model, person_model, head_is_user

def _head_class_ids(detector):
    """Figure out which class id(s) a head model uses for 'head'.
    - If a class is literally named 'head', filter to it.
    - If it's a single-class model, keep all (None).
    - Otherwise keep all as a safe default.
    Result is cached on the detector object itself."""
    cached = getattr(detector, "_faceblur_head_classes", "unset")
    if cached != "unset":
        return cached
    ids = None
    try:
        names = detector.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        head_ids = [int(i) for i, n in items if "head" in str(n).lower()]
        if head_ids:
            ids = head_ids
    except Exception:
        ids = None
    try:
        detector._faceblur_head_classes = ids
    except Exception:
        pass
    return ids

def clear_detector_cache():
    """Release YOLO models from memory and free CUDA cache."""
    global _detector_cache
    _detector_cache = {}
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
#  CENSOR HELPERS
# ══════════════════════════════════════════════════════════

def blur_region(frame, x1, y1, x2, y2, intensity=51):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    k = intensity if intensity % 2 == 1 else intensity + 1
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame

def pixelate_region(frame, x1, y1, x2, y2, block_size=15):
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return frame
    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame

def black_bar_region(frame, x1, y1, x2, y2):
    frame[y1:y2, x1:x2] = 0
    return frame

def expand_bbox(x1, y1, x2, y2, padding, fw, fh):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw = (x2 - x1) / 2 * (1 + padding)
    hh = (y2 - y1) / 2 * (1 + padding)
    return max(0, int(cx-hw)), max(0, int(cy-hh)), min(fw, int(cx+hw)), min(fh, int(cy+hh))

class FaceTracker:
    """
    Wraps OpenCV trackers to smooth face boxes between YOLO detections.
    On detection frames: updates trackers with new YOLO boxes.
    On skip frames: advances trackers to get interpolated positions.
    """
    def __init__(self):
        self._trackers = []   # list of (tracker, box)
        self._last_boxes = []

    def update_from_detection(self, frame, boxes):
        """Reset trackers from fresh YOLO detections."""
        self._trackers = []
        self._last_boxes = list(boxes)
        for box in boxes:
            x1, y1, x2, y2 = box[:4]   # boxes may carry a 5th score element
            try:
                tr = cv2.legacy.TrackerCSRT_create()
                tr.init(frame, (x1, y1, x2 - x1, y2 - y1))
                self._trackers.append(tr)
            except Exception:
                pass   # fall back to static boxes if CSRT unavailable

    def get_tracked_boxes(self, frame):
        """Advance trackers and return current box positions."""
        if not self._trackers:
            return self._last_boxes
        result = []
        for tr in self._trackers:
            try:
                ok, (tx, ty, tw, th) = tr.update(frame)
                if ok and tw > 0 and th > 0:
                    result.append((int(tx), int(ty), int(tx + tw), int(ty + th)))
            except Exception:
                pass
        return result if result else self._last_boxes

# ── Temporal box smoothing (motion-aware anti-flicker) ──
# Raw per-frame detections flicker: a blurred head is found on frame n, missed
# on n+1, found on n+2. A plain EMA + fixed hold smooths static footage but
# fails on DYNAMIC footage (headcam): held boxes stay at old SCREEN positions
# while the camera pans, and the EMA lags behind fast motion. The smoother here:
#   * CAMERA-MOTION COMPENSATION: global frame-to-frame shift is estimated by
#     phase correlation on a downscaled grayscale; every track moves WITH the
#     scene, so a held box stays glued to the spot in the world, not the screen.
#   * VELOCITY: each track keeps its own (camera-compensated) velocity and
#     coasts along it through missed detections.
#   * SNAP, DON'T LAG: if a matched detection moved more than half a box-size,
#     the track jumps straight to it; smoothing only damps sub-box jitter.
#   * TENTATIVE vs CONFIRMED: a brand-new track is still censored IMMEDIATELY
#     (privacy first) but dies after SMOOTH_HOLD_TENTATIVE missed frames unless
#     it racks up SMOOTH_CONFIRM_HITS detections -- so a 1-frame false positive
#     blinks for ~2 frames instead of being amplified into a 13-frame blob.
SMOOTH_ALPHA          = 0.6   # EMA weight of the new detection (jitter damping)
SMOOTH_SNAP_FRAC      = 0.5   # displacement > this fraction of box size -> snap
SMOOTH_HOLD           = 8     # missed frames a CONFIRMED track survives
SMOOTH_HOLD_TENTATIVE = 2     # missed frames a brand-new (1-hit) track survives
SMOOTH_CONFIRM_HITS   = 3     # kept for compatibility; hold now ramps with hits
SMOOTH_IOU            = 0.20  # IoU match gate...
SMOOTH_DIST_FRAC      = 0.80  # ...or center distance under this x box size
SMOOTH_MOTION_W       = 256   # downscale width for camera-motion estimation

class BoxSmoother:
    """Turns raw per-frame detections into stable, motion-following tracks.
    Call update(boxes, frame) every frame; pass the (unblurred) frame so the
    smoother can estimate camera motion. Returns (x1, y1, x2, y2, score)."""

    def __init__(self, alpha=SMOOTH_ALPHA, hold=SMOOTH_HOLD,
                 hold_tentative=SMOOTH_HOLD_TENTATIVE,
                 confirm_hits=SMOOTH_CONFIRM_HITS, iou=SMOOTH_IOU):
        self.alpha = float(alpha)
        self.hold = int(hold)
        self.hold_tentative = int(hold_tentative)
        self.confirm_hits = int(confirm_hits)
        self.iou_t = float(iou)
        self.tracks = []        # dicts: x1 y1 x2 y2 vx vy score hits miss
        self._prev_small = None
        self._prev_scale = 1.0

    # -- camera motion ----------------------------------------------------
    def _camera_shift(self, frame):
        """Global content shift (dx, dy) between the previous frame and this
        one, in full-frame pixels. (0, 0) if it cannot be estimated."""
        try:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = SMOOTH_MOTION_W / float(g.shape[1])
            small = cv2.resize(g, (SMOOTH_MOTION_W, max(2, int(g.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA).astype("float32")
            prev = self._prev_small
            self._prev_small = small
            self._prev_scale = scale
            if prev is None or prev.shape != small.shape:
                return 0.0, 0.0
            (dx, dy), resp = cv2.phaseCorrelate(prev, small)
            if resp < 0.05:   # heavy blur / scene cut: shift unreliable
                return 0.0, 0.0
            return dx / scale, dy / scale
        except Exception:
            return 0.0, 0.0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _iou(t, b):
        ix1, iy1 = max(t["x1"], b[0]), max(t["y1"], b[1])
        ix2, iy2 = min(t["x2"], b[2]), min(t["y2"], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        a_t = (t["x2"] - t["x1"]) * (t["y2"] - t["y1"])
        a_b = (b[2] - b[0]) * (b[3] - b[1])
        union = a_t + a_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(o):
        if isinstance(o, dict):
            return (o["x1"] + o["x2"]) / 2.0, (o["y1"] + o["y2"]) / 2.0
        return (o[0] + o[2]) / 2.0, (o[1] + o[3]) / 2.0

    def _shift_track(self, tr, dx, dy):
        tr["x1"] += dx; tr["x2"] += dx
        tr["y1"] += dy; tr["y2"] += dy

    # -- main ---------------------------------------------------------------
    def update(self, boxes, frame=None):
        # 1) Move every track with the camera, then along its own velocity.
        cam_dx = cam_dy = 0.0
        if frame is not None:
            cam_dx, cam_dy = self._camera_shift(frame)
        for tr in self.tracks:
            self._shift_track(tr, cam_dx + tr["vx"], cam_dy + tr["vy"])

        dets = [tuple(map(float, b[:4])) + (float(b[4]) if len(b) > 4 else 1.0,)
                for b in boxes]

        # 2) Greedy matching: confirmed tracks pick first. A detection matches
        #    a track if boxes overlap OR centers are within a box-size.
        unmatched = list(dets)
        for tr in sorted(self.tracks, key=lambda t: -t["hits"]):
            size = max(tr["x2"] - tr["x1"], tr["y2"] - tr["y1"], 1.0)
            tcx, tcy = self._center(tr)
            best, best_cost = None, None
            for b in unmatched:
                bcx, bcy = self._center(b)
                dist = ((bcx - tcx) ** 2 + (bcy - tcy) ** 2) ** 0.5
                if self._iou(tr, b) >= self.iou_t or dist <= SMOOTH_DIST_FRAC * size:
                    if best_cost is None or dist < best_cost:
                        best, best_cost = b, dist
            if best is None:
                tr["miss"] += 1
                tr["vx"] *= 0.9    # decay velocity while coasting blind
                tr["vy"] *= 0.9
                continue
            unmatched.remove(best)
            bcx, bcy = self._center(best)
            # Velocity correction: (bcx - tcx) is the residual AFTER the
            # prediction already moved the track by vx, so the correct update
            # is v += k*residual (v = 0.5v + 0.5r would converge to half the
            # true velocity). Clamped to one box-size per frame.
            tr["vx"] = max(-size, min(size, tr["vx"] + 0.6 * (bcx - tcx)))
            tr["vy"] = max(-size, min(size, tr["vy"] + 0.6 * (bcy - tcy)))
            disp = best_cost
            a = 1.0 if disp > SMOOTH_SNAP_FRAC * size else self.alpha
            tr["x1"] = a * best[0] + (1 - a) * tr["x1"]
            tr["y1"] = a * best[1] + (1 - a) * tr["y1"]
            tr["x2"] = a * best[2] + (1 - a) * tr["x2"]
            tr["y2"] = a * best[3] + (1 - a) * tr["y2"]
            tr["score"] = max(tr["score"] * 0.9, best[4])
            tr["hits"] += 1
            tr["miss"] = 0

        # 3) Cull with a GRADUATED hold: the more detections a track has
        #    accumulated, the longer it may coast blind. A 1-frame false
        #    positive lives only 1 + SMOOTH_HOLD_TENTATIVE frames, while a
        #    head detected even every 3rd frame keeps ratcheting up evidence
        #    (hits=1 -> hold 2, 2 -> 4, 3 -> 6, 4+ -> SMOOTH_HOLD) and is
        #    never dropped between detections.
        kept = []
        for tr in self.tracks:
            limit = min(self.hold, self.hold_tentative + 2 * (tr["hits"] - 1))
            if tr["miss"] <= limit:
                kept.append(tr)
        self.tracks = kept

        # 4) Spawn new tracks for unmatched detections (cover immediately).
        for b in unmatched:
            self.tracks.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                                "vx": 0.0, "vy": 0.0,
                                "score": b[4], "hits": 1, "miss": 0})

        return [(int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"]),
                 t["score"]) for t in self.tracks]

def _run_yolo(detector, frame, confidence, classes=None, imgsz=None):
    """Run YOLO on a single frame and return list of (x1,y1,x2,y2,score).
    classes: optional list of class ids to keep (e.g. [0]=person for COCO).
    imgsz: inference size. When None, ultralytics uses its default (640). Set it
    to the model's training size to avoid train/inference scale mismatch."""
    kwargs = dict(conf=confidence, verbose=False, workers=0, classes=classes)
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    results = detector(frame, **kwargs)
    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), score))
    return boxes

def _dedup_boxes(boxes, iou_thresh=0.4):
    """Remove duplicate boxes using IoU threshold."""
    if not boxes:
        return boxes
    kept = []
    for b in sorted(boxes, key=lambda x: -x[4]):  # sort by confidence desc
        x1, y1, x2, y2, s = b
        duplicate = False
        for kb in kept:
            kx1, ky1, kx2, ky2, _ = kb
            # compute IoU
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter/union > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(b)
    return kept

def detect_faces(frame, detector, confidence, detect_scale=0.5, edge_strip=True, classes=None,
                 edge_conf_drop=0.1, imgsz=None):
    """
    Run YOLO on downscaled frame + edge strips for partial/out-of-frame faces.
    Returns list of (x1, y1, x2, y2, score).
    classes: optional list of class ids to keep (e.g. [0]=person).
    imgsz: inference size passed to YOLO (None = ultralytics default 640). Set to
    the detector's training size to keep inference scale-matched to training.
    """
    h, w = frame.shape[:2]

    # Main detection pass (downscaled)
    if detect_scale < 1.0:
        sw = max(32, int(w * detect_scale))
        sh = max(32, int(h * detect_scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        raw = _run_yolo(detector, small, confidence, classes, imgsz=imgsz)
        # Scale coords back up
        boxes = [(int(x1/detect_scale), int(y1/detect_scale),
                  int(x2/detect_scale), int(y2/detect_scale), s)
                 for (x1, y1, x2, y2, s) in raw]
    else:
        boxes = _run_yolo(detector, frame, confidence, classes, imgsz=imgsz)

    # Edge strip passes (20% of each side, full resolution)
    if edge_strip:
        edge = 0.20
        strips = [
            (0,              0,              int(w*edge),    h),   # left
            (int(w*(1-edge)), 0,             w,              h),   # right
            (0,              0,              w,              int(h*edge)),  # top
            (0,              int(h*(1-edge)), w,             h),   # bottom
        ]
        for (sx1, sy1, sx2, sy2) in strips:
            strip = frame[sy1:sy2, sx1:sx2]
            if strip.size == 0:
                continue
            raw = _run_yolo(detector, strip, max(0.1, confidence - edge_conf_drop), classes, imgsz=imgsz)
            for (x1, y1, x2, y2, s) in raw:
                # Translate back to full frame coords
                boxes.append((x1+sx1, y1+sy1, x2+sx1, y2+sy1, s))

    return _dedup_boxes(boxes)

def _person_to_head(boxes):
    """Head region from a person box, sized by SHOULDER WIDTH (not body height,
    which breaks when the body is cut off). Generous on purpose so it covers the
    EARS and the back/crown of the head, not just the face. Top-centered on the
    person box and extended slightly above its top for the crown/hair. This is
    the YELLOW box in test_detect.py."""
    out = []
    for (x1, y1, x2, y2, s) in boxes:
        bw = x2 - x1
        if bw <= 0:
            continue
        cx = (x1 + x2) // 2
        hw = max(1, int(bw * HEAD_REGION_W_FRAC) // 2)   # half-width of head
        hh = max(1, int(bw * HEAD_REGION_H_FRAC))        # head height
        top = int(y1 - HEAD_REGION_TOP * hh)             # reach above for the crown
        out.append((cx - hw, top, cx + hw, top + hh, s))
    return out

# Prior-guided re-scoring ("rescue") parameters. The person->head pathway is
# used here NOT as a box source but as a soft spatial prior over the head
# model's own detections: a weak head box that overlaps a person-derived head
# region is boosted past the operating point, while an isolated weak box is
# not. This is the configuration the paper's ablation recommends (rescue only,
# no raw geometry-box injection), which recovered ~95% of the recall gain of
# the raw union while giving back most of its precision cost.
PRIOR_RESCUE_FLOOR = 0.10   # candidate floor for the head model when rescuing
PRIOR_RESCUE_ALPHA = 1.0    # boost strength: conf *= (1 + ALPHA * overlap)
PRIOR_RESCUE_OVERLAP = 0.30 # IoU (or center-inside) needed to count as "inside"

def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def _prior_overlap(box, priors):
    """Max IoU of `box` against the person-derived head regions, with a floor
    of 0.5 when the box CENTER falls inside a prior (the prior is a coarse
    region, so IoU alone under-credits a correct-but-small head)."""
    best = 0.0
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    for pb in priors:
        v = _box_iou(box, pb)
        if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
            v = max(v, 0.5)
        if v > best:
            best = v
    return best

def _rescore_with_prior(head_boxes, prior_regions, op_conf):
    """Prior-guided rescue. head_boxes: (x1,y1,x2,y2,score) collected at the low
    PRIOR_RESCUE_FLOOR. prior_regions: (x1,y1,x2,y2[,score]) head regions from
    _person_to_head. Returns the boxes whose BOOSTED score clears op_conf, with
    the boosted score attached. No raw prior boxes are ever added (rescue only),
    so gear-forward misplaced regions cannot paint a censor on their own."""
    priors = [tuple(p[:4]) for p in prior_regions]
    out = []
    for b in head_boxes:
        x1, y1, x2, y2 = b[:4]
        s = b[4] if len(b) > 4 else 1.0
        ov = _prior_overlap((x1, y1, x2, y2), priors) if priors else 0.0
        s2 = min(1.0, s * (1.0 + PRIOR_RESCUE_ALPHA * ov))
        if s2 >= op_conf:
            out.append((x1, y1, x2, y2, s2))
    return out

def _face_to_head(boxes):
    """Grow a FACE box outward into a whole-HEAD box: faces sit in the lower-
    centre of the head, so the head extends sideways (ears) and upward (crown),
    plus some margin behind. Used only when whole-head mode is on, so a head the
    head/person passes missed but the FACE model caught still gets fully covered
    instead of leaving ears/back showing."""
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        s = b[4] if len(b) > 4 else 1.0
        w = x2 - x1; h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        nw = w * FACE_TO_HEAD_W; nh = h * FACE_TO_HEAD_H
        cy -= FACE_TO_HEAD_UP * nh                       # head centre is above the face centre
        out.append((int(cx - nw / 2), int(cy - nh / 2),
                    int(cx + nw / 2), int(cy + nh / 2), s))
    return out

def _contained(inner, outer, thresh=0.6):
    """True if >= thresh of `inner`'s area falls inside `outer`."""
    ix1, iy1, ix2, iy2 = inner[:4]
    ox1, oy1, ox2, oy2 = outer[:4]
    ax1, ay1 = max(ix1, ox1), max(iy1, oy1)
    ax2, ay2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_inner = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / area_inner >= thresh

def _merge_face_head(face_boxes, head_boxes):
    """DEPRECATED / UNUSED. Kept for reference only.
    detect_objects now suppresses covered RAW face boxes and grows only the
    survivors inline, instead of growing every face first and merging here
    (which let a grown face survive containment and produce a second censor on
    a head already covered by a head/person box).

    Union of both detectors.
    Head boxes are bigger and cover the whole head, so they are kept as-is.
    Face boxes are kept only where no head box already covers them, so the
    face model acts as a safety net for heads the head model missed.
    (A face box sits *inside* its head box, so IoU dedup alone won't merge
    them -- we use containment instead.)"""
    merged = list(head_boxes)
    for fb in face_boxes:
        if not any(_contained(fb, hb) for hb in head_boxes):
            merged.append(fb)
    return _dedup_boxes(merged)

def _person_head_active(head_model, head_is_user):
    """DEPRECATED / UNUSED. The person->head pathway is now controlled by the
    explicit "Person->head aid" GUI toggle (use_person_aid), not by the
    automatic PERSON_HEAD_MODE policy. Kept for backward reference only.

    Whether the person->head GEOMETRY pass should contribute boxes, per
    PERSON_HEAD_MODE. The pass estimates a head from a body box and can land on
    forward-held gear/weapons, so it is demoted once a real head model is doing
    the work. See PERSON_HEAD_MODE for the policy options."""
    mode = PERSON_HEAD_MODE
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "any_off":
        return head_model is None
    # default "user_off": disable when YOUR own head.pt is the active model
    return not head_is_user


def detect_objects(frame, face_detector, head_model, person_model, confidence,
                   detect_scale=0.5, edge_strip=False, detect_head=True,
                   head_is_user=False, use_face=False, use_person_aid=False):
    """Head-primary detection with independent, opt-in aids.

    Primary pass:
      head_model : direct head boxes (the fine-tuned head.pt) at full
                   resolution and HEAD_INFER_IMGSZ. This is the main and, by
                   default, only detector.

    Optional aids (each independently toggleable):
      use_person_aid : run the COCO person detector, convert each person box to
                       an estimated head region, and use those regions as a SOFT
                       PRIOR that rescues weak head detections (see
                       _rescore_with_prior). Not a box source -- an isolated
                       mislocalized region cannot create a censor on its own.
      edge_strip     : run the head model over the four frame-edge strips too,
                       recovering heads cut off at the boundary.
      use_face       : run the face model as a safety net; a face box is kept
                       (grown to head size) only where no head box already
                       covers it, so it fills genuine head-model misses.

    Returns list of (x1, y1, x2, y2, score).

    Side effect: fills LAST_DETECT_BREAKDOWN with per-source boxes for the debug
    overlay: {'face': [...], 'person': [...], 'head': [...]}.
    """
    global LAST_DETECT_BREAKDOWN
    LAST_DETECT_BREAKDOWN = {"face": [], "person": [], "head": []}

    # Backwards-compatible fallback: if head is not the active method and no head
    # model is available, degrade to the legacy face-only behavior so nothing
    # silently returns nothing.
    if not detect_head or head_model is None:
        if use_face and face_detector is not None:
            face_boxes = detect_faces(frame, face_detector, confidence,
                                      detect_scale, edge_strip)
            LAST_DETECT_BREAKDOWN["face"] = list(face_boxes)
            return _dedup_boxes(_face_to_head(face_boxes))
        # No head model and face off: nothing to do.
        return []

    # --- primary head pass -------------------------------------------------
    head_classes = _head_class_ids(head_model)
    # When the person aid is on we collect at a LOW floor so weak-but-real heads
    # survive to be rescued; otherwise we collect at the operating confidence.
    # Heads (backs/partials) score lower than faces, so the base floor already
    # sits below the face operating point.
    base_conf = max(0.15, confidence - HEAD_CONF_DROP)
    collect_conf = min(base_conf, PRIOR_RESCUE_FLOOR) if use_person_aid else base_conf
    # edge_conf_drop=0.0: head_conf is already lowered; a second drop on the
    # full-res strips was a false-positive factory (kept from the original).
    model_heads = detect_faces(frame, head_model, collect_conf, 1.0,
                               edge_strip, classes=head_classes,
                               edge_conf_drop=0.0, imgsz=HEAD_INFER_IMGSZ)

    # --- optional person->head prior (rescue) ------------------------------
    if use_person_aid and person_model is not None:
        # Full resolution regardless of the Detect-scale slider: small/partial/
        # side-on bodies vanish at 0.5x. Person boxes are large, so full-res is
        # cheap for the recall it buys.
        person_raw = detect_faces(frame, person_model, base_conf, 1.0,
                                  edge_strip=False, classes=[HEAD_PERSON_CLASS])
        prior_regions = _person_to_head(person_raw)
        LAST_DETECT_BREAKDOWN["person"] = list(prior_regions)
        model_heads = _rescore_with_prior(model_heads, prior_regions, base_conf)
    else:
        # No aid: enforce the normal operating floor on the collected heads.
        model_heads = [b for b in model_heads
                       if (b[4] if len(b) > 4 else 1.0) >= base_conf]

    LAST_DETECT_BREAKDOWN["head"] = list(model_heads)
    head_union = _dedup_boxes(model_heads)

    # --- optional face safety net ------------------------------------------
    if use_face and face_detector is not None:
        face_boxes = detect_faces(frame, face_detector, confidence,
                                  detect_scale, edge_strip)
        LAST_DETECT_BREAKDOWN["face"] = list(face_boxes)
        uncovered_faces = [fb for fb in face_boxes
                           if not any(_contained(fb, hb, FACE_COVERED_FRAC)
                                      for hb in head_union)]
        return _dedup_boxes(list(head_union) + _face_to_head(uncovered_faces))

    return head_union

# Per-source boxes of the most recent detect_objects call (for debug overlay).
LAST_DETECT_BREAKDOWN = {"face": [], "person": [], "head": []}

# Debug overlay colors (BGR): which pass produced which box.
SOURCE_COLORS = (("face",   (255, 255,   0), "FACE"),       # cyan
                 ("person", (  0, 255, 255), "PERSON->HEAD"),# yellow
                 ("head",   (  0,   0, 255), "HEAD MODEL"))  # red

def draw_source_debug(frame, breakdown):
    """Thin per-source outlines + legend, so debug mode shows WHICH detector
    found each head (face model vs person->region vs head.pt)."""
    y = 18
    for key, color, label in SOURCE_COLORS:
        boxes = breakdown.get(key, [])
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, "{}: {}".format(label, len(boxes)), (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 16
    return frame

def apply_censor(frame, boxes, mode, padding, intensity, block_size, debug):
    """Apply censor with confidence-weighted blur intensity."""
    h, w = frame.shape[:2]
    for box in boxes:
        if len(box) == 5:
            x1, y1, x2, y2, score = box
        else:
            x1, y1, x2, y2 = box
            score = 1.0
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, padding, w, h)
        # Scale blur intensity by confidence (heatmap effect)
        conf_intensity = max(11, int(intensity * score))
        if conf_intensity % 2 == 0:
            conf_intensity += 1
        if   mode == "Blur":      frame = blur_region(frame, x1, y1, x2, y2, conf_intensity)
        elif mode == "Pixelate":  frame = pixelate_region(frame, x1, y1, x2, y2, block_size)
        elif mode == "Black Bar": frame = black_bar_region(frame, x1, y1, x2, y2)
        if debug:
            color = (0, int(255*score), int(255*(1-score)))  # green=high, red=low conf
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "{:.0f}%".format(score*100),
                        (x1, max(y1-4, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1)
    return frame

def process_frame(frame, detector, mode, padding, intensity, block_size, confidence, debug,
                  detect_scale=0.5):
    boxes = detect_faces(frame, detector, confidence, detect_scale)
    frame = apply_censor(frame, boxes, mode, padding, intensity, block_size, debug)
    return frame, len(boxes)

def make_output_path(input_path, outdir, suffix="_blurred"):
    base   = os.path.splitext(os.path.basename(input_path))[0]
    folder = outdir if outdir else os.path.dirname(os.path.abspath(input_path))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, base + suffix + ".mp4")

def save_report(report_path, files_data):
    """Save processing report as JSON."""
    try:
        with open(report_path, "w") as f:
            json.dump(files_data, f, indent=2)
    except Exception:
        pass

def get_resume_frame(tmp_path, fps):
    """Check how many frames exist in a partial output file."""
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, n - 2)  # -2 for safety margin
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════
#  FFMPEG HELPER
# ══════════════════════════════════════════════════════════

def get_ffmpeg_path():
    """
    Resolve ffmpeg.exe path:
    1. Bundled inside PyInstaller exe (sys._MEIPASS)
    2. Next to the exe / script
    3. System PATH
    """
    import sys
    ffmpeg_exe = "ffmpeg.exe"

    # PyInstaller bundle
    if hasattr(sys, "_MEIPASS"):
        bundled = os.path.join(sys._MEIPASS, ffmpeg_exe)
        if os.path.exists(bundled):
            return bundled

    # Next to exe/script
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    local = os.path.join(base_dir, ffmpeg_exe)
    if os.path.exists(local):
        return local

    # System PATH
    import shutil
    system = shutil.which("ffmpeg")
    if system:
        return system

    return None

# Single-pass encoder settings (video is encoded exactly ONCE here).
PIPE_PRESET = "veryfast"   # libx264 preset; keeps encode from bottlenecking detection
PIPE_CRF    = "23"         # quality (lower = better/larger); 23 is x264's default

def open_ffmpeg_pipe(ffmpeg_path, original_path, output_path, W, H, fps,
                     stderr_fp, log_fn=None):
    """Start an ffmpeg process that encodes censored frames -- fed as raw BGR24
    bytes on stdin -- directly to the final H.264 mp4, muxing the ORIGINAL audio
    in the SAME pass.

    This replaces the old "write XVID temp, then re-encode to H.264" design, which
    encoded every frame TWICE (XVID, then H.264) with a decode in between -- double
    the encode time, double the generation loss, plus a temp file. Here each frame
    is encoded once and the audio is muxed inline, so there is no separate merge
    step (and the encode time is naturally part of the processing ETA).

    Returns the Popen (write frames to .stdin) or None if ffmpeg can't start.
    `-map 1:a:0?` makes audio optional, so silent sources don't fail.
    """
    import subprocess
    fps_str = "{:.6f}".format(fps if fps and fps > 0 else 30.0)
    cmd = [
        ffmpeg_path, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "{}x{}".format(W, H), "-r", fps_str, "-i", "-",  # input 0: frames
        "-i", original_path,                                    # input 1: audio src
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", PIPE_PRESET, "-crf", PIPE_CRF,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=stderr_fp)
    except Exception as e:
        if log_fn:
            log_fn("  [WARN] could not start ffmpeg encoder ({}).\n".format(e), "warning")
        return None

def merge_audio(original_path, muted_path, output_path, ffmpeg_path, log_fn=None):
    """
    LEGACY two-pass merge (only used as a fallback when the single-pass pipe in
    open_ffmpeg_pipe can't start). Copies audio from the original into an already
    re-encoded video. Returns True on success, False on failure.
    """
    import subprocess
    if log_fn:
        log_fn("  Merging audio...\n", "dim")
    cmd = [
        ffmpeg_path,
        "-y",                        # overwrite output
        "-i", muted_path,            # processed video (no audio)
        "-i", original_path,         # original (has audio)
        "-c:v", "libx264",           # re-encode with H.264 for smaller files
        "-c:a", "aac",               # encode audio as AAC
        "-map", "0:v:0",             # video from processed file
        "-map", "1:a:0",             # audio from original file
        "-shortest",                 # end when shortest stream ends
        output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            if log_fn:
                log_fn("  Audio merged OK.\n", "dim")
            return True
        else:
            if log_fn:
                log_fn("  [WARN] ffmpeg error: {}\n".format(result.stderr[-200:]), "warning")
            return False
    except Exception as e:
        if log_fn:
            log_fn("  [WARN] ffmpeg failed: {}\n".format(e), "warning")
        return False


# ══════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════

BG      = "#0f0f0f"
BG2     = "#1a1a1a"
BG3     = "#2a2a2a"
BORDER  = "#3a3a3a"
ACCENT  = "#00e5ff"
ACCENT2 = "#ff3c6e"
TEXT    = "#f0f0f0"
TDIM    = "#aaaaaa"   # was #666666 - much more readable
TMID    = "#cccccc"   # was #999999 - more readable
SUCCESS = "#00e88a"
WARNING = "#ffb347"

FH = ("Courier New", 20, "bold")
FL = ("Courier New", 10, "bold")
FS = ("Courier New",  9)
FM = ("Courier New",  9)
FV = ("Courier New", 11, "bold")


# ══════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self, gpu_info=None):
        super().__init__()
        self.title("FACEBLUR v1.4")
        self.configure(bg=BG)
        self.geometry("960x780")
        self.minsize(900, 700)
        self._files        = []
        self._cancel_flag  = threading.Event()
        # IMPORTANT: bind every Variable to THIS root (master=self). The splash
        # is a separate Tk() and is still alive when App is constructed, so a
        # Variable created without a master would attach to the splash's
        # interpreter. The widgets live in THIS interpreter, so after the splash
        # is destroyed, .get()/.set() and the widgets would read/write DIFFERENT
        # Tcl variables -- every live UI change (e.g. frame skip, smooth boxes)
        # would be silently ignored and processing would run with stale values.
        self._outdir       = tk.StringVar(self)
        self._mode         = tk.StringVar(self, value="Blur")
        self._model_key    = tk.StringVar(self, value=list(YOLO_MODELS.keys())[0])
        self._conf         = tk.DoubleVar(self, value=0.40)
        self._pad          = tk.DoubleVar(self, value=0.25)
        self._blur_k       = tk.IntVar(self, value=51)
        self._pixel_sz     = tk.IntVar(self, value=15)
        self._debug        = tk.BooleanVar(self, value=False)
        self._skip_frames  = tk.IntVar(self, value=2)
        self._detect_scale = tk.DoubleVar(self, value=0.50)
        self._gpu_info     = gpu_info if gpu_info is not None else detect_gpu()
        self._suffix       = tk.StringVar(self, value="_blurred")
        self._edge_strip   = tk.BooleanVar(self, value=False)
        self._detect_head  = tk.BooleanVar(self, value=True)
        self._use_face     = tk.BooleanVar(self, value=False)
        self._use_person_aid = tk.BooleanVar(self, value=False)
        self._head_size    = tk.StringVar(self, value=HEAD_MODEL_DEFAULT_KEY)
        self._smooth_boxes = tk.BooleanVar(self, value=True)
        self._export_report = tk.BooleanVar(self, value=False)
        self._file_status  = {}
        self._build()

    # ── BUILD ─────────────────────────────────────────────

    def _build(self):
        # ── TOP BAR ──
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(top, text="FACEBLUR", bg=BG, fg=ACCENT, font=FH).pack(side="left")
        tk.Label(top, text="YOLOv11 head & face censoring", bg=BG, fg=TDIM, font=FS).pack(side="left", padx=12)
        tk.Label(top, text="v1.4", bg=BG, fg=TDIM, font=("Courier New", 8)).pack(side="left")
        tk.Label(top, text="made by werehappy", bg=BG, fg=TDIM, font=("Courier New", 8)).pack(side="left", padx=4)
        # GPU/CPU indicator (right side of top bar)
        if self._gpu_info["torch_cuda"]:
            device_text  = "GPU: " + (self._gpu_info.get("gpu_name") or "CUDA device")
            device_color = SUCCESS
        elif self._gpu_info["has_nvidia"]:
            device_text  = "GPU available — not enabled"
            device_color = WARNING
        else:
            device_text  = "CPU only"
            device_color = WARNING
        self._device_lbl = tk.Label(top, text=device_text, bg=BG,
                                    fg=device_color, font=FL)
        self._device_lbl.pack(side="right")
        # Recovery path: GPU present but no CUDA torch -> let the user install
        # the GPU build into faceblur_env with one click (no reinstall needed).
        self._enable_gpu_btn = None
        if self._gpu_info["has_nvidia"] and not self._gpu_info["torch_cuda"]:
            self._enable_gpu_btn = self._btn(top, "ENABLE GPU",
                                             self._enable_gpu, accent=True)
            self._enable_gpu_btn.pack(side="right", padx=(0, 10))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(10, 0))

        self._gpu_banner = None

        # ── MAIN AREA ──
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=10)

        left  = tk.Frame(main, bg=BG, width=370)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        # Right panel in a canvas for scrolling when sections collapse/expand
        right_outer = tk.Frame(main, bg=BG)
        right_outer.pack(side="left", fill="both", expand=True)
        right = tk.Frame(right_outer, bg=BG)
        right.pack(fill="x", anchor="n")

        # ── LEFT PANEL ──
        # file list card
        fc = tk.LabelFrame(left, text=" INPUT FILES ", bg=BG2, fg=TDIM,
                           font=FL, relief="flat", bd=1,
                           highlightbackground=BORDER, highlightthickness=1)
        fc.pack(fill="x", pady=(0, 8))

        hdr_row = tk.Frame(fc, bg=BG2)
        hdr_row.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(hdr_row, text="FILE", bg=BG2, fg=TDIM, font=FL, width=20, anchor="w").pack(side="left")
        tk.Label(hdr_row, text="STATUS", bg=BG2, fg=TDIM, font=FL, anchor="e").pack(side="right")
        self._count_lbl = tk.Label(hdr_row, text="0 files", bg=BG2, fg=ACCENT, font=FS)
        self._count_lbl.pack(side="right", padx=8)

        sb1 = tk.Scrollbar(fc, bg=BG2, troughcolor=BG3)
        sb1.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self._lb = tk.Listbox(fc, bg=BG2, fg=TEXT, font=FS, height=8,
                              selectbackground=BG3, selectforeground=ACCENT,
                              relief="flat", bd=0, highlightthickness=0,
                              activestyle="none", yscrollcommand=sb1.set)
        self._lb.pack(fill="x", padx=(8, 0), pady=(0, 4))
        sb1.config(command=self._lb.yview)

        # Drag and drop support
        tk.Label(fc, text="or drag & drop files here",
                 bg=BG2, fg=TDIM, font=("Courier New", 8)).pack(pady=(0, 4))
        self._setup_drag_drop(self._lb)
        self._setup_drag_drop(fc)

        br = tk.Frame(fc, bg=BG2)
        br.pack(fill="x", padx=8, pady=(0, 8))
        self._btn(br, "+ ADD",   self._add_files,   accent=True).pack(side="left")
        self._btn(br, "REMOVE",  self._remove_sel).pack(side="left", padx=4)
        self._btn(br, "CLEAR",   self._clear_files, danger=True).pack(side="left")

        # output dir card
        oc = tk.LabelFrame(left, text=" OUTPUT FOLDER ", bg=BG2, fg=TDIM,
                           font=FL, relief="flat", bd=1,
                           highlightbackground=BORDER, highlightthickness=1)
        oc.pack(fill="x", pady=(0, 8))
        er = tk.Frame(oc, bg=BG2)
        er.pack(fill="x", padx=8, pady=8)
        tk.Entry(er, textvariable=self._outdir, bg=BG3, fg=TEXT, font=FS,
                 relief="flat", bd=4, insertbackground=ACCENT
                 ).pack(side="left", fill="x", expand=True)
        self._btn(er, "BROWSE", self._pick_outdir).pack(side="left", padx=(4, 0))
        tk.Label(oc, text="Blank = same folder as source", bg=BG2, fg=TDIM,
                 font=FS).pack(anchor="w", padx=8, pady=(0, 8))

        # thumbnail preview card
        self._thumb_card = tk.LabelFrame(left, text=" PREVIEW ", bg=BG2, fg=TDIM,
                                          font=FL, relief="flat", bd=1,
                                          highlightbackground=BORDER, highlightthickness=1)
        self._thumb_card.pack(fill="x", pady=(0, 8))
        self._thumb_lbl = tk.Label(self._thumb_card, bg=BG2, fg=TDIM,
                                    text="Select a file to preview",
                                    font=("Courier New", 8))
        self._thumb_lbl.pack(pady=8)
        # Update preview when selection changes
        self._lb.bind("<<ListboxSelect>>", self._update_preview)

        # Tips button — replaces inline tips card
        tips_frame = tk.Frame(left, bg=BG)
        tips_frame.pack(fill="x", pady=(4, 0))
        self._btn(tips_frame, "?  TIPS & SHORTCUTS",
                  self._show_tips).pack(fill="x")

        # ── RIGHT PANEL ──
        # censor mode
        self._section(right, "CENSOR MODE")
        mr = tk.Frame(right, bg=BG2)
        mr.pack(fill="x", padx=8, pady=(0, 8))
        for m in ["Blur", "Pixelate", "Black Bar"]:
            btn = tk.Label(mr, text=m, font=FL, padx=12, pady=5,
                           cursor="hand2", relief="flat")
            btn.pack(side="left", padx=(0, 2))
            btn.bind("<Button-1>", lambda e, v=m: (
                self._mode.set(v), self._refresh_mode_btns(),
                self._update_suffix_preview()))
            btn.bind("<Enter>", lambda e, b=btn: b.config(
                bg="#505050" if b["bg"] != ACCENT else "#33ecff",
                fg="#0f0f0f"))
            btn.bind("<Leave>", lambda e, b=btn: self._refresh_mode_btns())
        self._mode_btns = mr.winfo_children()
        self._refresh_mode_btns()

        # head model (PRIMARY detector) — button row like the old model picker
        self._section(right, "HEAD MODEL")
        hkr = tk.Frame(right, bg=BG2)
        hkr.pack(fill="x", padx=8, pady=(0, 4))
        for k in HEAD_MODELS.keys():
            btn = tk.Label(hkr, text=k, font=FL, padx=10, pady=4,
                           cursor="hand2", relief="flat")
            btn.pack(side="left", padx=(0, 2))
            btn.bind("<Button-1>", lambda e, v=k: (
                self._head_size.set(v), self._refresh_head_btns()))
            btn.bind("<Enter>", lambda e, b=btn: b.config(
                bg="#505050" if b["bg"] != ACCENT else "#33ecff",
                fg="#0f0f0f"))
            btn.bind("<Leave>", lambda e, b=btn: self._refresh_head_btns())
        self._head_btns = hkr.winfo_children()
        self._refresh_head_btns()
        tk.Label(right, text="Primary detector \u2014 bundled with the app; "
                 "bigger = more accurate, slower",
                 bg=BG, fg=TDIM, font=FS).pack(anchor="w", padx=8, pady=(0, 6))

        # ── Collapsible PARAMETERS ──
        self._params_body = self._collapsible(right, "PARAMETERS")
        self._slider(self._params_body, "Confidence  ", self._conf,   0.10, 1.00, 0.05)
        self._slider(self._params_body, "Padding     ", self._pad,    0.00, 0.60, 0.05)
        self._slider(self._params_body, "Blur kernel ", self._blur_k, 11,   101,  2)
        self._slider(self._params_body, "Pixel size  ", self._pixel_sz, 5,  40,   1)

        # ── Collapsible OPTIONS ──
        self._opts_body = self._collapsible(right, "OPTIONS")

        opts_row1 = tk.Frame(self._opts_body, bg=BG)
        opts_row1.pack(fill="x", padx=8, pady=(0, 2))
        tk.Checkbutton(opts_row1, text="Show debug boxes",
                       variable=self._debug,
                       bg=BG, fg=TMID, font=FS, selectcolor=BG3,
                       activebackground=BG, activeforeground=ACCENT
                       ).pack(side="left")
        tk.Checkbutton(opts_row1, text="Edge strip detection",
                       variable=self._edge_strip,
                       bg=BG, fg=TMID, font=FS, selectcolor=BG3,
                       activebackground=BG, activeforeground=ACCENT
                       ).pack(side="left", padx=(12, 0))

        opts_row2 = tk.Frame(self._opts_body, bg=BG)
        opts_row2.pack(fill="x", padx=8, pady=(0, 2))
        tk.Checkbutton(opts_row2, text="Export report (.json)",
                       variable=self._export_report,
                       bg=BG, fg=TMID, font=FS, selectcolor=BG3,
                       activebackground=BG, activeforeground=ACCENT
                       ).pack(side="left")
        tk.Checkbutton(opts_row2, text="Person\u2192head aid (rescue)",
                       variable=self._use_person_aid,
                       bg=BG, fg=TMID, font=FS, selectcolor=BG3,
                       activebackground=BG, activeforeground=ACCENT
                       ).pack(side="left", padx=(12, 0))
        tk.Checkbutton(opts_row2, text="Face safety net",
                       variable=self._use_face,
                       bg=BG, fg=TMID, font=FS, selectcolor=BG3,
                       activebackground=BG, activeforeground=ACCENT
                       ).pack(side="left", padx=(12, 0))
        tk.Checkbutton(opts_row2, text="Smooth boxes (anti-flicker)",
                       variable=self._smooth_boxes,
                       bg=BG, fg=TMID, font=FS, selectcolor=BG3,
                       activebackground=BG, activeforeground=ACCENT
                       ).pack(side="left", padx=(12, 0))

        # Face-model selector: ONLY the face safety net uses this model, so it
        # is enabled only when that toggle is on (the person->head aid uses a
        # fixed COCO person model, not this one). Grayed out otherwise.
        face_row = tk.Frame(self._opts_body, bg=BG)
        face_row.pack(fill="x", padx=8, pady=(2, 4))
        self._face_model_lbl = tk.Label(face_row, text="Face model:", bg=BG,
                                        fg=TMID, font=FS)
        self._face_model_lbl.pack(side="left")
        self._face_om = tk.OptionMenu(face_row, self._model_key, *YOLO_MODELS.keys())
        self._face_om.config(bg=BG3, fg=TEXT, font=FS, activebackground=BG3,
                             activeforeground=ACCENT, highlightthickness=0,
                             bd=0, cursor="hand2")
        try:
            self._face_om["menu"].config(bg=BG3, fg=TEXT, activebackground=ACCENT,
                                         activeforeground=BG, font=FS)
        except Exception:
            pass
        self._face_om.pack(side="left", padx=6)
        self._face_model_hint = tk.Label(
            face_row, text="(used only by the face safety net; downloaded on "
            "first use)", bg=BG, fg=TDIM, font=FS)
        self._face_model_hint.pack(side="left")
        # React to the face-net toggle to enable/disable this selector.
        self._use_face.trace_add("write", lambda *a: self._refresh_face_model_row())
        self._refresh_face_model_row()

        suffix_row = tk.Frame(self._opts_body, bg=BG)
        suffix_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(suffix_row, text="Output suffix:", bg=BG, fg=TMID,
                 font=FS).pack(side="left")
        tk.Label(suffix_row, text="auto (based on mode)", bg=BG, fg=TDIM,
                 font=FS).pack(side="left", padx=6)
        self._suffix_preview = tk.Label(suffix_row, text="_blurred",
                                        bg=BG, fg=ACCENT, font=FL)
        self._suffix_preview.pack(side="left", padx=4)

        # ── Collapsible PERFORMANCE ──
        self._perf_body = self._collapsible(right, "PERFORMANCE")
        self._slider(self._perf_body, "Frame skip  ", self._skip_frames,  1, 6,   1)
        self._slider(self._perf_body, "Detect scale", self._detect_scale, 0.25, 1.0, 0.25)
        tk.Label(self._perf_body,
                 text="Frame skip: detect every N frames  |  Scale: resize before detection",
                 bg=BG, fg=TDIM, font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(0, 4))

        # progress
        self._section(right, "PROGRESS")
        self._status_lbl = tk.Label(right, text="Ready", bg=BG, fg=TDIM, font=FS, anchor="w")
        self._status_lbl.pack(fill="x", padx=8, pady=(0, 4))

        pb_frame = tk.Frame(right, bg=BG3, height=6)
        pb_frame.pack(fill="x", padx=8, pady=(0, 6))
        pb_frame.pack_propagate(False)
        self._pb_fill = tk.Frame(pb_frame, bg=ACCENT)
        self._pb_fill.place(x=0, y=0, width=0, height=6)
        self._pb_frame = pb_frame

        sr = tk.Frame(right, bg=BG)
        sr.pack(fill="x", padx=8, pady=(0, 8))
        self._lbl_file  = self._stat(sr, "FILE",  "-")
        self._lbl_frame = self._stat(sr, "FRAME", "-")
        self._lbl_faces = self._stat(sr, "FACES", "-")
        self._lbl_fps   = self._stat(sr, "FPS",   "-")
        self._lbl_eta   = self._stat(sr, "ETA",   "-")

        # action buttons
        ar = tk.Frame(right, bg=BG)
        ar.pack(fill="x", padx=8, pady=(4, 8))
        self._proc_btn   = self._btn(ar, "PROCESS", self._start, accent=True)
        self._cancel_btn = self._btn(ar, "CANCEL",  self._cancel, danger=True)
        self._reset_btn  = self._btn(ar, "RESET DEFAULTS", self._reset_defaults)
        self._proc_btn.pack(side="left")
        self._cancel_btn.pack(side="left", padx=8)
        self._reset_btn.pack(side="right")
        self._set_running(False)

        # ── LOG (bottom, fixed height) ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 0))
        lh = tk.Frame(self, bg=BG)
        lh.pack(fill="x", padx=20, pady=(4, 2))
        tk.Label(lh, text="LOG", bg=BG, fg=TDIM, font=FL).pack(side="left")
        self._btn(lh, "CLEAR", self._clear_log).pack(side="right")

        log_frame = tk.Frame(self, bg=BG, height=160)
        log_frame.pack(fill="x", padx=20, pady=(0, 12))
        log_frame.pack_propagate(False)

        sb2 = tk.Scrollbar(log_frame, bg=BG2, troughcolor=BG3)
        sb2.pack(side="right", fill="y")
        self._log_text = tk.Text(log_frame, bg=BG, fg=TMID, font=FM,
                                 relief="flat", bd=0, wrap="word",
                                 yscrollcommand=sb2.set, state="disabled",
                                 selectbackground=BG3, insertbackground=ACCENT)
        self._log_text.pack(fill="both", expand=True)
        sb2.config(command=self._log_text.yview)

        self._log_text.tag_config("accent",  foreground=ACCENT)
        self._log_text.tag_config("success", foreground=SUCCESS)
        self._log_text.tag_config("warning", foreground=WARNING)
        self._log_text.tag_config("error",   foreground=ACCENT2)
        self._log_text.tag_config("dim",     foreground=TDIM)

        import sys
        # Load saved settings
        self._load_settings()
        # Save settings on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Keyboard shortcuts
        self.bind("<Control-o>", lambda e: self._add_files())
        self.bind("<Control-O>", lambda e: self._add_files())
        self.bind("<Control-Return>", lambda e: self._start())
        self.bind("<Escape>", lambda e: self._cancel())
        self.bind("<Delete>", lambda e: self._remove_sel())
        # Restore window geometry
        settings = load_settings()
        if "geometry" in settings:
            try: self.geometry(settings["geometry"])
            except Exception: pass
        self._update_suffix_preview()
        self._write_log("FACEBLUR ready.  (YOLOv11-face)\n", "accent")
        self._write_log("Python: {}{}\n".format(
            sys.executable,
            " [bundled]" if getattr(sys, "frozen", False) else ""), "dim")
        gi = self._gpu_info
        if gi["torch_cuda"]:
            self._write_log("Device: GPU ({})\n".format(gi["gpu_name"]), "success")
        elif gi["has_nvidia"]:
            import sys
            is_gpu_build = getattr(sys, "frozen", False) and "GPU" in sys.executable.upper()
            if is_gpu_build:
                self._write_log("Device: CPU  (GPU build but CUDA unavailable)\n", "warning")
                self._write_log("  Your Nvidia driver may be outdated.\n", "warning")
                self._write_log("  Update at: nvidia.com/drivers  then restart FACEBLUR.\n", "warning")
            else:
                self._write_log("Device: CPU  (Nvidia GPU detected but no CUDA torch)\n", "warning")
        else:
            self._write_log("Device: CPU only  (no Nvidia GPU detected)\n", "dim")

    # ── WIDGET HELPERS ────────────────────────────────────

    def _btn(self, parent, text, cmd, accent=False, danger=False):
        bg = ACCENT2 if danger else (ACCENT if accent else "#3a3a3a")
        fg = "#0f0f0f" if (accent or danger) else "#ffffff"
        b = tk.Label(parent, text=text, bg=bg, fg=fg, font=FL,
                     cursor="hand2", padx=12, pady=6, relief="flat")
        b._bg = bg; b._accent = accent; b._danger = danger
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>",    lambda e: b.config(bg="#ff6b8a" if danger else ("#33ecff" if accent else "#4a4a4a")))
        b.bind("<Leave>",    lambda e: b.config(bg=b._bg))
        return b

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=BG, fg=TDIM, font=FL
                 ).pack(anchor="w", padx=8, pady=(10, 4))

    def _slider(self, parent, label, var, from_, to, res):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=8, pady=1)

        tk.Label(row, text=label, bg=BG, fg=TMID, font=FL,
                 width=13, anchor="w").pack(side="left")

        # Use tkinter's built-in showvalue with custom styling
        # digits controls decimal places shown
        digits = 2 if res < 1 else 0
        sc = tk.Scale(row, variable=var, from_=from_, to=to, resolution=res,
                      orient="horizontal", showvalue=True,
                      digits=digits,
                      bg=BG, fg=ACCENT,
                      font=("Courier New", 9, "bold"),
                      troughcolor="#1e1e1e",
                      activebackground="#ffffff",
                      highlightthickness=0,
                      bd=0, sliderlength=20, sliderrelief="raised", width=10)
        sc.pack(side="left", padx=4, fill="x", expand=True)

    def _collapsible(self, parent, title, default_open=False):
        """Create a collapsible section. Returns the body frame to pack widgets into."""
        hdr = tk.Frame(parent, bg=BG, cursor="hand2")
        hdr.pack(fill="x", pady=(6, 0))

        arrow_lbl = tk.Label(hdr, text="▼" if default_open else "▶",
                             bg=BG, fg=ACCENT, font=FL, width=2)
        arrow_lbl.pack(side="left", padx=(8, 2))
        tk.Label(hdr, text=title, bg=BG, fg=TMID,
                 font=FL).pack(side="left")
        tk.Frame(hdr, bg=BORDER, height=1).pack(side="left", fill="x",
                                                 expand=True, padx=8)

        # Body frame packed immediately after header
        body = tk.Frame(parent, bg=BG)
        if default_open:
            body.pack(fill="x")

        is_open = [default_open]

        def toggle(e=None):
            if is_open[0]:
                body.pack_forget()
                arrow_lbl.config(text="▶")
            else:
                body.pack(fill="x", after=hdr)
                arrow_lbl.config(text="▼")
            is_open[0] = not is_open[0]

        hdr.bind("<Button-1>", toggle)
        arrow_lbl.bind("<Button-1>", toggle)
        hdr.bind("<Enter>", lambda e: hdr.config(bg="#1a1a1a"))
        hdr.bind("<Leave>", lambda e: hdr.config(bg=BG))

        return body  # caller packs widgets into this frame

    def _reset_defaults(self):
        """Reset all parameters to recommended defaults."""
        self._conf.set(0.40)
        self._pad.set(0.25)
        self._blur_k.set(51)
        self._pixel_sz.set(15)
        self._skip_frames.set(2)
        self._detect_scale.set(0.50)
        self._debug.set(False)
        self._edge_strip.set(False)
        self._detect_head.set(True)
        self._use_face.set(False)
        self._use_person_aid.set(False)
        self._head_size.set(HEAD_MODEL_DEFAULT_KEY)
        self._smooth_boxes.set(True)
        self._export_report.set(False)
        self._mode.set("Blur")
        self._model_key.set(list(YOLO_MODELS.keys())[0])
        self._refresh_mode_btns()
        self._refresh_model_btns()
        self._refresh_head_btns()
        self._refresh_face_model_row()
        self._update_suffix_preview()
        # Delete settings file so it saves fresh
        try:
            if os.path.exists(SETTINGS_FILE):
                os.remove(SETTINGS_FILE)
        except Exception:
            pass
        self._write_log("Settings reset to defaults.\n", "accent")

    def _update_suffix_preview(self):
        """Update the suffix preview label to match current mode."""
        suffix = MODE_SUFFIX.get(self._mode.get(), "_blurred")
        self._suffix.set(suffix)
        if hasattr(self, "_suffix_preview"):
            self._suffix_preview.config(text=suffix)

    def _show_tips(self):
        """Open tips in a small popup window."""
        win = tk.Toplevel(self)
        win.title("FACEBLUR — Tips")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.geometry("380x320")
        # Center over main window
        x = self.winfo_x() + self.winfo_width()//2 - 190
        y = self.winfo_y() + self.winfo_height()//2 - 160
        win.geometry("+{}+{}".format(x, y))
        win.attributes("-topmost", True)

        tk.Label(win, text="TIPS", bg=BG, fg=ACCENT,
                 font=FL).pack(anchor="w", padx=16, pady=(14, 8))
        tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=16)

        tips = [
            ("Frame skip 2-3",    "2-3x faster, barely noticeable"),
            ("Detect scale 0.5",  "Half-res detection, 2x faster"),
            ("Nano model",        "Fastest, good for most videos"),
            ("Confidence 0.3",    "Catches partial/side faces"),
            ("Padding 0.3-0.4",   "Covers hair and chin edges"),
            ("Edge strip ON",     "Detects faces near frame border"),
            ("GPU = green label", "Top-right shows active device"),
            ("Ctrl+O",            "Add files"),
            ("Ctrl+Enter",        "Start processing"),
            ("Escape",            "Cancel processing"),
        ]
        for title, desc in tips:
            row = tk.Frame(win, bg=BG)
            row.pack(fill="x", padx=16, pady=3)
            tk.Label(row, text=u"▸ " + title, bg=BG, fg=ACCENT,
                     font=("Courier New", 9, "bold"), anchor="w",
                     width=18).pack(side="left")
            tk.Label(row, text=desc, bg=BG, fg=TMID,
                     font=("Courier New", 9), anchor="w").pack(side="left")

        self._btn(win, "CLOSE", win.destroy).pack(pady=(12, 14))

    def _stat(self, parent, label, value):
        f = tk.Frame(parent, bg=BG)
        f.pack(side="left", padx=(0, 20))
        tk.Label(f, text=label, bg=BG, fg=TDIM, font=FS).pack()
        lbl = tk.Label(f, text=value, bg=BG, fg=ACCENT, font=FV)
        lbl.pack()
        return lbl

    def _style_radio_btn(self, btn, active):
        btn.config(bg=ACCENT if active else "#3a3a3a",
                   fg="#0f0f0f" if active else "#cccccc")

    def _refresh_mode_btns(self):
        cur = self._mode.get()
        for b in self._mode_btns:
            active = (b["text"] == cur)
            b.config(bg=ACCENT if active else "#3a3a3a",
                     fg="#0f0f0f" if active else "#cccccc")
            def _enter(e, btn=b):
                is_active = btn["text"] == self._mode.get()
                btn.config(bg="#33ecff" if is_active else "#505050",
                           fg="#0f0f0f")
            def _leave(e, btn=b):
                is_active = btn["text"] == self._mode.get()
                btn.config(bg=ACCENT if is_active else "#3a3a3a",
                           fg="#0f0f0f" if is_active else "#cccccc")
            b.bind("<Enter>", _enter)
            b.bind("<Leave>", _leave)

    def _refresh_model_btns(self):
        # The face model is now an OptionMenu (see _refresh_face_model_row), not
        # a button row. Kept as a safe no-op so older call sites don't break.
        return

    def _refresh_head_btns(self):
        """Highlight the selected head-model size button (primary detector)."""
        cur = self._head_size.get()
        for b in getattr(self, "_head_btns", []):
            active = (b["text"] == cur)
            b.config(bg=ACCENT if active else "#3a3a3a",
                     fg="#0f0f0f" if active else "#cccccc")
            def _enter(e, btn=b):
                is_active = btn["text"] == self._head_size.get()
                btn.config(bg="#33ecff" if is_active else "#505050",
                           fg="#0f0f0f")
            def _leave(e, btn=b):
                is_active = btn["text"] == self._head_size.get()
                btn.config(bg=ACCENT if is_active else "#3a3a3a",
                           fg="#0f0f0f" if is_active else "#cccccc")
            b.bind("<Enter>", _enter)
            b.bind("<Leave>", _leave)

    def _refresh_face_model_row(self):
        """Enable the face-model selector only when the face safety net is on;
        it is the sole consumer of the selected model. Gray it out otherwise so
        it is clear the choice has no effect."""
        on = bool(self._use_face.get())
        state = "normal" if on else "disabled"
        try:
            self._face_om.config(state=state)
            self._face_model_lbl.config(fg=TMID if on else TDIM)
            self._face_model_hint.config(
                text=("(used only by the face safety net; downloaded on first use)"
                      if on else "(enable 'Face safety net' to use a face model)"))
        except Exception:
            pass

    # ── DRAG AND DROP ─────────────────────────────────────

    def _setup_drag_drop(self, widget):
        try:
            widget.drop_target_register("DND_Files")
            widget.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            try:
                self.tk.call("package", "require", "tkdnd")
                widget.drop_target_register("DND_Files")
                widget.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass

    def _on_drop(self, event):
        import re
        raw = event.data
        paths = re.findall(r'[{]([^}]+)[}]|(\S+)', raw)
        paths = [p[0] or p[1] for p in paths]
        exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        for p in paths:
            p = p.strip()
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                if p not in self._files:
                    self._files.append(p)
                    self._lb.insert("end", os.path.basename(p))
        n = len(self._files)
        self._count_lbl.config(text="{} file{}".format(n, "s" if n != 1 else ""))

    # ── SETTINGS ──────────────────────────────────────────

    def _save_settings(self):
        data = {
            "settings_version": SETTINGS_VERSION,
            "mode": self._mode.get(), "model_key": self._model_key.get(),
            "confidence": self._conf.get(), "padding": self._pad.get(),
            "blur_k": self._blur_k.get(), "pixel_sz": self._pixel_sz.get(),
            "debug": self._debug.get(), "skip_frames": self._skip_frames.get(),
            "detect_scale": self._detect_scale.get(), "outdir": self._outdir.get(),
            "suffix": self._suffix.get(), "edge_strip": self._edge_strip.get(),
            "export_report": self._export_report.get(), "geometry": self.geometry(),
            "detect_head": self._detect_head.get(),
            "head_size": self._head_size.get(),
            "smooth_boxes": self._smooth_boxes.get(),
            "use_face": self._use_face.get(),
            "use_person_aid": self._use_person_aid.get(),
        }
        save_settings(data)

    def _load_settings(self):
        data = load_settings()
        if not data:
            return
        try:
            keys = ["mode", "model_key", "confidence", "padding", "blur_k",
                    "pixel_sz", "debug", "skip_frames", "detect_scale",
                    "outdir", "suffix", "edge_strip", "export_report",
                    "detect_head", "smooth_boxes", "head_size",
                    "use_face", "use_person_aid"]
            targets = [self._mode, self._model_key, self._conf, self._pad,
                       self._blur_k, self._pixel_sz, self._debug,
                       self._skip_frames, self._detect_scale, self._outdir,
                       self._suffix, self._edge_strip, self._export_report,
                       self._detect_head, self._smooth_boxes, self._head_size,
                       self._use_face, self._use_person_aid]
            for k, t in zip(keys, targets):
                if k in data:
                    t.set(data[k])
            self._refresh_mode_btns()
            self._refresh_model_btns()
            self._refresh_head_btns()
            self._refresh_face_model_row()
        except Exception:
            pass

    # ── TOAST ─────────────────────────────────────────────

    def _toast(self, title, message):
        """Show Windows notification using PowerShell (no extra packages needed)."""
        def _do():
            try:
                import subprocess
                # Use PowerShell BurntToast or fallback to msg/balloon
                ps_script = (
                    "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
                    "ContentType = WindowsRuntime] | Out-Null; "
                    "[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, "
                    "ContentType = WindowsRuntime] | Out-Null; "
                    "$template = [Windows.UI.Notifications.ToastNotificationManager]"
                    "::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
                    "$template.SelectSingleNode('//text[@id=1]').InnerText = '{title}'; "
                    "$template.SelectSingleNode('//text[@id=2]').InnerText = '{msg}'; "
                    "$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
                    "[Windows.UI.Notifications.ToastNotificationManager]"
                    "::CreateToastNotifier('FACEBLUR').Show($toast)"
                ).format(title=title.replace("'", ""), msg=message.replace("'", ""))
                subprocess.Popen(
                    ["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                     "-Command", ps_script],
                    creationflags=0x08000000  # CREATE_NO_WINDOW
                )
            except Exception:
                pass  # notifications are optional, fail silently
        threading.Thread(target=_do, daemon=True).start()

    # ── GPU ENABLE ────────────────────────────────────────

    def _enable_gpu(self):
        self._enable_gpu_btn.config(text="Installing...", bg=BG3, fg=TDIM, cursor="arrow")
        self._enable_gpu_btn.unbind("<Button-1>")
        gi = self._gpu_info

        def _do():
            success = install_torch_for_cuda(gi["cuda_ver"], log_fn=self._write_log,
                                             has_nvidia=gi.get("has_nvidia", False))
            if success:
                try:
                    import importlib, torch
                    importlib.reload(torch)
                    gpu_now = torch.cuda.is_available()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_now else None
                except Exception:
                    gpu_now = False; gpu_name = None
                if gpu_now:
                    self._write_log("\nGPU active: {}\n".format(gpu_name), "success")
                    self.after(0, lambda: self._device_lbl.config(
                        text="GPU: {}".format(gpu_name), fg=SUCCESS))
                else:
                    self._write_log("\nInstalled! Restart to activate GPU.\n", "success")
                    self.after(0, lambda: messagebox.showinfo(
                        "Restart required",
                        "GPU support installed!\n\nPlease restart FACEBLUR."))
                    self.after(0, lambda: self._enable_gpu_btn.config(
                        text="RESTART APP", bg=ACCENT, fg=BG, cursor="hand2"))
                    self.after(0, lambda: self._enable_gpu_btn.bind(
                        "<Button-1>", lambda e: self._restart_app()))
            else:
                self._write_log("\nGPU install failed.\n", "error")
                self.after(0, lambda: self._enable_gpu_btn.config(
                    text="FAILED - retry", bg=ACCENT2, fg=BG, cursor="hand2"))
                self.after(0, lambda: self._enable_gpu_btn.bind(
                    "<Button-1>", lambda e: self._enable_gpu()))
        threading.Thread(target=_do, daemon=True).start()

    def _restart_app(self):
        import subprocess
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000
        subprocess.Popen([sys.executable] + sys.argv, **kwargs)
        self.destroy()

    # ── PREVIEW ───────────────────────────────────────────

    def _update_preview(self, event=None):
        sel = self._lb.curselection()
        if not sel:
            return
        path = self._files[sel[0]]

        def _load():
            try:
                cap = cv2.VideoCapture(path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 10))
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    return
                h, w = frame.shape[:2]
                scale = min(320/w, 120/h)
                nw, nh = int(w*scale), int(h*scale)
                frame = cv2.resize(frame, (nw, nh))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image, ImageTk
                photo = ImageTk.PhotoImage(Image.fromarray(frame))
                def _show():
                    if hasattr(self._thumb_lbl, "_photo"):
                        self._thumb_lbl._photo = None
                    self._thumb_lbl.config(image=photo, text="")
                    self._thumb_lbl._photo = photo
                self.after(0, _show)
            except Exception:
                self.after(0, lambda: self._thumb_lbl.config(
                    text="Preview unavailable", image=""))
        threading.Thread(target=_load, daemon=True).start()

    def _on_close(self):
        self._save_settings()
        self.destroy()

    # ── LOG ───────────────────────────────────────────────

    def _write_log(self, msg, tag=None):
        def _do():
            self._log_text.config(state="normal")
            self._log_text.insert("end", msg, tag or ())
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        self.after(0, _do)

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ── PROGRESS ──────────────────────────────────────────

    def _set_progress(self, pct, status=None, file_n=None, frame_n=None,
                      face_n=None, fps=None, eta=None):
        def _do():
            w = self._pb_frame.winfo_width()
            self._pb_fill.place(x=0, y=0,
                                width=int(w * max(0.0, min(1.0, pct))), height=6)
            if status  is not None: self._status_lbl.config(text=status)
            if file_n  is not None: self._lbl_file.config(text=str(file_n))
            if frame_n is not None: self._lbl_frame.config(text=str(frame_n))
            if face_n  is not None: self._lbl_faces.config(text=str(face_n))
            if fps     is not None: self._lbl_fps.config(text=str(fps))
            if eta     is not None: self._lbl_eta.config(text=str(eta))
        self.after(0, _do)

    def _set_running(self, running):
        def _do():
            if running:
                self._proc_btn.config(state="disabled", fg=TDIM, bg=BG3, cursor="arrow")
                self._cancel_btn.config(fg=BG, bg=ACCENT2, cursor="hand2")
            else:
                self._proc_btn.config(state="normal", fg=BG, bg=ACCENT, cursor="hand2")
                self._cancel_btn.config(fg=TDIM, bg=BG3, cursor="arrow")
        self.after(0, _do)

    # ── FILE LIST ─────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                       ("All files", "*.*")])
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                self._lb.insert("end", os.path.basename(p))
        n = len(self._files)
        self._count_lbl.config(text="{} file{}".format(n, "s" if n != 1 else ""))

    def _remove_sel(self):
        for i in reversed(list(self._lb.curselection())):
            self._lb.delete(i); del self._files[i]
        n = len(self._files)
        self._count_lbl.config(text="{} file{}".format(n, "s" if n != 1 else ""))

    def _clear_files(self):
        self._lb.delete(0, "end"); self._files.clear()
        self._count_lbl.config(text="0 files")

    def _pick_outdir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self._outdir.set(d)

    # ── PROCESSING ────────────────────────────────────────

    def _update_queue_status(self, path, status):
        icons  = {"waiting": "  [ ] ", "processing": "  [>>]",
                  "done": "  [OK]", "failed": "  [!!]"}
        colors = {"waiting": TDIM, "processing": ACCENT,
                  "done": SUCCESS, "failed": ACCENT2}
        self._file_status[path] = status
        def _do():
            self._lb.delete(0, "end")
            for p in self._files:
                s    = self._file_status.get(p, "waiting")
                name = os.path.basename(p)
                if len(name) > 28: name = name[:25] + "..."
                self._lb.insert("end", "{}{}".format(name, icons.get(s, "")))
                self._lb.itemconfig("end", fg=colors.get(s, TEXT))
        self.after(0, _do)

    def _preview_first_frame(self, path, cfg):
        def _do():
            try:
                detector = get_detector(cfg["model_key"], log_fn=None)
                head_model = None
                person_model = None
                head_is_user = False
                if cfg.get("detect_head"):
                    try:
                        head_model, person_model, head_is_user = get_head_detector(
                            head_size=cfg.get("head_size"), log_fn=None)
                    except Exception:
                        head_model = person_model = None
                        head_is_user = False
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read()
                cap.release()
                if not ok: return
                boxes = detect_objects(frame, detector, head_model, person_model,
                                       cfg["confidence"], cfg["detect_scale"],
                                       cfg["edge_strip"], cfg.get("detect_head", True),
                                       head_is_user=head_is_user,
                                       use_face=cfg.get("use_face", False),
                                       use_person_aid=cfg.get("use_person_aid", False))
                preview = apply_censor(frame.copy(), boxes,
                                       cfg["mode"], cfg["padding"],
                                       cfg["intensity"], cfg["block_size"],
                                       debug=True)
                h, w = preview.shape[:2]
                scale = min(320/w, 120/h)
                preview = cv2.resize(preview, (int(w*scale), int(h*scale)))
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                from PIL import Image, ImageTk
                photo = ImageTk.PhotoImage(Image.fromarray(preview))
                def _show():
                    if hasattr(self._thumb_lbl, "_photo"):
                        self._thumb_lbl._photo = None
                    self._thumb_lbl.config(image=photo, text="")
                    self._thumb_lbl._photo = photo
                self.after(0, _show)
                self._write_log("  Preview: {} box(es) on first frame\n".format(
                    len(boxes)), "dim")
                # Diagnostic: what does each whole-head pass find on its own?
                if cfg.get("detect_head"):
                    try:
                        hc = max(0.15, cfg["confidence"] - HEAD_CONF_DROP)
                        if person_model is not None and cfg.get("use_person_aid", False):
                            hr = detect_faces(frame, person_model, hc,
                                              1.0, edge_strip=False,
                                              classes=[HEAD_PERSON_CLASS])
                            hb = _person_to_head(hr)
                            self._write_log("  Person \u2192 head regions: {} "
                                            "on first frame\n".format(len(hb)),
                                            "dim" if hb else "warning")
                        elif person_model is not None:
                            self._write_log("  Person \u2192 head aid: OFF\n", "dim")
                        if head_model is not None:
                            hb2 = detect_faces(frame, head_model, hc, 1.0,
                                               cfg["edge_strip"],
                                               classes=_head_class_ids(head_model),
                                               imgsz=HEAD_INFER_IMGSZ)
                            self._write_log("  Head model alone: {} head(s) "
                                            "on first frame\n".format(len(hb2)),
                                            "dim" if hb2 else "warning")
                    except Exception:
                        pass
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _start(self):
        if not self._files:
            messagebox.showwarning("No files", "Add at least one video file first.")
            return
        self._cancel_flag.clear()
        self._clear_log()
        self._set_progress(0, status="Starting...")
        self._set_running(True)
        for p in self._files:
            self._file_status[p] = "waiting"
        cfg = {
            "files":         list(self._files),
            "mode":          self._mode.get(),
            "model_key":     self._model_key.get(),
            "confidence":    float(self._conf.get()),
            "padding":       float(self._pad.get()),
            "intensity":     int(self._blur_k.get()),
            "block_size":    int(self._pixel_sz.get()),
            "debug":         self._debug.get(),
            "outdir":        self._outdir.get().strip() or None,
            "skip_frames":   int(self._skip_frames.get()),
            "detect_scale":  float(self._detect_scale.get()),
            "suffix":        MODE_SUFFIX.get(self._mode.get(), "_blurred"),
            "edge_strip":    self._edge_strip.get(),
            "detect_head":   self._detect_head.get(),
            "head_size":     self._head_size.get(),
            "use_face":      self._use_face.get(),
            "use_person_aid": self._use_person_aid.get(),
            "smooth_boxes":  self._smooth_boxes.get(),
            "export_report": self._export_report.get(),
        }
        if self._files:
            self._preview_first_frame(self._files[0], cfg)
        self._update_queue_status(self._files[0], "waiting")
        threading.Thread(target=self._run, args=(cfg,), daemon=True).start()

    def _cancel(self):
        self._cancel_flag.set()
        self._write_log("Cancelling...\n", "warning")

    def _run(self, cfg):
        try:
            self._run_inner(cfg)
        except Exception:
            import traceback
            self._write_log("\n[EXCEPTION]\n{}\n".format(
                traceback.format_exc()), "error")
        finally:
            self._set_running(False)

    def _run_inner(self, cfg):
        files   = cfg["files"]
        n_files = len(files)
        self._write_log("Loading model: {}...\n".format(cfg["model_key"]), "accent")
        try:
            detector = get_detector(cfg["model_key"], log_fn=self._write_log)
        except Exception as e:
            self._write_log("[ERROR] {}\nRun: pip install ultralytics\n".format(e),
                            "error")
            return
        self._write_log("Model OK. Processing {} file(s)...\n".format(n_files), "accent")

        head_model = None
        person_model = None
        head_is_user = False
        if cfg.get("detect_head", True):
            self._write_log("Head detection ON (primary method). Loading head model...\n", "accent")
            try:
                head_model, person_model, head_is_user = get_head_detector(
                    head_size=cfg.get("head_size"), log_fn=self._write_log)
                methods = []
                person_on = person_model is not None and cfg.get("use_person_aid", False)
                if head_model is not None:
                    methods.append("head model (primary)")
                if person_on:
                    methods.append("person\u2192head rescue aid")
                if cfg.get("edge_strip"):
                    methods.append("edge strips")
                if cfg.get("use_face"):
                    methods.append("face safety net")
                if head_model is not None:
                    self._write_log("Head detection OK ({}).\n".format(
                        " + ".join(methods)), "accent")
                    try:
                        self._write_log("  Head classes: {}\n".format(
                            head_model.names), "dim")
                    except Exception:
                        pass
                    if person_model is not None and not person_on:
                        self._write_log("  Person\u2192head aid: OFF "
                                        "(enable it to rescue weak head "
                                        "detections).\n", "dim")
                else:
                    self._write_log("[ERROR] Head detection is ON but NO head "
                                    "model loaded.\n        The model downloads on "
                                    "first run; check internet access and the "
                                    "warnings above.\n", "error")
                    try:
                        self.after(0, lambda: messagebox.showwarning(
                            "Whole-head detection unavailable",
                            "Could not load the head/person detector, so this run "
                            "will censor faces only.\n\nThis usually means the "
                            "first-run model download was blocked. Check your "
                            "internet connection (or drop a head.pt next to the app) "
                            "and try again."))
                    except Exception:
                        pass
            except Exception as e:
                self._write_log("[WARN] Head model failed ({}). "
                                "Continuing face-only.\n".format(e), "warning")
                head_model = person_model = None

        report_data = []
        for fi, path in enumerate(files, 1):
            if self._cancel_flag.is_set():
                break
            self._update_queue_status(path, "processing")
            fname        = os.path.basename(path)
            outpath      = make_output_path(path, cfg["outdir"], cfg["suffix"])
            t_file_start = time.time()

            self._write_log("\n[{}/{}] {}\n".format(fi, n_files, fname), "accent")
            self._write_log("  -> {}\n".format(outpath), "dim")

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self._write_log("  [ERROR] Cannot open file\n", "error")
                self._update_queue_status(path, "failed")
                continue

            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._write_log("  {}x{}  {:.1f}fps  {} frames\n".format(
                W, H, fps, total if total > 0 else "?"), "dim")

            if W == 0 or H == 0:
                self._write_log("  [ERROR] Bad dimensions\n", "error")
                cap.release(); continue

            # --- Output: single-pass H.264 encode + inline audio mux. No temp
            #     file, no double encode. Falls back to the legacy cv2 writer +
            #     second-pass merge ONLY if ffmpeg can't start the pipe. ---
            ffmpeg_path = get_ffmpeg_path()
            using_pipe  = False
            proc        = None
            writer      = None
            tmp_path    = None
            stderr_log  = outpath + ".ffmpeg.log"
            stderr_fp   = None
            if ffmpeg_path:
                try:
                    stderr_fp = open(stderr_log, "wb")
                    proc = open_ffmpeg_pipe(ffmpeg_path, path, outpath, W, H, fps,
                                            stderr_fp, log_fn=self._write_log)
                except Exception:
                    proc = None
                if proc is not None:
                    using_pipe = True
                    self._write_log("  Encoder: H.264 single-pass (audio muxed inline)\n", "dim")
                else:
                    if stderr_fp is not None:
                        try: stderr_fp.close()
                        except Exception: pass
                        stderr_fp = None
            if not using_pipe:
                # Legacy fallback: temp video now, audio merged in a 2nd pass.
                tmp_path = outpath + ".tmp_raw.avi"
                writer   = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (W, H))
                if not writer.isOpened():
                    tmp_path = outpath + ".tmp_raw.mp4"
                    writer   = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
                if not writer.isOpened():
                    self._write_log("  [ERROR] Cannot create output\n", "error")
                    cap.release(); continue
                self._write_log("  Encoder: {} temp + ffmpeg merge (ffmpeg pipe "
                                "unavailable)\n".format(
                                    "XVID" if tmp_path.endswith(".avi") else "mp4v"),
                                "warning")

            frame_n     = 0
            faces_n     = 0
            skip        = max(1, cfg["skip_frames"])
            dscale      = cfg["detect_scale"]
            t_start     = time.time()
            # The CSRT tracker only does anything BETWEEN detections, i.e. when
            # frame-skipping. At skip=1 every frame is detected, so the tracker's
            # interpolation is never used -- don't pay for it, and don't risk it
            # holding a stale box. (This is separate from the Smooth-boxes option.)
            use_tracker = skip > 1
            tracker     = FaceTracker() if use_tracker else None
            boxes       = []
            smoother    = BoxSmoother() if cfg.get("smooth_boxes", True) else None
            src_totals  = {"face": 0, "person": 0, "head": 0}
            logged_first_breakdown = False
            # Make the active temporal processing visible, so "is smoothing on?"
            # is never a guess. With both OFF, boxes are raw per-frame detections.
            self._write_log("  Temporal: smoothing {} | between-frame tracking {}\n".format(
                "ON" if smoother is not None else "OFF",
                "ON (skip={})".format(skip) if use_tracker else "OFF (every frame)"),
                "dim")

            while not self._cancel_flag.is_set():
                ok, frame = cap.read()
                if not ok: break

                if frame_n % skip == 0:
                    boxes = detect_objects(frame, detector, head_model, person_model,
                                           cfg["confidence"], dscale,
                                           cfg["edge_strip"],
                                           cfg.get("detect_head", True),
                                           head_is_user=head_is_user,
                                           use_face=cfg.get("use_face", False),
                                           use_person_aid=cfg.get("use_person_aid", False))
                    if use_tracker:
                        try: tracker.update_from_detection(frame, boxes)
                        except Exception: use_tracker = False
                    current_boxes = boxes
                else:
                    if use_tracker:
                        try: current_boxes = tracker.get_tracked_boxes(frame)
                        except Exception:
                            use_tracker = False; current_boxes = boxes
                    else:
                        current_boxes = boxes

                # Per-source bookkeeping (which pass is actually finding heads?)
                if frame_n % skip == 0:
                    for k in src_totals:
                        src_totals[k] += len(LAST_DETECT_BREAKDOWN.get(k, []))
                    if not logged_first_breakdown:
                        logged_first_breakdown = True
                        if cfg.get("detect_head"):
                            self._write_log(
                                "  Frame 0 by source: face={} person->head={} "
                                "head.pt={}\n".format(
                                    len(LAST_DETECT_BREAKDOWN.get("face", [])),
                                    len(LAST_DETECT_BREAKDOWN.get("person", [])),
                                    len(LAST_DETECT_BREAKDOWN.get("head", []))), "dim")

                # Temporal smoothing: stabilize boxes + hold through missed
                # detections so the censor doesn't flicker.
                if smoother is not None:
                    current_boxes = smoother.update(current_boxes, frame)

                frame = apply_censor(frame, current_boxes,
                                     cfg["mode"], cfg["padding"],
                                     cfg["intensity"], cfg["block_size"],
                                     cfg["debug"])
                if cfg["debug"] and cfg.get("detect_head") and frame_n % skip == 0:
                    frame = draw_source_debug(frame, LAST_DETECT_BREAKDOWN)
                if using_pipe:
                    if frame.shape[1] != W or frame.shape[0] != H:
                        frame = cv2.resize(frame, (W, H))
                    try:
                        proc.stdin.write(frame.tobytes())
                    except (BrokenPipeError, OSError):
                        self._write_log("  [ERROR] encoder closed early "
                                        "(see encode error below).\n", "error")
                        break
                else:
                    writer.write(frame)
                frame_n += 1
                faces_n += len(current_boxes)

                if frame_n % 30 == 0:
                    pct      = (frame_n / total) if total > 0 else 0.5
                    elapsed  = time.time() - t_start
                    fps_proc = frame_n / elapsed if elapsed > 0 else 0
                    if total > 0 and fps_proc > 0:
                        rem = int((total - frame_n) / fps_proc)
                        h, m, s = rem // 3600, (rem % 3600) // 60, rem % 60
                        if h > 0:   eta_str = "{}h {:02d}m {:02d}s".format(h, m, s)
                        elif m > 0: eta_str = "{}m {:02d}s".format(m, s)
                        else:       eta_str = "{}s".format(s)
                    else:
                        eta_str = "-"
                    # Cap at 0.99 during processing - only hit 1.0 when truly done
                    safe_pct = min(0.99, pct)
                    self._set_progress(
                        safe_pct,
                        status="Processing {} - frame {}{}".format(
                            fname, frame_n,
                            "/{}".format(total) if total > 0 else ""),
                        file_n="{}/{}".format(fi, n_files),
                        frame_n=frame_n, face_n=faces_n,
                        fps="{:.1f}".format(fps_proc), eta=eta_str)

            cap.release()

            # Finalize the encoder. For the single-pass pipe, closing stdin makes
            # ffmpeg flush and write the moov atom (usually well under a second);
            # the audio is already muxed. Show a real "Finalizing" state with the
            # ETA cleared so it never freezes on a stale number.
            if not self._cancel_flag.is_set():
                self._set_progress(
                    min(0.99, (frame_n / total) if total > 0 else 0.99),
                    status="Finalizing {} (encoding/audio)...".format(fname),
                    file_n="{}/{}".format(fi, n_files),
                    frame_n=frame_n, face_n=faces_n, eta="0s")

            enc_err   = ""
            ok_output = False
            if using_pipe:
                try:
                    if proc.stdin: proc.stdin.close()
                except Exception: pass
                rc = proc.wait()
                if stderr_fp is not None:
                    try: stderr_fp.close()
                    except Exception: pass
                try:
                    with open(stderr_log, "r", errors="replace") as _f:
                        enc_err = _f.read().strip()
                except Exception:
                    enc_err = ""
                try: os.remove(stderr_log)
                except Exception: pass
                ok_output = (rc == 0 and os.path.exists(outpath)
                             and os.path.getsize(outpath) > 0)
            else:
                if writer is not None:
                    writer.release()

            if cfg.get("detect_head"):
                self._write_log("  Detections by source: face={} "
                                "person->head={} head.pt={}\n".format(
                                    src_totals["face"], src_totals["person"],
                                    src_totals["head"]), "dim")
                if src_totals["person"] == 0 and src_totals["head"] == 0:
                    self._write_log("  [WARN] Whole-head passes found NOTHING in "
                                    "this file.\n         Check: confidence too high? "
                                    "person model loaded? head.pt present?\n", "warning")
            del tracker; frame = None; current_boxes = None
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception: pass

            if self._cancel_flag.is_set():
                self._write_log("  Cancelled. Partial file kept.\n", "warning")
                self._update_queue_status(path, "waiting")
            elif frame_n == 0:
                self._write_log("  [ERROR] 0 frames read!\n", "error")
                self._update_queue_status(path, "failed")
                for p in (outpath if using_pipe else tmp_path,):
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass
            else:
                if using_pipe:
                    if not ok_output:
                        self._write_log("  [ERROR] encode failed: {}\n".format(
                            (enc_err[-200:] or "unknown error")), "error")
                else:
                    # Legacy two-pass fallback (only when the ffmpeg pipe couldn't start)
                    if not tmp_path or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                        self._write_log("  [ERROR] Output empty\n", "error")
                    else:
                        ffmpeg = get_ffmpeg_path()
                        if ffmpeg:
                            merged = merge_audio(path, tmp_path, outpath, ffmpeg,
                                                 log_fn=self._write_log)
                            if merged and os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            elif not merged:
                                self._write_log("  Falling back to muted output.\n", "warning")
                                if os.path.exists(tmp_path):
                                    if os.path.exists(outpath): os.remove(outpath)
                                    os.rename(tmp_path, outpath)
                        else:
                            self._write_log("  [WARN] ffmpeg not found - no audio.\n", "warning")
                            if os.path.exists(outpath): os.remove(outpath)
                            os.rename(tmp_path, outpath)

                if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
                    mb        = os.path.getsize(outpath) / 1048576
                    t_elapsed = time.time() - t_file_start
                    self._write_log(
                        "  Done. {} frames, {} faces, {:.1f} MB, {:.1f}s\n".format(
                            frame_n, faces_n, mb, t_elapsed), "success")
                    self._set_progress(1.0, status="Done: {}".format(fname),
                                       file_n="{}/{}".format(fi, n_files),
                                       frame_n=frame_n, face_n=faces_n, eta="0s")
                    self._update_queue_status(path, "done")
                    report_data.append({
                        "file": fname, "output": outpath,
                        "frames": frame_n, "faces_total": faces_n,
                        "duration_s": round(t_elapsed, 2),
                        "fps": round(frame_n/t_elapsed, 1) if t_elapsed > 0 else 0,
                        "settings": {k: v for k, v in cfg.items() if k != "files"},
                    })
                else:
                    self._update_queue_status(path, "failed")

        clear_detector_cache()
        if not self._cancel_flag.is_set():
            self._write_log("\nAll files finished.\n", "success")
            self._toast("FACEBLUR",
                        "Processing complete! {} file(s) done.".format(n_files))
            if cfg.get("export_report") and report_data:
                outdir = cfg.get("outdir") or os.path.dirname(
                    os.path.abspath(files[0]))
                rpath = os.path.join(outdir, "faceblur_report.json")
                save_report(rpath, report_data)
                self._write_log("Report saved: {}\n".format(rpath), "dim")
        self._set_progress(
            0 if self._cancel_flag.is_set() else 1.0,
            status="Cancelled." if self._cancel_flag.is_set() else "All done!",
            eta="-" if self._cancel_flag.is_set() else "0s")


# ══════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ══════════════════════════════════════════════════════════

class SplashScreen(tk.Tk):
    def __init__(self):
        super().__init__()
        # Thread that owns the Tk interpreter. All widget calls must happen here;
        # set_status() uses this to marshal updates from the background loader.
        self._tk_thread = threading.get_ident()
        self.overrideredirect(True)
        self.configure(bg="#0f0f0f")
        self.attributes("-topmost", True)
        W, H = 420, 220
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("{}x{}+{}+{}".format(W, H, (sw-W)//2, (sh-H)//2))
        border = tk.Frame(self, bg="#00e5ff", padx=1, pady=1)
        border.place(x=0, y=0, width=W, height=H)
        inner = tk.Frame(border, bg="#0f0f0f")
        inner.pack(fill="both", expand=True)
        tk.Label(inner, text="FACEBLUR", bg="#0f0f0f", fg="#00e5ff",
                 font=("Courier New", 32, "bold")).pack(pady=(30, 4))
        tk.Label(inner, text="YOLOv11 head & face censoring  v1.4  |  made by werehappy",
                 bg="#0f0f0f", fg="#444444",
                 font=("Courier New", 9)).pack()
        self._status = tk.Label(inner, text="Starting...",
                                bg="#0f0f0f", fg="#666666",
                                font=("Courier New", 9))
        self._status.pack(pady=(20, 4))
        pb_outer = tk.Frame(inner, bg="#1a1a1a", height=3)
        pb_outer.pack(fill="x", padx=40)
        pb_outer.pack_propagate(False)
        self._pb = tk.Frame(pb_outer, bg="#00e5ff", height=3)
        self._pb.place(x=0, y=0, width=0, height=3)
        self._pb_width = 340
        # Busy-animation state (used for the long first-run torch download).
        self._indeterminate = False
        self._anim_job = None
        self._indet_base = ""
        self._indet_start = 0.0

        # The splash is borderless (no X) and topmost. If loading ever fails,
        # it must still be closeable instead of floating forever.
        self._closed = False
        self._err_btn = None
        self.bind("<Escape>", lambda e: self.close())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._close_x = tk.Label(inner, text="\u2715", bg="#0f0f0f",
                                 fg="#444444", font=("Courier New", 11, "bold"),
                                 cursor="hand2")
        self._close_x.place(relx=1.0, x=-12, y=8, anchor="ne")
        self._close_x.bind("<Button-1>", lambda e: self.close())

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def show_error(self, msg):
        """Convert the borderless splash into a normal, closeable window so the
        user can read the failure and dismiss it (no more floating ghost)."""
        try:
            self.overrideredirect(False)
            self.attributes("-topmost", False)
            self.title("FACEBLUR - startup error")
        except Exception:
            pass
        try:
            self._status.config(text=msg, fg="#ff6b8a")
        except Exception:
            pass
        if not self._err_btn:
            try:
                self._err_btn = tk.Label(self, text="CLOSE", bg="#ff6b8a",
                                         fg="#0f0f0f", cursor="hand2",
                                         font=("Courier New", 10, "bold"),
                                         padx=16, pady=6)
                self._err_btn.bind("<Button-1>", lambda e: self.close())
                self._err_btn.place(relx=0.5, rely=1.0, y=-14, anchor="s")
            except Exception:
                pass
        try:
            self.update()
        except Exception:
            pass

    def report_callback_exception(self, exc, val, tb):
        # Tk swallows exceptions raised inside callbacks (e.g. our after()-
        # scheduled launch). Capture them and surface a closeable error window.
        import traceback
        _append_crash("".join(traceback.format_exception(exc, val, tb)))
        self.show_error("Startup error - see faceblur_crash.txt")

    def _apply_status(self, msg, pct):
        if getattr(self, "_closed", False):
            return
        if self._indeterminate:
            # The marquee animation owns the bar and the status text; just record
            # the latest message so the animator shows it with the elapsed clock.
            self._indet_base = msg
            return
        try:
            self._status.config(text=msg)
            self._pb.place(x=0, y=0,
                           width=int(self._pb_width * max(0.0, min(1.0, pct))),
                           height=3)
        except Exception:
            pass

    def start_indeterminate(self, base_msg="Working"):
        """Busy animation + elapsed clock for long, percentage-less work (the
        first-run torch download, which can be a 2.5GB CUDA wheel). A moving bar
        and a ticking timer prove the app is alive, not frozen. Any thread."""
        if getattr(self, "_closed", False):
            return
        if threading.get_ident() != self._tk_thread:
            try:
                self.after(0, self.start_indeterminate, base_msg)
            except Exception:
                pass
            return
        if self._indeterminate:
            self._indet_base = base_msg
            return
        self._indeterminate = True
        self._indet_base = base_msg
        self._indet_start = time.time()
        self._marquee_x = 0
        self._marquee_dir = 1
        self._animate_marquee()

    def _animate_marquee(self):
        if getattr(self, "_closed", False) or not self._indeterminate:
            return
        seg = 70
        track = self._pb_width
        self._marquee_x += self._marquee_dir * 14
        if self._marquee_x <= 0:
            self._marquee_x, self._marquee_dir = 0, 1
        elif self._marquee_x >= track - seg:
            self._marquee_x, self._marquee_dir = track - seg, -1
        try:
            self._pb.place(x=self._marquee_x, y=0, width=seg, height=3)
            elapsed = int(time.time() - self._indet_start)
            self._status.config(text="{}  ({}:{:02d})".format(
                self._indet_base, elapsed // 60, elapsed % 60))
        except Exception:
            pass
        try:
            self._anim_job = self.after(45, self._animate_marquee)
        except Exception:
            pass

    def stop_indeterminate(self):
        """Stop the busy animation and hand the bar back to set_status. Any thread."""
        if threading.get_ident() != self._tk_thread:
            try:
                self.after(0, self.stop_indeterminate)
            except Exception:
                pass
            return
        self._indeterminate = False
        job = getattr(self, "_anim_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
            self._anim_job = None

    def set_status(self, msg, pct):
        if getattr(self, "_closed", False):
            return
        # Tkinter is single-threaded: widgets may only be touched from the
        # thread that owns the interpreter (the one running mainloop). The
        # background loader calls this, so when we're off that thread we MUST
        # marshal the update via after() instead of poking widgets and calling
        # update() directly. Cross-thread update() races mainloop and is what
        # made the first launch intermittently hang / need several reopens.
        if threading.get_ident() == self._tk_thread:
            self._apply_status(msg, pct)
            try:
                self.update_idletasks()
            except Exception:
                pass
        else:
            try:
                self.after(0, self._apply_status, msg, pct)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

def _get_venv_path():
    """Get path to the faceblur venv next to the exe."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "faceblur_env")

def _add_venv_to_path():
    """Add the faceblur venv site-packages to sys.path so torch is importable."""
    venv = _get_venv_path()
    _log = ["Venv path: {}".format(venv),
            "Venv exists: {}".format(os.path.exists(venv))]

    if not os.path.exists(venv):
        _write_debug(_log)
        return False

    # Walk the entire venv to find site-packages regardless of structure
    found = False
    for root, dirs, files in os.walk(venv):
        if os.path.basename(root) == "site-packages":
            if root not in sys.path:
                sys.path.insert(0, root)
                _log.append("Added: {}".format(root))
            else:
                _log.append("Already on path: {}".format(root))
            found = True   # present on path counts as found

    # Also add Scripts/bin for DLLs
    for scripts_dir in ["Scripts", "bin"]:
        sp = os.path.join(venv, scripts_dir)
        if os.path.exists(sp) and sp not in sys.path:
            sys.path.insert(0, sp)
            _log.append("Added scripts: {}".format(sp))

    # The exe only bundles the stdlib modules OUR code uses. torch pulls in
    # extra stdlib modules (e.g. pickletools) that the frozen exe lacks, which
    # makes "import torch" fail. The bundled embeddable Python here ships the
    # FULL stdlib (in python3XX.zip), so add it as a fallback. Appended, not
    # inserted, so the exe's own (version-matched) stdlib always wins and the
    # zip only fills genuine gaps.
    import glob
    for z in sorted(glob.glob(os.path.join(venv, "python3*.zip"))):
        if z not in sys.path:
            sys.path.append(z)
            _log.append("Added stdlib zip: {}".format(z))
    for extra in (venv, os.path.join(venv, "DLLs"), os.path.join(venv, "Lib")):
        if os.path.isdir(extra) and extra not in sys.path:
            sys.path.append(extra)
            _log.append("Added stdlib dir: {}".format(extra))

    _log.append("Result: {}".format("found" if found else "site-packages not found"))
    _write_debug(_log)
    return found

def _write_debug(lines):
    try:
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
               else os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "faceblur_debug.txt"), "w") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

def _crash_log_path():
    try:
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
               else os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, "faceblur_crash.txt")
    except Exception:
        return "faceblur_crash.txt"

def _append_crash(text):
    try:
        import datetime
        with open(_crash_log_path(), "a") as f:
            f.write("\n=== {} ===\n{}\n".format(
                datetime.datetime.now().isoformat(), text))
    except Exception:
        pass

def _torch_is_installed():
    """Check if torch is available, including from venv."""
    _add_venv_to_path()
    try:
        import torch
        return True
    except Exception:
        # Log the reason: if this keeps failing in the frozen exe, the app
        # reinstalls torch on every launch (slow boot). The trace tells us why.
        import traceback
        _append_crash("torch import check failed (would trigger reinstall):\n"
                      + traceback.format_exc())
        return False

def _install_torch_first_run(splash):
    """
    First launch only: torch is not bundled in the exe (keeps the download
    small). We detect the GPU, then use the bundled embeddable Python in
    faceblur_env to pip-install the matching torch build into
    faceblur_env\\Lib\\site-packages. After this, _add_venv_to_path() makes
    it importable for every future launch, so this runs exactly once.
    Requires an internet connection on first run.
    """
    import time

    _log = []
    def _status(msg, pct):
        _log.append(msg)
        try:
            splash.set_status(msg, pct)
        except Exception:
            pass

    # Make sure the bundled python is reachable before we try to use it.
    python_exe = get_python_executable()
    if python_exe is None:
        _status("Setup error: bundled Python not found.", 0.5)
        _write_debug(["_install_torch_first_run: get_python_executable() returned None",
                      "venv path: {}".format(_get_venv_path())])
        time.sleep(4)
        return detect_gpu()

    _status("First run: detecting GPU...", 0.2)
    gpu = detect_gpu()
    cuda_ver = gpu.get("cuda_ver")
    if gpu.get("has_nvidia") and cuda_ver:
        _status("Downloading GPU libraries (one-time, a few minutes)...", 0.35)
    else:
        _status("Downloading libraries (one-time, a few minutes)...", 0.35)

    # install_torch_for_cuda() uses get_python_executable() internally and
    # streams pip output through this callback.
    def _pip_log(line, tag=None):
        _log.append(line.rstrip("\n"))
        # Keep the splash bar gently moving while pip works.
        _status("Installing libraries...", 0.55)

    ok = False
    # Show a moving bar + elapsed clock across the (long, percentage-less) pip
    # download/install so the splash is visibly ALIVE. On a machine with an
    # Nvidia GPU this pulls a ~2.5GB CUDA build, which legitimately takes many
    # minutes -- without this it looks frozen at "Installing libraries...".
    if gpu.get("has_nvidia") and cuda_ver:
        _base = "Downloading GPU libraries (one-time, ~2.5GB)"
    else:
        _base = "Installing libraries (one-time)"
    try:
        splash.start_indeterminate(_base)
    except Exception:
        pass
    try:
        ok = install_torch_for_cuda(cuda_ver, log_fn=_pip_log,
                                    has_nvidia=gpu.get("has_nvidia", False))
    except Exception as e:
        _log.append("install_torch_for_cuda raised: {}".format(e))
        ok = False
    finally:
        try:
            splash.stop_indeterminate()
        except Exception:
            pass

    _write_debug(_log)

    if not ok:
        _status("Could not install libraries. Check your internet "
                "connection and relaunch.", 0.6)
        time.sleep(5)
        # Return whatever we know; the app will report torch missing.
        return detect_gpu()

    # Make the freshly installed packages importable in THIS process.
    # The site-packages dir was already on sys.path (added empty at startup),
    # so the path-finder cache is stale - invalidate it before importing torch.
    _status("Finalizing...", 0.8)
    _add_venv_to_path()
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass

    # Re-detect now that torch is importable (picks up torch.cuda state).
    return detect_gpu()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # ---- crash diagnostics -------------------------------------------------
    # The console can vanish on a native crash (e.g. a numpy/torch ABI clash),
    # leaving no trace. Dump native faults to a file next to the exe.
    try:
        import faulthandler
        _crash_fp = open(_crash_log_path(), "w")
        faulthandler.enable(file=_crash_fp)
    except Exception:
        _crash_fp = None

    # Inject venv site-packages so torch installed by installer is found
    _add_venv_to_path()

    # cv2/numpy are bundled in the exe. If a build ever ships without them,
    # log the real error and show a readable message instead of the bare
    # PyInstaller "Failed to execute script" dialog with an empty crash file.
    try:
        import cv2 as _cv2
        import numpy as _np
    except Exception as _imp_err:
        import traceback
        _append_crash("Top-level import failed:\n" + traceback.format_exc())
        try:
            import tkinter as _tk
            from tkinter import messagebox as _mb
            _r = _tk.Tk(); _r.withdraw()
            _mb.showerror(
                "FACEBLUR - missing components",
                "Failed to load required components ({}).\n\n"
                "This usually means the build did not bundle OpenCV/NumPy.\n"
                "Details written to faceblur_crash.txt next to the app."
                .format(type(_imp_err).__name__))
            _r.destroy()
        except Exception:
            pass
        sys.exit(1)
    globals()["cv2"] = _cv2
    globals()["np"]  = _np

    splash = SplashScreen()
    splash.update_idletasks()
    splash.update()

    _loaded = {}

    def _fail(msg, exc=None):
        # Marshal an error onto the splash from any thread and make it
        # closeable. Always runs on the Tk thread via after().
        if exc is not None:
            import traceback
            _append_crash("".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__)))
        try:
            splash.after(0, lambda: splash.show_error(msg))
        except Exception:
            pass

    def _background_load():
        try:
            # First run: torch not installed yet — install it now
            if not _torch_is_installed():
                splash.set_status("First run setup...", 0.1)
                _loaded["gpu_info"] = _install_torch_first_run(splash)
            else:
                splash.set_status("Detecting GPU...", 0.5)
                _loaded["gpu_info"] = detect_gpu()

            splash.set_status("Ready.", 0.9)
            splash.after(0, _launch_app)
        except Exception as e:
            # Crash during load: don't leave the splash floating forever.
            _fail("Startup error - see faceblur_crash.txt", e)

    def _launch_app():
        try:
            splash.set_status("Building interface...", 1.0)
            splash.update()
            app = App(gpu_info=_loaded.get("gpu_info"))
            # The splash was Tk's default root. Hand that role to the app before
            # destroying the splash, so any parent-less dialogs (messageboxes)
            # target the live app rather than the destroyed splash interpreter.
            try:
                tk._default_root = app
            except Exception:
                pass
            splash.destroy()
            app.mainloop()
        except Exception as e:
            # Show a closeable error window instead of re-raising (a re-raise
            # here is swallowed by Tk and the splash would float forever).
            _fail("Startup error - see faceblur_crash.txt", e)

    threading.Thread(target=_background_load, daemon=True).start()
    splash.mainloop()
