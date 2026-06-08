"""
face_blur.py — Face Detection & Censoring GUI Application
Uses YOLOv8-face + OpenCV + Tkinter
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
VERSION = "1.0"

# Settings persistence
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".faceblur_settings.json")

SETTINGS_VERSION = "1.0"

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
        # Try common locations relative to the exe
        candidates = [
            os.path.join(exe_dir, "python.exe"),
            os.path.join(exe_dir, "_internal", "python.exe"),
        ]
        # Also check CONDA_PREFIX if set
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

def install_torch_for_cuda(cuda_ver, log_fn=None):
    """Install correct torch build using the real Python executable."""
    import subprocess, sys

    ver = float(cuda_ver) if cuda_ver else 0
    if ver >= 12.1:
        index = "https://download.pytorch.org/whl/cu121"
        label = "CUDA 12.1+"
    elif ver >= 11.8:
        index = "https://download.pytorch.org/whl/cu118"
        label = "CUDA 11.8"
    else:
        index = "https://download.pytorch.org/whl/cpu"
        label = "CPU (CUDA too old)"

    if log_fn:
        log_fn("Installing torch for {}...\n".format(label), "accent")

    python_exe = get_python_executable()
    if python_exe is None:
        if log_fn:
            log_fn("[ERROR] Could not find Python executable to run pip.\n", "error")
        return False

    if log_fn:
        log_fn("  Using Python: {}\n".format(python_exe), "dim")

    cmd = [python_exe, "-m", "pip", "install",
           "torch", "torchvision",
           "--index-url", index, "--upgrade"]

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

def get_detector(model_key, log_fn=None):
    filename, url = YOLO_MODELS[model_key]
    if filename not in _detector_cache:
        from ultralytics import YOLO
        if not os.path.exists(filename):
            if log_fn:
                log_fn("  Downloading {} (first time only)...\n".format(filename), "warning")
            import urllib.request
            urllib.request.urlretrieve(url, filename)
            if log_fn:
                log_fn("  Download complete.\n", "success")
        model = YOLO(filename)
        # Enable FP16 (half precision) on GPU for ~2x faster inference
        try:
            import torch
            if torch.cuda.is_available():
                model.model.half()
        except Exception:
            pass
        _detector_cache[filename] = model
    return _detector_cache[filename]

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
        for (x1, y1, x2, y2) in boxes:
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

def _run_yolo(detector, frame, confidence):
    """Run YOLO on a single frame and return list of (x1,y1,x2,y2,score)."""
    results = detector(frame, conf=confidence, verbose=False, workers=0)
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

def detect_faces(frame, detector, confidence, detect_scale=0.5, edge_strip=True):
    """
    Run YOLO on downscaled frame + edge strips for partial/out-of-frame faces.
    Returns list of (x1, y1, x2, y2, score).
    """
    h, w = frame.shape[:2]

    # Main detection pass (downscaled)
    if detect_scale < 1.0:
        sw = max(32, int(w * detect_scale))
        sh = max(32, int(h * detect_scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
        raw = _run_yolo(detector, small, confidence)
        # Scale coords back up
        boxes = [(int(x1/detect_scale), int(y1/detect_scale),
                  int(x2/detect_scale), int(y2/detect_scale), s)
                 for (x1, y1, x2, y2, s) in raw]
    else:
        boxes = _run_yolo(detector, frame, confidence)

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
            raw = _run_yolo(detector, strip, max(0.1, confidence - 0.1))
            for (x1, y1, x2, y2, s) in raw:
                # Translate back to full frame coords
                boxes.append((x1+sx1, y1+sy1, x2+sx1, y2+sy1, s))

    return _dedup_boxes(boxes)

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

def merge_audio(original_path, muted_path, output_path, ffmpeg_path, log_fn=None):
    """
    Use ffmpeg to copy audio from original into the processed video.
    Returns True on success, False on failure.
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
        self.title("FACEBLUR v1.0")
        self.configure(bg=BG)
        self.geometry("960x780")
        self.minsize(900, 700)
        self._files        = []
        self._cancel_flag  = threading.Event()
        self._outdir       = tk.StringVar()
        self._mode         = tk.StringVar(value="Blur")
        self._model_key    = tk.StringVar(value=list(YOLO_MODELS.keys())[0])
        self._conf         = tk.DoubleVar(value=0.40)
        self._pad          = tk.DoubleVar(value=0.25)
        self._blur_k       = tk.IntVar(value=51)
        self._pixel_sz     = tk.IntVar(value=15)
        self._debug        = tk.BooleanVar(value=False)
        self._skip_frames  = tk.IntVar(value=2)
        self._detect_scale = tk.DoubleVar(value=0.50)
        self._gpu_info     = gpu_info if gpu_info is not None else detect_gpu()
        self._suffix       = tk.StringVar(value="_blurred")
        self._edge_strip   = tk.BooleanVar(value=True)
        self._export_report = tk.BooleanVar(value=False)
        self._file_status  = {}
        self._build()

    # ── BUILD ─────────────────────────────────────────────

    def _build(self):
        # ── TOP BAR ──
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(top, text="FACEBLUR", bg=BG, fg=ACCENT, font=FH).pack(side="left")
        tk.Label(top, text="YOLOv8 face censoring", bg=BG, fg=TDIM, font=FS).pack(side="left", padx=12)
        tk.Label(top, text="v1.0", bg=BG, fg=TDIM, font=("Courier New", 8)).pack(side="left")
        tk.Label(top, text="made by werehappy", bg=BG, fg=TDIM, font=("Courier New", 8)).pack(side="left", padx=4)
        # GPU/CPU indicator (right side of top bar)
        if self._gpu_info["torch_cuda"]:
            device_text  = "GPU: " + self._gpu_info["gpu_name"]
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
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(10, 0))

        # GPU banner removed - installer handles GPU/CPU version selection
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

        # model
        self._section(right, "YOLO MODEL")
        mkr = tk.Frame(right, bg=BG2)
        mkr.pack(fill="x", padx=8, pady=(0, 4))
        for k in YOLO_MODELS.keys():
            btn = tk.Label(mkr, text=k, font=FL, padx=10, pady=4,
                           cursor="hand2", relief="flat")
            btn.pack(side="left", padx=(0, 2))
            btn.bind("<Button-1>", lambda e, v=k: (
                self._model_key.set(v), self._refresh_model_btns()))
            btn.bind("<Enter>", lambda e, b=btn: b.config(
                bg="#505050" if b["bg"] != ACCENT else "#33ecff",
                fg="#0f0f0f"))
            btn.bind("<Leave>", lambda e, b=btn: self._refresh_model_btns())
        self._model_btns = mkr.winfo_children()
        self._refresh_model_btns()
        tk.Label(right, text="Downloaded automatically on first use (~6-25 MB)",
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
        self._write_log("FACEBLUR ready.  (YOLOv8-face)\n", "accent")
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
        self._edge_strip.set(True)
        self._export_report.set(False)
        self._mode.set("Blur")
        self._model_key.set(list(YOLO_MODELS.keys())[0])
        self._refresh_mode_btns()
        self._refresh_model_btns()
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
        cur = self._model_key.get()
        for b in self._model_btns:
            active = (b["text"] == cur)
            b.config(bg=ACCENT if active else "#3a3a3a",
                     fg="#0f0f0f" if active else "#cccccc")
            def _enter(e, btn=b):
                is_active = btn["text"] == self._model_key.get()
                btn.config(bg="#33ecff" if is_active else "#505050",
                           fg="#0f0f0f")
            def _leave(e, btn=b):
                is_active = btn["text"] == self._model_key.get()
                btn.config(bg=ACCENT if is_active else "#3a3a3a",
                           fg="#0f0f0f" if is_active else "#cccccc")
            b.bind("<Enter>", _enter)
            b.bind("<Leave>", _leave)

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
        }
        save_settings(data)

    def _load_settings(self):
        data = load_settings()
        if not data:
            return
        try:
            keys = ["mode", "model_key", "confidence", "padding", "blur_k",
                    "pixel_sz", "debug", "skip_frames", "detect_scale",
                    "outdir", "suffix", "edge_strip", "export_report"]
            targets = [self._mode, self._model_key, self._conf, self._pad,
                       self._blur_k, self._pixel_sz, self._debug,
                       self._skip_frames, self._detect_scale, self._outdir,
                       self._suffix, self._edge_strip, self._export_report]
            for k, t in zip(keys, targets):
                if k in data:
                    t.set(data[k])
            self._refresh_mode_btns()
            self._refresh_model_btns()
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
            success = install_torch_for_cuda(gi["cuda_ver"], log_fn=self._write_log)
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
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read()
                cap.release()
                if not ok: return
                boxes = detect_faces(frame, detector, cfg["confidence"],
                                     cfg["detect_scale"], cfg["edge_strip"])
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
                self._write_log("  Preview: {} face(s) on first frame\n".format(
                    len(boxes)), "dim")
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

            tmp_path = outpath + ".tmp_raw.avi"
            fourcc   = cv2.VideoWriter_fourcc(*"XVID")
            writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                tmp_path = outpath + ".tmp_raw.mp4"
                fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
            if not writer.isOpened():
                self._write_log("  [ERROR] Cannot create output\n", "error")
                cap.release(); continue
            self._write_log("  Codec: {}\n".format(
                "XVID" if tmp_path.endswith(".avi") else "mp4v"), "dim")

            frame_n     = 0
            faces_n     = 0
            skip        = max(1, cfg["skip_frames"])
            dscale      = cfg["detect_scale"]
            t_start     = time.time()
            tracker     = FaceTracker()
            use_tracker = True
            boxes       = []

            while not self._cancel_flag.is_set():
                ok, frame = cap.read()
                if not ok: break

                if frame_n % skip == 0:
                    boxes = detect_faces(frame, detector,
                                         cfg["confidence"], dscale,
                                         cfg["edge_strip"])
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

                frame = apply_censor(frame, current_boxes,
                                     cfg["mode"], cfg["padding"],
                                     cfg["intensity"], cfg["block_size"],
                                     cfg["debug"])
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

            cap.release(); writer.release()
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
                if os.path.exists(tmp_path): os.remove(tmp_path)
            elif not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                self._write_log("  [ERROR] Output empty\n", "error")
            else:
                ffmpeg = get_ffmpeg_path()
                if ffmpeg:
                    merged = merge_audio(path, tmp_path, outpath, ffmpeg,
                                         log_fn=self._write_log)
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                    if not merged:
                        self._write_log("  Falling back to muted output.\n", "warning")
                        if os.path.exists(tmp_path):
                            os.rename(tmp_path, outpath)
                else:
                    self._write_log("  [WARN] ffmpeg not found - no audio.\n", "warning")
                    if os.path.exists(outpath): os.remove(outpath)
                    os.rename(tmp_path, outpath)

                if os.path.exists(outpath):
                    mb        = os.path.getsize(outpath) / 1048576
                    t_elapsed = time.time() - t_file_start
                    self._write_log(
                        "  Done. {} frames, {} faces, {:.1f} MB, {:.1f}s\n".format(
                            frame_n, faces_n, mb, t_elapsed), "success")
                    self._set_progress(1.0, status="Done: {}".format(fname),
                                       file_n="{}/{}".format(fi, n_files),
                                       frame_n=frame_n, face_n=faces_n)
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
            status="Cancelled." if self._cancel_flag.is_set() else "All done!")


# ══════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ══════════════════════════════════════════════════════════

class SplashScreen(tk.Tk):
    def __init__(self):
        super().__init__()
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
        tk.Label(inner, text="YOLOv8 face censoring  v1.0  |  made by werehappy",
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

    def set_status(self, msg, pct):
        self._status.config(text=msg)
        self._pb.place(x=0, y=0,
                       width=int(self._pb_width * max(0.0, min(1.0, pct))),
                       height=3)
        self.update()


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    multiprocessing.freeze_support()

    import cv2 as _cv2
    import numpy as _np
    globals()["cv2"] = _cv2
    globals()["np"]  = _np

    splash = SplashScreen()
    splash.update_idletasks()
    splash.update()

    _loaded = {}

    def _background_load():
        splash.set_status("Detecting GPU...", 0.5)
        _loaded["gpu_info"] = detect_gpu()
        splash.set_status("Ready.", 0.9)
        splash.after(0, _launch_app)

    def _launch_app():
        splash.set_status("Building interface...", 1.0)
        splash.update()
        app = App(gpu_info=_loaded.get("gpu_info"))
        splash.destroy()
        app.mainloop()

    threading.Thread(target=_background_load, daemon=True).start()
    splash.mainloop()
