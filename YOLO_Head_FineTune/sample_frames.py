r"""
sample_frames.py  --  pull diverse, well-spaced frames from your clips to label.

Good training data beats a clever training command. This script walks each clip
and saves frames that are (a) spread across time and (b) visually different from
the ones already kept, so you don't waste labeling effort on 50 near-identical
frames of the same hallway.

USAGE (same conda env you build FACEBLUR with):
    conda activate faceblur
    python sample_frames.py footage_folder --out dataset_raw --per-clip 80
    python sample_frames.py footage_folder -r --out dataset_raw   (incl. subfolders)
    python sample_frames.py clip1.mp4 clip2.mp4 --out dataset_raw --per-clip 80
    python sample_frames.py footage\*.mp4 --out dataset_raw --per-clip 120
    python sample_frames.py clip.mp4 --out dataset_raw --every 15   (fixed stride)

You can pass a FOLDER (all clips inside it are used), individual clip names, or
globs -- or any mix of the three.

OUTPUT:
    <out>/<clipname>_000123.jpg ...   one jpg per kept frame

KNOBS:
    --per-clip N   aim for ~N frames per clip, spread across its length (default 80)
    --every N      instead of per-clip targeting, keep 1 frame every N (overrides)
    --diff D       skip a frame if it's <D different from the last kept one
                   (0..1, default 0.12; raise to keep fewer/more-different frames)
    --min-blur B   drop frames blurrier than B (variance-of-Laplacian; default 0
                   = keep everything, INCLUDING blurry frames -- you WANT some)

Tip: keep a healthy share of HARD frames (motion blur, backs of heads, helmets,
partial/cut-off heads). Those are exactly the cases a public model misses, so
they're the most valuable thing to label.
"""
import os
import sys
import glob
import argparse

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
              ".ts", ".mts", ".m2ts", ".wmv", ".flv",
              ".mpg", ".mpeg", ".3gp", ".ogv")


def frame_signature(frame):
    """Tiny grayscale fingerprint for cheap 'how different is this frame' checks."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, (32, 32)).astype(np.float32) / 255.0
    return s


def diff(a, b):
    """Mean absolute difference between two signatures (0..1)."""
    return float(np.mean(np.abs(a - b)))


def blur_score(frame):
    """Variance of Laplacian -- higher = sharper. Low = motion-blurred."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def diagnose_inputs(inputs, recursive):
    """Explain WHY no videos were found, per input, so the fix is obvious."""
    print("[diagnose] checking why nothing matched "
          "(searched extensions: %s)" % ", ".join(VIDEO_EXTS))
    for item in inputs:
        if os.path.isdir(item):
            entries = os.listdir(item)
            subdirs = [e for e in entries if os.path.isdir(os.path.join(item, e))]
            exts = sorted({os.path.splitext(e)[1].lower()
                           for e in entries
                           if os.path.isfile(os.path.join(item, e))
                           and os.path.splitext(e)[1]})
            print("  folder '%s': %d item(s), %d subfolder(s)"
                  % (item, len(entries), len(subdirs)))
            if exts:
                print("    file types present at top level: %s"
                      % ", ".join(exts))
            if subdirs and not recursive:
                print("    -> clips may be inside subfolders. Re-run with -r "
                      "to include them, e.g.:")
                print("       python sample_frames.py \"%s\" -r --out ..." % item)
            if exts and not any(e in VIDEO_EXTS for e in exts):
                print("    -> none of those are recognized video types. If one "
                      "of them IS a video, tell me the extension and I'll add it.")
        elif not os.path.exists(item):
            print("  '%s': path not found (check the folder name / relative "
                  "location -- you are in:\n     %s)" % (item, os.getcwd()))
        else:
            ext = os.path.splitext(item)[1].lower()
            print("  '%s': not a recognized video type (ext '%s')." % (item, ext))


def expand_inputs(inputs, recursive=False):
    """Resolve each argument into video file paths. An argument may be:
      - a folder  -> all videos inside it (optionally recursing into subfolders)
      - a glob     -> its matches (e.g. footage\\*.mp4)
      - a single file path
    """
    files = []
    for item in inputs:
        if os.path.isdir(item):
            # A directory: pull every video file inside it.
            if recursive:
                for root, _dirs, names in os.walk(item):
                    for nm in names:
                        files.append(os.path.join(root, nm))
            else:
                for nm in os.listdir(item):
                    files.append(os.path.join(item, nm))
        else:
            hits = glob.glob(item)
            files.extend(hits if hits else [item])
    # De-duplicate while preserving order, keep only existing video files.
    seen = set()
    out = []
    for f in files:
        if (os.path.splitext(f)[1].lower() in VIDEO_EXTS
                and os.path.exists(f) and f not in seen):
            seen.add(f)
            out.append(f)
    return out


def sample_clip(path, out_dir, per_clip, every, diff_thresh, min_blur):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[!] cannot open:", path)
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    name = os.path.splitext(os.path.basename(path))[0]

    if every and every > 0:
        stride = every
    else:
        stride = max(1, total // max(1, per_clip)) if total else 15

    print("[*] %s : %d frames, stride %d" % (os.path.basename(path), total, stride))

    kept = 0
    last_sig = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                if min_blur > 0 and blur_score(frame) < min_blur:
                    idx += 1
                    continue
                sig = frame_signature(frame)
                if last_sig is None or diff(sig, last_sig) >= diff_thresh:
                    out = os.path.join(out_dir, "%s_%06d.jpg" % (name, idx))
                    cv2.imwrite(out, frame)
                    kept += 1
                    last_sig = sig
        idx += 1
    cap.release()
    print("    kept %d frames -> %s" % (kept, out_dir))
    return kept


def main():
    ap = argparse.ArgumentParser(description="Sample diverse frames from clips for labeling")
    ap.add_argument("inputs", nargs="+",
                    help="video clip(s), a FOLDER of clips, or globs")
    ap.add_argument("--out", default="dataset_raw", help="output folder (default dataset_raw)")
    ap.add_argument("--per-clip", type=int, default=80, help="target frames per clip (default 80)")
    ap.add_argument("--every", type=int, default=0, help="keep 1 frame every N (overrides --per-clip)")
    ap.add_argument("--diff", type=float, default=0.12, help="min visual difference to keep (0..1, default 0.12)")
    ap.add_argument("--min-blur", type=float, default=0.0, help="drop frames blurrier than this (default 0 = keep all)")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="when a folder is given, also search its subfolders")
    args = ap.parse_args()

    files = expand_inputs(args.inputs, recursive=args.recursive)
    if not files:
        print("[!] no video files found in:", args.inputs)
        diagnose_inputs(args.inputs, args.recursive)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    grand = 0
    for f in files:
        grand += sample_clip(f, args.out, args.per_clip, args.every, args.diff, args.min_blur)

    print()
    print("[DONE] %d frames saved to %s/" % (grand, args.out))
    print("Next: label them (see TRAINING_GUIDE.md). Box EVERY head -- front,")
    print("side, back, helmeted, blurred, partial/cut-off. One class: 'head'.")


if __name__ == "__main__":
    main()
