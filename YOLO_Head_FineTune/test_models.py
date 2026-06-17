r"""
test_models.py  --  shoot-out of AVAILABLE head detectors on your CQB footage.

Goal: find the best off-the-shelf head.pt for backs of heads, fully-side faces,
and 1/4 cut-off faces -- without training anything. It compares, on YOUR clips:

    PERSON       person box -> head region (BLUE person, YELLOW head region)
    CROWDHUMAN   YOLOv8 head model trained on CrowdHuman   (GREEN)
    SCUT-N       SCUT-HEAD nano  (what was tested before)  (MAGENTA)
    SCUT-M       SCUT-HEAD medium                          (RED)

CrowdHuman is the most promising: it's trained on crowded street scenes full of
occluded, side-on, back-facing and frame-cut heads (unlike SCUT-HEAD, which is
classroom photos). It must be downloaded ONCE by hand (Google Drive):

    https://github.com/Owen718/Head-Detection-Yolov8
    -> "Pre-trained YoloV8 Head Detection Model" Google Drive link
    -> save the .pt file as  crowdhuman.pt  next to this script

SCUT models auto-download. Missing models are skipped, not fatal.

All models run with the recall boosters that matter for this footage:
full resolution, low confidence (0.10), and optional test-time augmentation.

USAGE (same conda env as FACEBLUR):
    conda activate faceblur
    python test_models.py path\to\clip.mp4
    python test_models.py clip1.mp4 clip2.mp4 --frames 8
    python test_models.py clip.mp4 --tta          (slower, more recall)
    python test_models.py frame.jpg

OUTPUT:
    <clip>_models_NN.jpg   one annotated image per sampled frame, all models
    a per-frame count table + summary, and a recommendation.

Reading the result: the winner is whichever GREEN/MAGENTA/RED model fires on
the heads that YELLOW misses (the bodyless ones). Copy that model file to
head.pt next to FACEBLUR -- the app runs head.pt AND the person
method together (union), so head.pt can only ADD coverage, never lose any.
"""
import os
import sys
import argparse
import urllib.request

PERSON_MODEL = "yolo11n.pt"   # auto-downloads via ultralytics
PERSON_CLASS = 0

# (file, url-or-None, label, BGR color)
HEAD_MODELS = [
    ("crowdhuman.pt", None,
     "CROWDHUMAN", (0, 255, 0)),       # GREEN  - manual download, see header
    ("head_scut_n.pt",
     "https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/nano.pt",
     "SCUT-N", (255, 0, 255)),         # MAGENTA
    ("head_scut_m.pt",
     "https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/medium.pt",
     "SCUT-M", (0, 0, 255)),           # RED
]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def person_to_head(x1, y1, x2, y2):
    """Same geometry as FACEBLUR: 0.45w wide, 0.55w tall, top-centered."""
    bw = x2 - x1
    cx = (x1 + x2) / 2.0
    hw = max(1.0, bw * 0.45 / 2.0)
    hh = max(1.0, bw * 0.55)
    return int(cx - hw), int(y1), int(cx + hw), int(y1 + hh)


def load_models():
    from ultralytics import YOLO
    loaded = []   # (label, model, color, is_person)
    try:
        m = YOLO(PERSON_MODEL)
        loaded.append(("PERSON", m, (255, 0, 0), True))   # BLUE
        print("[*] PERSON model loaded (%s)" % PERSON_MODEL)
    except Exception as e:
        print("[!] person model failed:", repr(e))
    for fname, url, label, color in HEAD_MODELS:
        if not os.path.exists(fname):
            if url is None:
                print("[!] %s missing -- %s skipped." % (fname, label))
                print("    Download it once (see header docstring) for the most")
                print("    promising candidate on this kind of footage.")
                continue
            try:
                print("[*] downloading %s ..." % fname)
                urllib.request.urlretrieve(url, fname)
            except Exception as e:
                print("[!] %s download failed (%r) -- skipped." % (label, e))
                continue
        if os.path.getsize(fname) < 100000:
            print("[!] %s is tiny (%d bytes) -- not a real model, skipped."
                  % (fname, os.path.getsize(fname)))
            continue
        try:
            m = YOLO(fname)
            loaded.append((label, m, color, False))
            print("[*] %s loaded (%s), classes=%s" % (label, fname, m.names))
        except Exception as e:
            print("[!] %s failed to load (%r) -- skipped." % (label, e))
    return loaded


def run_model(model, frame, conf, is_person, tta):
    kw = dict(conf=conf, verbose=False, augment=tta)
    if is_person:
        kw["classes"] = [PERSON_CLASS]
    res = model(frame, **kw)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out


def annotate(models, frame, out_path, conf, tta):
    import cv2
    counts = {}
    y_legend = 26
    for label, model, color, is_person in models:
        boxes = run_model(model, frame, conf, is_person, tta)
        counts[label] = len(boxes)
        for (x1, y1, x2, y2, s) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if is_person:
                hx1, hy1, hx2, hy2 = person_to_head(x1, y1, x2, y2)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)  # YELLOW
        cv2.putText(frame, "%s: %d" % (label, counts[label]), (8, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_legend += 26
    cv2.imwrite(out_path, frame)
    return counts


def main():
    ap = argparse.ArgumentParser(description="Compare available head detectors on your footage")
    ap.add_argument("inputs", nargs="+", help="video clip(s) or image(s)")
    ap.add_argument("--frames", type=int, default=6, help="frames sampled per clip (default 6)")
    ap.add_argument("--conf", type=float, default=0.10, help="confidence floor (default 0.10 -- recall first)")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (multi-scale+flip, slower, more recall)")
    args = ap.parse_args()

    import cv2
    models = load_models()
    if not models:
        print("[!] no models available."); sys.exit(1)
    print("[*] running at FULL resolution, conf=%.2f, TTA=%s" % (args.conf, args.tta))
    print()

    totals = {label: 0 for (label, _, _, _) in models}
    for src in args.inputs:
        if not os.path.exists(src):
            print("[!] not found:", src); continue
        base, ext = os.path.splitext(src)
        if ext.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(src)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            print("[*] %s : %d frames, sampling %d" % (os.path.basename(src), total, args.frames))
            for i in range(args.frames):
                pos = int(total * (i + 0.5) / args.frames) if total else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if not ok:
                    continue
                out = "%s_models%02d.jpg" % (base, i)
                counts = annotate(models, frame, out, args.conf, args.tta)
                for k, v in counts.items():
                    totals[k] += v
                print("    frame %6d : %s  -> %s" % (
                    pos, "  ".join("%s=%d" % kv for kv in counts.items()), out))
            cap.release()
        else:
            frame = cv2.imread(src)
            if frame is None:
                print("[!] cannot read:", src); continue
            out = base + "_models.jpg"
            counts = annotate(models, frame, out, args.conf, args.tta)
            for k, v in counts.items():
                totals[k] += v
            print("[*] %s : %s -> %s" % (
                src, "  ".join("%s=%d" % kv for kv in counts.items()), out))

    print()
    print("[SUMMARY] total detections: %s" % "  ".join("%s=%d" % kv for kv in totals.items()))
    print()
    print("HOW TO DECIDE (open the saved *_models*.jpg and look, counts alone lie):")
    print("  * YELLOW already covers everyone with a torso. The question is ONLY:")
    print("    which colored model fires on heads that have NO yellow box?")
    print("  * If one head model wins those frames -> copy its file to head.pt")
    print("    next to FACEBLUR. The app unions head.pt WITH the person method, so")
    print("    it only adds coverage.")
    print("  * If --tta clearly beats no-TTA for the winner, note it; TTA is not")
    print("    used in-app (too slow per frame) but tells you the model has more")
    print("    recall available -> lower the in-app confidence instead.")
    print("  * If NO model catches the bodyless back-of-heads, no public model")
    print("    will: that is the fine-tuning case (see TRAINING_GUIDE.md).")


if __name__ == "__main__":
    main()
