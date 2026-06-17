r"""
test_detect.py  --  compare PERSON vs HEAD detection on your CQB footage.

The head model can't see helmeted/blurred heads. Person detection (COCO) is far
more robust to blur/occlusion/partial bodies, so this checks whether we can get
reliable coverage by detecting the PERSON and censoring the head region instead.

USAGE (same conda env you build FACEBLUR with):
    conda activate faceblur
    python test_detect.py path\to\your_cqb_clip.mp4
    python test_detect.py path\to\a_frame.jpg

Boxes drawn on the saved frames:
    BLUE   = person box (COCO yolo11)         <- robust to blur/helmets
    YELLOW = head region derived from person  <- what would get censored
    GREEN  = head-model box (head.pt)         <- for comparison (often empty here)

Send me: the printed per-frame counts and a couple of saved *_cmp.jpg frames
(ideally with a back-of-head). If BLUE/YELLOW cover the people but GREEN doesn't,
the person-region approach is the way to go and I'll wire it into the app.
"""
import os, sys, urllib.request

PERSON_MODEL    = "yolo11n.pt"   # COCO; ultralytics auto-downloads on first use
HEAD_MODEL_FILE = "head.pt"
HEAD_MODEL_URL  = "https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/nano.pt"
PERSON_CLASS    = 0
CONF            = 0.20           # low, full-res; we want recall


def person_to_head(x1, y1, x2, y2):
    """Head region from a person box, sized by WIDTH (robust to truncated bodies):
    head ~ 0.45*width wide, 0.55*width tall, top-centered on the person box."""
    bw = x2 - x1
    cx = (x1 + x2) // 2
    hw = max(1, int(bw * 0.45) // 2)
    hh = max(1, int(bw * 0.55))
    return (cx - hw, y1, cx + hw, y1 + hh)


def load_person():
    from ultralytics import YOLO
    try:
        m = YOLO(PERSON_MODEL)   # auto-downloads from ultralytics on first run
        print("[*] person model loaded:", PERSON_MODEL, "| classes incl. person =",
              0 in m.names)
        return m
    except Exception as e:
        print("[!] person model failed to load/download:", repr(e))
        return None


def load_head():
    from ultralytics import YOLO
    if not os.path.exists(HEAD_MODEL_FILE):
        try:
            print("[*] downloading head model...")
            urllib.request.urlretrieve(HEAD_MODEL_URL, HEAD_MODEL_FILE)
        except Exception as e:
            print("[!] head model download failed:", repr(e), "(continuing without it)")
            return None
    try:
        return YOLO(HEAD_MODEL_FILE)
    except Exception as e:
        print("[!] head model load failed:", repr(e))
        return None


def annotate(frame, person, head, out_path):
    import cv2
    np_person = np_head = 0
    if person is not None:
        r = person(frame, conf=CONF, classes=[PERSON_CLASS], verbose=False)[0]
        np_person = len(r.boxes)
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)          # BLUE person
            hx1, hy1, hx2, hy2 = person_to_head(x1, y1, x2, y2)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)    # YELLOW head region
    if head is not None:
        r = head(frame, conf=CONF, verbose=False)[0]
        np_head = len(r.boxes)
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)          # GREEN head-model
    cv2.imwrite(out_path, frame)
    return np_person, np_head


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    src = sys.argv[1]
    if not os.path.exists(src):
        print("[!] not found:", src); sys.exit(1)

    import cv2
    person = load_person()
    head = load_head()
    if person is None and head is None:
        print("[!] no models available; cannot continue."); sys.exit(1)

    base = os.path.splitext(src)[0]
    ext = os.path.splitext(src)[1].lower()

    if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"):
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        print("[*] video: %d frames | sampling 6" % total)
        tot_p = tot_h = 0
        for i in range(6):
            pos = int(total * (i + 0.5) / 6) if total else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            out = "%s_cmp%02d.jpg" % (base, i)
            p, h = annotate(frame, person, head, out)
            tot_p += p; tot_h += h
            print("    frame %6d : person=%d  head=%d  -> %s" % (pos, p, h, out))
        cap.release()
        print("\n[SUMMARY] across 6 sampled frames: person=%d  head=%d" % (tot_p, tot_h))
        print("  If person >> head, the person-region approach is the fix.")
        print("  If person is also ~0, the footage is too blurred/cropped for")
        print("  any off-the-shelf model and we should fine-tune on your frames.")
    else:
        frame = cv2.imread(src)
        if frame is None:
            print("[!] could not read image"); sys.exit(1)
        out = base + "_cmp.jpg"
        p, h = annotate(frame, person, head, out)
        print("[*] person=%d  head=%d  -> %s" % (p, h, out))


if __name__ == "__main__":
    main()
