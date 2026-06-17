r"""
test_head.py  --  standalone head-detection diagnostic for FACEBLUR.

Purpose: figure out, with zero involvement from the GUI/build/settings, whether
the head model actually detects heads (incl. the back of the head) in YOUR video.

USAGE (in the same conda env you build FACEBLUR with):
    conda activate faceblur
    python test_head.py path\to\a_frame.jpg
    python test_head.py path\to\your_video.mp4

What it does:
  * loads the head model (local head.pt if present, else downloads it)
  * runs it at LOW confidence so nothing is filtered out
  * for an image: saves <name>_heads.jpg with green boxes drawn
  * for a video : samples 6 frames across the clip, saves <name>_headNN.jpg
  * prints the model's classes and how many heads it found per frame

Send me: the printed output, and one of the saved _heads images (especially a
frame that contains a back-of-head). That tells us if the MODEL is the problem
or the INTEGRATION is.
"""
import os, sys, urllib.request

HEAD_MODEL_FILE = "head.pt"
HEAD_MODEL_URL  = "https://raw.githubusercontent.com/Abcfsa/YOLOv8_head_detector/main/nano.pt"
# More accurate (52 MB): .../main/medium.pt
CONF = 0.15   # deliberately low, to see everything the model can find


def load_model():
    from ultralytics import YOLO
    if not os.path.exists(HEAD_MODEL_FILE):
        print("[*] %s not found -- downloading from:\n    %s" % (HEAD_MODEL_FILE, HEAD_MODEL_URL))
        try:
            urllib.request.urlretrieve(HEAD_MODEL_URL, HEAD_MODEL_FILE)
            print("[*] download OK (%d bytes)" % os.path.getsize(HEAD_MODEL_FILE))
        except Exception as e:
            print("[!] DOWNLOAD FAILED:", repr(e))
            print("    -> This is likely why the app shows face-only.")
            print("    -> Your network may block raw.githubusercontent.com.")
            print("    -> Fix: download a head model in a browser, rename to head.pt,")
            print("       put it next to this script and the app, and rerun.")
            sys.exit(1)
    size = os.path.getsize(HEAD_MODEL_FILE)
    if size < 100000:   # a real model is multiple MB; tiny = HTML error page
        print("[!] %s is only %d bytes -- not a real model (probably an error page)."
              % (HEAD_MODEL_FILE, size))
        print("    Delete it and download a proper head model as head.pt.")
        sys.exit(1)
    m = YOLO(HEAD_MODEL_FILE)
    print("[*] model loaded. classes =", m.names)
    return m


def annotate(model, frame, out_path):
    import cv2
    res = model(frame, conf=CONF, verbose=False)[0]
    n = len(res.boxes)
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        conf = float(b.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "%.2f" % conf, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, frame)
    return n


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    src = sys.argv[1]
    if not os.path.exists(src):
        print("[!] file not found:", src)
        sys.exit(1)

    import cv2
    model = load_model()
    base, _ = os.path.splitext(src)
    ext = os.path.splitext(src)[1].lower()

    if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"):
        cap = cv2.VideoCapture(src)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        print("[*] video: %d frames" % total)
        n_samples = 6
        grand = 0
        for i in range(n_samples):
            pos = int(total * (i + 0.5) / n_samples) if total else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            out = "%s_head%02d.jpg" % (base, i)
            n = annotate(model, frame, out)
            grand += n
            print("    frame %6d : %d head(s)  -> %s" % (pos, n, out))
        cap.release()
        print("[*] total heads across sampled frames:", grand)
        if grand == 0:
            print("[!] The model found NO heads in your footage.")
            print("    -> Tell me what kind of video this is (real people? anime?")
            print("       game capture? low light?). The default model is trained on")
            print("       real-photo heads (SCUT-HEAD) and won't work on drawings.")
    else:
        frame = cv2.imread(src)
        if frame is None:
            print("[!] could not read image:", src)
            sys.exit(1)
        out = base + "_heads.jpg"
        n = annotate(model, frame, out)
        print("[*] heads found: %d  -> saved %s" % (n, out))
        if n == 0:
            print("[!] No heads found. Tell me what kind of content this is.")


if __name__ == "__main__":
    main()
