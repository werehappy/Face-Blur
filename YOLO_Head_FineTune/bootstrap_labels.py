r"""
bootstrap_labels.py  --  pre-label frames so you CORRECT boxes instead of
drawing them from scratch in Yolo_Label.

It runs a model you already have over a folder of frames and writes a YOLO-format
.txt next to each image (same basename), plus a classes.txt. Yolo_Label then
shows these as existing boxes; your job becomes fixing them and ADDING the
heads the model missed (backs of heads, helmets) -- a fraction of the work.

Everything is single-class: every detection is written as class 0 = head, which
is what your training dataset uses.

TWO BOOTSTRAP MODES:
  --mode person   (default) COCO person detector -> head REGION (top-center,
                  shoulder-width). Same geometry as FACEBLUR. Best RECALL for
                  bootstrap: boxes anyone with a visible torso, including
                  blurred/side/back/helmeted heads.
  --mode head     run a dedicated head model (e.g. head.pt) directly and use its
                  boxes. Tighter, but misses the hard cases (that's why you train).

USAGE (run in the env that has ultralytics + torch, e.g. your training env):
    conda activate yolotrain
    python bootstrap_labels.py dataset_raw
    python bootstrap_labels.py dataset_raw --mode head --model head.pt
    python bootstrap_labels.py dataset_raw --conf 0.20

IMPORTANT: these are a FIRST PASS, not ground truth. The person/head models
systematically MISS helmeted backs-of-heads -- exactly what you're training to
fix -- so scan every frame and add the boxes they skipped. If you trust them
blindly you'll bake the same blind spot into your data.
"""
import os
import sys
import glob
import argparse

PERSON_MODEL = "head_n.pt"   # COCO; ultralytics auto-downloads on first use
PERSON_CLASS = 0
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def person_to_head(x1, y1, x2, y2):
    """Person box -> head region. Same geometry as FACEBLUR:
    0.45*width wide, 0.55*width tall, top-centered on the person box."""
    bw = x2 - x1
    cx = (x1 + x2) / 2.0
    hw = max(1.0, bw * 0.45 / 2.0)
    hh = max(1.0, bw * 0.55)
    return cx - hw, float(y1), cx + hw, y1 + hh


def to_yolo_line(x1, y1, x2, y2, W, H):
    """Pixel xyxy -> normalized 'class cx cy w h' (class always 0 = head)."""
    x1 = max(0.0, min(x1, W)); x2 = max(0.0, min(x2, W))
    y1 = max(0.0, min(y1, H)); y2 = max(0.0, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return "0 %.6f %.6f %.6f %.6f" % (cx, cy, w, h)


def list_images(folder):
    out = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(folder, "*" + ext)))
        out.extend(glob.glob(os.path.join(folder, "*" + ext.upper())))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Pre-label frames for Yolo_Label correction")
    ap.add_argument("folder", help="folder of frames (e.g. dataset_raw)")
    ap.add_argument("--mode", choices=["person", "head"], default="person",
                    help="person->head region (default, best recall) or a head model directly")
    ap.add_argument("--model", default=None,
                    help="model file (default: yolo11n.pt for person mode, head.pt for head mode)")
    ap.add_argument("--conf", type=float, default=0.20, help="confidence floor (default 0.20)")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print("[!] not a folder:", args.folder); sys.exit(1)

    imgs = list_images(args.folder)
    if not imgs:
        print("[!] no images found in", args.folder); sys.exit(1)
    print("[*] %d images in %s" % (len(imgs), args.folder))

    from ultralytics import YOLO
    if args.mode == "person":
        model_file = args.model or PERSON_MODEL
    else:
        model_file = args.model or "head.pt"
        if not os.path.exists(model_file):
            print("[!] head model not found:", model_file,
                  "\n    pass --model path\\to\\your_head_model.pt"); sys.exit(1)
    print("[*] mode=%s  model=%s  conf=%.2f" % (args.mode, model_file, args.conf))
    model = YOLO(model_file)

    # one class only
    with open(os.path.join(args.folder, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("head\n")

    total_boxes = 0
    labeled = 0
    for i, img in enumerate(imgs):
        kw = dict(conf=args.conf, verbose=False)
        if args.mode == "person":
            kw["classes"] = [PERSON_CLASS]
        r = model(img, **kw)[0]
        H, W = r.orig_shape  # (height, width)
        lines = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            if args.mode == "person":
                x1, y1, x2, y2 = person_to_head(x1, y1, x2, y2)
            line = to_yolo_line(x1, y1, x2, y2, W, H)
            if line:
                lines.append(line)

        txt = os.path.splitext(img)[0] + ".txt"
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        total_boxes += len(lines)
        if lines:
            labeled += 1
        if (i + 1) % 25 == 0 or i + 1 == len(imgs):
            print("    %d/%d images  (%d boxes so far)" % (i + 1, len(imgs), total_boxes))

    print()
    print("[DONE] wrote labels for %d images (%d had boxes), %d boxes total."
          % (len(imgs), labeled, total_boxes))
    print("       classes.txt written (single class: head).")
    print()
    print("NEXT: open '%s' in Yolo_Label. Correct the boxes and ADD every head" % args.folder)
    print("      the model missed -- especially helmeted backs-of-heads. Those")
    print("      missed cases are the whole reason you're training.")


if __name__ == "__main__":
    main()
