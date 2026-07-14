"""
diagnose_heads.py - standalone head-detector diagnostic.

Runs a YOLO head model over an image, a folder of images, or a video, draws the
detected boxes (with confidence), and saves annotated copies. You can pass
several --imgsz values at once to see whether a false positive (e.g. a weapon
illuminator detected as a head) appears at your training size (960) but not at
another size -- which would point to a train/inference mismatch rather than a
data problem.

Examples (Windows cmd):

  # one frame at training size
  python diagnose_heads.py frame.jpg --model runs\\detect\\head_v2\\weights\\best.pt

  # same frame, compare 960 vs 1920 side by side
  python diagnose_heads.py frame.jpg --model best.pt --imgsz 960 1920

  # a whole folder of suspect frames
  python diagnose_heads.py suspect_frames\\ --model best.pt --imgsz 960

  # a video, sampling every 15th frame
  python diagnose_heads.py clip.mp4 --model best.pt --imgsz 960 --stride 15

  # overlay the COCO person->head geometric estimate for comparison
  python diagnose_heads.py frame.jpg --model best.pt --person-model yolo11n.pt

Output goes to --out (default: diag_out\\). Each saved image is annotated, and a
summary is printed per input listing every detection's confidence and box
center (as a fraction of width/height) so you can tell which box is on the head
and which is on the illuminator.

With --person-model, a COCO person detector is also run on every frame and each
person box is converted to an estimated head box (top-center, shoulder-width
sized -- the app's geometric fallback). Those estimates are drawn in BLUE with
a 'g' prefix on the confidence label, so you can compare the trained head model
against the person->head pathway frame by frame, including on the gear-induced
false positives that pathway is known for. Set --geom-w-frac / --geom-aspect to
the same constants the application uses.
"""

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose a YOLO head model on images / folders / videos.")
    p.add_argument("source",
                   help="Path to an image, a folder of images, or a video file.")
    p.add_argument("--model", required=True,
                   help="Path to the model weights (.pt or .onnx).")
    p.add_argument("--person-model", default=None,
                   help="Optional COCO person model; its person->head geometric "
                        "estimates are overlaid in blue for comparison.")
    p.add_argument("--person-class", type=int, default=0,
                   help="Class index of 'person' in the person model (default: 0).")
    p.add_argument("--geom-w-frac", type=float, default=0.40,
                   help="Estimated head width as a fraction of person-box width "
                        "(default 0.40; match the app's fallback constant).")
    p.add_argument("--geom-aspect", type=float, default=1.10,
                   help="Estimated head height as a multiple of head width "
                        "(default 1.10).")
    p.add_argument("--imgsz", type=int, nargs="+", default=[960],
                   help="One or more inference sizes to try (default: 960). "
                        "Pass several to compare, e.g. --imgsz 960 1920.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (default: 0.25).")
    p.add_argument("--device", default="0",
                   help="CUDA device index or 'cpu' (default: 0).")
    p.add_argument("--out", default="diag_out",
                   help="Output folder for annotated images (default: diag_out).")
    p.add_argument("--stride", type=int, default=30,
                   help="For videos: process every Nth frame (default: 30).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="For videos: stop after this many processed frames "
                        "(0 = no limit).")
    return p.parse_args()


def collect_images(source: Path):
    """Return a list of image paths from a file or folder."""
    if source.is_dir():
        return sorted(p for p in source.rglob("*")
                      if p.suffix.lower() in IMG_EXTS)
    return [source]


def head_from_person(box, w_frac, aspect):
    """Estimate a head box (xyxy) from a person box: top-center, head width =
    w_frac * person-box width, height = aspect * head width."""
    x1, y1, x2, _y2 = box
    hw = (x2 - x1) * w_frac
    hh = hw * aspect
    cx = (x1 + x2) / 2.0
    return (cx - hw / 2.0, y1, cx + hw / 2.0, y1 + hh)


def geom_estimates(result, w_frac, aspect):
    """Convert a person-model result into [(conf, head_xyxy)] estimates."""
    out = []
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        conf = float(boxes.conf[i].item())
        pbox = tuple(float(v) for v in boxes.xyxy[i].tolist())
        out.append((conf, head_from_person(pbox, w_frac, aspect)))
    return out


def draw_and_save(cv2, frame, result, out_path, imgsz, geom=None):
    """Draw result boxes on a copy of frame, save to out_path, return summary.
    geom, if given, is [(conf, head_xyxy)] from the person->head pathway and is
    drawn in blue with a 'g' label prefix."""
    img = frame.copy()
    h, w = img.shape[:2]
    lines = []
    boxes = result.boxes
    n = 0 if boxes is None else len(boxes)
    for i in range(n):
        conf = float(boxes.conf[i].item())
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        # Color by confidence: high-conf in red so illuminator fires stand out.
        color = (0, 0, 255) if conf >= 0.7 else (0, 200, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = "{:.2f}".format(conf)
        cv2.putText(img, label, (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        lines.append("      conf={:.3f}  center=({:.3f},{:.3f})  "
                     "size=({:.3f},{:.3f})".format(
                         conf, cx, cy, (x2 - x1) / w, (y2 - y1) / h))
    for conf, (x1, y1, x2, y2) in geom or []:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        color = (255, 128, 0)  # blue-ish (BGR) for the geometric pathway
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, "g{:.2f}".format(conf), (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        lines.append("      geom conf={:.3f}  center=({:.3f},{:.3f})  "
                     "size=({:.3f},{:.3f})".format(
                         conf, cx, cy, (x2 - x1) / w, (y2 - y1) / h))
    cv2.imwrite(str(out_path), img)
    return n, lines


def main():
    args = parse_args()

    try:
        import cv2
    except Exception as e:
        sys.exit("[ERROR] opencv not available: {}\n"
                 "        pip install opencv-python".format(e))
    try:
        from ultralytics import YOLO
    except Exception as e:
        sys.exit("[ERROR] ultralytics not available: {}\n"
                 "        pip install ultralytics".format(e))

    source = Path(args.source)
    if not source.exists():
        sys.exit("[ERROR] source not found: {}".format(source))
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit("[ERROR] model not found: {}".format(model_path))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading model: {}".format(model_path))
    model = YOLO(str(model_path))

    pmodel = None
    if args.person_model:
        pm_path = Path(args.person_model)
        if not pm_path.exists():
            sys.exit("[ERROR] person model not found: {}".format(pm_path))
        print("Loading person model (geometric baseline): {}".format(pm_path))
        pmodel = YOLO(str(pm_path))

    is_video = source.is_file() and source.suffix.lower() in VID_EXTS

    for imgsz in args.imgsz:
        out_dir = out_root / "imgsz_{}".format(imgsz)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\n=== imgsz={}  conf={}  device={} ===".format(
            imgsz, args.conf, args.device))

        total_dets = 0
        total_highconf = 0
        processed = 0

        if is_video:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                print("[WARN] could not open video: {}".format(source))
                continue
            idx = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % max(1, args.stride) != 0:
                    continue
                res = model.predict(frame, imgsz=imgsz, conf=args.conf,
                                    device=args.device, verbose=False)[0]
                geom = None
                if pmodel is not None:
                    pres = pmodel.predict(frame, imgsz=imgsz, conf=args.conf,
                                          device=args.device,
                                          classes=[args.person_class],
                                          verbose=False)[0]
                    geom = geom_estimates(pres, args.geom_w_frac, args.geom_aspect)
                name = "frame_{:06d}.jpg".format(idx)
                n, lines = draw_and_save(cv2, frame, res, out_dir / name, imgsz,
                                         geom=geom)
                hi = sum(1 for ln in lines if "geom" not in ln
                         and float(ln.split("conf=")[1].split()[0]) >= 0.7)
                total_dets += n
                total_highconf += hi
                processed += 1
                if n:
                    print("  {}: {} det(s){}".format(
                        name, n, "  <-- has >=0.70" if hi else ""))
                    for ln in lines:
                        print(ln)
                if args.max_frames and processed >= args.max_frames:
                    break
            cap.release()
        else:
            images = collect_images(source)
            if not images:
                print("[WARN] no images found in {}".format(source))
                continue
            for img_path in images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print("[WARN] could not read {}".format(img_path))
                    continue
                res = model.predict(frame, imgsz=imgsz, conf=args.conf,
                                    device=args.device, verbose=False)[0]
                geom = None
                if pmodel is not None:
                    pres = pmodel.predict(frame, imgsz=imgsz, conf=args.conf,
                                          device=args.device,
                                          classes=[args.person_class],
                                          verbose=False)[0]
                    geom = geom_estimates(pres, args.geom_w_frac, args.geom_aspect)
                n, lines = draw_and_save(
                    cv2, frame, res, out_dir / img_path.name, imgsz, geom=geom)
                hi = sum(1 for ln in lines if "geom" not in ln
                         and float(ln.split("conf=")[1].split()[0]) >= 0.7)
                total_dets += n
                total_highconf += hi
                processed += 1
                print("  {}: {} det(s){}".format(
                    img_path.name, n, "  <-- has >=0.70" if hi else ""))
                for ln in lines:
                    print(ln)

        print("  ---- imgsz={} summary: {} frame(s), {} detection(s), "
              "{} at conf>=0.70 ----".format(
                  imgsz, processed, total_dets, total_highconf))

    print("\nDone. Annotated images saved under: {}".format(out_root.resolve()))
    print("Red boxes = conf >= 0.70 (the band where your illuminator fires).")
    if pmodel is not None:
        print("Blue 'g' boxes = person->head geometric estimates from {}.".format(
            args.person_model))
    if len(args.imgsz) > 1:
        print("Compare the imgsz_* folders: if the illuminator box appears at "
              "one size but not another, it's a train/inference size mismatch, "
              "not a data problem.")


if __name__ == "__main__":
    main()
