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

Output goes to --out (default: diag_out\\). Each saved image is annotated, and a
summary is printed per input listing every detection's confidence and box
center (as a fraction of width/height) so you can tell which box is on the head
and which is on the illuminator.
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


def draw_and_save(cv2, frame, result, out_path, imgsz):
    """Draw result boxes on a copy of frame, save to out_path, return summary."""
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
                name = "frame_{:06d}.jpg".format(idx)
                n, lines = draw_and_save(cv2, frame, res, out_dir / name, imgsz)
                hi = sum(1 for ln in lines
                         if float(ln.split("conf=")[1].split()[0]) >= 0.7)
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
                n, lines = draw_and_save(
                    cv2, frame, res, out_dir / img_path.name, imgsz)
                hi = sum(1 for ln in lines
                         if float(ln.split("conf=")[1].split()[0]) >= 0.7)
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
    if len(args.imgsz) > 1:
        print("Compare the imgsz_* folders: if the illuminator box appears at "
              "one size but not another, it's a train/inference size mismatch, "
              "not a data problem.")


if __name__ == "__main__":
    main()
