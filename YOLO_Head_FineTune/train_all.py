r"""
train_all.py -- train several head-model sizes back-to-back, unattended.

Trains yolo11n, yolo11s, yolo11m (by default) on the SAME dataset/settings,
one after another, saving each as head_<size>.pt and printing a comparison so
you can pick the best one to ship -- or wire all three into an app selector.

Built for "start it Friday, read results Monday":
  * each model is isolated in try/except, so one failure doesn't lose the rest
  * batch=-1 auto-sizes per model (m needs a smaller batch than n)
  * a summary table + summary.txt is written at the end

USAGE:
    conda activate yolotrain
    python train_all.py --data head_dataset/data.yaml
    python train_all.py --data head_dataset/data.yaml --models n s m --imgsz 960

IMPORTANT before leaving it:
  * Confirm CUDA: python -c "import torch;print(torch.cuda.is_available())"  -> True
  * Confirm the split is clean: python check_split.py --root head_dataset
  * Disable Windows sleep: powercfg /change standby-timeout-ac 0
  * imgsz MUST match the app's HEAD_INFER_IMGSZ for whichever model you ship
    (default 960). Training at 1280 means setting the app to 1280 too.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Train n/s/m head models back-to-back.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--models", nargs="+", default=["n", "s", "m"],
                    help="sizes to train, from n s m l x (default: n s m)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="training size (default 960; MUST match app HEAD_INFER_IMGSZ)")
    ap.add_argument("--epochs", type=int, default=300, help="epoch ceiling per model")
    ap.add_argument("--patience", type=int, default=60, help="early-stop patience")
    ap.add_argument("--batch", default="-1",
                    help="batch (default -1 = auto-size per model to VRAM)")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--out", default=".", help="where to copy the renamed head_<size>.pt")
    return ap.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        sys.exit("[ERROR] ultralytics not importable: {}".format(e))

    if not Path(args.data).exists():
        sys.exit("[ERROR] data yaml not found: {}".format(args.data))

    try:
        batch = int(args.batch)
    except ValueError:
        batch = args.batch  # allow float like 0.8 for auto-fraction

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []   # (size, status, recall, map50, map5095, weight_path, minutes)

    print("=" * 64)
    print("Plan: train sizes {} at imgsz={}, epochs<= {} patience {}".format(
        ", ".join(args.models), args.imgsz, args.epochs, args.patience))
    print("=" * 64)

    for size in args.models:
        base = "yolo11{}.pt".format(size)
        run_name = "head_{}".format(size)
        print("\n" + "#" * 64)
        print("# TRAINING {}  (base {})".format(run_name, base))
        print("#" * 64)
        t0 = time.time()
        try:
            model = YOLO(base)
            model.train(
                data=args.data,
                imgsz=args.imgsz,
                batch=batch,
                epochs=args.epochs,
                patience=args.patience,
                cos_lr=True,
                close_mosaic=15,
                device=args.device,
                seed=0,
                deterministic=True,
                name=run_name,
                exist_ok=True,
            )
            best = Path("runs/detect") / run_name / "weights" / "best.pt"
            dest = out_dir / "head_{}.pt".format(size)
            recall = map50 = map5095 = None
            if best.exists():
                shutil.copy(str(best), str(dest))
                # Evaluate the best checkpoint at the same imgsz the app will use.
                try:
                    metrics = YOLO(str(best)).val(
                        data=args.data, imgsz=args.imgsz,
                        device=args.device, verbose=False)
                    recall = float(getattr(metrics.box, "mr", getattr(metrics.box, "r", float("nan"))))
                    map50 = float(metrics.box.map50)
                    map5095 = float(metrics.box.map)
                except Exception as e:
                    print("[warn] eval failed for {}: {}".format(run_name, e))
            mins = (time.time() - t0) / 60.0
            results.append((size, "ok", recall, map50, map5095, str(dest), mins))
            print("[done] {} in {:.1f} min -> {}".format(run_name, mins, dest))
        except Exception as e:
            mins = (time.time() - t0) / 60.0
            print("[FAIL] {} after {:.1f} min: {}".format(run_name, mins, e))
            results.append((size, "FAILED", None, None, None, "", mins))
            # keep going to the next size

    # ---- summary ------------------------------------------------------------
    lines = []
    lines.append("=" * 72)
    lines.append("SUMMARY  (imgsz={})".format(args.imgsz))
    lines.append("-" * 72)
    lines.append("{:<6} {:<8} {:>8} {:>8} {:>9} {:>8}  {}".format(
        "size", "status", "recall", "mAP50", "mAP5095", "minutes", "weights"))
    for (size, status, recall, map50, map5095, dest, mins) in results:
        def fmt(v):
            return "{:.3f}".format(v) if isinstance(v, float) else "   -  "
        lines.append("{:<6} {:<8} {:>8} {:>8} {:>9} {:>8.1f}  {}".format(
            size, status, fmt(recall), fmt(map50), fmt(map5095), mins, dest))
    lines.append("-" * 72)
    lines.append("For a privacy blur, RECALL is the metric that matters most.")
    lines.append("Pick the size by recall-vs-speed: ship that head_<size>.pt as")
    lines.append("head.pt, and set HEAD_INFER_IMGSZ={} in face_blur.py.".format(args.imgsz))
    lines.append("=" * 72)

    report = "\n".join(lines)
    print("\n" + report)
    summary_path = out_dir / "training_summary.txt"
    try:
        summary_path.write_text(report, encoding="utf-8")
        print("\n[written] {}".format(summary_path))
    except Exception as e:
        print("[warn] could not write summary file: {}".format(e))


if __name__ == "__main__":
    main()
