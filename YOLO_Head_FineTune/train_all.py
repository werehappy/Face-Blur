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

    # retrain after a warmup-inflated best.pt, and guard against regressions
    python train_all.py --data head_dataset/data.yaml --models n ^
        --lr0 0.01 --warmup-epochs 1.0 --baseline-recall 0.737

    # use hyperparameters found by the tuner (train_head.py search phase);
    # explicit flags still override individual values from the yaml
    python train_all.py --data head_dataset/data.yaml ^
        --hyp runs/detect/tune/best_hyperparameters.yaml --baseline-recall 0.737

HYPERPARAMETER PRECEDENCE:
  explicit CLI flag  >  --hyp yaml  >  safe defaults (lr0=0.01, warmup_epochs=1.0).
  Only recognized tunable keys are taken from the yaml; anything else is
  ignored with a note. The effective values are printed at startup and
  recorded in the summary header, so every run is self-documenting.

LEARNING RATE (why a first-epoch best.pt happens):
  Fine-tuning from COCO weights at the ultralytics default lr0=0.03 is unstable
  -- the LR warmup evaluates epoch 1 at a gentle rate (so it looks great), then
  the full 0.03 blows the weights apart and val cls_loss spikes; later epochs
  only claw back. Ultralytics picks best.pt by val fitness, so that inflated
  epoch 1 can win and you ship a barely-trained model. Defaults here are the
  safer lr0=0.01 with a 1-epoch warmup; pass --baseline-recall to have any run
  that fails to beat your current model flagged instead of silently shipped.

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
from datetime import datetime
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
    ap.add_argument("--lr0", type=float, default=None,
                    help="initial learning rate (resolved default 0.01; fine-tuning from "
                         "COCO weights is unstable at the ultralytics 0.03 default -- "
                         "that high LR is what let a warmup-inflated epoch 1 win best.pt). "
                         "Overrides --hyp if both are given.")
    ap.add_argument("--warmup-epochs", type=float, default=None,
                    help="LR warmup length in epochs (resolved default 1.0; a long 3-epoch "
                         "warmup evaluates epoch 1 at a gentle LR and can crown it "
                         "best.pt before the real training even starts). Overrides --hyp.")
    ap.add_argument("--hyp", default=None,
                    help="path to a tuned hyperparameter yaml (e.g. the "
                         "best_hyperparameters.yaml written by the ultralytics tuner / "
                         "train_head.py search). Loaded first; any explicit CLI flag "
                         "below overrides it; unset values fall back to safe defaults.")
    ap.add_argument("--lrf", type=float, default=None,
                    help="final LR fraction for the cosine schedule (optional override)")
    ap.add_argument("--scale", type=float, default=None,
                    help="scale augmentation gain (optional; targets the tiny-top-view "
                         "vs large-first-person head size spread)")
    ap.add_argument("--degrees", type=float, default=None,
                    help="rotation augmentation in degrees (optional; ~10 suits "
                         "helmet-cam roll)")
    ap.add_argument("--hsv-v", type=float, default=None,
                    help="value/brightness augmentation gain (optional; lighting varies "
                         "across sources)")
    ap.add_argument("--mosaic", type=float, default=None,
                    help="mosaic augmentation probability (optional; helps small-object "
                         "recall)")
    ap.add_argument("--baseline-recall", type=float, default=None,
                    help="recall of the model you currently ship (e.g. 0.737). If set, "
                         "any newly trained size that does not beat it is flagged in the "
                         "summary as a REGRESSION so you don't ship it by mistake.")
    ap.add_argument("--batch", default="-1",
                    help="batch (default -1 = auto-size per model to VRAM)")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--out", default="runs",
                    help="root results folder (default: runs); this run goes in "
                         "runs/<date_time>/")
    ap.add_argument("--tag", default=None,
                    help="name for this run's subfolder (default: the run date/time, "
                         "e.g. 2026-07-14_09-30-05)")
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

    # ---- resolve hyperparameters: --hyp yaml < explicit CLI < nothing ------
    # Keys the ultralytics tuner emits that are valid model.train() kwargs.
    TUNABLE = {
        "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
        "warmup_momentum", "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective",
        "flipud", "fliplr", "bgr", "mosaic", "mixup", "copy_paste",
    }
    hyp = {}
    if args.hyp:
        hyp_path = Path(args.hyp)
        if not hyp_path.exists():
            sys.exit("[ERROR] --hyp file not found: {}".format(hyp_path))
        import yaml
        with open(str(hyp_path), encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        ignored = sorted(set(loaded) - TUNABLE)
        hyp = {k: v for k, v in loaded.items() if k in TUNABLE}
        print("[hyp] loaded {} tuned value(s) from {}".format(len(hyp), hyp_path))
        if ignored:
            print("[hyp] ignored non-tunable keys: {}".format(", ".join(ignored)))
    # Explicit CLI flags override the yaml.
    for k, v in [("lr0", args.lr0), ("lrf", args.lrf),
                 ("warmup_epochs", args.warmup_epochs), ("scale", args.scale),
                 ("degrees", args.degrees), ("hsv_v", args.hsv_v),
                 ("mosaic", args.mosaic)]:
        if v is not None:
            hyp[k] = v
    # Safe defaults for the two values that caused the epoch-1 best.pt problem,
    # applied only if neither the yaml nor the CLI set them.
    hyp.setdefault("lr0", 0.01)
    hyp.setdefault("warmup_epochs", 1.0)
    print("[hyp] effective: {}".format(
        ", ".join("{}={}".format(k, hyp[k]) for k in sorted(hyp))))

    # ---- one tidy folder for this whole run --------------------------------
    #   runs/<date_time>/
    #       head_n/ head_s/ head_m/   -> Ultralytics output per size
    #                                    (curves, logs, weights/best.pt)
    #       head_n.pt head_s.pt ...   -> renamed best checkpoints (what you ship)
    #       training_summary.txt
    tag = args.tag or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(args.out) / tag
    weights_dir = run_dir          # shipped .pt files sit at the run root
    yolo_dir = run_dir             # ultralytics head_<size>/ folders too
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []   # (size, status, recall, map50, map5095, weight_path, minutes, best_epoch)

    print("=" * 64)
    print("Output folder: {}".format(run_dir.resolve()))
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
                project=str(yolo_dir),
                name=run_name,
                exist_ok=True,
                **hyp,
            )
            best = yolo_dir / run_name / "weights" / "best.pt"
            dest = weights_dir / "head_{}.pt".format(size)
            recall = map50 = map5095 = None
            best_epoch = None
            # Which epoch became best.pt? Ultralytics picks by fitness on the VAL
            # split; a warmup-inflated epoch 1 winning is the classic sign that
            # lr0 is too high and the run peaked before it really trained.
            res_csv = yolo_dir / run_name / "results.csv"
            if res_csv.exists():
                try:
                    import csv as _csv
                    with open(str(res_csv), encoding="utf-8") as f:
                        rows = list(_csv.DictReader(f))
                    def _fit(r):
                        return (0.1 * float(r["metrics/mAP50(B)"])
                                + 0.9 * float(r["metrics/mAP50-95(B)"]))
                    best_row = max(rows, key=_fit)
                    best_epoch = int(float(best_row["epoch"]))
                    if best_epoch <= 2 and len(rows) > 3:
                        print("[WARN] best.pt for {} came from epoch {} of {} -- the "
                              "run likely peaked during warmup and got worse after. "
                              "Lower lr0 (currently {}) and/or shorten "
                              "warmup_epochs (currently {}).".format(
                                  run_name, best_epoch, len(rows), hyp["lr0"],
                                  hyp["warmup_epochs"]))
                except Exception as e:
                    print("[warn] could not parse {}: {}".format(res_csv, e))
            if best.exists():
                shutil.copy(str(best), str(dest))
                # Evaluate the best checkpoint at the same imgsz the app will use.
                try:
                    metrics = YOLO(str(best)).val(
                        data=args.data, imgsz=args.imgsz,
                        device=args.device, verbose=False,
                        project=str(yolo_dir), name="{}_val".format(run_name),
                        exist_ok=True)
                    recall = float(getattr(metrics.box, "mr", getattr(metrics.box, "r", float("nan"))))
                    map50 = float(metrics.box.map50)
                    map5095 = float(metrics.box.map)
                except Exception as e:
                    print("[warn] eval failed for {}: {}".format(run_name, e))
            mins = (time.time() - t0) / 60.0
            results.append((size, "ok", recall, map50, map5095, str(dest), mins, best_epoch))
            print("[done] {} in {:.1f} min -> {}".format(run_name, mins, dest))
        except Exception as e:
            mins = (time.time() - t0) / 60.0
            print("[FAIL] {} after {:.1f} min: {}".format(run_name, mins, e))
            results.append((size, "FAILED", None, None, None, "", mins, None))
            # keep going to the next size

    # ---- summary ------------------------------------------------------------
    lines = []
    lines.append("=" * 84)
    lines.append("SUMMARY  (imgsz={}, lr0={}, warmup_epochs={}{})".format(
        args.imgsz, hyp["lr0"], hyp["warmup_epochs"],
        ", hyp={}".format(args.hyp) if args.hyp else ""))
    lines.append("-" * 84)
    lines.append("{:<6} {:<8} {:>8} {:>8} {:>9} {:>8} {:>6}  {}".format(
        "size", "status", "recall", "mAP50", "mAP5095", "minutes", "bestEp", "weights/flag"))
    flagged = []
    for (size, status, recall, map50, map5095, dest, mins, best_epoch) in results:
        def fmt(v):
            # NaN (v != v) prints as '-' too, not 'nan'
            return "{:.3f}".format(v) if isinstance(v, float) and v == v else "   -  "
        note = dest
        # Flag a checkpoint that regressed against the model you already ship.
        if (args.baseline_recall is not None and isinstance(recall, float)
                and recall == recall and recall < args.baseline_recall):
            note = "REGRESSION vs baseline {:.3f} -- DO NOT SHIP".format(args.baseline_recall)
            flagged.append(size)
        ep_str = str(best_epoch) if best_epoch is not None else "-"
        # Mark a suspiciously early best epoch inline.
        if best_epoch is not None and best_epoch <= 2:
            ep_str += "!"
        lines.append("{:<6} {:<8} {:>8} {:>8} {:>9} {:>8.1f} {:>6}  {}".format(
            size, status, fmt(recall), fmt(map50), fmt(map5095), mins, ep_str, note))
    lines.append("-" * 84)
    lines.append("For a privacy blur, RECALL is the metric that matters most.")
    lines.append("Pick the size by recall-vs-speed: ship that head_<size>.pt")
    lines.append("as head.pt, and set HEAD_INFER_IMGSZ={} in face_blur.py.".format(args.imgsz))
    if any(ep is not None and ep <= 2 for *_r, ep in results):
        lines.append("")
        lines.append("NOTE: a '!' on bestEp means best.pt came from epoch 1-2 -- the run")
        lines.append("peaked during warmup and degraded after. Retrain with a lower --lr0")
        lines.append("and/or shorter --warmup-epochs before trusting that checkpoint.")
    if flagged:
        lines.append("")
        lines.append("WARNING: size(s) {} scored below your --baseline-recall and would be".format(
            ", ".join(flagged)))
        lines.append("a regression -- keep shipping your existing model until a run beats it.")
    lines.append("=" * 84)

    report = "\n".join(lines)
    print("\n" + report)
    summary_path = run_dir / "training_summary.txt"
    try:
        summary_path.write_text(report, encoding="utf-8")
        print("\n[written] {}".format(summary_path))
    except Exception as e:
        print("[warn] could not write summary file: {}".format(e))

    print("\nAll results for this run are in: {}".format(run_dir.resolve()))
    print("  head_<size>.pt  -> the files you can ship")
    print("  head_<size>/    -> Ultralytics training curves, plots, and logs")
    print("  training_summary.txt")


if __name__ == "__main__":
    main()
