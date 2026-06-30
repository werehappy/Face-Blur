r"""
train_head.py -- maximum-accuracy head training on a GTX 1060 3GB.

Strategy (time is free, VRAM is the constraint, data is the ceiling):
  PHASE 1  hyperparameter search at LOW res (fast, fits VRAM, many iterations)
           -> finds good augmentation + learning-rate values.
  PHASE 2  one LONG final run at HIGH res with those values (slow via the
           Windows shared-memory spill, but that's fine if it runs all weekend).
  PHASE 3  evaluate on a held-out TEST split (per-domain recall is what matters).

PREREQUISITES (these set the accuracy ceiling -- the script cannot fix them):
  * Split train/val/test by CLIP, never by random frame. Adjacent frames are
    near-identical; a random split leaks them across train/val and your metrics
    lie. data.yaml should point to separate train/ val/ test/ image folders.
  * Keep a real TEST set per domain (helmet-cam and catwalk separately) that
    never touches training.
  * Balance domains by HEAD COUNT, not frame count (catwalk's 8+ heads/frame
    can swamp helmet-cam's <=3).
  * Include your illuminator NEGATIVE frames (heads labeled, light unlabeled).

USAGE (in your training env):
    conda activate yolotrain
    python train_head.py --data head_dataset/data.yaml            # full plan
    python train_head.py --data head_dataset/data.yaml --no-tune  # skip search
    python train_head.py --data head_dataset/data.yaml --final-imgsz 960  # faster app

AFTER TRAINING:
    * best.pt is at runs/detect/<name>/weights/best.pt -- drop it next to the app
      as head.pt.
    * Set HEAD_INFER_IMGSZ in face_blur.py to the SAME value as --final-imgsz,
      or the illuminator/scale mismatch comes back.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Two-phase head training for a 1060 3GB.")
    ap.add_argument("--data", required=True, help="path to data.yaml (train/val/test).")
    ap.add_argument("--model", default="yolo11s.pt",
                    help="base weights (default yolo11s.pt; 's' is the accuracy/"
                         "app-speed sweet spot, since head.pt now carries all "
                         "hard-head recall and runs on CPU for many users).")
    ap.add_argument("--name", default="head_final", help="run name.")
    # Phase 1 (tune)
    ap.add_argument("--no-tune", action="store_true", help="skip the search phase.")
    ap.add_argument("--tune-imgsz", type=int, default=640,
                    help="search resolution (default 640; small = fast = fits VRAM).")
    ap.add_argument("--tune-batch", type=int, default=16, help="search batch (default 16).")
    ap.add_argument("--tune-epochs", type=int, default=20, help="epochs per search trial.")
    ap.add_argument("--iterations", type=int, default=15, help="number of search trials.")
    # Phase 2 (final)
    ap.add_argument("--final-imgsz", type=int, default=1280,
                    help="final resolution (default 1280; biggest lever for small/"
                         "distant heads). MUST match the app's HEAD_INFER_IMGSZ.")
    ap.add_argument("--final-batch", type=int, default=8,
                    help="final batch (default 8; spills to RAM at 1280 on 3GB, "
                         "which is slow but fine if time is free).")
    ap.add_argument("--epochs", type=int, default=300, help="final-run epoch ceiling.")
    ap.add_argument("--patience", type=int, default=60,
                    help="early stop after this many epochs with no val gain.")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'.")
    ap.add_argument("--no-test", action="store_true", help="skip the test-split eval.")
    return ap.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        sys.exit("[ERROR] ultralytics not importable: {}\n"
                 "        conda activate yolotrain ; pip install -U ultralytics".format(e))

    if not Path(args.data).exists():
        sys.exit("[ERROR] data yaml not found: {}".format(args.data))

    best_hyp = {}

    # ---- PHASE 1: hyperparameter search (low res, fast, many trials) ---------
    if not args.no_tune:
        print("\n========== PHASE 1: hyperparameter search "
              "(imgsz={}, {} trials x {} epochs) ==========".format(
                  args.tune_imgsz, args.iterations, args.tune_epochs))
        tune_name = args.name + "_tune"
        model = YOLO(args.model)
        try:
            model.tune(
                data=args.data,
                epochs=args.tune_epochs,
                iterations=args.iterations,
                imgsz=args.tune_imgsz,
                batch=args.tune_batch,
                optimizer="AdamW",
                device=args.device,
                seed=0,
                plots=False,
                save=False,
                val=True,
                name=tune_name,
            )
        except Exception as e:
            print("[WARN] tune phase failed ({}). Continuing with default "
                  "hyperparameters.".format(e))

        # Load the evolved hyperparameters, if the search produced them.
        hyp_path = Path("runs/detect") / tune_name / "best_hyperparameters.yaml"
        if not hyp_path.exists():
            hyp_path = Path("runs/detect/tune/best_hyperparameters.yaml")
        if hyp_path.exists():
            try:
                import yaml
                with open(hyp_path, "r") as f:
                    best_hyp = yaml.safe_load(f) or {}
                print("[ok] loaded tuned hyperparameters from {}".format(hyp_path))
            except Exception as e:
                print("[WARN] could not read {} ({}); using defaults.".format(hyp_path, e))
        else:
            print("[WARN] no best_hyperparameters.yaml found; using defaults.")

    # Keep only augmentation / optimization keys; drop anything that would
    # conflict with the explicit final-run args below.
    drop = {"epochs", "imgsz", "batch", "device", "data", "patience", "name",
            "model", "save", "val", "plots", "iterations"}
    final_hyp = {k: v for k, v in best_hyp.items() if k not in drop}

    # ---- PHASE 2: long final run at high resolution -------------------------
    print("\n========== PHASE 2: final run "
          "(model={}, imgsz={}, batch={}, epochs<= {} patience {}) ==========".format(
              args.model, args.final_imgsz, args.final_batch, args.epochs, args.patience))
    print("    (At imgsz=1280 on 3GB this spills into system RAM and is slow -- "
          "expected. Check the ETA after epoch 1 and lower --epochs if it won't "
          "fit your weekend.)")
    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        imgsz=args.final_imgsz,
        batch=args.final_batch,
        epochs=args.epochs,
        patience=args.patience,
        cos_lr=True,            # cosine LR decay -> slightly better final point
        close_mosaic=15,        # disable mosaic for the last 15 epochs to clean up
        device=args.device,
        seed=0,
        deterministic=True,
        cache=False,            # caching uses system RAM, which the spill needs
        plots=True,
        name=args.name,
    )
    train_kwargs.update(final_hyp)   # tuned augmentation / lr values
    results = model.train(**train_kwargs)

    weights = Path("runs/detect") / args.name / "weights" / "best.pt"
    print("\n[ok] final weights: {}".format(weights))

    # ---- PHASE 3: held-out test evaluation ----------------------------------
    if not args.no_test:
        print("\n========== PHASE 3: test-split evaluation ==========")
        try:
            best = YOLO(str(weights))
            # Evaluate at the SAME imgsz the app will run at.
            metrics = best.val(data=args.data, split="test",
                               imgsz=args.final_imgsz, device=args.device)
            print("    For a privacy blur, RECALL is the number that matters "
                  "(a miss = a leak). Check recall on your hard cases, per domain.")
            print("    box recall: {}".format(getattr(metrics.box, "r", "n/a")))
            print("    box mAP50 : {}".format(getattr(metrics.box, "map50", "n/a")))
        except Exception as e:
            print("[WARN] test eval skipped ({}). Does data.yaml define a 'test' "
                  "split?".format(e))

    print("\nDONE.")
    print("Next: copy {} next to the app as head.pt, and set HEAD_INFER_IMGSZ={} "
          "in face_blur.py to match.".format(weights, args.final_imgsz))


if __name__ == "__main__":
    main()
