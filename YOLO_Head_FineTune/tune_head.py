r"""
tune_head.py -- search hyperparameters for the head model, producing the
best_hyperparameters.yaml that train_all.py --hyp consumes.

Runs the ultralytics tuner: N short trainings with mutated hyperparameters,
scored by validation fitness. Output lands in <out>/tune/:
    best_hyperparameters.yaml   <- feed this to train_all.py --hyp
    tune_results.csv            <- every trial's numbers
    tune_fitness.png            <- search progress

USAGE (2080/3060-class GPU, overnight budget):
    python tune_head.py --data head_dataset/data.yaml
    python tune_head.py --data head_dataset/data.yaml --imgsz 960 --iterations 10

Then train with the result:
    python train_all.py --data head_dataset/data.yaml ^
        --hyp runs/tune_<date>/tune/best_hyperparameters.yaml ^
        --baseline-recall 0.737

NOTES
  * Trials start from lr0=0.01 / warmup_epochs=1.0 (the stable fine-tuning
    values), NOT the ultralytics 0.03 default -- otherwise every trial
    re-inherits the LR that caused the epoch-1 best.pt problem.
  * The tuner scores on the VALIDATION split. head_test stays untouched:
    tune -> one final train -> one test evaluation. Do not iterate against
    the test set.
  * --imgsz 640 is a faster proxy search; scale-related picks should be
    sanity-checked at 960. If time allows, search at 960 directly with
    fewer iterations.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Hyperparameter search for the head model.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="base weights to tune around (default yolo11n.pt; tune the "
                         "size you plan to ship)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="search resolution (default 640 = fast proxy; use 960 to "
                         "match deployment exactly, with fewer --iterations)")
    ap.add_argument("--epochs", type=int, default=25,
                    help="epochs per trial (default 25; must comfortably clear the "
                         "1-epoch warmup so trials measure real training)")
    ap.add_argument("--iterations", type=int, default=15,
                    help="number of trials (default 15)")
    ap.add_argument("--batch", default="-1", help="batch (-1 = auto-size to VRAM)")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--out", default="runs", help="root results folder (default: runs)")
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
        batch = args.batch

    run_dir = Path(args.out) / "tune_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Tuning {} at imgsz={} : {} iterations x {} epochs".format(
        args.model, args.imgsz, args.iterations, args.epochs))
    print("Output: {}".format(run_dir.resolve()))

    model = YOLO(args.model)
    model.tune(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        iterations=args.iterations,
        batch=batch,
        device=args.device,
        # stable starting point -- trials mutate FROM here, not from 0.03
        lr0=0.01,
        warmup_epochs=1.0,
        cos_lr=True,
        optimizer="AdamW",
        seed=0,
        # keep every trial cheap: no per-trial val plots or checkpoint saves
        plots=False,
        save=False,
        val=True,
        project=str(run_dir),
        name="tune",
        exist_ok=True,
    )

    best = run_dir / "tune" / "best_hyperparameters.yaml"
    print("\nDone. If the search succeeded, the tuned values are at:")
    print("  {}".format(best.resolve()))
    print("\nNext:")
    print("  python train_all.py --data {} --hyp {} --baseline-recall <your current recall>".format(
        args.data, best))


if __name__ == "__main__":
    main()
