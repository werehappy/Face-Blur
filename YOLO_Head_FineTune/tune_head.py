r"""
tune_head.py -- search hyperparameters for the head model, producing the
best_hyperparameters.yaml that train_all.py --hyp consumes.

Runs the ultralytics tuner: N short trainings with mutated hyperparameters,
scored by validation fitness. Output lands in <out>/tune/:
    best_hyperparameters.yaml   <- feed this to train_all.py --hyp
    tune_results.csv            <- every trial's numbers
    tune_fitness.png            <- search progress

USAGE (3060-class GPU):
    python tune_head.py --data head_dataset/data.yaml
        (10 trials x 25 epochs at imgsz 960 -- deployment-matched, a long
         overnight run)
    python tune_head.py --data head_dataset/data.yaml --imgsz 640 --iterations 15
        (faster proxy search; sanity-check scale/mosaic picks at 960 after)

Then train with the result:
    python train_all.py --data head_dataset/data.yaml ^
        --hyp runs/tune_<date>/tune/best_hyperparameters.yaml ^
        --baseline-recall 0.737

NOTES
  * Trials start from lr0=0.01 / warmup_epochs=1.0 (the stable fine-tuning
    values), NOT the ultralytics 0.03 default -- otherwise every trial
    re-inherits the LR that caused the epoch-1 best.pt problem.
  * The default searches at 960 because this dataset's weakness is small
    top-view heads: a 640 search shrinks exactly the objects that matter and
    optimizes scale-linked augmentation for the wrong pixel sizes.
  * 10 trials is a small, noisy sample. After the run, open tune_results.csv
    and confirm the best trial beats trial 1 (the near-default start) by more
    than trial-to-trial noise -- if it doesn't, tuning didn't matter on this
    data; keep the defaults and say so in the paper.
  * The tuner scores on the VALIDATION split. head_test stays untouched:
    tune -> one final train -> one test evaluation. Do not iterate against
    the test set.
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
    ap.add_argument("--imgsz", type=int, default=960,
                    help="search resolution (default 960 = matches deployment "
                         "HEAD_INFER_IMGSZ, so scale-linked picks transfer directly; "
                         "pass 640 for a ~2x faster proxy search whose scale/mosaic "
                         "picks must be sanity-checked at 960)")
    ap.add_argument("--epochs", type=int, default=25,
                    help="epochs per trial (default 25; must comfortably clear the "
                         "1-epoch warmup so trials measure real training)")
    ap.add_argument("--iterations", type=int, default=10,
                    help="number of trials (default 10 at 960; raise to ~15 if "
                         "searching at 640)")
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
    # IMPORTANT: ultralytics' Tuner resolves RELATIVE project paths against its
    # own default runs root (runs/detect/...), which nests everything under
    # runs/detect/<your-relative-path> and leaves the intended folder empty.
    # An absolute path is used verbatim.
    project_dir = run_dir.resolve()

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
        project=str(project_dir),
        name="tune",
        exist_ok=True,
    )

    best = project_dir / "tune" / "best_hyperparameters.yaml"
    print("\nDone. If the search succeeded, the tuned values are at:")
    print("  {}".format(best.resolve()))
    print("\nNext:")
    print("  python train_all.py --data {} --hyp {} --baseline-recall <your current recall>".format(
        args.data, best))


if __name__ == "__main__":
    main()
