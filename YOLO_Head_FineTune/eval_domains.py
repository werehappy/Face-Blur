r"""
eval_domains.py -- measure head-model quality PER DOMAIN (helmet-cam vs catwalk).

A pooled mAP can look healthy while one domain quietly underperforms. This runs
your model on a labeled split (val or test) but splits the images by domain
(using clip-name prefixes) and reports recall / precision / mAP for each domain
separately -- plus the number of labeled heads (instances) per domain, so the
instance imbalance that tilts training is visible.

For a privacy blur, RECALL is the number that matters (a miss = a leak).

USAGE:
    conda activate yolotrain
    python eval_domains.py --data head_dataset/data.yaml --model head_s.pt
    python eval_domains.py --data head_dataset/data.yaml --model runs/detect/head_m/weights/best.pt --split val
    # custom domain rules (name:prefix1,prefix2 ; matched by clip-name startswith):
    python eval_domains.py --data head_dataset/data.yaml --model head_s.pt ^
        --domain catwalk:new_clip,cat_ --domain helmet:clip

Defaults assume your naming scheme: catwalk = new_clip*/cat_*, helmet = clip*.

Evaluate at the SAME imgsz the app runs (HEAD_INFER_IMGSZ, default 960).
"""

import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")


def clip_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def parse_domains(entries):
    """entries like ['catwalk:new_clip,cat_', 'helmet:clip'] ->
    ordered list of (name, [prefixes])."""
    out = []
    for e in entries:
        if ":" not in e:
            print("[warn] ignoring bad --domain '{}' (need name:prefix)".format(e))
            continue
        name, prefixes = e.split(":", 1)
        out.append((name.strip(), [p.strip() for p in prefixes.split(",") if p.strip()]))
    return out


def domain_of(clip, domains):
    for name, prefixes in domains:
        if any(clip.startswith(p) for p in prefixes):
            return name
    return "(other)"


def count_instances(img_path, images_root, labels_root):
    """Number of labeled boxes for an image (0 for empty/missing label)."""
    rel = os.path.relpath(img_path, images_root)
    stem = os.path.splitext(rel)[0]
    lbl = os.path.join(labels_root, stem + ".txt")
    if not os.path.exists(lbl):
        return 0
    try:
        with open(lbl, "r") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description="Per-domain head-model evaluation.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", required=True, help="weights (.pt)")
    ap.add_argument("--split", default="val", help="split to evaluate: val or test (default val)")
    ap.add_argument("--imgsz", type=int, default=960, help="inference size (match the app, default 960)")
    ap.add_argument("--conf", type=float, default=None, help="confidence threshold (default: ultralytics default)")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--domain", action="append", default=None,
                    help="domain rule name:prefix1,prefix2 (repeatable). "
                         "Default: catwalk:new_clip,cat_ and helmet:clip")
    ap.add_argument("--images-dir", default=None,
                    help="override the split images dir (else <data path>/images/<split>)")
    args = ap.parse_args()

    try:
        import yaml
        from ultralytics import YOLO
    except Exception as e:
        sys.exit("[ERROR] need ultralytics + pyyaml: {}".format(e))

    if not os.path.exists(args.data):
        sys.exit("[ERROR] data yaml not found: {}".format(args.data))
    if not os.path.exists(args.model):
        sys.exit("[ERROR] model not found: {}".format(args.model))

    with open(args.data, "r") as f:
        dcfg = yaml.safe_load(f) or {}
    base = dcfg.get("path") or os.path.dirname(os.path.abspath(args.data))
    names = dcfg.get("names", {0: "head"})

    images_dir = args.images_dir or os.path.join(base, "images", args.split)
    labels_dir = os.path.join(base, "labels", args.split)
    if not os.path.isdir(images_dir):
        sys.exit("[ERROR] split images dir not found: {}\n"
                 "        pass --images-dir, or check --split.".format(images_dir))

    domains = parse_domains(args.domain) if args.domain else \
        [("catwalk", ["new_clip", "cat_"]), ("helmet", ["clip"])]

    # Bucket images by domain, and tally labeled instances per domain.
    imgs = []
    for root, _d, fnames in os.walk(images_dir):
        for nm in fnames:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(root, nm))
    if not imgs:
        sys.exit("[ERROR] no images in {}".format(images_dir))

    buckets = defaultdict(list)
    frames = defaultdict(int)
    instances = defaultdict(int)
    for p in imgs:
        dom = domain_of(clip_of(p), domains)
        buckets[dom].append(p)
        frames[dom] += 1
        instances[dom] += count_instances(p, images_dir, labels_dir)

    print("=" * 68)
    print("Split '{}': {} images. Model: {}  imgsz={}".format(
        args.split, len(imgs), args.model, args.imgsz))
    print("Domain composition (labeled heads = what the loss actually sees):")
    for dom in list(buckets.keys()):
        print("  {:<10} {:>5} frames  {:>6} labeled heads".format(
            dom, frames[dom], instances[dom]))
    print("=" * 68)

    model = YOLO(args.model)
    results_rows = []
    tmpdir = tempfile.mkdtemp(prefix="domeval_")

    # Evaluate each domain, plus an ALL row for the pooled baseline.
    eval_sets = [(dom, buckets[dom]) for dom in buckets]
    eval_sets.append(("ALL", imgs))

    for dom, paths in eval_sets:
        if not paths:
            continue
        list_txt = os.path.join(tmpdir, "imgs_{}.txt".format(dom.strip("()")))
        with open(list_txt, "w") as f:
            for p in paths:
                f.write(os.path.abspath(p) + "\n")
        tmp_yaml = os.path.join(tmpdir, "data_{}.yaml".format(dom.strip("()")))
        with open(tmp_yaml, "w") as f:
            # ultralytics requires both train and val keys even for val-only runs;
            # point train at the same list (it isn't trained, only validated).
            yaml.safe_dump({"path": base, "train": list_txt,
                            "val": list_txt, "names": names}, f)

        kw = dict(data=tmp_yaml, imgsz=args.imgsz, device=args.device,
                  split="val", verbose=False, plots=False)
        if args.conf is not None:
            kw["conf"] = args.conf
        try:
            m = model.val(**kw)
            row = (dom, len(paths),
                   float(getattr(m.box, "mr", float("nan"))),
                   float(getattr(m.box, "mp", float("nan"))),
                   float(getattr(m.box, "map50", float("nan"))),
                   float(getattr(m.box, "map", float("nan"))))
        except Exception as e:
            print("[warn] eval failed for {}: {}".format(dom, e))
            row = (dom, len(paths), float("nan"), float("nan"),
                   float("nan"), float("nan"))
        results_rows.append(row)

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("PER-DOMAIN RESULTS (imgsz={})".format(args.imgsz))
    print("-" * 68)
    print("{:<10} {:>7} {:>9} {:>10} {:>9} {:>9}".format(
        "domain", "images", "recall", "precision", "mAP50", "mAP5095"))
    for (dom, n, mr, mp, m50, m95) in results_rows:
        def f(v):
            return "{:.3f}".format(v) if v == v else "  -  "  # v==v filters NaN
        print("{:<10} {:>7} {:>9} {:>10} {:>9} {:>9}".format(
            dom, n, f(mr), f(mp), f(m50), f(m95)))
    print("-" * 68)
    print("RECALL is the metric that matters for a privacy blur (a miss = leak).")
    print("Watch for one domain's recall trailing the other. If helmet trails")
    print("catwalk, the instance imbalance above is the likely cause -- rebalance")
    print("by adding helmet-cam heads (not frames) rather than splitting models.")
    print("=" * 68)


if __name__ == "__main__":
    main()
