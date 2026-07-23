r"""
tune_fusion.py -- sweep the person->head RESCUE parameters (person model, alpha,
beta, floor) for each head model and report BOTH the best config per person
model (to see the recall/precision trade as the person detector scales) AND the
overall best config per head model.

Why this exists
---------------
The rescue parameters must be selected on the VALIDATION split, never on the
held-out test set. Selecting them on test (running test twice and keeping the
better setting) turns the test set into a tuning set and makes the reported
test number optimistic. Run this on val, freeze the winner, then run
eval_compare.py on test ONCE with those values.

    python tune_fusion.py --data head_dataset/data.yaml --split val ^
        --model head_n:runs/.../head_n/weights/best.pt ^
        --model head_s:runs/.../head_s/weights/best.pt ^
        --model head_m:runs/.../head_m/weights/best.pt ^
        --fuse-person yolo11n.pt,yolo11s.pt,yolo11m.pt ^
        --imgsz 960 --conf 0.25

Person model as a 4th axis
--------------------------
The prior's quality depends on the person detector: a stronger person model
finds more bodies (small / occluded / side-on) -> more valid head regions -> more
weak head detections can be rescued, but also more spurious regions in clutter.
So the person model (--fuse-person, comma list of n/s/m) is swept too. Each
person model needs its OWN inference pass; alpha/beta/floor are then free
in-memory re-scorings. Cost = (#person models) person passes + (#head models)
head passes, and the whole grid is evaluated in memory.

Efficiency
----------
Each head model is run ONCE at the lowest floor, collecting every candidate box;
each person model is run ONCE. Every (person, alpha, beta, floor) combination is
scored purely in memory (same matcher/AP as eval_compare).

Selection metric
----------------
Because the augmentation exists to raise RECALL at the operating point under the
privacy asymmetry (a missed head is a real failure, a spurious blur is
cosmetic), the default selection metric is F-beta with beta=2 (recall weighted
4x precision) at --conf. Change with --select {f2,f1,recall_at_prec,map50}.
--min-precision-style guard: --precision-drop keeps precision within a margin of
the no-rescue baseline (default 0.05; set 1.0 to disable).

Note: this sweeps the eval's fusion pathway (head_from_person geometry, w_frac/
aspect). If you have aligned the eval geometry to the app's _person_to_head, the
chosen parameters transfer to the app; otherwise they are correct for the paper
tables but only approximate for the deployed app.
"""

import argparse
import os
import sys
from collections import defaultdict

# Reuse the exact matcher / metrics / geometry / IO from eval_compare so the
# swept parameters mean the same thing when you later run eval_compare on test.
import eval_compare as ec

IMG_EXTS = getattr(ec, "IMG_EXTS", [".jpg", ".jpeg", ".png", ".bmp"])


def parse_args():
    ap = argparse.ArgumentParser(description="Sweep rescue params per head model on val.")
    ap.add_argument("--data", required=True, help="data.yaml (use your VAL split, not test)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--model", action="append", dest="models", type=ec.parse_model,
                    default=[], help="name:weights (repeatable), e.g. head_s:.../best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25, help="operating confidence")
    ap.add_argument("--device", default=None)
    ap.add_argument("--geom-iou", type=float, default=0.5, help="match IoU (matches eval_compare)")
    # prior source: now a comma list -> a 4th sweep axis (each needs its own
    # person-inference pass, unlike alpha/beta/floor which are free in memory).
    ap.add_argument("--fuse-person", default="yolo11n.pt,yolo11s.pt,yolo11m.pt",
                    help="comma list of person models to compare as the prior "
                         "source (default n,s,m)")
    ap.add_argument("--person-class", type=int, default=0)
    ap.add_argument("--geom-w-frac", type=float, default=0.40)
    ap.add_argument("--geom-aspect", type=float, default=1.10)
    ap.add_argument("--geom-conf-floor", type=float, default=0.05)
    # sweep grids
    ap.add_argument("--alphas", default="0.5,1.0,1.5,2.0",
                    help="comma list of alpha (boost strength) values")
    ap.add_argument("--betas", default="0.0,0.5",
                    help="comma list of beta (unclaimed-prior injection) values")
    ap.add_argument("--floors", default="0.05,0.10,0.15",
                    help="comma list of candidate floors")
    # selection
    ap.add_argument("--select", default="f2",
                    choices=["f2", "f1", "recall_at_prec", "map50"],
                    help="metric to maximize when picking the best config")
    ap.add_argument("--precision-drop", type=float, default=0.05,
                    help="max precision the winner may give up vs the no-rescue "
                         "baseline (guards against recall-at-any-cost). Set 1.0 to disable.")
    ap.add_argument("--no-map", action="store_true",
                    help="skip mAP in the report (faster; mAP is not the default selector)")
    ap.add_argument("--out", default="fusion_sweep.csv",
                    help="write the full grid to this CSV")
    return ap.parse_args()


def _grid(s):
    return [float(x) for x in s.split(",") if x.strip()]


def fbeta(recall, precision, beta):
    if recall != recall or precision != precision or (precision + recall) == 0:
        return float("nan")
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else float("nan")


def resolve_dirs(args):
    import yaml
    with open(args.data) as f:
        dcfg = yaml.safe_load(f) or {}
    base = dcfg.get("path") or os.path.dirname(os.path.abspath(args.data))
    images_dir = args.images_dir or os.path.join(base, "images", args.split)
    labels_dir = os.path.join(base, "labels", args.split)
    if not os.path.isdir(images_dir):
        sys.exit("[ERROR] split images dir not found: {} "
                 "(is --split correct? your test folder is literally named 'val')"
                 .format(images_dir))
    imgs = []
    for root, _d, fnames in os.walk(images_dir):
        for nm in fnames:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(root, nm))
    if not imgs:
        sys.exit("[ERROR] no images in {}".format(images_dir))
    return images_dir, labels_dir, sorted(imgs)


def build_priors(args, person_weights, imgs):
    """One person pass with a SPECIFIC person model -> {abs_path: [(conf, region)]}."""
    from ultralytics import YOLO
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        torch, cuda = None, False
    print("[prior] person model {} over {} images...".format(person_weights, len(imgs)))
    person = YOLO(person_weights)
    priors = {}
    for p in imgs:
        res = person.predict(p, imgsz=args.imgsz,
                             conf=max(0.05, args.geom_conf_floor),
                             device=args.device, classes=[args.person_class],
                             verbose=False)[0]
        regs = []
        if res.boxes is not None and len(res.boxes):
            confs = res.boxes.conf.detach().cpu().tolist()
            xyxys = res.boxes.xyxy.detach().cpu().tolist()
            for c, pb in zip(confs, xyxys):
                regs.append((float(c),
                             ec.head_from_person(tuple(pb), args.geom_w_frac, args.geom_aspect)))
        priors[os.path.abspath(p)] = regs
        del res
        if cuda:
            torch.cuda.empty_cache()
    del person
    ec._free_cuda()
    return priors


def collect_candidates(args, name, weights, imgs, images_dir, labels_dir, min_floor):
    """Run one head model ONCE at min_floor, return (cand_by_img, gts_by_img).
    cand_by_img[key] = [(raw_conf, xyxy), ...] for every candidate >= min_floor."""
    from ultralytics import YOLO
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        torch, cuda = None, False
    print("[collect] {} at floor {} over {} images...".format(name, min_floor, len(imgs)))
    model = YOLO(weights)
    cand_by_img, gts_by_img = {}, {}
    for p in imgs:
        key = os.path.abspath(p)
        res = model.predict(p, imgsz=args.imgsz, conf=min_floor,
                            device=args.device, verbose=False)[0]
        cand = []
        if res.boxes is not None and len(res.boxes):
            confs = res.boxes.conf.detach().cpu().tolist()
            xyxys = res.boxes.xyxy.detach().cpu().tolist()
            for c, b in zip(confs, xyxys):
                cand.append((float(c), tuple(b)))
        cand_by_img[key] = cand
        gts_by_img[key] = ec.load_gt_boxes(p, images_dir, labels_dir, res.orig_shape)
        del res
        if cuda:
            torch.cuda.empty_cache()
    del model
    ec._free_cuda()
    return cand_by_img, gts_by_img


def fuse_in_memory(cand_by_img, priors, alpha, beta, floor):
    """Apply rescue+optional-injection to pre-collected candidates. Pure Python,
    no inference. Mirrors eval_compare._apply_prior + the fused floor logic."""
    out = {}
    for key, cands in cand_by_img.items():
        regs = priors.get(key, [])          # [(person_conf, region_xyxy), ...]
        # ec._prior_overlap expects (conf, box) pairs, matching eval_compare.
        reg_pairs = regs
        preds = []
        kept_boxes = []
        for c, box in cands:
            if c < floor:
                continue  # candidate floor
            ov = ec._prior_overlap(box, reg_pairs) if reg_pairs else 0.0
            preds.append((min(1.0, c * (1.0 + alpha * ov)), box))
            kept_boxes.append(box)
        if beta > 0:
            for pc, pb in regs:
                if all(ec.iou_xyxy(pb, hb) < 0.30 for hb in kept_boxes):
                    preds.append((pc * beta, pb))
        out[key] = preds
    return out


def score(preds_by_img, gts_by_img, keys, conf, iou, want_map):
    recall, precision, f1, _ = ec.op_metrics(preds_by_img, gts_by_img, keys, conf, iou)
    map50 = float("nan")
    if want_map:
        map50 = ec.ap_at(preds_by_img, gts_by_img, keys, 0.5)
    return recall, precision, f1, map50


def main():
    args = parse_args()
    if not args.models:
        sys.exit("[ERROR] pass at least one --model name:weights")
    if args.split.lower() in ("test",) or "test" in os.path.basename(os.path.dirname(args.data)).lower():
        print("[warn] this looks like it may be your TEST data. Tune on VAL; "
              "run eval_compare on test only ONCE with the frozen winner.\n")

    alphas, betas, floors = _grid(args.alphas), _grid(args.betas), _grid(args.floors)
    person_models = [p.strip() for p in args.fuse_person.split(",") if p.strip()]
    min_floor = min(floors)
    want_map = not args.no_map

    images_dir, labels_dir, imgs = resolve_dirs(args)
    keys = [os.path.abspath(p) for p in imgs]

    # Build priors ONCE per person model (each needs its own inference pass).
    priors_by_person = {}
    for pw in person_models:
        if not os.path.exists(pw) and not pw.startswith("yolo"):
            print("[warn] person model not found and not an auto-download name: {}".format(pw))
        priors_by_person[pw] = build_priors(args, pw, imgs)

    rows = []                       # every grid point
    best_per_model = {}             # head name -> overall best row
    best_per_model_person = {}      # head name -> {person -> best row}

    for name, weights in args.models:
        if not os.path.exists(weights):
            print("[warn] skipping {}: weights not found ({})".format(name, weights))
            continue
        cand_by_img, gts_by_img = collect_candidates(
            args, name, weights, imgs, images_dir, labels_dir, min_floor)

        # no-rescue baseline (candidates thresholded at --conf, no prior)
        base_preds = {k: [(c, b) for c, b in v if c >= args.conf]
                      for k, v in cand_by_img.items()}
        b_r, b_p, b_f1, b_map = score(base_preds, gts_by_img, keys,
                                      args.conf, args.geom_iou, want_map)
        print("\n=== {} ===".format(name))
        print("  baseline (no rescue): R={:.3f} P={:.3f} F1={:.3f} mAP50={}".format(
            b_r, b_p, b_f1, "{:.3f}".format(b_map) if b_map == b_map else "-"))
        rows.append(dict(model=name, person="-", alpha="-", beta="-", floor="-",
                         recall=b_r, precision=b_p, f1=b_f1, map50=b_map,
                         f2=fbeta(b_r, b_p, 2), baseline=True))

        prec_floor = (b_p - args.precision_drop) if args.precision_drop < 1.0 else -1.0

        def selval(r, p, f1, mp, f2):
            if args.select == "f2":
                return f2
            if args.select == "f1":
                return f1
            if args.select == "map50":
                return mp
            return r  # recall_at_prec

        overall_best = None
        per_person = {}
        for pw in person_models:
            priors = priors_by_person[pw]
            person_best = None
            for fl in floors:
                for al in alphas:
                    for be in betas:
                        fused = fuse_in_memory(cand_by_img, priors, al, be, fl)
                        r, p, f1, mp = score(fused, gts_by_img, keys,
                                             args.conf, args.geom_iou, want_map)
                        f2 = fbeta(r, p, 2)
                        row = dict(model=name, person=_short(pw), alpha=al, beta=be,
                                   floor=fl, recall=r, precision=p, f1=f1,
                                   map50=mp, f2=f2, baseline=False)
                        rows.append(row)
                        sel = selval(r, p, f1, mp, f2)
                        if p == p and prec_floor >= 0 and p < prec_floor:
                            continue
                        if sel == sel:
                            cand_row = dict(row); cand_row["_sel"] = sel
                            if person_best is None or sel > person_best["_sel"]:
                                person_best = cand_row
                            if overall_best is None or sel > overall_best["_sel"]:
                                overall_best = cand_row
            if person_best is not None:
                per_person[_short(pw)] = person_best
                dR = person_best["recall"] - b_r
                dP = person_best["precision"] - b_p
                print("  [{}]  best alpha={} beta={} floor={}  ->  "
                      "R={:.3f} ({:+.3f}) P={:.3f} ({:+.3f}) F2={:.3f}".format(
                          _short(pw), person_best["alpha"], person_best["beta"],
                          person_best["floor"], person_best["recall"], dR,
                          person_best["precision"], dP, person_best["f2"]))
            else:
                print("  [{}]  no config passed the precision guard".format(_short(pw)))

        best_per_model_person[name] = per_person
        if overall_best is None:
            best_per_model[name] = dict(model=name, person="-", alpha="-",
                                        beta="-", floor="-", recall=b_r,
                                        precision=b_p, f1=b_f1, map50=b_map,
                                        f2=fbeta(b_r, b_p, 2))
            print("  [warn] no config satisfied the precision guard; baseline is best.")
        else:
            best_per_model[name] = overall_best
            print("  OVERALL BEST  person={} alpha={} beta={} floor={}  ->  "
                  "R={:.3f} P={:.3f} F1={:.3f} F2={:.3f} mAP50={}".format(
                      overall_best["person"], overall_best["alpha"],
                      overall_best["beta"], overall_best["floor"],
                      overall_best["recall"], overall_best["precision"],
                      overall_best["f1"], overall_best["f2"],
                      "{:.3f}".format(overall_best["map50"]) if overall_best["map50"] == overall_best["map50"] else "-"))

    # ---- write full grid CSV ----
    import csv
    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "person", "alpha", "beta", "floor", "recall",
                     "precision", "f1", "f2", "map50", "baseline"])
        for r in rows:
            wr.writerow([r["model"], r["person"], r["alpha"], r["beta"], r["floor"],
                         _f(r["recall"]), _f(r["precision"]), _f(r["f1"]),
                         _f(r["f2"]), _f(r["map50"]), int(r["baseline"])])

    # ---- per-person-model breakdown ----
    print("\n" + "=" * 78)
    print("PER PERSON-MODEL WINNERS  (selection: {}; conf={})".format(args.select, args.conf))
    print("=" * 78)
    print("{:<8} {:<9} {:>6} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6}".format(
        "head", "person", "alpha", "beta", "floor", "recall", "prec", "F1", "F2"))
    for name, pp in best_per_model_person.items():
        for person, b in pp.items():
            print("{:<8} {:<9} {:>6} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6}".format(
                name, person, str(b["alpha"]), str(b["beta"]), str(b["floor"]),
                _f(b["recall"]), _f(b["precision"]), _f(b["f1"]), _f(b["f2"])))

    # ---- overall winner per head model ----
    print("\n" + "=" * 78)
    print("OVERALL BEST CONFIG PER HEAD MODEL")
    print("=" * 78)
    print("{:<8} {:<9} {:>6} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6}".format(
        "head", "person", "alpha", "beta", "floor", "recall", "prec", "F1", "mAP50"))
    for name, b in best_per_model.items():
        print("{:<8} {:<9} {:>6} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6}".format(
            name, str(b.get("person", "-")), str(b["alpha"]), str(b["beta"]),
            str(b["floor"]), _f(b["recall"]), _f(b["precision"]), _f(b["f1"]),
            _f(b.get("map50", float("nan")))))
    print("\nFull grid written to {}".format(args.out))
    print("\nNext step: run eval_compare.py on TEST ONCE per head model with the "
          "overall-best\n  --fuse-person, --fuse-alpha, --fuse-beta, --fuse-floor above, "
          "and report those numbers.")


def _short(path):
    """yolo11n.pt -> yolo11n; keep a readable tag for tables/logs."""
    return os.path.splitext(os.path.basename(path))[0]


def _f(x):
    try:
        return "{:.4f}".format(x) if x == x else ""
    except Exception:
        return ""


if __name__ == "__main__":
    main()
