#!/usr/bin/env python3
r"""
eval_compare.py -- compare SEVERAL head models per domain and emit paper-ready
output (CSV + LaTeX table + Markdown table + a comparison figure).

This is the reporting layer that sits on top of eval_domains.py's conventions:
same data.yaml + split layout, same clip-prefix domain bucketing, same imgsz
960 default. Where eval_domains.py evaluates ONE model, this runs every model
you pass over every domain and lines them up for Section 6 of the paper.

For each model x domain it reports, at the app's operating point (conf 0.25 by
default) and matched imgsz:
    Recall (headline -- a miss is a privacy leak), Precision, F1
and, threshold-independent:
    mAP@0.5, mAP@0.5:0.95
plus labeled-head instance counts per domain (sample size + the imbalance signal).

Two val passes are used per model x domain: one at low confidence for correct
mAP, one at the operating confidence for Recall/Precision. Skip mAP with
--no-map for speed or for non-head-class baselines.

USAGE
-----
    conda activate yolotrain
    python eval_compare.py --data head_dataset/data.yaml --split test ^
        --model head_n:head_n.pt --model head_s:head_s.pt --model head_m:head_m.pt

    # custom domain rules (name:prefix1,prefix2 ; clip-name startswith), and a title
    python eval_compare.py --data head_dataset/data.yaml --split test ^
        --model head_s:head_s.pt ^
        --domain catwalk:new_clip,cat_ --domain helmet:clip ^
        --title "Base vs fine-tuned, held-aside clips"

OUTPUT (into --out-dir, default paper_eval/)
    eval_comparison.csv    every model x domain row
    eval_comparison.md     readable comparison table
    eval_comparison.tex    LaTeX (booktabs) recall matrix, ready to \input
    recall_by_domain.png   grouped bar chart: per-domain recall by model

GEOMETRIC BASELINE (COCO person -> estimated head)
---------------------------------------------------
A stock yolo11 outputs `person`, not heads, so ultralytics val cannot score it
against head ground truth. This script therefore ships its own operating-point
matcher and AP computation for that pathway: pass a COCO person model with
--geom-model and each detected person box is converted to an estimated head
box (top-center, sized from shoulder width -- Section 4.3 of the paper), then
matched against the head labels at IoU 0.5. The row lands in the same CSV /
Markdown / LaTeX / figure as the fine-tuned models.

    python eval_compare.py --data head_dataset/data.yaml --split test ^
        --model head_s:head_s.pt ^
        --geom-model person2head_n:yolo11n.pt ^
        --geom-w-frac 0.40 --geom-aspect 1.10

Set --geom-w-frac / --geom-aspect to the SAME constants the application uses
for its geometric fallback, otherwise you are evaluating a different baseline
than the one the runtime policy disables.
"""

import os as _os
# Must be set before torch initializes CUDA (torch is imported lazily below):
# reduces fragmentation ("N GiB reserved but unallocated") on long multi-model runs.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")


# --------------------------------------------------------------------------- #
# data bucketing (mirrors eval_domains.py so results are consistent)
# --------------------------------------------------------------------------- #
def clip_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def parse_domains(entries):
    out = []
    for e in entries or []:
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
    rel = os.path.relpath(img_path, images_root)
    stem = os.path.splitext(rel)[0]
    lbl = os.path.join(labels_root, stem + ".txt")
    if not os.path.exists(lbl):
        return 0
    try:
        with open(lbl) as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def parse_model(spec):
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("model must be name:weights, got '{}'".format(spec))
    return parts[0], parts[1]


# --------------------------------------------------------------------------- #
# person->head geometric baseline (COCO person model -> estimated head boxes)
# --------------------------------------------------------------------------- #
def head_from_person(box, w_frac, aspect):
    """Estimate a head box (xyxy) from a person box: top-center, head width =
    w_frac * person-box width (shoulder width proxy), height = aspect * width.
    Height is tied to the head's own width, not the person height, so partially
    visible / truncated bodies do not produce absurdly tall head boxes."""
    x1, y1, x2, _y2 = box
    hw = (x2 - x1) * w_frac
    hh = hw * aspect
    cx = (x1 + x2) / 2.0
    return (cx - hw / 2.0, y1, cx + hw / 2.0, y1 + hh)


IMG_SHAPES = {}  # abs_img_path -> (h, w), filled by load_gt_boxes for the
                 # border/interior split (needs frame dims to test edge proximity)


def load_gt_boxes(img_path, images_root, labels_root, shape):
    """Read YOLO-format labels for one image, return pixel xyxy boxes."""
    h, w = shape[0], shape[1]
    IMG_SHAPES[os.path.abspath(img_path)] = (h, w)
    rel = os.path.relpath(img_path, images_root)
    stem = os.path.splitext(rel)[0]
    lbl = os.path.join(labels_root, stem + ".txt")
    out = []
    if not os.path.exists(lbl):
        return out
    with open(lbl) as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5:
                continue
            _c, cx, cy, bw, bh = (float(v) for v in parts[:5])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_image(preds, gts, iou_thr):
    """Greedy conf-descending matching within one image.
    preds: [(conf, xyxy)]. Returns [(conf, is_tp)] for every prediction."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
    matched = [False] * len(gts)
    out = []
    for i in order:
        conf, box = preds[i]
        best, best_iou = -1, iou_thr
        for g, gt in enumerate(gts):
            if matched[g]:
                continue
            v = iou_xyxy(box, gt)
            if v >= best_iou:
                best, best_iou = g, v
        if best >= 0:
            matched[best] = True
            out.append((conf, True))
        else:
            out.append((conf, False))
    return out


def _resample_keys(keys, level, rng):
    """Return a resampled (with replacement) key list for one bootstrap draw.
      clip  -- resample whole CLIPS (the honest default: heads within a clip are
               correlated -- same person, lighting, consecutive frames -- so the
               independent unit is the clip, not the head or frame). With C
               clips, draw C clips with replacement and concatenate their keys.
      image -- resample images with replacement (ignores within-clip correlation;
               CIs will be too narrow -- provided only for comparison).
    Duplicate keys are intentional: op_metrics/ap_at iterate the list and so a
    doubled clip doubles its TP/FP/GT contribution, which is the correct
    bootstrap behaviour.
    """
    if level == "image":
        n = len(keys)
        return [keys[rng.randrange(n)] for _ in range(n)]
    # clip level
    by_clip = {}
    for k in keys:
        by_clip.setdefault(clip_of(k), []).append(k)
    clips = list(by_clip.keys())
    c = len(clips)
    out = []
    for _ in range(c):
        out.extend(by_clip[clips[rng.randrange(c)]])
    return out


def _percentile(sorted_vals, q):
    """Linear-interpolated percentile (q in [0,1]) of a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[lo]


def bootstrap_metric_cis(preds_by_img, gts_by_img, keys, conf, iou, n_boot,
                         level, ci, seed, want_map=False):
    """Bootstrap CIs for recall/precision/f1 (and optionally mAP50) over `keys`.
    Returns {metric: (lo, hi)}. Metrics are recomputed with op_metrics/ap_at on
    each resample, so the CI means exactly what the point estimate means."""
    import random
    rng = random.Random(seed)
    acc = {"recall": [], "precision": [], "f1": [], "map50": []}
    for _ in range(n_boot):
        rk = _resample_keys(keys, level, rng)
        r, p, f1, _ = op_metrics(preds_by_img, gts_by_img, rk, conf, iou)
        acc["recall"].append(r); acc["precision"].append(p); acc["f1"].append(f1)
        if want_map:
            acc["map50"].append(ap_at(preds_by_img, gts_by_img, rk, 0.5))
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for m, vals in acc.items():
        vals = [v for v in vals if v == v]  # drop NaN
        if not vals:
            out[m] = (float("nan"), float("nan"))
            continue
        vals.sort()
        out[m] = (_percentile(vals, lo_q), _percentile(vals, hi_q))
    return out


def bootstrap_recall_ci(preds_by_img, gts_by_img, keys, conf, iou, n_boot,
                        level, ci, seed):
    """Recall-only bootstrap CI (for per-source / population rows)."""
    return bootstrap_metric_cis(preds_by_img, gts_by_img, keys, conf, iou,
                                n_boot, level, ci, seed, want_map=False)["recall"]


def bootstrap_paired_delta_ci(base_preds, var_preds, gts_by_img, keys, conf, iou,
                              n_boot, level, ci, seed, metric="recall"):
    """CI for the DIFFERENCE (variant - base) of a metric, resampling the SAME
    clips for both arms on each draw. This is the correct test for 'did the
    augmentation help?': if the CI excludes 0, the improvement is significant at
    the (1-ci) level. Returns (delta_point, lo, hi)."""
    import random
    rng = random.Random(seed)
    def metric_of(preds, ks):
        r, p, f1, _ = op_metrics(preds, gts_by_img, ks, conf, iou)
        return {"recall": r, "precision": p, "f1": f1}[metric]
    point = metric_of(var_preds, keys) - metric_of(base_preds, keys)
    diffs = []
    for _ in range(n_boot):
        rk = _resample_keys(keys, level, rng)
        d = metric_of(var_preds, rk) - metric_of(base_preds, rk)
        if d == d:
            diffs.append(d)
    diffs.sort()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return point, _percentile(diffs, lo_q), _percentile(diffs, hi_q)


def _ci_str(lo, hi):
    if lo != lo or hi != hi:
        return ""
    return "[{:.3f}, {:.3f}]".format(lo, hi)


def _attach_ci(args, preds_by_img, gts_by_img, keys, row, want_map=False):
    """If bootstrap is on, compute CIs for this row's metrics over `keys` and
    store them as recall_ci/precision_ci/f1_ci/map50_ci on the row dict."""
    if getattr(args, "bootstrap", 0) <= 0:
        return
    cis = bootstrap_metric_cis(preds_by_img, gts_by_img,
                               [os.path.abspath(k) for k in keys],
                               args.conf, args.geom_iou, args.bootstrap,
                               args.bootstrap_level, args.bootstrap_ci,
                               args.bootstrap_seed, want_map=want_map)
    row["recall_ci"] = cis["recall"]
    row["precision_ci"] = cis["precision"]
    row["f1_ci"] = cis["f1"]
    if want_map:
        row["map50_ci"] = cis["map50"]


def op_metrics(preds_by_img, gts_by_img, keys, conf, iou_thr):
    """Recall/Precision/F1 at an operating confidence over a subset of images."""
    tp = fp = n_gt = 0
    for k in keys:
        gts = gts_by_img.get(k, [])
        n_gt += len(gts)
        preds = [p for p in preds_by_img.get(k, []) if p[0] >= conf]
        for _c, is_tp in match_image(preds, gts, iou_thr):
            if is_tp:
                tp += 1
            else:
                fp += 1
    recall = tp / n_gt if n_gt else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall)) else float("nan"))
    return recall, precision, f1, n_gt


def ap_at(preds_by_img, gts_by_img, keys, iou_thr):
    """All-point-interpolated AP at one IoU threshold over a subset of images."""
    scored = []
    n_gt = 0
    for k in keys:
        gts = gts_by_img.get(k, [])
        n_gt += len(gts)
        scored.extend(match_image(preds_by_img.get(k, []), gts, iou_thr))
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    scored.sort(key=lambda t: -t[0])
    tp_cum = fp_cum = 0
    rec, prec = [], []
    for _conf, is_tp in scored:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        rec.append(tp_cum / n_gt)
        prec.append(tp_cum / (tp_cum + fp_cum))
    # precision envelope (monotone non-increasing from the right)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap, prev_r = 0.0, 0.0
    for r, p in zip(rec, prec):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def _report_vram(tag):
    """Print allocated CUDA memory so leaks are visible in the console."""
    try:
        import torch
        if torch.cuda.is_available():
            gib = torch.cuda.memory_allocated() / (1024 ** 3)
            print("  [mem] {}: {:.2f} GiB allocated".format(tag, gib))
    except Exception:
        pass


def _free_cuda():
    """Reclaim GPU memory AFTER the caller has dropped its model reference
    (model = None). Must run after, not before: ultralytics models sit in
    reference cycles (model <-> predictor/validator callbacks), so they are
    only collectable by gc once no outside reference remains. Without this,
    every model evaluated in one invocation stays resident and the last
    stage OOMs with ~20 GiB "allocated" on a 12 GiB card.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def eval_geom_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows):
    """Run each --geom-model once over all images, convert person boxes to head
    estimates, then score every domain subset with the built-in matcher."""
    from ultralytics import YOLO

    # Confidence floor for the person model. We do NOT need the near-zero tail
    # that object-detection mAP curves use: a person box below ~0.05 conf yields
    # a garbage head estimate, and keeping hundreds of them per image (at conf
    # 0.001) blows up both the preds dict and GPU memory. 0.05 keeps the PR
    # curve meaningful while cutting the retained box count by 10-50x.
    floor = args.conf if args.no_map else max(0.05, args.geom_conf_floor)

    # Free CUDA between models/frames; harmless on CPU.
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False

    for name, weights in args.geom_models:
        if not os.path.exists(weights):
            print("[warn] skipping geom '{}': weights not found ({})".format(name, weights))
            continue
        print("\n=== geometric baseline: {} ({} person -> head, w_frac={}, aspect={}) ===".format(
            name, weights, args.geom_w_frac, args.geom_aspect))
        model = YOLO(weights)

        preds_by_img, gts_by_img = {}, {}
        # stream=True yields one Results at a time; batch=1 and per-frame cache
        # clearing keep peak GPU memory to a single image regardless of test-set
        # size. We copy out plain Python floats and never retain the Results.
        # One predict() call PER IMAGE. Never hand predict() the whole list:
        # some ultralytics versions ignore/reject the batch kwarg for list
        # sources and build ONE batch out of every image, which tries to
        # allocate tens of GiB in a single conv (observed: 17.88 GiB).
        for img_path in imgs:
            res = model.predict(img_path, imgsz=args.imgsz, conf=floor,
                                device=args.device, classes=[args.person_class],
                                verbose=False)[0]
            key = os.path.abspath(img_path)
            boxes = res.boxes
            preds = []
            n = 0 if boxes is None else len(boxes)
            if n:
                # pull the whole tensor to CPU once, not element-by-element
                confs = boxes.conf.detach().cpu().tolist()
                xyxys = boxes.xyxy.detach().cpu().tolist()
                for conf, pbox in zip(confs, xyxys):
                    preds.append((float(conf),
                                  head_from_person(tuple(pbox),
                                                   args.geom_w_frac, args.geom_aspect)))
            preds_by_img[key] = preds
            gts_by_img[key] = load_gt_boxes(img_path, images_dir, labels_dir, res.orig_shape)
            del res, boxes
            if _cuda:
                torch.cuda.empty_cache()
        for dom, paths in eval_sets:
            keys = [os.path.abspath(p) for p in paths]
            recall, precision, f1, _ = op_metrics(preds_by_img, gts_by_img, keys,
                                                  args.conf, args.geom_iou)
            map50 = map5095 = float("nan")
            if not args.no_map:
                map50 = ap_at(preds_by_img, gts_by_img, keys, 0.5)
                aps = [ap_at(preds_by_img, gts_by_img, keys, 0.5 + 0.05 * j) for j in range(10)]
                aps = [a for a in aps if a == a]
                map5095 = sum(aps) / len(aps) if aps else float("nan")
            n_heads = heads[dom] if dom != "ALL" else sum(heads.values())
            _r = dict(model=name, domain=dom, images=len(paths), heads=n_heads,
                             recall=recall, precision=precision, f1=f1,
                             map50=map50, map50_95=map5095)
            _attach_ci(args, preds_by_img, gts_by_img, paths, _r, want_map=not args.no_map)
            rows.append(_r)
            print("  {:<12} R={} P={} F1={} mAP50={}".format(
                dom, _fmt(recall), _fmt(precision), _fmt(f1), _fmt(map50)))
        model = None
        _free_cuda()
        _report_vram("after releasing " + name)


# --------------------------------------------------------------------------- #
# metric collection (needs ultralytics)
# --------------------------------------------------------------------------- #

def _prior_overlap(box, priors):
    """Soft overlap between a head detection and the person->head prior set.
    Max IoU over priors; a detection whose CENTER falls inside a prior counts
    at least 0.5 even when box sizes disagree (the prior is a coarse region,
    not a tight box, so IoU alone under-credits correct-but-small heads)."""
    best = 0.0
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    for _pc, pb in priors:
        v = iou_xyxy(box, pb)
        if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
            v = max(v, 0.5)
        if v > best:
            best = v
    return best



def _build_priors(args, imgs):
    """One person-model pass -> {abs_img_path: [(person_conf, head_region_xyxy)]}.
    Shared by the fused (--fuse) and combined (--edge-fuse) modes."""
    from ultralytics import YOLO
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False
    print("\n=== building person->head priors ({}, w_frac={}, aspect={}) ===".format(
        args.fuse_person, args.geom_w_frac, args.geom_aspect))
    person = YOLO(args.fuse_person)
    priors_by_img = {}
    # per-image predict; see eval_geom_models for why the list form is unsafe
    for img_path in imgs:
        res = person.predict(img_path, imgsz=args.imgsz, conf=max(0.05, args.geom_conf_floor),
                             device=args.device, classes=[args.person_class],
                             verbose=False)[0]
        pri = []
        if res.boxes is not None and len(res.boxes):
            confs = res.boxes.conf.detach().cpu().tolist()
            xyxys = res.boxes.xyxy.detach().cpu().tolist()
            for c, pb in zip(confs, xyxys):
                pri.append((float(c), head_from_person(tuple(pb),
                                                       args.geom_w_frac, args.geom_aspect)))
        priors_by_img[os.path.abspath(img_path)] = pri
        del res
        if _cuda:
            torch.cuda.empty_cache()
    person = None
    _free_cuda()
    _report_vram("after releasing person prior model")
    return priors_by_img


def _apply_prior(preds, priors, args):
    """Prior-guided re-scoring of candidate head boxes (the fusion step).
    preds: [(conf, xyxy)]. Returns the fused prediction list:
    detections boosted by overlap with priors; unclaimed priors injected
    at person_conf * beta (skipped entirely when beta == 0)."""
    fused = []
    boxes = [b for _c, b in preds]
    for c, hb in preds:
        ov = _prior_overlap(hb, priors)
        fused.append((min(1.0, c * (1.0 + args.fuse_alpha * ov)), hb))
    if args.fuse_beta > 0:
        for pc, pb in priors:
            if all(iou_xyxy(pb, hb) < 0.30 for hb in boxes):
                fused.append((pc * args.fuse_beta, pb))
    return fused


def _edge_collect(model, img_path, args, floor, strip_op_conf):
    """The app's edge-strip detect pass on ONE image: full frame + four
    full-resolution strips, boxes translated to frame coords, deduplicated
    confidence-descending at IoU 0.4 (face_blur.py _dedup_boxes semantics).
    Returns (preds, (h, w)) or (None, None) if the image is unreadable."""
    import cv2
    frame = cv2.imread(img_path)
    if frame is None:
        return None, None
    h, w = frame.shape[:2]
    e = args.edge_frac
    crops = [(frame, 0, 0)]
    for sx1, sy1, sx2, sy2 in [(0, 0, int(w*e), h), (int(w*(1-e)), 0, w, h),
                               (0, 0, w, int(h*e)), (0, int(h*(1-e)), w, h)]:
        strip = frame[sy1:sy2, sx1:sx2]
        if strip.size:
            crops.append((strip, sx1, sy1))
    preds = []
    for idx, (im, ox, oy) in enumerate(crops):
        res = model.predict(im, imgsz=args.imgsz, conf=floor,
                            device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            confs = res.boxes.conf.detach().cpu().tolist()
            xyxys = res.boxes.xyxy.detach().cpu().tolist()
            for c, b in zip(confs, xyxys):
                c = float(c)
                # strip boxes that clear the strip threshold but not --conf
                # are promoted to exactly --conf so op_metrics matches the
                # deployed behaviour (strips may run at a lower conf)
                if idx > 0 and strip_op_conf <= c < args.conf:
                    c = args.conf
                preds.append((c, (b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy)))
        del res
    return _dedup_conf_desc(preds, 0.40), (h, w)


def _is_border_gt(box, shape, margin):
    """True if a GT head box touches the frame edge within `margin` (fraction of
    the frame dimension). These are the frame-truncated heads that edge-strip
    inference targets; everything else is 'interior'."""
    h, w = shape
    mx, my = margin * w, margin * h
    x1, y1, x2, y2 = box
    return (x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my)


def _pop_recall(preds_by_img, gts_by_img, keys, conf, iou_thr, margin, border):
    """Recall over ONLY the border (or interior) GT population. Predictions are
    matched against the full GT set (so a pred matching an interior head is not
    stolen by the border tally), then we count TPs whose matched GT is in the
    requested population. Returns (recall, n_pop)."""
    tp = n_pop = 0
    for k in keys:
        gts = gts_by_img.get(k, [])
        shape = IMG_SHAPES.get(k)
        if shape is None:
            continue
        flags = [_is_border_gt(g, shape, margin) for g in gts]
        n_pop += sum(1 for fb in flags if fb == border)
        preds = [p for p in preds_by_img.get(k, []) if p[0] >= conf]
        # greedy conf-desc match to GT indices (mirror match_image, but keep the
        # matched GT index so we can attribute each TP to its population)
        order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
        matched = [False] * len(gts)
        for i in order:
            _c, pbox = preds[i]
            best, best_iou = -1, iou_thr
            for g, gt in enumerate(gts):
                if matched[g]:
                    continue
                v = iou_xyxy(pbox, gt)
                if v >= best_iou:
                    best, best_iou = g, v
            if best >= 0:
                matched[best] = True
                if flags[best] == border:
                    tp += 1
    recall = tp / n_pop if n_pop else float("nan")
    return recall, n_pop


def _append_pop_rows(args, preds_by_img, gts_by_img, keys, rows, row_name):
    """Append two extra rows: border-touching and interior GT recall. Only the
    ALL image set is used (populations are geometric, not per-domain). Guarded by
    --edge-margin > 0."""
    if getattr(args, "edge_margin", 0) <= 0:
        return
    for border, tag in ((True, "border"), (False, "interior")):
        rec, n_pop = _pop_recall(preds_by_img, gts_by_img, keys,
                                 args.conf, args.geom_iou, args.edge_margin, border)
        rows.append(dict(model=row_name, domain=tag, images=len(keys), heads=n_pop,
                         recall=rec, precision=float("nan"), f1=float("nan"),
                         map50=float("nan"), map50_95=float("nan")))
        print("  {:<12} R={} (n={})".format(tag, _fmt(rec), n_pop))


def _score_rows(args, preds_by_img, gts_by_img, eval_sets, heads, rows, row_name):
    """Score one prediction set over every domain subset and append rows."""
    for dom, paths in eval_sets:
        keys = [os.path.abspath(p) for p in paths]
        recall, precision, f1, _ = op_metrics(preds_by_img, gts_by_img, keys,
                                              args.conf, args.geom_iou)
        map50 = map5095 = float("nan")
        if not args.no_map:
            map50 = ap_at(preds_by_img, gts_by_img, keys, 0.5)
            aps = [ap_at(preds_by_img, gts_by_img, keys, 0.5 + 0.05 * j) for j in range(10)]
            aps = [a for a in aps if a == a]
            map5095 = sum(aps) / len(aps) if aps else float("nan")
        n_heads = heads[dom] if dom != "ALL" else sum(heads.values())
        rows.append(dict(model=row_name, domain=dom, images=len(paths), heads=n_heads,
                         recall=recall, precision=precision, f1=f1,
                         map50=map50, map50_95=map5095))
        print("  {:<12} R={} P={} F1={} mAP50={}".format(
            dom, _fmt(recall), _fmt(precision), _fmt(f1), _fmt(map50)))
    _append_pop_rows(args, preds_by_img, gts_by_img,
                     [os.path.abspath(p) for p in dict(eval_sets)["ALL"]],
                     rows, row_name)


def eval_combo_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows, priors_by_img=None):
    """Combined mode (--edge-fuse): edge strips AND the person->head prior
    together, i.e. the app's actual runtime configuration. Per image: the
    edge-strip pass collects and dedups candidates, then the prior re-scores
    them (and injects unclaimed priors if beta > 0). Rows: '<name>+edge+prior'.
    Costs 5 head inferences per image plus one shared person pass."""
    from ultralytics import YOLO
    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False

    if priors_by_img is None:
        priors_by_img = _build_priors(args, imgs)

    floor = 0.05
    strip_op_conf = max(0.10, args.conf - args.edge_conf_drop)
    for name, weights in args.edge_fuse:
        if not os.path.exists(weights):
            print("[warn] skipping edge-fuse '{}': weights not found ({})".format(name, weights))
            continue
        print("\n=== combined model: {}+edge+prior ({}; frac={}, alpha={}, beta={}) ===".format(
            name, weights, args.edge_frac, args.fuse_alpha, args.fuse_beta))
        model = YOLO(weights)
        preds_by_img, gts_by_img = {}, {}
        for img_path in imgs:
            key = os.path.abspath(img_path)
            preds, shape = _edge_collect(model, img_path, args, floor, strip_op_conf)
            if preds is None:
                continue
            preds_by_img[key] = _apply_prior(preds, priors_by_img.get(key, []), args)
            gts_by_img[key] = load_gt_boxes(img_path, images_dir, labels_dir, shape)
            if _cuda:
                torch.cuda.empty_cache()
        model = None
        _free_cuda()
        _report_vram("after releasing " + name)
        _score_rows(args, preds_by_img, gts_by_img, eval_sets, heads, rows, name + "+edge+prior")


def eval_fused_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows, priors_by_img=None):
    """Prior-guided ("heatmap") evaluation: the COCO person detector defines
    soft head-region priors; the head model's detections are re-scored by their
    overlap with those priors, and unclaimed priors are added as weak
    geometry-only boxes.

      fused_conf(head box)  = min(1, conf * (1 + alpha * overlap))
      fused_conf(prior box) = person_conf * beta      (only if no head box
                                                       overlaps that prior)

    The head model runs at a LOW floor (--fuse-floor, default 0.10): weak
    detections inside a prior get boosted past the operating point (the
    "rescue"); weak detections in the open stay below it. Scored with the
    same matcher/AP as the geometric baseline. NOTE: the AP curve is
    truncated at the floor, so fused mAP is slightly conservative.
    """
    from ultralytics import YOLO

    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False

    # NOTE: no existence check here -- ultralytics auto-downloads official
    # names like "yolo11n.pt"; a bad local path fails loudly below instead.

    if priors_by_img is None:
        priors_by_img = _build_priors(args, imgs)

    # ---- pass 2: each head model at a low floor, fused with the priors ----
    for name, weights in args.fuse:
        if not os.path.exists(weights):
            print("[warn] skipping fused '{}': weights not found ({})".format(name, weights))
            continue
        print("\n=== fused model: {}+prior ({}; floor={}, alpha={}, beta={}) ===".format(
            name, weights, args.fuse_floor, args.fuse_alpha, args.fuse_beta))
        model = YOLO(weights)
        preds_by_img, gts_by_img = {}, {}
        base_by_img = {}   # candidates WITHOUT the prior (for the paired delta)
        # per-image predict; see eval_geom_models for why the list form is unsafe
        for img_path in imgs:
            res = model.predict(img_path, imgsz=args.imgsz, conf=args.fuse_floor,
                                device=args.device, verbose=False)[0]
            key = os.path.abspath(img_path)
            priors = priors_by_img.get(key, [])
            preds = []
            head_boxes = []
            base_here = []
            if res.boxes is not None and len(res.boxes):
                confs = res.boxes.conf.detach().cpu().tolist()
                xyxys = res.boxes.xyxy.detach().cpu().tolist()
                for c, hb in zip(confs, xyxys):
                    hb = tuple(hb)
                    ov = _prior_overlap(hb, priors)
                    preds.append((min(1.0, float(c) * (1.0 + args.fuse_alpha * ov)), hb))
                    head_boxes.append(hb)
                    base_here.append((float(c), hb))   # unrescored candidate
            # unclaimed priors -> weak geometry-only boxes (never full conf:
            # gear-forward poses misplace the region, see face_blur.py notes)
            for pc, pb in priors:
                if all(iou_xyxy(pb, hb) < 0.30 for hb in head_boxes):
                    preds.append((pc * args.fuse_beta, pb))
            preds_by_img[key] = preds
            base_by_img[key] = base_here
            gts_by_img[key] = load_gt_boxes(img_path, images_dir, labels_dir, res.orig_shape)
            del res
            if _cuda:
                torch.cuda.empty_cache()
        for dom, paths in eval_sets:
            keys = [os.path.abspath(p) for p in paths]
            recall, precision, f1, _ = op_metrics(preds_by_img, gts_by_img, keys,
                                                  args.conf, args.geom_iou)
            map50 = map5095 = float("nan")
            if not args.no_map:
                map50 = ap_at(preds_by_img, gts_by_img, keys, 0.5)
                aps = [ap_at(preds_by_img, gts_by_img, keys, 0.5 + 0.05 * j) for j in range(10)]
                aps = [a for a in aps if a == a]
                map5095 = sum(aps) / len(aps) if aps else float("nan")
            n_heads = heads[dom] if dom != "ALL" else sum(heads.values())
            _r = dict(model=name + "+prior", domain=dom, images=len(paths), heads=n_heads,
                             recall=recall, precision=precision, f1=f1,
                             map50=map50, map50_95=map5095)
            _attach_ci(args, preds_by_img, gts_by_img, paths, _r, want_map=not args.no_map)
            rows.append(_r)
            print("  {:<12} R={} P={} F1={} mAP50={}".format(
                dom, _fmt(recall), _fmt(precision), _fmt(f1), _fmt(map50)))
        _append_pop_rows(args, preds_by_img, gts_by_img,
                         [os.path.abspath(p) for p in dict(eval_sets)["ALL"]],
                         rows, name + "+prior")
        # Paired significance test: does the prior significantly change recall/
        # precision vs the same model WITHOUT rescoring, on the same resampled
        # clips? If the CI excludes 0 the effect is significant at (1-ci).
        if getattr(args, "bootstrap", 0) > 0:
            allk = [os.path.abspath(p) for p in dict(eval_sets)["ALL"]]
            for metric in ("recall", "precision"):
                pt, lo, hi = bootstrap_paired_delta_ci(
                    base_by_img, preds_by_img, gts_by_img, allk, args.conf,
                    args.geom_iou, args.bootstrap, args.bootstrap_level,
                    args.bootstrap_ci, args.bootstrap_seed, metric=metric)
                sig = "significant" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
                print("  [delta] {} {}: {:+.3f}  {:.0%} CI [{:+.3f}, {:+.3f}]  {}".format(
                    name + "+prior", metric, pt, args.bootstrap_ci, lo, hi, sig))
                rows.append(dict(model=name + "+prior\u0394", domain=metric,
                                 images="", heads="", recall=pt, precision="",
                                 f1="", map50="", map50_95="",
                                 recall_ci=(lo, hi)))
        model = None
        _free_cuda()
        _report_vram("after releasing " + name)



def _dedup_conf_desc(preds, iou_thresh=0.40):
    """Confidence-descending IoU dedup, replicating face_blur.py's
    _dedup_boxes(): the highest-confidence box wins; any box overlapping a
    kept box above the threshold is suppressed. preds: [(conf, xyxy)]."""
    kept = []
    for conf, box in sorted(preds, key=lambda t: -t[0]):
        if all(iou_xyxy(box, kb) <= iou_thresh for _c, kb in kept):
            kept.append((conf, box))
    return kept


def eval_edge_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows):
    """Edge-strip ablation: replicate the app's detect pass (face_blur.py
    detect_faces with edge_strip=True) and score it with the custom matcher.

    Per image: one full-frame pass, plus four full-resolution strips covering
    --edge-frac (default 0.20) of each side, run at (conf - --edge-conf-drop,
    floored at 0.10); strip boxes are translated back to frame coordinates and
    the union is deduplicated confidence-descending at IoU 0.4, exactly as in
    the app. Rows are named '<name>+edge'; run the same weights via --model
    for the no-strip ablation row. Costs 5 inferences per image.

    NOTE: candidate boxes are collected from conf 0.05 up so the AP curve is
    meaningful, but the strip conf-drop is applied RELATIVE to --conf when
    filtering at the operating point, mirroring deployment.
    """
    from ultralytics import YOLO
    import cv2

    try:
        import torch
        _cuda = torch.cuda.is_available()
    except Exception:
        torch, _cuda = None, False

    floor = 0.05
    strip_op_conf = max(0.10, args.conf - args.edge_conf_drop)

    for name, weights in args.edge:
        if not os.path.exists(weights):
            print("[warn] skipping edge '{}': weights not found ({})".format(name, weights))
            continue
        print("\n=== edge-strip model: {}+edge ({}; frac={}, strip conf={}) ===".format(
            name, weights, args.edge_frac, strip_op_conf))
        model = YOLO(weights)
        preds_by_img, gts_by_img = {}, {}
        for img_path in imgs:
            key = os.path.abspath(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            e = args.edge_frac
            crops = [(frame, 0, 0)]
            for sx1, sy1, sx2, sy2 in [(0, 0, int(w*e), h), (int(w*(1-e)), 0, w, h),
                                       (0, 0, w, int(h*e)), (0, int(h*(1-e)), w, h)]:
                strip = frame[sy1:sy2, sx1:sx2]
                if strip.size:
                    crops.append((strip, sx1, sy1))
            preds = []
            for idx, (im, ox, oy) in enumerate(crops):
                res = model.predict(im, imgsz=args.imgsz, conf=floor,
                                    device=args.device, verbose=False)[0]
                if res.boxes is not None and len(res.boxes):
                    confs = res.boxes.conf.detach().cpu().tolist()
                    xyxys = res.boxes.xyxy.detach().cpu().tolist()
                    for c, b in zip(confs, xyxys):
                        c = float(c)
                        # strips run at a (possibly) lower operating conf than
                        # the main pass; below-op boxes are kept for the AP
                        # curve but flagged so op_metrics at --conf matches
                        # deployment: promote strip boxes that clear the strip
                        # threshold but not --conf up to --conf exactly.
                        if idx > 0 and strip_op_conf <= c < args.conf:
                            c = args.conf
                        preds.append((c, (b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy)))
                del res
            if _cuda:
                torch.cuda.empty_cache()
            preds_by_img[key] = _dedup_conf_desc(preds, 0.40)
            gts_by_img[key] = load_gt_boxes(img_path, images_dir, labels_dir, (h, w))
        model = None
        _free_cuda()
        _report_vram("after releasing " + name)

        for dom, paths in eval_sets:
            keys = [os.path.abspath(p) for p in paths]
            recall, precision, f1, _ = op_metrics(preds_by_img, gts_by_img, keys,
                                                  args.conf, args.geom_iou)
            map50 = map5095 = float("nan")
            if not args.no_map:
                map50 = ap_at(preds_by_img, gts_by_img, keys, 0.5)
                aps = [ap_at(preds_by_img, gts_by_img, keys, 0.5 + 0.05 * j) for j in range(10)]
                aps = [a for a in aps if a == a]
                map5095 = sum(aps) / len(aps) if aps else float("nan")
            n_heads = heads[dom] if dom != "ALL" else sum(heads.values())
            _r = dict(model=name + "+edge", domain=dom, images=len(paths), heads=n_heads,
                             recall=recall, precision=precision, f1=f1,
                             map50=map50, map50_95=map5095)
            _attach_ci(args, preds_by_img, gts_by_img, paths, _r, want_map=not args.no_map)
            rows.append(_r)
            print("  {:<12} R={} P={} F1={} mAP50={}".format(
                dom, _fmt(recall), _fmt(precision), _fmt(f1), _fmt(map50)))
        _append_pop_rows(args, preds_by_img, gts_by_img,
                         [os.path.abspath(p) for p in dict(eval_sets)["ALL"]],
                         rows, name + "+edge")


def evaluate(model, tmp_yaml, imgsz, device, conf):
    """One ultralytics val pass. Returns (recall, precision, map50, map5095)."""
    kw = dict(data=tmp_yaml, imgsz=imgsz, device=device, split="val",
              verbose=False, plots=False)
    if conf is not None:
        kw["conf"] = conf
    m = model.val(**kw)
    b = m.box
    return (float(getattr(b, "mr", float("nan"))),
            float(getattr(b, "mp", float("nan"))),
            float(getattr(b, "map50", float("nan"))),
            float(getattr(b, "map", float("nan"))))


def collect_rows(args):
    import yaml
    from ultralytics import YOLO

    with open(args.data) as f:
        dcfg = yaml.safe_load(f) or {}
    base = dcfg.get("path") or os.path.dirname(os.path.abspath(args.data))
    names = dcfg.get("names", {0: "head"})

    images_dir = args.images_dir or os.path.join(base, "images", args.split)
    labels_dir = os.path.join(base, "labels", args.split)
    if not os.path.isdir(images_dir):
        sys.exit("[ERROR] split images dir not found: {} (check --split/--images-dir)".format(images_dir))

    domains = parse_domains(args.domain) or [("catwalk", ["new_clip", "cat_"]), ("helmet", ["clip"])]

    imgs = []
    for root, _d, fnames in os.walk(images_dir):
        for nm in fnames:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(root, nm))
    if not imgs:
        sys.exit("[ERROR] no images in {}".format(images_dir))

    buckets = defaultdict(list)
    frames = defaultdict(int)
    heads = defaultdict(int)
    for p in imgs:
        dom = domain_of(clip_of(p), domains)
        buckets[dom].append(p)
        frames[dom] += 1
        heads[dom] += count_instances(p, images_dir, labels_dir)

    ordered = [d for d, _ in domains if d in buckets]
    ordered += [d for d in buckets if d not in ordered]  # include (other)

    print("=" * 70)
    print("Split '{}': {} images across {} domain(s). imgsz={} conf={}".format(
        args.split, len(imgs), len(ordered), args.imgsz, args.conf))
    for d in ordered:
        print("  {:<12} {:>5} frames  {:>6} labeled heads".format(d, frames[d], heads[d]))
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="evalcmp_")
    rows = []
    eval_sets = [(d, buckets[d]) for d in ordered]
    eval_sets.append(("ALL", imgs))
    for name, weights in args.models:
        if not os.path.exists(weights):
            print("[warn] skipping '{}': weights not found ({})".format(name, weights))
            continue
        print("\n=== model: {} ({}) ===".format(name, weights))
        model = YOLO(weights)

        # For bootstrap CIs and/or the border/interior split the base model needs
        # per-box predictions (ultralytics .val() does not expose them). Run ONE
        # predict pass up front and reuse it; otherwise skip for speed.
        base_preds_by_img = base_gts_by_img = None
        if getattr(args, "bootstrap", 0) > 0 or getattr(args, "edge_margin", 0) > 0:
            base_preds_by_img, base_gts_by_img = {}, {}
            for p in imgs:
                k = os.path.abspath(p)
                res = model.predict(p, imgsz=args.imgsz, conf=args.conf,
                                    device=args.device, verbose=False)[0]
                pr = []
                if res.boxes is not None and len(res.boxes):
                    cs = res.boxes.conf.detach().cpu().tolist()
                    bs = res.boxes.xyxy.detach().cpu().tolist()
                    pr = [(float(c), tuple(b)) for c, b in zip(cs, bs)]
                base_preds_by_img[k] = pr
                base_gts_by_img[k] = load_gt_boxes(p, images_dir, labels_dir, res.orig_shape)
                del res

        for dom, paths in eval_sets:
            list_txt = os.path.join(tmpdir, "imgs_{}_{}.txt".format(name, dom.strip("()")))
            with open(list_txt, "w") as f:
                for p in paths:
                    f.write(os.path.abspath(p) + "\n")
            tmp_yaml = os.path.join(tmpdir, "data_{}_{}.yaml".format(name, dom.strip("()")))
            with open(tmp_yaml, "w") as f:
                yaml.safe_dump({"path": base, "train": list_txt, "val": list_txt, "names": names}, f)

            try:
                recall, precision, _, _ = evaluate(model, tmp_yaml, args.imgsz, args.device, args.conf)
                map50 = map5095 = float("nan")
                if not args.no_map:
                    _, _, map50, map5095 = evaluate(model, tmp_yaml, args.imgsz, args.device, None)
            except Exception as e:  # noqa
                print("  [warn] eval failed for {}/{}: {}".format(name, dom, e))
                recall = precision = map50 = map5095 = float("nan")

            f1 = (2 * precision * recall / (precision + recall)
                  if (precision == precision and recall == recall and (precision + recall)) else float("nan"))
            n_heads = heads[dom] if dom != "ALL" else sum(heads.values())
            _r = dict(model=name, domain=dom, images=len(paths), heads=n_heads,
                             recall=recall, precision=precision, f1=f1,
                             map50=map50, map50_95=map5095)
            if base_preds_by_img is not None:
                _attach_ci(args, base_preds_by_img, base_gts_by_img, paths, _r,
                           want_map=not args.no_map)
            rows.append(_r)
            print("  {:<12} R={} P={} F1={} mAP50={}".format(
                dom, _fmt(recall), _fmt(precision), _fmt(f1), _fmt(map50)))

        # Border/interior populations for the BASE model (reuses the up-front
        # predict pass built above; only emitted when --edge-margin > 0).
        if getattr(args, "edge_margin", 0) > 0 and base_preds_by_img is not None:
            _append_pop_rows(args, base_preds_by_img, base_gts_by_img,
                             [os.path.abspath(p) for p in imgs], rows, name)
        model = None
        base_preds_by_img = base_gts_by_img = None
        _free_cuda()
        _report_vram("after releasing " + name)

    if args.geom_models:
        eval_geom_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows)
    priors_cache = None
    if args.fuse or args.edge_fuse:
        priors_cache = _build_priors(args, imgs)
    if args.fuse:
        eval_fused_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows,
                          priors_by_img=priors_cache)
    if args.edge:
        eval_edge_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows)
    if args.edge_fuse:
        eval_combo_models(args, imgs, eval_sets, heads, images_dir, labels_dir, rows,
                          priors_by_img=priors_cache)
    return rows, ordered


# --------------------------------------------------------------------------- #
# reporting (pure functions -- no ultralytics needed)
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    return "{:.{}f}".format(v, nd) if isinstance(v, (int, float)) and v == v else "-"


def write_csv(rows, path):
    fields = ["model", "domain", "images", "heads", "recall", "precision", "f1", "map50", "map50_95"]
    # add CI columns only if any row carries them
    has_ci = any("recall_ci" in r for r in rows)
    ci_fields = []
    if has_ci:
        ci_fields = ["recall_lo", "recall_hi", "precision_lo", "precision_hi",
                     "f1_lo", "f1_hi", "map50_lo", "map50_hi"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ci_fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fields}
            if has_ci:
                for m, lo, hi in (("recall_ci", "recall_lo", "recall_hi"),
                                  ("precision_ci", "precision_lo", "precision_hi"),
                                  ("f1_ci", "f1_lo", "f1_hi"),
                                  ("map50_ci", "map50_lo", "map50_hi")):
                    v = r.get(m)
                    if v and v[0] == v[0]:
                        row[lo] = "{:.4f}".format(v[0])
                        row[hi] = "{:.4f}".format(v[1])
                    else:
                        row[lo] = row[hi] = ""
            w.writerow(row)


def _models_in_order(rows):
    seen = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def _domains_in_order(rows, ordered_domains):
    doms = [d for d in ordered_domains]
    if any(r["domain"] == "ALL" for r in rows):
        doms.append("ALL")
    return doms


def write_markdown(rows, ordered_domains, path, metric_note):
    models = _models_in_order(rows)
    doms = _domains_in_order(rows, ordered_domains)
    lut = {(r["model"], r["domain"]): r for r in rows}
    lines = ["# Per-domain model comparison", "", metric_note, ""]
    # Recall matrix (headline)
    lines.append("## Recall by domain (headline)")
    lines.append("")
    lines.append("| Domain | Heads | " + " | ".join(models) + " |")
    lines.append("|" + "---|" * (len(models) + 2))
    for d in doms:
        heads = next((lut[(m, d)]["heads"] for m in models if (m, d) in lut), "-")
        cells = [_fmt(lut[(m, d)]["recall"]) if (m, d) in lut else "-" for m in models]
        tag = " _(reference only)_" if d == "ALL" else ""
        lines.append("| {}{} | {} | {} |".format(d, tag, heads, " | ".join(cells)))
    # Full metrics
    lines += ["", "## Full metrics", "",
              "| Model | Domain | Images | Heads | Recall | Precision | F1 | mAP50 | mAP50-95 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["model"], r["domain"], r["images"], r["heads"],
            _fmt(r["recall"]), _fmt(r["precision"]), _fmt(r["f1"]),
            _fmt(r["map50"]), _fmt(r["map50_95"])))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(rows, ordered_domains, path, caption, conf):
    models = _models_in_order(rows)
    doms = [d for d in _domains_in_order(rows, ordered_domains) if d != "ALL"]
    lut = {(r["model"], r["domain"]): r for r in rows}
    col = "l r " + " ".join(["r"] * len(models))
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{" + _tex_escape(caption) + r"}",
           r"\label{tab:head_recall}",
           r"\begin{tabular}{" + col + r"}", r"\toprule",
           "Domain & Heads & " + " & ".join(_tex_escape(m) for m in models) + r" \\",
           r"\midrule"]
    for d in doms:
        heads = next((lut[(m, d)]["heads"] for m in models if (m, d) in lut), "-")
        cells = [_fmt(lut[(m, d)]["recall"], 3) if (m, d) in lut else "-" for m in models]
        out.append("{} & {} & {} \\\\".format(_tex_escape(d), heads, " & ".join(cells)))
    out += [r"\bottomrule", r"\end{tabular}",
            r"\vspace{2pt}",
            r"\\{\footnotesize Recall at operating confidence " + str(conf) +
            r", imgsz 960. Per domain; pooled omitted.}",
            r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def make_figure(rows, ordered_domains, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa
        print("[warn] matplotlib not available, skipping figure: {}".format(e))
        return
    models = _models_in_order(rows)
    doms = [d for d in ordered_domains if d != "ALL"]
    lut = {(r["model"], r["domain"]): r for r in rows}
    colors = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5", "#854F0B", "#3B6D11"]

    x = np.arange(len(doms))
    n = max(len(models), 1)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(doms), 4.2), dpi=150)
    for i, m in enumerate(models):
        vals = [lut[(m, d)]["recall"] if (m, d) in lut and lut[(m, d)]["recall"] == lut[(m, d)]["recall"] else 0
                for d in doms]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=m,
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "{:.2f}".format(v),
                    ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    ax.set_xticks(x)
    ax.set_xticklabels(doms)
    ax.set_ylabel("Recall (operating point)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12, color="#1F3A5F")
    ax.legend(frameon=False, fontsize=9, ncol=min(len(models), 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Compare head models per domain for the paper.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", action="append", dest="models", type=parse_model, default=[],
                    metavar="name:weights", help="repeatable; e.g. --model head_s:head_s.pt")
    ap.add_argument("--geom-model", action="append", dest="geom_models", type=parse_model, default=[],
                    metavar="name:weights",
                    help="repeatable; COCO person model scored via the person->head "
                         "geometric baseline, e.g. --geom-model person2head_n:yolo11n.pt")
    ap.add_argument("--geom-w-frac", type=float, default=0.40,
                    help="estimated head width as a fraction of the person-box width "
                         "(default 0.40; match the app's fallback constant)")
    ap.add_argument("--geom-aspect", type=float, default=1.10,
                    help="estimated head height as a multiple of head width (default 1.10)")
    ap.add_argument("--geom-iou", type=float, default=0.5,
                    help="IoU match threshold for the geometric baseline (default 0.5, "
                         "same as the ultralytics val default)")
    ap.add_argument("--person-class", type=int, default=0,
                    help="class index of 'person' in the geom model (COCO default 0)")
    ap.add_argument("--geom-conf-floor", type=float, default=0.05,
                    help="lowest person-detection confidence kept for the geometric "
                         "baseline mAP curve (default 0.05; raise toward --conf if you "
                         "hit GPU OOM on dense scenes, lower for a longer PR tail)")
    ap.add_argument("--edge-fuse", action="append", dest="edge_fuse", type=parse_model, default=[],
                    help="name:weights evaluated with edge strips AND the person->head prior "
                         "together -- the app's runtime configuration (repeatable). Rows: "
                         "'<name>+edge+prior'. Uses the --edge-* and --fuse-* parameters.")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="if > 0, add bootstrap confidence intervals to every "
                         "recall/precision/f1 (and mAP) figure, using this many "
                         "resamples (e.g. 1000). Off by default.")
    ap.add_argument("--bootstrap-level", choices=["clip", "image"], default="clip",
                    help="bootstrap resampling unit (default clip: the honest "
                         "choice, since heads within a clip are correlated).")
    ap.add_argument("--bootstrap-ci", type=float, default=0.95,
                    help="confidence level for the intervals (default 0.95)")
    ap.add_argument("--bootstrap-seed", type=int, default=0,
                    help="RNG seed for reproducible intervals (default 0)")
    ap.add_argument("--edge-margin", type=float, default=0.0,
                    help="if > 0, also report border-touching vs interior GT "
                         "recall (fraction of frame dim within which a GT box "
                         "counts as frame-truncated; try 0.02). Adds 'border' "
                         "and 'interior' rows per model to the CSV. The base "
                         "model gets one extra predict pass to compute these.")
    ap.add_argument("--edge", action="append", dest="edge", type=parse_model, default=[],
                    help="name:weights of a head model to evaluate WITH the app's "
                         "edge-strip pass (repeatable). Produces rows named '<name>+edge'; "
                         "run the same weights via --model for the no-strip ablation row.")
    ap.add_argument("--edge-frac", type=float, default=0.20,
                    help="strip width as a fraction of each side (default 0.20, as in the app)")
    ap.add_argument("--edge-conf-drop", type=float, default=0.0,
                    help="strip confidence drop relative to --conf (default 0.0, the app's "
                         "head-pass setting; the face pass uses 0.1)")
    ap.add_argument("--fuse", action="append", dest="fuse", type=parse_model, default=[],
                    help="name:weights of a head model to evaluate WITH the person->head "
                         "prior (repeatable). Produces rows named '<name>+prior'. "
                         "Run the same weights via --model too for the unguided ablation row.")
    ap.add_argument("--fuse-person", default="yolo11n.pt",
                    help="COCO person weights used to build the priors (default yolo11n.pt)")
    ap.add_argument("--fuse-floor", type=float, default=0.10,
                    help="head-model confidence floor for fusion candidates (default 0.10). "
                         "Weak boxes above this can be rescued by the prior.")
    ap.add_argument("--fuse-alpha", type=float, default=1.0,
                    help="prior boost strength: conf*(1+alpha*overlap) (default 1.0)")
    ap.add_argument("--fuse-beta", type=float, default=0.5,
                    help="confidence multiplier for geometry-only boxes from unclaimed "
                         "priors (default 0.5); set 0 to disable adding them")
    ap.add_argument("--split", default="test", help="split to evaluate (default test)")
    ap.add_argument("--imgsz", type=int, default=960, help="inference size (match the app, default 960)")
    ap.add_argument("--conf", type=float, default=0.25, help="operating-point confidence (default 0.25)")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--domain", action="append", default=None,
                    help="domain rule name:prefix1,prefix2 (repeatable)")
    ap.add_argument("--images-dir", default=None, help="override split images dir")
    ap.add_argument("--no-map", action="store_true", help="skip the mAP pass (faster)")
    ap.add_argument("--out-dir", default="paper_eval", help="output directory")
    ap.add_argument("--title", default="Head-detection recall by domain",
                    help="figure title / table caption")
    args = ap.parse_args()

    if not args.data:
        ap.error("missing --data")
    if not (args.models or args.geom_models):
        ap.error("need at least one --model or --geom-model")
    if not os.path.exists(args.data):
        sys.exit("[ERROR] data yaml not found: {}".format(args.data))

    os.makedirs(args.out_dir, exist_ok=True)
    rows, ordered = collect_rows(args)
    if not rows:
        sys.exit("[ERROR] no results collected.")

    note = ("Recall/Precision/F1 at operating confidence {}, imgsz {}. "
            "mAP is threshold-independent. Recall is the headline metric "
            "(a missed head is a privacy leak). Pooled 'ALL' is reference only.").format(args.conf, args.imgsz)

    csv_p = os.path.join(args.out_dir, "eval_comparison.csv")
    md_p = os.path.join(args.out_dir, "eval_comparison.md")
    tex_p = os.path.join(args.out_dir, "eval_comparison.tex")
    fig_p = os.path.join(args.out_dir, "recall_by_domain.png")

    write_csv(rows, csv_p)
    write_markdown(rows, ordered, md_p, note)
    write_latex(rows, ordered, tex_p, args.title, args.conf)
    make_figure(rows, ordered, fig_p, args.title)

    print("\nWrote:")
    for p in (csv_p, md_p, tex_p, fig_p):
        print("  " + p)
    print("\nHeadline = per-domain recall. Do not report the pooled 'ALL' row as the result.")


if __name__ == "__main__":
    main()
