r"""
find_misses.py -- surface the frames where your head model MISSES labeled heads,
so your next labeling round targets the model's actual weak spots instead of
random frames.

For a privacy blur, a MISS is a leak: a real (labeled) head the model did not
detect. This tool runs the model on an already-labeled split, and for each
labeled head checks whether any prediction covers it (IoU >= threshold at the
confidence the APP actually runs). Heads with no covering prediction are misses.

It then:
  - ranks frames by miss count (worst first),
  - restricts to a domain (default: catwalk, your weak domain),
  - stages each flagged frame + its EXISTING label into a YOLO-layout folder you
    can open directly in your labeling tool (X-AnyLabeling / CVAT), and
  - writes an annotated *preview* JPG (missed heads in RED, caught heads in GREEN,
    model boxes in YELLOW) named with the miss count so your file browser sorts
    the worst frames to the top.

Run it against the SAME imgsz + confidence the app uses, so a "miss" here means a
miss in production, not at some eval-only operating point.

USAGE (yolotrain env):
    conda activate yolotrain

    # default: catwalk misses on the val split, using shipping nano at the app's
    # operating point (imgsz 960, conf 0.25, IoU-match 0.5)
    python find_misses.py --data head_dataset/data.yaml --model head_n.pt

    # look at train too (bigger pool of hard cases to learn the *pattern* of misses)
    python find_misses.py --data head_dataset/data.yaml --model head_n.pt --split train

    # both domains, cap output to the 200 worst frames
    python find_misses.py --data head_dataset/data.yaml --model head_n.pt \
        --only all --limit 200

    # point at an arbitrary labeled folder instead of a split
    python find_misses.py --images path/to/imgs --labels path/to/lbls --model head_n.pt

Domain rules default to your naming (pre- OR post-rename): catwalk = catwalk*/
new_clip*/cat_*, helmet = helmet*/clip*. Override with --domain name:prefix,...

READ THE NOTE the tool prints at the end about what re-labeling already-labeled
frames does and does NOT buy you -- the real leverage is in what the misses tell
you to go collect.
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")


# --------------------------------------------------------------------------- #
# Pure helpers (no cv2 / ultralytics) -- unit-testable on their own.
# --------------------------------------------------------------------------- #
def clip_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def parse_domains(entries):
    """['catwalk:new_clip,cat_', 'helmet:clip'] -> [(name, [prefixes]), ...]."""
    out = []
    for e in entries:
        if ":" not in e:
            print("[warn] ignoring bad --domain '{}' (need name:prefix)".format(e))
            continue
        name, prefixes = e.split(":", 1)
        out.append((name.strip(),
                    [p.strip() for p in prefixes.split(",") if p.strip()]))
    return out


def domain_of(clip, domains):
    for name, prefixes in domains:
        if any(clip.startswith(p) for p in prefixes):
            return name
    return "(other)"


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Normalized YOLO (cx,cy,w,h) -> absolute (x1,y1,x2,y2) pixels."""
    bw = w * img_w
    bh = h * img_h
    x1 = cx * img_w - bw / 2.0
    y1 = cy * img_h - bh / 2.0
    return (x1, y1, x1 + bw, y1 + bh)


def iou_xyxy(a, b):
    """IoU of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_misses(gt_boxes, pred_boxes, iou_thr):
    """For each GT box, is it covered by some prediction (IoU >= thr)?

    Returns (missed_idx, matched_idx, unmatched_pred_idx).
    A GT is 'matched' if its best IoU over all preds >= thr. A pred is
    'unmatched' if it covers no GT (candidate false-positive OR an unlabeled
    head worth adding).
    """
    missed, matched = [], []
    pred_hit = [False] * len(pred_boxes)
    for gi, g in enumerate(gt_boxes):
        best_iou, best_pi = 0.0, -1
        for pi, p in enumerate(pred_boxes):
            v = iou_xyxy(g, p)
            if v > best_iou:
                best_iou, best_pi = v, pi
        if best_iou >= iou_thr:
            matched.append(gi)
            if best_pi >= 0:
                pred_hit[best_pi] = True
        else:
            missed.append(gi)
    unmatched_pred = [i for i, hit in enumerate(pred_hit) if not hit]
    return missed, matched, unmatched_pred


def load_gt(label_path, img_w, img_h, cls_filter=0):
    """Read a YOLO label file -> list of (x1,y1,x2,y2) for class cls_filter."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls_filter is not None and cls != cls_filter:
                continue
            cx, cy, w, h = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
            boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def resolve_dirs(args):
    """Return (images_dir, labels_dir, split_tag)."""
    if args.images:
        labels = args.labels or args.images.replace("images", "labels")
        return args.images, labels, os.path.basename(args.images.rstrip("/\\"))
    # derive dataset root from --data (data.yaml) or --root
    if args.root:
        root = args.root
    elif args.data:
        root = os.path.dirname(os.path.abspath(args.data))
    else:
        root = "head_dataset"
    images = os.path.join(root, "images", args.split)
    labels = os.path.join(root, "labels", args.split)
    return images, labels, args.split


def list_images(images_dir):
    out = []
    for root, _d, fnames in os.walk(images_dir):
        for nm in fnames:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, nm))
    return sorted(out)


def label_for(img_path, images_dir, labels_dir):
    rel = os.path.relpath(img_path, images_dir)
    stem = os.path.splitext(rel)[0]
    return os.path.join(labels_dir, stem + ".txt")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Find frames where the head model misses labeled heads.")
    ap.add_argument("--data", help="path to data.yaml (used to locate dataset root)")
    ap.add_argument("--root", help="dataset root (alternative to --data)")
    ap.add_argument("--images", help="explicit images dir (overrides --data/--root/--split)")
    ap.add_argument("--labels", help="explicit labels dir (with --images)")
    ap.add_argument("--split", default="val", help="val|train|test (default val)")
    ap.add_argument("--model", required=True, help="head model weights, e.g. head_n.pt")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="inference size; MUST match the app's HEAD_INFER_IMGSZ (default 960)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence at the app's operating point (default 0.25)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU a prediction needs to count as covering a head (default 0.5)")
    ap.add_argument("--domain", action="append", default=[],
                    help="name:prefix,... (repeatable). Default catwalk/helmet.")
    ap.add_argument("--only", default="catwalk",
                    help="restrict output to this domain, or 'all' (default catwalk)")
    ap.add_argument("--min-misses", type=int, default=1,
                    help="only stage frames with at least this many missed heads (default 1)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of frames staged (worst-first). 0 = no cap.")
    ap.add_argument("--out", default="miss_review", help="output dir (default miss_review)")
    ap.add_argument("--device", default=None, help="cuda device index or 'cpu' (optional)")
    args = ap.parse_args()

    images_dir, labels_dir, split_tag = resolve_dirs(args)
    if not os.path.isdir(images_dir):
        sys.exit("[ERROR] images dir not found: {}".format(images_dir))
    if not os.path.isdir(labels_dir):
        print("[warn] labels dir not found: {} (frames with no label = 0 heads)"
              .format(labels_dir))

    domains = parse_domains(args.domain) if args.domain else [
        ("catwalk", ["catwalk", "new_clip", "cat_"]),
        ("helmet", ["helmet", "clip"]),
    ]

    imgs = list_images(images_dir)
    if not imgs:
        sys.exit("[ERROR] no images in {}".format(images_dir))

    # Lazy imports so the pure helpers above stay testable without these deps.
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit("[ERROR] needs opencv + ultralytics in this env: {}".format(e))

    model = YOLO(args.model)
    predict_kw = dict(imgsz=args.imgsz, conf=args.conf, verbose=False)
    if args.device is not None:
        predict_kw["device"] = args.device

    want = None if args.only.lower() == "all" else args.only
    rows = []          # per-frame stats for the manifest + ranking
    dom_gt = defaultdict(int)
    dom_missed = defaultdict(int)

    print("=" * 70)
    print("Model: {}   split: {}   imgsz={} conf={} iou-match={}".format(
        args.model, split_tag, args.imgsz, args.conf, args.iou))
    print("Scanning {} images ...".format(len(imgs)))

    for i, img_path in enumerate(imgs, 1):
        dom = domain_of(clip_of(img_path), domains)
        im = cv2.imread(img_path)
        if im is None:
            print("[warn] unreadable image, skipping: {}".format(img_path))
            continue
        h, w = im.shape[:2]

        gt = load_gt(label_for(img_path, images_dir, labels_dir), w, h)
        r = model.predict(source=img_path, **predict_kw)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            for xyxy in r.boxes.xyxy.tolist():
                preds.append(tuple(xyxy))

        missed, matched, unmatched = match_misses(gt, preds, args.iou)
        dom_gt[dom] += len(gt)
        dom_missed[dom] += len(missed)

        rows.append({
            "image": img_path, "domain": dom, "w": w, "h": h,
            "gt": len(gt), "matched": len(matched), "missed": len(missed),
            "unmatched_pred": len(unmatched),
            "gt_boxes": gt, "missed_idx": missed, "matched_idx": matched,
            "pred_boxes": preds, "unmatched_pred_idx": unmatched,
        })
        if i % 100 == 0:
            print("  ...{}/{}".format(i, len(imgs)))

    # ---- per-domain recall summary (the honest scoreboard) ----------------- #
    print("=" * 70)
    print("Per-domain recall at this operating point (recall = caught / labeled):")
    for dom in sorted(dom_gt.keys()):
        g = dom_gt[dom]
        miss = dom_missed[dom]
        rec = (g - miss) / g if g else 0.0
        print("  {:<10} {:>6} heads   {:>5} missed   recall {:.3f}".format(
            dom, g, miss, rec))
    print("=" * 70)

    # ---- select + rank frames to stage ------------------------------------ #
    flagged = [r for r in rows
               if r["missed"] >= args.min_misses
               and (want is None or r["domain"] == want)]
    flagged.sort(key=lambda r: (r["missed"], r["unmatched_pred"]), reverse=True)
    if args.limit and len(flagged) > args.limit:
        flagged = flagged[:args.limit]

    if not flagged:
        print("No frames with >= {} missed head(s) in domain '{}'. "
              "Either the model is strong here or you filtered too tightly."
              .format(args.min_misses, args.only))
        return

    # ---- stage originals + labels + annotated previews --------------------- #
    stage_img = os.path.join(args.out, "to_label", "images", split_tag)
    stage_lbl = os.path.join(args.out, "to_label", "labels", split_tag)
    prev_dir = os.path.join(args.out, "previews")
    for d in (stage_img, stage_lbl, prev_dir):
        os.makedirs(d, exist_ok=True)

    RED, GREEN, YELLOW = (0, 0, 255), (0, 200, 0), (0, 220, 220)
    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["rank", "preview", "image", "domain",
                      "labeled_heads", "caught", "missed", "unmatched_pred"])
        for rank, r in enumerate(flagged, 1):
            src = r["image"]
            base = os.path.basename(src)
            stem, ext = os.path.splitext(base)

            # copy original image + its existing label (opens pre-labeled in tool)
            shutil.copy2(src, os.path.join(stage_img, base))
            src_lbl = label_for(src, images_dir, labels_dir)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(stage_lbl, stem + ".txt"))
            else:
                open(os.path.join(stage_lbl, stem + ".txt"), "w").close()

            # annotated preview (NOT for labeling -- a visual guide)
            im = cv2.imread(src)
            for j, b in enumerate(r["gt_boxes"]):
                col = RED if j in r["missed_idx"] else GREEN
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            for j in r["unmatched_pred_idx"]:
                x1, y1, x2, y2 = [int(v) for v in r["pred_boxes"][j]]
                cv2.rectangle(im, (x1, y1), (x2, y2), YELLOW, 1)
            tag = "MISS {}/{}".format(r["missed"], r["gt"])
            cv2.putText(im, tag, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, RED, 2, cv2.LINE_AA)
            # zero-padded miss count -> file browser sorts worst-first
            prev_name = "{:03d}_{}.jpg".format(r["missed"], stem)
            cv2.imwrite(os.path.join(prev_dir, prev_name), im)

            wtr.writerow([rank, prev_name, src, r["domain"],
                          r["gt"], r["matched"], r["missed"], r["unmatched_pred"]])

    total_missed = sum(r["missed"] for r in flagged)
    print("Staged {} frames ({} missed heads) for review.".format(
        len(flagged), total_missed))
    print("  originals + labels:  {}".format(os.path.join(args.out, "to_label")))
    print("  ranked previews:     {}  (RED=missed  GREEN=caught  YELLOW=model box)"
          .format(prev_dir))
    print("  manifest:            {}".format(manifest))
    print("=" * 70)
    print("HOW TO GET LEVERAGE FROM THIS (read before you label):")
    print("- These frames are ALREADY labeled. Re-labeling the same boxes adds")
    print("  nothing -- the value is in what the misses have in COMMON.")
    print("- Open the previews sorted worst-first. Ask: what do the RED boxes")
    print("  share? (tiny/distant heads, back-of-head, heavy occlusion, motion")
    print("  blur, one specific catwalk clip/lighting?)  That pattern is your")
    print("  labeling target.")
    print("- Then go SAMPLE NEW catwalk frames of that kind (sample_frames.py),")
    print("  label them, and add them to train. New hard examples move recall;")
    print("  recopying old ones does not.")
    print("- YELLOW boxes with no RED/GREEN under them are model detections with")
    print("  no label: either a real head you FORGOT to label (add it -- an")
    print("  unlabeled head teaches 'not a head') or a false positive (ignore).")
    print("- Re-run this exact command after retraining; compare the per-domain")
    print("  recall line above. That before/after is the whole point.")
    print("=" * 70)


if __name__ == "__main__":
    main()
