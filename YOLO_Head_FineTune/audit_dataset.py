r"""
audit_dataset.py -- READ-ONLY inventory of a YOLO dataset before cleanup.

Finds the problems that quietly corrupt training/eval:
  * EXACT DUPLICATE images (identical content, possibly under different names)
    -- these inflate counts and can leak across train/val.
  * images with a MISSING label .txt, and ORPHAN labels (label with no image).
  * per-clip frame counts per split (spot a clip sampled twice, or a clip that
    appears in more than one split).

It changes NOTHING. Review its output, then use it to decide what to delete/move.

USAGE:
    python audit_dataset.py --root head_dataset
"""

import argparse
import hashlib
import os
import re
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")
SPLITS = ("train", "val", "test")


def clip_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.hexdigest()


def images_in(root, split):
    d = os.path.join(root, "images", split)
    out = []
    if os.path.isdir(d):
        for r, _dd, names in os.walk(d):
            for nm in names:
                if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                    out.append(os.path.join(r, nm))
    return out


def label_for(img_path, root, split):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(root, "labels", split, stem + ".txt")


def main():
    ap = argparse.ArgumentParser(description="Read-only YOLO dataset audit.")
    ap.add_argument("--root", default="head_dataset")
    args = ap.parse_args()

    present = [s for s in SPLITS if os.path.isdir(os.path.join(args.root, "images", s))]
    if not present:
        print("[!] no images/train|val|test under {}".format(args.root))
        return

    all_imgs = []
    per_split = {}
    for s in present:
        imgs = images_in(args.root, s)
        per_split[s] = imgs
        all_imgs.extend((s, p) for p in imgs)

    # ---- per-clip counts ----------------------------------------------------
    print("=" * 66)
    print("PER-CLIP FRAME COUNTS")
    clip_split = defaultdict(dict)
    for s in present:
        cc = defaultdict(int)
        for p in per_split[s]:
            cc[clip_of(p)] += 1
        for c in sorted(cc):
            clip_split[c][s] = cc[c]
    for c in sorted(clip_split):
        parts = ", ".join("{}={}".format(s, clip_split[c][s]) for s in clip_split[c])
        multi = "  <-- IN MULTIPLE SPLITS" if len(clip_split[c]) > 1 else ""
        print("  {:<26} {}{}".format(c, parts, multi))
    print("-" * 66)

    # ---- exact duplicate images (content hash) ------------------------------
    print("EXACT DUPLICATE IMAGES (identical content)")
    by_hash = defaultdict(list)
    for s, p in all_imgs:
        h = file_hash(p)
        if h:
            by_hash[h].append((s, p))
    dup_groups = [v for v in by_hash.values() if len(v) > 1]
    if dup_groups:
        total_extra = sum(len(g) - 1 for g in dup_groups)
        print("  {} group(s) of identical images; {} redundant copies.".format(
            len(dup_groups), total_extra))
        cross = 0
        for g in dup_groups:
            splits = {s for s, _ in g}
            flag = "  <-- ACROSS SPLITS (leakage!)" if len(splits) > 1 else ""
            if len(splits) > 1:
                cross += 1
            print("   duplicate set{}:".format(flag))
            for s, p in g:
                print("      [{}] {}".format(s, os.path.relpath(p, args.root)))
        if cross:
            print("  [!!] {} duplicate set(s) span train+val -- that IS leakage.".format(cross))
    else:
        print("  none.")
    print("-" * 66)

    # ---- missing labels / orphan labels -------------------------------------
    print("LABEL INTEGRITY")
    missing = []
    for s in present:
        for p in per_split[s]:
            if not os.path.exists(label_for(p, args.root, s)):
                missing.append((s, p))
    orphans = []
    for s in present:
        ld = os.path.join(args.root, "labels", s)
        if os.path.isdir(ld):
            img_stems = {os.path.splitext(os.path.basename(p))[0] for p in per_split[s]}
            for nm in os.listdir(ld):
                if nm.endswith(".txt"):
                    stem = nm[:-4]
                    if stem not in img_stems:
                        orphans.append((s, os.path.join(ld, nm)))
    print("  images with NO label .txt: {}".format(len(missing)))
    for s, p in missing[:10]:
        print("      [{}] {}".format(s, os.path.basename(p)))
    if len(missing) > 10:
        print("      ... and {} more".format(len(missing) - 10))
    print("  orphan labels (no image): {}".format(len(orphans)))
    for s, p in orphans[:10]:
        print("      [{}] {}".format(s, os.path.basename(p)))
    print("=" * 66)
    print("Nothing was changed. Use this to decide what to delete/rename/move.")


if __name__ == "__main__":
    main()
