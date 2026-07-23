r"""
count_dataset.py -- count clips, frames, and labeled heads per domain and split,
for the dataset tables in Section 5.

Domain rule (as specified): a clip is
  * catwalk  if its name starts with 'catwalk'
  * helmet   if its name starts with 'helmet'
  * mixed    otherwise
Clip identity is derived from each frame's filename by stripping the trailing
frame index (one '_<digits>' group), IDENTICAL to eval_compare.py's clip_of, so
these counts match the domain bucketing used everywhere else in the pipeline.
Example: 'catwalk_00_000123.jpg' -> clip 'catwalk_00' -> domain 'catwalk'.

USAGE (point it at the split roots you have):
    python count_dataset.py ^
        --split train:head_dataset/images/train ^
        --split val:head_dataset/images/val ^
        --split test:head_test/images/val

Each --split is name:images_dir. Labels are looked up by swapping 'images' ->
'labels' in the path (YOLO layout) unless --labels is given per split as
name:images_dir:labels_dir. Pass --list-clips to print every clip with its
frame count so you can sanity-check the grouping.
"""

import argparse
import os
import re
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")   # same as eval_compare.py


def clip_of(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


# Some helmet-cam clips are date-named rather than 'helmet_*' (they came from a
# different capture session). List them here so they classify as helmet
# everywhere. Keep this in sync with eval_compare.py's --domain patterns.
HELMET_EXTRA_PREFIXES = ("test0706", "test0707")

def domain_of(clip):
    c = clip.lower()
    if c.startswith("catwalk"):
        return "catwalk"
    if c.startswith("helmet") or c.startswith(HELMET_EXTRA_PREFIXES):
        return "helmet"
    return "mixed"


def count_heads(img_path, images_dir, labels_dir):
    """Number of label rows for one image (0 if no label file)."""
    rel = os.path.relpath(img_path, images_dir)
    lbl = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")
    if not os.path.isfile(lbl):
        return 0
    n = 0
    with open(lbl) as f:
        for ln in f:
            if len(ln.split()) >= 5:
                n += 1
    return n


def parse_split(entry):
    parts = entry.split(":")
    if len(parts) == 2:
        name, images_dir = parts
        labels_dir = None
    elif len(parts) == 3:
        name, images_dir, labels_dir = parts
    else:
        raise SystemExit("[ERROR] bad --split '{}' (need name:images_dir "
                         "or name:images_dir:labels_dir)".format(entry))
    if labels_dir is None:
        # YOLO convention: images/... -> labels/...
        if os.sep + "images" + os.sep in images_dir or images_dir.endswith(os.sep + "images"):
            labels_dir = images_dir.replace(os.sep + "images", os.sep + "labels")
        else:
            labels_dir = images_dir.replace("/images", "/labels").replace("\\images", "\\labels")
    return name, images_dir, labels_dir


def gather(images_dir):
    imgs = []
    for root, _d, files in os.walk(images_dir):
        for nm in files:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(root, nm))
    return sorted(imgs)


def main():
    ap = argparse.ArgumentParser(description="Count clips/frames/heads per domain and split.")
    ap.add_argument("--split", action="append", dest="splits", required=True,
                    help="name:images_dir  (or name:images_dir:labels_dir). Repeatable.")
    ap.add_argument("--no-heads", action="store_true",
                    help="skip counting labeled heads (faster; only clips + frames)")
    ap.add_argument("--list-clips", action="store_true",
                    help="print every clip with its frame/head counts for verification")
    ap.add_argument("--out", default=None, help="optional CSV to write the per-domain table")
    args = ap.parse_args()

    # results[split][domain] = {clips:set, frames:int, heads:int}
    results = defaultdict(lambda: defaultdict(lambda: {"clips": set(), "frames": 0, "heads": 0}))
    per_clip = defaultdict(lambda: {"domain": None, "frames": 0, "heads": 0, "split": None})

    for entry in args.splits:
        name, images_dir, labels_dir = parse_split(entry)
        if not os.path.isdir(images_dir):
            raise SystemExit("[ERROR] images dir not found: {}".format(images_dir))
        imgs = gather(images_dir)
        if not imgs:
            print("[warn] no images under {}".format(images_dir))
        for p in imgs:
            clip = clip_of(p)
            dom = domain_of(clip)
            cell = results[name][dom]
            cell["clips"].add(clip)
            cell["frames"] += 1
            key = (name, clip)
            per_clip[key]["domain"] = dom
            per_clip[key]["split"] = name
            per_clip[key]["frames"] += 1
            if not args.no_heads:
                h = count_heads(p, images_dir, labels_dir)
                cell["heads"] += h
                per_clip[key]["heads"] += h

    # ---- per-split, per-domain table ----
    domains_order = ["helmet", "catwalk", "mixed"]
    split_order = [parse_split(e)[0] for e in args.splits]

    print("\n" + "=" * 72)
    print("CLIP / FRAME / HEAD COUNTS BY SPLIT AND DOMAIN")
    print("=" * 72)
    header = "{:<8} {:<9} {:>6} {:>8} {:>8}".format("split", "domain", "clips", "frames", "heads")
    print(header)
    print("-" * 72)
    grand = {"clips": 0, "frames": 0, "heads": 0}
    csv_rows = [("split", "domain", "clips", "frames", "heads")]
    for sname in split_order:
        sub_c = sub_f = sub_h = 0
        for dom in domains_order:
            if dom not in results[sname]:
                continue
            cell = results[sname][dom]
            nc, nf, nh = len(cell["clips"]), cell["frames"], cell["heads"]
            print("{:<8} {:<9} {:>6} {:>8} {:>8}".format(
                sname, dom, nc, nf, (nh if not args.no_heads else "-")))
            csv_rows.append((sname, dom, nc, nf, nh))
            sub_c += nc; sub_f += nf; sub_h += nh
        print("{:<8} {:<9} {:>6} {:>8} {:>8}".format(
            sname, "TOTAL", sub_c, sub_f, (sub_h if not args.no_heads else "-")))
        csv_rows.append((sname, "TOTAL", sub_c, sub_f, sub_h))
        print("-" * 72)
        grand["clips"] += sub_c; grand["frames"] += sub_f; grand["heads"] += sub_h

    print("{:<8} {:<9} {:>6} {:>8} {:>8}".format(
        "ALL", "", grand["clips"], grand["frames"],
        (grand["heads"] if not args.no_heads else "-")))

    # cross-split domain totals (useful for "helmet clips across the dataset")
    print("\n" + "=" * 72)
    print("DOMAIN TOTALS ACROSS ALL SPLITS")
    print("=" * 72)
    print("{:<9} {:>6} {:>8} {:>8}".format("domain", "clips", "frames", "heads"))
    dom_tot = defaultdict(lambda: {"clips": set(), "frames": 0, "heads": 0})
    for sname in split_order:
        for dom, cell in results[sname].items():
            dom_tot[dom]["clips"] |= cell["clips"]
            dom_tot[dom]["frames"] += cell["frames"]
            dom_tot[dom]["heads"] += cell["heads"]
    for dom in domains_order:
        if dom not in dom_tot:
            continue
        d = dom_tot[dom]
        print("{:<9} {:>6} {:>8} {:>8}".format(
            dom, len(d["clips"]), d["frames"], (d["heads"] if not args.no_heads else "-")))
    # note: a clip name appearing in two splits would be double-counted here;
    # clip-level splitting should prevent that. Flag it if it happens.
    all_clip_names = defaultdict(set)
    for (sname, clip) in per_clip:
        all_clip_names[clip].add(sname)
    leaked = {c: s for c, s in all_clip_names.items() if len(s) > 1}
    if leaked:
        print("\n[WARN] {} clip name(s) appear in MORE THAN ONE split -- possible "
              "leakage (or reused names):".format(len(leaked)))
        for c, s in sorted(leaked.items()):
            print("   {} -> {}".format(c, ", ".join(sorted(s))))

    # ---- optional per-clip listing ----
    if args.list_clips:
        print("\n" + "=" * 72)
        print("PER-CLIP DETAIL (verify the grouping is correct)")
        print("=" * 72)
        print("{:<8} {:<9} {:<28} {:>7} {:>7}".format("split", "domain", "clip", "frames", "heads"))
        for (sname, clip) in sorted(per_clip, key=lambda k: (k[0], per_clip[k]["domain"], k[1])):
            info = per_clip[(sname, clip)]
            print("{:<8} {:<9} {:<28} {:>7} {:>7}".format(
                sname, info["domain"], clip, info["frames"],
                (info["heads"] if not args.no_heads else "-")))

    # ---- optional CSV ----
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows(csv_rows)
        print("\nWrote per-domain table to {}".format(args.out))


if __name__ == "__main__":
    main()
