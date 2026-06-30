r"""
check_split.py -- inspect a YOLO dataset's train/val/test split BY CLIP.

It groups every image by its clip (the filename minus the trailing _NNNNNN frame
index, e.g. clip1_000039.jpg -> clip 'clip1'), then reports per split:
  * how many clips and frames
  * frames per clip
and -- the important part -- whether any clip appears in more than one split.
A clip in both train and val means near-identical adjacent frames leak across the
split, so your val/test metrics are optimistic and can't be trusted.

This script only READS; it never moves or changes files.

USAGE:
    python check_split.py                       # assumes ./head_dataset
    python check_split.py --root head_dataset
    python check_split.py --train head_dataset/images/train --val head_dataset/images/val

If you also pass --domains, give substrings that identify each domain, e.g.:
    python check_split.py --root head_dataset --domains helmet catwalk clip
and it will report how many clips/frames per domain per split, so you can see
whether each domain is represented in val/test (needed to measure per-domain
recall).
"""

import argparse
import os
import re
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# clip = filename with a trailing _<digits> stripped off
CLIP_RE = re.compile(r"^(.*?)_\d+$")


def clip_of(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def list_images(d):
    if not d or not os.path.isdir(d):
        return []
    out = []
    for root, _dirs, names in os.walk(d):
        for nm in names:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, nm))
    return out


def clips_in(images):
    """Return {clip: frame_count} for a list of image paths."""
    d = defaultdict(int)
    for p in images:
        d[clip_of(p)] += 1
    return d


def main():
    ap = argparse.ArgumentParser(description="Check a YOLO dataset split by clip.")
    ap.add_argument("--root", default="head_dataset",
                    help="dataset root containing images/train, images/val[, images/test]")
    ap.add_argument("--train", default=None, help="explicit train images dir (overrides --root)")
    ap.add_argument("--val", default=None, help="explicit val images dir")
    ap.add_argument("--test", default=None, help="explicit test images dir")
    ap.add_argument("--domains", nargs="*", default=None,
                    help="substrings identifying domains, e.g. helmet catwalk clip")
    args = ap.parse_args()

    splits = {}
    if args.train or args.val or args.test:
        if args.train: splits["train"] = args.train
        if args.val:   splits["val"] = args.val
        if args.test:  splits["test"] = args.test
    else:
        for s in ("train", "val", "test"):
            d = os.path.join(args.root, "images", s)
            if os.path.isdir(d):
                splits[s] = d

    if not splits:
        print("[!] no split folders found. Pass --root or --train/--val/--test.")
        print("    looked under: {}/images/{{train,val,test}}".format(args.root))
        return

    split_clips = {}
    print("=" * 60)
    for s, d in splits.items():
        imgs = list_images(d)
        cc = clips_in(imgs)
        split_clips[s] = cc
        print("[{}]  {}".format(s.upper(), d))
        print("    {} image(s) across {} clip(s)".format(len(imgs), len(cc)))
        for clip in sorted(cc):
            print("      {:<28} {:>5} frames".format(clip, cc[clip]))
        print("-" * 60)

    # ---- leakage check: any clip in more than one split? --------------------
    all_clips = set()
    for cc in split_clips.values():
        all_clips |= set(cc.keys())
    leaks = []
    for clip in sorted(all_clips):
        where = [s for s in split_clips if clip in split_clips[s]]
        if len(where) > 1:
            leaks.append((clip, where))

    print("LEAKAGE CHECK")
    if leaks:
        print("  [!!] These clips appear in MORE THAN ONE split -- near-identical")
        print("       frames leak across the split, so your metrics are inflated:")
        for clip, where in leaks:
            print("       {:<28} in {}".format(clip, " + ".join(where)))
        print("  Fix: move WHOLE clips to a single split (split by clip, not frame).")
    else:
        print("  OK: every clip lives in exactly one split. No frame leakage.")
    print("-" * 60)

    # ---- optional per-domain breakdown --------------------------------------
    if args.domains:
        print("PER-DOMAIN BREAKDOWN (by filename substring)")
        for s in split_clips:
            print("  [{}]".format(s.upper()))
            matched_frames = 0
            for dom in args.domains:
                clips = [c for c in split_clips[s] if dom.lower() in c.lower()]
                frames = sum(split_clips[s][c] for c in clips)
                matched_frames += frames
                print("    {:<12} {:>3} clip(s), {:>5} frame(s)".format(
                    dom, len(clips), frames))
            total = sum(split_clips[s].values())
            if matched_frames < total:
                print("    {:<12} {:>3} clip(s), {:>5} frame(s)  (unmatched by any "
                      "domain substring)".format("(other)", "?", total - matched_frames))
        print("  -> For per-domain recall, each domain should appear in val/test,")
        print("     not only in train.")
        print("-" * 60)

    print("Reminder: a privacy blur cares about RECALL per domain. A pooled score")
    print("can hide one domain underperforming behind another doing well.")


if __name__ == "__main__":
    main()
