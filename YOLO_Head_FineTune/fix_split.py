r"""
fix_split.py -- move WHOLE clips between train/val/test (images + labels).

Use this to repair a frame-leaked split: a clip whose frames are spread across
two splits must live entirely in one. It moves each image AND its matching label
.txt together, so nothing gets orphaned.

DRY-RUN BY DEFAULT. It prints what it WOULD move and changes nothing until you
add --apply.

Clip = filename minus the trailing _NNNNNN frame index, matched EXACTLY
(so 'clip1' does not match 'new_clip1').

USAGE:
    # see the plan (no changes):
    python fix_split.py --root head_dataset --to-train new_clip4
    # actually do it:
    python fix_split.py --root head_dataset --to-train new_clip4 --apply

    # you can target several clips / several destinations at once:
    python fix_split.py --root head_dataset --to-val clip5 clip16 --to-train new_clip4 --apply
"""

import argparse
import os
import re
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")
SPLITS = ("train", "val", "test")


def clip_of(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def images_in(root, split):
    d = os.path.join(root, "images", split)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, n) for n in os.listdir(d)
            if os.path.splitext(n)[1].lower() in IMG_EXTS]


def plan_moves(root, assignments):
    """assignments: {clip: target_split}. Returns list of (img_src, img_dst,
    lbl_src, lbl_dst) for every frame that needs to move."""
    moves = []
    for split in SPLITS:
        for img in images_in(root, split):
            c = clip_of(img)
            target = assignments.get(c)
            if target and target != split:
                stem = os.path.splitext(os.path.basename(img))[0]
                ext = os.path.splitext(img)[1]
                img_dst = os.path.join(root, "images", target, stem + ext)
                lbl_src = os.path.join(root, "labels", split, stem + ".txt")
                lbl_dst = os.path.join(root, "labels", target, stem + ".txt")
                moves.append((img, img_dst, lbl_src, lbl_dst))
    return moves


def main():
    ap = argparse.ArgumentParser(description="Move whole clips between splits.")
    ap.add_argument("--root", default="head_dataset")
    ap.add_argument("--to-train", nargs="*", default=[], help="clips to move into train")
    ap.add_argument("--to-val", nargs="*", default=[], help="clips to move into val")
    ap.add_argument("--to-test", nargs="*", default=[], help="clips to move into test")
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry-run)")
    args = ap.parse_args()

    assignments = {}
    for c in args.to_train: assignments[c] = "train"
    for c in args.to_val:   assignments[c] = "val"
    for c in args.to_test:  assignments[c] = "test"
    if not assignments:
        print("[!] nothing to do -- pass --to-train/--to-val/--to-test CLIP ...")
        return

    moves = plan_moves(args.root, assignments)
    if not moves:
        print("[ok] nothing to move: those clips are already in their target "
              "split (or weren't found).")
        return

    # summarize per clip
    by_clip = {}
    for (img, img_dst, _ls, _ld) in moves:
        c = clip_of(img)
        by_clip.setdefault(c, {"n": 0, "to": assignments[c]})
        by_clip[c]["n"] += 1

    print("Planned moves ({}):".format("APPLYING" if args.apply else "DRY-RUN"))
    for c, info in sorted(by_clip.items()):
        print("  {:<28} {:>4} frame(s) -> {}".format(c, info["n"], info["to"]))

    missing_labels = 0
    for (img, img_dst, lbl_src, lbl_dst) in moves:
        if not os.path.exists(lbl_src):
            missing_labels += 1
    if missing_labels:
        print("  [note] {} image(s) have no matching label .txt (background "
              "frames, or a labels/ mismatch). Their images still move; check "
              "if that's expected.".format(missing_labels))

    if not args.apply:
        print("\nDRY-RUN only. Re-run with --apply to perform these moves.")
        return

    moved_i = moved_l = 0
    for (img, img_dst, lbl_src, lbl_dst) in moves:
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        shutil.move(img, img_dst)
        moved_i += 1
        if os.path.exists(lbl_src):
            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
            shutil.move(lbl_src, lbl_dst)
            moved_l += 1
    print("\n[done] moved {} image(s) and {} label(s).".format(moved_i, moved_l))
    print("Re-run check_split.py to confirm the leak is gone.")


if __name__ == "__main__":
    main()
