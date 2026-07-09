r"""
sample_one_per_clip.py -- copy ONE frame from each clip into a review folder,
so you can eyeball which domain each clip actually is before renaming.

The middle frame of each clip is copied and prefixed with its split, e.g.
    review_clips/train__clip5.jpg
    review_clips/val__new_clip4.jpg
Open the folder, glance at each image, and confirm your domain mapping from what
you SEE rather than from memory.

USAGE:
    python sample_one_per_clip.py --root head_dataset --out review_clips
"""

import argparse
import os
import re
import shutil
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")
SPLITS = ("train", "val", "test")


def clip_of(name):
    stem = os.path.splitext(name)[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="head_dataset")
    ap.add_argument("--out", default="review_clips")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for s in SPLITS:
        d = os.path.join(args.root, "images", s)
        if not os.path.isdir(d):
            continue
        by_clip = defaultdict(list)
        for nm in os.listdir(d):
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                by_clip[clip_of(nm)].append(nm)
        for clip, files in by_clip.items():
            files.sort()
            mid = files[len(files) // 2]   # a representative middle frame
            ext = os.path.splitext(mid)[1]
            dst = os.path.join(args.out, "{}__{}{}".format(s, clip, ext))
            shutil.copy(os.path.join(d, mid), dst)
            n += 1
    print("[done] copied {} sample frame(s) (one per clip) to {}/".format(n, args.out))
    print("Open that folder, confirm each clip's domain, THEN run rename_clips.py.")


if __name__ == "__main__":
    main()
