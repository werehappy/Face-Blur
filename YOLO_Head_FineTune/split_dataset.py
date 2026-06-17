r"""
split_dataset.py  --  turn your labeled frames into the train/val layout YOLO
wants, and write data.yaml. Splits BY CLIP so near-identical frames from one
moment never land in both train and val (the #1 way to get great metrics and
bad real-world results).

INPUT: the folder you labeled in Yolo_Label (images + matching .txt + classes.txt),
e.g. dataset_raw/

OUTPUT:
    head_dataset/
    ├── images/train , images/val
    ├── labels/train , labels/val
    └── data.yaml          (absolute path, single class 'head')

USAGE (any env with python; no heavy deps needed):
    python split_dataset.py dataset_raw
    python split_dataset.py dataset_raw --out head_dataset --val 0.2

Clip grouping: frames from sample_frames.py are named <clip>_<frame>.jpg, so the
clip id is the name with the trailing _<digits> stripped. Whole clips go to
train or val together. With only one clip it falls back to a random per-frame
split and warns you (some leakage; pull frames from more clips for a real model).
"""
import os
import re
import sys
import glob
import random
import shutil
import argparse

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CLIP_RE = re.compile(r"^(.*?)[_-]?\d+$")   # strip trailing digits to get clip id


def list_images(folder):
    out = []
    for ext in IMG_EXTS:
        out.extend(glob.glob(os.path.join(folder, "*" + ext)))
        out.extend(glob.glob(os.path.join(folder, "*" + ext.upper())))
    return sorted(set(out))


def clip_id(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    m = CLIP_RE.match(stem)
    return m.group(1) if m and m.group(1) else stem


def main():
    ap = argparse.ArgumentParser(description="Split labeled frames into YOLO train/val")
    ap.add_argument("folder", help="labeled folder (images + .txt), e.g. dataset_raw")
    ap.add_argument("--out", default="head_dataset", help="output dataset root (default head_dataset)")
    ap.add_argument("--val", type=float, default=0.2, help="val fraction (default 0.2)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print("[!] not a folder:", args.folder); sys.exit(1)

    imgs = list_images(args.folder)
    if not imgs:
        print("[!] no images found in", args.folder); sys.exit(1)

    # how many actually have labels?
    labeled = [im for im in imgs if os.path.exists(os.path.splitext(im)[0] + ".txt")]
    print("[*] %d images, %d have a .txt label file" % (len(imgs), len(labeled)))
    if not labeled:
        print("[!] No .txt files next to your images. Did you save in Yolo_Label?")
        print("    Yolo_Label writes <image>.txt as you navigate frames.")
        sys.exit(1)

    # group by clip
    groups = {}
    for im in imgs:
        groups.setdefault(clip_id(im), []).append(im)
    clips = sorted(groups)
    random.seed(args.seed)

    val_set = set()
    if len(clips) >= 2:
        random.shuffle(clips)
        n_val = max(1, round(len(clips) * args.val))
        val_clips = set(clips[:n_val])
        for c in val_clips:
            val_set.update(groups[c])
        print("[*] %d clips: %d train / %d val (split BY CLIP -> no leakage)"
              % (len(clips), len(clips) - len(val_clips), len(val_clips)))
        print("    val clips:", ", ".join(sorted(val_clips)))
    else:
        print("[!] Only one clip detected -> falling back to a random per-frame split.")
        print("    There WILL be some train/val leakage; for a trustworthy model,")
        print("    sample frames from several different clips.")
        shuffled = imgs[:]
        random.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * args.val))
        val_set = set(shuffled[:n_val])

    # make dirs
    out = args.out
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    n = {"train": 0, "val": 0}
    for im in imgs:
        split = "val" if im in val_set else "train"
        # image
        shutil.copy2(im, os.path.join(out, "images", split, os.path.basename(im)))
        # label (empty file if none -> treated as a negative/background frame)
        txt_src = os.path.splitext(im)[0] + ".txt"
        txt_dst = os.path.join(out, "labels", split,
                               os.path.splitext(os.path.basename(im))[0] + ".txt")
        if os.path.exists(txt_src):
            shutil.copy2(txt_src, txt_dst)
        else:
            open(txt_dst, "w").close()
        n[split] += 1

    # data.yaml (absolute path so it works from anywhere)
    root_abs = os.path.abspath(out).replace("\\", "/")
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: %s\n" % root_abs)
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: head\n")

    print()
    print("[DONE] train=%d  val=%d  ->  %s/" % (n["train"], n["val"], out))
    print("       data.yaml written: %s" % yaml_path)
    print()
    print("TRAIN NOW (in your training env):")
    print("  conda activate yolotrain")
    print("  yolo detect train model=yolo11n.pt data=%s epochs=100 imgsz=960 batch=8 patience=30 name=head_v1"
          % yaml_path.replace("\\", "/"))


if __name__ == "__main__":
    main()
