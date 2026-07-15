r"""
prepare_benchmark.py -- convert public head-detection benchmarks (VOC-style
XML annotations) into YOLO layout so eval_compare.py can score models on them.

Supports the standard layouts of:
  * SCUT-HEAD Part A / Part B   (JPEGImages/ Annotations/ ImageSets/Main/*.txt)
  * HollywoodHeads              (JPEGImages/ Annotations/ Splits/*.txt)
  * Casablanca and other VOC-style sets, if arranged the same way
    (use --images-dir/--ann-dir/--split-file to point at nonstandard names)

USAGE:
    python prepare_benchmark.py --src SCUT_HEAD_Part_B --out bench/scut_b --split test
    python prepare_benchmark.py --src HollywoodHeads   --out bench/hh     --split test
    python prepare_benchmark.py --src Casablanca --images-dir frames ^
        --ann-dir annotations --split-file splits/test.txt --out bench/casa --split test

Then evaluate (same operating point as the paper):
    python eval_compare.py --data bench/scut_b/data.yaml --split test ^
        --model head_n:head_n.pt --model head_s:head_s.pt --model head_m:head_m.pt ^
        --imgsz 960 --conf 0.25

Every head-like class name is mapped to class 0 ('head'). Boxes are clamped to
the image and degenerate boxes are dropped (counted in the report).
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HEAD_NAMES = {"head", "person_head", "person-head", "human_head"}
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args():
    ap = argparse.ArgumentParser(description="Convert a VOC-style head benchmark to YOLO.")
    ap.add_argument("--src", required=True, help="benchmark root folder")
    ap.add_argument("--out", required=True, help="output dataset folder")
    ap.add_argument("--split", default="test",
                    help="which split to convert (default test); matched against "
                         "<src>/ImageSets/Main/<split>.txt or <src>/Splits/<split>.txt")
    ap.add_argument("--images-dir", default=None,
                    help="images subfolder if not JPEGImages/")
    ap.add_argument("--ann-dir", default=None,
                    help="annotations subfolder if not Annotations/")
    ap.add_argument("--split-file", default=None,
                    help="explicit split list file (one image stem per line); "
                         "overrides --split lookup")
    ap.add_argument("--copy-images", action="store_true",
                    help="copy images instead of symlinking (use on Windows or "
                         "across drives)")
    ap.add_argument("--difficult", choices=["drop", "keep"], default="drop",
                    help="how to treat VOC 'difficult' objects (default: drop). "
                         "Published HollywoodHeads results (Vu et al., 2015) score "
                         "difficult heads as neither TP nor FP; YOLO-format labels "
                         "cannot express ignore regions, so 'drop' removes them from "
                         "GT -- recall then matches their protocol, but detections "
                         "on difficult heads count as FP, making your precision/mAP "
                         "conservative relative to published numbers. Note this in "
                         "any comparison table.")
    return ap.parse_args()


def find_dir(src: Path, override, candidates):
    if override:
        d = src / override
        if not d.is_dir():
            sys.exit("[ERROR] folder not found: {}".format(d))
        return d
    for c in candidates:
        if (src / c).is_dir():
            return src / c
    sys.exit("[ERROR] none of {} found under {} -- pass the folder explicitly"
             .format(candidates, src))


def find_split_file(src: Path, override, split):
    if override:
        f = Path(override)
        if not f.is_file():
            f = src / override
        if not f.is_file():
            sys.exit("[ERROR] split file not found: {}".format(override))
        return f
    for c in ["ImageSets/Main/{}.txt".format(split), "Splits/{}.txt".format(split)]:
        if (src / c).is_file():
            return src / c
    return None  # fall back to "every annotated image"


def voc_boxes(xml_path: Path):
    """Yield (name, difficult, xmin, ymin, xmax, ymax, img_w, img_h) from one VOC xml."""
    root = ET.parse(str(xml_path)).getroot()
    size = root.find("size")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        difficult = (obj.findtext("difficult") or "0").strip() in ("1", "true")
        bb = obj.find("bndbox")
        if bb is None:
            # Some releases (e.g. Casablanca) contain <object> entries without
            # a <bndbox>. Signal it with name "(no bndbox)" so main() can count.
            yield (name, difficult, None, None, None, None, w, h)
            continue
        coords = [bb.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")]
        if any(c is None for c in coords):
            yield (name, difficult, None, None, None, None, w, h)
            continue
        yield (name, difficult,
               float(coords[0]), float(coords[1]),
               float(coords[2]), float(coords[3]), w, h)


def main():
    args = parse_args()
    src = Path(args.src)
    if not src.is_dir():
        sys.exit("[ERROR] source folder not found: {}".format(src))

    img_dir = find_dir(src, args.images_dir, ["JPEGImages", "Images", "images", "frames"])
    ann_dir = find_dir(src, args.ann_dir, ["Annotations", "annotations"])
    split_file = find_split_file(src, args.split_file, args.split)

    if split_file:
        stems = [ln.strip().split()[0] for ln in
                 split_file.read_text().splitlines() if ln.strip()]
        print("Split '{}': {} entries from {}".format(args.split, len(stems), split_file))
    else:
        stems = sorted(p.stem for p in ann_dir.glob("*.xml"))
        print("[warn] no split file found -- converting ALL {} annotated images. "
              "If the benchmark defines an official test split, use it, or your "
              "numbers will not be comparable to published ones.".format(len(stems)))

    out = Path(args.out)
    out_img = out / "images" / args.split
    out_lbl = out / "labels" / args.split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_imgs = n_boxes = n_dropped = n_nonhead = n_missing = n_difficult = n_noboxes = 0
    nonhead_names = {}
    for stem in stems:
        xml_path = ann_dir / (stem + ".xml")
        img_path = None
        for ext in IMG_EXTS:
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        if not xml_path.exists() or img_path is None:
            n_missing += 1
            continue

        lines = []
        n_difficult_here = 0
        for name, difficult, x1, y1, x2, y2, w, h in voc_boxes(xml_path):
            if x1 is None:
                n_noboxes += 1
                continue
            if name not in HEAD_NAMES:
                n_nonhead += 1
                nonhead_names[name] = nonhead_names.get(name, 0) + 1
                continue
            if difficult and args.difficult == "drop":
                n_difficult_here += 1
                continue
            # clamp to image, drop degenerate
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                n_dropped += 1
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(cx, cy, bw, bh))
        n_difficult += n_difficult_here

        (out_lbl / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        dest = out_img / img_path.name
        if not dest.exists():
            if args.copy_images:
                shutil.copy(str(img_path), str(dest))
            else:
                try:
                    dest.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy(str(img_path), str(dest))  # e.g. Windows w/o privilege
        n_imgs += 1
        n_boxes += len(lines)

    (out / "data.yaml").write_text(
        "path: {}\nnames:\n  0: head\n".format(out.resolve()))

    print("\nConverted: {} images, {} head boxes -> {}".format(n_imgs, n_boxes, out))
    if n_missing:
        print("  [warn] {} split entries missing an image or xml".format(n_missing))
    if n_dropped:
        print("  [warn] {} degenerate boxes dropped".format(n_dropped))
    if n_noboxes:
        print("  [note] {} <object> entries had no/incomplete <bndbox> and were skipped".format(n_noboxes))
    if n_difficult:
        print("  [note] {} 'difficult' heads {} (policy: --difficult {}). Published"
              " protocols score these as ignore regions; with 'drop', your recall"
              " matches published protocol but precision/mAP is conservative."
              .format(n_difficult,
                      "removed from GT" if args.difficult == "drop" else "kept as GT",
                      args.difficult))
    if n_nonhead:
        print("  [note] {} non-head objects ignored: {}".format(
            n_nonhead, ", ".join("{} x{}".format(k, v)
                                 for k, v in sorted(nonhead_names.items()))))
    print("\nEvaluate with:")
    print("  python eval_compare.py --data {} --split {} --model head_s:head_s.pt "
          "--imgsz 960 --conf 0.25".format(out / "data.yaml", args.split))


if __name__ == "__main__":
    main()
