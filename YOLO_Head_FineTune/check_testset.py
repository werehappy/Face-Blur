#!/usr/bin/env python3
r"""
check_testset.py -- verify a held-aside test set is VALID before you trust any
recall number from eval_compare.py.

It answers the four questions that have burned this project:
  1. Bucketing:   does every clip fall into a named reference source, or into
                  "(other)"? (the cause of your pooled-only results)
  2. Labels:      does every image have a NON-EMPTY .txt? (unlabeled frames
                  silently zero recall)
  3. Sample size: how many labeled HEAD INSTANCES per source? (30 is noise;
                  aim for a few hundred)
  4. Provenance:  do any test clips also appear in the TRAINING set? (that is
                  leakage, and it inflates every metric)

USAGE
-----
    python check_testset.py --test-root head_test --split val \
        --train-root head_dataset \
        --domain firstperson:firstperson --domain topview:topview \
        --min-instances 200

Domain rules use the same name:prefix1,prefix2 format as eval_compare.py, and
clip names are parsed the same way (frame index stripped from the tail).
"""

import argparse
import os
import re
import sys
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")


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


def count_instances(label_path):
    if not os.path.exists(label_path):
        return None  # missing file (distinct from empty)
    try:
        with open(label_path) as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return None


def list_images(images_dir):
    out = []
    for root, _d, names in os.walk(images_dir):
        for nm in names:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, nm))
    return out


def clip_names_under(root):
    """All clip names found in any images/* split of a dataset root."""
    clips = set()
    base = os.path.join(root, "images")
    if not os.path.isdir(base):
        return clips
    for img in list_images(base):
        clips.add(clip_of(img))
    return clips


def main():
    ap = argparse.ArgumentParser(description="Validate a held-aside test set.")
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--train-root", default=None, help="training dataset root, for provenance check")
    ap.add_argument("--domain", action="append", default=None,
                    help="reference-source rule name:prefix1,prefix2 (repeatable)")
    ap.add_argument("--min-instances", type=int, default=200,
                    help="warn if a source has fewer labeled heads than this")
    ap.add_argument("--no-split", action="store_true",
                    help="treat the whole test set as one bucket (no per-source diagnostic); "
                         "use this when clips are mixed and cannot be cleanly assigned")
    args = ap.parse_args()

    domains = parse_domains(args.domain) or [("firstperson", ["firstperson", "helmet"]),
                                             ("topview", ["topview", "catwalk"])]
    img_dir = os.path.join(args.test_root, "images", args.split)
    lbl_dir = os.path.join(args.test_root, "labels", args.split)
    if not os.path.isdir(img_dir):
        sys.exit("[ERROR] {} not found (check --test-root/--split)".format(img_dir))

    images = list_images(img_dir)
    if not images:
        sys.exit("[ERROR] no images under {}".format(img_dir))

    issues = []  # (severity, message); severity in {"FAIL","WARN"}

    # per-clip / per-source tallies
    src_frames = defaultdict(int)
    src_labeled = defaultdict(int)
    src_instances = defaultdict(int)
    src_clips = defaultdict(set)
    other_clips = set()
    missing_lbl = []
    empty_lbl = []

    for img in images:
        clip = clip_of(img)
        src = "all" if args.no_split else domain_of(clip, domains)
        src_frames[src] += 1
        src_clips[src].add(clip)
        if src == "(other)":
            other_clips.add(clip)
        stem = os.path.splitext(os.path.basename(img))[0]
        n = count_instances(os.path.join(lbl_dir, stem + ".txt"))
        if n is None:
            missing_lbl.append(stem)
        elif n == 0:
            empty_lbl.append(stem)
            src_labeled[src] += 1  # counts as labeled-but-empty (valid negative)
        else:
            src_labeled[src] += 1
            src_instances[src] += n

    # ---- report -----------------------------------------------------------
    print("=" * 70)
    print("TEST-SET REPORT  ({} images under {})".format(len(images), img_dir))
    print("=" * 70)
    print("{:<14} {:>6} {:>8} {:>10} {:>7}".format("source", "clips", "frames", "instances", "min?"))
    for src in list(src_frames):
        inst = src_instances[src]
        flag = "OK" if inst >= args.min_instances else "LOW"
        print("{:<14} {:>6} {:>8} {:>10} {:>7}".format(
            src, len(src_clips[src]), src_frames[src], inst, flag))
        if src == "(other)":
            issues.append(("FAIL", "clips fell into (other) — bucketing rules don't match "
                           "their names: {}".format(sorted(other_clips))))
        elif inst < args.min_instances:
            issues.append(("WARN", "source '{}' has only {} labeled heads (< {})".format(
                src, inst, args.min_instances)))
        if len(src_clips[src]) < 2 and src != "(other)":
            issues.append(("WARN", "source '{}' has only {} clip — one clip is a thin, "
                           "non-representative test".format(src, len(src_clips[src]))))
    print("-" * 70)

    # labels
    if missing_lbl:
        issues.append(("FAIL", "{} image(s) have NO label .txt (recall will be wrong). "
                       "e.g. {}".format(len(missing_lbl), missing_lbl[:5])))
    n_empty = len(empty_lbl)
    if n_empty:
        print("note: {} frame(s) have empty labels (fine ONLY if truly head-free negatives)".format(n_empty))

    # provenance
    if args.train_root:
        train_clips = clip_names_under(args.train_root)
        test_clips = set().union(*src_clips.values()) if src_clips else set()
        overlap = sorted(train_clips & test_clips)
        if overlap:
            issues.append(("FAIL", "LEAKAGE: these clips are in BOTH train and test: {}".format(overlap)))
        else:
            print("provenance: no exact clip-name overlap with training set "
                  "({} train clips checked)".format(len(train_clips)))
            print("            (cannot detect same source footage under a different name — "
                  "confirm that yourself)")
    else:
        issues.append(("WARN", "no --train-root given: provenance/leakage NOT checked"))

    # verdict
    print("=" * 70)
    fails = [m for s, m in issues if s == "FAIL"]
    warns = [m for s, m in issues if s == "WARN"]
    for m in fails:
        print("  FAIL: " + m)
    for m in warns:
        print("  warn: " + m)
    print("-" * 70)
    if fails:
        print("VERDICT: NOT READY — fix the FAIL item(s) above before trusting eval numbers.")
        sys.exit(1)
    elif warns:
        print("VERDICT: usable, but weak — address the warnings for a paper-grade test set.")
    else:
        print("VERDICT: READY — sources bucketed, all frames labeled, instances sufficient, no leakage.")


if __name__ == "__main__":
    main()
