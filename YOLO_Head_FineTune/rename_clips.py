r"""
rename_clips.py -- rename whole clips to domain-encoded names.

Renames every frame of a clip from its current name to <domain>_<NN>_<frameidx>,
moving each label .txt with its image, so the domain is baked into the filename
and every tool (check_split, eval_domains, ...) buckets correctly forever.

Clips are assigned to a domain by PREFIX rule (matched with startswith), then
numbered sequentially per domain (natural-sorted). Split membership is NOT
changed -- a clip stays in whatever split folder it's already in.

DRY-RUN BY DEFAULT. Prints the full old->new mapping; changes nothing until --apply.

USAGE:
    # your case:
    python rename_clips.py --root head_dataset ^
        --rule new_clip=catwalk --rule cat_clip=catwalk --rule clip=helmet
    # then, to actually do it:
    python rename_clips.py --root head_dataset ^
        --rule new_clip=catwalk --rule cat_clip=catwalk --rule clip=helmet --apply

Note: rules are applied longest-prefix-first, so 'new_clip'/'cat_clip' win over
'clip'. After renaming, evaluate with:
    python eval_domains.py --data head_dataset/data.yaml --model head_s.pt ^
        --domain helmet:helmet --domain catwalk:catwalk
"""

import argparse
import os
import re
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_(\d+)$")   # (clip, frameindex)
SPLITS = ("train", "val", "test")


def split_clip_idx(stem):
    m = CLIP_RE.match(stem)
    if m:
        return m.group(1), m.group(2)
    return stem, None


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", s)]


def images_in(root, split):
    d = os.path.join(root, "images", split)
    out = []
    if os.path.isdir(d):
        for nm in os.listdir(d):
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                out.append(nm)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rename clips to domain-encoded names.")
    ap.add_argument("--root", default="head_dataset")
    ap.add_argument("--rule", action="append", default=[],
                    help="prefix=domain (repeatable), e.g. --rule clip=helmet")
    ap.add_argument("--map", action="append", default=[], dest="maps",
                    help="EXACT clip=domain override (repeatable), beats --rule, "
                         "e.g. --map new_clip4=helmet")
    ap.add_argument("--apply", action="store_true", help="perform the rename (default: dry-run)")
    args = ap.parse_args()

    if not args.rule and not args.maps:
        print("[!] no --rule/--map given. Example:")
        print("    --rule new_clip=catwalk --rule cat_clip=catwalk --rule clip=helmet "
              "--map new_clip4=helmet")
        return

    # exact per-clip overrides (highest priority)
    exact = {}
    for m in args.maps:
        if "=" not in m:
            print("[warn] bad map '{}'".format(m)); continue
        c, dom = m.split("=", 1)
        exact[c.strip()] = dom.strip()

    # longest prefix first so specific rules win
    rules = []
    for r in args.rule:
        if "=" not in r:
            print("[warn] bad rule '{}'".format(r)); continue
        pfx, dom = r.split("=", 1)
        rules.append((pfx.strip(), dom.strip()))
    rules.sort(key=lambda x: -len(x[0]))

    def domain_of(clip):
        if clip in exact:
            return exact[clip]
        for pfx, dom in rules:
            if clip.startswith(pfx):
                return dom
        return None

    present = [s for s in SPLITS if os.path.isdir(os.path.join(args.root, "images", s))]
    if not present:
        print("[!] no images/train|val|test under {}".format(args.root)); return

    # gather clips (across all splits) and which split each lives in
    clip_split = {}      # clip -> split
    clip_frames = defaultdict(int)
    for s in present:
        for nm in images_in(args.root, s):
            clip, idx = split_clip_idx(os.path.splitext(nm)[0])
            clip_split[clip] = s
            clip_frames[clip] += 1

    # assign new names: <domain>_<NN>, numbered per domain by natural sort
    domain_clips = defaultdict(list)
    unmatched = []
    for clip in clip_split:
        dom = domain_of(clip)
        if dom is None:
            unmatched.append(clip)
        else:
            domain_clips[dom].append(clip)

    newname = {}   # old clip -> new clip
    for dom in sorted(domain_clips):
        for i, clip in enumerate(sorted(domain_clips[dom], key=natural_key), 1):
            newname[clip] = "{}_{:02d}".format(dom, i)

    # ---- report -------------------------------------------------------------
    print("=" * 64)
    print("CLIP RENAME PLAN ({})".format("APPLYING" if args.apply else "DRY-RUN"))
    print("-" * 64)
    for clip in sorted(newname, key=natural_key):
        print("  {:<16} -> {:<14} ({} frames, {})".format(
            clip, newname[clip], clip_frames[clip], clip_split[clip]))
    if unmatched:
        print("\n  [!] NOT matched by any rule (left unchanged):")
        for c in sorted(unmatched, key=natural_key):
            print("      {:<16} ({} frames, {})".format(c, clip_frames[c], clip_split[c]))
        print("      Add a --rule for these, or they keep their names.")
    print("-" * 64)

    if not args.apply:
        print("DRY-RUN only. Re-run with --apply to rename.")
        return

    # ---- apply: rename image + label together, in place, per split ----------
    renamed_imgs = renamed_lbls = 0
    for s in present:
        img_dir = os.path.join(args.root, "images", s)
        lbl_dir = os.path.join(args.root, "labels", s)
        for nm in list(images_in(args.root, s)):
            stem, ext = os.path.splitext(nm)
            clip, idx = split_clip_idx(stem)
            if clip not in newname or idx is None:
                continue
            new_stem = "{}_{}".format(newname[clip], idx)
            # image
            os.rename(os.path.join(img_dir, nm),
                      os.path.join(img_dir, new_stem + ext))
            renamed_imgs += 1
            # label
            old_lbl = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(old_lbl):
                os.rename(old_lbl, os.path.join(lbl_dir, new_stem + ".txt"))
                renamed_lbls += 1
    print("[done] renamed {} image(s) and {} label(s).".format(renamed_imgs, renamed_lbls))
    print("Re-run check_split.py / audit_dataset.py to confirm, then eval with")
    print("--domain helmet:helmet --domain catwalk:catwalk")


if __name__ == "__main__":
    main()
