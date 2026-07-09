#!/usr/bin/env python3
r"""
eval_compare.py -- compare SEVERAL head models per domain and emit paper-ready
output (CSV + LaTeX table + Markdown table + a comparison figure).

This is the reporting layer that sits on top of eval_domains.py's conventions:
same data.yaml + split layout, same clip-prefix domain bucketing, same imgsz
960 default. Where eval_domains.py evaluates ONE model, this runs every model
you pass over every domain and lines them up for Section 6 of the paper.

For each model x domain it reports, at the app's operating point (conf 0.25 by
default) and matched imgsz:
    Recall (headline -- a miss is a privacy leak), Precision, F1
and, threshold-independent:
    mAP@0.5, mAP@0.5:0.95
plus labeled-head instance counts per domain (sample size + the imbalance signal).

Two val passes are used per model x domain: one at low confidence for correct
mAP, one at the operating confidence for Recall/Precision. Skip mAP with
--no-map for speed or for non-head-class baselines.

USAGE
-----
    conda activate yolotrain
    python eval_compare.py --data head_dataset/data.yaml --split test ^
        --model head_n:head_n.pt --model head_s:head_s.pt --model head_m:head_m.pt

    # custom domain rules (name:prefix1,prefix2 ; clip-name startswith), and a title
    python eval_compare.py --data head_dataset/data.yaml --split test ^
        --model head_s:head_s.pt ^
        --domain catwalk:new_clip,cat_ --domain helmet:clip ^
        --title "Base vs fine-tuned, held-aside clips"

OUTPUT (into --out-dir, default paper_eval/)
    eval_comparison.csv    every model x domain row
    eval_comparison.md     readable comparison table
    eval_comparison.tex    LaTeX (booktabs) recall matrix, ready to \input
    recall_by_domain.png   grouped bar chart: per-domain recall by model

BASELINE NOTE
-------------
A stock yolo11 outputs `person`, not heads, so ultralytics val cannot score it
against head ground truth. To include the person->head geometric baseline (the
pre-fine-tuning method), score it with the operating-point matcher in
evaluate_heads.py and add its row to the CSV by hand, or ask to have that path
merged in here.
"""

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLIP_RE = re.compile(r"^(.*?)_\d+$")


# --------------------------------------------------------------------------- #
# data bucketing (mirrors eval_domains.py so results are consistent)
# --------------------------------------------------------------------------- #
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


def count_instances(img_path, images_root, labels_root):
    rel = os.path.relpath(img_path, images_root)
    stem = os.path.splitext(rel)[0]
    lbl = os.path.join(labels_root, stem + ".txt")
    if not os.path.exists(lbl):
        return 0
    try:
        with open(lbl) as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def parse_model(spec):
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("model must be name:weights, got '{}'".format(spec))
    return parts[0], parts[1]


# --------------------------------------------------------------------------- #
# metric collection (needs ultralytics)
# --------------------------------------------------------------------------- #
def evaluate(model, tmp_yaml, imgsz, device, conf):
    """One ultralytics val pass. Returns (recall, precision, map50, map5095)."""
    kw = dict(data=tmp_yaml, imgsz=imgsz, device=device, split="val",
              verbose=False, plots=False)
    if conf is not None:
        kw["conf"] = conf
    m = model.val(**kw)
    b = m.box
    return (float(getattr(b, "mr", float("nan"))),
            float(getattr(b, "mp", float("nan"))),
            float(getattr(b, "map50", float("nan"))),
            float(getattr(b, "map", float("nan"))))


def collect_rows(args):
    import yaml
    from ultralytics import YOLO

    with open(args.data) as f:
        dcfg = yaml.safe_load(f) or {}
    base = dcfg.get("path") or os.path.dirname(os.path.abspath(args.data))
    names = dcfg.get("names", {0: "head"})

    images_dir = args.images_dir or os.path.join(base, "images", args.split)
    labels_dir = os.path.join(base, "labels", args.split)
    if not os.path.isdir(images_dir):
        sys.exit("[ERROR] split images dir not found: {} (check --split/--images-dir)".format(images_dir))

    domains = parse_domains(args.domain) or [("catwalk", ["new_clip", "cat_"]), ("helmet", ["clip"])]

    imgs = []
    for root, _d, fnames in os.walk(images_dir):
        for nm in fnames:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(root, nm))
    if not imgs:
        sys.exit("[ERROR] no images in {}".format(images_dir))

    buckets = defaultdict(list)
    frames = defaultdict(int)
    heads = defaultdict(int)
    for p in imgs:
        dom = domain_of(clip_of(p), domains)
        buckets[dom].append(p)
        frames[dom] += 1
        heads[dom] += count_instances(p, images_dir, labels_dir)

    ordered = [d for d, _ in domains if d in buckets]
    ordered += [d for d in buckets if d not in ordered]  # include (other)

    print("=" * 70)
    print("Split '{}': {} images across {} domain(s). imgsz={} conf={}".format(
        args.split, len(imgs), len(ordered), args.imgsz, args.conf))
    for d in ordered:
        print("  {:<12} {:>5} frames  {:>6} labeled heads".format(d, frames[d], heads[d]))
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="evalcmp_")
    rows = []
    for name, weights in args.models:
        if not os.path.exists(weights):
            print("[warn] skipping '{}': weights not found ({})".format(name, weights))
            continue
        print("\n=== model: {} ({}) ===".format(name, weights))
        model = YOLO(weights)

        eval_sets = [(d, buckets[d]) for d in ordered]
        eval_sets.append(("ALL", imgs))
        for dom, paths in eval_sets:
            list_txt = os.path.join(tmpdir, "imgs_{}_{}.txt".format(name, dom.strip("()")))
            with open(list_txt, "w") as f:
                for p in paths:
                    f.write(os.path.abspath(p) + "\n")
            tmp_yaml = os.path.join(tmpdir, "data_{}_{}.yaml".format(name, dom.strip("()")))
            with open(tmp_yaml, "w") as f:
                yaml.safe_dump({"path": base, "train": list_txt, "val": list_txt, "names": names}, f)

            try:
                recall, precision, _, _ = evaluate(model, tmp_yaml, args.imgsz, args.device, args.conf)
                map50 = map5095 = float("nan")
                if not args.no_map:
                    _, _, map50, map5095 = evaluate(model, tmp_yaml, args.imgsz, args.device, None)
            except Exception as e:  # noqa
                print("  [warn] eval failed for {}/{}: {}".format(name, dom, e))
                recall = precision = map50 = map5095 = float("nan")

            f1 = (2 * precision * recall / (precision + recall)
                  if (precision == precision and recall == recall and (precision + recall)) else float("nan"))
            n_heads = heads[dom] if dom != "ALL" else sum(heads.values())
            rows.append(dict(model=name, domain=dom, images=len(paths), heads=n_heads,
                             recall=recall, precision=precision, f1=f1,
                             map50=map50, map50_95=map5095))
            print("  {:<12} R={} P={} F1={} mAP50={}".format(
                dom, _fmt(recall), _fmt(precision), _fmt(f1), _fmt(map50)))
    return rows, ordered


# --------------------------------------------------------------------------- #
# reporting (pure functions -- no ultralytics needed)
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    return "{:.{}f}".format(v, nd) if isinstance(v, (int, float)) and v == v else "-"


def write_csv(rows, path):
    fields = ["model", "domain", "images", "heads", "recall", "precision", "f1", "map50", "map50_95"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _models_in_order(rows):
    seen = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def _domains_in_order(rows, ordered_domains):
    doms = [d for d in ordered_domains]
    if any(r["domain"] == "ALL" for r in rows):
        doms.append("ALL")
    return doms


def write_markdown(rows, ordered_domains, path, metric_note):
    models = _models_in_order(rows)
    doms = _domains_in_order(rows, ordered_domains)
    lut = {(r["model"], r["domain"]): r for r in rows}
    lines = ["# Per-domain model comparison", "", metric_note, ""]
    # Recall matrix (headline)
    lines.append("## Recall by domain (headline)")
    lines.append("")
    lines.append("| Domain | Heads | " + " | ".join(models) + " |")
    lines.append("|" + "---|" * (len(models) + 2))
    for d in doms:
        heads = next((lut[(m, d)]["heads"] for m in models if (m, d) in lut), "-")
        cells = [_fmt(lut[(m, d)]["recall"]) if (m, d) in lut else "-" for m in models]
        tag = " _(reference only)_" if d == "ALL" else ""
        lines.append("| {}{} | {} | {} |".format(d, tag, heads, " | ".join(cells)))
    # Full metrics
    lines += ["", "## Full metrics", "",
              "| Model | Domain | Images | Heads | Recall | Precision | F1 | mAP50 | mAP50-95 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["model"], r["domain"], r["images"], r["heads"],
            _fmt(r["recall"]), _fmt(r["precision"]), _fmt(r["f1"]),
            _fmt(r["map50"]), _fmt(r["map50_95"])))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex(rows, ordered_domains, path, caption, conf):
    models = _models_in_order(rows)
    doms = [d for d in _domains_in_order(rows, ordered_domains) if d != "ALL"]
    lut = {(r["model"], r["domain"]): r for r in rows}
    col = "l r " + " ".join(["r"] * len(models))
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{" + _tex_escape(caption) + r"}",
           r"\label{tab:head_recall}",
           r"\begin{tabular}{" + col + r"}", r"\toprule",
           "Domain & Heads & " + " & ".join(_tex_escape(m) for m in models) + r" \\",
           r"\midrule"]
    for d in doms:
        heads = next((lut[(m, d)]["heads"] for m in models if (m, d) in lut), "-")
        cells = [_fmt(lut[(m, d)]["recall"], 3) if (m, d) in lut else "-" for m in models]
        out.append("{} & {} & {} \\\\".format(_tex_escape(d), heads, " & ".join(cells)))
    out += [r"\bottomrule", r"\end{tabular}",
            r"\vspace{2pt}",
            r"\\{\footnotesize Recall at operating confidence " + str(conf) +
            r", imgsz 960. Per domain; pooled omitted.}",
            r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def make_figure(rows, ordered_domains, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa
        print("[warn] matplotlib not available, skipping figure: {}".format(e))
        return
    models = _models_in_order(rows)
    doms = [d for d in ordered_domains if d != "ALL"]
    lut = {(r["model"], r["domain"]): r for r in rows}
    colors = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5", "#854F0B", "#3B6D11"]

    x = np.arange(len(doms))
    n = max(len(models), 1)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(doms), 4.2), dpi=150)
    for i, m in enumerate(models):
        vals = [lut[(m, d)]["recall"] if (m, d) in lut and lut[(m, d)]["recall"] == lut[(m, d)]["recall"] else 0
                for d in doms]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=m,
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "{:.2f}".format(v),
                    ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    ax.set_xticks(x)
    ax.set_xticklabels(doms)
    ax.set_ylabel("Recall (operating point)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12, color="#1F3A5F")
    ax.legend(frameon=False, fontsize=9, ncol=min(len(models), 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D3D1C7", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Compare head models per domain for the paper.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", action="append", dest="models", type=parse_model, default=[],
                    metavar="name:weights", help="repeatable; e.g. --model head_s:head_s.pt")
    ap.add_argument("--split", default="test", help="split to evaluate (default test)")
    ap.add_argument("--imgsz", type=int, default=960, help="inference size (match the app, default 960)")
    ap.add_argument("--conf", type=float, default=0.25, help="operating-point confidence (default 0.25)")
    ap.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    ap.add_argument("--domain", action="append", default=None,
                    help="domain rule name:prefix1,prefix2 (repeatable)")
    ap.add_argument("--images-dir", default=None, help="override split images dir")
    ap.add_argument("--no-map", action="store_true", help="skip the mAP pass (faster)")
    ap.add_argument("--out-dir", default="paper_eval", help="output directory")
    ap.add_argument("--title", default="Head-detection recall by domain",
                    help="figure title / table caption")
    args = ap.parse_args()

    for req, msg in [(args.data, "--data"), (args.models, "at least one --model")]:
        if not req:
            ap.error("missing {}".format(msg))
    if not os.path.exists(args.data):
        sys.exit("[ERROR] data yaml not found: {}".format(args.data))

    os.makedirs(args.out_dir, exist_ok=True)
    rows, ordered = collect_rows(args)
    if not rows:
        sys.exit("[ERROR] no results collected.")

    note = ("Recall/Precision/F1 at operating confidence {}, imgsz {}. "
            "mAP is threshold-independent. Recall is the headline metric "
            "(a missed head is a privacy leak). Pooled 'ALL' is reference only.").format(args.conf, args.imgsz)

    csv_p = os.path.join(args.out_dir, "eval_comparison.csv")
    md_p = os.path.join(args.out_dir, "eval_comparison.md")
    tex_p = os.path.join(args.out_dir, "eval_comparison.tex")
    fig_p = os.path.join(args.out_dir, "recall_by_domain.png")

    write_csv(rows, csv_p)
    write_markdown(rows, ordered, md_p, note)
    write_latex(rows, ordered, tex_p, args.title, args.conf)
    make_figure(rows, ordered, fig_p, args.title)

    print("\nWrote:")
    for p in (csv_p, md_p, tex_p, fig_p):
        print("  " + p)
    print("\nHeadline = per-domain recall. Do not report the pooled 'ALL' row as the result.")


if __name__ == "__main__":
    main()
