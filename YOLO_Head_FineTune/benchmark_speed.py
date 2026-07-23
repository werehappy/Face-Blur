r"""
benchmark_speed.py -- CPU vs CUDA inference-speed benchmark for the head models
and the deployed runtime configurations (base / +prior / +edge / +edge+prior).

WHAT IT MEASURES
    End-to-end per-frame wall-clock time (preprocess + inference + postprocess/
    NMS) at batch 1 -- i.e. exactly what the app pays per frame -- for each
    (device x model x config). Reports mean, median, p90 ms/frame and the median
    FPS. Median is the honest headline: the first frames carry one-time costs
    (model load, CUDA context, cuDNN autotune) that are excluded via --warmup.

CONFIGS (what each frame runs)
    base       : 1 head-model forward
    prior      : 1 head forward + 1 person forward (the person->head rescue aid)
    edge       : 5 head forwards (full frame + 4 edge strips)
    edge_prior : 5 head forwards + 1 person forward
    (The rescoring arithmetic / dedup is negligible vs. inference and is not
    separately timed; the passes above are >99% of per-frame cost.)

HONESTY GUARDS
    * --warmup frames are run and discarded before timing (CUDA especially).
    * torch.cuda.synchronize() is called before stopping the clock on CUDA, so
      we measure GPU compute, not async launch time (a common way to report
      falsely-fast GPU numbers).
    * The exact CPU and GPU names are printed and written to the CSV; a speed
      number is meaningless without the hardware, so quote it in the paper.

USAGE
    python benchmark_speed.py --images head_test/images/val ^
        --model head_n:head_n.pt --model head_s:head_s.pt --model head_m:head_m.pt ^
        --person yolo11m.pt --devices cpu,cuda ^
        --configs base,prior,edge,edge_prior ^
        --imgsz 960 --conf 0.25 --warmup 10 --frames 100 --out speed.csv
"""

import argparse
import os
import statistics
import sys
import time

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_model(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError("expected name:weights, got '{}'".format(s))
    name, path = s.split(":", 1)
    return name, path


def parse_args():
    ap = argparse.ArgumentParser(description="CPU vs CUDA speed benchmark.")
    ap.add_argument("--model", action="append", dest="models", type=parse_model,
                    default=[], help="name:weights head model (repeatable)")
    ap.add_argument("--person", default="yolo11m.pt",
                    help="person model used by the +prior/+edge_prior configs")
    ap.add_argument("--images", required=True, help="folder of frames to benchmark on")
    ap.add_argument("--devices", default="cpu,cuda",
                    help="comma list: cpu, cuda (cuda skipped with a note if unavailable)")
    ap.add_argument("--configs", default="base,prior,edge,edge_prior",
                    help="comma list of: base, prior, edge, edge_prior")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--edge-frac", type=float, default=0.20)
    ap.add_argument("--warmup", type=int, default=10, help="frames discarded before timing")
    ap.add_argument("--frames", type=int, default=100, help="frames to time (subsampled evenly)")
    ap.add_argument("--check-accuracy", action="store_true",
                    help="also report mean detections/frame per device to confirm "
                         "CPU and CUDA agree (sanity check, not a headline)")
    ap.add_argument("--out", default="speed.csv")
    return ap.parse_args()


def gather(folder, n):
    imgs = []
    for root, _d, files in os.walk(folder):
        for nm in files:
            if os.path.splitext(nm)[1].lower() in IMG_EXTS:
                imgs.append(os.path.join(root, nm))
    imgs.sort()
    if not imgs:
        sys.exit("[ERROR] no images under {}".format(folder))
    if n and len(imgs) > n:
        step = len(imgs) / n
        imgs = [imgs[int(i * step)] for i in range(n)]
    return imgs


def hw_info():
    cpu = "unknown-CPU"
    try:
        import platform
        cpu = platform.processor() or platform.machine() or "unknown-CPU"
        if sys.platform.startswith("linux"):
            for ln in open("/proc/cpuinfo"):
                if ln.lower().startswith("model name"):
                    cpu = ln.split(":", 1)[1].strip(); break
    except Exception:
        pass
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return cpu, gpu


def run_config(head, person, frame, config, imgsz, conf, device, edge_frac):
    """Issue the inference passes one frame's `config` requires. Returns the
    number of head detections (for the optional accuracy check)."""
    import cv2
    ndet = 0
    # head pass(es)
    passes = [frame]
    if config in ("edge", "edge_prior"):
        h, w = frame.shape[:2]
        e = edge_frac
        for sx1, sy1, sx2, sy2 in [(0, 0, int(w*e), h), (int(w*(1-e)), 0, w, h),
                                   (0, 0, w, int(h*e)), (0, int(h*(1-e)), w, h)]:
            strip = frame[sy1:sy2, sx1:sx2]
            if strip.size:
                passes.append(strip)
    for im in passes:
        r = head.predict(im, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if r.boxes is not None:
            ndet += len(r.boxes)
    # person pass
    if config in ("prior", "edge_prior") and person is not None:
        person.predict(frame, imgsz=imgsz, conf=conf, device=device,
                       classes=[0], verbose=False)
    return ndet


def time_frames(head, person, frames, config, imgsz, conf, device, edge_frac,
                warmup, cuda):
    import torch
    # warmup
    for i in range(min(warmup, len(frames))):
        run_config(head, person, frames[i % len(frames)], config, imgsz, conf, device, edge_frac)
    if cuda:
        torch.cuda.synchronize()
    times, dets = [], []
    for fr in frames:
        t0 = time.perf_counter()
        nd = run_config(head, person, fr, config, imgsz, conf, device, edge_frac)
        if cuda:
            torch.cuda.synchronize()   # <-- essential: GPU calls are async
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
        dets.append(nd)
    return times, dets


def summarize(times):
    times_sorted = sorted(times)
    mean = statistics.mean(times)
    median = statistics.median(times)
    p90 = times_sorted[min(len(times_sorted) - 1, int(0.9 * len(times_sorted)))]
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    fps = 1000.0 / median if median > 0 else float("nan")
    return mean, median, p90, std, fps


def main():
    args = parse_args()
    if not args.models:
        sys.exit("[ERROR] pass at least one --model name:weights")

    import cv2
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        torch, cuda_ok = None, False

    devices = []
    for d in args.configs and args.devices.split(","):
        d = d.strip()
        if d == "cuda" and not cuda_ok:
            print("[note] CUDA not available; skipping the cuda device.")
            continue
        devices.append(d)
    if not devices:
        sys.exit("[ERROR] no runnable device (cpu always works; cuda needs a GPU).")

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cpu_name, gpu_name = hw_info()
    print("CPU: {}".format(cpu_name))
    print("GPU: {}".format(gpu_name or "none detected"))
    print("imgsz={} conf={} warmup={} frames={}".format(
        args.imgsz, args.conf, args.warmup, args.frames))

    frames_paths = gather(args.images, args.frames)
    frames = [cv2.imread(p) for p in frames_paths]
    frames = [f for f in frames if f is not None]
    print("Loaded {} frames from {}\n".format(len(frames), args.images))

    from ultralytics import YOLO

    rows = []
    need_person = any(c in ("prior", "edge_prior") for c in configs)

    for device in devices:
        cuda = (device == "cuda")
        dev_label = "CUDA ({})".format(gpu_name) if cuda else "CPU"
        print("=" * 70)
        print("DEVICE: {}".format(dev_label))
        print("=" * 70)
        person = None
        if need_person:
            person = YOLO(args.person)
        for name, weights in args.models:
            if not os.path.exists(weights):
                print("[warn] skipping {}: weights not found ({})".format(name, weights))
                continue
            head = YOLO(weights)
            print("\n {} ({})".format(name, weights))
            print(" {:<11} {:>9} {:>9} {:>9} {:>8} {:>8}".format(
                "config", "mean ms", "med ms", "p90 ms", "std", "FPS"))
            for config in configs:
                if config in ("prior", "edge_prior") and person is None:
                    continue
                times, dets = time_frames(head, person, frames, config, args.imgsz,
                                          args.conf, device, args.edge_frac,
                                          args.warmup, cuda)
                mean, median, p90, std, fps = summarize(times)
                print(" {:<11} {:>9.1f} {:>9.1f} {:>9.1f} {:>8.1f} {:>8.1f}".format(
                    config, mean, median, p90, std, fps))
                row = dict(device=device, gpu=gpu_name or "", cpu=cpu_name,
                           model=name, config=config, imgsz=args.imgsz,
                           mean_ms=round(mean, 2), median_ms=round(median, 2),
                           p90_ms=round(p90, 2), std_ms=round(std, 2),
                           fps=round(fps, 2), frames=len(times))
                if args.check_accuracy:
                    row["mean_dets"] = round(statistics.mean(dets), 3)
                rows.append(row)
            del head
            if cuda:
                torch.cuda.empty_cache()
        del person
        if cuda:
            torch.cuda.empty_cache()

    # optional accuracy-consistency note
    if args.check_accuracy:
        print("\n[accuracy check] mean detections/frame should match across "
              "devices for the same model+config (identical weights/math):")
        by = {}
        for r in rows:
            by.setdefault((r["model"], r["config"]), {})[r["device"]] = r.get("mean_dets")
        for (m, c), d in by.items():
            if "cpu" in d and "cuda" in d:
                same = abs((d["cpu"] or 0) - (d["cuda"] or 0)) < 1e-6
                print("  {:<8} {:<11} cpu={} cuda={} {}".format(
                    m, c, d["cpu"], d["cuda"], "OK" if same else "DIFFER (float noise?)"))

    import csv
    fields = ["device", "gpu", "cpu", "model", "config", "imgsz", "mean_ms",
              "median_ms", "p90_ms", "std_ms", "fps", "frames"]
    if args.check_accuracy:
        fields.append("mean_dets")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("\nWrote {}".format(args.out))
    # speedup summary
    print("\nCUDA speedup over CPU (median ms):")
    idx = {(r["device"], r["model"], r["config"]): r["median_ms"] for r in rows}
    seen = False
    for (dev, m, c), v in idx.items():
        if dev == "cpu" and ("cuda", m, c) in idx:
            g = idx[("cuda", m, c)]
            if g > 0:
                print("  {:<8} {:<11} {:.1f}x  ({:.0f} ms CPU -> {:.0f} ms CUDA)".format(
                    m, c, v / g, v, g)); seen = True
    if not seen:
        print("  (need both cpu and cuda runs to compute speedup)")


if __name__ == "__main__":
    main()
