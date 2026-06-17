# TRAINING_GUIDE — Fine-tune your own `head.pt` for FACEBLUR

This is the reliable fix for footage that public head models miss: helmeted,
motion-blurred, side/back, and partially cut-off heads (CQB / body-cam). You
label a few hundred frames from your **own** clips, fine-tune YOLO on them, and
drop the result in next to FACEBLUR as `head.pt`. The app loads it with no code
change and unions it with the person→head method, so it can only add coverage.

The hard part is **labeling**, not training. Training is a one-line command.
Budget your effort accordingly.

---

## 0. What you'll end up with

```
best.pt  ->  rename to  head.pt  ->  put next to FACEBLUR.exe
```

A single-class (`head`) detector trained on frames that look like your real
footage. Plan on **2–3 rounds**: train, see where it fails, label more of those
failure cases, retrain.

---

## 1. How much data

| Goal | Labeled frames | Realistic result |
|---|---|---|
| Minimum proof-of-life | ~150–300 | Catches obvious heads; misses hard ones |
| Solid first model | ~500–800 | Good on footage like what you labeled |
| Strong | 1,500+ | Robust across lighting/clips |

Frames matter less than **head instances** and **diversity**. 500 frames with
2–4 heads each (~1,500 boxes) covering different clips, lighting, distances, and
head poses beats 2,000 near-identical frames from one hallway.

Pull frames from **several different clips**. Keep at least one whole clip aside
that you never label — that's your honest test set (see Step 7).

---

## 2. Set up a training environment

Training wants a **GPU build of torch**. Your `faceblur` env may have CPU torch
(the app downloads its own at runtime), and CPU training is painfully slow. Use a
separate env so you don't disturb the app's env:

```bash
conda create -n yolotrain python=3.10 -y
conda activate yolotrain
pip install ultralytics "numpy<2"
# GPU torch (CUDA 12.1 build; use cu118 if your driver is older):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If that prints `CUDA: True`, you're set. If `False`, training still works on CPU
but expect it to be many times slower; reduce `imgsz` and `epochs` to compensate.

---

## 3. Pull frames to label

Use the included `sample_frames.py` (saves diverse, well-spaced, de-duplicated
frames so you don't label 50 copies of the same shot):

```bash
conda activate faceblur     # this script only needs opencv
python sample_frames.py clip1.mp4 clip2.mp4 clip3.mp4 --out dataset_raw --per-clip 80
```

This drops ~240 varied jpgs into `dataset_raw/`. Deliberately include **hard**
frames — motion blur, backs of heads, helmets, heads cut off by the frame edge.
Those are the cases public models fail on, so they're the most valuable to label.

---

## 4. Label the frames (one class: `head`)

Pick any tool that exports **YOLO format**:

- **Label Studio** (`pip install label-studio`) — local, free, no upload.
- **CVAT** — local via Docker, strong for video/boxes.
- **Roboflow** — web, easiest UI, free tier; can also do the train/val split and
  augmentation for you. Fine if you're OK uploading footage to a cloud service.

**Define exactly one class: `head`** (class id `0`).

Labeling rules that make or break the model:
- Box the **whole head**: hair/helmet top down to the chin/jaw, ear to ear.
- Label **every** head you can identify — front, **side, back**, helmeted,
  blurred, and **partial/cut-off** heads (box the visible part, right up to the
  frame edge).
- If you truly can't tell a blob is a head, skip it — don't guess. Consistency
  matters more than catching every last pixel.
- Be consistent about tightness. Slightly generous is fine (the app pads anyway);
  wildly loose boxes teach the model to fire on walls.

### YOLO label format (what the tool exports)

One `.txt` per image, same basename. Each line is one box:

```
0 x_center y_center width height
```

All four numbers are **normalized 0–1** (pixel value ÷ image width or height),
and the leading `0` is the `head` class id. An image with no heads gets an empty
`.txt` (or no file) — empty frames are useful negatives, keep some.

---

## 5. Arrange the dataset

YOLO expects images and labels in parallel folders, split into train/val:

```
head_dataset/
├── images/
│   ├── train/   img0001.jpg, img0002.jpg, ...
│   └── val/     img0500.jpg, ...
├── labels/
│   ├── train/   img0001.txt, img0002.txt, ...
│   └── val/     img0500.txt, ...
└── data.yaml
```

Use roughly an **80/20 train/val split**. Critical: put frames from your
held-aside clip(s) only in `val` (or in neither) — if near-identical frames from
the same moment land in both train and val, your metrics will look great and the
real footage will still fail. Split **by clip**, not by random frame.

`data.yaml`:

```yaml
path: C:/path/to/head_dataset      # absolute path to the dataset root
train: images/train
val: images/val
names:
  0: head
```

(A ready-to-edit `data.yaml` ships next to this guide.)

---

## 6. Train

From the training env, pick a small base model and fine-tune. `yolo11n.pt` (nano)
is fast and matches the app's stack; step up to `yolo11s.pt` if recall is short.

**CLI (simplest):**

```bash
conda activate yolotrain
yolo detect train model=yolo11n.pt data=head_dataset/data.yaml ^
    epochs=100 imgsz=960 batch=8 patience=30 ^
    name=head_v1
```

**Python (same thing, if you prefer a script):**

```python
from ultralytics import YOLO
m = YOLO("yolo11n.pt")          # start from COCO-pretrained nano
m.train(data="head_dataset/data.yaml",
        epochs=100, imgsz=960, batch=8, patience=30, name="head_v1")
```

Settings that matter for your footage:
- **`imgsz=960`** (or `1280` if VRAM allows). CQB heads are small and blurry;
  bigger input = more small-head recall. This is the single biggest lever. Lower
  to `640` if you hit out-of-memory; raise `batch` down with it.
- **`batch=8`** — drop to `4` on OOM, raise on a big GPU.
- **`epochs=100` + `patience=30`** — it stops early when val stops improving.
- Ultralytics' default augmentation (mosaic, flips, HSV) already helps blur/pose
  variation; you don't need to tune it for a first model.

Output lands in `runs/detect/head_v1/`. The trained weights are
`runs/detect/head_v1/weights/best.pt`.

---

## 7. Check it honestly

Look at the validation curves/metrics ultralytics prints (`runs/detect/head_v1/`):
`mAP50` and especially **recall** (you care about not *missing* heads more than
the odd false box — the app pads and you can over-cover for privacy).

Then test on the clip you **held aside** (never labeled), which is the only
honest measure. Use the diagnostic that already ships with the app:

```bash
copy runs\detect\head_v1\weights\best.pt head.pt
python test_head.py path\to\held_aside_clip.mp4
```

Open the saved `_headNN.jpg` frames and look at the backs-of-heads and blurred
heads specifically — counts alone lie.

---

## 8. Iterate (this is where the quality comes from)

Wherever the model still misses heads on the held-aside clip:
1. Pull more frames from those exact moments (`sample_frames.py` on that clip).
2. Label them (focus on the failure type — e.g. backs of heads).
3. Add them to `images/train` + `labels/train` and retrain (`name=head_v2`).

Two or three rounds of "train → find misses → label the misses → retrain" gets
far more than one giant first batch. This is normal; plan for it.

---

## 9. Ship it

Once you're happy:

```
runs\detect\head_vN\weights\best.pt   ->   rename to   head.pt
```

Put `head.pt` next to `FACEBLUR.exe` (or wire it into the installer's `[Files]`
section so every user gets it). The app auto-detects the `head` class from the
model's own names, runs it at full resolution with edge strips, and unions it
with the person→head method — so your model only **adds** coverage. Turn on
**Detect whole head** and **Show debug boxes**: red outlines are your `head.pt`
firing.

---

## Quick reference

```bash
# 1. frames to label
python sample_frames.py clip1.mp4 clip2.mp4 --out dataset_raw --per-clip 80
# 2. label in Label Studio / CVAT / Roboflow  -> export YOLO format, class 'head'
# 3. arrange into head_dataset/{images,labels}/{train,val} + data.yaml (80/20, split by clip)
# 4. train
yolo detect train model=yolo11n.pt data=head_dataset/data.yaml epochs=100 imgsz=960 batch=8 patience=30 name=head_v1
# 5. test on a held-aside clip
copy runs\detect\head_v1\weights\best.pt head.pt
python test_head.py held_aside_clip.mp4
# 6. iterate on the misses, then ship head.pt next to FACEBLUR
```

---

## Common pitfalls

- **Great metrics, bad real footage** → train/val leakage. Split by clip, not by
  random frame.
- **Model fires on walls/gear** → loose or inconsistent boxes, or too few empty
  negatives. Tighten labels, keep some head-free frames.
- **Misses small distant heads** → `imgsz` too low. Raise to 960/1280.
- **Out-of-memory during train** → lower `batch`, then `imgsz`.
- **Misses backs of heads specifically** → you didn't label enough of them.
  That's the whole reason you're training; over-represent them.
- **CPU-only and glacially slow** → install the CUDA torch build (Step 2), or
  cut `epochs`/`imgsz` for a rough first pass.
