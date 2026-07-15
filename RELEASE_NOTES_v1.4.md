# FACEBLUR v1.4 — Release Notes

**Release date:** July 15, 2026
**Made by werehappy**

---

## Headline

**Head detection is now the app's primary method, and the runtime augmentations are explicit, independent, opt-in toggles.**

Earlier versions ran the face model on every frame and treated the head model as an add-on, with the person→head pathway governed by an all-or-nothing automatic policy. v1.4 inverts this: the fine-tuned head model is the main — and by default, only — detector, and everything else is an aid you switch on when you need it.

---

## What changed

### Head model is primary
- The bundled fine-tuned head model (`head_n.pt` / `head_s.pt` / `head_m.pt`, selectable by size) is now the main detector and runs by default.
- It fires on the head itself, so it covers backs, sides, partial, helmeted, and motion-blurred heads, and works with no face — or no body — visible.
- The face model no longer runs automatically; it is available as an optional safety net (see below).

### Three independent aids, all OFF by default
1. **Person→head aid (rescue).** A COCO person detector estimates a head region from each body and uses those regions as a **soft spatial prior**. The head model is run at a low candidate floor, and a weak head detection that overlaps a region is boosted past the operating point (`conf × (1 + overlap)`); an isolated weak detection is dropped.
   - The person-derived regions are **never added as boxes on their own**. A region that lands on forward-held gear (e.g. a weapon illuminator) can no longer create a censor by itself — it can only reinforce a real but low-confidence head detection.
   - This replaces the previous all-or-nothing "union the raw geometry boxes, or disable the pass entirely" behavior. In measured evaluation, rescue-only recovered most of the recall of the old raw union while giving back most of its precision cost.
2. **Edge-strip inference.** Runs the head model over the four frame borders at full resolution to recover heads truncated at the edge of frame.
3. **Face safety net.** Runs a face model and keeps a face box (grown to head size) only where no head box already covers it, filling genuine head-model misses. A face already inside a head box is dropped rather than censored twice.

### Removed / retired
- The old **"Detect whole head"** checkbox is gone; head detection is on by default and the aids are separate toggles.
- The **`PERSON_HEAD_MODE`** automatic policy is retired in favor of the explicit person→head aid toggle. (The constant and its helper remain in the source, marked deprecated, for backward reference.)

### Persistence & defaults
- The two new toggles (person→head aid, face safety net) are saved with your other settings and restored on next launch.
- Fresh install / Reset to Defaults: head detection ON; person→head aid, edge strip, and face safety net all OFF; smooth boxes ON.

---

## Recommended settings for hard footage

For dynamic-camera, motion-blur, or cut-off-head footage (CQB/body-cam, dense crowds):

| Setting | Value |
|---|---|
| Frame skip | 1 |
| Detect scale | 1.00 |
| Confidence | ~0.25 or lower |
| Padding | 0.30+ |
| Person→head aid | ON |
| Edge strip | ON (if heads are frequently cut off at frame borders) |
| Head model size | small or medium |

---

## Notes & compatibility

- **Inference size must still match training size.** The head models are trained at 960 and the app runs them at `HEAD_INFER_IMGSZ = 960`. If you retrain at a different size, set `HEAD_INFER_IMGSZ` to match, or scale-mismatch false positives return.
- **No change to the build/distribution flow.** `build_installer.bat` and `installer.iss` are version-bumped to 1.4; torch is still excluded from the exe and downloaded on first run, and all three head models are bundled next to the exe.
- **Settings from v1.3 load cleanly.** Missing keys fall back to the new defaults, so an upgraded install simply starts with the aids off until you enable them.

---

## Upgrade guidance

If you previously relied on whole-head mode with the face union, the closest v1.4 equivalent is: head detection ON (default) + **Face safety net** ON. To additionally recover weak heads on hard footage, also enable the **Person→head aid**.

---

*FACEBLUR v1.4 — made by werehappy*
