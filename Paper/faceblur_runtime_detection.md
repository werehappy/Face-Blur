# Runtime Head Detection in FACEBLUR: The Role of the Fine-Tuned `head.pt` Model

## Overview

FACEBLUR performs privacy-preserving anonymization by locating heads and faces in
video frames and applying a censoring pass (blur) over the detected regions. The
system draws on two complementary detection sources: a dedicated face model, and a
head-region estimator. The fine-tuned single-class detector produced by the training
pipeline — shipped as `head.pt` and placed next to `FACEBLUR.exe` — is the component
that supplies head-level coverage for the hard cases that generic face detection
cannot handle: helmeted heads, motion-blurred heads, side and back views, partially
occluded or frame-cut heads, and dense crowds.

When the operator enables the *Detect whole head* option, the application inspects the
loaded weights and auto-detects the head class directly from the model's own metadata,
rather than relying on a hard-coded class index. It then runs `head.pt` at the
configured inference resolution, `HEAD_INFER_IMGSZ`, with edge strips (additional
inference passes near the frame borders) to recover heads that are cut off at the
image boundary. The final set of censored regions is the union of the head detections
and the face-model detections.

This section documents the runtime detection logic in detail, with particular
attention to a behavioral change in how the head source is combined with a secondary,
geometry-based head estimator — the so-called *person→head pass* — and the rationale
for disabling that estimator when a user-supplied `head.pt` is present.

## Two Sources of Head Coverage

Historically, FACEBLUR obtained head coverage from two mechanisms operating in
parallel:

The first is the learned head detector, `head.pt`, which predicts head bounding boxes
directly. The second is a geometric *person→head* pass, which does not detect heads
directly at all. Instead, it runs a COCO-style person detector to locate human bodies
and then estimates the likely head region from each body box using fixed geometric
heuristics (broadly, the upper portion of the torso box). This person→head estimator
is attractive as a fallback because it will fire on the head of essentially anyone with
a visible body, including cases where the head itself is blurred, turned away, or
otherwise unrecognizable to a face or head detector.

In earlier versions of the application, the two sources were always combined by union:
the censored region was `head.pt ∪ person→head ∪ face`. Under this arrangement the
learned `head.pt` could only *add* coverage — it could never remove a region that the
person→head pass had already proposed. This made the geometric pass a permanent safety
net but also tied the system's false-positive behavior to that pass.

## The Failure Mode: Geometry-Estimated Boxes on Forward-Held Objects

The union-always design carried a systematic weakness. Because the person→head pass
estimates a head region from body geometry rather than observing a head, it can place a
"head" box on whatever occupies the estimated region — even when no head is there. In
FACEBLUR's target footage (close-quarters and body-cam material), operators hold
equipment forward of the body: the person→head estimate frequently landed on
forward-held gear rather than on the actual head. The recurring, documented instance of
this was a weapon illuminator, whose bright, roughly head-sized, forward-projected
region matched the geometric estimate closely enough to trigger a spurious detection at
high confidence.

Two distinct causes contributed to this class of false positive, and the pipeline
documentation is careful to separate them:

1. **Geometric mis-estimation.** The person→head heuristic assumes the head sits in a
   predictable location relative to the torso. Forward-held gear violates that
   assumption, so the estimated box covers the object instead of the head.

2. **Train/inference scale mismatch.** Independently, running the *learned* head model
   at an inference resolution different from its training resolution causes it to
   misread objects at scales it never trained on, which was itself observed to produce
   illuminator false positives. This is addressed separately (see *Inference-Resolution
   Matching* below) and is not fixed by disabling the geometric pass.

## Current Behavior: Disabling the Person→Head Pass When `head.pt` Is Present

The application no longer unions the geometric pass unconditionally. Its behavior is
governed by a configuration constant, `PERSON_HEAD_MODE`, defined in `face_blur.py`.
The default value is `"user_off"`, which encodes the following policy:

- **When a user-supplied `head.pt` is loaded**, the person→head geometry pass is
  disabled. The censored region reduces to `head.pt ∪ face`. This removes the entire
  family of geometry-induced false positives — including the weapon-illuminator case —
  because the estimator that produced them is no longer running.

- **When no user `head.pt` is present** (that is, only the auto-downloaded default head
  model is available, or none at all), the person→head pass still runs as the robust
  fallback, preserving the historical safety-net behavior for users who have not
  trained their own model.

The consequence of this policy is a shift in responsibility. Under the old union, the
geometric pass guaranteed a floor of head coverage regardless of the learned model's
quality. Under `"user_off"`, once an operator supplies their own `head.pt`, that model
alone must carry all of the hard-case recall — the helmeted, blurred, back-of-head, and
crowd instances that the geometric fallback previously backstopped. This is why the
training pipeline places heightened emphasis on honest, per-domain evaluation: the
elimination of the geometric fallback makes the learned model's measured recall the
operative guarantee of coverage, rather than a supplement to it.

The choice to disable the geometric pass, rather than to filter its outputs, reflects a
deliberate trade. Filtering (for example, suppressing person→head boxes that overlap
bright foreground objects) would retain the fallback while attempting to remove its
worst errors, but at the cost of additional heuristics and residual failure cases.
Disabling it outright is simpler and removes the failure class entirely, on the premise
that an operator who has invested in training a domain-specific `head.pt` no longer
needs the generic geometric estimate.

## The Configuration Surface: `PERSON_HEAD_MODE`

`PERSON_HEAD_MODE` is exposed as a tunable constant so that the combination policy can
be changed without altering the surrounding logic. It admits four values:

- `"user_off"` (default): disable the person→head pass whenever a *user-supplied*
  `head.pt` is loaded; otherwise keep it. This is the behavior described above.

- `"any_off"`: disable the person→head pass whenever *any* head model loads, including
  the auto-downloaded default. This is more aggressive than the default, trusting even
  the bundled model to obviate the geometric fallback.

- `"always"`: the legacy behavior — always union the person→head pass with the other
  sources, regardless of which head model is loaded.

- `"never"`: disable the person→head pass entirely, in all configurations.

Exposing the policy as an enumerated constant rather than a boolean makes the
per-configuration semantics explicit and allows the fallback's aggressiveness to be
tuned to a deployment's tolerance for missed heads versus spurious blurs.

## Inference-Resolution Matching (`HEAD_INFER_IMGSZ`)

An orthogonal but closely related runtime parameter is `HEAD_INFER_IMGSZ`, the
resolution at which the application runs the learned head model, defaulting to 960. The
governing rule is that this inference resolution must match the resolution at which
`head.pt` was trained. A mismatch causes the model to interpret objects at scales it
never encountered during training, which manifests as scale-dependent false positives —
and was, as noted above, an independent contributor to the illuminator problem.

The practical implication for deployment is that changing training resolution and
changing inference resolution are coupled operations. Training at 960 requires no change
(`HEAD_INFER_IMGSZ = 960`). Training at 1280 improves recall on small and distant heads
but requires setting `HEAD_INFER_IMGSZ = 1280` to match, and incurs a slower head pass —
a meaningful cost on the CPU-only machines on which many users run the application.
Because the two settings are linked, a scale-dependent false positive is diagnosed and
corrected by matching the resolutions, not by retraining the model — a distinction the
pipeline documentation makes explicit, since retraining a model to fix what is in fact a
configuration mismatch would waste effort and leave the underlying problem in place.

## Deduplication and Visualization

Two secondary runtime behaviors round out the detection logic.

First, the union of sources is deduplicated at the censoring stage: a face that is
already contained within a head box is not censored twice. This prevents redundant
processing where the face and head sources overlap, which is the common case for
front-facing subjects.

Second, the application provides a *Show debug boxes* mode that renders each detection
source in a distinct color, allowing an operator to attribute every blurred region to
its origin: red boxes indicate `head.pt` firing, cyan indicates the face model, and
yellow indicates the person→head geometric region. A diagnostic consequence of the
default policy follows directly: when a user `head.pt` is loaded, no yellow boxes appear,
because the geometric pass is disabled. The absence of yellow is therefore a positive
confirmation that the geometry fallback is inactive and that coverage is being supplied
entirely by the learned model and the face detector.

## Summary

FACEBLUR's runtime head detection combines a learned single-class detector (`head.pt`)
with a face model and, conditionally, a geometry-based person→head estimator. The
current default policy (`PERSON_HEAD_MODE = "user_off"`) disables the geometric estimator
whenever the operator supplies their own `head.pt`, reducing the detection set to
`head.pt ∪ face`. This change eliminates a class of false positives in which the
geometric estimate landed on forward-held equipment, at the cost of transferring full
responsibility for hard-case recall onto the learned model. The inference resolution
`HEAD_INFER_IMGSZ` must match the model's training resolution to avoid a separate,
scale-dependent family of false positives. Together, the mode-conditional union, the
resolution-matching requirement, and the debug visualization define a detection pipeline
whose correctness depends directly on the quality of the operator's fine-tuned model —
which is the central motivation for the rigorous, per-domain evaluation prescribed by
the training pipeline.
