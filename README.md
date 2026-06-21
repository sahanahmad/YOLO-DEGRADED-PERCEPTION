# YOLOv8 Perception Pipeline for Contested Environments

Applied research project: real-time object detection, tracking, and sensor
fusion for degraded-visibility, GNSS-denied conditions. Built incrementally,
one component at a time.

## Status (update as work progresses)

| Component | Status | Notes |
|---|---|---|
| 1. YOLOv8 detection pipeline | Not started | |
| 2. Multi-object tracking (Kalman/EKF) | Not started | Designed to consume calibrated R from Component 3 |
| 3. Fog/low-light augmentation + confidence calibration | **In progress** | Current focus — see `src/calibration/` |
| 4. Camera/LiDAR fusion | Not started | |
| 5. Edge inference (ONNX/TensorRT) | Not started | |

## Component 3 — current scope

**Core idea:** Per-layer activation variation under fog/low-light
augmentation, used as a signal to recalibrate YOLOv8 detection confidence.
Recalibrated confidence maps to measurement noise covariance (R) for the
downstream Kalman tracker, so the tracker automatically shifts trust toward
motion-model prediction as visual reliability degrades.

**Dataset (primary): RTTS (RESIDE)**
Real-world hazy traffic imagery, 4,322 annotated images, 5 classes
(car/bus/bicycle/motorcycle/person), VOC2007 XML format. No registration
gate. ~41k bounding boxes total, with a "difficult" flag on a subset
(11,606 boxes) usable as a coarse degradation/quality proxy in the absence
of a continuous severity dial.

**Future consideration (not currently in scope): Seeing Through Fog / DENSE**
(Bijelic et al., CVPR 2020). Multimodal (camera + LiDAR + radar) with a
fog-chamber subset offering controlled, repeatable real fog density.
Revisit when scoping Component 4 (camera/LiDAR fusion) — requires a
registration/approval process, so not suited to the current near-term
timeline. Not pursued for Component 3.

**Possible secondary validation set: Foggy Driving** (101 real-world
images) — too small to be primary, may be useful later as an additional
real-world sanity check.

**Target venue:** Mid-tier (WACV workshop / ICPR / IEEE autonomous systems
conference) — scope and claims calibrated accordingly, not a top-tier
submission.

## Honesty note

This is active work in progress. Discussion of this project — in papers,
interviews, or otherwise — should reflect what's actually built vs.
designed vs. planned, per the status table above.
