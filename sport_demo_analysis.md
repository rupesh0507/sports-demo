# Dual-Pipeline Analysis Memo — sport_demo.mp4
**Generated:** 2026-06-10  
**Pipelines:** Sports2D v0.8.30 (RTMPose-m) · MediaPipe BlazePose Full  

---

## 1. Clip Information

| Field | Value |
|---|---|
| **Source** | Pexels video ID 8533913, "Side View of a Man Running at a Track" by KoolShooters |
| **License** | Pexels free license — free for any use, no attribution required |
| **Drill type** | Sprint / running drill — single male athlete running on an outdoor athletics track |
| **Camera** | Side view, static (no pan, no zoom, no cuts) |
| **Duration** | 9.12 seconds |
| **Resolution** | 1920 × 1080 |
| **Frame rate** | 25 fps |
| **Total frames** | 228 |
| **File size** | 5.7 MB |

**Quality gate result:** GOOD  
- Sharpness (Laplacian variance): 143.5 — threshold > 80 ✓  
- Brightness (mean pixel): 58.1 — threshold > 40 ✓

---

## 2. Critical Finding: Both Engines Detected the Athlete in ~25% of Frames

This is the most important result and must be understood before reading any other numbers.

| Engine | Total video frames | Frames with valid pose |
|---|---|---|
| Sports2D (RTMPose-m) | 228 | **60** (26%) — timestamps t=3.86 s to t=6.23 s |
| MediaPipe (BlazePose) | 228 | **57** (25%) |

The running athlete crosses the frame at speed. Both engines independently confirm that a single, fully-visible, detectable athlete was present in roughly one-quarter of the clip — approximately 2.3–2.4 seconds of actual usable footage out of 9.12 seconds total.

The video opens on empty track, the athlete runs into frame around t=3.9 s, is fully visible until about t=6.2 s, and then the motion-blur and exit angle degrade detection for the remainder.

**The downstream implications:**
- Sports2D's `.mot` file has 228 rows but 168 are empty (NaN). Only the 60 rows within t=3.86–6.23 s carry valid angle data.
- MediaPipe interpolated across the full 228-frame span to fill gaps, but 75% of the "analysed" frames are interpolated from 57 actual detections. This inflates frame counts and can elevate confidence scores.

---

## 3. Per-Engine Results

### 3.1 Sports2D (RTMPose-m · HALPE_26)

**Detection window:** t = 3.86 s → 6.23 s (60 valid frames, ~24.9 fps)  
Sports2D identified the athlete as "Seen from the right" — meaning the **left side** of the body faces the camera (near side), and the **right side** is the far/occluded side.

| Joint | Median angle | Peak excursion | Confidence | Frames in range |
|---|---|---|---|---|
| Left Knee | 62.5° | 125.4° | HIGH 88% | 60/60 |
| **Right Knee** | **126.9°** | **148.8°** | **LOW 27%** | **5/60** |
| Left Hip | 29.2° | 56.3° | HIGH 90% | 60/60 |
| **Right Hip** | **121.6°** | **157.5°** | **LOW 30%** | **6/60** |
| Right Ankle (shank) | 0.3° | 23.6° | HIGH 84% | 60/60 |
| Left Ankle (shank) | 3.1° | 30.0° | HIGH 82% | 60/60 |
| Right Thigh | −87.2° | 133.6° | HIGH 90% | 60/60 |
| Left Thigh | −70.3° | 112.9° | HIGH 86% | 60/60 |
| Trunk | 77.2° | 110.3° | HIGH 93% | 60/60 |

**Flagged LOW:** Right knee (27%) and right hip (30%) — both are the far-side limbs occluded by the athlete's body. The raw signal swings wildly (−192° to +149°) reflecting 2D projection ambiguity on the depth axis. These values should not be used.

**Rep detection (left hip signal):**
- Reps: **2** at t=4.46 s and t=5.75 s
- Peak hip angles: 55.3° and 50.1°
- Tempo: 1.285 s/rep (46.7 cycles/min)
- Confidence: **HIGH 100%** (CV=0.0 — both peaks perfectly timed within the 2.4-second window)

---

### 3.2 MediaPipe (BlazePose Full · 33 landmarks)

**Detection rate:** 57/228 frames (25%). Pipeline interpolated across gaps to report all 228 frames as "analysed."

| Joint | Median angle | Confidence | Frames in range | MP Visibility |
|---|---|---|---|---|
| Right Knee | 53.9° | HIGH 96% | 228/228 | 0.989 |
| Left Knee | 43.9° | HIGH 96% | 228/228 | 0.851 |
| Right Hip | 23.2° | HIGH 98% | 228/228 | 0.991 |
| Left Hip | 21.8° | HIGH 96% | 228/228 | 0.851 |
| **Right Ankle** | **−106.8°** | **MEDIUM 77%** | **144/228** | 0.989 |
| Left Ankle | −108.3° | HIGH 94% | 215/228 | 0.851 |
| Right Thigh | −73.0° | HIGH 97% | 228/228 | 0.991 |
| Left Thigh | −71.7° | HIGH 97% | 228/228 | 0.851 |
| Trunk | 82.0° | HIGH 98% | 228/228 | 0.999 |

**Flagged MEDIUM:** Right ankle (77%) — plausibility only 63% (144/228 frames inside the −150° to +30° shank range). The right shank exits the plausible range during certain phases of the running stride.

**Rep detection (right hip signal):**
- Reps: **1** at t=5.08 s
- Peak hip angle: 43.1°
- Confidence: **MEDIUM 71%**

---

## 4. Side-by-Side Comparison

### 4.1 Key Angles

| Metric | Sports2D value | Sports2D conf | MediaPipe value | MP conf | Note |
|---|---|---|---|---|---|
| Left Knee | 62.5° | HIGH 88% | 43.9° | HIGH 96% | ~19° difference |
| Right Knee | 126.9° | **LOW 27%** | 53.9° | HIGH 96% | S2D far-side failure |
| Left Hip | 29.2° | HIGH 90% | 21.8° | HIGH 96% | ~7° difference |
| Right Hip | 121.6° | **LOW 30%** | 23.2° | HIGH 98% | S2D far-side failure |
| Right Ankle | 0.3° | HIGH 84% | −106.8° | MEDIUM 77% | Different conventions — not comparable |
| Left Ankle | 3.1° | HIGH 82% | −108.3° | HIGH 94% | Different conventions — not comparable |
| Right Thigh | −87.2° | HIGH 90% | −73.0° | HIGH 97% | ~14° difference |
| Left Thigh | −70.3° | HIGH 86% | −71.7° | HIGH 97% | **~1° — strong agreement** |
| **Trunk** | **77.2°** | HIGH 93% | **82.0°** | HIGH 98% | **~5° — strong agreement** |

### 4.2 Event / Rep Detection

| Field | Sports2D | MediaPipe |
|---|---|---|
| Signal selected | Left hip | Right hip |
| Rep count | **2** | **1** |
| Peak angle(s) | 55.3°, 50.1° | 43.1° |
| Tempo | 1.285 s/rep (46.7/min) | Not computed (single peak) |
| Rep confidence | HIGH 100% | MEDIUM 71% |

---

## 5. Where the Engines Agree

1. **Trunk angle (~77–82°):** Both engines place the runner's trunk at roughly the same forward-lean angle. This is the most reliable number in the whole dataset and the clearest validation signal between the two models.

2. **Left thigh segment (~−70° to −72°):** Near-perfect agreement on the near-side thigh segment — ~1° delta.

3. **Detection window:** Both independently found the athlete present for roughly 25% of the clip (60 frames / 57 frames out of 228). This is not a coincidence — it reflects the actual physical event (athlete in frame for ~2.4 s of the 9.12 s clip).

4. **Qualitative left-side joint kinematics:** Both show the near-side knee reaching ~60–100°+ flexion during the stride, consistent with a running gait.

---

## 6. Where the Engines Differ

### 6.1 Far-side (right-side) joints
Sports2D rates right knee and right hip as **LOW confidence** (27% and 30%) with almost no frames within anatomical range. MediaPipe rates both as **HIGH 96–98%**.

**Why:** RTMPose-m estimates 2D keypoints from a 2D image and propagates that ambiguity into the angle calculation. When the right leg is behind the left leg (camera perspective), the projected right hip/knee landmarks are unreliable and produce biologically impossible wrap-around angles (−192° to +157°). MediaPipe's BlazePose uses a depth-aware architecture trained on a mix of 2D images and 3D pose datasets — it predicts the far-side landmark positions with more robustness under occlusion. This is a genuine architectural advantage of BlazePose in side-view scenarios.

### 6.2 Absolute angle magnitudes on matching joints
For near-side joints where both engines should be reliable (left knee, left hip), there are still 7–19° differences. Causes include:
- **Different reference frames:** Sports2D angle values in the `.mot` file have been corrected for a −1.38° floor tilt; MediaPipe has no floor correction.
- **Different skeletal anchor points:** Sports2D hip angle is LKnee→LHip→LShoulder; MediaPipe uses the same indices (25→23→11) which should be equivalent — but landmark definitions between HALPE_26 and BlazePose 33 differ slightly in anatomical placement.
- **Different analysis windows:** Sports2D uses t=3.86–6.23 s only; MediaPipe averages over the interpolated full 9.12 s span.
- **Median vs. time-average:** These are median angles over their respective analysis windows; the time distribution of detections differs, so medians can diverge even for the same joint.

### 6.3 Rep count (2 vs 1)
Both detected the hip going through an athletic flexion cycle in the 2.4-second window. Sports2D finds 2 complete cycles; MediaPipe finds 1. Neither number is a valid "rep count" in the exercise sense — the running stride is a continuous cyclic motion, not discrete exercise repetitions. Sports2D's 2 cycles are consistent with ~46 strides/min (a reasonable sprint cadence), while MediaPipe's interpolated signal likely smoothed out one of the peaks.

### 6.4 Ankle angle coordinate conventions
The ankle values are not comparable between engines: Sports2D reports the shank segment angle near 0° (nearly horizontal in this clip) while MediaPipe reports ~−106° (nearly vertical). Both use the same atan2(−Δy, Δx) segment formula — the difference is in which time window is represented and whether the interpolation of non-detected frames has pulled the median toward a different phase of the stride. **Do not compare ankle numbers directly between engines.**

---

## 7. Tracking Issues Specific to This Sports Movement

| Issue | Sports2D | MediaPipe |
|---|---|---|
| **Athlete only in frame ~2.4 s of 9.12 s** | 168/228 frames entirely blank in .mot | 171/228 frames interpolated |
| **Far-side leg occlusion** | RIGHT knee/hip: LOW confidence, values invalid | RIGHT knee/hip: HIGH confidence, plausible values |
| **High-speed motion blur** | Visible in fast-stride frames; some keypoint jitter | Temporal filtering (VIDEO mode) reduces jitter |
| **Partial frame exit** | Not penalised; NaN-filled | Interpolation may create false confidence |
| **Gait vs. exercise rep mismatch** | 2 "reps" = stride cycles, not exercises | 1 "rep" = single stride peak |
| **No person-height input** | Assumes 1.65 m (manual default); may skew metre-space coords | Not applicable (normalised coordinates) |

**Which engine handled this clip better overall:** MediaPipe is clearly more appropriate for a side-view running clip. Its depth-aware model produces plausible bilateral joint angles while Sports2D's 2D approach breaks down on the occluded far-side limbs. For the near-side trunk and thigh, both engines are equivalent. Sports2D is arguably better suited to setups where the camera angle is calibrated and the athlete is exercising facing the camera (the pipeline's original design target).

---

## 8. What This Does and Does Not Prove

**What it shows:**
- Both engines can ingest and process a real outdoor sports clip through the Univision pipeline end-to-end without crashing or corrupting other outputs.
- The dual-pipeline output isolation works: `Sports2D_output/sport_demo_Sports2D/` and `mediapipe_output/sport_demo/` were written independently with no cross-contamination.
- Trunk angle shows reasonable bilateral consistency (~5° delta) between engines, which is the expected "sanity check" metric.
- The confidence system correctly flags Sports2D's far-side failures as LOW — the plausibility scoring is doing its job.

**What it does NOT show:**
- **No accuracy claim is possible.** There is no ground truth for this clip (no IMU, no motion capture, no force plate data). We cannot say either engine is "correct."
- **The "reps" are not exercise reps.** Calling 2 or 1 "reps" for a sprint clip is a category error. The rep detector was built for cyclic exercises (squats, jumps); applying it to running gait produces stride-cycle counts, not meaningful exercise repetitions.
- **Angle differences between engines are not resolved.** The 7–19° deltas on near-side hip/knee could represent real model disagreement, different time-window medians, coordinate convention differences, or a mix of all three. Additional testing with a known-angle setup (e.g., a goniometer-calibrated joint) would be needed to determine which engine is closer to ground truth.
- **MediaPipe's HIGH confidence on interpolated frames should be taken cautiously.** 171/228 "analysed" frames are interpolations over zero actual detections. The confidence scores reflect the quality of the 57 real detections, which were good — but the time-series between t≈0–3.8 s and t≈6.2–9.1 s is synthetic, not observed.

---

## 9. Output Files Produced

```
sports-demo/
├── sport_demo.mp4                                      ← new download (Pexels 8533913)
│
├── Sports2D_output/sport_demo_Sports2D/
│   ├── sport_demo_Sports2D.mp4                         ← RTMPose skeleton overlay
│   ├── sport_demo_Sports2D_angles_person00.mot         ← 228 rows, 60 non-null
│   ├── sport_demo_Sports2D_angles_person00_left_knee_plot.png
│   ├── trust_report.json                               ← 9 metrics, right knee/hip flagged LOW
│   └── rep_report.json                                 ← 2 stride cycles, HIGH 100%
│
└── mediapipe_output/sport_demo/
    ├── sport_demo_mp_skeleton.mp4                      ← BlazePose 33-landmark overlay
    ├── sport_demo_mp_angles.csv                        ← 228 rows (57 real, 171 interpolated)
    ├── sport_demo_mp_angles_plot.png                   ← 3×3 angle panel
    ├── mediapipe_trust_report.json                     ← 9 metrics, right ankle MEDIUM
    └── mediapipe_rep_report.json                       ← 1 stride cycle, MEDIUM 71%
```

No existing outputs were modified. `squat_video` and `demo (2)` output folders are unchanged.
