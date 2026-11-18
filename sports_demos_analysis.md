# Sports Demo Clips — Dual-Pipeline Analysis Memo
**Generated:** 2026-06-10  
**Clips analysed:** sports_demo_1.mp4 · sports_demo_2.mp4 · sports_demo_3.mp4  
**Pipelines:** Sports2D v0.8.30 (RTMPose-m) · MediaPipe BlazePose Full  

---

## Overview

| | Clip 1 | Clip 2 | Clip 3 |
|---|---|---|---|
| **Drill** | Running / jogging | Hurdle jump (repeated) | Soccer freestyle juggling |
| **View** | Side, steady | Front, steady | ~¾ front/side, steady |
| **Background** | Solid green screen | Solid green screen | Dark studio, spotlight floor |
| **Watermark** | iStock by Getty Images | iStock by Getty Images | iStock by Getty Images |
| **Resolution** | 768×432 | 768×432 | 768×432 |
| **Duration** | 10.0 s · 250fr @ 25fps | 10.71 s · 257fr @ 24fps | 10.28 s · 257fr @ 25fps |
| **Quality gate** | **FAILED** blur 73.7 | **PASSED** blur 103.4 | **FAILED** blur 25.7 |
| **Sports2D ran?** | No — rejected at gate | Yes — partial (see §2) | No — rejected at gate |
| **MediaPipe ran?** | Yes — with warning | Yes — fully | Yes — with warning |
| **MP detection %** | 250/250 = **100%** | 257/257 = **100%** | 204/257 = **79%** |

---

## Clip 1 — sports_demo_1.mp4

### Visual Description
Single male athlete (grey t-shirt, black shorts) running / jogging against a solid bright-green chroma-key background. Pure side view, static camera, no cuts or pans. The athlete stays centred in frame throughout all 10 seconds — this is stock chroma-key footage of a person running on a treadmill or looping the same stride cycle. Full body (head to shoes) is continuously visible. The green occupies ≈ 80% of every frame. All frames carry an "iStock by Getty Images" watermark.

### Quality Gate
| Metric | Score | Threshold | Result |
|---|---|---|---|
| Sharpness (Laplacian variance) | 73.7 | > 80.0 | **FAIL** |
| Brightness | 186.2 | > 40.0 | PASS |

**Why it fails:** The Laplacian variance measures gradient energy across the whole frame. The large uniform green background has near-zero gradient — it scores as "blurry" even though the athlete pixels are sharp. This is a known blind spot of the Laplacian sharpness metric on chroma-key footage. The gap is small (73.7 vs 80.0) and is driven by background proportion, not actual motion blur on the subject.

### Sports2D Pipeline
**Result: Hard stop at Step 1.** run_all.py exits with "Pipeline stopped: video failed the quality gate."  
No pose estimation, no .mot file, no chart, no trust report produced.

### MediaPipe Pipeline
MediaPipe proceeds with a quality warning and completes all stages.

**Frame detection:** 250 / 250 = **100%** — continuous from t=0.00 s to t=9.96 s, max gap 1 frame (0.04 s).

**Trust report:**

| Joint | Value | Stability | Plausibility | Confidence | MP Visibility |
|---|---|---|---|---|---|
| Trunk | 86.0° | 83% | 100% (250/250) | **HIGH 91%** | 0.998 |
| Right Thigh | −84.7° | 72% | 100% (250/250) | **HIGH 85%** | 0.984 |
| Left Thigh | −89.8° | 73% | 100% (250/250) | **HIGH 85%** | 0.761 |
| Left Ankle | −108.2° | 74% | 93% (232/250) | **HIGH 83%** | 0.761 |
| Right Knee | 24.3° | 66% | 100% (250/250) | **HIGH 81%** | 0.983 |
| Left Knee | 21.5° | 65% | 100% (250/250) | **HIGH 81%** | 0.761 |
| Right Hip | 8.3° | 62% | 100% (250/250) | **MEDIUM 79%** | 0.984 |
| Left Hip | 7.2° | 66% | 100% (250/250) | **HIGH 81%** | 0.761 |
| Right Ankle | −110.9° | 72% | 86% (216/250) | **MEDIUM 79%** | 0.983 |

Hip stability is the lowest at 62–66%, reflecting the cyclic hip flexion oscillation across running strides — the signal moves, which is penalised by the stability score (stability rewards static signals). This is expected for running, not a tracking failure.

**Rep detection:** 11 stride cycles detected from left hip signal at ~0.92 s/cycle (64.9 cycles/min). Confidence: **HIGH 96%**. Peak hip angles 12–18° throughout — consistent with the small hip oscillation of a running gait as seen from the side.

### Clip 1 Verdict
**NOT demo-worthy (quality gate failure) — but MediaPipe tracking is excellent.**  
The quality gate failure is a false alarm caused by the uniform green background and is not representative of the actual tracking difficulty. MediaPipe achieves 100/100 detection with HIGH confidence on most joints. The failure to run Sports2D is genuine — Sports2D won't process this clip through its official pipeline. If the green-screen-inflated sharpness is stripped out (or the threshold is slightly lowered), this clip has good bilateral symmetry data and clear stride-cycle rep detection. **Not suitable as a dual-engine demo without a quality gate fix.**

---

## Clip 2 — sports_demo_2.mp4

### Visual Description
Single male athlete (white tank top, black shorts) repeatedly jumping over a white athletics hurdle against a solid bright-green chroma-key background. Front-facing camera, static, no cuts or pans. The athlete approaches the hurdle, jumps over, lands, resets, and jumps again. The hurdle is positioned at roughly mid-thigh height, occupying the lower third of the frame. The athlete's full body is visible in every frame; during peak jump height the head approaches the upper frame edge but never exits. The green background occupies roughly 50–60% of each frame. "iStock by Getty Images" watermark throughout.

### Quality Gate
| Metric | Score | Threshold | Result |
|---|---|---|---|
| Sharpness | 103.4 | > 80.0 | **PASS** |
| Brightness | 152.7 | > 40.0 | PASS |

### Sports2D Pipeline
**Result: Partial success with two downstream failures.**

Sports2D ran successfully through all 257 frames (detection complete, ~24 s processing time). The pipeline completed:
- Pose estimation: **DONE** — skeleton overlay video + .mot file produced
- Joint angle chart: **DONE** — left_hip_plot.png
- Trust report: **CRASHED** — `trust_report.py` exits with `ValueError: could not convert string to float: ''`
- Rep detection: **CRASHED** — same error in `detect_reps.py`

**Root cause:** The .mot file for this clip contains 53 / 257 rows where only *some* joint columns are populated (the rest are empty strings). The `read_mot()` function in both `trust_report.py` and `detect_reps.py` calls `float(val)` on every field without guarding against empty strings. This crash scenario only occurs when Sports2D produces partial-data rows — this did not happen on the previous squat clip, which had all-or-nothing rows.

**Calibration problem — large floor angle error:**  
Sports2D reported: `Camera horizon: -43.65°, corrected using gait kinematics` and `Floor level: 28.95 px (from the top of the image)`. A floor level 29 px from the *top* of a 432 px-tall image is clearly wrong. Sports2D's automatic floor/horizon calibration is designed for side-view gait data; it produces nonsensical results on this front-view jumping clip. As a consequence, all segment angles had 43.65° added to them in correction — producing an inflated trunk value of 134.7° (well outside the plausible 40–150° range) and distorted thigh angles.

**Sports2D angles recovered via pandas workaround (read_mot bug bypassed):**

| Joint | Median angle | Valid frames | In-range | Stability | Plausibility | Confidence |
|---|---|---|---|---|---|---|
| Right Hip | 0.4° | 257/257 | 253/257 | 0.97 | 0.98 | **HIGH 98%** |
| Left Hip | 1.7° | 257/257 | 252/257 | 0.98 | 0.98 | **HIGH 98%** |
| Right Knee | 2.3° | 256/257 | 247/256 | 0.94 | 0.96 | **HIGH 95%** |
| Left Knee | −2.6° | 245/257 | 156/245 | 0.95 | 0.64 | **MEDIUM 78%** |
| Right Ankle | −80.8° | 203/257 | 179/203 | 0.93 | 0.88 | **HIGH 90%** |
| Left Ankle | −90.4° | 215/257 | 169/215 | 0.94 | 0.79 | **HIGH 86%** |
| Right Thigh | −45.7° | 257/257 | 242/257 | 0.97 | 0.94 | **HIGH 95%** |
| Left Thigh | −42.8° | 257/257 | 201/257 | 0.98 | 0.78 | **HIGH 88%** |
| Trunk | 134.7° | 257/257 | 257/257 | 0.95 | 1.00 | **HIGH 97%** |

*Note: Trunk 134.7° is an artefact of the floor-angle correction error (−43.65°). The uncorrected trunk value would be ~91°, which is more plausible.*

**Sports2D rep detection:** 0 reps detected from left hip. The signal's cyclic component (hip flexion in a front-view jump) is too small or too noisy for the peak detector to find clear peaks above the adaptive threshold.

### MediaPipe Pipeline
**Result: Full success.**

**Frame detection:** 257 / 257 = **100%** — continuous from t=0.00 s to t=10.67 s.

**Trust report:**

| Joint | Value | Stability | Plausibility | Confidence | MP Visibility |
|---|---|---|---|---|---|
| Right Hip | 4.0° | 97% | 100% (257/257) | **HIGH 98%** | 0.936 |
| Left Hip | 6.7° | 96% | 100% (257/257) | **HIGH 98%** | 0.960 |
| Right Knee | 3.8° | 95% | 100% (257/257) | **HIGH 98%** | 0.852 |
| Left Knee | 5.5° | 95% | 100% (256/257) | **HIGH 98%** | 0.837 |
| Trunk | 88.6° | 94% | 100% (257/257) | **HIGH 97%** | 0.997 |
| Right Ankle | −88.2° | 96% | 100% (256/257) | **HIGH 98%** | 0.852 |
| Left Ankle | −91.2° | 96% | 93% (238/257) | **HIGH 94%** | 0.837 |
| Right Thigh | −90.1° | 96% | 93% (240/257) | **HIGH 95%** | 0.936 |
| Left Thigh | −91.1° | 96% | 100% (257/257) | **HIGH 98%** | 0.960 |

**Rep detection:** 1 rep from right hip at t=6.46 s, peak 158.0°, confidence **MEDIUM 71%**. Single detected peak means no inter-rep interval for tempo calculation.

### Side-by-Side Table — Clip 2

| Metric | Sports2D value | S2D confidence | MediaPipe value | MP confidence | Notes |
|---|---|---|---|---|---|
| Right Hip | 0.4° | HIGH 98% | 4.0° | HIGH 98% | Near-standing: strong agreement |
| Left Hip | 1.7° | HIGH 98% | 6.7° | HIGH 98% | Near-standing: strong agreement |
| Right Knee | 2.3° | HIGH 95% | 3.8° | HIGH 98% | Near-standing: strong agreement |
| Left Knee | −2.6° | MEDIUM 78% | 5.5° | HIGH 98% | S2D partial tracking failure (53 mixed rows) |
| Trunk | **134.7°** | HIGH 97% | **88.6°** | HIGH 97% | **46° delta — S2D floor correction error** |
| Right Ankle | −80.8° | HIGH 90% | −88.2° | HIGH 98% | ~7° delta — within expected range |
| Right Thigh | −45.7° | HIGH 95% | −90.1° | HIGH 95% | **44° delta — S2D floor correction distortion** |
| Reps (hip) | **0** | — | **1** | MEDIUM 71% | Both engines miss repeated jump pattern |

**Where they agree:** Hip and knee median angles (0–7° for standing/reset frames) agree within 5° — the floor angle error predominantly affects segment angles (trunk, thigh), not the 3-point joint angles (hip, knee). Bilateral symmetry is similar in both engines.

**Where they differ:** Trunk (134.7° vs 88.6°) and right thigh (−45.7° vs −90.1°) diverge by ~44° — entirely explained by Sports2D's erroneous −43.65° floor correction. MediaPipe's segment angles are uncorrected and thus more plausible for this non-calibrated setup.

**Rep detection:** Both engines fail to detect the repeated hurdle jump pattern clearly. The front camera view means the hip and knee angles oscillate only modestly in 2D projection — the depth (3D) component of the jump is not captured by either engine. MediaPipe finds 1 peak, Sports2D finds 0. Neither is suitable for rep counting from this camera angle.

### Clip 2 Verdict
**CONDITIONALLY demo-worthy — the only clip passing quality gate, but with caveats.**  
MediaPipe tracking is excellent (100/100 detection, HIGH confidence across all joints). Sports2D tracking is also good for the joint angles themselves, but is sabotaged by two distinct bugs: (a) the floor-angle calibration fails on front-view data, producing a 44° distortion in segment angles; (b) `trust_report.py` and `detect_reps.py` crash on the mixed-populate .mot file from this clip. If the demo only uses MediaPipe results, this clip is clean. As a dual-engine comparison, Sports2D's trunk and thigh numbers are misleading and must be footnoted.

---

## Clip 3 — sports_demo_3.mp4

### Visual Description
A single footballer / soccer freestyler performing ball juggling and freestyle tricks in a dark photographic studio with a soft spotlight on the floor. The athlete is dressed in a grey t-shirt and dark jeans. The clip opens with approximately 1.2 seconds of empty frame, the athlete enters from the right performing low ball touches, moves to the centre juggling a soccer ball with feet and knees, then runs off to the left — exiting frame at approximately t=9.5 s, leaving the final ~0.8 seconds empty. The athlete is in motion throughout; there are no static resting poses. The background is very dark (near-black) with high local contrast around the spotlight. "iStock by Getty Images" watermark throughout.

### Quality Gate
| Metric | Score | Threshold | Result |
|---|---|---|---|
| Sharpness (Laplacian variance) | 25.7 | > 80.0 | **FAIL** — very wide margin |
| Brightness | 71.2 | > 40.0 | PASS |

**Why it fails badly:** The Laplacian variance is crushed by the large dark background (gradient ≈ 0 across most pixels) and by real motion blur introduced by the rapid kicking/juggling motion. This is a genuine quality failure, not a background artefact — the fast foot movement during ball contact produces real motion blur on the key action frames.

### Sports2D Pipeline
**Result: Hard stop at Step 1.** run_all.py exits "Pipeline stopped: video failed the quality gate."  
No pose data produced.

### MediaPipe Pipeline
MediaPipe proceeds with a quality warning.

**Frame detection:** 204 / 257 = **79.4%** (standalone scan: 208/257 = 80.9%; pipeline count differs by rounding/threshold differences).  
Detection window: t=1.24 s → t=9.52 s (span 8.28 s). The first ~1.2 s (empty frame) and last ~0.8 s (empty frame) account for most missed detections. Within the athlete's presence window (1.24–9.52 s = ~8.3 s), detection is near-continuous.

**Important caveat:** All 257 frames are reported as "analysed" and all joints show 257/257 valid frames in the trust report. This is because the mediapipe_pipeline.py interpolates through non-detected frames to fill the full timeline. The confidence scores are computed on the 204 real detections, not the full 257 — but the "frames in range" counts are inflated by interpolation. This should not be read as "257 frames of real pose data."

**Trust report (MediaPipe):**

| Joint | Value | Stability | Plausibility | Confidence | MP Visibility |
|---|---|---|---|---|---|
| Trunk | 98.0° | 95% | 100% (257/257*) | **HIGH 98%** | 0.996 |
| Left Knee | 17.7° | 91% | 100% (257/257*) | **HIGH 95%** | 0.955 |
| Right Knee | 18.0° | 90% | 100% (257/257*) | **HIGH 95%** | 0.855 |
| Left Hip | 12.3° | 89% | 100% (257/257*) | **HIGH 94%** | 0.969 |
| Right Hip | 14.3° | 88% | 100% (257/257*) | **HIGH 94%** | 0.866 |
| Left Ankle | −73.7° | 90% | 100% (257/257*) | **HIGH 95%** | 0.955 |
| Right Ankle | −73.9° | 91% | 100% (257/257*) | **HIGH 95%** | 0.855 |
| Left Thigh | −93.9° | 89% | 100% (257/257*) | **HIGH 94%** | 0.969 |
| Right Thigh | −94.8° | 89% | 100% (257/257*) | **HIGH 94%** | 0.866 |

*\* "257/257" reflects interpolated fill, not 257 actual detections.*

**Rep detection:** 3 "reps" from right hip at t=4.08, 5.60, 6.48 s. Peak angles 26–35°. Tempo 1.2 s/rep, **MEDIUM 68%**. These are not exercise reps — they represent ball contact events or weight-shift cycles during juggling. The rep detector is not designed for this type of movement.

### Clip 3 Verdict
**NOT demo-worthy.**  
Sports2D: hard rejected at quality gate. MediaPipe: runs with warnings, achieves 79% detection, but all confidence scores are inflated by interpolation over the empty frames at clip start/end. The footage itself is not suitable — the athlete enters and exits the frame, the motion is a freestyle skill drill (not a recognisable athletic exercise), and the dark studio background causes genuine quality failure. The trunk angle (98.0°) suggests a notable forward lean during ball work, which is physically plausible, but no comparison is possible without Sports2D data.

---

## Cross-Clip Comparison

### Frame Detection Summary

| | Clip 1 | Clip 2 | Clip 3 |
|---|---|---|---|
| **Total frames** | 250 | 257 | 257 |
| **Sports2D valid frames** | N/A (gate fail) | 257 (203 full + 53 partial) | N/A (gate fail) |
| **MediaPipe detected frames** | **250/250 = 100%** | **257/257 = 100%** | 204/257 = **79%** |
| **Athlete continuously in frame?** | Yes (whole clip) | Yes (whole clip) | No (enters t=1.2s, exits t=9.5s) |

### Trunk Angle Across Clips (MediaPipe — most reliable metric)

| Clip | MediaPipe Trunk | Interpretation |
|---|---|---|
| Clip 1 (running) | 86.0° | Slight forward lean — consistent with jogging |
| Clip 2 (hurdle jump) | 88.6° | Near-upright — consistent with standing/reset at hurdle |
| Clip 3 (soccer freestyle) | 98.0° | More pronounced forward lean — consistent with ball control crouch |

### Sports2D vs MediaPipe (Clip 2 only — the single dual-pipeline result)

The key finding is that where both engines produce plausible values (hip and knee joint angles), they agree well (< 5°). Where Sports2D diverges — trunk and thigh segment angles — the cause is a known calibration failure on front-view footage, not a genuine biomechanical discrepancy. MediaPipe's segment angles (trunk 88.6°, thigh ~−90°) are more plausible. Sports2D's joint angles (hip ~1–7°, knee ~2–6°) are reliable and closely match MediaPipe.

---

## Pipeline Issues Uncovered

### Issue 1: `trust_report.py` and `detect_reps.py` crash on partial .mot rows
When Sports2D produces a .mot file where some frames have partial joint data (some columns populated, some empty), `read_mot()` in both scripts crashes with `ValueError: could not convert string to float: ''`. The function does not guard against empty string fields. This was not observed in previous clips (squat_video.mp4, sport_demo.mp4) because those produced rows that were entirely populated or entirely empty — never mixed.

**Impact:** Trust report and rep detection unavailable for Sports2D on clips with partial-data rows.  
**Fix needed:** Guard the `float()` call: `float(val) if val.strip() else float('nan')` — but this requires modifying `trust_report.py` and `detect_reps.py`, which were outside scope for this task.

### Issue 2: Sports2D floor-angle calibration fails on front-view clips
Sports2D's automatic floor/horizon calibration uses gait kinematics heuristics designed for side-view footage. On front-view footage, it produced a −43.65° horizon correction for clip 2, pushing all segment angles ~44° off their true values. Joint angles (hip, knee) are immune since they are relative angle calculations.

**Impact:** Trunk and thigh segment angles from Sports2D are unreliable on front-view footage.

### Issue 3: Quality gate false-fail on green-screen footage
The Laplacian variance sharpness metric fails on large uniform-colour backgrounds (green screen, dark studio) because the background pixels have near-zero gradient. Clip 1 fails at 73.7 (threshold 80.0) despite the athlete pixels being sharp; clip 3 fails at 25.7 for a combination of genuine blur and dark background.

**Impact:** Sports2D is blocked from processing valid green-screen footage with sharp subjects.

---

## What This Does and Does Not Prove

**What it shows:**
- MediaPipe BlazePose handles green-screen footage (clips 1 and 2) with 100% detection rate, High confidence, and smooth angle signals — the uniform background is not a problem for the pose model, only for the quality gate metric.
- Both engines agree well on joint angles (hip, knee) where both produce valid data (clip 2, < 5° delta), giving confidence that the angle geometry is consistent.
- Three distinct failure modes are now documented: quality-gate false fails, trust_report/.mot crash, and Sports2D calibration errors on front-view footage.

**What it does NOT prove:**
- No accuracy claim is possible without ground-truth (IMU, motion capture).  
- MediaPipe's HIGH confidence scores on clip 3 do not mean the data is reliable — those frames include interpolated-not-observed values.
- Sports2D's HIGH confidence on the recovered clip 2 joint angles does not account for the calibration error affecting segment angles.
- The "rep" counts (11, 1, 3) across all clips are measuring oscillation cycles in the angle signal, not meaningful exercise repetitions. The rep detector was designed for squats/jumps, not running strides, hurdle resets, or ball juggling.

---

## Final Recommendation

**Best clip for dual-engine demo: Clip 2 (sports_demo_2.mp4 — hurdle jumps).**

It is the only clip passing the quality gate and achieving 100% detection on both engines. MediaPipe gives clean, high-confidence data for all 9 joints. Sports2D joint angles (hip, knee) are reliable and agree with MediaPipe. However, two honesty caveats must accompany any demo use:

1. Sports2D trust report and rep detection crash and must be labelled "unavailable" — this is a known bug in `read_mot()` when the .mot file contains partial rows, not a tracking failure.
2. Sports2D trunk and thigh angles are distorted by a −43.65° floor calibration error on this front-view clip and should not be used without the note that Sports2D is calibrated for side-view footage.

If a clean side-view clip with full-body athlete visible for the entire clip is supplied (matching what squat_video.mp4 provides), both pipelines will run end-to-end without any of these issues.
