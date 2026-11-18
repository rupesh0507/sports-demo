"""
trust_report.py
---------------
Generates a "trust label" for every joint angle measured by Sports2D.

For each metric it records:
  - metric name and value
  - confidence score (derived from signal stability — how smooth vs jumpy the signal is)
  - video quality grade (blur + brightness, reused from check_video_quality.py logic)
  - pose model details (RTMPose-m / HALPE_26)

Output: prints human-readable records to the console AND saves a JSON file
        alongside the Sports2D output.

Usage (preferred — derive folder from video name):
  python trust_report.py squat_video.mp4

Usage (explicit paths — still works):
  python trust_report.py path/to/angles.mot path/to/video.mp4

When given a video filename, the script derives the output folder as:
  Sports2D_output/<videoname>_Sports2D/<videoname>_Sports2D_angles_person00.mot
"""

import sys
import os
import json
import glob
import datetime
import numpy as np
import cv2


# ── 1. CONFIGURATION ────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

BLUR_THRESHOLD       = 80.0
BRIGHTNESS_THRESHOLD = 40.0
SAMPLE_FRAMES        = 20

# RTMPose model used by Sports2D 0.8.30 in "balanced" mode
MODEL_INFO = {
    "name":            "RTMPose-m",
    "skeleton":        "HALPE_26  (26 keypoints, body + feet)",
    "backend":         "OpenVINO CPU  (auto-selected; no CUDA found)",
    "sports2d_version": "0.8.30",
    "reference":       "Jiang et al. 2023 — RTMPose: Real-Time Multi-Person Pose Estimation "
                       "(arXiv:2303.07399)"
}

# ── 2. HELPERS ───────────────────────────────────────────────────────────────

def mot_path_from_video(video_path):
    """
    Derive the Sports2D angles .mot path from a video filename.
    squat_video.mp4 → Sports2D_output/squat_video_Sports2D/squat_video_Sports2D_angles_person00.mot
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.join("Sports2D_output", f"{stem}_Sports2D")
    return os.path.join(folder, f"{stem}_Sports2D_angles_person00.mot")


def read_mot(path):
    """Read a Sports2D / OpenSim .mot file into parallel lists."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i + 1 for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    header = lines[start].strip().split("\t")
    data = {col: [] for col in header}
    for line in lines[start + 1:]:
        parts = line.strip().split("\t")
        if len(parts) != len(header):
            continue
        for col, val in zip(header, parts):
            data[col].append(float(val) if val.strip() else float('nan'))
    return {k: np.array(v) for k, v in data.items()}



# ── Plausible ranges (Sports2D conventions) ──────────────────────────────────
#
# Joint angles: 0° = limb straight, positive = bent.
# Segment angles: anticlockwise from horizontal; shank/thigh point downward
#   (naturally negative for a standing person).
#
# These ranges are derived from inspecting actual Sports2D output, NOT from
# anatomical textbooks — the two conventions are different.
#
PLAUSIBLE_RANGES = {
    "right knee":   (-20,  160),   # joint flexion
    "left knee":    (-20,  160),
    "right hip":    (-60,  160),   # joint flexion (signed)
    "left hip":     (-60,  160),
    "right ankle":  (-150,  30),   # shank-from-horizontal
    "left ankle":   (-150,  30),
    "right thigh":  (-150,  10),   # thigh-from-horizontal
    "left thigh":   (-150,  10),
    "trunk":        (  40, 150),   # trunk-from-horizontal (~90° upright)
}


# ── NEW: Keypoint dependencies per metric (HALPE_26 names) ───────────────────
# Maps each angle column to the keypoints whose positions determine it.
# Used for lineage tagging; ready for confidence lookup if Sports2D ever saves scores.
METRIC_KEYPOINTS = {
    "right knee":  ["RHip", "RKnee", "RAnkle"],       # hip-knee-ankle angle
    "left knee":   ["LHip", "LKnee", "LAnkle"],
    "right hip":   ["RKnee", "RHip", "RShoulder"],     # knee-hip-shoulder angle
    "left hip":    ["LKnee", "LHip", "LShoulder"],
    "right ankle": ["RAnkle", "RKnee"],                # shank segment from horizontal
    "left ankle":  ["LAnkle", "LKnee"],
    "right thigh": ["RHip", "RKnee"],                  # thigh segment from horizontal
    "left thigh":  ["LHip", "LKnee"],
    "trunk":       ["Hip", "Neck"],                    # trunk segment (computed midpoints)
}

# ── NEW: Metric classification ────────────────────────────────────────────────
# "direct"   — angle computed straight from visible keypoint positions
# "inferred" — would require 2D→3D lifting (not used here)
# "proxy"    — stands in for a quantity not directly measurable (not used here)
METRIC_TYPE = {col: "direct" for col in METRIC_KEYPOINTS}

# ── NEW: Why model_keypoint_confidence is null ────────────────────────────────
# RTMPose outputs per-keypoint scores (0–1, shape Nframes×26) during inference.
# Sports2D v0.8.30 uses them internally in process.py:
#   • line 1882-1883: filter keypoints below keypoint_likelihood_threshold (0.3)
#   • line 2044:      build score_data DataFrame (Nframes × 26 per person)
#   • line 1938/2546: draw green→red colour scale on skeleton video
#   • line 1200-1209: floor-line estimation
# BUT score_data is never written to disk. The TRC files store only X/Y/Z
# positions (Z = 0 for 2D). To obtain raw scores, modify Sports2D/process.py
# to save score_data to CSV alongside the TRC output (e.g. after line 2044).
MODEL_KEYPOINT_CONFIDENCE_NOTE = (
    "Not available in Sports2D v0.8.30 output files. "
    "RTMPose per-keypoint scores exist in memory as score_data "
    "(pd.DataFrame, shape Nframes×26) at Sports2D/process.py ~line 2044, "
    "but are never written to disk. The TRC files store only keypoint "
    "X/Y positions; Z is always 0.0. "
    "To capture scores, patch process.py to export score_data to CSV."
)


def normalize_to_range(signal, lo, hi):
    """
    Per-frame: shift each value by the smallest integer multiple of 360°
    that places it inside [lo, hi].  Corrects Sports2D's occasional 360°-wrap
    artifacts (e.g. -348° reported instead of 12°) without touching genuine
    tracking-failure frames.
    """
    mid = (lo + hi) / 2.0
    result = signal.copy().astype(float)
    for i, v in enumerate(result):
        if lo <= v <= hi:
            continue
        best, best_dist = v, abs(v - mid)
        for k in (1, 2, 3):
            for cand in (v + k * 360, v - k * 360):
                if lo <= cand <= hi:
                    d = abs(cand - mid)
                    if d < best_dist:
                        best, best_dist = cand, d
        result[i] = best
    return result


def stability_confidence(signal):
    """
    Frame-to-frame smoothness: 0–1.
    A noise ratio below 5 % of range → ~1.0.
    Above 30 % → ~0.0.
    """
    NOISE_CAP = 0.30
    diffs = np.abs(np.diff(signal))
    frame_diff = float(np.mean(diffs))
    sig_range = float(np.max(signal) - np.min(signal))
    raw_noise = frame_diff / max(sig_range, 1.0)
    return round(1.0 - min(raw_noise / NOISE_CAP, 1.0), 3)


def plausibility_confidence(signal, col_name):
    """
    Fraction of frames whose value falls within the plausible range AFTER
    360° wrap-normalisation.  Any frames still outside the range are genuine
    tracking failures and lower the score.
    """
    lo, hi = PLAUSIBLE_RANGES.get(col_name, (-360, 360))
    norm = normalize_to_range(signal, lo, hi)
    in_range = np.sum((norm >= lo) & (norm <= hi))
    return round(float(in_range) / len(norm), 3)


def confidence_label(score):
    if score >= 0.80:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


def check_video_quality(video_path):
    """Reuse the same blur + brightness logic as check_video_quality.py."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(SAMPLE_FRAMES, total), dtype=int)
    blur_scores, bright_scores = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright_scores.append(float(np.mean(gray)))
    cap.release()
    if not blur_scores:
        return None
    avg_blur = round(float(np.mean(blur_scores)), 1)
    avg_bright = round(float(np.mean(bright_scores)), 1)
    passed = (avg_blur >= BLUR_THRESHOLD) and (avg_bright >= BRIGHTNESS_THRESHOLD)
    return {
        "grade":               "GOOD" if passed else "BAD",
        "blur_score":          avg_blur,
        "brightness_score":    avg_bright,
        "blur_threshold":      BLUR_THRESHOLD,
        "brightness_threshold": BRIGHTNESS_THRESHOLD,
    }


def representative_value(signal, col_name):
    """
    Report the peak and median computed only from in-range (valid) frames.
    Using in-range frames avoids tracking-failure outliers inflating the value.
    """
    lo, hi = PLAUSIBLE_RANGES.get(col_name, (-1e9, 1e9))
    norm = normalize_to_range(signal, lo, hi)
    valid = norm[(norm >= lo) & (norm <= hi)]
    if len(valid) == 0:
        return round(float(np.median(norm)), 1), "median  (no in-range frames)"
    peak = round(float(np.max(np.abs(valid))), 1)
    median = round(float(np.median(valid)), 1)
    return median, f"median of {len(valid)}/{len(signal)} valid frames  (peak excursion {peak}°)"


# ── 3. MAIN ──────────────────────────────────────────────────────────────────

def build_report(mot_path, video_path):
    data   = read_mot(mot_path)
    qcheck = check_video_quality(video_path)
    time   = data.pop("time", None)
    n_frames = len(next(iter(data.values())))

    measurements = []
    for col, raw_signal in data.items():
        lo, hi = PLAUSIBLE_RANGES.get(col, (-360, 360))
        # Normalise 360°-wrap artifacts, then work with the normalised signal
        signal = normalize_to_range(raw_signal, lo, hi)
        stab  = stability_confidence(signal)
        plaus = plausibility_confidence(signal, col)   # uses already-normalised signal
        conf  = round(float(np.sqrt(stab * plaus)), 3)
        value, vtype = representative_value(signal, col)
        sig_min = round(float(np.min(signal)), 1)
        sig_max = round(float(np.max(signal)), 1)
        measurements.append({
            "metric":               col.replace(" ", "_"),
            "display_name":        col.title(),
            "value_degrees":       value,
            "value_type":          vtype,
            "signal_range_degrees": [sig_min, sig_max],
            "plausible_range_degrees": [lo, hi],
            "frames_analyzed":     n_frames,
            "frames_in_range":     int(np.sum((signal >= lo) & (signal <= hi))),
            "stability_score":     stab,
            "plausibility_score":  plaus,
            "confidence_score":    conf,
            "confidence_label":    confidence_label(conf),
            "confidence_basis":    "geometric mean of signal-stability and anatomical-plausibility scores",
            # ── NEW additive fields — existing fields above are unchanged ──────
            "metric_type":                 METRIC_TYPE.get(col, "direct"),
            "produced_by":                 "deterministic_formula",
            "processing_timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
            "metric_keypoints":            METRIC_KEYPOINTS.get(col, []),
            "model_keypoint_confidence":   None,
            "model_keypoint_confidence_note": MODEL_KEYPOINT_CONFIDENCE_NOTE,
        })

    report = {
        "report_metadata": {
            "generated_at":  datetime.datetime.now().isoformat(timespec="seconds"),
            "video_file":    os.path.basename(video_path),
            "video_quality": qcheck,
            "pose_model":    MODEL_INFO,
            # ── NEW additive field ─────────────────────────────────────────────
            "model_keypoint_confidence_availability": MODEL_KEYPOINT_CONFIDENCE_NOTE,
        },
        "measurements": measurements,
    }
    return report


def print_report(report):
    meta  = report["report_metadata"]
    qc    = meta["video_quality"]
    model = meta["pose_model"]

    sep = "=" * 62
    print(f"\n{sep}")
    print("  SPORTS ANALYTICS  —  TRUST REPORT")
    print(f"  Generated : {meta['generated_at']}")
    print(f"  Video     : {meta['video_file']}")
    print(f"{sep}")

    print(f"\n  VIDEO QUALITY GATE")
    print(f"    Grade      : {qc['grade']}")
    print(f"    Sharpness  : {qc['blur_score']}  (threshold > {qc['blur_threshold']})")
    print(f"    Brightness : {qc['brightness_score']}  (threshold > {qc['brightness_threshold']})")

    print(f"\n  POSE MODEL")
    print(f"    Name       : {model['name']}")
    print(f"    Skeleton   : {model['skeleton']}")
    print(f"    Backend    : {model['backend']}")
    print(f"    Version    : Sports2D {model['sports2d_version']}")
    print(f"    Reference  : {model['reference']}")

    print(f"\n  MEASUREMENTS  ({len(report['measurements'])} joints / segments)")
    print(f"  {'Metric':<20}  {'Value':>8}  {'Stability':>10}  {'Plausibility':>13}  {'Overall':>13}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*13}  {'-'*13}")
    for m in report["measurements"]:
        stab_s  = f"{m['stability_score']:.0%}"
        plaus_s = f"{m['plausibility_score']:.0%}  ({m['frames_in_range']}/{m['frames_analyzed']}fr)"
        conf_s  = f"{m['confidence_score']:.0%}  {m['confidence_label']}"
        print(f"  {m['display_name']:<20}  {m['value_degrees']:>7.1f}°  {stab_s:>10}  {plaus_s:>13}  {conf_s:>13}")
    print()


def print_example_record(report):
    # Pick the measurement with the highest confidence to show as the example
    valid = [m for m in report["measurements"]
             if m["confidence_score"] is not None and m["confidence_score"] == m["confidence_score"]]
    best = max(valid, key=lambda m: m["confidence_score"]) if valid else report["measurements"][0]
    meta = report["report_metadata"]
    print("=" * 62)
    print("  EXAMPLE — ONE TRUSTED MEASUREMENT RECORD")
    print("=" * 62)
    print(json.dumps({
        "metric":               best["metric"],
        "display_name":         best["display_name"],
        "value_degrees":        best["value_degrees"],
        "value_type":           best["value_type"],
        "signal_range_degrees": best["signal_range_degrees"],
        "frames_analyzed":      best["frames_analyzed"],
        "confidence_score":     best["confidence_score"],
        "confidence_label":     best["confidence_label"],
        "confidence_basis":     best["confidence_basis"],
        "video_quality_grade":  meta["video_quality"]["grade"],
        "video_blur_score":     meta["video_quality"]["blur_score"],
        "video_brightness":     meta["video_quality"]["brightness_score"],
        "pose_model":           meta["pose_model"]["name"],
        "skeleton":             meta["pose_model"]["skeleton"],
        "sports2d_version":     meta["pose_model"]["sports2d_version"],
    }, indent=2))
    print()


# ── 4. ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trust_report.py <video.mp4>")
        print("   or: python trust_report.py <angles.mot> <video.mp4>")
        sys.exit(1)

    arg1 = sys.argv[1]

    # Single video-file argument → derive both paths from the video name
    if os.path.splitext(arg1)[1].lower() in VIDEO_EXTENSIONS:
        video_path = arg1
        mot_path   = mot_path_from_video(video_path)
    else:
        # Explicit mot + video paths (original usage)
        mot_path   = arg1
        video_path = sys.argv[2] if len(sys.argv) > 2 else None
        if video_path is None:
            print("ERROR: when passing a .mot file directly, also provide the video path.")
            print("  python trust_report.py angles.mot video.mp4")
            sys.exit(1)

    if not os.path.isfile(mot_path):
        print(f"ERROR: angles file not found: {mot_path}")
        print("Make sure Sports2D has been run on this video first.")
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"ERROR: video file not found: {video_path}")
        sys.exit(1)

    report = build_report(mot_path, video_path)

    print_report(report)
    print_example_record(report)

    out_path = os.path.join(os.path.dirname(mot_path), "trust_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full JSON saved: {out_path}")
