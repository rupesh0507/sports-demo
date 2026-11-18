"""
detect_reps.py
--------------
Reads an existing Sports2D .mot angle file and counts exercise repetitions.

STRICTLY READ-ONLY — does not modify any existing file or output.
Re-uses the same 360°-normalisation and plausible-range masking as
plot_joint_angle.py so that tracking-failure frames never create false reps.

Usage:
  python detect_reps.py squat_video.mp4
  python detect_reps.py demo_video.mp4

Output:
  Console summary + rep_report.json saved inside that video's output folder.
  trust_report.json is never touched.
"""

import sys
import os
import json
import datetime

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ── Plausible angle ranges (mirrored from plot_joint_angle.py — read-only) ───
# These are kept here verbatim so this script is fully standalone.
# Do not edit; if plot_joint_angle.py ranges change, update here to match.
PLAUSIBLE_RANGES = {
    "right knee":   (-20,  160),
    "left knee":    (-20,  160),
    "right hip":    (-60,  160),
    "left hip":     (-60,  160),
    "right ankle":  (-150,  30),
    "left ankle":   (-150,  30),
    "right thigh":  (-150,  10),
    "left thigh":   (-150,  10),
    "trunk":        (  40, 150),
}

# Keyword preference for column selection (higher = preferred for rep counting)
KEYWORD_BONUS = {
    "hip":    5000,   # hip flexion/abduction shows clearest rep signal
    "knee":   2000,   # knee flexion works for squats but often noisier
    "ankle":   500,
    "thigh":   300,
    "trunk":   100,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_to_range(signal, lo, hi):
    """
    Per-frame: shift each value by the nearest integer multiple of 360° that
    places it inside [lo, hi]. Corrects Sports2D 360°-wrap artefacts.
    Identical to the function in plot_joint_angle.py and trust_report.py.
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


def confidence_label(score):
    if score >= 0.80:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


def mot_path_from_video(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.join("Sports2D_output", f"{stem}_Sports2D")
    mot = os.path.join(folder, f"{stem}_Sports2D_angles_person00.mot")
    return mot, folder


def read_mot(path):
    with open(path) as f:
        lines = f.readlines()
    start = next(i + 1 for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    header = lines[start].strip().split("\t")
    data = {}
    for line in lines[start + 1:]:
        parts = line.strip().split("\t")
        if len(parts) != len(header):
            continue
        for col, val in zip(header, parts):
            data.setdefault(col, []).append(float(val) if val.strip() else float('nan'))
    return {k: np.array(v) for k, v in data.items()}


# ── Signal selection ──────────────────────────────────────────────────────────

def select_rep_signal(data):
    """
    Choose the angle column that will give the clearest rep-counting signal.

    Scoring: valid_frame_fraction * in_range_variance + keyword_bonus
      - valid_frame_fraction rewards columns that weren't lost to tracking failures
      - in_range_variance rewards joints that actually moved (not static joints)
      - keyword_bonus prefers hip > knee > ankle (hip flexion is most reliable
        for both squats and jumping jacks)

    Returns (col_name, reason_string).
    """
    best_col, best_score, best_reason = None, -1.0, ""

    for col, raw in data.items():
        if col == "time":
            continue
        lo, hi = PLAUSIBLE_RANGES.get(col, (-360, 360))
        norm = normalize_to_range(raw, lo, hi)
        valid = norm[(norm >= lo) & (norm <= hi)]

        if len(valid) < 10:
            continue

        valid_frac = len(valid) / len(raw)
        in_range_var = float(np.var(valid))

        kw_bonus = 0
        for kw, bonus in KEYWORD_BONUS.items():
            if kw in col.lower():
                kw_bonus = bonus
                break

        score = valid_frac * in_range_var + kw_bonus
        reason = (f"valid_frac={valid_frac:.2f}, in_range_var={in_range_var:.1f}, "
                  f"kw_bonus={kw_bonus}  →  total={score:.1f}")

        if score > best_score:
            best_score = score
            best_col = col
            best_reason = reason

    return best_col, best_reason


# ── Rep counting ──────────────────────────────────────────────────────────────

def detect_peaks(signal, time_arr, fps, col_name):
    """
    Find exercise reps as peaks in the smoothed, normalised angle signal.

    Returns:
      peaks       — frame indices of detected peaks
      smoothed    — the smoothed signal array (same length as signal)
      valid_mask  — boolean mask of in-range (trusted) frames
      params      — dict describing the detection parameters used
    """
    lo, hi = PLAUSIBLE_RANGES.get(col_name, (-360, 360))
    norm = normalize_to_range(signal, lo, hi)
    valid_mask = (norm >= lo) & (norm <= hi)

    # Interpolate across gaps (find_peaks can't handle NaN)
    indices = np.arange(len(norm))
    valid_idx = indices[valid_mask]
    if len(valid_idx) < 10:
        return np.array([]), norm, valid_mask, {}

    interp = np.interp(indices, valid_idx, norm[valid_mask])

    # Savitzky–Golay smoothing: window ≈ 0.4 s (shorter than 0.8 s so that
    # small-amplitude peaks in fast exercises like jumping jacks are not
    # flattened away before peak detection runs).
    raw_win = max(int(fps * 0.4), 5)
    win = raw_win if raw_win % 2 == 1 else raw_win + 1   # must be odd
    win = min(win, len(interp) - 2)
    if win < 5:
        win = 5
    smoothed = savgol_filter(interp, win, 3)

    # Adaptive peak detection parameters.
    #
    # min_height = mean + 0.15 * (max - mean)
    #   Just above the mean so tiny micro-movements (e.g. 4-5° blips in
    #   jumping jacks between real reps) are filtered, while genuinely
    #   small-amplitude reps that a 0.30 formula would drop are kept.
    #
    # prominence = 0.25 * range
    #   Each peak must stand 25% of the total range above its neighbours.
    #   This removes "shoulder" artefacts (e.g. a 116 degree secondary
    #   local maximum during the ascent from a 130 degree squat).
    valid_smooth = smoothed[valid_mask]
    sig_range  = float(np.max(valid_smooth) - np.min(valid_smooth))
    sig_mean   = float(np.mean(valid_smooth))
    min_height = sig_mean + 0.15 * (float(np.max(valid_smooth)) - sig_mean)
    min_dist   = max(int(fps * 0.5), 5)   # >= 0.5 s between reps
    prominence = max(sig_range * 0.25, 5.0)

    peaks, _ = find_peaks(
        smoothed,
        height=min_height,
        distance=min_dist,
        prominence=prominence,
    )

    params = {
        "smoothing_window_frames": win,
        "min_peak_height_deg": round(min_height, 1),
        "min_peak_height_basis": "signal_mean + 0.15 * (signal_max - signal_mean)",
        "min_distance_frames": min_dist,
        "prominence_deg": round(prominence, 1),
        "signal_range_deg": round(sig_range, 1),
    }
    return peaks, smoothed, valid_mask, params


# ── Confidence ────────────────────────────────────────────────────────────────

def rep_count_confidence(peaks, time_arr, valid_mask):
    """
    Confidence in the rep count:  sqrt( regularity × valid_fraction )

    regularity = 1 – min(CV / 0.5, 1)
      Coefficient of variation of inter-rep intervals.
      CV = 0   → perfectly regular → regularity = 1
      CV = 0.5 → very irregular   → regularity = 0

    valid_fraction = fraction of all frames in anatomical range.
    """
    valid_frac = float(np.sum(valid_mask)) / len(valid_mask)

    if len(peaks) == 0:
        return 0.0, {
            "valid_fraction": round(valid_frac, 3),
            "regularity_score": 0.0,
            "note": "no reps detected — confidence is 0",
        }

    if len(peaks) == 1:
        regularity = 0.5   # can't measure regularity with one rep
        score = round(float(np.sqrt(regularity * valid_frac)), 3)
        return score, {
            "valid_fraction": round(valid_frac, 3),
            "regularity_score": regularity,
            "note": "only 1 rep detected — cannot measure inter-rep regularity",
        }

    intervals = np.diff([float(time_arr[p]) for p in peaks])
    cv = float(np.std(intervals)) / float(np.mean(intervals))
    regularity = max(0.0, 1.0 - cv / 0.5)
    score = round(float(np.sqrt(regularity * valid_frac)), 3)

    return score, {
        "valid_fraction": round(valid_frac, 3),
        "regularity_score": round(regularity, 3),
        "inter_rep_cv": round(cv, 3),
        "mean_interval_s": round(float(np.mean(intervals)), 3),
        "std_interval_s": round(float(np.std(intervals)), 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_reps.py <video.mp4>")
        sys.exit(1)

    video_arg = sys.argv[1]
    stem = os.path.splitext(os.path.basename(video_arg))[0]
    mot_path, out_folder = mot_path_from_video(video_arg)

    if not os.path.isfile(mot_path):
        print(f"\nERROR: angle file not found:\n  {mot_path}")
        print("Run Sports2D on this video first (python run_sports2d.py <video>).")
        sys.exit(1)

    print(f"\nReading angles from: {mot_path}")

    data     = read_mot(mot_path)
    time_arr = data.get("time", np.array([]))
    n_frames = len(time_arr)

    if n_frames < 2:
        print("ERROR: too few frames in .mot file.")
        sys.exit(1)

    fps        = round(1.0 / float(np.mean(np.diff(time_arr))), 2)
    duration_s = round(float(time_arr[-1]), 2)

    # ── Select best signal ────────────────────────────────────────────────────
    angle_data = {k: v for k, v in data.items() if k != "time"}
    best_col, selection_reason = select_rep_signal(angle_data)

    if best_col is None:
        print("ERROR: no usable angle signal found.")
        sys.exit(1)

    # ── Normalise and count reps ──────────────────────────────────────────────
    lo, hi    = PLAUSIBLE_RANGES.get(best_col, (-360, 360))
    raw       = data[best_col]
    norm      = normalize_to_range(raw, lo, hi)
    valid_mask = (norm >= lo) & (norm <= hi)
    valid_count = int(np.sum(valid_mask))
    valid_frac  = valid_count / n_frames

    peaks, smoothed, _, params = detect_peaks(raw, time_arr, fps, best_col)

    rep_times  = [round(float(time_arr[p]), 3) for p in peaks]
    rep_peaks  = [round(float(smoothed[p]), 1) for p in peaks]

    tempo_s    = None
    tempo_rpm  = None
    if len(rep_times) >= 2:
        intervals  = np.diff(rep_times)
        tempo_s    = round(float(np.mean(intervals)), 3)
        tempo_rpm  = round(60.0 / tempo_s, 1)

    conf_score, conf_details = rep_count_confidence(peaks, time_arr, valid_mask)
    conf_lbl   = confidence_label(conf_score)

    # ── Honest notes ──────────────────────────────────────────────────────────
    notes = []
    if valid_frac < 0.40:
        notes.append(
            f"Only {valid_frac:.0%} of frames had in-range angles — "
            "tracking quality was low. Rep count may be unreliable."
        )
    if len(peaks) == 0 and valid_frac >= 0.50:
        notes.append(
            "No clear rep peaks detected despite adequate signal quality. "
            "Possible reasons: movement amplitude too small for the threshold, "
            "exercise is not cyclic on this joint, or the clip contains only "
            "partial reps (start/end without completing full cycles)."
        )
    if len(peaks) > 0 and conf_score < 0.55:
        notes.append(
            f"Rep count ({len(peaks)}) detected but confidence is {conf_lbl} — "
            "inter-rep timing was irregular. Treat the count as approximate."
        )
    if len(peaks) == 1:
        notes.append(
            "Only 1 rep detected. Clip may be too short, or the exercise "
            "starts/ends mid-rep. Inter-rep regularity cannot be measured."
        )

    # ── Console output ────────────────────────────────────────────────────────
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  REP DETECTION REPORT")
    print(f"  Video     : {os.path.basename(video_arg)}")
    print(f"  Duration  : {duration_s} s  |  {n_frames} frames @ {fps} fps")
    print(f"{SEP}")
    print(f"  Signal used   : {best_col}")
    print(f"  Valid frames  : {valid_count}/{n_frames}  ({valid_frac:.0%})")
    print(f"  Plaus. range  : [{lo}°, {hi}°]")
    print(f"{SEP}")
    if len(peaks) == 0:
        print(f"  Rep count     : 0  — no reps detected")
    else:
        print(f"  Rep count     : {len(peaks)}")
        if tempo_s:
            print(f"  Tempo         : {tempo_s} s/rep  ({tempo_rpm} reps/min)")
        print(f"  Rep times (s) : {rep_times}")
        print(f"  Peak values   : {rep_peaks}°")
    print(f"  Confidence    : {conf_score:.0%}  {conf_lbl}")
    if notes:
        print()
        for n in notes:
            print(f"  ⚠  {n}")
    print(f"{SEP}\n")

    # ── JSON output ───────────────────────────────────────────────────────────
    report = {
        "generated_at":              datetime.datetime.now().isoformat(timespec="seconds"),
        "video_file":                os.path.basename(video_arg),
        "duration_seconds":          duration_s,
        "total_frames":              n_frames,
        "fps":                       fps,
        "signal_used":               best_col,
        "signal_selection_reason":   selection_reason,
        "plausible_range_degrees":   [lo, hi],
        "valid_frames":              valid_count,
        "valid_fraction":            round(valid_frac, 3),
        "rep_count":                 len(peaks),
        "rep_times_seconds":         rep_times,
        "per_rep_peak_degrees":      rep_peaks,
        "tempo_seconds_per_rep":     tempo_s,
        "tempo_reps_per_minute":     tempo_rpm,
        "rep_count_confidence":      conf_score,
        "rep_count_confidence_label": conf_lbl,
        "confidence_details":        conf_details,
        "confidence_basis":          (
            "sqrt( regularity × valid_fraction ).  "
            "Regularity = 1 – min(CV_of_inter_rep_intervals / 0.5, 1).  "
            "HIGH ≥ 80 %, MEDIUM 55–79 %, LOW < 55 %."
        ),
        "peak_detection_params":     params,
        "notes":                     notes,
    }

    out_path = os.path.join(out_folder, "rep_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Rep report saved: {out_path}")


if __name__ == "__main__":
    main()
