"""
mediapipe_pipeline.py
---------------------
Complete standalone MediaPipe-based sports analytics pipeline.

For a given video it:
  1. Quality gate  — blur + brightness check
  2. BlazePose     — 33 landmarks per frame via MediaPipe Tasks API
  3. Angles        — right/left knee, hip, ankle, thigh, trunk
                     matching Sports2D angular conventions
  4. Post-process  — interpolation → Hampel outlier rejection →
                     4th-order Butterworth 6 Hz low-pass →
                     360° wrap normalisation → plausible-range masking
  5. Trust report  — stability × plausibility confidence +
                     MediaPipe native per-keypoint visibility
  6. Rep detection — adaptive peak detection (same algorithm as detect_reps.py)
  7. Outputs       → mediapipe_output/<videoname>/

Output files
  <name>_mp_skeleton.mp4         skeleton overlay video
  <name>_mp_angles.csv           per-frame joint angles (9 columns + time)
  <name>_mp_angles_plot.png      multi-panel angle chart
  mediapipe_trust_report.json    per-angle confidence + visibility
  mediapipe_rep_report.json      rep count, tempo, confidence

Sports2D_output/ is NEVER read or written.

Usage:
  python mediapipe_pipeline.py squat_video.mp4
  python mediapipe_pipeline.py "demo (2).mp4"
"""

import sys
import os
import json
import math
import datetime
import urllib.request

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI window
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mediapipe as mp

# ── MediaPipe Tasks API ────────────────────────────────────────────────────────
_tv             = mp.tasks.vision
_BaseOptions    = mp.tasks.BaseOptions
LANDMARK_NAMES  = [lm.name for lm in _tv.PoseLandmark]   # 33 names

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_full.task")

OUTPUT_ROOT = "mediapipe_output"

# ── Quality-gate thresholds (same as check_video_quality.py) ──────────────────
BLUR_THRESHOLD       = 80.0
BRIGHTNESS_THRESHOLD = 40.0
SAMPLE_FRAMES        = 20

# ── Joint-angle plausible ranges (same as Sports2D / plot_joint_angle.py) ─────
# Joint angles: 0° = limb straight, positive = bent.
# Segment angles: anticlockwise from horizontal (y-down image coords).
PLAUSIBLE_RANGES = {
    "right knee":   (-20,  160),
    "left knee":    (-20,  160),
    "right hip":    (-60,  160),
    "left hip":     (-60,  160),
    "right ankle":  (-150,  30),   # shank segment from horizontal
    "left ankle":   (-150,  30),
    "right thigh":  (-150,  10),   # thigh segment from horizontal
    "left thigh":   (-150,  10),
    "trunk":        (  40, 150),   # trunk segment (~90° when upright)
}

ANGLE_COLS = [
    "right knee", "left knee",
    "right hip",  "left hip",
    "right ankle","left ankle",
    "right thigh","left thigh",
    "trunk",
]

# MediaPipe landmark indices that determine each angle.
# Used for per-angle visibility reporting.
ANGLE_LANDMARKS = {
    "right knee":   [24, 26, 28],    # RHip, RKnee, RAnkle
    "left knee":    [23, 25, 27],    # LHip, LKnee, LAnkle
    "right hip":    [26, 24, 12],    # RKnee, RHip, RShoulder
    "left hip":     [25, 23, 11],    # LKnee, LHip, LShoulder
    "right ankle":  [26, 28],        # RKnee → RAnkle (shank segment)
    "left ankle":   [25, 27],        # LKnee → LAnkle
    "right thigh":  [24, 26],        # RHip → RKnee
    "left thigh":   [23, 25],        # LHip → LKnee
    "trunk":        [23, 24, 11, 12],# mid-hip → mid-shoulder
}

# Keyword preference for column selection (rep counting)
KEYWORD_BONUS = {
    "hip":   5000,
    "knee":  2000,
    "ankle":  500,
    "thigh":  300,
    "trunk":  100,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model():
    if os.path.isfile(MODEL_FILE):
        return
    print("Downloading BlazePose model (~9 MB) …")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(MODEL_FILE, "wb") as f:
        f.write(r.read())
    print(f"  Saved: {MODEL_FILE}")


def quality_gate(video_path):
    """Blur + brightness check. Returns dict with grade and scores."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"grade": "ERROR", "blur_score": 0.0, "brightness_score": 0.0}
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
        return {"grade": "ERROR", "blur_score": 0.0, "brightness_score": 0.0}
    avg_blur   = round(float(np.mean(blur_scores)), 1)
    avg_bright = round(float(np.mean(bright_scores)), 1)
    passed = (avg_blur >= BLUR_THRESHOLD) and (avg_bright >= BRIGHTNESS_THRESHOLD)
    return {
        "grade":               "GOOD" if passed else "BAD",
        "blur_score":          avg_blur,
        "brightness_score":    avg_bright,
        "blur_threshold":      BLUR_THRESHOLD,
        "brightness_threshold": BRIGHTNESS_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Angle geometry
# ─────────────────────────────────────────────────────────────────────────────

def _joint_angle(ax, ay, bx, by, cx, cy):
    """
    Flexion angle at vertex B, where A-B-C are landmark positions.
    Returns 0° when limb is straight, increases toward 180° as joint bends.
    Uses 2D normalised image coordinates (x in [0,1], y in [0,1], y increases down).
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    mag = math.sqrt((v1x**2 + v1y**2) * (v2x**2 + v2y**2))
    if mag < 1e-9:
        return float("nan")
    cos_theta = max(-1.0, min(1.0, (v1x*v2x + v1y*v2y) / mag))
    # 180° - interior angle gives 0° for straight, positive for flexion.
    return 180.0 - math.degrees(math.acos(cos_theta))


def _segment_angle(px, py, dx, dy):
    """
    Segment angle from proximal (px,py) to distal (dx,dy), measured
    anticlockwise from horizontal.  Image coords (y increases downward),
    so the y component is negated to match standard screen-to-math convention.
    A vertical downward-pointing segment (thigh/shank during standing) → ~-90°.
    A vertical upward-pointing segment (trunk during standing) → ~+90°.
    """
    return math.degrees(math.atan2(-(dy - py), dx - px))


def compute_frame_angles(lm):
    """
    Given a list of 33 NormalizedLandmark objects (one frame), return a dict
    of angle name → float (degrees).  Returns all-NaN dict if lm is None.
    """
    if lm is None:
        return {col: float("nan") for col in ANGLE_COLS}

    x = [l.x for l in lm]
    y = [l.y for l in lm]

    mid_hip_x = (x[23] + x[24]) / 2
    mid_hip_y = (y[23] + y[24]) / 2
    mid_sho_x = (x[11] + x[12]) / 2
    mid_sho_y = (y[11] + y[12]) / 2

    return {
        # 3-point joint angles (0° = straight, positive = flexion)
        "right knee":  _joint_angle(x[24], y[24], x[26], y[26], x[28], y[28]),
        "left knee":   _joint_angle(x[23], y[23], x[25], y[25], x[27], y[27]),
        "right hip":   _joint_angle(x[26], y[26], x[24], y[24], x[12], y[12]),
        "left hip":    _joint_angle(x[25], y[25], x[23], y[23], x[11], y[11]),
        # Segment angles from horizontal (shank proxy for "ankle", thigh, trunk)
        "right ankle": _segment_angle(x[26], y[26], x[28], y[28]),  # knee → ankle
        "left ankle":  _segment_angle(x[25], y[25], x[27], y[27]),
        "right thigh": _segment_angle(x[24], y[24], x[26], y[26]),  # hip → knee
        "left thigh":  _segment_angle(x[23], y[23], x[25], y[25]),
        "trunk":       _segment_angle(mid_hip_x, mid_hip_y, mid_sho_x, mid_sho_y),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_nans(signal):
    """Linear interpolation across NaN gaps. Leaves leading/trailing NaNs."""
    arr = signal.copy().astype(float)
    idx = np.arange(len(arr))
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return arr
    arr[valid.sum() > 0] = np.interp(idx, idx[valid], arr[valid])
    return arr


def _hampel_filter(signal, window=5, sigma=3.0):
    """Replace outliers (> sigma × 1.4826 × MAD) with the local rolling median."""
    result = signal.copy().astype(float)
    for i in range(len(signal)):
        lo = max(0, i - window)
        hi = min(len(signal), i + window + 1)
        win_data = signal[lo:hi]
        valid = win_data[~np.isnan(win_data)]
        if len(valid) < 3:
            continue
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        if mad < 1e-6:
            continue
        if not np.isnan(signal[i]) and abs(signal[i] - med) > sigma * 1.4826 * mad:
            result[i] = med
    return result


def _butterworth_lp(signal, fps, cutoff=6.0, order=4):
    """4th-order zero-phase Butterworth low-pass filter."""
    nyq = fps / 2.0
    if cutoff >= nyq:
        return signal.copy()
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def normalize_to_range(signal, lo, hi):
    """
    Per-frame 360° wrap correction: shift each value by the nearest
    integer multiple of 360° that places it inside [lo, hi].
    """
    mid = (lo + hi) / 2.0
    result = signal.copy().astype(float)
    for i, v in enumerate(result):
        if np.isnan(v) or lo <= v <= hi:
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


def postprocess_signal(raw_signal, fps, col_name):
    """
    Full post-processing chain:
      1. Interpolate NaN gaps
      2. Hampel outlier rejection
      3. Butterworth 6 Hz low-pass
      4. 360° wrap normalisation
      5. Plausible-range masking (NaN for out-of-range frames)

    Returns (processed_signal, masked_signal) where masked_signal has NaN
    for frames outside the plausible range (tracking failures).
    """
    lo, hi = PLAUSIBLE_RANGES.get(col_name, (-360, 360))

    interp  = _interpolate_nans(raw_signal)
    hampel  = _hampel_filter(interp)
    smooth  = _butterworth_lp(hampel, fps)
    normed  = normalize_to_range(smooth, lo, hi)

    masked = normed.copy()
    out_of_range = (normed < lo) | (normed > hi)
    masked[out_of_range] = float("nan")

    return normed, masked


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trust metrics
# ─────────────────────────────────────────────────────────────────────────────

def stability_confidence(signal):
    """Frame-to-frame smoothness: 0–1. Low noise → close to 1."""
    NOISE_CAP = 0.30
    diffs = np.abs(np.diff(signal[~np.isnan(signal)]))
    if len(diffs) == 0:
        return 0.0
    frame_diff = float(np.mean(diffs))
    sig_range  = float(np.nanmax(signal) - np.nanmin(signal))
    raw_noise  = frame_diff / max(sig_range, 1.0)
    return round(1.0 - min(raw_noise / NOISE_CAP, 1.0), 3)


def plausibility_confidence(signal, col_name):
    """Fraction of frames within the plausible range after 360° normalisation."""
    lo, hi = PLAUSIBLE_RANGES.get(col_name, (-360, 360))
    normed  = normalize_to_range(signal, lo, hi)
    in_rng  = np.sum((normed >= lo) & (normed <= hi))
    return round(float(in_rng) / len(normed), 3)


def confidence_label(score):
    if score >= 0.80:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Rep detection  (algorithm from detect_reps.py — copied verbatim)
# ─────────────────────────────────────────────────────────────────────────────

def detect_peaks(signal, time_arr, fps, col_name):
    lo, hi = PLAUSIBLE_RANGES.get(col_name, (-360, 360))
    norm = normalize_to_range(signal, lo, hi)
    valid_mask = (norm >= lo) & (norm <= hi)

    indices   = np.arange(len(norm))
    valid_idx = indices[valid_mask]
    if len(valid_idx) < 10:
        return np.array([]), norm, valid_mask, {}

    interp = np.interp(indices, valid_idx, norm[valid_mask])

    raw_win = max(int(fps * 0.4), 5)
    win = raw_win if raw_win % 2 == 1 else raw_win + 1
    win = min(win, len(interp) - 2)
    if win < 5:
        win = 5
    smoothed = savgol_filter(interp, win, 3)

    valid_smooth = smoothed[valid_mask]
    sig_range  = float(np.max(valid_smooth) - np.min(valid_smooth))
    sig_mean   = float(np.mean(valid_smooth))
    min_height = sig_mean + 0.15 * (float(np.max(valid_smooth)) - sig_mean)
    min_dist   = max(int(fps * 0.5), 5)
    prominence = max(sig_range * 0.25, 5.0)

    peaks, _ = find_peaks(smoothed, height=min_height,
                          distance=min_dist, prominence=prominence)
    params = {
        "smoothing_window_frames": win,
        "min_peak_height_deg":     round(min_height, 1),
        "min_peak_height_basis":   "signal_mean + 0.15 * (signal_max - signal_mean)",
        "min_distance_frames":     min_dist,
        "prominence_deg":          round(prominence, 1),
        "signal_range_deg":        round(sig_range, 1),
    }
    return peaks, smoothed, valid_mask, params


def rep_count_confidence(peaks, time_arr, valid_mask):
    valid_frac = float(np.sum(valid_mask)) / len(valid_mask)
    if len(peaks) == 0:
        return 0.0, {"valid_fraction": round(valid_frac, 3),
                     "regularity_score": 0.0,
                     "note": "no reps detected — confidence is 0"}
    if len(peaks) == 1:
        regularity = 0.5
        score = round(float(np.sqrt(regularity * valid_frac)), 3)
        return score, {"valid_fraction": round(valid_frac, 3),
                       "regularity_score": regularity,
                       "note": "only 1 rep detected — cannot measure inter-rep regularity"}
    intervals  = np.diff([float(time_arr[p]) for p in peaks])
    cv         = float(np.std(intervals)) / float(np.mean(intervals))
    regularity = max(0.0, 1.0 - cv / 0.5)
    score      = round(float(np.sqrt(regularity * valid_frac)), 3)
    return score, {"valid_fraction":   round(valid_frac, 3),
                   "regularity_score": round(regularity, 3),
                   "inter_rep_cv":     round(cv, 3),
                   "mean_interval_s":  round(float(np.mean(intervals)), 3),
                   "std_interval_s":   round(float(np.std(intervals)), 3)}


def select_rep_signal(angle_data):
    """Pick the angle column that gives the clearest rep-counting signal."""
    best_col, best_score, best_reason = None, -1.0, ""
    for col, raw in angle_data.items():
        lo, hi = PLAUSIBLE_RANGES.get(col, (-360, 360))
        norm   = normalize_to_range(raw, lo, hi)
        valid  = norm[(norm >= lo) & (norm <= hi) & ~np.isnan(norm)]
        if len(valid) < 10:
            continue
        valid_frac    = len(valid) / len(raw)
        in_range_var  = float(np.var(valid))
        kw_bonus = 0
        for kw, bonus in KEYWORD_BONUS.items():
            if kw in col.lower():
                kw_bonus = bonus
                break
        score = valid_frac * in_range_var + kw_bonus
        if score > best_score:
            best_score  = score
            best_col    = col
            best_reason = (f"valid_frac={valid_frac:.2f}, "
                           f"in_range_var={in_range_var:.1f}, "
                           f"kw_bonus={kw_bonus}  →  total={score:.1f}")
    return best_col, best_reason


# ─────────────────────────────────────────────────────────────────────────────
# 6. BlazePose extraction + skeleton overlay
# ─────────────────────────────────────────────────────────────────────────────

def run_blazepose(video_path, out_skeleton_path, out_folder):
    """
    Run BlazePose on every frame.  Saves skeleton overlay video.

    Returns:
      all_landmarks  — list[list|None], one entry per frame.
                       Each entry is either None (no detection) or
                       a list of 33 NormalizedLandmark objects.
      visibility_arr — np.ndarray shape (N_frames, 33)  float32
                       visibility scores; NaN where no detection.
      fps            — float
      n_frames       — int
    """
    ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_skeleton_path, fourcc, fps, (width, height))

    options = _tv.PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=_tv.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    du = _tv.drawing_utils
    ds = _tv.drawing_styles
    conn = _tv.PoseLandmarksConnections.POSE_LANDMARKS

    all_landmarks = []
    vis_list      = []

    print(f"  Running BlazePose on {n_frames} frames …")
    with _tv.PoseLandmarker.create_from_options(options) as lmkr:
        fi = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int(fi * (1000.0 / fps))
            result   = lmkr.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                all_landmarks.append(lm)
                vis_list.append([l.visibility for l in lm])
                annotated = frame_bgr.copy()
                du.draw_landmarks(annotated, lm, conn,
                    landmark_drawing_spec=ds.get_default_pose_landmarks_style())
            else:
                all_landmarks.append(None)
                vis_list.append([float("nan")] * 33)
                annotated = frame_bgr.copy()

            writer.write(annotated)
            fi += 1
            if fi % 50 == 0:
                print(f"    … {fi}/{n_frames}")

    cap.release()
    writer.release()

    visibility_arr = np.array(vis_list, dtype=float)  # shape (N, 33)
    return all_landmarks, visibility_arr, fps, n_frames


# ─────────────────────────────────────────────────────────────────────────────
# 7. Angle computation for all frames
# ─────────────────────────────────────────────────────────────────────────────

def extract_angles(all_landmarks, fps):
    """
    Compute raw angles for all frames, then post-process each signal.

    Returns:
      time_arr       — np.ndarray  (N,)
      raw_angles     — dict col → np.ndarray (N,)   raw before post-process
      proc_angles    — dict col → np.ndarray (N,)   smoothed, normalised
      masked_angles  — dict col → np.ndarray (N,)   NaN for out-of-range
    """
    n = len(all_landmarks)
    time_arr = np.arange(n) / fps

    raw_angles = {col: np.full(n, float("nan")) for col in ANGLE_COLS}

    for fi, lm in enumerate(all_landmarks):
        frame_angs = compute_frame_angles(lm)
        for col in ANGLE_COLS:
            raw_angles[col][fi] = frame_angs[col]

    proc_angles   = {}
    masked_angles = {}
    for col in ANGLE_COLS:
        proc, masked = postprocess_signal(raw_angles[col], fps, col)
        proc_angles[col]   = proc
        masked_angles[col] = masked

    return time_arr, raw_angles, proc_angles, masked_angles


# ─────────────────────────────────────────────────────────────────────────────
# 8. Output writers
# ─────────────────────────────────────────────────────────────────────────────

def save_angles_csv(out_folder, stem, time_arr, masked_angles):
    """Write per-frame angles to CSV."""
    path = os.path.join(out_folder, f"{stem}_mp_angles.csv")
    header = "time," + ",".join(ANGLE_COLS)
    rows = [header]
    for fi, t in enumerate(time_arr):
        vals = [f"{t:.4f}"]
        for col in ANGLE_COLS:
            v = masked_angles[col][fi]
            vals.append("" if np.isnan(v) else f"{v:.4f}")
        rows.append(",".join(vals))
    with open(path, "w") as f:
        f.write("\n".join(rows))
    return path


def save_angle_plot(out_folder, stem, time_arr, masked_angles):
    """3×3 subplot figure with all 9 angles."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(ANGLE_COLS):
        ax = axes[i]
        y  = masked_angles[col]
        ax.plot(time_arr, y, linewidth=1.5, color="#1f77b4", label=col.title())
        ax.fill_between(time_arr, y, alpha=0.10, color="#1f77b4")
        lo, hi = PLAUSIBLE_RANGES.get(col, (None, None))
        if lo is not None:
            ax.set_ylim(lo - 5, hi + 5)
        ax.set_title(col.title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Degrees", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    fig.suptitle(f"MediaPipe Joint Angles — {stem}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_folder, f"{stem}_mp_angles_plot.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_trust_report(video_path, stem, qc, proc_angles, visibility_arr):
    """Build the trust report dict."""
    n_frames = len(next(iter(proc_angles.values())))
    measurements = []
    for col in ANGLE_COLS:
        sig  = proc_angles[col]
        lo, hi = PLAUSIBLE_RANGES[col]

        valid_mask = (sig >= lo) & (sig <= hi) & ~np.isnan(sig)
        in_range   = int(np.sum(valid_mask))

        stab  = stability_confidence(sig)
        plaus = round(in_range / n_frames, 3)
        conf  = round(float(np.sqrt(stab * plaus)), 3)

        # MediaPipe native visibility — min over contributing landmarks per frame
        lm_indices = ANGLE_LANDMARKS[col]
        lm_vis_all = visibility_arr[:, lm_indices]   # shape (N, k)
        lm_vis_min = np.nanmin(lm_vis_all, axis=1)   # weakest link per frame
        mean_vis   = round(float(np.nanmean(lm_vis_min)), 4)

        valid_sig  = sig[valid_mask]
        sig_min    = round(float(np.nanmin(sig)), 1)
        sig_max    = round(float(np.nanmax(sig)), 1)
        value      = round(float(np.median(valid_sig)), 1) if in_range > 0 else float("nan")

        measurements.append({
            "metric":                   col.replace(" ", "_"),
            "display_name":             col.title(),
            "value_degrees":            value,
            "signal_range_degrees":     [sig_min, sig_max],
            "plausible_range_degrees":  [lo, hi],
            "frames_analyzed":          n_frames,
            "frames_in_range":          in_range,
            "stability_score":          stab,
            "plausibility_score":       plaus,
            "confidence_score":         conf,
            "confidence_label":         confidence_label(conf),
            "confidence_basis":         "sqrt(stability × plausibility)",
            "mediapipe_landmark_indices": lm_indices,
            "mediapipe_mean_visibility":  mean_vis,
        })

    return {
        "report_metadata": {
            "generated_at":  datetime.datetime.now().isoformat(timespec="seconds"),
            "video_file":    os.path.basename(video_path),
            "pipeline":      "mediapipe_pipeline.py",
            "pose_model":    "BlazePose Full (pose_landmarker_full.task)",
            "video_quality": qc,
        },
        "measurements": measurements,
    }


def save_trust_report(out_folder, trust_report):
    path = os.path.join(out_folder, "mediapipe_trust_report.json")
    with open(path, "w") as f:
        json.dump(trust_report, f, indent=2)
    return path


def build_rep_report(video_path, time_arr, proc_angles, fps):
    """Run rep detection and build the rep report dict."""
    n_frames   = len(time_arr)
    duration_s = round(float(time_arr[-1]), 2) if len(time_arr) > 0 else 0.0

    # Select best signal
    best_col, selection_reason = select_rep_signal(proc_angles)
    if best_col is None:
        return {
            "generated_at":   datetime.datetime.now().isoformat(timespec="seconds"),
            "video_file":     os.path.basename(video_path),
            "rep_count":      0,
            "error":          "no usable angle signal found",
        }

    lo, hi = PLAUSIBLE_RANGES.get(best_col, (-360, 360))
    raw    = proc_angles[best_col]
    norm   = normalize_to_range(raw, lo, hi)
    valid_mask  = (norm >= lo) & (norm <= hi) & ~np.isnan(norm)
    valid_count = int(np.sum(valid_mask))
    valid_frac  = valid_count / n_frames

    peaks, smoothed, _, params = detect_peaks(raw, time_arr, fps, best_col)

    rep_times = [round(float(time_arr[p]), 3) for p in peaks]
    rep_peaks = [round(float(smoothed[p]), 1) for p in peaks]

    tempo_s = tempo_rpm = None
    if len(rep_times) >= 2:
        intervals  = np.diff(rep_times)
        tempo_s    = round(float(np.mean(intervals)), 3)
        tempo_rpm  = round(60.0 / tempo_s, 1)

    conf_score, conf_details = rep_count_confidence(peaks, time_arr, valid_mask)
    conf_lbl = confidence_label(conf_score)

    notes = []
    if valid_frac < 0.40:
        notes.append(f"Only {valid_frac:.0%} of frames had in-range angles.")
    if len(peaks) == 0 and valid_frac >= 0.50:
        notes.append("No clear rep peaks detected despite adequate signal quality.")
    if len(peaks) > 0 and conf_score < 0.55:
        notes.append(f"Rep count ({len(peaks)}) detected but confidence is {conf_lbl}.")

    return {
        "generated_at":               datetime.datetime.now().isoformat(timespec="seconds"),
        "video_file":                 os.path.basename(video_path),
        "duration_seconds":           duration_s,
        "total_frames":               n_frames,
        "fps":                        round(fps, 2),
        "signal_used":                best_col,
        "signal_selection_reason":    selection_reason,
        "plausible_range_degrees":    [lo, hi],
        "valid_frames":               valid_count,
        "valid_fraction":             round(valid_frac, 3),
        "rep_count":                  len(peaks),
        "rep_times_seconds":          rep_times,
        "per_rep_peak_degrees":       rep_peaks,
        "tempo_seconds_per_rep":      tempo_s,
        "tempo_reps_per_minute":      tempo_rpm,
        "rep_count_confidence":       conf_score,
        "rep_count_confidence_label": conf_lbl,
        "confidence_details":         conf_details,
        "confidence_basis":           (
            "sqrt( regularity × valid_fraction ). "
            "Regularity = 1 – min(CV_of_inter_rep_intervals / 0.5, 1). "
            "HIGH ≥ 80%, MEDIUM 55–79%, LOW < 55%."
        ),
        "peak_detection_params":      params,
        "notes":                      notes,
    }


def save_rep_report(out_folder, rep_report):
    path = os.path.join(out_folder, "mediapipe_rep_report.json")
    with open(path, "w") as f:
        json.dump(rep_report, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 9. Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_trust_summary(trust_report):
    meta = trust_report["report_metadata"]
    qc   = meta["video_quality"]
    SEP  = "=" * 68
    print(f"\n{SEP}")
    print("  MEDIAPIPE TRUST REPORT")
    print(f"  Generated : {meta['generated_at']}")
    print(f"  Video     : {meta['video_file']}")
    print(f"  Model     : {meta['pose_model']}")
    print(f"{SEP}")
    print(f"  VIDEO QUALITY GATE")
    print(f"    Grade      : {qc['grade']}")
    print(f"    Sharpness  : {qc['blur_score']}  (threshold > {qc['blur_threshold']})")
    print(f"    Brightness : {qc['brightness_score']}  (threshold > {qc['brightness_threshold']})")
    print(f"\n  MEASUREMENTS  ({len(trust_report['measurements'])} angles)")
    HDR = f"  {'Metric':<20}  {'Value':>8}  {'Stability':>10}  {'Plausibility':>13}  {'Confidence':>12}  {'MP Vis':>8}"
    print(HDR)
    print("  " + "-" * 80)
    for m in trust_report["measurements"]:
        val_s   = f"{m['value_degrees']:.1f}°" if not (
            isinstance(m['value_degrees'], float) and math.isnan(m['value_degrees'])) else "  n/a"
        stab_s  = f"{m['stability_score']:.0%}"
        plaus_s = f"{m['plausibility_score']:.0%}  ({m['frames_in_range']}/{m['frames_analyzed']}fr)"
        conf_s  = f"{m['confidence_score']:.0%}  {m['confidence_label']}"
        vis_s   = f"{m['mediapipe_mean_visibility']:.3f}"
        print(f"  {m['display_name']:<20}  {val_s:>8}  {stab_s:>10}  {plaus_s:>13}  {conf_s:>12}  {vis_s:>8}")
    print()


def print_rep_summary(rep_report):
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  REP DETECTION REPORT  (MediaPipe)")
    print(f"  Video    : {rep_report['video_file']}")
    d = rep_report.get('duration_seconds', '?')
    n = rep_report.get('total_frames', '?')
    f = rep_report.get('fps', '?')
    print(f"  Duration : {d} s  |  {n} frames @ {f} fps")
    print(f"{SEP}")
    if "error" in rep_report:
        print(f"  ERROR: {rep_report['error']}")
    else:
        print(f"  Signal used   : {rep_report['signal_used']}")
        vf = rep_report['valid_frames']
        tf = rep_report['total_frames']
        vfrac = rep_report['valid_fraction']
        lo, hi = rep_report['plausible_range_degrees']
        print(f"  Valid frames  : {vf}/{tf}  ({vfrac:.0%})")
        print(f"  Plaus. range  : [{lo}°, {hi}°]")
        print(f"{SEP}")
        rc = rep_report['rep_count']
        if rc == 0:
            print("  Rep count     : 0  — no reps detected")
        else:
            print(f"  Rep count     : {rc}")
            ts = rep_report['tempo_seconds_per_rep']
            tr = rep_report['tempo_reps_per_minute']
            if ts:
                print(f"  Tempo         : {ts} s/rep  ({tr} reps/min)")
            print(f"  Rep times (s) : {rep_report['rep_times_seconds']}")
            print(f"  Peak values   : {rep_report['per_rep_peak_degrees']}°")
        cs = rep_report['rep_count_confidence']
        cl = rep_report['rep_count_confidence_label']
        print(f"  Confidence    : {cs:.0%}  {cl}")
        for note in rep_report.get("notes", []):
            print(f"  ⚠  {note}")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path):
    if not os.path.isfile(video_path):
        print(f"ERROR: file not found — {video_path}")
        sys.exit(1)

    stem       = os.path.splitext(os.path.basename(video_path))[0]
    out_folder = os.path.join(OUTPUT_ROOT, stem)
    os.makedirs(out_folder, exist_ok=True)

    print(f"\n{'='*58}")
    print(f"  MediaPipe Pipeline — {stem}")
    print(f"  Output folder: {out_folder}")
    print(f"{'='*58}\n")

    # Stage 1 — Quality gate
    print("Stage 1  Quality gate …")
    qc = quality_gate(video_path)
    print(f"  Grade: {qc['grade']}  (blur={qc['blur_score']}, "
          f"brightness={qc['brightness_score']})")
    if qc["grade"] == "BAD":
        print("  WARNING: video failed quality gate — results may be unreliable.")

    # Stage 2 — BlazePose
    skeleton_path = os.path.join(out_folder, f"{stem}_mp_skeleton.mp4")
    print(f"\nStage 2  BlazePose extraction …")
    all_lm, vis_arr, fps, n_frames = run_blazepose(video_path, skeleton_path, out_folder)
    detected = sum(1 for l in all_lm if l is not None)
    print(f"  Frames: {n_frames}  |  Detected: {detected}  |  FPS: {fps:.1f}")
    print(f"  Skeleton video: {skeleton_path}")

    # Stage 3 — Angle extraction + post-processing
    print(f"\nStage 3  Computing and post-processing angles …")
    time_arr, raw_angles, proc_angles, masked_angles = extract_angles(all_lm, fps)
    for col in ANGLE_COLS:
        valid = np.sum(~np.isnan(masked_angles[col]))
        print(f"  {col:<20}  valid frames: {valid}/{n_frames}")

    # Stage 4 — Save angles CSV
    csv_path = save_angles_csv(out_folder, stem, time_arr, masked_angles)
    print(f"\n  Angles CSV: {csv_path}")

    # Stage 5 — Save angle plot
    plot_path = save_angle_plot(out_folder, stem, time_arr, masked_angles)
    print(f"  Angle plot: {plot_path}")

    # Stage 6 — Trust report
    print(f"\nStage 4  Trust report …")
    trust = build_trust_report(video_path, stem, qc, proc_angles, vis_arr)
    trust_path = save_trust_report(out_folder, trust)
    print_trust_summary(trust)
    print(f"  Trust report saved: {trust_path}")

    # Stage 7 — Rep detection
    print(f"Stage 5  Rep detection …")
    rep_report  = build_rep_report(video_path, time_arr, proc_angles, fps)
    rep_path    = save_rep_report(out_folder, rep_report)
    print_rep_summary(rep_report)
    print(f"  Rep report saved: {rep_path}")

    print(f"\nAll outputs in: {os.path.abspath(out_folder)}")
    print(f"{'='*58}\n")

    return out_folder


def main():
    if len(sys.argv) < 2:
        print("Usage: python mediapipe_pipeline.py <video.mp4>")
        sys.exit(1)
    run_pipeline(sys.argv[1])


if __name__ == "__main__":
    main()
