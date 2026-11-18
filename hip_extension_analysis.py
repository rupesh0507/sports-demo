"""
hip_extension_analysis.py
--------------------------
SME Brief Pattern A1 — Hip Extension at Toe-Off (Sprint Mechanics)

Standalone script. Does NOT modify any existing pipeline files.
Input  : run_sideview.mp4  (1920×1080 @ 24fps, 280 frames)
Output : hip_extension_output/
    hip_extension_results.json      — per-toe-off summary
    hip_extension_plot.png          — hip angle over time with toe-off markers

Angle convention (INCLUDED ANGLE, matches SME brief D2/A1 thresholds):
    angle_at(knee, hip, shoulder) = arccos(dot / magnitudes)
    180° = perfectly straight (thigh inline with trunk)
    Decreases as joint bends OR as thigh moves away from trunk
    Labels:  FULL_EXTENSION ≥ 160°
             PARTIAL        145–160°
             RESTRICTED     < 145°
             UNDETECTABLE   landmark visibility < 0.5 or nan

MediaPipe landmarks (BlazePose 33-point):
    Hips: 23 (L), 24 (R)   Knees: 25 (L), 26 (R)   Ankles: 27 (L), 28 (R)
    Shoulders: 11 (L), 12 (R)

Hip angle for rear leg: angle at HIP between KNEE and same-side SHOULDER.
Rear leg = the ankle that is behind the body (smaller x in L→R running).
Toe-off detection: local maximum in the rear ankle's vertical (y) time series.
"""

import cv2, numpy as np, json, os, sys, urllib.request, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

import mediapipe as mp

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "run_sideview.mp4"
OUT_DIR      = "hip_extension_output"
FPS_NATIVE   = 24.0          # measured fps of this clip
FPS_PREF     = 30.0          # SME D3 preferred minimum fps for pattern A1

# SME Brief A1 thresholds (INCLUDED ANGLE: arccos convention, 180° = straight)
THR_FULL     = 160.0         # ≥ 160° → FULL_EXTENSION
THR_PARTIAL  = 145.0         # 145–160° → PARTIAL; < 145° → RESTRICTED

VIS_MIN      = 0.5           # minimum landmark visibility to trust an angle
FPS_PENALTY  = 0.10          # subtract 10% confidence for sub-30fps clip

# Minimum x-separation for "rear" ankle classification (normalised coords).
# Prevents mis-labelling when both feet are crossing mid-air.
MIN_X_GAP    = 0.03

# Phase guard: trunk lean from vertical above this threshold → acceleration phase.
# During acceleration a sprinter leans >20° forward; at max speed typically <15°.
ACCEL_TRUNK_LEAN_DEG  = 20.0

# Plausibility guard (SME D3 rule): any toe-off reading more than this many degrees
# below the full-speed median is treated as a bad detection, not a real RESTRICTED label.
PLAUSIBILITY_DROP_DEG = 30.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── MediaPipe setup ────────────────────────────────────────────────────────────
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode           = mp.tasks.vision.RunningMode
BaseOptions           = mp.tasks.BaseOptions

# Prefer the model file already in the project folder (same as mediapipe_pipeline.py),
# fall back to the ~/.cache path used by mediapipe_scan.py.
_local_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pose_landmarker_full.task")
_cache_model = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe",
                            "pose_landmarker_full.task")
if os.path.isfile(_local_model):
    MODEL_PATH = _local_model
elif os.path.isfile(_cache_model):
    MODEL_PATH = _cache_model
else:
    MODEL_PATH = _local_model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading MediaPipe model (~9 MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        MODEL_PATH,
    )
    print("  Downloaded.")

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def included_angle(ax, ay, bx, by, cx, cy):
    """
    INCLUDED angle in degrees at vertex B formed by A-B-C.
    Uses arccos directly: 180° = straight, decreases as the joint bends.
    Distinct from the existing pipeline's _joint_angle (which returns 180 - this).
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    mag = math.sqrt((v1x**2 + v1y**2) * (v2x**2 + v2y**2))
    if mag < 1e-9:
        return float("nan")
    cos_t = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / mag))
    return math.degrees(math.acos(cos_t))


def lm_xy(lms, idx):
    if idx >= len(lms):
        return float("nan"), float("nan")
    return lms[idx].x, lms[idx].y


def lm_vis(lms, idx):
    if idx >= len(lms):
        return 0.0
    return lms[idx].visibility


def hip_incl(lms, side):
    """
    Included angle at HIP between KNEE (a) → HIP (b) → SHOULDER (c).
    Returns (angle_deg, min_visibility_of_3_landmarks).
    """
    if side == "left":
        kx, ky = lm_xy(lms, 25)   # L knee
        hx, hy = lm_xy(lms, 23)   # L hip
        sx, sy = lm_xy(lms, 11)   # L shoulder
        vis = min(lm_vis(lms, 25), lm_vis(lms, 23), lm_vis(lms, 11))
    else:
        kx, ky = lm_xy(lms, 26)   # R knee
        hx, hy = lm_xy(lms, 24)   # R hip
        sx, sy = lm_xy(lms, 12)   # R shoulder
        vis = min(lm_vis(lms, 26), lm_vis(lms, 24), lm_vis(lms, 12))

    if any(math.isnan(v) for v in (kx, ky, hx, hy, sx, sy)):
        return float("nan"), vis
    return included_angle(kx, ky, hx, hy, sx, sy), vis


def smooth_valid(arr, win=5, poly=2):
    """Savitzky-Golay, filling NaN with linear interpolation before smoothing."""
    arr = np.asarray(arr, dtype=float)
    valid = ~np.isnan(arr)
    if np.sum(valid) < win + 1:
        return arr.copy()
    x = np.arange(len(arr))
    filled = np.interp(x, x[valid], arr[valid])
    smoothed = savgol_filter(filled, window_length=win, polyorder=poly)
    result = arr.copy()
    result[valid] = smoothed[valid]
    return result


def trunk_lean_deg(lms):
    """
    Forward/backward lean of the trunk from vertical (degrees, 0° = upright).
    Computed from the mid-hip → mid-shoulder segment in normalised image coords
    (y increases downward).  Used to identify the acceleration phase.
    Returns nan if any required landmark is missing.
    """
    sho_x = (lms[11].x + lms[12].x) / 2
    sho_y = (lms[11].y + lms[12].y) / 2
    hip_x = (lms[23].x + lms[24].x) / 2
    hip_y = (lms[23].y + lms[24].y) / 2
    dx = sho_x - hip_x
    dy = sho_y - hip_y
    tlen = math.sqrt(dx * dx + dy * dy)
    if tlen < 1e-9:
        return float("nan")
    # dot of trunk vector with vertical-up = (0, -1)  →  -dy / tlen
    cos_lean = max(-1.0, min(1.0, -dy / tlen))
    return math.degrees(math.acos(cos_lean))


# ── Run pose estimation ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit(f"ERROR: Cannot open {VIDEO_PATH}")

fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_NATIVE
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nPattern A1: Hip Extension at Toe-Off")
print(f"  Video : {VIDEO_PATH}  ({W}x{H} @ {fps:.1f}fps  {fc} frames  {fc/fps:.1f}s)")
print(f"  Model : {os.path.basename(MODEL_PATH)}")
print(f"  Running pose estimation...")

frames_data = []   # list of per-frame dicts

with PoseLandmarker.create_from_options(options) as lm_det:
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms  = int((fi / fps) * 1000)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = lm_det.detect_for_video(mp_img, ts_ms)

        row = dict(
            frame=fi, time=fi / fps, detected=False,
            L_ax=float("nan"), L_ay=float("nan"),
            R_ax=float("nan"), R_ay=float("nan"),
            L_hip_angle=float("nan"), L_hip_vis=0.0,
            R_hip_angle=float("nan"), R_hip_vis=0.0,
            trunk_lean=float("nan"),
        )

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            row["detected"] = True
            row["L_ax"], row["L_ay"] = lm_xy(lms, 27)   # L ankle
            row["R_ax"], row["R_ay"] = lm_xy(lms, 28)   # R ankle
            row["L_hip_angle"], row["L_hip_vis"] = hip_incl(lms, "left")
            row["R_hip_angle"], row["R_hip_vis"] = hip_incl(lms, "right")
            row["trunk_lean"] = trunk_lean_deg(lms)

        frames_data.append(row)
        fi += 1

cap.release()
n_frames   = len(frames_data)
n_detected = sum(1 for f in frames_data if f["detected"])
print(f"  Processed {n_frames} frames — pose detected in {n_detected} ({100*n_detected/n_frames:.1f}%)")

# ── Infer running direction (left→right or right→left) ─────────────────────────
det = [f for f in frames_data if f["detected"]]
mid_x = [0.5 * (f["L_ax"] + f["R_ax"]) for f in det if not math.isnan(f["L_ax"])]
if len(mid_x) < 10:
    sys.exit("ERROR: Too few detected frames to determine running direction.")
x_early = float(np.nanmean(mid_x[: len(mid_x) // 3]))
x_late  = float(np.nanmean(mid_x[-len(mid_x) // 3 :]))
running_lr = x_late > x_early   # True = left → right
print(f"  Running direction: {'Left → Right' if running_lr else 'Right → Left'}")

# ── Build time series arrays ───────────────────────────────────────────────────
times      = np.array([f["time"]        for f in frames_data])
L_ax       = np.array([f["L_ax"]        for f in frames_data])
L_ay       = np.array([f["L_ay"]        for f in frames_data])
R_ax       = np.array([f["R_ax"]        for f in frames_data])
R_ay       = np.array([f["R_ay"]        for f in frames_data])
L_hip_ang  = np.array([f["L_hip_angle"] for f in frames_data])
R_hip_ang  = np.array([f["R_hip_angle"] for f in frames_data])
L_hip_vis  = np.array([f["L_hip_vis"]   for f in frames_data])
R_hip_vis  = np.array([f["R_hip_vis"]   for f in frames_data])
trunk_lean    = np.array([f["trunk_lean"] for f in frames_data])

# Smooth ankle y for stable peak detection; smooth trunk lean for phase detection
L_ay_sm       = smooth_valid(L_ay)
R_ay_sm       = smooth_valid(R_ay)
trunk_lean_sm = smooth_valid(trunk_lean, win=9, poly=2)   # wider window for phase boundary

# ── Toe-off detection ──────────────────────────────────────────────────────────
# In image coords y increases downward: ground contact → high y, foot in air → low y.
# Toe-off = local MAXIMUM in the ankle y time series (foot at its lowest point before lift).
# We then check: is this ankle the REAR ankle at that frame?
# Rear ankle = smaller x in L→R running (or larger x in R→L).

def detect_toeoffs_for_ankle(ank_y_sm, ank_x, other_x, label, running_lr):
    """
    Return list of (frame_index, side_label) for toe-off events of one ankle.
    min_dist_frames: at 24fps, ~2Hz stride → 12 frames/stride. Use 8 as minimum.
    """
    valid = ~np.isnan(ank_y_sm)
    if np.sum(valid) < 12:
        return []

    y_min = float(np.nanmin(ank_y_sm))
    y_max = float(np.nanmax(ank_y_sm))
    y_range = y_max - y_min
    if y_range < 0.005:
        return []

    # Height threshold: peak must be in the top 65% of the observed y range
    # (filters out peaks during mid-swing when the foot is high)
    y_thresh = y_min + 0.65 * y_range

    peaks, _ = find_peaks(
        np.where(valid, ank_y_sm, y_min - 1),
        height=y_thresh,
        distance=8,         # ≥ 8 frames between peaks (≥ 0.33s at 24fps)
        prominence=0.015,   # must be a true stance peak, not tracking jitter
    )

    toe_offs = []
    for p in peaks:
        ax = ank_x[p]
        ox = other_x[p]
        if math.isnan(ax) or math.isnan(ox):
            continue
        x_diff = ax - ox                          # negative if ax is behind in L→R
        if running_lr:
            is_rear = x_diff < -MIN_X_GAP         # this ankle is to the LEFT
        else:
            is_rear = x_diff > MIN_X_GAP          # this ankle is to the RIGHT
        if is_rear:
            toe_offs.append(p)
    return toe_offs

L_toeoffs = detect_toeoffs_for_ankle(L_ay_sm, L_ax, R_ax, "left",  running_lr)
R_toeoffs = detect_toeoffs_for_ankle(R_ay_sm, R_ax, L_ax, "right", running_lr)

all_toeoffs = sorted([(int(fi), "left")  for fi in L_toeoffs] +
                     [(int(fi), "right") for fi in R_toeoffs],
                     key=lambda t: t[0])

print(f"  Toe-offs detected: {len(all_toeoffs)}  "
      f"(left={len(L_toeoffs)}, right={len(R_toeoffs)})")

# ── Label each toe-off ─────────────────────────────────────────────────────────

def confidence(vis, angle, fps_clip):
    """
    Landmark-visibility-based confidence, reduced by 10% for sub-30fps clips.
    Returns 0.0–1.0.
    """
    if math.isnan(angle) or vis < VIS_MIN:
        return 0.0
    base = float(vis)
    if fps_clip < FPS_PREF:
        base = max(0.0, base - FPS_PENALTY)
    return round(base, 3)


def label_angle(angle, vis):
    if math.isnan(angle) or vis < VIS_MIN:
        return "UNDETECTABLE"
    if angle >= THR_FULL:
        return "FULL_EXTENSION"
    if angle >= THR_PARTIAL:
        return "PARTIAL"
    return "RESTRICTED"


toe_off_results = []
for fi, side in all_toeoffs:
    t   = frames_data[fi]["time"]
    ang = frames_data[fi][f"{side[0].upper()}_hip_angle"]
    vis = frames_data[fi][f"{side[0].upper()}_hip_vis"]
    lbl = label_angle(ang, vis)
    conf = confidence(vis, ang, fps)

    # Flag readings near threshold boundaries (within 3°): ±1-frame timing
    # uncertainty at 24fps can shift these into the adjacent label.
    ambig = (not math.isnan(ang) and vis >= VIS_MIN and
             (abs(ang - THR_FULL) < 3.0 or abs(ang - THR_PARTIAL) < 3.0))

    toe_off_results.append(dict(
        frame=fi,
        time_s=round(t, 3),
        side=side,
        hip_angle_deg=(round(ang, 1) if not math.isnan(ang) else None),
        label=lbl,
        confidence=conf,
        visibility=round(vis, 3),
        ambiguous=ambig,
        note=("Near threshold — ±1 frame at 24fps may shift label." if ambig else ""),
    ))

# ── Phase guard (SME A1: pattern defined for full-speed running only) ──────────
# Any toe-off whose frame has trunk lean > ACCEL_TRUNK_LEAN_DEG is in the
# acceleration phase: the athlete's posture does not match the A1 definition.
# Re-label as UNDETECTABLE rather than a speed-inappropriate FULL/PARTIAL/RESTRICTED.
for r in toe_off_results:
    lean = float(trunk_lean_sm[r["frame"]]) if not math.isnan(trunk_lean_sm[r["frame"]]) else 0.0
    r["trunk_lean_at_toeoff_deg"] = round(lean, 1)
    if lean > ACCEL_TRUNK_LEAN_DEG and r["label"] != "UNDETECTABLE":
        r["label"]      = "UNDETECTABLE"
        r["confidence"] = 0.0
        r["note"]       = (
            f"Acceleration phase (trunk lean {lean:.1f}° > {ACCEL_TRUNK_LEAN_DEG:.0f}° threshold) — "
            f"Pattern A1 applies to full-speed running only."
        )

# ── Plausibility guard (SME D3 rule) ──────────────────────────────────────────
# Among events that survived the phase guard, compute the median hip angle.
# Any reading more than PLAUSIBILITY_DROP_DEG below that median is a bad detection
# (tracking artefact, not a real sprint toe-off) — flag UNDETECTABLE, never RESTRICTED.
_fs_angles = [
    r["hip_angle_deg"] for r in toe_off_results
    if r["label"] != "UNDETECTABLE" and r["hip_angle_deg"] is not None
]
if _fs_angles:
    _fs_median = float(np.median(_fs_angles))
    for r in toe_off_results:
        if r["label"] == "UNDETECTABLE":
            continue
        ang = r["hip_angle_deg"]
        if ang is None:
            continue
        drop = _fs_median - ang
        if drop > PLAUSIBILITY_DROP_DEG:
            r["label"]      = "UNDETECTABLE"
            r["confidence"] = 0.0
            r["note"]       = (
                f"Implausible angle ({ang:.1f}° is {drop:.1f}° below "
                f"full-speed median {_fs_median:.1f}°) — "
                f"D3 rule: flagged UNDETECTABLE, not RESTRICTED."
            )

# ── Overall assessment ─────────────────────────────────────────────────────────
valid_results = [r for r in toe_off_results if r["label"] != "UNDETECTABLE"]
if not valid_results:
    overall = "UNDETECTABLE — no usable toe-off readings from this clip."
else:
    label_counts = {}
    for r in valid_results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    dominant = max(label_counts, key=label_counts.get)
    counts_str = ", ".join(f"{v}x {k}" for k, v in sorted(label_counts.items()))
    angles = [r["hip_angle_deg"] for r in valid_results if r["hip_angle_deg"] is not None]
    avg_ang = float(np.mean(angles)) if angles else float("nan")
    overall = (
        f"Dominant: {dominant}  ({counts_str}). "
        f"Mean hip angle at toe-off: {avg_ang:.1f}°. "
        f"24fps caveat: ±{1000/fps:.0f}ms timing uncertainty applies to all readings."
    )

# ── Write JSON ─────────────────────────────────────────────────────────────────
fps_caveat = (
    f"Clip is {fps:.0f}fps; SME brief D3 prefers ≥{FPS_PREF:.0f}fps for this pattern. "
    f"Timing resolution: ±{1000/fps:.0f}ms per frame. "
    f"The exact toe-off frame may be off by ±1 frame, causing a ~5–10° hip-angle error. "
    f"Confidence reduced by {int(FPS_PENALTY*100)}% on all non-UNDETECTABLE readings."
)

summary = dict(
    clip=VIDEO_PATH,
    resolution=f"{W}x{H}",
    fps=fps,
    total_frames=n_frames,
    detected_frames=n_detected,
    running_direction="left_to_right" if running_lr else "right_to_left",
    fps_caveat=fps_caveat,
    angle_convention=(
        "INCLUDED ANGLE (arccos): 180° = straight limb, "
        "decreases as joint bends. "
        "FULL_EXTENSION ≥ 160°, PARTIAL 145–160°, RESTRICTED < 145°."
    ),
    toe_offs=toe_off_results,
    overall_assessment=overall,
)

json_path = os.path.join(OUT_DIR, "hip_extension_results.json")
with open(json_path, "w") as jf:
    json.dump(summary, jf, indent=2)
print(f"  JSON → {json_path}")

# ── Plot ───────────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "FULL_EXTENSION": "#2e7d32",   # dark green
    "PARTIAL":        "#f57c00",   # orange
    "RESTRICTED":     "#c62828",   # dark red
    "UNDETECTABLE":   "#9e9e9e",   # grey
}

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2, 1]})

# ── Panel 1: Hip angles + toe-off markers ──────────────────────────────────────
ax1 = axes[0]
ax1.plot(times, L_hip_ang, color="#1565c0", lw=1.5, alpha=0.8, label="Left hip (incl. angle)")
ax1.plot(times, R_hip_ang, color="#b71c1c", lw=1.5, alpha=0.8, label="Right hip (incl. angle)")

# Threshold lines
ax1.axhline(THR_FULL,    color=LABEL_COLORS["FULL_EXTENSION"], ls=":", lw=1.2,
            label=f"Full ext. threshold ({THR_FULL:.0f}°)", alpha=0.7)
ax1.axhline(THR_PARTIAL, color=LABEL_COLORS["PARTIAL"],        ls=":", lw=1.2,
            label=f"Partial threshold ({THR_PARTIAL:.0f}°)", alpha=0.7)

# Toe-off event markers
for r in toe_off_results:
    c = LABEL_COLORS[r["label"]]
    ax1.axvline(r["time_s"], color=c, lw=1.2, ls="--", alpha=0.55)
    if r["hip_angle_deg"] is not None:
        marker = "^" if r["side"] == "left" else "v"
        ax1.scatter(r["time_s"], r["hip_angle_deg"], color=c, s=70,
                    marker=marker, zorder=5, linewidths=0)

ax1.set_ylim(100, 195)
ax1.set_ylabel("Hip angle — included (°)", fontsize=10)
ax1.set_title("Pattern A1: Hip Extension at Toe-Off  |  run_sideview.mp4  "
              f"({fps:.0f}fps ⚠ below 30fps preference)", fontsize=11)
ax1.legend(fontsize=8, loc="lower right", ncol=2)
ax1.grid(axis="y", alpha=0.2)

# Annotate toe-off events
for r in toe_off_results:
    if r["hip_angle_deg"] is not None:
        lbl_short = {"FULL_EXTENSION": "FE", "PARTIAL": "P",
                     "RESTRICTED": "R", "UNDETECTABLE": "?"}[r["label"]]
        ax1.annotate(
            lbl_short,
            xy=(r["time_s"], r["hip_angle_deg"] + 2),
            fontsize=7, ha="center",
            color=LABEL_COLORS[r["label"]],
        )

# ── Panel 2: Ankle Y positions + toe-off markers ───────────────────────────────
ax2 = axes[1]
ax2.plot(times, L_ay_sm, color="#1565c0", lw=1.5, alpha=0.7, label="Left ankle y (smoothed)")
ax2.plot(times, R_ay_sm, color="#b71c1c", lw=1.5, alpha=0.7, label="Right ankle y (smoothed)")
ax2.invert_yaxis()   # flip so "up" in the chart = foot raised above ground
ax2.set_ylabel("Ankle y (norm, ↑ = raised)", fontsize=10)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(axis="y", alpha=0.2)

for r in toe_off_results:
    c = LABEL_COLORS[r["label"]]
    ax2.axvline(r["time_s"], color=c, lw=1, ls="--", alpha=0.4)
# Mark detected peaks
for fi in L_toeoffs:
    if not math.isnan(L_ay_sm[fi]):
        ax2.scatter(times[fi], L_ay_sm[fi], color="#1565c0", s=40,
                    marker="x", zorder=5, linewidths=1.5)
for fi in R_toeoffs:
    if not math.isnan(R_ay_sm[fi]):
        ax2.scatter(times[fi], R_ay_sm[fi], color="#b71c1c", s=40,
                    marker="x", zorder=5, linewidths=1.5)

# ── Panel 3: Per-event confidence ─────────────────────────────────────────────
ax3 = axes[2]
if toe_off_results:
    bar_x = [r["time_s"]    for r in toe_off_results]
    bar_h = [r["confidence"] for r in toe_off_results]
    bar_c = [LABEL_COLORS[r["label"]] for r in toe_off_results]
    ax3.bar(bar_x, bar_h, width=0.12, color=bar_c, alpha=0.75)

ax3.axhline(0.5, color="#616161", ls=":", lw=1, label="Min. confidence 0.5")
ax3.axhline(1.0 - FPS_PENALTY, color="#fb8c00", ls="--", lw=1,
            label=f"24fps ceiling ({1.0-FPS_PENALTY:.2f})")
ax3.set_ylim(0, 1.05)
ax3.set_ylabel("Confidence", fontsize=10)
ax3.set_xlabel("Time (s)", fontsize=10)
ax3.legend(fontsize=8)
ax3.text(0.01, 0.6, f"⚠ 24fps: confidence capped at {1.0-FPS_PENALTY:.0%}",
         transform=ax3.transAxes, fontsize=8, color="#e65100")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "hip_extension_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot → {plot_path}")

# ── Printed summary ────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  PATTERN A1: HIP EXTENSION AT TOE-OFF — RESULTS")
print("=" * 70)
print(f"  Clip       : {VIDEO_PATH}  ({W}x{H} @ {fps:.0f}fps)")
print(f"  Frames     : {n_frames} total, {n_detected} with pose ({100*n_detected/n_frames:.1f}%)")
print()
print("  ⚠  24fps CAVEAT (SME D3 flag)")
print(f"     This clip is {fps:.0f}fps; SME brief D3 prefers ≥{FPS_PREF:.0f}fps for Pattern A1.")
print(f"     Timing resolution: ±{1000/fps:.0f}ms per frame.")
print( "     The exact toe-off frame may be off by ±1 frame → ~5–10° angle error.")
print(f"     Confidence reduced by {int(FPS_PENALTY*100)}% on all readings.")
print()
print(f"  Toe-offs detected: {len(all_toeoffs)}")
print()

if not toe_off_results:
    print("  No toe-off events detected. Check tracking or clip quality.")
else:
    hdr = f"  {'#':>2}  {'t(s)':>6}  {'Side':6}  {'Hip angle':>10}  {'Label':18}  {'Conf':>6}  Note"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, r in enumerate(toe_off_results, 1):
        ang_str = f"{r['hip_angle_deg']:.1f}°" if r["hip_angle_deg"] is not None else "  N/A "
        flag    = " ⚠BOUNDARY" if r["ambiguous"] else ""
        print(f"  {i:>2}  {r['time_s']:>6.2f}  {r['side']:6}  {ang_str:>10}  "
              f"{r['label']:18}  {r['confidence']:>6.2f}  {flag}")

print()
print(f"  OVERALL: {overall}")
print()

if valid_results:
    print("  Breakdown by label:")
    for lbl in ["FULL_EXTENSION", "PARTIAL", "RESTRICTED"]:
        subset = [r for r in valid_results if r["label"] == lbl]
        if subset:
            ang_list = [r["hip_angle_deg"] for r in subset
                        if r["hip_angle_deg"] is not None]
            ang_str = ", ".join(f"{a:.1f}°" for a in ang_list)
            print(f"    {lbl:18}: {len(subset)} event(s)  [{ang_str}]")

undetectable = [r for r in toe_off_results if r["label"] == "UNDETECTABLE"]
if undetectable:
    print(f"    UNDETECTABLE       : {len(undetectable)} event(s)  "
          f"(frames {', '.join(str(r['frame']) for r in undetectable)})")

print()
print(f"  Outputs saved to:  {OUT_DIR}/")
print(f"    hip_extension_results.json")
print(f"    hip_extension_plot.png")
print("=" * 70)
