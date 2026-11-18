"""
robust_pipeline.py
------------------
Robustness layer — 7 sequential stages:

  Stage 1  Inspect        — resolution, fps, rotation tag, codec, duration
  Stage 2  Normalize      — bake rotation, re-encode to H.264 if needed
  Stage 3  Quality gate   — blur + brightness check (same thresholds as
                            check_video_quality.py, NOT imported from it)
  Stage 4  MediaPipe pose — BlazePose on the normalized video; overlay video saved
  Stage 5  Subject select — pick main subject when multiple people detected
  Stage 6  Framing check  — flag OOF landmarks and "too far" via normalised coords
  Stage 7  Confidence     — per-landmark mean visibility; overall model confidence

Does NOT touch or import any existing pipeline script (check_video_quality.py,
run_sports2d.py, plot_joint_angle.py, trust_report.py, detect_reps.py,
run_all.py, app.py).

Writes ALL output to:  robust_output/<videoname>/
  <name>_normalized.mp4   — corrected video (H.264, pixels upright)
  <name>_mediapipe.mp4    — skeleton-overlay video
  inspect_report.json     — stages 1-3 results
  pose_report.json        — stages 4-7 results

Usage:
    python robust_pipeline.py somevideo.mp4
"""

import sys
import os
import json
import shutil
import subprocess
import datetime
import urllib.request

import cv2
import numpy as np
import imageio_ffmpeg
import mediapipe as mp

# ─────────────────────────────────────────────────────────────────────────────
# Quality-gate thresholds — kept here verbatim (same values as
# check_video_quality.py) so this file is fully standalone. The original
# script is never imported or modified.
# ─────────────────────────────────────────────────────────────────────────────
BLUR_THRESHOLD       = 80.0   # Laplacian variance — below this = too blurry
BRIGHTNESS_THRESHOLD = 40.0   # mean pixel value   — below this = too dark
SAMPLE_FRAMES        = 20     # evenly-spaced frames to sample for QC

OUTPUT_ROOT = "robust_output"

# ── MediaPipe BlazePose model ─────────────────────────────────────────────────
# "Full" variant: ~9 MB, good balance of accuracy and speed.
# Downloaded once, cached in the project folder.
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_full.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

MAX_PERSONS = 3   # detect up to this many people simultaneously per frame

# ── Framing-check thresholds ──────────────────────────────────────────────────
# BlazePose normalises landmark coordinates to [0, 1].
# Values outside this range mean the body part is outside the visible frame.
OUT_OF_FRAME_MARGIN    = 0.05   # allow 5% outside before calling it OOF
OUT_OF_FRAME_WARN_FRAC = 0.15   # warn if landmark OOF in > 15% of frames
TOO_FAR_BBOX_AREA      = 0.04   # subject bbox area < 4% of frame → too far

# Key landmarks to check during framing.
# Split into "critical" (torso/head — OOF here is PROBLEMATIC) and
# "extremity" (hands/feet — OOF is expected and only a WARNING).
_CRITICAL_LM = {0: "NOSE",
                11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER",
                23: "LEFT_HIP",      24: "RIGHT_HIP"}
_EXTREMITY_LM = {7: "LEFT_EAR",    8: "RIGHT_EAR",
                 13: "LEFT_ELBOW", 14: "RIGHT_ELBOW",
                 15: "LEFT_WRIST", 16: "RIGHT_WRIST",
                 25: "LEFT_KNEE",  26: "RIGHT_KNEE",
                 27: "LEFT_ANKLE", 28: "RIGHT_ANKLE"}
ALL_KEY_LM = {**_CRITICAL_LM, **_EXTREMITY_LM}

# ── MediaPipe Tasks API (v0.10+) ──────────────────────────────────────────────
# mp.solutions.* is removed in 0.10+; everything lives under mp.tasks.
_tv            = mp.tasks.vision
_BaseOptions   = mp.tasks.BaseOptions
LANDMARK_NAMES = [lm.name for lm in _tv.PoseLandmark]   # 33 names, index = enum value


# ─────────────────────────────────────────────────────────────────────────────
# Tool helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_ffmpeg_exe():
    """Return the path to the ffmpeg binary bundled by imageio_ffmpeg."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_ffprobe_exe():
    """
    imageio_ffmpeg bundles ffmpeg; ffprobe is normally in the same folder.
    We try the sibling binary first, then fall back to system PATH.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    candidate = os.path.join(
        os.path.dirname(ffmpeg),
        os.path.basename(ffmpeg).replace("ffmpeg", "ffprobe"),
    )
    if os.path.isfile(candidate):
        return candidate
    return "ffprobe"   # hope it's on PATH


def run_ffprobe(video_path, ffprobe_exe):
    """
    Run  ffprobe -print_format json -show_streams -show_format
    and return the parsed dict, or None if ffprobe is unavailable.
    """
    cmd = [
        ffprobe_exe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Inspect
# ─────────────────────────────────────────────────────────────────────────────

def inspect_video(video_path, ffprobe_exe):
    """
    Read and report all relevant properties of the video file.

    OpenCV supplies : stored width/height, fps, frame count, raw FourCC.
    ffprobe supplies: exact codec name, rotation metadata, duration, bitrate.

    Rotation metadata comes in two forms depending on how the video was made:
      tags.rotate       — older MP4s, most Android phones, GoPro
                          stored as positive CW degrees (e.g. 90 = 90° CW)
      side_data_list    — newer ffmpeg display-matrix format
                          stored as a negative CCW angle (e.g. -90 means
                          90° CW baked into the display matrix)
    Both are normalised here to a single positive CW rotation_degrees value.
    """
    print("\n[Stage 1 — INSPECT]")

    # ── OpenCV: basic geometry ────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"  ERROR: OpenCV cannot open '{video_path}'")

    cv_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cv_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv_fps   = cap.get(cv2.CAP_PROP_FPS)
    cv_fc    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
    # Decode the 4-byte FourCC integer into a human-readable string (e.g. "avc1")
    cv_fourcc = "".join(chr((fourcc_i >> 8 * i) & 0xFF) for i in range(4)).strip()
    cap.release()
    cv_duration = cv_fc / cv_fps if cv_fps > 0 else 0.0

    # ── ffprobe: codec, rotation, bitrate ─────────────────────────────────────
    probe       = run_ffprobe(video_path, ffprobe_exe)
    codec_name  = "unknown"
    rotation    = 0
    duration_s  = round(cv_duration, 2)
    bitrate_kbps = None

    if probe:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            codec_name = stream.get("codec_name", "unknown")

            # Older format: tags.rotate  (positive CW integer, e.g. "90")
            tag_rot = stream.get("tags", {}).get("rotate")
            if tag_rot is not None:
                rotation = int(tag_rot) % 360

            # Newer format: side_data_list → display matrix
            # ffprobe encodes this as the negative CCW angle, e.g. -90 = 90° CW.
            for sd in stream.get("side_data_list", []):
                r = sd.get("rotation")
                if r is not None:
                    rotation = int(-r) % 360  # flip sign, normalise to [0,360)
            break  # only need the first video stream

        fmt = probe.get("format", {})
        if fmt.get("duration"):
            duration_s = round(float(fmt["duration"]), 2)
        if fmt.get("bit_rate"):
            bitrate_kbps = round(int(fmt["bit_rate"]) / 1000)

    # ── Effective (viewer-visible) dimensions ─────────────────────────────────
    # If rotation is 90 or 270, width and height are swapped for the viewer.
    eff_w = cv_h if rotation in (90, 270) else cv_w
    eff_h = cv_w if rotation in (90, 270) else cv_h
    orientation = "portrait" if eff_h > eff_w else "landscape"

    props = {
        "width_stored":          cv_w,
        "height_stored":         cv_h,
        "resolution_stored":     f"{cv_w}×{cv_h}",
        "width_effective":       eff_w,
        "height_effective":      eff_h,
        "resolution_effective":  f"{eff_w}×{eff_h}",
        "orientation":           orientation,
        "fps":                   round(cv_fps, 3),
        "frame_count":           cv_fc,
        "duration_seconds":      duration_s,
        "codec_fourcc_opencv":   cv_fourcc,
        "codec_name_ffprobe":    codec_name,
        "rotation_degrees":      rotation,
        "bitrate_kbps":          bitrate_kbps,
        "ffprobe_available":     probe is not None,
    }

    # ── Console output ────────────────────────────────────────────────────────
    BAR = "─" * 54
    print(f"  {BAR}")
    print(f"  File          : {os.path.basename(video_path)}")
    print(f"  Stored size   : {cv_w}×{cv_h}")
    print(f"  Effective size: {eff_w}×{eff_h}  ({orientation})")
    print(f"  FPS           : {props['fps']}")
    print(f"  Duration      : {duration_s} s  ({cv_fc} frames)")
    print(f"  Codec         : {codec_name}  [OpenCV FourCC: '{cv_fourcc}']")
    rot_note = "  ⚠ stored pixels are rotated — normalization required" if rotation != 0 else "  (none)"
    print(f"  Rotation tag  : {rotation}°{rot_note}")
    # Portrait pixels (height > width) without a rotation tag: the camera
    # genuinely stored a portrait frame.  We accept portrait as valid; we
    # will re-encode to H.264 if needed but will NOT auto-rotate the pixels.
    if orientation == "portrait" and rotation == 0:
        print(f"  ⚠ Portrait pixels detected (stored {cv_w}×{cv_h}, no rotation tag).")
        print(f"    Keeping portrait orientation; will re-encode to H.264 if needed.")
    if bitrate_kbps:
        print(f"  Bitrate       : {bitrate_kbps} kbps")
    print(f"  ffprobe       : {'available ✓' if probe is not None else 'NOT found — using OpenCV only'}")
    print(f"  {BAR}")

    return props


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Normalize
# ─────────────────────────────────────────────────────────────────────────────

# Codec names that need no re-encoding (all mean H.264 / AVC).
H264_NAMES = {"h264", "avc", "avc1", "x264"}

# ffmpeg -vf transpose values:
#   transpose=1  →  rotate 90° CW   (use when rotation tag says 90°)
#   transpose=2  →  rotate 90° CCW  (use when rotation tag says 270°)
#   Two transpose=1 filters chained →  180° (when tag says 180°)
_TRANSPOSE = {
    90:  "transpose=1",
    180: "transpose=1,transpose=1",
    270: "transpose=2",
}


def normalize_video(video_path, out_path, props, ffmpeg_exe):
    """
    Produce a clean, upright, H.264 copy of the video at out_path.

    What triggers normalization:
      • rotation_degrees != 0  → the video was recorded rotated; bake the
                                  correction into the pixel data and clear
                                  the rotation metadata tag.
      • codec is not H.264     → re-encode to libx264 so downstream tools
                                  (Sports2D, browsers) can always read it.

    If neither condition applies, the video is simply copied — no re-encoding.
    """
    print("\n[Stage 2 — NORMALIZE]")

    rotation   = props["rotation_degrees"]
    raw_codec  = props["codec_name_ffprobe"].lower()
    fourcc     = props["codec_fourcc_opencv"].lower()

    needs_rotate = rotation != 0
    # If ffprobe wasn't available, raw_codec = "unknown"; treat as needing recode.
    needs_recode = (raw_codec not in H264_NAMES) and (fourcc not in H264_NAMES)

    ops = []
    if needs_rotate:
        ops.append(f"bake rotation ({rotation}° CW → upright pixels)")
    if needs_recode:
        ops.append(f"re-encode to H.264  (was: {raw_codec or fourcc})")

    if not ops:
        orient_note = " (portrait orientation preserved)" if props["orientation"] == "portrait" else ""
        print(f"  ✓ Already H.264 with no rotation tag — copying as-is{orient_note}")
        print(f"    → {out_path}")
        shutil.copy2(video_path, out_path)
        return {
            "action":     "copy",
            "operations": [],
            "reason":     f"already H.264, rotation=0{orient_note}",
            "output":     os.path.abspath(out_path),
        }

    for i, op in enumerate(ops, 1):
        print(f"  Operation {i}: {op}")
    print(f"  Writing to  : {out_path}")

    # Build the ffmpeg command.
    cmd = [ffmpeg_exe, "-y", "-i", video_path]

    if needs_rotate:
        vf = _TRANSPOSE.get(rotation)
        if vf:
            cmd += ["-vf", vf]
        else:
            # Unknown rotation angle — skip transpose, log warning.
            print(f"  WARNING: unsupported rotation {rotation}° — skipping transpose")

    # Always encode to libx264 when normalizing (fast + compatible).
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]

    # Clear the rotation metadata tag so players don't double-rotate.
    cmd += ["-metadata:s:v:0", "rotate=0"]

    # Move moov atom to the start so the file can stream without full download.
    cmd += ["-movflags", "+faststart"]

    cmd.append(out_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Print the last 600 chars of stderr — usually the most informative part.
        print(f"  ERROR: ffmpeg failed:\n{result.stderr[-600:]}")
        sys.exit(1)

    mb = os.path.getsize(out_path) / 1e6
    print(f"  ✓ Done — {mb:.1f} MB")

    return {
        "action":     "normalized",
        "operations": ops,
        "ffmpeg_cmd": " ".join(cmd),
        "output":     os.path.abspath(out_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Quality Gate
# ─────────────────────────────────────────────────────────────────────────────

def quality_gate(video_path):
    """
    Blur + brightness check — same algorithm and thresholds as
    check_video_quality.py, but returns structured data instead of only
    printing. The original script is NOT imported.

    Blur metric  : Laplacian variance of a grayscale frame.
                   Low variance = uniform image = blurry = unreliable edges.
    Bright metric: Mean grayscale pixel value.
                   Low value = dark frame = pose model struggles.
    Both are averaged over SAMPLE_FRAMES evenly-spaced frames.
    """
    print("\n[Stage 3 — QUALITY GATE]")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"  ERROR: cannot open '{video_path}'")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        sys.exit(f"  ERROR: no frames in '{video_path}'")

    sample_idx = np.linspace(0, total - 1, min(SAMPLE_FRAMES, total), dtype=int)

    blur_scores, bright_scores = [], []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        bright_scores.append(float(np.mean(gray)))
    cap.release()

    if not blur_scores:
        sys.exit("  ERROR: could not read any frames")

    avg_blur   = float(np.mean(blur_scores))
    avg_bright = float(np.mean(bright_scores))

    issues = []
    if avg_blur   < BLUR_THRESHOLD:
        issues.append(f"too blurry  (blur {avg_blur:.1f} < {BLUR_THRESHOLD})")
    if avg_bright < BRIGHTNESS_THRESHOLD:
        issues.append(f"too dark  (brightness {avg_bright:.1f} < {BRIGHTNESS_THRESHOLD})")

    passed = (len(issues) == 0)
    grade  = "GOOD" if passed else "REJECTED"

    print(f"  Blur score  : {avg_blur:.1f}   (threshold > {BLUR_THRESHOLD})")
    print(f"  Brightness  : {avg_bright:.1f}   (threshold > {BRIGHTNESS_THRESHOLD})")
    if passed:
        print(f"  ✓ {grade} — ready for pose analysis")
    else:
        print(f"  ✗ {grade}")
        for issue in issues:
            print(f"    ⚠  {issue}")

    return {
        "grade":                grade,
        "passed":               passed,
        "blur_score":           round(avg_blur, 2),
        "brightness":           round(avg_bright, 2),
        "blur_threshold":       BLUR_THRESHOLD,
        "brightness_threshold": BRIGHTNESS_THRESHOLD,
        "issues":               issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Summary + JSON report
# ─────────────────────────────────────────────────────────────────────────────

def save_report(out_folder, video_path, props, norm_result, qc_result):
    """Serialise everything to inspect_report.json and return the path."""
    report = {
        "generated_at":      datetime.datetime.now().isoformat(timespec="seconds"),
        "input_video":        os.path.abspath(video_path),
        "original_properties": props,
        "normalization":      norm_result,
        "quality_gate":       qc_result,
    }
    path = os.path.join(out_folder, "inspect_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_summary(video_path, props, norm_result, qc_result, report_path):
    """Final human-readable summary block."""
    BAR  = "─" * 56
    BAR2 = "═" * 56
    qc_icon = "✓" if qc_result["passed"] else "✗"

    norm_desc = (
        "none needed"
        if norm_result["action"] == "copy"
        else "; ".join(norm_result["operations"])
    )

    print(f"\n  {BAR2}")
    print(f"  ROBUST PIPELINE — SUMMARY")
    print(f"  {BAR}")
    print(f"  Input          : {os.path.basename(video_path)}")
    print(f"  Stored size    : {props['resolution_stored']}")
    if props["rotation_degrees"] != 0:
        print(f"  Effective size : {props['resolution_effective']}  (after rotation)")
    print(f"  Orientation    : {props['orientation']}")
    print(f"  Codec          : {props['codec_name_ffprobe']}")
    print(f"  Duration       : {props['duration_seconds']} s  "
          f"({props['frame_count']} frames @ {props['fps']} fps)")
    print(f"  Rotation tag   : {props['rotation_degrees']}°")
    print(f"  {BAR}")
    print(f"  Normalization  : {norm_desc}")
    print(f"  Output video   : {norm_result['output']}")
    print(f"  {BAR}")
    print(f"  Quality grade  : {qc_icon} {qc_result['grade']}  "
          f"(blur {qc_result['blur_score']:.1f}, "
          f"brightness {qc_result['brightness']:.1f})")
    if not qc_result["passed"]:
        for iss in qc_result["issues"]:
            print(f"    ⚠  {iss}")
    print(f"  {BAR}")
    print(f"  Report saved   : {report_path}")
    print(f"  {BAR2}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Stages 4-7 — MediaPipe pose, subject selection, framing, confidence
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_pose_model():
    """Download pose_landmarker_full.task if not already present (~9 MB)."""
    if os.path.isfile(MODEL_FILE):
        return
    print(f"  Downloading BlazePose model (~9 MB) …")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(MODEL_FILE, "wb") as f:
        f.write(r.read())
    print(f"  Saved: {MODEL_FILE}")


def _bbox_area(landmarks):
    """
    Normalised bounding-box area from a person's 33 landmark coordinates.
    Values in [0, 1]²; result is 0–1 (fraction of the full frame area).
    Ignores landmarks extrapolated far outside the frame (|coord| > 3).
    """
    xs = [lm.x for lm in landmarks if abs(lm.x) < 3]
    ys = [lm.y for lm in landmarks if abs(lm.y) < 3]
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _subject_centroid(landmarks):
    """Return (cx, cy) centroid of the 33 landmarks (screen coords 0-1)."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _select_subject(all_persons):
    """
    Given a list of per-person landmark lists, choose the 'main subject'.

    Strategy: largest normalised bounding-box area.
    A larger bbox means the person occupies more of the frame — the obvious
    foreground subject in a training/sports video.

    Returns (subject_index, subject_landmarks, areas_list).
    """
    areas = [_bbox_area(p) for p in all_persons]
    best  = int(np.argmax(areas))
    return best, all_persons[best], areas


def _draw_pose(frame_bgr, subject_lms, other_lms_list):
    """
    Draw BlazePose skeleton onto frame (in-place copy).
    Selected subject → default teal/orange style.
    Non-selected persons → muted gray (so they don't dominate the view).
    """
    du  = _tv.drawing_utils
    ds  = _tv.drawing_styles
    plc = _tv.PoseLandmarksConnections

    # Muted spec for background people
    gray_dot  = du.DrawingSpec(color=(110, 110, 110), thickness=1, circle_radius=2)
    gray_line = du.DrawingSpec(color=(70,  70,  70),  thickness=1)

    out = frame_bgr.copy()
    for lms in other_lms_list:
        du.draw_landmarks(out, lms, plc.POSE_LANDMARKS,
                          landmark_drawing_spec=gray_dot,
                          connection_drawing_spec=gray_line)
    if subject_lms is not None:
        du.draw_landmarks(out, subject_lms, plc.POSE_LANDMARKS,
                          landmark_drawing_spec=ds.get_default_pose_landmarks_style())
    return out


def run_pose_stages(norm_video_path, out_folder, stem):
    """
    Stages 4-7 combined: run BlazePose on the normalized video, collect
    per-frame landmark data, then compute subject-selection summary,
    framing verdict, and per-landmark confidence.

    Returns a dict that becomes pose_report.json.
    """

    # ── Stage 4 — MEDIAPIPE POSE ──────────────────────────────────────────────
    print("\n[Stage 4 — MEDIAPIPE POSE]")
    _ensure_pose_model()

    cap   = cv2.VideoCapture(norm_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_path = os.path.join(out_folder, f"{stem}_mediapipe.mp4")
    writer = cv2.VideoWriter(
        overlay_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h),
    )

    options = _tv.PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=_tv.RunningMode.VIDEO,
        num_poses=MAX_PERSONS,
        min_pose_detection_confidence=0.4,   # slightly lower to catch partially
        min_pose_presence_confidence=0.4,    # visible people in crowded frames
        min_tracking_confidence=0.4,
    )

    print(f"  Model    : pose_landmarker_full  (BlazePose, 33 landmarks)")
    print(f"  Video    : {total} frames  {w}×{h}  {fps:.1f} fps")
    print(f"  Max people per frame : {MAX_PERSONS}")
    print(f"  Overlay  : {overlay_path}")

    # Storage for all collected frame data
    per_frame  = []   # list of dicts, one per frame WITH detection
    frame_idx  = 0
    max_people = 0

    with _tv.PoseLandmarker.create_from_options(options) as lm:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts  = int(frame_idx * 1000.0 / fps)
            result = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )

            n = len(result.pose_landmarks)
            max_people = max(max_people, n)

            if n > 0:
                subj_idx, subj_lms, areas = _select_subject(result.pose_landmarks)
                others = [result.pose_landmarks[i]
                          for i in range(n) if i != subj_idx]
                per_frame.append({
                    "frame":       frame_idx,
                    "n_detected":  n,
                    "subj_idx":    subj_idx,
                    "subj_lms":    subj_lms,
                    "bbox_area":   areas[subj_idx],
                })
                annotated = _draw_pose(bgr, subj_lms, others)
                # Label the subject
                head = subj_lms[0]
                px   = max(0, int(head.x * w) - 30)
                py   = max(12, int(head.y * h) - 12)
                cv2.putText(annotated, "SUBJECT", (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 180), 1,
                            cv2.LINE_AA)
            else:
                annotated = bgr.copy()

            cv2.putText(annotated, f"f{frame_idx}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  … {frame_idx}/{total} frames")

    cap.release()
    writer.release()

    n_detected = len(per_frame)
    print(f"  ✓ {n_detected}/{total} frames with detection  "
          f"(max {max_people} {'person' if max_people == 1 else 'people'} in a frame)")

    # ── Stage 5 — SUBJECT SELECTION ───────────────────────────────────────────
    print("\n[Stage 5 — SUBJECT SELECTION]")
    print(f"  Max people seen in any single frame : {max_people}")
    if max_people <= 1:
        print(f"  ✓ Single person throughout — no selection needed")
    else:
        print(f"  Strategy : largest bounding box per frame")
        multi_frames = sum(1 for f in per_frame if f["n_detected"] > 1)
        print(f"  Frames with >1 person : {multi_frames}/{n_detected}")
        # Summarise who was selected when multiple present
        sel_counts = {}
        for f in per_frame:
            if f["n_detected"] > 1:
                sel_counts[f["subj_idx"]] = sel_counts.get(f["subj_idx"], 0) + 1
        for idx, cnt in sorted(sel_counts.items()):
            print(f"    Person slot {idx} selected in {cnt} multi-person frames")

    mean_bbox  = (float(np.mean([f["bbox_area"] for f in per_frame]))
                  if per_frame else 0.0)
    subj_selection = {
        "max_persons_per_frame":  max_people,
        "strategy":               "largest_bounding_box",
        "frames_with_detection":  n_detected,
        "frames_no_detection":    total - n_detected,
        "mean_subject_bbox_area": round(mean_bbox, 4),
    }

    # ── Stage 6 — FRAMING CHECK ───────────────────────────────────────────────
    print("\n[Stage 6 — FRAMING CHECK]")

    if not per_frame:
        print("  ✗ No frames had detections — cannot check framing.")
        framing = {
            "verdict": "UNKNOWN",
            "subject_too_far": None,
            "mean_bbox_area_normalized": 0.0,
            "key_landmark_issues": [],
            "note": "no detections in any frame",
        }
    else:
        lo = -OUT_OF_FRAME_MARGIN
        hi =  1.0 + OUT_OF_FRAME_MARGIN

        # Count OOF occurrences per key landmark
        oof_counts = {idx: 0 for idx in ALL_KEY_LM}
        for fdata in per_frame:
            lms = fdata["subj_lms"]
            for idx in ALL_KEY_LM:
                lm = lms[idx]
                if lm.x < lo or lm.x > hi or lm.y < lo or lm.y > hi:
                    oof_counts[idx] += 1

        oof_frac = {idx: oof_counts[idx] / n_detected for idx in ALL_KEY_LM}

        issues        = []
        problematic   = False
        has_warning   = False

        print(f"  Landmark OOF analysis  (>{OUT_OF_FRAME_WARN_FRAC:.0%} threshold)")
        for idx, name in ALL_KEY_LM.items():
            frac = oof_frac[idx]
            if frac > OUT_OF_FRAME_WARN_FRAC:
                is_critical = idx in _CRITICAL_LM
                sev = "PROBLEMATIC" if is_critical else "WARNING"
                if is_critical:
                    problematic = True
                else:
                    has_warning = True
                issues.append({
                    "landmark":         name,
                    "index":            idx,
                    "frames_out":       oof_counts[idx],
                    "fraction":         round(frac, 3),
                    "severity":         sev,
                })
                print(f"  ⚠  {name:<20} OOF {oof_counts[idx]:3d}/{n_detected} frames "
                      f"({frac:.0%})  [{sev}]")

        if not issues:
            print(f"  ✓ All key landmarks within frame bounds")

        too_far = mean_bbox < TOO_FAR_BBOX_AREA
        print(f"  Subject bbox area     : {mean_bbox:.1%}  of frame"
              f"  ({'✗ TOO FAR' if too_far else '✓ OK'})")

        if too_far:
            has_warning = True

        if problematic:
            verdict = "PROBLEMATIC"
        elif has_warning or too_far:
            verdict = "WARNING"
        else:
            verdict = "GOOD"

        icon = "✓" if verdict == "GOOD" else "⚠" if verdict == "WARNING" else "✗"
        print(f"  {icon} Framing verdict : {verdict}")

        framing = {
            "verdict":                   verdict,
            "subject_too_far":           too_far,
            "mean_bbox_area_normalized": round(mean_bbox, 4),
            "key_landmark_issues":       issues,
        }

    # ── Stage 7 — CONFIDENCE HARVEST ─────────────────────────────────────────
    print("\n[Stage 7 — CONFIDENCE HARVEST]")

    # Accumulate visibility values for every landmark index across all frames
    vis_by_lm = [[] for _ in range(33)]
    for fdata in per_frame:
        lms = fdata["subj_lms"]
        for i, lm in enumerate(lms):
            vis_by_lm[i].append(float(lm.visibility))

    per_lm_vis = {}
    for i, name in enumerate(LANDMARK_NAMES):
        vals = vis_by_lm[i]
        per_lm_vis[name] = {
            "index":            i,
            "mean_visibility":  round(float(np.mean(vals)), 4) if vals else None,
            "min_visibility":   round(float(np.min(vals)),  4) if vals else None,
            "max_visibility":   round(float(np.max(vals)),  4) if vals else None,
            "frames_measured":  len(vals),
        }

    all_vals = [v for sub in vis_by_lm for v in sub]
    overall  = round(float(np.mean(all_vals)), 4) if all_vals else None

    # Sort for printing: lowest mean first (worst → needs attention)
    sorted_by_vis = sorted(
        [(name, d["mean_visibility"]) for name, d in per_lm_vis.items()
         if d["mean_visibility"] is not None],
        key=lambda x: x[1],
    )
    print(f"  Overall mean visibility : {overall:.4f}" if overall else "  No data")
    print(f"  5 weakest landmarks:")
    for name, v in sorted_by_vis[:5]:
        bar = "█" * int(v * 20)
        print(f"    {name:<28}  {v:.3f}  {bar}")
    print(f"  5 strongest landmarks:")
    for name, v in sorted_by_vis[-5:][::-1]:
        bar = "█" * int(v * 20)
        print(f"    {name:<28}  {v:.3f}  {bar}")

    # ── Assemble report ───────────────────────────────────────────────────────
    pose_report = {
        "generated_at":          datetime.datetime.now().isoformat(timespec="seconds"),
        "video_file":            os.path.basename(norm_video_path),
        "normalized_video":      os.path.abspath(norm_video_path),
        "overlay_video":         os.path.abspath(overlay_path),
        "mediapipe_model":       "pose_landmarker_full",
        "frames_total":          total,
        "frames_with_detection": n_detected,
        "subject_selection":     subj_selection,
        "framing":               framing,
        "per_landmark_visibility": per_lm_vis,
        "overall_mean_visibility": overall,
    }
    return pose_report


def save_pose_report(out_folder, pose_report):
    """Write pose_report.json and return the path."""
    path = os.path.join(out_folder, "pose_report.json")
    with open(path, "w") as f:
        json.dump(pose_report, f, indent=2)
    return path


def print_pose_summary(pose_report, report_path):
    """Print a clean summary block for stages 4-7."""
    BAR  = "─" * 56
    BAR2 = "═" * 56
    fr   = pose_report["framing"]
    sc   = pose_report["subject_selection"]
    ov   = pose_report["overall_mean_visibility"]

    icon_framing = "✓" if fr["verdict"] == "GOOD" else "⚠" if fr["verdict"] == "WARNING" else "✗"

    print(f"\n  {BAR2}")
    print(f"  POSE ANALYSIS — SUMMARY")
    print(f"  {BAR}")
    print(f"  Frames processed  : {pose_report['frames_total']}")
    det  = pose_report["frames_with_detection"]
    tot  = pose_report["frames_total"]
    pct  = det / tot * 100 if tot else 0
    print(f"  Frames with pose  : {det}/{tot}  ({pct:.1f}%)")
    print(f"  Max people/frame  : {sc['max_persons_detected'] if 'max_persons_detected' in sc else sc['max_persons_per_frame']}")
    print(f"  Mean subject size : {sc['mean_subject_bbox_area']:.1%} of frame")
    print(f"  {BAR}")
    print(f"  Framing verdict   : {icon_framing} {fr['verdict']}")
    if fr.get("subject_too_far"):
        print(f"    ⚠  Subject bounding box < {TOO_FAR_BBOX_AREA:.0%} of frame — may be too far")
    for iss in fr.get("key_landmark_issues", []):
        print(f"    ⚠  {iss['landmark']}: OOF {iss['fraction']:.0%} of frames [{iss['severity']}]")
    if not fr.get("key_landmark_issues") and not fr.get("subject_too_far"):
        print(f"    All key landmarks within frame bounds")
    print(f"  {BAR}")
    print(f"  Model confidence  : {ov:.4f}  (mean visibility across all landmarks)")

    # Weakest 3 landmarks (only print if notably low)
    lmv = pose_report["per_landmark_visibility"]
    weak = sorted(
        [(n, d["mean_visibility"]) for n, d in lmv.items() if d["mean_visibility"] is not None],
        key=lambda x: x[1],
    )
    if weak and weak[0][1] < 0.7:
        print(f"  Notable weak spots:")
        for name, v in weak[:3]:
            if v < 0.7:
                print(f"    {name}: {v:.3f}")
    print(f"  {BAR}")
    print(f"  Overlay video     : {pose_report['overlay_video']}")
    print(f"  Pose report       : {report_path}")
    print(f"  {BAR2}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python robust_pipeline.py <video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        sys.exit(f"ERROR: file not found — {video_path}")

    stem       = os.path.splitext(os.path.basename(video_path))[0]
    out_folder = os.path.join(OUTPUT_ROOT, stem)
    os.makedirs(out_folder, exist_ok=True)

    ffmpeg_exe  = get_ffmpeg_exe()
    ffprobe_exe = get_ffprobe_exe()

    norm_out = os.path.join(out_folder, f"{stem}_normalized.mp4")

    # ── Run pipeline stages ──────────────────────────────────────────────────
    props       = inspect_video(video_path, ffprobe_exe)
    norm_result = normalize_video(video_path, norm_out, props, ffmpeg_exe)
    qc_result   = quality_gate(norm_out)

    # ── Save + print summary ─────────────────────────────────────────────────
    report_path = save_report(out_folder, video_path, props, norm_result, qc_result)
    print_summary(video_path, props, norm_result, qc_result, report_path)

    if not qc_result["passed"]:
        print("  ✗ Pipeline halted — video is not usable for pose analysis.\n")
        sys.exit(2)

    # ── Stages 4-7 — MediaPipe pose, subject selection, framing, confidence ───
    pose_report = run_pose_stages(norm_out, out_folder, stem)
    pose_report_path = save_pose_report(out_folder, pose_report)
    print_pose_summary(pose_report, pose_report_path)


if __name__ == "__main__":
    main()
