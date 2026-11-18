"""
Plots a joint angle over time from a Sports2D output file.

Supported formats:
  .mot  — Sports2D / OpenSim tab-separated (auto-detected)
  .csv  — plain CSV with a time column

Usage (preferred — derive folder from video name):
  python plot_joint_angle.py squat_video.mp4
  python plot_joint_angle.py squat_video.mp4 "right hip"

Usage (explicit file path — still works):
  python plot_joint_angle.py Sports2D_output/squat_video_Sports2D/squat_video_Sports2D_angles_person00.mot
  python plot_joint_angle.py demo_angles.csv RKnee_flexion

When given a video filename, the script derives the output folder as:
  Sports2D_output/<videoname>_Sports2D/<videoname>_Sports2D_angles_person00.mot
"""

import sys
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Anatomically plausible ranges per Sports2D column ───────────────────────
#
# Joint angles: 0° = limb straight, positive = bent.
# Segment angles: anticlockwise from horizontal; shank/thigh point downward
#   (negative for a person standing upright).
#
PLAUSIBLE_RANGES = {
    "right knee":   (-20,  160),
    "left knee":    (-20,  160),
    "right hip":    (-60,  160),
    "left hip":     (-60,  160),
    "right ankle":  (-150,  30),   # shank-from-horizontal, typically -100 to -30°
    "left ankle":   (-150,  30),
    "right thigh":  (-150,  10),   # thigh-from-horizontal, typically -120 to -60°
    "left thigh":   (-150,  10),
    "trunk":        (  40, 150),   # trunk-from-horizontal, ~90° when upright
}


def normalize_to_range(values, lo, hi):
    """
    Per-frame: shift each value by the smallest integer multiple of 360°
    that places it inside [lo, hi].  This corrects 360°-wrap artifacts
    (e.g. Sports2D reporting -348° instead of 12°) without touching
    genuinely-out-of-range tracking-failure frames.
    """
    mid = (lo + hi) / 2.0
    result = np.array(values, dtype=float)
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


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise every column that has a known plausible range."""
    df = df.copy()
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            df[col] = normalize_to_range(df[col].values, lo, hi)
    return df


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_mot(path: str) -> pd.DataFrame:
    """Read an OpenSim .mot file into a DataFrame (skips the header section)."""
    with open(path, "r") as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            start = i + 1
            break
    return pd.read_csv(path, sep="\t", skiprows=start)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def mot_path_from_video(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.join("Sports2D_output", f"{stem}_Sports2D")
    return os.path.join(folder, f"{stem}_Sports2D_angles_person00.mot")


def find_data_file(hint: str = None) -> str:
    if hint:
        ext = os.path.splitext(hint)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return mot_path_from_video(hint)
        if os.path.isfile(hint):
            return hint
    patterns = [
        "Sports2D_output/**/*angles*.mot",
        "Sports2D_output/**/*.mot",
        "Sports2D_output/**/*angles*.csv",
        "**/*angles*.csv",
        "demo_angles.csv",
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return matches[0]
    return None


# ── Column selection ──────────────────────────────────────────────────────────

def pick_best_column(df: pd.DataFrame, hint: str = None) -> str:
    """
    Pick the most informative angle column to plot.
    Scores by IN-RANGE variance (tracking-failure columns produce huge raw
    variance but have few valid frames — they should not win).
    """
    numeric = df.select_dtypes(include="number").columns.tolist()
    candidates = [c for c in numeric if "time" not in c.lower()]

    if hint and hint in df.columns:
        return hint

    preferred_keywords = ["hip", "knee", "trunk", "shoulder", "ankle"]
    scored = []
    for col in candidates:
        lo, hi = PLAUSIBLE_RANGES.get(col, (-1e9, 1e9))
        valid = df[col].dropna()
        valid = valid[(valid >= lo) & (valid <= hi)]
        in_range_var = float(valid.var()) if len(valid) > 1 else 0.0
        kw_bonus = 0
        for rank, kw in enumerate(preferred_keywords):
            if kw in col.lower():
                kw_bonus = (len(preferred_keywords) - rank) * 500
                break
        scored.append((col, in_range_var + kw_bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else candidates[0]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_angle(file_path: str, joint_col: str = None) -> None:
    print(f"Reading: {file_path}")
    if file_path.endswith(".mot"):
        df = read_mot(file_path)
    else:
        df = pd.read_csv(file_path)

    # Fix 360° wrap artifacts; values still outside range become NaN (chart gaps)
    df = apply_normalization(df)
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.any():
                n_bad = int(mask.sum())
                print(f"  [{col}] {n_bad} out-of-range frames masked (tracking failure)")
                df.loc[mask, col] = float("nan")

    print(f"Columns: {[c for c in df.columns if 'time' not in c.lower()]}")

    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if time_col is None:
        df.insert(0, "time", range(len(df)))
        time_col = "time"

    angle_col = pick_best_column(df, joint_col)
    valid_vals = df[angle_col].dropna()
    print(f"Plotting: {angle_col}  (range in this video: "
          f"{valid_vals.min():.1f}° – {valid_vals.max():.1f}°)")

    x = df[time_col]
    y = df[angle_col]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, y, linewidth=2, color="#1f77b4", label=angle_col.title())
    ax.fill_between(x, y, alpha=0.12, color="#1f77b4")

    lo, hi = PLAUSIBLE_RANGES.get(angle_col, (None, None))
    if lo is not None:
        ax.set_ylim(lo - 5, hi + 5)

    ax.set_title(f"Joint Angle Over Time — {angle_col.title()}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.legend(fontsize=11)

    out_png = file_path.rsplit(".", 1)[0] + f"_{angle_col.replace(' ', '_')}_plot.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Chart saved: {out_png}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    file_arg  = sys.argv[1] if len(sys.argv) > 1 else None
    joint_arg = sys.argv[2] if len(sys.argv) > 2 else None

    path = find_data_file(file_arg)
    if not path:
        print("No data file found.")
        print("Pass a video name:  python plot_joint_angle.py squat_video.mp4")
        print("Or a direct path:   python plot_joint_angle.py Sports2D_output/.../angles.mot")
        sys.exit(1)

    if not os.path.isfile(path):
        print(f"ERROR: derived angles file does not exist: {path}")
        print("Make sure Sports2D has been run on this video first.")
        sys.exit(1)

    plot_angle(path, joint_arg)
