"""
run_sports2d_multi.py
---------------------
Experimental multi-person Sports2D tracking script.

SEPARATE from run_sports2d.py — does NOT write to Sports2D_output/.
All output goes to: multiperson_output/<videoname>_Sports2D/

Usage:
  python run_sports2d_multi.py <video_path>

Example:
  python run_sports2d_multi.py "istockphoto-2257185924-640_adpp_is.mp4"

What it does:
  1. Runs Sports2D with nb_persons_to_detect=2 (both players)
  2. Saves skeleton-overlay video + per-person .mot angle files
  3. Prints a tracking summary: persons detected, frames per person,
     tracking continuity, and any obvious identity swaps.

Does NOT run trust_report.py or detect_reps.py.
"""

import sys
import os
import glob
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def read_mot_frames(path):
    """
    Read a Sports2D .mot file and return:
      total_rows     — total data rows (including NaN/empty rows)
      valid_rows     — rows where ALL joint columns have a value
      partial_rows   — rows where SOME columns have a value
      empty_rows     — rows where NO joint column has a value
      time_span      — (t_first_valid, t_last_valid) in seconds, or (None, None)
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Find header end
    start = next((i + 1 for i, l in enumerate(lines)
                  if l.strip().lower() == "endheader"), None)
    if start is None:
        return 0, 0, 0, 0, (None, None)

    header = lines[start].strip().split("\t")
    data_lines = lines[start + 1:]
    angle_cols = [h for h in header if h != "time"]

    total_rows = 0
    valid_rows = 0
    partial_rows = 0
    empty_rows = 0
    t_first = None
    t_last  = None

    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        total_rows += 1

        try:
            t = float(parts[0])
        except ValueError:
            continue

        angle_vals = [parts[i] for i, h in enumerate(header) if h != "time"
                      and i < len(parts)]
        n_populated = sum(1 for v in angle_vals if v.strip() != "")
        n_total     = len(angle_cols)

        if n_populated == 0:
            empty_rows += 1
        elif n_populated == n_total:
            valid_rows += 1
            if t_first is None:
                t_first = t
            t_last = t
        else:
            partial_rows += 1
            if t_first is None:
                t_first = t
            t_last = t

    return total_rows, valid_rows, partial_rows, empty_rows, (t_first, t_last)


def check_identity_stability(mot_path, fps, window_sec=0.5):
    """
    Crude identity-swap heuristic: sample the 'Left hip' or 'Right hip' column
    and check for large frame-to-frame jumps (> 30° in one frame = likely swap).
    Returns (swap_count, max_jump_deg).
    """
    with open(mot_path, encoding="utf-8") as f:
        lines = f.readlines()

    start = next((i + 1 for i, l in enumerate(lines)
                  if l.strip().lower() == "endheader"), None)
    if start is None:
        return 0, 0.0

    header = [h.strip() for h in lines[start].strip().split("\t")]

    # Pick hip column — prefer left hip, fall back to right hip
    target = None
    for candidate in ["left hip", "right hip", "left knee", "right knee"]:
        if candidate in header:
            target = candidate
            break
    if target is None:
        return 0, 0.0

    col_idx = header.index(target)
    values = []
    for line in lines[start + 1:]:
        parts = line.strip().split("\t")
        if len(parts) > col_idx and parts[col_idx].strip():
            try:
                values.append(float(parts[col_idx]))
            except ValueError:
                pass

    if len(values) < 2:
        return 0, 0.0

    diffs      = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
    max_jump   = max(diffs)
    swap_count = sum(1 for d in diffs if d > 30.0)   # >30° in one frame = suspect
    return swap_count, max_jump


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_sports2d_multi.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    stem       = os.path.splitext(os.path.basename(video_path))[0]
    result_dir = "multiperson_output"
    # Sports2D appends _Sports2D to the stem automatically
    expected_out = os.path.join(result_dir, f"{stem}_Sports2D")

    os.makedirs(result_dir, exist_ok=True)

    print("\n" + "=" * 62)
    print("  MULTI-PERSON Sports2D EXPERIMENT")
    print(f"  Video  : {video_path}")
    print(f"  Output : {expected_out}")
    print("=" * 62 + "\n")

    # ── Run Sports2D in multi-person mode ──────────────────────────────────
    try:
        from Sports2D import Sports2D
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        Sports2D.process({
            "base": {
                "video_input":            video_path,
                "nb_persons_to_detect":   2,          # track BOTH players
                "person_ordering_method": "largest_size",  # stable inter-frame ID
                "show_realtime_results":  False,
                "save_vid":               True,        # skeleton overlay video
                "save_img":               False,
                "save_pose":              True,
                "calculate_angles":       True,
                "save_angles":            True,
                "result_dir":             result_dir,  # → multiperson_output/
            },
            "pose": {
                "pose_model":    "body_with_feet",
                "mode":          "balanced",
                "det_frequency": 4,
            },
            "angles": {
                "joint_angles": [
                    "Right knee", "Left knee",
                    "Right hip",  "Left hip",
                    "Right ankle","Left ankle",
                ],
                "segment_angles": [
                    "Right thigh", "Left thigh", "Trunk",
                ],
            },
            "post-processing": {
                "show_graphs": False,
                "save_graphs": False,
                "filter":      True,
            },
        })

    except ImportError:
        print("Sports2D not installed. Activate venv and run: pip install sports2d")
        sys.exit(1)
    except Exception as e:
        print(f"\nSports2D error: {e}")
        raise

    # ── Post-run analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  POST-RUN TRACKING SUMMARY")
    print("=" * 62)

    # Find .mot files in the output folder
    mot_pattern = os.path.join(result_dir, "**", "*.mot")
    mot_files   = sorted(glob.glob(mot_pattern, recursive=True))
    # Filter to only those belonging to this video
    mot_files = [f for f in mot_files if stem in os.path.basename(f)]

    if not mot_files:
        print(f"  WARNING: No .mot files found under {result_dir}/ for {stem}.")
        print("  Sports2D may have failed or produced output in an unexpected path.")
    else:
        person_ids = sorted(set(
            os.path.basename(f).split("person")[-1].replace(".mot","")
            for f in mot_files if "person" in os.path.basename(f)
        ))

        print(f"\n  Persons tracked : {len(person_ids)}  "
              f"({', '.join('person' + p for p in person_ids)})")
        print()

        for mot in mot_files:
            pname = "person" + os.path.basename(mot).split("person")[-1].replace(".mot","")
            total, valid, partial, empty, (t0, t1) = read_mot_frames(mot)
            swap_count, max_jump = check_identity_stability(mot, fps)

            span = f"{t0:.2f}s – {t1:.2f}s" if t0 is not None else "n/a"
            print(f"  [{pname}]  {os.path.basename(mot)}")
            print(f"    Total rows   : {total}")
            print(f"    Full rows    : {valid}   (all joints populated)")
            print(f"    Partial rows : {partial} (some joints populated)")
            print(f"    Empty rows   : {empty}   (no joints)")
            print(f"    Active window: {span}")
            if valid + partial > 0:
                coverage = 100 * (valid + partial) / max(total, 1)
                print(f"    Coverage     : {coverage:.1f}%  of .mot timeline")
            print(f"    ID-swap heuristic: {swap_count} frame-jumps > 30°  "
                  f"(max jump = {max_jump:.1f}°)")
            if swap_count > 5:
                print(f"    *** IDENTITY INSTABILITY SUSPECTED — "
                      f"{swap_count} large jumps ***")
            elif swap_count == 0:
                print(f"    Identity looks STABLE (no large angle jumps)")
            else:
                print(f"    Identity MOSTLY STABLE ({swap_count} minor jumps — "
                      f"check skeleton video to confirm)")
            print()

    # List all output files produced
    print("  Output files produced:")
    if os.path.isdir(expected_out):
        for fname in sorted(os.listdir(expected_out)):
            fpath = os.path.join(expected_out, fname)
            size  = os.path.getsize(fpath) / 1e3
            print(f"    {fname}  ({size:.0f} KB)")
    else:
        # Try searching result_dir directly
        all_files = []
        for root, dirs, files in os.walk(result_dir):
            for f in files:
                if stem in f:
                    all_files.append(os.path.join(root, f))
        if all_files:
            for f in sorted(all_files):
                size = os.path.getsize(f) / 1e3
                print(f"    {f}  ({size:.0f} KB)")
        else:
            print(f"    (no files found in {result_dir}/ — check for errors above)")

    print("\n" + "=" * 62)
    print("  EXPERIMENT COMPLETE")
    print(f"  Skeleton video: {expected_out}/<name>_Sports2D.mp4")
    print("  Next step: inspect the skeleton video to confirm both players")
    print("  are tracked cleanly throughout the rally.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
