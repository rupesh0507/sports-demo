"""
run_all.py
----------
Runs the full sports-analytics pipeline on a single video.

Usage:
  python run_all.py <video.mp4>

Steps (in order):
  1. check_video_quality.py  — reject blurry / dark footage before wasting GPU time
  2. run_sports2d.py         — AI pose estimation → skeleton video + joint angles
  3. plot_joint_angle.py     — chart the most informative angle over time
  4. trust_report.py         — attach confidence scores to every measurement

Output always goes to:  Sports2D_output/<videoname>_Sports2D/
Each video gets its own isolated folder; nothing is shared or overwritten.
"""

import sys
import os
import subprocess

# ── helpers ──────────────────────────────────────────────────────────────────

PYTHON = sys.executable   # use the same interpreter that launched this script


def banner(step_num, total, label):
    width = 62
    print("\n" + "=" * width)
    print(f"  STEP {step_num}/{total}  —  {label}")
    print("=" * width)


def run_step(label, cmd, check_output_for=None):
    """
    Run a pipeline step as a subprocess.

    check_output_for: if set, scan stdout for this string and fail if absent.
    Returns (success: bool, stdout: str).
    """
    print(f"  Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n  FAILED  (exit code {result.returncode})")
        return False, ""

    return True, ""


def run_step_capture(label, cmd, check_output_for=None):
    """
    Like run_step but captures output so we can inspect it.
    Used for the quality-check step where we need to read GOOD/BAD.
    """
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Always echo the output so the user sees it
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        print(f"\n  FAILED  (exit code {result.returncode})")
        return False, result.stdout

    if check_output_for and check_output_for not in result.stdout:
        return False, result.stdout

    return True, result.stdout

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <video.mp4>")
        sys.exit(1)

    video = sys.argv[1]

    if not os.path.isfile(video):
        print(f"ERROR: video file not found: {video}")
        sys.exit(1)

    stem       = os.path.splitext(os.path.basename(video))[0]
    out_folder = os.path.join("Sports2D_output", f"{stem}_Sports2D")

    print("\n" + "=" * 62)
    print("  SPORTS ANALYTICS PIPELINE")
    print(f"  Video      : {video}")
    print(f"  Output     : {out_folder}")
    print("=" * 62)

    TOTAL = 4

    # ── Step 1: Quality check ────────────────────────────────────────────────
    banner(1, TOTAL, "Video Quality Check")
    ok, stdout = run_step_capture(
        "quality check",
        [PYTHON, "check_video_quality.py", video],
        check_output_for="GOOD",
    )
    if not ok:
        print("\n  Pipeline stopped: video failed the quality gate.")
        print("  Fix the video (better lighting / focus) and try again.")
        sys.exit(1)
    print("  Quality gate: PASSED")

    # ── Step 2: Sports2D pose estimation ────────────────────────────────────
    banner(2, TOTAL, "AI Pose Estimation  (Sports2D)")
    ok, _ = run_step(
        "Sports2D",
        [PYTHON, "run_sports2d.py", video],
    )
    if not ok:
        print("\n  Pipeline stopped: Sports2D failed.")
        print("  Check that the venv is active and sports2d is installed.")
        sys.exit(1)
    print(f"  Pose estimation: DONE  →  {out_folder}")

    # ── Step 3: Joint angle chart ────────────────────────────────────────────
    banner(3, TOTAL, "Joint Angle Chart")
    ok, _ = run_step(
        "chart",
        [PYTHON, "plot_joint_angle.py", video],
    )
    if not ok:
        print("\n  Pipeline stopped: chart generation failed.")
        sys.exit(1)
    print(f"  Chart: DONE  →  {out_folder}")

    # ── Step 4: Trust report ─────────────────────────────────────────────────
    banner(4, TOTAL, "Trust Report")
    ok, _ = run_step(
        "trust report",
        [PYTHON, "trust_report.py", video],
    )
    if not ok:
        print("\n  Pipeline stopped: trust report failed.")
        sys.exit(1)
    print(f"  Trust report: DONE  →  {out_folder}/trust_report.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"  Video      : {video}")
    print(f"  Output     : {out_folder}")
    print("  Files generated:")
    if os.path.isdir(out_folder):
        for f in sorted(os.listdir(out_folder)):
            print(f"    {f}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
