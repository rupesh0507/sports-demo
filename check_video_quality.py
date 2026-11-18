"""
Video Quality Checker
Checks if a video is good enough to analyze with Sports2D.
- Blur: Laplacian variance (low = blurry)
- Brightness: mean pixel value (low = too dark)

Fallback: when a video fails blur (but passes brightness), a local-block
sharpness check rescues cases where a large uniform background (green screen,
dark studio backdrop) dilutes the whole-frame metric even though the subject
is sharp.  Rescued videos are annotated PASS_VIA_LOCAL_SHARPNESS.
The fallback never runs on videos that already pass; it cannot change a PASS
to a FAIL.

Usage: python check_video_quality.py <video_path>
       python check_video_quality.py  (runs on both test_good.mp4 and test_bad.mp4)
"""

import sys
import cv2
import numpy as np

BLUR_THRESHOLD       = 80.0   # below this = too blurry  (whole-frame Laplacian variance)
BRIGHTNESS_THRESHOLD = 40.0   # below this = too dark     (mean pixel value)
SAMPLE_FRAMES        = 20     # evenly-spaced frames to sample from the video

# ── Local-sharpness fallback constants ───────────────────────────────────────
# Used only when the whole-frame blur fails but brightness is fine.
# Divide each sampled frame into LOCAL_GRID × LOCAL_GRID blocks, average the
# Laplacian variance of the top LOCAL_TOP_N sharpest blocks.  A flat
# background contributes near-zero variance blocks; a sharp subject in even
# a small region of the frame produces high-variance blocks.
LOCAL_BLUR_THRESHOLD = 400.0  # mean top-block score must reach this to rescue
LOCAL_GRID           = 8      # grid divisions per axis → 64 blocks per frame
LOCAL_TOP_N          = 3      # average the top-N sharpest blocks per frame


def _top_block_sharpness(gray):
    """
    Divide a grayscale frame into LOCAL_GRID × LOCAL_GRID blocks.
    Return the mean Laplacian variance of the top LOCAL_TOP_N sharpest blocks.
    Returns 0.0 if blocks are too small to be meaningful.
    """
    h, w = gray.shape
    bh, bw = h // LOCAL_GRID, w // LOCAL_GRID
    if bh < 4 or bw < 4:
        return 0.0
    scores = []
    for r in range(LOCAL_GRID):
        for c in range(LOCAL_GRID):
            block = gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            scores.append(float(cv2.Laplacian(block, cv2.CV_64F).var()))
    scores.sort(reverse=True)
    return float(np.mean(scores[:LOCAL_TOP_N]))


def check_video(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR – cannot open '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"ERROR – '{video_path}' has no frames")
        cap.release()
        return

    # Pick evenly-spaced frame indices to sample
    sample_indices = np.linspace(0, total_frames - 1, min(SAMPLE_FRAMES, total_frames), dtype=int)

    blur_scores       = []
    brightness_scores = []
    gray_frames       = []   # kept for the local-sharpness fallback (20 frames, small cost)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_scores.append(float(np.mean(gray)))
        gray_frames.append(gray)

    cap.release()

    if not blur_scores:
        print(f"ERROR – could not read any frames from '{video_path}'")
        return

    avg_blur       = float(np.mean(blur_scores))
    avg_brightness = float(np.mean(brightness_scores))

    print(f"\n{'='*55}")
    print(f"  File      : {video_path}")
    print(f"  Blur score: {avg_blur:.1f}  (threshold: >{BLUR_THRESHOLD})")
    print(f"  Brightness: {avg_brightness:.1f}  (threshold: >{BRIGHTNESS_THRESHOLD})")
    print(f"{'='*55}")

    issues = []
    if avg_blur < BLUR_THRESHOLD:
        issues.append(f"too blurry (score {avg_blur:.1f} < {BLUR_THRESHOLD})")
    if avg_brightness < BRIGHTNESS_THRESHOLD:
        issues.append(f"too dark (brightness {avg_brightness:.1f} < {BRIGHTNESS_THRESHOLD})")

    # ── Local-sharpness fallback ──────────────────────────────────────────────
    # Triggered only when: blur fails AND brightness passes.
    # Never runs on videos that already pass (avg_blur >= BLUR_THRESHOLD).
    # Cannot demote a passing video — only promotes a blur-only failure.
    local_score = None
    rescued     = False
    if avg_blur < BLUR_THRESHOLD and avg_brightness >= BRIGHTNESS_THRESHOLD:
        per_frame = [_top_block_sharpness(g) for g in gray_frames]
        local_score = float(np.mean(per_frame)) if per_frame else 0.0
        if local_score >= LOCAL_BLUR_THRESHOLD:
            rescued = True
            issues  = []   # brightness was already fine; clear the blur issue

    # ── Verdict ───────────────────────────────────────────────────────────────
    if issues:
        print(f"  BAD – rejected  |  Reason: {' and '.join(issues)}")
        if local_score is not None:
            print(f"  Local top-block sharpness: {local_score:.1f}  "
                  f"(< {LOCAL_BLUR_THRESHOLD} threshold — no sharp region found)")
    elif rescued:
        print(f"  GOOD – ready to analyze  [PASS_VIA_LOCAL_SHARPNESS]")
        print(f"  Note: whole-frame blur {avg_blur:.1f} < {BLUR_THRESHOLD} threshold, but")
        print(f"        local top-block sharpness {local_score:.1f} >= {LOCAL_BLUR_THRESHOLD}.")
        print(f"        Likely cause: uniform background diluting whole-frame metric.")
    else:
        print("  GOOD – ready to analyze")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_video(sys.argv[1])
    else:
        # Default: test both sample clips if they exist
        for clip in ["test_good.mp4", "test_bad.mp4"]:
            check_video(clip)
