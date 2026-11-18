"""
Creates two synthetic test videos:
  test_good.mp4 – bright, sharp, high contrast (quality checker should PASS)
  test_bad.mp4  – dark and blurry (quality checker should FAIL)

Run once: python create_test_videos.py
"""

import cv2
import numpy as np

FPS = 30
DURATION_SECS = 3
W, H = 640, 480


def make_good_video(path: str) -> None:
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * DURATION_SECS):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Bright white background
        frame[:] = (200, 200, 200)
        # Moving dark circle – sharp edges give high Laplacian variance
        cx = int(W * 0.1 + (W * 0.8) * (i / (FPS * DURATION_SECS)))
        cv2.circle(frame, (cx, H // 2), 60, (30, 30, 30), -1)
        cv2.putText(frame, "GOOD VIDEO", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        out.write(frame)
    out.release()
    print(f"Created: {path}")


def make_bad_video(path: str) -> None:
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * DURATION_SECS):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Very dark background
        frame[:] = (15, 15, 15)
        # Soft blurry blob
        cx = int(W * 0.1 + (W * 0.8) * (i / (FPS * DURATION_SECS)))
        cv2.circle(frame, (cx, H // 2), 60, (25, 25, 25), -1)
        # Heavy blur to destroy sharpness
        frame = cv2.GaussianBlur(frame, (31, 31), 10)
        out.write(frame)
    out.release()
    print(f"Created: {path}")


if __name__ == "__main__":
    make_good_video("test_good.mp4")
    make_bad_video("test_bad.mp4")
    print("Done. Now run: python check_video_quality.py")
