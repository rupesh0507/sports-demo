"""
Standalone MediaPipe pose-detection scan for sprint1_testset verification.
Uses the Tasks API (mediapipe 0.10.x) — no legacy solutions needed.

Usage:
    python detect_scan.py <video_path> [--mode single|multi]

Reports:
    - total frames sampled
    - % frames with ≥1 pose detected  (single mode)
    - % frames with ≥2 poses detected (multi mode)
    - key landmark visibility for relevant body region
"""

import sys
import os
import cv2
import json
import numpy as np

# Model path — same directory as mediapipe_pipeline.py uses
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "pose_landmarker_full.task",
)

SAMPLE_RATE = 1   # analyse every Nth frame (1 = every frame, 3 = every 3rd)
MAX_FRAMES  = 300 # cap to avoid very long clips taking forever


def run_scan(video_path: str, mode: str = "single") -> dict:
    import mediapipe as mp
    _tv          = mp.tasks.vision
    _BaseOptions = mp.tasks.BaseOptions

    num_poses = 1 if mode == "single" else 4
    options = _tv.PoseLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=os.path.abspath(MODEL_FILE)),
        running_mode=_tv.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_checked  = 0
    frames_any_pose = 0
    frames_multi    = 0   # ≥2 poses
    hip_vis_sum     = 0.0
    knee_vis_sum    = 0.0
    hip_vis_n       = 0

    # Landmark indices (MediaPipe 33-point BlazePose)
    # 23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee
    # 27=left_ankle, 28=right_ankle, 11=left_shoulder, 12=right_shoulder
    HIP_IDS   = [23, 24]
    KNEE_IDS  = [25, 26]
    ANKLE_IDS = [27, 28]

    frame_idx = 0
    with _tv.PoseLandmarker.create_from_options(options) as lmkr:
        while frames_checked < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % SAMPLE_RATE != 0:
                continue

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lmkr.detect(mp_img)

            frames_checked += 1
            n_poses = len(result.pose_landmarks) if result.pose_landmarks else 0

            if n_poses >= 1:
                frames_any_pose += 1
                lm = result.pose_landmarks[0]
                hip_v  = np.mean([lm[i].visibility for i in HIP_IDS])
                knee_v = np.mean([lm[i].visibility for i in KNEE_IDS])
                hip_vis_sum  += hip_v
                knee_vis_sum += knee_v
                hip_vis_n    += 1

            if n_poses >= 2:
                frames_multi += 1

    cap.release()

    pct_any  = 100.0 * frames_any_pose / frames_checked if frames_checked else 0.0
    pct_multi = 100.0 * frames_multi   / frames_checked if frames_checked else 0.0
    avg_hip_vis  = hip_vis_sum  / hip_vis_n if hip_vis_n else 0.0
    avg_knee_vis = knee_vis_sum / hip_vis_n if hip_vis_n else 0.0

    return {
        "video"            : os.path.basename(video_path),
        "resolution"       : f"{width}x{height}",
        "fps"              : round(fps, 2),
        "total_frames"     : total_frames,
        "frames_checked"   : frames_checked,
        "pct_pose_detected": round(pct_any, 1),
        "pct_two_poses"    : round(pct_multi, 1),
        "avg_hip_visibility"  : round(avg_hip_vis,  3),
        "avg_knee_visibility" : round(avg_knee_vis, 3),
        "mode"             : mode,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_scan.py <video> [single|multi]")
        sys.exit(1)
    vpath = sys.argv[1]
    mode  = sys.argv[2] if len(sys.argv) > 2 else "single"
    result = run_scan(vpath, mode)
    print(json.dumps(result, indent=2))
