"""
RTMPose (HALPE_26) frame-by-frame verification scan.
Uses the same model + backend that Sports2D uses internally.
Writes all output to sprint1_testset/ — never touches Sports2D_output/.

Usage:
    python rtmpose_scan.py <video_path>

Reports per-frame detection of hip, knee, ankle on the camera-facing side,
plus contiguous-run analysis for A1/A4 suitability.
"""

import sys
import os
import json
import cv2
import numpy as np

# HALPE_26 keypoint indices used by Sports2D / RTMPose-m
# L=left, R=right (from the athlete's own perspective)
L_HIP, R_HIP     = 11, 12
L_KNEE, R_KNEE   = 13, 14
L_ANKLE, R_ANKLE = 15, 16
L_SHOULDER, R_SHOULDER = 5, 6

# Model paths (cached by Sports2D on first run)
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "rtmlib", "hub", "checkpoints")
DET_MODEL  = os.path.join(CACHE, "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx")
POSE_MODEL = os.path.join(CACHE, "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.onnx")

# Confidence threshold matching Sports2D's keypoint_likelihood_threshold
CONF_THR = 0.3

# Minimum hip+knee+ankle confidence to count a frame as "fully tracked"
def lower_body_detected(scores, side):
    """Return True if hip+knee+ankle for the given side all exceed CONF_THR."""
    if side == "right":
        idxs = [R_HIP, R_KNEE, R_ANKLE]
    else:
        idxs = [L_HIP, L_KNEE, L_ANKLE]
    if scores is None or len(scores) == 0:
        return False
    return all(scores[i] >= CONF_THR for i in idxs)


def scan_video(video_path: str) -> dict:
    from rtmlib import Body

    print(f"Loading RTMPose models…")
    body = Body(
        det=DET_MODEL,
        det_input_size=(640, 640),
        pose=POSE_MODEL,
        pose_input_size=(192, 256),  # rtmlib (W,H); model ONNX expects (B,3,H,W)=(B,3,256,192)
        mode="balanced",
        backend="openvino",
        device="cpu",
    )
    print("Models loaded.")

    cap = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_results = []   # list of dicts per frame

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run RTMPose
        keypoints, scores = body(frame)   # returns arrays

        # keypoints: (N,K,2)  scores: (N,K)
        if keypoints is not None and len(keypoints) > 0:
            kp   = keypoints[0]   # first person
            sc   = scores[0]
            person_detected = True
            r_ok = lower_body_detected(sc, "right")
            l_ok = lower_body_detected(sc, "left")
            # Mean confidence for lower body keypoints
            lower_idxs = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
            mean_lower = float(np.mean([sc[i] for i in lower_idxs]))
            r_hip_sc  = float(sc[R_HIP])
            r_knee_sc = float(sc[R_KNEE])
            r_ank_sc  = float(sc[R_ANKLE])
            l_hip_sc  = float(sc[L_HIP])
            l_knee_sc = float(sc[L_KNEE])
            l_ank_sc  = float(sc[L_ANKLE])
        else:
            person_detected = False
            r_ok = l_ok = False
            mean_lower = 0.0
            r_hip_sc = r_knee_sc = r_ank_sc = 0.0
            l_hip_sc = l_knee_sc = l_ank_sc = 0.0

        frame_results.append({
            "frame": frame_idx,
            "person": person_detected,
            "right_ok": r_ok,
            "left_ok": l_ok,
            "either_ok": r_ok or l_ok,
            "mean_lower": mean_lower,
            "r_hip": r_hip_sc, "r_knee": r_knee_sc, "r_ank": r_ank_sc,
            "l_hip": l_hip_sc, "l_knee": l_knee_sc, "l_ank": l_ank_sc,
        })

    cap.release()

    n = len(frame_results)
    if n == 0:
        return {"error": "no frames read"}

    # Summary stats
    pct_person  = 100.0 * sum(r["person"]    for r in frame_results) / n
    pct_right   = 100.0 * sum(r["right_ok"]  for r in frame_results) / n
    pct_left    = 100.0 * sum(r["left_ok"]   for r in frame_results) / n
    pct_either  = 100.0 * sum(r["either_ok"] for r in frame_results) / n
    mean_lower_all = float(np.mean([r["mean_lower"] for r in frame_results]))

    # Dominant side = whichever side has higher detection rate
    dominant_side = "right" if pct_right >= pct_left else "left"
    dominant_pct  = max(pct_right, pct_left)

    # Longest contiguous run of "either_ok" frames
    max_run = cur_run = 0
    for r in frame_results:
        if r["either_ok"]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    # Duration of longest run in seconds
    longest_run_sec = round(max_run / fps, 2) if fps > 0 else 0

    # Per-frame breakdown (just print, don't bloat JSON)
    result = {
        "video"              : os.path.basename(video_path),
        "resolution"         : f"{width}x{height}",
        "fps"                : round(fps, 2),
        "total_frames"       : total,
        "frames_processed"   : n,
        "pct_person_detected": round(pct_person,  1),
        "pct_right_lower"    : round(pct_right,   1),
        "pct_left_lower"     : round(pct_left,    1),
        "pct_either_side"    : round(pct_either,  1),
        "dominant_side"      : dominant_side,
        "dominant_side_pct"  : round(dominant_pct, 1),
        "mean_lower_body_conf": round(mean_lower_all, 3),
        "longest_continuous_run_frames": max_run,
        "longest_continuous_run_sec"   : longest_run_sec,
        "conf_threshold_used": CONF_THR,
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rtmpose_scan.py <video>")
        sys.exit(1)

    vpath = sys.argv[1]
    result = scan_video(vpath)
    print(json.dumps(result, indent=2))
