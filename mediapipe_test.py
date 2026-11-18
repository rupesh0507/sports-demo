"""
mediapipe_test.py
-----------------
Learning script — uses MediaPipe Pose (BlazePose) to:
  1. Process every frame of a video and draw the 33-landmark skeleton.
  2. Save the annotated video to mediapipe_test_output/<name>_mediapipe.mp4
  3. Print a landmark table for one frame in the middle of the video.

Uses the NEW MediaPipe Tasks API (v0.10+).
The BlazePose model file (pose_landmarker_full.task, ~9 MB) is auto-downloaded
from Google's public bucket and cached in the sports-demo folder.

This script is completely standalone.
It does NOT import or modify any other file in this project.

Usage:
  python mediapipe_test.py squat_video.mp4
  python mediapipe_test.py demo_video.mp4
"""

import sys
import os
import urllib.request
import cv2
import mediapipe as mp

# ── MediaPipe Tasks API (v0.10+) ───────────────────────────────────────────────
# The new API lives entirely under mp.tasks.
# mp.tasks.vision.PoseLandmarker      — the model runner
# mp.tasks.vision.drawing_utils       — draws dots and lines on frames
# mp.tasks.vision.drawing_styles      — default teal/orange colour scheme
# mp.tasks.vision.PoseLandmark        — enum with 33 landmark names
# mp.tasks.vision.PoseLandmarksConnections — which dots to connect with lines
tv            = mp.tasks.vision
drawing_utils = tv.drawing_utils
drawing_styles = tv.drawing_styles
PoseLandmarker        = tv.PoseLandmarker
PoseLandmarkerOptions = tv.PoseLandmarkerOptions
PoseLandmarksConnections = tv.PoseLandmarksConnections
RunningMode   = tv.RunningMode

BaseOptions = mp.tasks.BaseOptions

# Human-readable names for all 33 BlazePose landmarks.
LANDMARK_NAMES = [lm.name for lm in tv.PoseLandmark]

# ── Model ──────────────────────────────────────────────────────────────────────
# Google hosts three variants: lite (~3 MB), full (~9 MB), heavy (~27 MB).
# "full" is a good balance of accuracy and speed.
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
MODEL_FILE = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_model():
    """Download the BlazePose model file if not already present."""
    if os.path.isfile(MODEL_FILE):
        return
    print(f"Downloading BlazePose model (~9 MB) …")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(MODEL_FILE, "wb") as f:
        f.write(r.read())
    print(f"  Saved: {MODEL_FILE}")


def make_output_path(video_path):
    """Return (output_folder, output_video_path) for the given input video."""
    stem   = os.path.splitext(os.path.basename(video_path))[0]
    # Output goes into mediapipe_test_output/ — never into Sports2D_output/.
    folder = os.path.join(os.path.dirname(os.path.abspath(video_path)),
                          "mediapipe_test_output")
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"{stem}_mediapipe.mp4")
    return folder, out_path


def draw_pose_on_frame(frame_bgr, landmarks):
    """
    Draw the 33-point skeleton on a BGR frame (in-place copy).

    landmark_list is a flat list of NormalizedLandmark objects (one per person).
    draw_landmarks() takes:
      image            — numpy array (BGR, uint8)
      landmark_list    — flat list of NormalizedLandmark
      connections      — which pairs of landmarks to connect with lines
      landmark/connection drawing specs — colour and thickness
    """
    annotated = frame_bgr.copy()
    drawing_utils.draw_landmarks(
        annotated,
        landmarks,
        # PoseLandmarksConnections.POSE_LANDMARKS is the set of (i, j) pairs
        # that MediaPipe considers anatomically connected (shoulder-elbow, etc.)
        PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
    )
    return annotated


def print_landmark_table(landmarks, frame_index):
    """
    Print a table of all 33 BlazePose landmarks for one frame.

    MediaPipe normalises all coordinates to the frame dimensions:
      x, y  — fraction of frame width / height
               0.0 = left / top edge,  1.0 = right / bottom edge
      z     — depth relative to the hip midpoint, in the same scale as x.
               Negative = closer to the camera.
    visibility — per-keypoint confidence (0.0 unseen → 1.0 certain).
                 This is MediaPipe's own term for how likely the model thinks
                 the landmark is both visible and correctly placed.
    """
    if landmarks is None:
        print(f"[Frame {frame_index}] No person detected — cannot print landmark table.")
        return

    COL = "{:<4}  {:<32}  {:>8}  {:>8}  {:>8}  {:>10}"

    print()
    print(f"  BlazePose landmarks — frame {frame_index}")
    print("  " + "─" * 76)
    print(COL.format("#", "Landmark name", "x", "y", "z", "visibility"))
    print("  " + "─" * 76)
    for i, name in enumerate(LANDMARK_NAMES):
        lm = landmarks[i]
        print(COL.format(
            i,
            name,
            f"{lm.x:.4f}",
            f"{lm.y:.4f}",
            f"{lm.z:.4f}",
            f"{lm.visibility:.4f}",
        ))
    print("  " + "─" * 76)
    print()
    print("  NOTE: x and y are normalised 0–1 (fraction of frame width / height).")
    print("        z is relative depth in the same scale (negative = toward camera).")
    print("        visibility is MediaPipe's per-keypoint confidence"
          " (0 = unseen, 1 = certain).")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python mediapipe_test.py <video.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"ERROR: file not found — {video_path}")
        sys.exit(1)

    ensure_model()

    _, out_path = make_output_path(video_path)

    # ── Open input video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nInput  : {video_path}")
    print(f"Frames : {total_frames}  |  {fps:.1f} fps  |  {width}×{height}")
    print(f"Output : {out_path}")
    print()

    # The frame we will print the full landmark table for.
    table_frame_index = total_frames // 2

    # ── Open output video ───────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # ── Build PoseLandmarker options ────────────────────────────────────────────
    #
    # RunningMode.VIDEO  — processes frames in order using temporal filtering.
    #   (As opposed to IMAGE mode which treats each frame independently.)
    #   In VIDEO mode you call detect_for_video(mp_image, timestamp_ms).
    #
    # num_poses=1        — we only need the one athlete in the frame.
    # min_pose_detection_confidence — detector must be at least 50% sure a
    #   person is there before it runs the full landmark model.
    # min_tracking_confidence — once tracking starts, drop below this and it
    #   re-runs the full detector on that frame.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Process frames ──────────────────────────────────────────────────────────
    table_landmarks = None   # will hold landmarks from the middle frame

    print("Processing frames …")
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # MediaPipe Image wrapper expects RGB.
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # mp.Image wraps the numpy array without copying.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp must increase monotonically; use milliseconds.
            timestamp_ms = int(frame_index * (1000.0 / fps))

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # result.pose_landmarks is a list of one list-per-person.
            # For num_poses=1 we only look at index 0 when someone is detected.
            person_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            # Capture the middle frame's landmarks for the table.
            if frame_index == table_frame_index:
                table_landmarks = person_landmarks

            # Draw skeleton and frame counter.
            if person_landmarks is not None:
                annotated = draw_pose_on_frame(frame_bgr, person_landmarks)
            else:
                annotated = frame_bgr.copy()

            cv2.putText(
                annotated,
                f"frame {frame_index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            frame_index += 1

            if frame_index % 30 == 0:
                print(f"  … {frame_index}/{total_frames} frames processed")

    cap.release()
    writer.release()

    print(f"\nDone. {frame_index} frames written to:\n  {out_path}\n")

    # ── Print landmark table for the middle frame ───────────────────────────────
    print_landmark_table(table_landmarks, table_frame_index)


if __name__ == "__main__":
    main()
