"""
Wrapper to run Sports2D on a video file.

Usage:
  python run_sports2d.py <video_path>

What it does:
  1. Runs Sports2D pose estimation on the video
  2. Produces a video with skeleton overlay  (Sports2D_output/<name>_Sports2D.mp4)
  3. Saves joint angle CSV                  (Sports2D_output/<name>_angles.csv)

After this runs, use:
  python plot_joint_angle.py   (auto-finds the CSV)
"""

import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_sports2d.py <video_path>")
        print("Example: python run_sports2d.py demo_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    print(f"Running Sports2D on: {video_path}")
    print("First run downloads the pose model (~50 MB) — please wait...\n")

    try:
        from Sports2D import Sports2D

        # Config must be a NESTED dict matching Sports2D's DEFAULT_CONFIG structure.
        # Flat keys at the top level are silently ignored — always use the section names.
        Sports2D.process({
            "base": {
                "video_input": video_path,
                "nb_persons_to_detect": 1,
                # Avoid interactive click-to-select — pick the largest detected person
                "person_ordering_method": "largest_size",
                "show_realtime_results": False,   # no display window
                "save_vid": True,                 # skeleton overlay video
                "save_img": False,
                "save_pose": True,
                "calculate_angles": True,
                "save_angles": True,
                "result_dir": "Sports2D_output",
            },
            "pose": {
                "pose_model": "body_with_feet",
                "mode": "balanced",               # balanced = good speed/accuracy
                "det_frequency": 4,               # re-detect every 4 frames
            },
            "angles": {
                "joint_angles": [
                    "Right knee",
                    "Left knee",
                    "Right hip",
                    "Left hip",
                    "Right ankle",
                    "Left ankle",
                ],
                "segment_angles": [
                    "Right thigh",
                    "Left thigh",
                    "Trunk",
                ],
            },
            "post-processing": {
                "show_graphs": False,
                "save_graphs": False,
                "filter": True,
            },
        })

        print("\nDone! Check the Sports2D_output/ folder for:")
        print("  - *_Sports2D.mp4   — video with skeleton drawn on the person")
        print("  - *_angles.csv     — joint angles over time (one column per joint)")
        print("\nNext step: python plot_joint_angle.py")

    except ImportError:
        print("Sports2D not installed. Activate the venv and run: pip install sports2d")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
