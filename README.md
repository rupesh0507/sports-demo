# ⚡ Univision — Visual Intelligence Platform for Sports

> Turn raw sports video into trusted biomechanics intelligence — AI pose tracking, joint angles, rep detection, and confidence-scored metrics. No sensors. No lab. No special hardware.

Univision processes a plain video of an athlete and delivers:

- **AI skeleton overlay**: pose tracked on every frame
- **Joint angle time series**: smoothed, normalised angles for every major joint
- **Rep detection**: automatic counting with tempo and per-rep depth
- **Confidence-scored trust report**: every metric carries stability, plausibility and quality scores
- **Dual-engine UI**: choose between **Sports2D (RTMPose-m)** and **MediaPipe (BlazePose)**

> **Note:** This repository holds **source code only**. No videos, frames, images, model weights or generated outputs are committed. Use your own video files (see [Sample videos](#sample-videos)). The pose models download automatically on first run.

---

## Quick start

### 1. Prerequisites

- **Python 3.10** (tested with 3.10.11). Newer Python versions may not be supported by `sports2d` or `mediapipe`.
- Git
- Internet access on the first run, to download the pose models (~135 MB for RTMPose, ~9 MB for BlazePose)

### 2. Clone and install

**Windows (PowerShell)**

```powershell
git clone https://github.com/rupesh0507/sports-demo.git
cd sports-demo

py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
git clone https://github.com/rupesh0507/sports-demo.git
cd sports-demo

python3.10 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the web app (recommended)

```bash
streamlit run app.py
```

Open **http://localhost:8501**, then:

1. Drag and drop an MP4 video of an athlete exercising.
2. Choose a pose engine: **Sports2D (RTMPose-m)** or **MediaPipe (BlazePose)**.
3. Click **▶ Run Analysis**. Logs stream live, and a video that fails the quality gate is rejected with the reason.
4. Review the skeleton video, angle chart, trust table and rep detection.

### 4. Or run from the command line

```bash
# Full Sports2D pipeline: quality gate → pose → angle chart → trust report
python run_all.py path/to/your_video.mp4

# Rep detection (run after run_all.py)
python detect_reps.py path/to/your_video.mp4

# Full MediaPipe pipeline (all stages in one script)
python mediapipe_pipeline.py path/to/your_video.mp4
```

Results are written to `Sports2D_output/<name>_Sports2D/` or `mediapipe_output/<name>/`. Both folders are git-ignored.

---

## Sample videos

Videos are not stored in the repo. You can:

- Use any MP4 of one person exercising (squats, jumping jacks, lunges, sprints and similar). Keep the full body in frame and the camera still, with good lighting.
- Generate two synthetic quality-gate test clips:

  ```bash
  python create_test_videos.py      # writes test_good.mp4 and test_bad.mp4
  python check_video_quality.py test_good.mp4   # → GOOD
  python check_video_quality.py test_bad.mp4    # → BAD
  ```

Some research scripts expect a named input (for example `run_sideview.mp4`, a side-view sprint clip, for the hip-extension scripts). Put your own clip at that path before running them.

---

## Pose engines

The two pipelines are fully independent. Each writes to its own output folder and never reads from the other.

| | Sports2D (RTMPose-m) | MediaPipe (BlazePose) |
|---|---|---|
| **Model** | RTMPose-m (+ YOLOX-m detector) | BlazePose Full |
| **Keypoints** | 26 (HALPE_26) | 33 |
| **Backend** | OpenVINO / ONNX on CPU | MediaPipe Tasks API |
| **Output folder** | `Sports2D_output/<name>_Sports2D/` | `mediapipe_output/<name>/` |
| **Model download** | ~135 MB, automatic | ~9 MB (`pose_landmarker_full.task`), automatic |
| **Angle format** | `.mot` (OpenSim, tab-separated) | `.csv` |
| **Trust report** | `trust_report.json` | `mediapipe_trust_report.json` |
| **Rep report** | `rep_report.json` | `mediapipe_rep_report.json` |

## Architecture

```
                 ┌──────────── Video uploaded (app.py) ────────────┐
                 ▼                                                 ▼
          [Quality Gate]                                    [Quality Gate]
      check_video_quality.py                            (built into pipeline)
                 ▼                                                 ▼
        [Pose Estimation]                                [Pose Estimation]
         run_sports2d.py                              mediapipe_pipeline.py
       RTMPose-m / HALPE_26                          BlazePose Full / 33 lm
                 ▼                                                 ▼
       [Joint Angle Chart]  plot_joint_angle.py             (built in)
                 ▼                                                 ▼
         [Trust Report]     trust_report.py                 (built in)
                 ▼                                                 ▼
         [Rep Detection]    detect_reps.py                  (built in)
                 ▼                                                 ▼
     Sports2D_output/<name>_Sports2D/                mediapipe_output/<name>/
```

---

## Project structure

```
sports-demo/
├── app.py                        Streamlit UI: dual-engine frontend (entry point)
├── run_all.py                    CLI: Sports2D four-step pipeline
├── requirements.txt              Python dependencies
│
├── Sports2D pipeline
│   ├── check_video_quality.py    Blur + brightness quality gate (local-sharpness fallback)
│   ├── run_sports2d.py           RTMPose-m pose estimation wrapper
│   ├── run_sports2d_multi.py     Multi-person Sports2D run
│   ├── plot_joint_angle.py       Joint angle chart generator
│   ├── trust_report.py           Confidence scoring (stability × plausibility)
│   ├── detect_reps.py            Adaptive peak-detection rep counter
│   └── demo_angles.csv           Small example angle file for plot_joint_angle.py
│
├── MediaPipe pipeline
│   ├── mediapipe_pipeline.py     BlazePose full analytics pipeline
│   └── mediapipe_test.py         Minimal BlazePose skeleton demo
│
├── Robustness layer
│   └── robust_pipeline.py        7-stage input validation + pose report
│
├── Sprint / screening research
│   ├── hip_extension_analysis.py        Hip extension at toe-off (MediaPipe)
│   ├── hip_extension_validation.py      Cross-engine + stride-consistency validation
│   ├── hip_extension_validation_2d.py   Same-formula 2D comparison (BlazePose vs RTMPose)
│   ├── screening_scaffold.py            Per-metric QUALIFY / DISQUALIFY / NEEDS_REVIEW signals
│   ├── screening_thresholds.json        Provisional thresholds (not validated)
│   └── sprint1_testset/
│       ├── detect_scan.py               MediaPipe detection-rate scan
│       └── rtmpose_scan.py              RTMPose per-frame detection scan
│
├── create_test_videos.py         Generates synthetic good/bad quality-gate clips
├── sport_demo_analysis.md        Analysis notes
└── sports_demos_analysis.md      Analysis notes
```

Git-ignored (generated locally): `venv/`, `Sports2D_output/`, `mediapipe_output/`, `robust_output/`, `hip_extension_output/`, `hip_extension_validation/`, `screening_output/`, `multiperson_output/`, model files (`*.task`), and all video/image files.

---

## Module reference

| Script | Usage | What it does |
|---|---|---|
| `check_video_quality.py` | `python check_video_quality.py video.mp4` | Samples frames and rejects blurry (Laplacian variance ≤ 80) or dark (mean brightness ≤ 40) footage before the AI runs |
| `run_sports2d.py` | `python run_sports2d.py video.mp4` | Sports2D v0.8.30: YOLOX-m detection → RTMPose-m 26 keypoints → interpolation, Hampel filter, 6 Hz Butterworth, pixel→metre conversion |
| `plot_joint_angle.py` | `python plot_joint_angle.py video.mp4 ["right hip"]` | Picks the most informative joint, applies 360° wrap correction, masks tracking failures and saves a PNG |
| `trust_report.py` | `python trust_report.py video.mp4` | `confidence = √(stability × plausibility)`. HIGH ≥ 80%, MEDIUM 55–79%, LOW < 55% |
| `detect_reps.py` | `python detect_reps.py video.mp4` | Savitzky-Golay smoothing, then `find_peaks` with adaptive height, prominence and a 0.5 s minimum gap |
| `mediapipe_pipeline.py` | `python mediapipe_pipeline.py video.mp4` | Quality gate → BlazePose → 9 angles → post-processing → trust + native visibility → reps |
| `robust_pipeline.py` | `python robust_pipeline.py video.mp4` | Inspect → normalise rotation/codec (ffmpeg) → quality gate → BlazePose → subject select → framing check → confidence harvest |
| `run_sports2d_multi.py` | `python run_sports2d_multi.py video.mp4` | Multi-person Sports2D analysis |
| `hip_extension_analysis.py` | `python hip_extension_analysis.py` | Hip extension at toe-off on `run_sideview.mp4` → `hip_extension_output/` |
| `hip_extension_validation*.py` | `python hip_extension_validation.py` | Validates the hip-extension results against Sports2D (run after the analysis script) |
| `screening_scaffold.py` | `python screening_scaffold.py [clip_stem]` | Applies `screening_thresholds.json` to the analysis outputs |

### Rep detection: how it works

A peak in the smoothed angle signal counts as a rep only when it clears **all** of these:

- Height above `mean + 15% × (max − mean)`
- Prominence ≥ 25% of the signal range
- At least 0.5 s since the previous peak

Rep confidence is `√(regularity × valid_fraction)`. Isometric holds, or exercises where the dominant joint doesn't clearly flex and extend, return 0 reps.

### Plausible angle ranges

| Joint / segment | Range | Notes |
|---|---|---|
| Right / left knee | −20° to 160° | 0° = straight |
| Right / left hip | −60° to 160° | Negative = hyperextension |
| Right / left ankle (shank) | −150° to 30° | About −90° when standing |
| Right / left thigh | −150° to 10° | About −90° when standing |
| Trunk | 40° to 150° | About 90° when upright |

---

## Troubleshooting

- **`numpy` errors with Sports2D**: keep `numpy<2.0`, as pinned in `requirements.txt`.
- **`mp.solutions` not found**: MediaPipe 0.10+ removed the legacy API. This code uses the Tasks API (`mp.tasks.vision.PoseLandmarker`).
- **The skeleton video doesn't play in the browser**: `imageio-ffmpeg` provides the ffmpeg used to transcode results to browser-friendly H.264. Make sure it is installed.
- **The first run is slow**: the models download on first use and are cached after that.
- **opencv 4.11**: excluded because of a known incompatibility. Use 4.10.x.

---

## References

- **RTMPose**: Jiang et al. (2023). *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose.* arXiv:2303.07399
- **Sports2D**: Pagnon, D. (2024). https://github.com/davidpagnon/Sports2D
- **Pose2Sim**: Pagnon, D. et al. (2022). *Journal of Open Source Software.*
- **BlazePose / MediaPipe**: Bazarevsky et al. (2020). arXiv:2006.10204. https://developers.google.com/mediapipe
