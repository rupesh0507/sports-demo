"""
app.py — Video Analytics Frontend
Run: streamlit run app.py
"""

import os, sys
os.environ["MPLBACKEND"] = "Agg"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import glob, json, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import streamlit as st
import imageio_ffmpeg

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Univision — Visual Intelligence Platform for Sports",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

from trust_report import build_report, check_video_quality as _qc_check
import plot_joint_angle as _pja

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#FFFFFF!important}
[data-testid="stMain"]{background:#FFFFFF}
[data-testid="stSidebar"]{background:#F0FAF9;border-right:1px solid #B2DFDB}
[data-testid="stHeader"]{background:transparent}

.hero{background:linear-gradient(135deg,#00897B 0%,#00BFA5 60%,#B2DFDB 100%);
      border-radius:16px;padding:36px 40px 30px;margin-bottom:28px}
.hero-title{color:#fff;font-size:2.3rem;font-weight:800;letter-spacing:-0.5px;margin:0}
.hero-sub{color:rgba(255,255,255,.88);font-size:1.05rem;margin-top:8px}

.sec-hdr{color:#00695C;font-size:1.5rem;font-weight:700;
         border-left:4px solid #00BFA5;padding-left:12px;margin:0 0 14px 0}

/* Section h4/h5 headings in results area */
[data-testid="stMain"] h3{font-size:1.6rem!important;font-weight:800!important}
[data-testid="stMain"] h4{font-size:1.4rem!important;font-weight:700!important}
[data-testid="stMain"] h5{font-size:1.15rem!important;font-weight:700!important}

/* Trust table rendered as HTML */
.trust-table{width:100%;border-collapse:collapse;font-size:1.05rem}
.trust-table th{background:#E0F2F1;color:#00695C;padding:13px 16px;
                text-align:left;font-size:1.05rem;font-weight:700;
                border-bottom:2.5px solid #00BFA5;white-space:nowrap}
.trust-table td{padding:12px 16px;border-bottom:1px solid #E0F2F1;
                color:#1A2E2A;vertical-align:middle}
.trust-table tr:hover td{background:#F0FAF9}

div[data-testid="stButton"]>button[kind="primary"]{
  background:linear-gradient(135deg,#00897B,#00BFA5)!important;
  color:#fff!important;border:none;border-radius:10px;
  font-size:1.05rem;font-weight:700;padding:12px 0;width:100%;
  box-shadow:0 4px 14px rgba(0,137,123,.35)}
div[data-testid="stButton"]>button[kind="primary"]:hover{
  background:linear-gradient(135deg,#00695C,#00897B)!important;
  box-shadow:0 6px 18px rgba(0,137,123,.45);transform:translateY(-1px)}

.trust-caption{color:#78909C;font-size:1rem;font-style:italic;margin-top:8px}
[data-testid="stFileUploaderDropzone"]{
  background:#F0FAF9!important;border:2px dashed #00BFA5!important;border-radius:12px!important}
hr{border-color:#E0F2F1!important}
[data-testid="stExpander"]{border:1px solid #B2DFDB!important;border-radius:10px!important}

/* Fixed-height video player — consistent regardless of video aspect ratio.
   st.video renders a <video> inside [data-testid="stVideo"]. Capping height
   and using object-fit:contain keeps the full frame visible without cropping. */
[data-testid="stVideo"]{
  max-height:380px!important;overflow:hidden!important;
  background:#000;border-radius:10px}
[data-testid="stVideo"] video{
  max-height:380px!important;width:100%!important;
  object-fit:contain!important;background:#000}

/* Engine badge */
.engine-badge-s2d{background:#E0F7FA;color:#006064;padding:7px 20px;
                  border-radius:20px;font-weight:700;font-size:1.1rem;
                  display:inline-block;margin-bottom:20px}
.engine-badge-mp{background:#E8EAF6;color:#283593;padding:7px 20px;
                 border-radius:20px;font-weight:700;font-size:1.1rem;
                 display:inline-block;margin-bottom:20px}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sec(icon, title):
    st.markdown(f'<div class="sec-hdr">{icon}&nbsp; {title}</div>', unsafe_allow_html=True)

def confidence_cell_style(val):
    return {"HIGH":  "background-color:#E8F5E9;color:#1B5E20;font-weight:700",
            "MEDIUM":"background-color:#FFF3E0;color:#BF360C;font-weight:700",
            "LOW":   "background-color:#FFEBEE;color:#B71C1C;font-weight:700"}.get(str(val),"")


def render_trust_table(df):
    """Render the trust DataFrame as a styled HTML table."""
    _conf_style = {
        "HIGH":   "background:#E8F5E9;color:#1B5E20;font-weight:700;border-radius:6px;padding:4px 10px",
        "MEDIUM": "background:#FFF3E0;color:#BF360C;font-weight:700;border-radius:6px;padding:4px 10px",
        "LOW":    "background:#FFEBEE;color:#B71C1C;font-weight:700;border-radius:6px;padding:4px 10px",
    }
    html = ['<table class="trust-table">',
            '<thead><tr>',
            *[f'<th>{col}</th>' for col in df.columns],
            '</tr></thead><tbody>']
    for i, (_, row) in enumerate(df.iterrows()):
        row_bg = "" if i % 2 == 0 else 'style="background:#F8FFFE"'
        html.append(f'<tr {row_bg}>')
        for col in df.columns:
            val = row[col]
            if col == "Confidence":
                st_css = _conf_style.get(str(val), "")
                html.append(f'<td><span style="{st_css}">{val}</span></td>')
            else:
                html.append(f'<td>{val}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return "".join(html)

def find_fresh_chart(folder):
    hits = [f for f in glob.glob(os.path.join(folder,"*_plot.png"))
            if "debug" not in os.path.basename(f)]
    return max(hits, key=os.path.getmtime) if hits else None

def ensure_web_video(src):
    """Transcode to H.264 + faststart.  Returns path to web-compatible file."""
    if not src or not os.path.isfile(src):
        return None
    dst = src.replace(".mp4","_web.mp4")
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    try:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff,"-y","-i",src,
                        "-vcodec","libx264","-preset","fast","-crf","22",
                        "-pix_fmt","yuv420p","-movflags","+faststart",dst],
                       capture_output=True, timeout=180)
        return dst if os.path.isfile(dst) else src
    except Exception:
        return src

def read_video(path):
    with open(path,"rb") as f: return f.read()

def models_cached():
    d = os.path.join(os.path.expanduser("~"),".cache","rtmlib","hub","checkpoints")
    return len(glob.glob(os.path.join(d,"*.onnx"))) >= 2


# ── Session state ─────────────────────────────────────────────────────────────
_ss = st.session_state
for k,v in [("done",False),("stem",None),("engine_key","sports2d"),
             ("qc",None),("report",None),("chart",None),("preview",None),
             ("mp_trust",None),("mp_rep",None),("mp_chart",None)]:
    if k not in _ss: _ss[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
# Read currently selected engine from persisted widget state (safe on first run)
_cur_engine = st.session_state.get("engine_radio", "Sports2D (RTMPose-m)")
_sidebar_mp = "MediaPipe" in _cur_engine

with st.sidebar:
    st.markdown("""
<div style="background:linear-gradient(135deg,#00897B,#00BFA5);border-radius:10px;
            padding:14px 18px;margin-bottom:16px;text-align:center">
  <div style="color:#fff;font-size:1.4rem;font-weight:800;letter-spacing:0.5px">⚡ Univision</div>
  <div style="color:rgba(255,255,255,0.88);font-size:0.78rem;margin-top:3px">
    Visual Intelligence Platform for Sports</div>
</div>""", unsafe_allow_html=True)

    st.markdown("### 🧬 Pose Model")
    if _sidebar_mp:
        st.markdown("""
| | |
|---|---|
| **Pose model** | BlazePose Full |
| **Skeleton** | 33 landmarks (full body) |
| **Backend** | MediaPipe Tasks API v0.10+ |
| **Model file** | pose_landmarker_full.task |
""")
        if os.path.isfile(os.path.join(BASE_DIR, "pose_landmarker_full.task")):
            st.success("✅ BlazePose model cached")
        else:
            st.info("📥 First run downloads BlazePose model (~9 MB).")
    else:
        st.markdown("""
| | |
|---|---|
| **Pose model** | RTMPose-m |
| **Skeleton** | HALPE_26 (26 keypoints, body + feet) |
| **Backend** | OpenVINO CPU (auto-selected; no CUDA found) |
| **Sports2D** | v0.8.30 |
""")
        if models_cached():
            st.success("✅ Models cached — no download needed")
        else:
            st.info("📥 First run downloads models (~90 MB). Cached permanently after.")

    st.divider()
    st.markdown("### ⚙ Pipeline")
    if _sidebar_mp:
        for n,lbl in [("1","Quality gate — blur & brightness"),
                      ("2","BlazePose — 33 landmarks per frame"),
                      ("3","Joint angle chart (9 angles)"),
                      ("4","Trust report — confidence + visibility"),
                      ("5","Rep detection")]:
            st.markdown(f"**{n}.** {lbl}")
    else:
        for n,lbl in [("1","Quality gate — blur & brightness"),
                      ("2","AI pose estimation — RTMPose-m"),
                      ("3","Joint angle chart"),
                      ("4","Trust report — confidence scores")]:
            st.markdown(f"**{n}.** {lbl}")
    st.divider()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">⚡ Univision</div>
  <div class="hero-title" style="font-size:1.6rem;font-weight:600;margin-top:4px;
       letter-spacing:0.5px;color:rgba(255,255,255,0.95)">
      Visual Intelligence Platform for Sports</div>
  <div class="hero-sub" style="margin-top:12px;font-size:1.05rem">
      From raw video to trusted biomechanics — AI pose tracking, joint angles,
      rep detection &amp; confidence-scored metrics. No sensors. No lab.
  </div>
</div>""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
sec("📂","Upload Exercise Video")
uploaded = st.file_uploader(
    "Drag & drop or click to browse — MP4, AVI, MOV",
    type=["mp4","avi","mov"],
    help="Single-person exercise clip. Best with bright, clear footage.",
    label_visibility="visible",
)
if uploaded is None:
    st.markdown("""
    <div style="background:#F0FAF9;border:1px solid #B2DFDB;border-radius:14px;
                text-align:center;padding:50px 20px;color:#546E7A">
      <div style="font-size:3rem;margin-bottom:14px">🎥</div>
      <div style="font-size:1.15rem;font-weight:700;color:#00695C">Upload a video to begin</div>
      <div style="font-size:.9rem;margin-top:8px">
        All processing runs locally · Nothing leaves your machine</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Engine selector ───────────────────────────────────────────────────────────
st.divider()
sec("⚡", "Pose Engine")
engine_label = st.radio(
    "Select the pose estimation model:",
    ["Sports2D (RTMPose-m)", "MediaPipe (BlazePose)"],
    horizontal=True,
    key="engine_radio",
    help="Sports2D uses RTMPose-m (26 keypoints, HALPE_26). "
         "MediaPipe uses BlazePose Full (33 landmarks). "
         "Each engine writes to its own output folder — results never mix.",
)
engine_key = "sports2d" if "Sports2D" in engine_label else "mediapipe"

# ── Paths ─────────────────────────────────────────────────────────────────────
video_filename = uploaded.name
stem           = os.path.splitext(video_filename)[0]
video_path     = os.path.join(BASE_DIR, video_filename)
preview_web    = video_path.replace(".mp4","_web.mp4")

# Sports2D paths
s2d_folder     = os.path.join(BASE_DIR,"Sports2D_output",f"{stem}_Sports2D")
skeleton_mp4   = os.path.join(s2d_folder,f"{stem}_Sports2D.mp4")
skeleton_web   = skeleton_mp4.replace(".mp4","_web.mp4")
mot_path       = os.path.join(s2d_folder,f"{stem}_Sports2D_angles_person00.mot")
trust_json     = os.path.join(s2d_folder,"trust_report.json")

# MediaPipe paths
mp_folder      = os.path.join(BASE_DIR,"mediapipe_output", stem)
mp_skeleton    = os.path.join(mp_folder, f"{stem}_mp_skeleton.mp4")
mp_skeleton_web = mp_skeleton.replace(".mp4","_web.mp4")
mp_chart_png   = os.path.join(mp_folder, f"{stem}_mp_angles_plot.png")
mp_trust_json  = os.path.join(mp_folder, "mediapipe_trust_report.json")
mp_rep_json    = os.path.join(mp_folder, "mediapipe_rep_report.json")

# Save bytes to disk
with open(video_path,"wb") as fh:
    fh.write(uploaded.getbuffer())

# Clear session when video OR engine changes
if _ss.stem != stem or _ss.engine_key != engine_key:
    _ss.done=False; _ss.stem=stem; _ss.engine_key=engine_key
    _ss.qc=None; _ss.report=None; _ss.chart=None; _ss.preview=None
    _ss.mp_trust=None; _ss.mp_rep=None; _ss.mp_chart=None

# Show raw upload directly — no blocking transcode for the preview.
# Modern browsers play H.264/MP4 natively; the skeleton overlay is transcoded
# later (inside the pipeline, after analysis).
_ss.preview = video_path

# ── Preview + Run button ──────────────────────────────────────────────────────
st.divider()
sec("🎬","Uploaded Video")
col_vid, _, col_ctrl = st.columns([3, 0.2, 1.6])
with col_vid:
    st.video(read_video(_ss.preview))
with col_ctrl:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    _engine_note = ("RTMPose-m · Sports2D" if engine_key == "sports2d"
                    else "BlazePose · MediaPipe")
    st.markdown(
        f'<div style="font-size:.8rem;color:#78909C;margin-top:8px;text-align:center">'
        f'Engine: <b>{_engine_note}</b><br>Runs on CPU.</div>',
        unsafe_allow_html=True)

if not run and not _ss.done:
    st.stop()


# ── Pipeline ──────────────────────────────────────────────────────────────────
if run:
    st.divider()
    sec("⚙","Running Pipeline")

    # ── Sports2D path (unchanged) ─────────────────────────────────────────────
    if engine_key == "sports2d":
        with st.status("Analysing video…", expanded=True) as pstatus:

            st.write("**Step 1 / 5 — Checking video quality…**")
            qc = _qc_check(video_path)
            if qc is None:
                _msg = "OpenCV could not read the file. Try a standard H.264 MP4."
                print(f"[QC ERROR]  {video_filename}: could not open video")
                pstatus.update(label="❌ Could not open video", state="error", expanded=True)
                st.error(_msg)
                st.stop()
            if qc["grade"] == "BAD":
                issues = []
                if qc["blur_score"]       < qc["blur_threshold"]:
                    issues.append(f"too blurry (score {qc['blur_score']:.0f} < threshold {qc['blur_threshold']:.0f})")
                if qc["brightness_score"] < qc["brightness_threshold"]:
                    issues.append(f"too dark (brightness {qc['brightness_score']:.0f} < threshold {qc['brightness_threshold']:.0f})")
                _reason = " · ".join(issues)
                print(f"[QC FAIL]   {video_filename}: {_reason}")
                pstatus.update(label="❌ Video rejected", state="error", expanded=True)
                st.error(f"**Video rejected** — {_reason}")
                st.stop()
            print(f"[QC PASS]   {video_filename}: sharpness={qc['blur_score']:.0f}  brightness={qc['brightness_score']:.0f}")
            st.write(f"  ✅ Passed — sharpness {qc['blur_score']:.0f} · brightness {qc['brightness_score']:.0f}")

            st.write("**Step 2 / 5 — AI pose estimation…**")
            st.caption("Cached models." if models_cached() else "First run: downloading models (~90 MB)…")

            log_box  = st.empty()
            log_lines = []

            with subprocess.Popen(
                [sys.executable, os.path.join(BASE_DIR,"run_sports2d.py"), video_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=BASE_DIR, bufsize=1,
            ) as proc:
                for raw in proc.stdout:
                    line = raw.rstrip()
                    if line:
                        log_lines.append(line)
                        log_box.code("\n".join(log_lines[-12:]), language="text")
                proc.wait()
                returncode = proc.returncode

            if returncode != 0:
                print(f"[S2D ERROR] {video_filename}: Sports2D exited with code {returncode}")
                pstatus.update(label="❌ Pose estimation failed", state="error", expanded=True)
                st.error(f"Sports2D exited with code {returncode}. See log above.")
                st.stop()
            if not os.path.isfile(mot_path):
                print(f"[S2D ERROR] {video_filename}: expected output missing — {mot_path}")
                pstatus.update(label="❌ Output file missing", state="error", expanded=True)
                st.error(f"Expected: `{mot_path}`")
                st.stop()

            log_box.empty()
            st.write("  ✅ Pose estimation complete")

            ensure_web_video(skeleton_mp4)

            st.write("**Step 3 / 5 — Computing joint angle chart…**")
            chart_path = None
            try:
                _pja.plot_angle(mot_path)
                plt.close("all")
                chart_path = find_fresh_chart(s2d_folder)
                st.write(f"  ✅ Chart — {os.path.basename(chart_path) if chart_path else '—'}")
            except Exception as exc:
                st.warning(f"Chart issue: {exc}")

            st.write("**Step 4 / 5 — Building trust report…**")
            try:
                report = build_report(mot_path, video_path)
                os.makedirs(s2d_folder, exist_ok=True)
                with open(trust_json,"w") as fh:
                    json.dump(report, fh, indent=2)
                n_h = sum(1 for m in report["measurements"] if m["confidence_label"]=="HIGH")
                st.write(f"  ✅ Trust report — {n_h}/{len(report['measurements'])} joints HIGH confidence")
            except Exception as exc:
                pstatus.update(label="❌ Trust report failed", state="error")
                st.error(f"Error: {exc}"); st.stop()

            st.write("**Step 5 / 5 — Detecting reps…**")
            _rep_proc = subprocess.run(
                [sys.executable, os.path.join(BASE_DIR, "detect_reps.py"), video_path],
                capture_output=True, text=True, cwd=BASE_DIR, timeout=60,
            )
            if _rep_proc.returncode == 0:
                st.write("  ✅ Rep detection complete")
            else:
                st.warning("Rep detection had an issue — rep section may be empty.")

            pstatus.update(label="✅ Analysis complete!", state="complete", expanded=False)

        _ss.done=True; _ss.qc=qc; _ss.report=report; _ss.chart=chart_path

    # ── MediaPipe path ────────────────────────────────────────────────────────
    else:
        with st.status("Analysing video…", expanded=True) as pstatus:

            st.write("**Step 1 / 2 — Checking video quality…**")
            qc = _qc_check(video_path)
            if qc is None:
                _msg = "OpenCV could not read the file. Try a standard H.264 MP4."
                print(f"[QC ERROR]  {video_filename}: could not open video")
                pstatus.update(label="❌ Could not open video", state="error", expanded=True)
                st.error(_msg)
                st.stop()
            if qc["grade"] == "BAD":
                issues = []
                if qc["blur_score"]       < qc["blur_threshold"]:
                    issues.append(f"too blurry (score {qc['blur_score']:.0f} < threshold {qc['blur_threshold']:.0f})")
                if qc["brightness_score"] < qc["brightness_threshold"]:
                    issues.append(f"too dark (brightness {qc['brightness_score']:.0f} < threshold {qc['brightness_threshold']:.0f})")
                _reason = " · ".join(issues)
                print(f"[QC FAIL]   {video_filename}: {_reason}")
                pstatus.update(label="❌ Video rejected", state="error", expanded=True)
                st.error(f"**Video rejected** — {_reason}")
                st.stop()
            print(f"[QC PASS]   {video_filename}: sharpness={qc['blur_score']:.0f}  brightness={qc['brightness_score']:.0f}")
            st.write(f"  ✅ Passed — sharpness {qc['blur_score']:.0f} · brightness {qc['brightness_score']:.0f}")

            st.write("**Step 2 / 2 — BlazePose analysis (angles · chart · trust · reps)…**")

            log_box   = st.empty()
            log_lines = []

            with subprocess.Popen(
                [sys.executable, os.path.join(BASE_DIR,"mediapipe_pipeline.py"), video_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=BASE_DIR, bufsize=1,
            ) as proc:
                for raw in proc.stdout:
                    line = raw.rstrip()
                    if line:
                        log_lines.append(line)
                        log_box.code("\n".join(log_lines[-12:]), language="text")
                proc.wait()
                returncode = proc.returncode

            if returncode != 0:
                print(f"[MP ERROR]  {video_filename}: mediapipe_pipeline.py exited with code {returncode}")
                pstatus.update(label="❌ MediaPipe analysis failed", state="error", expanded=True)
                st.error(f"mediapipe_pipeline.py exited with code {returncode}. See log above.")
                st.stop()

            if not os.path.isfile(mp_trust_json):
                print(f"[MP ERROR]  {video_filename}: expected output missing — {mp_trust_json}")
                pstatus.update(label="❌ Output file missing", state="error", expanded=True)
                st.error(f"Expected: `{mp_trust_json}`")
                st.stop()

            log_box.empty()

            # Load results from the files mediapipe_pipeline.py wrote
            with open(mp_trust_json) as fh:
                mp_trust = json.load(fh)
            mp_rep = None
            if os.path.isfile(mp_rep_json):
                with open(mp_rep_json) as fh:
                    mp_rep = json.load(fh)

            mp_chart_path = mp_chart_png if os.path.isfile(mp_chart_png) else None

            # Transcode skeleton video for browser
            ensure_web_video(mp_skeleton)

            n_h = sum(1 for m in mp_trust.get("measurements",[]) if m["confidence_label"]=="HIGH")
            n_t = len(mp_trust.get("measurements",[]))
            st.write(f"  ✅ Analysis complete — {n_h}/{n_t} joints HIGH confidence")

            pstatus.update(label="✅ Analysis complete!", state="complete", expanded=False)

        _ss.done=True; _ss.qc=qc
        _ss.mp_trust=mp_trust; _ss.mp_rep=mp_rep; _ss.mp_chart=mp_chart_path


# ── Results ────────────────────────────────────────────────────────────────────
if not _ss.done:
    st.stop()

qc = _ss.qc

# Abort if no results for the selected engine
if _ss.engine_key == "sports2d" and not _ss.report:
    st.stop()
if _ss.engine_key == "mediapipe" and not _ss.mp_trust:
    st.stop()

st.divider()
sec("📋","Analysis Results")

# ── Engine badge ───────────────────────────────────────────────────────────────
if _ss.engine_key == "sports2d":
    st.markdown(
        '<span class="engine-badge-s2d">🏃 Results from: Sports2D (RTMPose-m · HALPE_26)</span>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<span class="engine-badge-mp">🤖 Results from: MediaPipe (BlazePose Full · 33 landmarks)</span>',
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SPORTS2D RESULTS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
if _ss.engine_key == "sports2d":
    report     = _ss.report
    chart_path = _ss.chart

    # 1. Quality gate
    st.markdown("#### 🔍 Quality Gate")
    if qc["grade"] == "GOOD":
        st.success(f"✅ **GOOD** &nbsp;|&nbsp; Sharpness: **{qc['blur_score']:.0f}** &nbsp;|&nbsp; Brightness: **{qc['brightness_score']:.0f}**")
    else:
        st.error(f"❌ **REJECTED** &nbsp;|&nbsp; Sharpness: **{qc['blur_score']:.0f}** &nbsp;|&nbsp; Brightness: **{qc['brightness_score']:.0f}**")
    st.divider()

    # 2. Skeleton overlay video
    st.markdown("#### 🦴 Body Tracking — Skeleton Overlay")
    play_vid = skeleton_web if os.path.isfile(skeleton_web) else \
               (skeleton_mp4 if os.path.isfile(skeleton_mp4) else None)
    if play_vid:
        st.video(read_video(play_vid))
        st.caption("RTMPose-m (HALPE_26, 26 keypoints) skeleton overlaid on every frame.")
    else:
        st.warning(f"Skeleton video not found: `{skeleton_mp4}`")
    st.divider()

    # 3. Joint angle chart
    st.markdown("#### 📈 Joint Angle Over Time")
    if chart_path and os.path.isfile(chart_path):
        st.image(chart_path, use_container_width=True)
        st.caption("Most stable joint plotted. Gaps = tracking confidence too low (excluded, not guessed).")
    else:
        st.warning("Chart not found.")
    st.divider()

    # 4. Trust report
    st.markdown("#### 🔒 Trust Report — Every Metric with Confidence")
    measurements = report.get("measurements",[])
    meta         = report.get("report_metadata",{})

    if measurements:
        rows = []
        for m in measurements:
            rows.append({
                "Metric":       m["display_name"],
                "Value (°)":    f"{m['value_degrees']:.1f}",
                "Confidence":   m["confidence_label"],
                "Score":        f"{m['confidence_score']:.0%}",
                "Stability":    f"{m['stability_score']:.0%}",
                "Valid frames": f"{m['frames_in_range']}/{m['frames_analyzed']}",
                "Quality grade":meta.get("video_quality",{}).get("grade","—"),
                "Type":         m.get("metric_type", "—"),
                "Produced by":  m.get("produced_by", "—"),
                "Keypoints":    " · ".join(m.get("metric_keypoints", [])) or "—",
            })
        df = pd.DataFrame(rows)
        st.markdown(render_trust_table(df), unsafe_allow_html=True)
        st.markdown(
            '<div class="trust-caption">Every measurement carries a confidence score '
            'and quality grade — no black-box numbers.</div>',
            unsafe_allow_html=True)
        kp_note = meta.get("model_keypoint_confidence_availability", "")
        if kp_note:
            with st.expander("ℹ️ Per-keypoint model confidence — known limitation"):
                st.write(kp_note)
    else:
        st.warning("No measurements in trust report.")

    # 5. Rep detection
    rep_json = os.path.join(s2d_folder, "rep_report.json")
    if os.path.isfile(rep_json):
        try:
            with open(rep_json) as _f:
                rr = json.load(_f)
        except Exception:
            rr = None

        if rr:
            st.divider()
            sec("🔄", "Rep Detection")

            rep_count  = rr.get("rep_count", 0)
            conf_lbl   = rr.get("rep_count_confidence_label", "—")
            conf_score = rr.get("rep_count_confidence", 0)
            tempo_s    = rr.get("tempo_seconds_per_rep")
            tempo_rpm  = rr.get("tempo_reps_per_minute")
            rep_peaks  = rr.get("per_rep_peak_degrees", [])
            signal_col = rr.get("signal_used", "—")
            rep_notes  = rr.get("notes", [])

            _badge_fg = {"HIGH":"#1B5E20","MEDIUM":"#BF360C","LOW":"#B71C1C"}.get(conf_lbl,"#546E7A")
            _badge_bg = {"HIGH":"#E8F5E9","MEDIUM":"#FFF3E0","LOW":"#FFEBEE"}.get(conf_lbl,"#F5F5F5")

            col_r, col_t, col_p = st.columns([1.3, 1.5, 2.5])
            with col_r:
                st.markdown("#### Reps detected")
                st.markdown(
                    f'<div style="font-size:3.8rem;font-weight:800;color:#00695C;'
                    f'line-height:1;margin-bottom:12px">{rep_count}</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<span style="background:{_badge_bg};color:{_badge_fg};padding:7px 20px;'
                    f'border-radius:20px;font-weight:700;font-size:1.1rem;">'
                    f'{conf_lbl} &nbsp;·&nbsp; {conf_score:.0%}</span>',
                    unsafe_allow_html=True)
            with col_t:
                if tempo_s and tempo_rpm:
                    st.metric("Tempo", f"{tempo_rpm:.0f} reps / min")
                    st.markdown(
                        f'<div style="font-size:1rem;color:#546E7A;margin-top:4px">'
                        f'Avg <b>{tempo_s:.1f} s</b> per rep</div>',
                        unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:1rem;color:#546E7A;margin-top:8px">'
                    f'Signal: <b>{signal_col}</b></div>',
                    unsafe_allow_html=True)
            with col_p:
                if rep_peaks:
                    st.markdown("#### Peak angle per rep (°)")
                    _bar_df = pd.DataFrame({
                        "Rep":      [f"Rep {i+1}" for i in range(len(rep_peaks))],
                        "Peak (°)": [round(v, 1) for v in rep_peaks],
                    }).set_index("Rep")
                    st.bar_chart(_bar_df, height=200, color="#00897B")
            if rep_notes:
                st.markdown("<br>", unsafe_allow_html=True)
                for _note in rep_notes:
                    st.warning(_note)

# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
else:
    mp_trust   = _ss.mp_trust
    mp_rep     = _ss.mp_rep
    mp_chart_p = _ss.mp_chart

    # 1. Quality gate
    st.markdown("#### 🔍 Quality Gate")
    if qc["grade"] == "GOOD":
        st.success(f"✅ **GOOD** &nbsp;|&nbsp; Sharpness: **{qc['blur_score']:.0f}** &nbsp;|&nbsp; Brightness: **{qc['brightness_score']:.0f}**")
    else:
        st.error(f"❌ **REJECTED** &nbsp;|&nbsp; Sharpness: **{qc['blur_score']:.0f}** &nbsp;|&nbsp; Brightness: **{qc['brightness_score']:.0f}**")
    st.divider()

    # 2. Skeleton overlay video
    st.markdown("#### 🦴 Body Tracking — Skeleton Overlay")
    _mp_play = mp_skeleton_web if os.path.isfile(mp_skeleton_web) else \
               (mp_skeleton if os.path.isfile(mp_skeleton) else None)
    if _mp_play:
        st.video(read_video(_mp_play))
        st.caption("BlazePose Full (33 landmarks) skeleton overlaid on every frame.")
    else:
        st.warning(f"Skeleton video not found: `{mp_skeleton}`")
    st.divider()

    # 3. Joint angle chart
    st.markdown("#### 📈 Joint Angles Over Time")
    if mp_chart_p and os.path.isfile(mp_chart_p):
        st.image(mp_chart_p, use_container_width=True)
        st.caption(
            "9 angles: right/left knee (flexion), right/left hip (flexion), "
            "right/left ankle (shank segment), right/left thigh (segment), trunk. "
            "Gaps = frames outside the plausible anatomical range.")
    else:
        st.warning("Angle chart not found.")
    st.divider()

    # 4. Trust report  (includes native BlazePose visibility column)
    st.markdown("#### 🔒 Trust Report — Confidence & Native BlazePose Visibility")
    mp_measurements = mp_trust.get("measurements", [])
    mp_meta         = mp_trust.get("report_metadata", {})

    if mp_measurements:
        rows = []
        for m in mp_measurements:
            rows.append({
                "Metric":        m["display_name"],
                "Value (°)":     f"{m['value_degrees']:.1f}",
                "Confidence":    m["confidence_label"],
                "Score":         f"{m['confidence_score']:.0%}",
                "Stability":     f"{m['stability_score']:.0%}",
                "Valid frames":  f"{m['frames_in_range']}/{m['frames_analyzed']}",
                "MP Visibility": f"{m['mediapipe_mean_visibility']:.3f}",
            })
        df = pd.DataFrame(rows)
        st.markdown(render_trust_table(df), unsafe_allow_html=True)
        st.markdown(
            '<div class="trust-caption">'
            '<b>Confidence</b> = √(stability × plausibility) — derived from signal smoothness and anatomical plausibility. '
            '<b>MP Visibility</b> = mean native BlazePose keypoint confidence for the landmarks that define this angle '
            '(min over contributing landmarks per frame, averaged across all frames). '
            'Both scores are independent; HIGH confidence with low visibility '
            'means the signal is smooth but the keypoint was partially occluded.</div>',
            unsafe_allow_html=True)
    else:
        st.warning("No measurements in trust report.")

    # 5. Rep detection
    if mp_rep:
        st.divider()
        sec("🔄", "Rep Detection")

        rep_count  = mp_rep.get("rep_count", 0)
        conf_lbl   = mp_rep.get("rep_count_confidence_label", "—")
        conf_score = mp_rep.get("rep_count_confidence", 0)
        tempo_s    = mp_rep.get("tempo_seconds_per_rep")
        tempo_rpm  = mp_rep.get("tempo_reps_per_minute")
        rep_peaks  = mp_rep.get("per_rep_peak_degrees", [])
        signal_col = mp_rep.get("signal_used", "—")
        rep_notes  = mp_rep.get("notes", [])

        _badge_fg = {"HIGH":"#1B5E20","MEDIUM":"#BF360C","LOW":"#B71C1C"}.get(conf_lbl,"#546E7A")
        _badge_bg = {"HIGH":"#E8F5E9","MEDIUM":"#FFF3E0","LOW":"#FFEBEE"}.get(conf_lbl,"#F5F5F5")

        col_r, col_t, col_p = st.columns([1.3, 1.5, 2.5])
        with col_r:
            st.markdown("#### Reps detected")
            st.markdown(
                f'<div style="font-size:3.8rem;font-weight:800;color:#00695C;'
                f'line-height:1;margin-bottom:12px">{rep_count}</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<span style="background:{_badge_bg};color:{_badge_fg};padding:7px 20px;'
                f'border-radius:20px;font-weight:700;font-size:1.1rem;">'
                f'{conf_lbl} &nbsp;·&nbsp; {conf_score:.0%}</span>',
                unsafe_allow_html=True)
        with col_t:
            if tempo_s and tempo_rpm:
                st.metric("Tempo", f"{tempo_rpm:.0f} reps / min")
                st.markdown(
                    f'<div style="font-size:1rem;color:#546E7A;margin-top:4px">'
                    f'Avg <b>{tempo_s:.1f} s</b> per rep</div>',
                    unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:1rem;color:#546E7A;margin-top:8px">'
                f'Signal: <b>{signal_col}</b></div>',
                unsafe_allow_html=True)
        with col_p:
            if rep_peaks:
                st.markdown("#### Peak angle per rep (°)")
                _bar_df = pd.DataFrame({
                    "Rep":      [f"Rep {i+1}" for i in range(len(rep_peaks))],
                    "Peak (°)": [round(v, 1) for v in rep_peaks],
                }).set_index("Rep")
                st.bar_chart(_bar_df, height=200, color="#00897B")
        if rep_notes:
            st.markdown("<br>", unsafe_allow_html=True)
            for _note in rep_notes:
                st.warning(_note)
