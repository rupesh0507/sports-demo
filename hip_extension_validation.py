"""
hip_extension_validation.py
----------------------------
Validation for Pattern A1: Hip Extension at Toe-Off.
Two tests:
  A. Cross-engine agreement — MediaPipe (BlazePose) vs Sports2D (RTMPose-m / HALPE_26).
  B. Stride consistency     — spread of hip-extension values across consecutive strides.

Strictly additive:
  - Does NOT modify hip_extension_analysis.py or any existing pipeline script.
  - Reads MediaPipe results from hip_extension_output/hip_extension_results.json
    (which already has phase + plausibility guards applied).
  - Runs Sports2D on run_sideview.mp4 with output → hip_extension_validation/s2d_raw/.
  - All outputs go to hip_extension_validation/.

Angle convention — INCLUDED ANGLE (arccos, 180° = straight):
  MediaPipe  : already in included-angle convention (from hip_extension_analysis.py).
  Sports2D   : MOT file stores a joint angle (0° = straight, same formula as
               mediapipe_pipeline.py's _joint_angle).
               Conversion: s2d_included = 180° − s2d_joint.
               If the raw value is negative (signed extension convention),
               the script detects this and applies 180° + s2d_joint instead.
  Same SME A1 thresholds applied to both: FULL ≥ 160°, PARTIAL 145–160°, RESTRICTED < 145°.

Validation A phase/plausibility guards:
  - Phase guard: MediaPipe JSON already excludes acceleration-phase events (trunk_lean > 20°).
    Sports2D is evaluated only at those same approved frames.
  - Plausibility guard: applied independently to Sports2D values using their own median.
"""

import math, json, os, sys, glob
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_PATH = "run_sideview.mp4"
MP_JSON    = os.path.join("hip_extension_output", "hip_extension_results.json")
VAL_DIR    = "hip_extension_validation"
S2D_DIR    = os.path.join(VAL_DIR, "s2d_raw")

# ── Constants (identical to hip_extension_analysis.py) ────────────────────────
THR_FULL              = 160.0
THR_PARTIAL           = 145.0
FPS_PREF              = 30.0
FPS_PENALTY           = 0.10
PLAUSIBILITY_DROP_DEG = 30.0    # independent guard for Sports2D values

os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(S2D_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def label_incl(angle):
    """Map an included angle to a SME A1 label string."""
    if angle is None or (isinstance(angle, float) and math.isnan(angle)):
        return "UNDETECTABLE"
    if angle >= THR_FULL:
        return "FULL_EXTENSION"
    if angle >= THR_PARTIAL:
        return "PARTIAL"
    return "RESTRICTED"


def parse_mot(path):
    """
    Parse a Sports2D .mot file.
    Returns (header: list[str], rows: list[dict]) where header is lowercase column names
    and each row dict maps column → float (nan for missing/unparseable values).
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"  ERROR reading MOT file: {e}")
        return [], []

    end_hdr = next((i for i, l in enumerate(lines)
                    if l.strip().lower() == "endheader"), None)
    if end_hdr is None:
        print("  ERROR: 'endheader' marker not found in MOT file.")
        return [], []

    header = [h.strip().lower() for h in lines[end_hdr + 1].strip().split("\t")]
    rows   = []
    for line in lines[end_hdr + 2:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        row = {}
        for j, col in enumerate(header):
            raw = parts[j].strip() if j < len(parts) else ""
            try:
                row[col] = float(raw) if raw else float("nan")
            except ValueError:
                row[col] = float("nan")
        rows.append(row)
    return header, rows


def s2d_incl_at_time(s2d_rows, target_t, side, fps):
    """
    Return (included_angle, raw_joint_angle) from Sports2D MOT data at target_t ± 0.6 frames.
    Convention conversion: incl = 180° − joint.
    If joint < 0 (signed extension convention), incl = 180° + joint instead.
    Returns (nan, nan) if no close row exists or the column is missing.
    """
    tol = 0.6 / max(fps, 1.0)
    best, best_dt = None, float("inf")
    for row in s2d_rows:
        dt = abs(row.get("time", float("nan")) - target_t)
        if dt < best_dt:
            best_dt, best = dt, row
    if best is None or best_dt > tol:
        return float("nan"), float("nan")

    joint = best.get(f"{side} hip", float("nan"))
    if math.isnan(joint):
        return float("nan"), float("nan")

    incl = 180.0 - joint
    if incl > 180.0:
        # Signed extension convention: negative joint → switch formula
        incl = 180.0 + joint
    return incl, joint


def stats(values, label):
    """Return a dict with descriptive stats for a list of floats."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n < 2:
        return dict(engine=label, n=n, mean=None, std=None,
                    range=None, min=None, max=None,
                    verdict=f"Too few valid readings ({n}) for consistency analysis.")
    mean_ = float(np.mean(vals))
    std_  = float(np.std(vals))
    rng_  = float(np.max(vals) - np.min(vals))
    if std_ <= 5.0:
        verdict = (f"CONSISTENT — std {std_:.1f}°, range {rng_:.1f}°. "
                   f"Strides cluster tightly; measurement is repeatable.")
    elif std_ <= 10.0:
        verdict = (f"MODERATE — std {std_:.1f}°, range {rng_:.1f}°. "
                   f"Acceptable natural stride-to-stride variability.")
    else:
        verdict = (f"SCATTERED — std {std_:.1f}°, range {rng_:.1f}°. "
                   f"High variability; measurement may be noisy or conditions unsteady.")
    return dict(engine=label, n=n, mean=round(mean_, 1), std=round(std_, 1),
                range=round(rng_, 1), min=round(float(np.min(vals)), 1),
                max=round(float(np.max(vals)), 1), verdict=verdict)


# ── Step 1: Load MediaPipe results ─────────────────────────────────────────────
if not os.path.isfile(MP_JSON):
    sys.exit(
        f"ERROR: {MP_JSON} not found.\n"
        f"Run hip_extension_analysis.py first to generate MediaPipe results."
    )

with open(MP_JSON, encoding="utf-8-sig") as f:
    mp_data = json.load(f)

fps          = mp_data["fps"]
mp_all       = mp_data["toe_offs"]
mp_valid     = [r for r in mp_all if r["label"] != "UNDETECTABLE"]
mp_undetect  = [r for r in mp_all if r["label"] == "UNDETECTABLE"]

print(f"\nPattern A1 Validation — {VIDEO_PATH}  ({fps:.0f}fps)")
print(f"  MediaPipe JSON   : {MP_JSON}")
print(f"  Total toe-offs   : {len(mp_all)}  "
      f"(valid: {len(mp_valid)}, UNDETECTABLE: {len(mp_undetect)})")
if not mp_valid:
    sys.exit("ERROR: No valid MediaPipe toe-offs to validate against.")


# ── Step 2: Run Sports2D (skip if MOT already exists from prior run) ───────────
existing_mot = sorted(
    glob.glob(os.path.join(S2D_DIR, "**", "*_angles_person00.mot"), recursive=True)
)

if existing_mot:
    mot_path = existing_mot[0]
    print(f"  Sports2D cached  : {os.path.relpath(mot_path)}")
else:
    print(f"\n  Running Sports2D on {VIDEO_PATH}  (may take a few minutes)...")
    try:
        from Sports2D import Sports2D as _S2D
        _S2D.process({
            "base": {
                "video_input":            VIDEO_PATH,
                "nb_persons_to_detect":   1,
                "person_ordering_method": "largest_size",
                "show_realtime_results":  False,
                "save_vid":               False,   # skip video — faster
                "save_img":               False,
                "save_pose":              True,
                "calculate_angles":       True,
                "save_angles":            True,
                "result_dir":             S2D_DIR,
            },
            "pose": {
                "pose_model":    "body_with_feet",
                "mode":          "balanced",
                "det_frequency": 4,
            },
            "angles": {
                "joint_angles":   ["Right hip", "Left hip"],
                "segment_angles": ["Trunk"],
            },
            "post-processing": {
                "show_graphs": False,
                "save_graphs": False,
                "filter":      True,
            },
        })
        existing_mot = sorted(
            glob.glob(os.path.join(S2D_DIR, "**", "*_angles_person00.mot"), recursive=True)
        )
        if not existing_mot:
            sys.exit("ERROR: Sports2D ran but no *_angles_person00.mot found. "
                     f"Check {S2D_DIR} for what was actually created.")
        mot_path = existing_mot[0]
        print(f"  Sports2D done    : {os.path.relpath(mot_path)}")
    except ImportError:
        sys.exit("ERROR: Sports2D not installed.  pip install sports2d")
    except Exception as exc:
        sys.exit(f"ERROR: Sports2D failed: {exc}")


# ── Step 3: Parse Sports2D MOT file ───────────────────────────────────────────
s2d_header, s2d_rows = parse_mot(mot_path)
if not s2d_rows:
    sys.exit(f"ERROR: Could not parse MOT file: {mot_path}")

print(f"  Sports2D MOT     : {len(s2d_rows)} rows  columns: {s2d_header}")

missing_cols = [c for c in ("right hip", "left hip") if c not in s2d_header]
if missing_cols:
    print(f"  WARNING: Expected column(s) {missing_cols} not in MOT header. "
          f"Comparison will produce NaN for those sides.")
    print(f"  Available columns: {s2d_header}")

# Detect convention: sample at toe-off frames
_s2d_samples = []
for ev in mp_valid:
    _, j = s2d_incl_at_time(s2d_rows, ev["time_s"], ev["side"], fps)
    if not math.isnan(j):
        _s2d_samples.append(j)
if _s2d_samples:
    _s2d_med = float(np.median(_s2d_samples))
    _conv = "signed (ext < 0)" if _s2d_med < 0 else "unsigned (0°=straight)"
    print(f"  S2D joint angle at toe-offs (median): {_s2d_med:.1f}°  → convention: {_conv}")
    print(f"  → converted included angle (180° ± joint): "
          f"{(180.0 + _s2d_med if _s2d_med < 0 else 180.0 - _s2d_med):.1f}°")


# ── Step 4: Build comparison table ────────────────────────────────────────────
table = []
for ev in mp_valid:
    t    = ev["time_s"]
    side = ev["side"]
    mp_a = ev["hip_angle_deg"]      # already included angle

    s2d_a, s2d_j = s2d_incl_at_time(s2d_rows, t, side, fps)
    s2d_lbl  = label_incl(s2d_a)
    s2d_note = ""

    table.append(dict(
        frame          = ev["frame"],
        time_s         = t,
        side           = side,
        mp_incl        = mp_a,
        mp_label       = ev["label"],
        s2d_joint      = round(s2d_j,  1) if not math.isnan(s2d_j)  else None,
        s2d_incl       = round(s2d_a,  1) if not math.isnan(s2d_a)  else None,
        s2d_label      = s2d_lbl,
        diff_deg       = None,
        trunk_lean_deg = ev.get("trunk_lean_at_toeoff_deg"),
        s2d_note       = s2d_note,
    ))

# Sports2D plausibility guard (independent — uses Sports2D's own median)
_s2d_valid_angles = [r["s2d_incl"] for r in table if r["s2d_incl"] is not None]
_s2d_median_fs    = float(np.median(_s2d_valid_angles)) if _s2d_valid_angles else float("nan")

for r in table:
    if r["s2d_incl"] is None:
        r["s2d_label"] = "UNDETECTABLE"
        r["s2d_note"]  = "No Sports2D reading near this frame."
        continue
    drop = _s2d_median_fs - r["s2d_incl"]
    if drop > PLAUSIBILITY_DROP_DEG:
        r["s2d_label"] = "UNDETECTABLE"
        r["s2d_note"]  = (
            f"S2D plausibility guard: {r['s2d_incl']:.1f}° is {drop:.1f}° below "
            f"S2D full-speed median {_s2d_median_fs:.1f}°."
        )

# Compute absolute difference where both engines returned usable values
diffs = []
for r in table:
    if (r["mp_incl"] is not None
            and r["s2d_incl"] is not None
            and r["s2d_label"] != "UNDETECTABLE"):
        d = abs(r["mp_incl"] - r["s2d_incl"])
        r["diff_deg"] = round(d, 1)
        diffs.append(d)


# ── Step 5: Agreement statistics (Validation A) ───────────────────────────────
n_comp = len(diffs)
if diffs:
    mad  = round(float(np.mean(diffs)),   2)
    maxd = round(float(np.max(diffs)),    1)
    stdd = round(float(np.std(diffs)),    2)
else:
    mad = maxd = stdd = None

if mad is None:
    agree_verdict = "INDETERMINATE — no frames where both engines returned usable values."
elif mad <= 5.0:
    agree_verdict = (
        f"STRONG agreement (MAD {mad:.1f}°) — measurement is model-independent. "
        f"Both BlazePose and RTMPose place the hip angle in the same zone."
    )
elif mad <= 10.0:
    agree_verdict = (
        f"MODERATE agreement (MAD {mad:.1f}°) — within the 5–10° expected inter-model tolerance. "
        f"Results are broadly consistent but carry ~{mad:.0f}° model-specific uncertainty."
    )
else:
    agree_verdict = (
        f"WEAK agreement (MAD {mad:.1f}°) — exceeds 10° tolerance. "
        f"The hip angle depends significantly on which model is used; "
        f"single-engine results should be treated with caution."
    )


# ── Step 6: Stride consistency (Validation B) ─────────────────────────────────
mp_angles  = [r["mp_incl"]  for r in table if r["mp_incl"]  is not None]
s2d_angles = [r["s2d_incl"] for r in table
              if r["s2d_incl"] is not None and r["s2d_label"] != "UNDETECTABLE"]

mp_cons  = stats(mp_angles,  "MediaPipe")
s2d_cons = stats(s2d_angles, "Sports2D")


# ── Step 7: Overall one-line verdict ──────────────────────────────────────────
if mad is not None and n_comp >= 3:
    mp_std = mp_cons.get("std")
    if mad <= 10.0 and mp_std is not None and mp_std <= 10.0:
        one_line = (
            f"TRUSTWORTHY AND REPEATABLE — engines agree to {mad:.1f}° MAD; "
            f"stride consistency std {mp_std:.1f}° (MediaPipe). "
            f"24fps caveat noted (±42ms timing, ~5–10° angle uncertainty per event)."
        )
    elif mad > 10.0:
        one_line = (
            f"UNCERTAIN — engine disagreement ({mad:.1f}° MAD) exceeds 10° tolerance; "
            f"results are model-dependent. Investigate convention or model accuracy."
        )
    else:
        mp_std_str = f"{mp_std:.1f}°" if mp_std is not None else "?"
        one_line = (
            f"PARTIALLY TRUSTWORTHY — agreement MAD {mad:.1f}° but stride scatter "
            f"std {mp_std_str} limits per-event precision. Pattern-level classification reliable."
        )
else:
    one_line = f"INSUFFICIENT DATA — {n_comp} comparable readings; verdict deferred."


# ── Write JSON ─────────────────────────────────────────────────────────────────
out = dict(
    clip=VIDEO_PATH,
    fps=fps,
    fps_caveat=(
        f"{fps:.0f}fps — below SME D3 preference (≥30fps). "
        f"Timing uncertainty ±{1000/fps:.0f}ms per frame; ~5–10° angle error at toe-off."
    ),
    angle_convention=(
        "Both engines compared in INCLUDED ANGLE convention (arccos, 180°=straight). "
        "Sports2D MOT joint angle converted via: incl = 180° − joint "
        "(or 180° + joint for negative/extension-signed values)."
    ),
    sources=dict(
        mediapipe_json=MP_JSON,
        sports2d_mot=os.path.relpath(mot_path),
    ),
    sports2d_convention_note=(
        f"Sports2D median joint angle at toe-off frames: "
        f"{_s2d_med:.1f}° → {_conv}." if _s2d_samples else "Convention not determined."
    ),
    validation_A=dict(
        description="Cross-engine agreement: MediaPipe (BlazePose) vs Sports2D (RTMPose-m HALPE_26)",
        n_mediapipe_valid=len(mp_valid),
        n_comparable=n_comp,
        mean_abs_diff_deg=mad,
        max_diff_deg=maxd,
        std_diff_deg=stdd,
        sports2d_plausibility_median=round(_s2d_median_fs, 1) if not math.isnan(_s2d_median_fs) else None,
        agreement_verdict=agree_verdict,
        event_table=table,
    ),
    validation_B=dict(
        description="Stride consistency: spread of hip-extension angles across full-speed toe-offs",
        mediapipe=mp_cons,
        sports2d=s2d_cons,
    ),
    overall_verdict=one_line,
)

json_path = os.path.join(VAL_DIR, "validation_results.json")
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(out, jf, indent=2)
print(f"\n  JSON → {json_path}")


# ── Printed summary ────────────────────────────────────────────────────────────
def fmt(v, prec=1):
    return f"{v:.{prec}f}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"

print()
print("=" * 74)
print("  PATTERN A1 VALIDATION: CROSS-ENGINE + STRIDE CONSISTENCY")
print("=" * 74)
print(f"  Clip : {VIDEO_PATH}  @{fps:.0f}fps  "
      f"⚠ below 30fps preference (±{1000/fps:.0f}ms timing uncertainty)")
print()

# ── Validation A ───────────────────────────────────────────────────────────────
print("  VALIDATION A — Cross-engine: MediaPipe (BlazePose) vs Sports2D (RTMPose-m)")
print("  " + "-" * 70)
hdr = (f"  {'t(s)':>6}  {'Side':5}  {'MP°':>7}  {'MP label':16}"
       f"  {'S2D jt°':>8}  {'S2D°':>7}  {'S2D label':16}  {'Δ°':>5}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in table:
    diff_s = fmt(r["diff_deg"])
    s2d_lbl = r["s2d_label"] + ("⚠" if r["s2d_note"] else "")
    print(f"  {r['time_s']:>6.2f}  {r['side']:5}"
          f"  {fmt(r['mp_incl']):>7}  {r['mp_label']:16}"
          f"  {fmt(r['s2d_joint']):>8}  {fmt(r['s2d_incl']):>7}  {s2d_lbl:16}  {diff_s:>5}")

print()
if _s2d_samples:
    print(f"  Sports2D angle convention at toe-off frames: {_conv} "
          f"(median raw = {_s2d_med:.1f}°)")
print(f"  Comparable events : {n_comp} / {len(mp_valid)}")
if mad is not None:
    print(f"  Mean absolute diff: {mad:.1f}°   max: {maxd:.1f}°   std: {stdd:.1f}°")
print(f"  Agreement verdict : {agree_verdict}")
print()

# ── Validation B ───────────────────────────────────────────────────────────────
print("  VALIDATION B — Stride consistency")
print("  " + "-" * 70)
for cs in (mp_cons, s2d_cons):
    label_e = cs["engine"]
    if cs["std"] is not None:
        vals_str = f"n={cs['n']}  mean={cs['mean']:.1f}°  std={cs['std']:.1f}°  "
        vals_str += f"range={cs['range']:.1f}°  [{cs['min']:.1f}°–{cs['max']:.1f}°]"
        print(f"  {label_e:12}: {vals_str}")
    else:
        print(f"  {label_e:12}: n={cs['n']}  (insufficient for statistics)")
    print(f"               → {cs['verdict']}")
    print()

print("  ONE-LINE VERDICT")
print("  " + "-" * 70)
print(f"  {one_line}")
print()
print(f"  Outputs → {VAL_DIR}/")
print(f"    validation_results.json")
print("=" * 74)
