"""
hip_extension_validation_2d.py
--------------------------------
Re-runs the cross-engine hip-extension comparison using Sports2D's 2D pixel
keypoints directly, bypassing the 3D metre-space projection that caused the
~16° systematic offset in hip_extension_validation.py.

Method:
  - Reads Sports2D's *_px_person00.trc (Butterworth-smoothed 2D pixel coords,
    already written by hip_extension_validation.py's Sports2D run).
  - For each MediaPipe full-speed toe-off frame, extracts the relevant
    (Knee, Hip, Shoulder) pixel coordinates for the correct side.
  - Computes the INCLUDED ANGLE with the IDENTICAL arccos formula used in
    hip_extension_analysis.py.  The formula is scale-invariant — pixel vs
    normalised coordinates produce the same angle.
  - Compares directly with MediaPipe's included angle at the same frame.

This is a true apples-to-apples comparison:
  same 3 anatomical landmarks, same formula, same 2D image-coordinate geometry.
  Any remaining disagreement is purely from the two pose models placing
  landmarks differently (BlazePose vs RTMPose-m), not from coordinate-space
  or projection differences.

Output:
  hip_extension_validation/validation_2d_results.json
  Printed summary with side-by-side table and verdict.

Does NOT modify hip_extension_analysis.py, hip_extension_validation.py,
run_sports2d.py, or any existing pipeline script.
"""

import math, json, os, sys, glob
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
VIDEO_PATH = "run_sideview.mp4"
MP_JSON    = os.path.join("hip_extension_output", "hip_extension_results.json")
VAL_DIR    = "hip_extension_validation"
TRC_GLOB   = os.path.join(VAL_DIR, "s2d_raw", "**", "*_px_person00.trc")
OUT_JSON   = os.path.join(VAL_DIR, "validation_2d_results.json")
PRIOR_JSON = os.path.join(VAL_DIR, "validation_results.json")   # 3D run results

# ── SME A1 constants (identical to hip_extension_analysis.py) ─────────────────
THR_FULL              = 160.0
THR_PARTIAL           = 145.0
PLAUSIBILITY_DROP_DEG = 30.0

# ── Helpers ────────────────────────────────────────────────────────────────────

def label_incl(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return "UNDETECTABLE"
    return ("FULL_EXTENSION" if a >= THR_FULL else
            "PARTIAL"        if a >= THR_PARTIAL else
            "RESTRICTED")


def included_angle(ax, ay, bx, by, cx, cy):
    """
    Included angle in degrees at vertex B formed by A-B-C.
    arccos(dot/magnitudes): 180° = straight, decreases as joint bends.
    Scale-invariant — pixel and normalised coordinates give identical results.
    Exact same implementation as hip_extension_analysis.py.
    """
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    mag = math.sqrt((v1x**2 + v1y**2) * (v2x**2 + v2y**2))
    if mag < 1e-9:
        return float("nan")
    cos_t = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / mag))
    return math.degrees(math.acos(cos_t))


def parse_trc_px(path):
    """
    Parse Sports2D's pixel TRC file (X/Y/Z format, Z always 0.0 for 2D video).

    TRC format observed in this project:
      Line 0 : PathFileType header
      Line 1 : DataRate / NumFrames / NumMarkers labels
      Line 2 : corresponding values
      Line 3 : "Frame#  Time  Hip(empty)(empty)  RHip(empty)(empty)  ..."
               Each marker name occupies 3 tab-columns (X, Y, Z).
               The marker name is in the first column; the next two are empty.
      Line 4 : "  X1  Y1  Z1  X2  Y2  Z2 ..."  (sub-column labels, ignored)
      Line 5+: data rows  "frame_num  time  x1 y1 z1  x2 y2 z2 ..."

    Returns:
      marker_xcol : dict  marker_name → column-index of its X value in data rows.
                    Y is at marker_xcol+1, Z at marker_xcol+2 (ignored).
      frame_to_row: dict  frame_number(int) → list[float]  (full tab-split row)
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Line 3: header row with marker names
    hdr_parts = lines[3].rstrip("\n").split("\t")
    marker_xcol = {}
    for col_i, h in enumerate(hdr_parts):
        name = h.strip()
        if name and name not in ("Frame#", "Time"):
            marker_xcol[name] = col_i  # X at this column, Y at col_i+1

    # Data rows start at line 5
    frame_to_row = {}
    for line in lines[5:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        try:
            frame_num = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        row = []
        for p in parts:
            try:
                row.append(float(p) if p.strip() else float("nan"))
            except ValueError:
                row.append(float("nan"))
        frame_to_row[frame_num] = row

    return marker_xcol, frame_to_row


def get_xy(frame_to_row, marker_xcol, frame_num, marker_name):
    """Return (x, y) pixel coords for a marker at frame_num, or (nan, nan)."""
    row   = frame_to_row.get(frame_num)
    x_col = marker_xcol.get(marker_name)
    if row is None or x_col is None:
        return float("nan"), float("nan")
    x = row[x_col]     if x_col     < len(row) else float("nan")
    y = row[x_col + 1] if x_col + 1 < len(row) else float("nan")
    return x, y


def hip_2d_incl(frame_to_row, marker_xcol, frame_num, side):
    """
    Compute included hip angle for 'left' or 'right' from Sports2D 2D pixels.
    Geometry: Knee → Hip → Shoulder  (same three landmarks as MediaPipe).
    Returns (angle_deg, all_landmarks_found: bool).
    """
    if side == "right":
        kx, ky = get_xy(frame_to_row, marker_xcol, frame_num, "RKnee")
        hx, hy = get_xy(frame_to_row, marker_xcol, frame_num, "RHip")
        sx, sy = get_xy(frame_to_row, marker_xcol, frame_num, "RShoulder")
    else:
        kx, ky = get_xy(frame_to_row, marker_xcol, frame_num, "LKnee")
        hx, hy = get_xy(frame_to_row, marker_xcol, frame_num, "LHip")
        sx, sy = get_xy(frame_to_row, marker_xcol, frame_num, "LShoulder")

    if any(math.isnan(v) for v in (kx, ky, hx, hy, sx, sy)):
        return float("nan"), False
    return included_angle(kx, ky, hx, hy, sx, sy), True


def vstats(values):
    """Descriptive stats dict for a list of floats, excluding nan/None."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return dict(n=len(v), mean=None, std=None, range=None, min=None, max=None)
    return dict(n=len(v),
                mean=round(float(np.mean(v)),  1),
                std=round(float(np.std(v)),    1),
                range=round(float(np.max(v) - np.min(v)), 1),
                min=round(float(np.min(v)),  1),
                max=round(float(np.max(v)),  1))


def fmt(v, d=1):
    return (f"{v:.{d}f}"
            if v is not None and not (isinstance(v, float) and math.isnan(v))
            else "N/A")


# ── Step 1: Load MediaPipe results ─────────────────────────────────────────────
if not os.path.isfile(MP_JSON):
    sys.exit(f"ERROR: {MP_JSON} not found. Run hip_extension_analysis.py first.")

with open(MP_JSON, encoding="utf-8-sig") as f:
    mp_data = json.load(f)

fps      = mp_data["fps"]
mp_valid = [r for r in mp_data["toe_offs"] if r["label"] != "UNDETECTABLE"]

print(f"\nPattern A1 — 2D apples-to-apples cross-engine comparison")
print(f"  {VIDEO_PATH}  @{fps:.0f}fps  |  MediaPipe valid events: {len(mp_valid)}")


# ── Step 2: Load prior 3D MAD for comparison ───────────────────────────────────
prior_mad = None
if os.path.isfile(PRIOR_JSON):
    try:
        with open(PRIOR_JSON, encoding="utf-8-sig") as f:
            prior = json.load(f)
        prior_mad = prior.get("validation_A", {}).get("mean_abs_diff_deg")
        print(f"  Prior 3D-projection run MAD: {prior_mad:.1f}°")
    except Exception:
        pass


# ── Step 3: Find Sports2D 2D-pixel TRC file ────────────────────────────────────
trc_files = sorted(glob.glob(TRC_GLOB, recursive=True))
if not trc_files:
    sys.exit(
        f"ERROR: No *_px_person00.trc found under {os.path.join(VAL_DIR, 's2d_raw')}.\n"
        f"Run hip_extension_validation.py first (it generates the Sports2D TRC file)."
    )
trc_path = trc_files[0]
print(f"  Sports2D 2D TRC  : {os.path.relpath(trc_path)}")


# ── Step 4: Parse TRC ──────────────────────────────────────────────────────────
marker_xcol, frame_to_row = parse_trc_px(trc_path)
print(f"  Markers in TRC   : {sorted(marker_xcol)}")
print(f"  Frames in TRC    : {len(frame_to_row)}")

required = {"RKnee", "RHip", "RShoulder", "LKnee", "LHip", "LShoulder"}
missing  = required - set(marker_xcol)
if missing:
    sys.exit(f"ERROR: Required markers {missing} not found. "
             f"Available: {sorted(marker_xcol)}")

# Sanity-check pixel magnitudes (should be in image-coordinate range, not 0–1)
sample_row = frame_to_row.get(0) or next(iter(frame_to_row.values()), None)
if sample_row is not None:
    rhip_x = sample_row[marker_xcol["RHip"]]
    print(f"  RHip.x at frame 0: {rhip_x:.1f}px  "
          f"({'pixels ✓' if rhip_x > 10 else 'normalised? check'})")


# ── Step 5: Compute Sports2D-2D hip angles at each toe-off frame ───────────────
table = []
for ev in mp_valid:
    frame = ev["frame"]
    side  = ev["side"]
    mp_a  = ev["hip_angle_deg"]

    s2d_a, ok = hip_2d_incl(frame_to_row, marker_xcol, frame, side)
    table.append(dict(
        frame        = frame,
        time_s       = ev["time_s"],
        side         = side,
        mp_incl      = mp_a,
        mp_label     = ev["label"],
        s2d_2d_incl  = round(s2d_a, 1) if ok else None,
        s2d_2d_label = label_incl(s2d_a) if ok else "UNDETECTABLE",
        diff_deg     = None,
        trunk_lean   = ev.get("trunk_lean_at_toeoff_deg"),
        s2d_note     = ("" if ok else f"No TRC data at frame {frame}."),
    ))


# ── Step 6: Sports2D-2D plausibility guard ─────────────────────────────────────
# Applied independently on Sports2D's own values — same D3 logic as the main analysis.
_s2d_raw = [r["s2d_2d_incl"] for r in table if r["s2d_2d_incl"] is not None]
_s2d_med = float(np.median(_s2d_raw)) if _s2d_raw else float("nan")
for r in table:
    if r["s2d_2d_incl"] is None:
        continue
    drop = _s2d_med - r["s2d_2d_incl"]
    if drop > PLAUSIBILITY_DROP_DEG:
        r["s2d_2d_label"] = "UNDETECTABLE"
        r["s2d_note"] = (
            f"S2D-2D plausibility guard: {r['s2d_2d_incl']:.1f}° is "
            f"{drop:.1f}° below S2D-2D median {_s2d_med:.1f}°."
        )


# ── Step 7: Compute differences ────────────────────────────────────────────────
diffs = []
for r in table:
    if (r["mp_incl"] is not None and r["s2d_2d_incl"] is not None
            and r["s2d_2d_label"] != "UNDETECTABLE"):
        d = abs(r["mp_incl"] - r["s2d_2d_incl"])
        r["diff_deg"] = round(d, 1)
        diffs.append(d)

n_comp = len(diffs)
mad    = round(float(np.mean(diffs)),  2) if diffs else None
maxd   = round(float(np.max(diffs)),   1) if diffs else None
stdd   = round(float(np.std(diffs)),   2) if diffs else None


# ── Step 8: Verdict ────────────────────────────────────────────────────────────
if mad is None:
    verdict = "INDETERMINATE — no comparable readings."
elif mad <= 5.0:
    if prior_mad and prior_mad > 10.0:
        verdict = (
            f"STRONG agreement (MAD {mad:.1f}°) — confirmed: the earlier {prior_mad:.1f}° "
            f"gap was caused by Sports2D's 3D metre-space projection. "
            f"In 2D image-coordinate space, both models agree within the expected "
            f"inter-model noise floor. The measurement is substantially model-independent."
        )
    else:
        verdict = (
            f"STRONG agreement (MAD {mad:.1f}°) — both models agree within the "
            f"expected inter-model noise floor in 2D coordinate space."
        )
elif mad <= 10.0:
    if prior_mad and mad < prior_mad - 2:
        verdict = (
            f"MODERATE agreement (MAD {mad:.1f}°). The 2D comparison substantially "
            f"reduced the gap from {prior_mad:.1f}° (3D) to {mad:.1f}° (2D) — "
            f"the 3D projection was the dominant cause. Residual disagreement is within "
            f"the 5–10° expected inter-model tolerance."
        )
    else:
        verdict = (
            f"MODERATE agreement (MAD {mad:.1f}°) — within the 5–10° inter-model "
            f"tolerance. Results are broadly consistent between engines."
        )
else:
    prior_str = f" (was {prior_mad:.1f}° in 3D mode)" if prior_mad else ""
    verdict = (
        f"STILL WEAK agreement (MAD {mad:.1f}°{prior_str}). "
        f"Even with an identical 2D formula, the engines disagree beyond 10°. "
        f"The systematic offset is NOT solely from the 3D projection; "
        f"the two models (BlazePose vs RTMPose-m) place the same anatomical "
        f"landmarks at measurably different pixel positions in this clip."
    )


# ── Step 9: Stride consistency ─────────────────────────────────────────────────
mp_angles  = [r["mp_incl"]     for r in table if r["mp_incl"] is not None]
s2d_angles = [r["s2d_2d_incl"] for r in table
              if r["s2d_2d_incl"] is not None and r["s2d_2d_label"] != "UNDETECTABLE"]
mp_st  = vstats(mp_angles);  mp_st["engine"]  = "MediaPipe"
s2d_st = vstats(s2d_angles); s2d_st["engine"] = "Sports2D-2D"


# ── Step 10: Write JSON ────────────────────────────────────────────────────────
os.makedirs(VAL_DIR, exist_ok=True)
out = dict(
    clip=VIDEO_PATH, fps=fps,
    fps_caveat=(f"{fps:.0f}fps — below SME D3 preference. "
                f"±{1000/fps:.0f}ms timing uncertainty per toe-off frame."),
    method=(
        "Sports2D 2D pixel coords from *_px_person00.trc (Butterworth-smoothed). "
        "Hip angle: included_angle(Knee, Hip, Shoulder) using arccos formula — "
        "identical to hip_extension_analysis.py. Scale-invariant: pixels and "
        "normalised coords produce the same angle."
    ),
    prior_3d_mad_deg=prior_mad,
    validation_A=dict(
        description=(
            "Cross-engine agreement: MediaPipe (BlazePose, normalised coords) vs "
            "Sports2D-2D (RTMPose-m HALPE_26, pixel coords, identical formula)"
        ),
        n_mediapipe_valid=len(mp_valid),
        n_comparable=n_comp,
        mean_abs_diff_deg=mad,
        max_diff_deg=maxd,
        std_diff_deg=stdd,
        s2d_2d_plausibility_median=(
            round(_s2d_med, 1) if not math.isnan(_s2d_med) else None
        ),
        agreement_verdict=verdict,
        event_table=table,
    ),
    validation_B=dict(
        mediapipe=mp_st,
        sports2d_2d=s2d_st,
    ),
)

with open(OUT_JSON, "w", encoding="utf-8") as jf:
    json.dump(out, jf, indent=2)
print(f"\n  JSON → {OUT_JSON}")


# ── Printed summary ────────────────────────────────────────────────────────────
print()
print("=" * 78)
print("  PATTERN A1 — 2D APPLES-TO-APPLES CROSS-ENGINE COMPARISON")
print("  Sports2D 2D pixel coords + identical arccos formula vs MediaPipe")
print("=" * 78)
print(f"  Clip : {VIDEO_PATH}  @{fps:.0f}fps  ⚠ ±{1000/fps:.0f}ms timing uncertainty")
if prior_mad is not None:
    print(f"  Prior 3D-projection MAD: {prior_mad:.1f}°  ← comparing against this")
print()

# Event table
hdr = (f"  {'t(s)':>6}  {'Side':5}  {'MP°':>7}  {'MP label':18}"
       f"  {'S2D-2D°':>8}  {'S2D-2D label':20}  {'Δ°':>5}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in table:
    lbl = r["s2d_2d_label"] + ("⚠" if r["s2d_note"] else "")
    print(f"  {r['time_s']:>6.2f}  {r['side']:5}"
          f"  {fmt(r['mp_incl']):>7}  {r['mp_label']:18}"
          f"  {fmt(r['s2d_2d_incl']):>8}  {lbl:20}  {fmt(r['diff_deg']):>5}")

print()
print(f"  Comparable events : {n_comp} / {len(mp_valid)}")
if mad is not None:
    print(f"  Mean absolute diff: {mad:.1f}°   max: {maxd:.1f}°   std: {stdd:.1f}°")
    if prior_mad is not None:
        delta = prior_mad - mad
        print(f"  Change from 3D run: {'REDUCED' if delta > 0 else 'INCREASED'} "
              f"by {abs(delta):.1f}°  ({prior_mad:.1f}° → {mad:.1f}°)")
print()
print(f"  VERDICT: {verdict}")
print()

# Stride consistency
print("  STRIDE CONSISTENCY")
print("  " + "-" * 70)
for st in (mp_st, s2d_st):
    if st["std"] is not None:
        cons = ("CONSISTENT" if st["std"] <= 5.0 else
                "MODERATE"   if st["std"] <= 10.0 else "SCATTERED")
        print(f"  {st['engine']:14}: n={st['n']}  mean={st['mean']:.1f}°  "
              f"std={st['std']:.1f}°  range={st['range']:.1f}°  "
              f"[{st['min']:.1f}°–{st['max']:.1f}°]  → {cons}")
    else:
        print(f"  {st['engine']:14}: n={st['n']}  — insufficient for consistency stats")
print()
print(f"  Output → {OUT_JSON}")
print("=" * 78)
