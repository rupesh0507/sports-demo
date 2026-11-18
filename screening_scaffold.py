"""
screening_scaffold.py
---------------------
Configurable qualify / disqualify SCAFFOLDING MODULE for athlete screening.

Reads existing analysis outputs (hip_extension_output/, trust reports, etc.)
and applies a configurable threshold table (screening_thresholds.json) to
produce coarse per-metric screening signals:

    QUALIFY     — measured value clearly meets the positive threshold,
                  confidence adequate, not near a boundary.
    DISQUALIFY  — measured value clearly fails, confidence adequate,
                  not near a boundary.
    NEEDS_REVIEW — confidence too low, OR value within boundary margin of a
                  threshold, OR source reading flagged ambiguous or
                  UNDETECTABLE.  Per SME brief D3, this is always preferred
                  over forcing a hard verdict on uncertain input.

IMPORTANT — this module NEVER produces a single "this athlete qualifies"
overall decision.  All output is per-metric, per-event, for scout review.

IMPORTANT — all thresholds loaded from screening_thresholds.json are
PROVISIONAL_NOT_VALIDATED placeholders.  Real NFL-level cutoffs require
combine data, scout validation, and statistical analysis.  Do not use for
actual athlete decisions.

Usage:
    python screening_scaffold.py                 # screens run_sideview (default)
    python screening_scaffold.py <clip_stem>     # screens <clip_stem>

Output:
    screening_output/<clip_stem>_screening.json
    Printed summary to console.

Does NOT modify any existing script or output folder.
"""

import json, os, sys, math, datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
THRESHOLDS_PATH = "screening_thresholds.json"
OUT_DIR         = "screening_output"

# ── Verdict constants ─────────────────────────────────────────────────────────
QUALIFY      = "QUALIFY"
DISQUALIFY   = "DISQUALIFY"
NEEDS_REVIEW = "NEEDS_REVIEW"

DISCLAIMER = (
    "PROVISIONAL SCREENING SCAFFOLD — NOT FOR OPERATIONAL USE.\n"
    "All thresholds are placeholders from the SME brief. NFL-level cutoffs\n"
    "require combine data, scout-validated ground truth, and statistical\n"
    "validation. Per SME brief D3: no hard verdict on weak or insufficient\n"
    "input. This output is a decision-layer structure for scout review only.\n"
    "No overall qualify / disqualify verdict is computed or implied."
)


# ── 1. Load threshold config ──────────────────────────────────────────────────

def load_thresholds(path=THRESHOLDS_PATH):
    """Load screening_thresholds.json. Exits clearly if the file is missing."""
    if not os.path.isfile(path):
        sys.exit(
            f"ERROR: Threshold config not found: {path}\n"
            f"Create {path} before running this module."
        )
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    # Sanity-check expected keys
    if "patterns" not in cfg:
        sys.exit(f"ERROR: {path} is missing the 'patterns' key.")
    return cfg


# ── 2. Core screening function ────────────────────────────────────────────────

def screen_metric(value, confidence, pattern_cfg, source_label=None, ambiguous=False):
    """
    Apply the screening threshold for one measurement.

    Returns (verdict, reason):
        verdict  — QUALIFY | DISQUALIFY | NEEDS_REVIEW
        reason   — human-readable string explaining the verdict

    Priority order (first triggered wins):
      1. Quality guard   — source_label UNDETECTABLE, or value None/NaN
      2. Confidence guard — confidence below minimum
      3. Ambiguity guard  — source analysis flagged reading as ambiguous
      4. Boundary guard   — value within boundary_margin_deg of any threshold
      5. Clear zones      — value clearly qualifies or clearly disqualifies
      6. Fallback         — middle zone not reached by guards → NEEDS_REVIEW

    Per SME brief D3: always prefer NEEDS_REVIEW over a hard call on
    uncertain input.
    """
    thr = pattern_cfg["thresholds"]
    qualify_min      = thr["qualify_min_deg"]
    disqualify_max   = thr["disqualify_max_deg"]
    margin           = thr["boundary_margin_deg"]
    min_conf         = thr["min_confidence"]
    clear_qualify    = thr["clear_qualify_zone_deg"]     # qualify_min + margin
    clear_disqualify = thr["clear_disqualify_zone_deg"]  # disqualify_max - margin

    # 1. Quality guard
    if source_label == "UNDETECTABLE":
        return NEEDS_REVIEW, (
            "Source analysis flagged UNDETECTABLE — quality insufficient "
            "(acceleration phase or visibility below minimum threshold)."
        )
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NEEDS_REVIEW, "Measurement is missing or NaN — quality insufficient."

    # 2. Confidence guard
    if confidence is not None and confidence < min_conf:
        return NEEDS_REVIEW, (
            f"Confidence {confidence:.2f} < minimum {min_conf} — "
            "signal not reliable enough for a hard verdict."
        )

    # 3. Ambiguity guard (source analysis flagged this event explicitly)
    if ambiguous:
        return NEEDS_REVIEW, (
            "Source analysis flagged reading as ambiguous — ±1 frame at "
            "recorded fps may shift the label (SME brief D3 applies)."
        )

    # 4. Boundary guard — check proximity to each threshold
    near_qualify    = abs(value - qualify_min)    <= margin
    near_disqualify = abs(value - disqualify_max) <= margin

    if near_qualify and near_disqualify:
        # Value is simultaneously close to both boundaries (very narrow mid-zone)
        return NEEDS_REVIEW, (
            f"{value:.1f}° is within ±{margin}° of BOTH the QUALIFY threshold "
            f"({qualify_min}°) and DISQUALIFY threshold ({disqualify_max}°) — "
            "precision insufficient for any hard call."
        )
    if near_qualify:
        return NEEDS_REVIEW, (
            f"{value:.1f}° is within ±{margin}° of the QUALIFY threshold "
            f"({qualify_min}°). Cross-engine measurement uncertainty (~11°) "
            "prevents a definitive verdict."
        )
    if near_disqualify:
        return NEEDS_REVIEW, (
            f"{value:.1f}° is within ±{margin}° of the DISQUALIFY threshold "
            f"({disqualify_max}°). Cross-engine measurement uncertainty (~11°) "
            "prevents a definitive verdict."
        )

    # 5. Clear zones — value is far enough from both boundaries
    if value >= clear_qualify:
        return QUALIFY, (
            f"{value:.1f}° >= {clear_qualify}° (QUALIFY threshold {qualify_min}° "
            f"+ {margin}° margin) — clearly in the extension territory."
        )
    if value <= clear_disqualify:
        return DISQUALIFY, (
            f"{value:.1f}° <= {clear_disqualify}° (DISQUALIFY threshold "
            f"{disqualify_max}° − {margin}° margin) — clearly restricted."
        )

    # 6. Fallback: value sits between clear zones but wasn't caught by guards
    #    (theoretically unreachable with a consistent threshold config, but
    #    included defensively so the function never returns without a verdict)
    return NEEDS_REVIEW, (
        f"{value:.1f}° falls between clear zones "
        f"({clear_disqualify}° – {clear_qualify}°) — NEEDS_REVIEW by default."
    )


# ── 3. Pattern A1: Hip Extension at Toe-Off ───────────────────────────────────

def screen_hip_extension(thresholds, mp_json_path):
    """
    Read hip_extension_analysis.py output JSON and apply Pattern A1 screening.

    Each toe-off event is assessed independently.  UNDETECTABLE events are
    tracked separately (excluded from qualify / disqualify counts) but still
    appear in the events list with a NEEDS_REVIEW verdict and reason.

    Returns a dict suitable for inclusion in the full screening report.
    """
    pattern_key = "A1_hip_extension_at_toeoff"
    if pattern_key not in thresholds["patterns"]:
        return {
            "pattern":      pattern_key,
            "error":        f"Pattern '{pattern_key}' not found in {THRESHOLDS_PATH}.",
            "events":       [],
            "clip_summary": None,
        }

    pattern_cfg = thresholds["patterns"][pattern_key]
    thr         = pattern_cfg["thresholds"]

    if not os.path.isfile(mp_json_path):
        return {
            "pattern":      pattern_key,
            "error":        f"Input JSON not found: {mp_json_path}",
            "events":       [],
            "clip_summary": None,
        }

    with open(mp_json_path, encoding="utf-8-sig") as f:
        src = json.load(f)

    clip     = src.get("clip", "unknown")
    fps      = src.get("fps", "unknown")
    toe_offs = src.get("toe_offs", [])

    events      = []
    n_qualify   = 0
    n_disqualify = 0
    n_review    = 0
    n_excluded  = 0   # UNDETECTABLE events (not counted in qualify/disqualify/review)

    for i, ev in enumerate(toe_offs):
        label      = ev.get("label", "")
        value      = ev.get("hip_angle_deg")
        confidence = ev.get("confidence")
        ambiguous  = ev.get("ambiguous", False)
        time_s     = ev.get("time_s")
        side       = ev.get("side", "?")
        note       = ev.get("note", "")

        if label == "UNDETECTABLE":
            # Excluded from pass/fail counts; shown in event list for transparency
            n_excluded += 1
            verdict = NEEDS_REVIEW
            reason  = (
                "Source analysis flagged UNDETECTABLE — quality insufficient "
                f"(acceleration phase or visibility below threshold). {note}"
            ).strip()
        else:
            verdict, reason = screen_metric(
                value=value,
                confidence=confidence,
                pattern_cfg=pattern_cfg,
                source_label=label,
                ambiguous=ambiguous,
            )
            if verdict == QUALIFY:
                n_qualify += 1
            elif verdict == DISQUALIFY:
                n_disqualify += 1
            else:
                n_review += 1

        events.append({
            "event_index":        i,
            "time_s":             time_s,
            "side":               side,
            "measured_value_deg": value,
            "confidence":         confidence,
            "source_label":       label,
            "ambiguous_flag":     ambiguous,
            "source_note":        note,
            "screening_verdict":  verdict,
            "verdict_reason":     reason,
            "threshold_used": {
                "qualify_min_deg":       thr["qualify_min_deg"],
                "disqualify_max_deg":    thr["disqualify_max_deg"],
                "boundary_margin_deg":   thr["boundary_margin_deg"],
                "clear_qualify_zone_deg":  thr["clear_qualify_zone_deg"],
                "clear_disqualify_zone_deg": thr["clear_disqualify_zone_deg"],
                "min_confidence":        thr["min_confidence"],
            },
            "threshold_status": "PROVISIONAL_NOT_VALIDATED",
        })

    # Aggregate stats across valid (non-UNDETECTABLE) events only
    n_valid = n_qualify + n_disqualify + n_review
    valid_angles = [
        ev["measured_value_deg"]
        for ev in events
        if ev["source_label"] != "UNDETECTABLE"
        and ev["measured_value_deg"] is not None
        and not (isinstance(ev["measured_value_deg"], float)
                 and math.isnan(ev["measured_value_deg"]))
    ]
    mean_angle = round(sum(valid_angles) / len(valid_angles), 1) if valid_angles else None

    # Narrative signal — human-readable summary, NOT a verdict
    if n_valid == 0:
        signal = (
            "No valid events — all flagged UNDETECTABLE. "
            "Pattern A1 cannot be assessed from this clip."
        )
    elif n_review == n_valid:
        signal = (
            f"All {n_review} valid events → NEEDS_REVIEW. "
            f"Measurements cluster near threshold boundaries "
            f"(mean {mean_angle}°, qualify threshold {thr['qualify_min_deg']}°, "
            f"±{thr['boundary_margin_deg']}° margin). "
            "Pattern suggests borderline-to-adequate extension. "
            "A definitive call requires ≥30fps video, multi-stride data, "
            "and cross-engine agreement."
        )
    elif n_qualify == n_valid:
        signal = (
            f"All {n_qualify} valid events → QUALIFY. "
            "Provisional positive signal — strong consistent hip extension. "
            f"Mean {mean_angle}°."
        )
    elif n_disqualify == n_valid:
        signal = (
            f"All {n_disqualify} valid events → DISQUALIFY. "
            "Provisional concern — consistently restricted extension. "
            f"Mean {mean_angle}°."
        )
    elif n_disqualify == 0:
        signal = (
            f"{n_qualify} QUALIFY, {n_review} NEEDS_REVIEW, 0 DISQUALIFY. "
            f"Provisional positive lean (mean {mean_angle}°). "
            "Scout review recommended to resolve NEEDS_REVIEW events."
        )
    elif n_qualify == 0:
        signal = (
            f"0 QUALIFY, {n_disqualify} DISQUALIFY, {n_review} NEEDS_REVIEW. "
            "Provisional concern. Scout review required."
        )
    else:
        signal = (
            f"Mixed: {n_qualify} QUALIFY, {n_disqualify} DISQUALIFY, "
            f"{n_review} NEEDS_REVIEW (mean {mean_angle}°). Scout review required."
        )

    return {
        "pattern":      pattern_key,
        "display_name": pattern_cfg["display_name"],
        "source_file":  mp_json_path,
        "clip":         clip,
        "fps_recorded": fps,
        "provisional_thresholds": {
            "qualify_min_deg":          thr["qualify_min_deg"],
            "disqualify_max_deg":       thr["disqualify_max_deg"],
            "boundary_margin_deg":      thr["boundary_margin_deg"],
            "clear_qualify_zone_deg":   thr["clear_qualify_zone_deg"],
            "clear_disqualify_zone_deg": thr["clear_disqualify_zone_deg"],
            "min_confidence":           thr["min_confidence"],
            "status":                   "PROVISIONAL_NOT_VALIDATED",
        },
        "events": events,
        "clip_summary": {
            "n_total_toe_offs":       len(toe_offs),
            "n_excluded_undetectable": n_excluded,
            "n_valid_assessed":       n_valid,
            "n_qualify":              n_qualify,
            "n_disqualify":           n_disqualify,
            "n_needs_review":         n_review,
            "mean_angle_valid_deg":   mean_angle,
            "signal":                 signal,
            "threshold_status":       "PROVISIONAL_NOT_VALIDATED",
        },
    }


# ── 4. Print summary ──────────────────────────────────────────────────────────

def print_summary(report):
    """Print a human-readable screening summary to the console."""
    SEP  = "=" * 72
    DASH = "─" * 70

    print(f"\n{SEP}")
    print("  ATHLETE SCREENING SCAFFOLD  —  PROVISIONAL SUMMARY")
    print(f"  Clip     : {report['clip']}")
    print(f"  Athlete  : {report['athlete_id']}")
    print(f"  Generated: {report['generated_at']}")
    print(f"{SEP}")

    print()
    print("  ⚠  DISCLAIMER")
    for line in DISCLAIMER.splitlines():
        print(f"     {line}")
    print()

    for pkey, pat in report["patterns"].items():

        if "error" in pat:
            print(f"  [{pkey}]  ERROR: {pat['error']}")
            print()
            continue

        thr = pat["provisional_thresholds"]
        cs  = pat["clip_summary"]

        print(f"  PATTERN: {pat['display_name']}")
        print(f"  {DASH}")
        print(f"  Source      : {pat['source_file']}")
        print(f"  FPS         : {pat['fps_recorded']}fps  ⚠ 24fps = ±42ms/frame timing")
        print(f"  Thresholds  : QUALIFY ≥{thr['qualify_min_deg']}°  |  "
              f"DISQUALIFY <{thr['disqualify_max_deg']}°  |  margin ±{thr['boundary_margin_deg']}°  "
              f"[PROVISIONAL]")
        print(f"  Clear zones : QUALIFY ≥{thr['clear_qualify_zone_deg']}°  |  "
              f"DISQUALIFY ≤{thr['clear_disqualify_zone_deg']}°"
              f"   (threshold ± margin)")
        print(f"  Confidence  : hard verdict requires ≥{thr['min_confidence']}")
        print()

        # Event table
        col_w = {"t": 6, "side": 5, "val": 8, "conf": 6, "src": 16, "verd": 13}
        hdr = (f"  {'t(s)':>{col_w['t']}}  {'Side':{col_w['side']}}  "
               f"{'Value':>{col_w['val']}}  {'Conf':>{col_w['conf']}}  "
               f"{'SrcLabel':{col_w['src']}}  {'Verdict':{col_w['verd']}}  Reason (truncated)")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for ev in pat["events"]:
            val_s  = (f"{ev['measured_value_deg']:.1f}°"
                      if ev['measured_value_deg'] is not None else "N/A")
            conf_s = (f"{ev['confidence']:.2f}"
                      if ev['confidence'] is not None else "N/A")
            amb    = "⚠" if ev["ambiguous_flag"] else " "
            excl   = "*" if ev["source_label"] == "UNDETECTABLE" else " "
            reason = ev["verdict_reason"]
            short  = (reason[:55] + "…") if len(reason) > 55 else reason
            print(f"  {ev['time_s']:>{col_w['t']}.2f}  "
                  f"{ev['side']:{col_w['side']}}  "
                  f"{val_s:>{col_w['val']}}  "
                  f"{conf_s:>{col_w['conf']}}{amb}  "
                  f"{ev['source_label']:{col_w['src']}}  "
                  f"{ev['screening_verdict']:{col_w['verd']}}{excl}  "
                  f"{short}")

        print()
        print(f"  (* UNDETECTABLE = excluded from counts)")
        print()
        print(f"  COUNTS  :  QUALIFY={cs['n_qualify']}  |  "
              f"DISQUALIFY={cs['n_disqualify']}  |  "
              f"NEEDS_REVIEW={cs['n_needs_review']}  |  "
              f"excluded_UNDETECTABLE={cs['n_excluded_undetectable']}")
        print(f"  MEAN    :  {cs['mean_angle_valid_deg']}°  (valid events only)")
        print(f"  SIGNAL  :  {cs['signal']}")
        print(f"  STATUS  :  {cs['threshold_status']}")
        print()

    print(f"  OVERALL VERDICT:  {report['overall_verdict']}")
    print()
    print(f"  Output : {report['output_file']}")
    print(f"{SEP}\n")


# ── 5. Entry point ────────────────────────────────────────────────────────────

def main():
    clip_stem = sys.argv[1] if len(sys.argv) > 1 else "run_sideview"

    # Load threshold config
    thresholds = load_thresholds(THRESHOLDS_PATH)

    # ── Screen Pattern A1 ─────────────────────────────────────────────────────
    mp_json = os.path.join("hip_extension_output", "hip_extension_results.json")
    a1      = screen_hip_extension(thresholds, mp_json)

    # ── Assemble report ───────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{clip_stem}_screening.json")

    report = {
        "clip":              a1.get("clip", clip_stem),
        "athlete_id":        clip_stem,
        "generated_at":      datetime.datetime.now().isoformat(timespec="seconds"),
        "threshold_config":  THRESHOLDS_PATH,
        "threshold_status":  "PROVISIONAL_NOT_VALIDATED",
        "disclaimer":        DISCLAIMER,
        # Explicit: no overall verdict is ever computed
        "overall_verdict": (
            "NOT_COMPUTED — per SME brief D3, no single overall screening verdict "
            "is produced. Individual pattern signals are for scout review only. "
            "Thresholds are provisional and unvalidated."
        ),
        "patterns": {
            "A1_hip_extension_at_toeoff": a1,
        },
        "output_file": out_path,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)


if __name__ == "__main__":
    main()
