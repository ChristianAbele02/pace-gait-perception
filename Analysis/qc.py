"""Quality control for the BIDS motion recordings.

Checks head tracking, sample quality, treadmill and behaviour. Writes a JSON report and
a figure per recording to ``derivatives/qc/``.

Usage::

    python -m Analysis.qc                 # QC every recording in the dataset
    python Analysis/qc.py                 # same
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import bids_io
    from . import clean
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import clean
    import plotstyle as S

# Frozen-pose runs longer than this are flagged as tracking dropouts (e.g. a ~9 s
# OptiTrack stall).
STALE_RUN_WARN_S = 1.0

S.apply()


def _valid_pos_mask(motion: pd.DataFrame) -> np.ndarray:
    """True where the local head position is within LOCAL_POS_VALID_RADIUS_M of the
    per-axis median; pre-init and dropout frames hold values like -505.
    """
    cols = ["head_local_pos_x", "head_local_pos_y", "head_local_pos_z"]
    cols = [c for c in cols if c in motion]
    if not cols:
        return np.ones(len(motion), bool)
    ok = np.ones(len(motion), bool)
    for c in cols:
        v = motion[c].to_numpy()
        med = np.nanmedian(v)
        ok &= np.abs(v - med) <= C.LOCAL_POS_VALID_RADIUS_M
    return ok


def _timing_qc(t: np.ndarray) -> dict:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    nominal = 1.0 / C.NOMINAL_FS
    dropped = int(np.sum(dt > 1.5 * nominal))
    return {
        "n_samples": int(t.size),
        "duration_s": round(float(t[-1] - t[0]), 2),
        "fs_effective_hz": round(1.0 / float(np.median(dt)), 2),
        "dt_median_ms": round(float(np.median(dt)) * 1e3, 2),
        "dt_max_ms": round(float(dt.max()) * 1e3, 2),
        "dropped_frames_gt_1p5x": dropped,
        "dropped_frame_pct": round(100 * dropped / dt.size, 2),
    }


def _tracking_qc(motion: pd.DataFrame, valid: np.ndarray) -> dict:
    out = {"n_invalid_pos_samples": int((~valid).sum()),
           "invalid_pos_pct": round(100 * (~valid).sum() / len(motion), 3),
           "channels": {}}
    for c in motion.columns:
        if c == "t":
            continue
        v = motion[c].to_numpy()
        vv = v[valid] if c.startswith("head_") else v
        vv = vv[np.isfinite(vv)]
        if vv.size == 0:
            out["channels"][c] = {"flat_zero": True, "note": "no finite samples"}
            continue
        nonzero_frac = float(np.mean(vv != 0))
        out["channels"][c] = {
            "min": round(float(vv.min()), 4),
            "max": round(float(vv.max()), 4),
            "std": round(float(vv.std()), 5),
            "nonzero_frac": round(nonzero_frac, 3),
            "flat_zero": bool(np.allclose(vv, 0)),
        }
    # orientation verdict: tracked vs flat channels
    ori = [c for c in motion.columns if "_ori_" in c]
    tracked = [c for c in ori if not out["channels"][c].get("flat_zero", False)]
    out["orientation_verdict"] = {
        "tracked_channels": tracked,
        "flat_zero_channels": [c for c in ori if c not in tracked],
    }
    # Frozen-pose frames: mocap_frame_advanced == 0 means the previous mocap frame was
    # reused (tracking stall) even if head_mocap_valid == 1. They are flat, so the spike
    # cleaner misses them; gait.py excises them. Without the flag, repeated poses are used
    # (clean.py).
    stale, stale_source = clean.stale_frames(motion)
    if stale is not None:
        t = motion["t"].to_numpy() if "t" in motion else np.arange(len(motion), dtype=float)
        longest, run_start = 0.0, None
        i = 0
        while i < stale.size:
            if stale[i]:
                j = i
                while j < stale.size and stale[j]:
                    j += 1
                dur = float(t[min(j, t.size - 1)] - t[i])
                if dur > longest:
                    longest, run_start = dur, float(t[i] - t[0])
                i = j
            else:
                i += 1
        out["stale_pose"] = {
            "n_stale": int(stale.sum()),
            "pct_stale": round(100.0 * stale.sum() / max(1, stale.size), 2),
            "longest_run_s": round(longest, 2),
            "longest_run_start_s": round(run_start, 1) if run_start is not None else None,
            "flag": longest > STALE_RUN_WARN_S,
            "source": stale_source,
        }
    return out


def _treadmill_qc(motion: pd.DataFrame) -> dict:
    if "treadmill_speed" not in motion:
        return {}
    s = motion["treadmill_speed"].to_numpy()
    glitches = int(np.sum(s > C.MAX_PLAUSIBLE_SPEED_MPS))
    moving = s[(s > 0.05) & (s < C.MAX_PLAUSIBLE_SPEED_MPS)]
    t = motion["t"].to_numpy()
    onset_idx = np.flatnonzero((s[:-1] <= 0.05) & (s[1:] > 0.05))
    return {
        "median_moving_speed_mps": round(float(np.median(moving)), 4) if moving.size else None,
        "moving_fraction": round(float(np.mean((s > 0.05) & (s < C.MAX_PLAUSIBLE_SPEED_MPS))), 3),
        "speed_glitches_gt_max": glitches,
        "first_movement_time_s": round(float(t[onset_idx[0] + 1]), 2) if onset_idx.size else None,
    }


# Vigilance thresholds. EASY trials are far above threshold, so an attentive participant
# gets nearly all right regardless of stride phase. They serve as the attention check in
# the experiment blocks (which have no catch trials) and make up 10 % of each block.
EASY_VIGILANCE_MIN = 0.80      # lower easy-trial accuracy is flagged
EASY_VIGILANCE_DROP = 0.15     # first-third minus last-third drop that is flagged
EASY_MIN_PER_THIRD = 5         # minimum easy trials per third


def _vigilance_qc(real: "pd.DataFrame") -> dict:
    """Attention check from EASY-trial accuracy, overall and by block thirds.

    Easy trials (10 % of each block) are near ceiling and insensitive to the stride-phase
    effect. Low or declining easy-trial accuracy indicates inattention.

    Only flagged. The exclusion gate (block_rules.py) does not use easy-trial accuracy.
    """
    out = {"n_easy": 0}
    if real.empty or "difficulty" not in real.columns:
        return out
    easy = real[real["difficulty"].astype(str).str.lower() == "easy"]
    n = len(easy)
    out["n_easy"] = int(n)
    if n == 0:
        return out
    acc = pd.to_numeric(easy["response_correct"], errors="coerce")
    out["easy_accuracy"] = round(float(acc.mean()), 3)
    out["easy_timeouts"] = int(easy["response"].isna().sum())
    # Thirds by trial order within the block.
    k = n // 3
    if k >= EASY_MIN_PER_THIRD:
        first = float(acc.iloc[:k].mean())
        last = float(acc.iloc[-k:].mean())
        out["easy_accuracy_first_third"] = round(first, 3)
        out["easy_accuracy_last_third"] = round(last, 3)
        out["easy_accuracy_drop"] = round(first - last, 3)
        out["n_easy_per_third"] = int(k)
    return out


def _behaviour_qc(events: pd.DataFrame) -> dict:
    if events.empty:
        return {"n_events": 0}
    real = events[events["is_training"] == 0]
    resp = real[real["response"].notna()]
    n_resp = len(resp)
    right = int((resp["response"] == "right").sum())
    left = int((resp["response"] == "left").sum())
    acc = real["response_correct"]
    out = {
        "n_events_total": int(len(events)),
        "n_training": int((events["is_training"] == 1).sum()),
        "n_analysis_trials": int(len(real)),
        "n_timeouts": int(real["response"].isna().sum()),
        "accuracy_overall": round(float(pd.to_numeric(acc, errors="coerce").mean()), 3),
        "accuracy_by_difficulty": {
            k: round(float(pd.to_numeric(g["response_correct"], errors="coerce").mean()), 3)
            for k, g in real.groupby("difficulty")
        },
        "response_left": left,
        "response_right": right,
        "response_right_bias": round(right / n_resp, 3) if n_resp else None,
        "stimulus_left": int((real["stimulus_side"] == "left").sum()),
        "stimulus_right": int((real["stimulus_side"] == "right").sum()),
        "median_rt_s": round(float(pd.to_numeric(resp["response_time"], errors="coerce").median()), 3)
                       if n_resp else None,
    }
    out.update(_vigilance_qc(real))
    # flags
    flags = []
    if out["response_right_bias"] is not None and not (0.35 <= out["response_right_bias"] <= 0.65):
        flags.append(f"response side bias: {right}R/{left}L "
                     f"({out['response_right_bias']:.0%} right)")
    if out["accuracy_overall"] < 0.6:
        flags.append(f"low overall accuracy ({out['accuracy_overall']:.0%})")
    ev = out.get("easy_accuracy")
    if ev is not None and ev < EASY_VIGILANCE_MIN:
        flags.append(f"vigilance: easy-trial accuracy only {ev:.0%} "
                     f"over {out['n_easy']} easy trials (expected >{EASY_VIGILANCE_MIN:.0%})")
    drop = out.get("easy_accuracy_drop")
    if drop is not None and drop >= EASY_VIGILANCE_DROP:
        flags.append(f"vigilance: easy-trial accuracy fell {drop:.0%} from the first to "
                     f"the last third of the block "
                     f"({out['easy_accuracy_first_third']:.0%} -> "
                     f"{out['easy_accuracy_last_third']:.0%})")
    out["flags"] = flags
    return out


_CLEAN_CHANNELS = (("head_local_pos_y", "vertical head (gait)", S.INK),
                   ("head_local_pos_x", "lateral head (sway)", S.BLUE))


def _figure(rec, motion, events, valid, report, cleaned, clean_per, out_png: Path) -> None:
    """QC sheet per block: head-position channels raw vs cleaned with correction sizes,
    belt speed and target onsets.

    Cleaning is only measured here (gait.py applies it); excised spikes are marked and the
    histogram shows correction sizes.
    """
    t = motion["t"].to_numpy()
    chans = [(c, lab, col) for c, lab, col in _CLEAN_CHANNELS
             if c in motion.columns and c in cleaned]
    n = len(chans) + 2
    fig = plt.figure(figsize=(15, 2.4 * n + 0.9))
    gs = fig.add_gridspec(n, 2, width_ratios=[4.2, 1], height_ratios=[1] * (n - 1) + [0.6],
                          hspace=0.6, wspace=0.14)
    cv = report.get("position_cleaning", {}).get("aggregate", {})
    fig.suptitle(f"Quality control — {S.nice_label(rec.label())}"
                 + (f"  ·  cleaning {cv.get('verdict')}" if cv else ""),
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    tags = iter("abcdefgh")
    first = None

    for row, (c, lab, col) in enumerate(chans):
        a = fig.add_subplot(gs[row, 0], sharex=first)
        first = first or a
        raw = motion[c].to_numpy()
        cln = np.asarray(cleaned[c], float)
        a.plot(t[valid], raw[valid], lw=0.6, color=S.MIST, label="raw", zorder=1)
        a.plot(t, cln, lw=0.6, color=col, label="cleaned", zorder=2)
        exc = ~np.isfinite(cln)
        if (~valid).any():
            a.scatter(t[~valid], np.full((~valid).sum(), np.nanmedian(raw[valid])), s=10,
                      color=S.RED, zorder=3, label=f"invalid / dropout ({int((~valid).sum())})")
        if exc.any():
            a.scatter(t[exc], raw[exc], s=14, color=S.ORANGE, marker="x", zorder=4,
                      label=f"excised spike ({int(exc.sum())})")
        a.legend(loc="lower left", ncol=4, fontsize=7.5)
        a.set_ylabel(f"{lab.split(' (')[0]}\n(m)")
        a.tick_params(labelbottom=False)
        info = clean_per.get(c, {})
        S.panel(a, next(tags), f"{lab.capitalize()}: {info.get('pct_flagged')}% flagged, "
                               f"{info.get('clean_coverage_pct')}% coverage", fontsize=10)

        # |raw - cleaned| where repaired; excised samples are compared with the cleaned median.
        h = fig.add_subplot(gs[row, 1])
        corr_mm = np.where(np.isfinite(cln), np.abs(raw - cln),
                           np.abs(raw - np.nanmedian(cln))) * 1000.0
        changed = corr_mm[corr_mm > 1e-4]
        if changed.size:
            h.hist(changed, bins=40, color=S.PURPLE, alpha=0.85)
            h.set_yscale("log")
            h.set_xlabel("correction (mm)")
        else:
            h.text(0.5, 0.5, "no sample\ncorrected", ha="center", va="center",
                   transform=h.transAxes, color=S.GREY, fontsize=9)
            h.set_xticks([]); h.set_yticks([])
        S.panel(h, "", f"{changed.size / max(raw.size, 1):.2%} corrected"
                       + (f", max {changed.max():.0f} mm" if changed.size else ""), fontsize=9)

    a = fig.add_subplot(gs[len(chans), 0], sharex=first)
    if "treadmill_speed" in motion:
        s = motion["treadmill_speed"].to_numpy()
        a.plot(t, np.where(s > C.MAX_PLAUSIBLE_SPEED_MPS, np.nan, s), lw=0.7, color=S.TEAL)
    td = report.get("treadmill") or {}
    td = td if td.get("median_moving_speed_mps") is not None else {}
    k = fig.add_subplot(gs[len(chans), 1])
    k.axis("off")
    k.text(0.02, 0.5, "\n".join((f"median {td.get('median_moving_speed_mps')} m/s",
                                 f"moving {td.get('moving_fraction') or 0:.0%} of the block",
                                 f"starts at {td.get('first_movement_time_s')} s",
                                 f"speed glitches: {td.get('speed_glitches_gt_max')}"))
           if td else "belt stopped (standing)", transform=k.transAxes, va="center",
           fontsize=9, color=S.INK)
    a.set_ylabel("belt speed\n(m/s)")
    a.tick_params(labelbottom=False)
    S.panel(a, next(tags), "Treadmill belt speed", fontsize=10)

    a = fig.add_subplot(gs[len(chans) + 1, 0], sharex=first)
    if not events.empty:
        for _, r in events.dropna(subset=["onset"]).iterrows():
            corr = r["response_correct"]
            a.axvline(r["onset"], lw=0.8, alpha=0.7,
                      color=S.OUTCOME["correct"] if corr == 1
                      else S.OUTCOME["wrong"] if corr == 0 else S.GREY)
    a.set_yticks([])
    a.set_xlabel("time (s)")
    S.panel(a, next(tags), "Target onsets  (green = correct, red = wrong, grey = timeout)",
            fontsize=10)

    b = report.get("behaviour") or {}
    acc = b.get("accuracy_by_difficulty") or {}
    k = fig.add_subplot(gs[len(chans) + 1, 1])
    if acc:
        levels = [lv for lv in ("EASY", "THR", "HARD") if lv in acc]
        k.barh(levels, [acc[lv] for lv in levels], color=S.GREEN, alpha=0.85)
        k.axvline(0.5, color=S.GREY, ls="--", lw=0.9)
        k.set_xlim(0, 1); k.invert_yaxis()
        k.set_xlabel("accuracy")
        S.panel(k, "", f"{b.get('accuracy_overall')} correct, "
                       f"{b.get('n_timeouts')} timeouts", fontsize=9)
    else:
        k.axis("off")
    S.save(fig, out_png)


def qc_recording(rec) -> dict:
    """QC of one recording: timing, tracking and behaviour.

    Checks effective sampling rate and dropped frames, tracking validity and position
    spikes, whether head orientation varies on all three axes (yaw-only is a known
    failure), response side bias and accuracy per difficulty level.

    Cleaning is measured, not applied; gait.py does the repair.
    """
    motion, channels, meta = bids_io.load_motion(rec)
    events = bids_io.load_events(rec)
    t = motion["t"].to_numpy()
    valid = _valid_pos_mask(motion)

    # Measure spikes/dropouts in the head-position channels (data are not changed).
    cleaned, clean_per, clean_agg, *_ = clean.clean_motion_positions(motion)

    report = {
        "recording": rec.label(),
        "task": rec.task,
        "tracking_system": rec.tracksys,
        "timing": _timing_qc(t),
        "tracking": _tracking_qc(motion, valid),
        "position_cleaning": {"aggregate": clean_agg, "per_channel": clean_per},
        "treadmill": _treadmill_qc(motion),
        "behaviour": _behaviour_qc(events),
    }

    out_dir = bids_io.deriv_dir("qc", rec.sub, rec.ses)
    (out_dir / f"{rec.label()}_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _figure(rec, motion, events, valid, report, cleaned, clean_per,
            bids_io.fig_dir("qc", rec.sub, rec.ses) / f"{rec.label()}_qc.png")
    return report


def _print_report(r: dict) -> None:
    print(f"\n=== QC: {r['recording']}  (task-{r['task']}) ===")
    tm = r["timing"]
    print(f"  timing   : {tm['n_samples']} samples, {tm['duration_s']}s, "
          f"{tm['fs_effective_hz']} Hz; dropped {tm['dropped_frames_gt_1p5x']} "
          f"({tm['dropped_frame_pct']}%), max gap {tm['dt_max_ms']} ms")
    tr = r["tracking"]
    print(f"  position : {tr['n_invalid_pos_samples']} invalid/dropout samples "
          f"({tr['invalid_pos_pct']}%)")
    pc = r.get("position_cleaning", {}).get("aggregate")
    if pc:
        print(f"  cleaning : {pc['pct_flagged']}% flagged, {pc['n_excised']} excised; "
              f"coverage {pc['clean_coverage_pct']}%, longest clean run "
              f"{pc['longest_clean_run_s']}s, max spike {pc['max_excursion_m']}m "
              f"-> {pc['verdict']}")
    ov = tr["orientation_verdict"]
    print(f"  orient.  : tracked={ov['tracked_channels'] or 'NONE'}; "
          f"flat-zero={len(ov['flat_zero_channels'])}/6 channels")
    td = r["treadmill"]
    if td:
        print(f"  treadmill: {td.get('median_moving_speed_mps')} m/s median, "
              f"moving {td.get('moving_fraction')}, glitches={td.get('speed_glitches_gt_max')}, "
              f"start {td.get('first_movement_time_s')}s")
    b = r["behaviour"]
    if b.get("n_events"):
        return
    print(f"  behaviour: {b['n_analysis_trials']} trials ({b['n_training']} training, "
          f"{b['n_timeouts']} timeouts); acc={b['accuracy_overall']} "
          f"{b['accuracy_by_difficulty']}")
    print(f"             responses {b['response_right']}R/{b['response_left']}L; "
          f"stimuli {b['stimulus_right']}R/{b['stimulus_left']}L; RT~{b['median_rt_s']}s")
    if b.get("n_easy"):
        drop = b.get("easy_accuracy_drop")
        print(f"             vigilance: easy-trial acc {b['easy_accuracy']:.0%} "
              f"over {b['n_easy']} trials"
              + (f", first->last third {b['easy_accuracy_first_third']:.0%}"
                 f"->{b['easy_accuracy_last_third']:.0%} ({drop:+.0%})"
                 if drop is not None else ""))
    for f in b.get("flags", []):
        print(f"  [FLAG]   : {f}")


def main() -> None:
    """Run QC on every recording and write figures and JSON."""
    # Excluded recordings are included; their QC documents the exclusion.
    recs = bids_io.find_recordings(include_excluded=True)
    if not recs:
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return
    behaviour = bids_io.excluded_recordings(scope="behaviour")
    gait = bids_io.excluded_recordings(scope="gait")
    for rec in recs:
        _print_report(qc_recording(rec))
        if rec.label() in gait:
            print(f"  [EXCLUDED from every analysis]: {gait[rec.label()]}")
        elif rec.label() in behaviour:
            print(f"  [EXCLUDED from behavioural analyses, gait kept]: "
                  f"{behaviour[rec.label()]}")
    print(f"\nQC reports + figures -> {C.DERIV_DIR / 'qc'}")


if __name__ == "__main__":
    main()
