"""Discrimination threshold per condition (H5): standing vs walking.

Davidson et al. (2024) found that walking shifts the detection threshold, not the slope.
This tests the threshold for the 2-AFC tilt task: per participant and condition a Weibull
is fitted to the experiment trials (three tilt levels), the 75 %-correct threshold is
taken, and the conditions are compared with a paired test across participants (as in
Davidson Fig 2).

Three tilt levels do not identify the slope (cf. the QUEST+ review, Fig 6), so slope and
lapse are fixed and only alpha is fitted. The per-condition slope/lapse come from the
calibration joint refit (``calibration_refit.refit_from_bids``), where the adaptive trials
identify the slope, and that module's slope-invariance LRT is reported, so equal slopes
are tested rather than assumed. Without calibration data the QUEST+ design values
(beta = 2.5, lambda = 0.05) are used. With a fixed slope per condition, alpha and the 75 %
threshold are monotonically related.

Timeouts and training trials are excluded, as elsewhere in the pipeline (qc.py,
descriptive.py).

    python -m Analysis.threshold_by_condition
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
from scipy.optimize import minimize

try:
    from . import bids_io
    from . import calibration_refit as cref
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import calibration_refit as cref
    import plotstyle as S

S.apply()

BETA_FIX = 2.5          # QUEST+ design slope (StaircaseManager.cs)
LAPSE_FIX = 0.05        # QUEST+ design lapse
GUESS = 0.5             # 2-AFC chance
TARGET = 0.75           # 75 %-correct threshold
CONDITIONS = ("standing", "walking")
COND_COL = {"standing": S.ORANGE, "walking": S.TEAL}


# ===========================================================================
# Fit alpha (threshold) with slope and lapse fixed
# ===========================================================================
def fit_alpha(tilt, correct, beta=BETA_FIX, lapse=LAPSE_FIX) -> dict | None:
    """ML estimate of the Weibull threshold (alpha) with slope/lapse fixed.

    Returns alpha (deg), the 75 %-correct threshold (deg) and n; None if there are too few
    or degenerate data. ``ok`` is False if the estimate hits the search bounds (e.g.
    accuracy at ceiling)."""
    tilt = np.asarray(tilt, float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(tilt) & np.isfinite(correct)
    tilt, correct = tilt[ok], correct[ok]
    if tilt.size < 8 or np.unique(correct).size < 2:
        return None

    def nll(la):
        p = cref.weibull(tilt, np.exp(la[0]), beta, lapse, GUESS)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -float(np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p)))

    a0 = float(np.clip(np.median(tilt), 0.3, 25.0))
    lo, hi = np.log(0.1), np.log(45.0)
    res = minimize(nll, x0=[np.log(a0)], method="L-BFGS-B", bounds=[(lo, hi)])
    a = float(np.exp(res.x[0]))
    railed = a <= 0.12 or a >= 40.0
    return {"alpha_deg": round(a, 4),
            "thresh75_deg": round(float(cref._inverse(TARGET, a, beta, lapse, GUESS)), 4),
            "beta": round(float(beta), 4), "lapse": round(float(lapse), 4),
            "n": int(tilt.size), "ok": bool(res.success) and not railed,
            "railed": bool(railed)}


def _calibration_slopes():
    """Per-condition (slope, lapse) from the offline calibration joint refit.

    Returns ``(slopes, lapses, invariance)``: dicts by condition (well-fit conditions only)
    and the slope-invariance LRT result (or None). Empty without calibration data; the
    caller then uses the QUEST+ design values.
    """
    try:
        fit = cref.refit_from_bids()
    except Exception:
        fit = None
    slopes, lapses, inv = {}, {}, None
    if fit:
        for cond, res in fit.get("by_condition", {}).items():
            j = res.get("joint", {})
            if j.get("ok") and cond in CONDITIONS:
                slopes[cond] = float(j["slope_beta"])
                lapses[cond] = float(j["lapse"])
        inv = fit.get("slope_invariance")
    return slopes, lapses, inv


def _trials(rec):
    """(tilt_deg, correct) of answered non-training trials of one recording."""
    ev = bids_io.load_events(rec)
    if ev.empty:
        return np.array([]), np.array([])
    ev = ev[(ev["is_training"] == 0) & ev["response"].notna()]
    tilt = pd.to_numeric(ev["tilt_angle_deg"], errors="coerce").to_numpy()
    correct = (ev["response_correct"] == 1).astype(float).to_numpy()
    return tilt, correct


def _participant_fits(block_trials: dict, slope_lapse) -> dict:
    """One threshold per participant and condition, fitted to all blocks pooled.

    Same estimator as in the power analysis: fixed-slope Weibull MLE on all standing
    (2 blocks, ~360 trials) and walking (8 blocks, ~1440 trials) trials. Each visit has
    its own tilt levels; with a fixed slope this only improves the threshold estimate.

    ``block_trials``: sub -> condition -> list of (tilt, correct) per block.
    ``slope_lapse``: condition -> (slope, lapse). Returns sub -> condition -> fit dict
    (with ``n_blocks``) or None.
    """
    per: dict[str, dict] = {}
    for sub, by_cond in block_trials.items():
        for c, blocks in by_cond.items():
            b, lam = slope_lapse(c)
            f = fit_alpha(np.concatenate([t for t, _ in blocks]),
                          np.concatenate([k for _, k in blocks]), beta=b, lapse=lam)
            if f:
                f["n_blocks"] = len(blocks)
            per.setdefault(sub, {})[c] = f
    return per


# ===========================================================================
# Crossover model: condition x time on task
# ===========================================================================
def _crossover_model(cells: dict[str, dict]) -> dict:
    """Within-participant 2 (condition) x 2 (position in session) model on log10 threshold.

    Conditions are blocked (one switch per visit), so within a visit condition is
    confounded with time on task: the second condition is measured on a more fatigued,
    more adapted participant. Counterbalancing between participants only would leave this
    aliased with the allocation group. The order therefore crosses over between
    visits (`ExperimentConductor.StandingFirst`), so each participant fills all four cells
    and all effects are within-participant:

      * ``condition``   = mean over positions of (walking - standing): H2, free of time
        on task;
      * ``position``    = mean over conditions of (second - first): the time-on-task
        effect (manipulation check);
      * ``interaction`` = whether the walking cost depends on fatigue.

    Contrasts are computed per participant on log10 threshold (i.e. ratios) and tested
    one-sample. Participants missing a cell (e.g. only one visit so far) are skipped and
    counted in ``n_incomplete``.
    """
    from scipy.stats import ttest_1samp, wilcoxon

    keys = [("standing", "first"), ("standing", "second"),
            ("walking", "first"), ("walking", "second")]
    rows, incomplete = {}, 0
    for sub, cell in cells.items():
        got = {k: float(np.mean(np.log10(cell[k]))) for k in keys if cell.get(k)}
        if len(got) < 4:
            incomplete += 1
            continue
        rows[sub] = got
    if len(rows) < 2:
        return {"n_complete": len(rows), "n_incomplete": incomplete,
                "status": "not estimable — needs >=2 participants with all four "
                          "(condition x position) cells, i.e. both crossover visits",
                "effects": None}

    subs = sorted(rows)
    g = lambda c, p: np.array([rows[s][(c, p)] for s in subs])
    sf, ss = g("standing", "first"), g("standing", "second")
    wf, ws = g("walking", "first"), g("walking", "second")
    contrasts = {
        "condition_walking_minus_standing": 0.5 * ((wf + ws) - (sf + ss)),
        "position_second_minus_first": 0.5 * ((ss + ws) - (sf + wf)),
        "interaction_condition_x_position": (ws - wf) - (ss - sf),
    }

    effects = {}
    for name, d in contrasts.items():
        t = ttest_1samp(d, 0.0)
        try:
            w_p = float(wilcoxon(d).pvalue)
        except ValueError:
            w_p = None
        effects[name] = {
            "mean_log10_ratio": round(float(np.mean(d)), 4),
            "ratio": round(float(10 ** np.mean(d)), 4),
            "t": round(float(t.statistic), 3),
            "p": round(float(t.pvalue), 4),
            "wilcoxon_p": (round(w_p, 4) if w_p is not None else None),
            "significant": bool(t.pvalue < 0.05),
        }
    return {"n_complete": len(rows), "n_incomplete": incomplete,
            "status": "ok", "participants": subs, "effects": effects,
            "note": "condition effect is the H2 test, purged of time-on-task; "
                    "position is the fatigue/adaptation effect; a significant "
                    "interaction means the walking cost itself depends on time on task."}


# ===========================================================================
# Analysis
# ===========================================================================
def analyse() -> dict | None:
    """Compare tilt-discrimination thresholds between standing and walking (H5).

    Weibull per participant and condition over the three difficulty levels, with slope and
    lapse from the calibration joint refit; slope invariance is tested with a nested LRT.

    With both crossover visits, a 2 (condition) x 2 (position) model on log10 threshold
    estimates the time-on-task effect directly. With one visit it is not estimable, and
    the output states that the paired test is still confounded with time on task.
    """
    recs = [r for r in bids_io.find_recordings() if r.condition in CONDITIONS]
    if not recs:
        print("No standing/walking recordings found. Run Analysis/bidsify.py first.")
        return None

    slopes, lapses, invariance = _calibration_slopes()

    def _bl(c):                                    # (slope, lapse) for condition c
        return slopes.get(c, BETA_FIX), lapses.get(c, LAPSE_FIX)

    block_trials: dict[str, dict] = {}             # sub -> {condition: [(tilt, correct) per block]}
    cells: dict[str, dict] = {}                    # sub -> {(condition, position): [thr75,...]}
    pooled = {c: ([], []) for c in CONDITIONS}     # condition -> (tilts, correct)
    for rec in recs:
        tilt, correct = _trials(rec)
        if tilt.size == 0:
            continue
        block_trials.setdefault(rec.sub, {}).setdefault(rec.condition, []).append((tilt, correct))
        # Position of the condition within its visit (time-on-task covariate); one fit
        # per block, kept per (condition, position) cell for the crossover model.
        pos = (bids_io.scan_meta(rec) or {}).get("condition_position")
        if pos in ("first", "second"):
            b, lam = _bl(rec.condition)
            f = fit_alpha(tilt, correct, beta=b, lapse=lam)
            if f and f["ok"]:
                cells.setdefault(rec.sub, {}).setdefault((rec.condition, pos), []).append(
                    f["thresh75_deg"])
        pooled[rec.condition][0].append(tilt)
        pooled[rec.condition][1].append(correct)

    # Paired test input: one threshold per participant and condition from all blocks.
    per = _participant_fits(block_trials, _bl)

    # paired 75 % thresholds (both conditions present and well-fit)
    paired = [(s, v["standing"]["thresh75_deg"], v["walking"]["thresh75_deg"])
              for s, v in per.items()
              if v.get("standing") and v.get("walking")
              and v["standing"] and v["walking"]
              and v["standing"]["ok"] and v["walking"]["ok"]]

    group = None
    if len(paired) >= 2:
        st = np.array([p[1] for p in paired]); wk = np.array([p[2] for p in paired])
        log_ratio = np.log10(wk / st)
        from scipy.stats import ttest_rel, wilcoxon
        t_p = float(ttest_rel(np.log10(wk), np.log10(st)).pvalue)
        try:
            w_p = float(wilcoxon(wk, st).pvalue)
        except ValueError:
            w_p = None
        group = {
            "n_participants": len(paired),
            "median_threshold_standing_deg": round(float(np.median(st)), 4),
            "median_threshold_walking_deg": round(float(np.median(wk)), 4),
            "median_walk_over_stand_ratio": round(float(10 ** np.median(log_ratio)), 4),
            "mean_log10_ratio": round(float(np.mean(log_ratio)), 4),
            "paired_ttest_log_p": round(t_p, 4),
            "wilcoxon_p": (round(w_p, 4) if w_p is not None else None),
            "direction": ("walking > standing (impaired)" if np.median(log_ratio) > 0
                          else "walking < standing"),
            "significant_ttest": bool(t_p < 0.05),
        }

    pooled_fit = {}
    for c in CONDITIONS:
        if pooled[c][0]:
            b, lam = _bl(c)
            pooled_fit[c] = fit_alpha(np.concatenate(pooled[c][0]),
                                      np.concatenate(pooled[c][1]), beta=b, lapse=lam)

    report = {
        "test": "H5 — discrimination threshold standing vs walking",
        "slope_source": ("calibration joint refit" if slopes else "QUEST+ design default"),
        "slope_per_condition_used": {c: round(_bl(c)[0], 4) for c in CONDITIONS},
        "lapse_per_condition_used": {c: round(_bl(c)[1], 4) for c in CONDITIONS},
        "slope_invariance_test": invariance,
        "per_participant": {s: {c: v.get(c) for c in CONDITIONS} for s, v in per.items()},
        "pooled_fit_illustrative": pooled_fit,
        "group_paired": group,
        "crossover": _crossover_model(cells),
    }
    out_dir = bids_io.deriv_dir("conditions")
    (out_dir / "threshold_by_condition.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    _figure(per, pooled, pooled_fit, paired, group, bool(slopes),
            bids_io.fig_dir("conditions") / "threshold_by_condition.png")
    _print(report, paired, invariance)
    bids_io.gate_report("threshold", condition=CONDITIONS, n_analysed=len(paired))
    bids_io.gate_report("threshold", unit="calibrations")
    return report


# ===========================================================================
# Figure
# ===========================================================================
def _figure(per, pooled, pooled_fit, paired, group, slope_from_cal, out: Path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.7),
                                   gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    # (a) pooled psychometric fit per condition (illustrative) and binned accuracy
    xg = np.geomspace(0.3, 25, 200)
    for c in CONDITIONS:
        if not pooled[c][0]:
            continue
        tilt = np.concatenate(pooled[c][0]); cor = np.concatenate(pooled[c][1])
        # binned accuracy in log-spaced tilt bins
        edges = np.geomspace(max(0.3, tilt.min()), min(25, tilt.max() + 1e-6), 9)
        idx = np.clip(np.digitize(tilt, edges) - 1, 0, len(edges) - 2)
        cen = np.sqrt(edges[:-1] * edges[1:])
        acc = np.array([cor[idx == b].mean() if np.any(idx == b) else np.nan
                        for b in range(len(edges) - 1)])
        axA.scatter(cen, 100 * acc, s=22, color=COND_COL[c], alpha=0.55, zorder=2)
        f = pooled_fit.get(c)
        if f:
            p = cref.weibull(xg, f["alpha_deg"], f["beta"], f["lapse"], GUESS)
            axA.plot(xg, 100 * p, color=COND_COL[c], lw=2.4, zorder=3,
                     label=f"{c}: 75% thr = {f['thresh75_deg']:.2f}° (β={f['beta']:.2f})")
            axA.axvline(f["thresh75_deg"], color=COND_COL[c], ls=":", lw=1.1, zorder=1)
    axA.axhline(100 * TARGET, color=S.GREY, ls="--", lw=1.0, zorder=0)
    axA.set_xscale("log"); axA.set_xlim(0.3, 25); axA.set_ylim(45, 100)
    axA.set_xticks([0.3, 1, 2, 5, 10, 25]); axA.set_xticklabels(["0.3", "1", "2", "5", "10", "25"])
    axA.set_xlabel("Gabor tilt (deg)"); axA.set_ylabel("P(correct) (%)")
    S.panel(axA, "a", "Pooled psychometric fit (illustrative)", fontsize=10)
    axA.legend(loc="lower right", fontsize=8.5)

    # (b) paired 75 % thresholds
    for s, st, wk in paired:
        axB.plot([0, 1], [st, wk], color=S.MIST, lw=1.2, zorder=1)
        axB.scatter([0], [st], color=COND_COL["standing"], s=40, zorder=2)
        axB.scatter([1], [wk], color=COND_COL["walking"], s=40, zorder=2)
    if group:
        axB.scatter([0], [group["median_threshold_standing_deg"]], marker="_", s=600,
                    color=COND_COL["standing"], lw=2.5, zorder=3)
        axB.scatter([1], [group["median_threshold_walking_deg"]], marker="_", s=600,
                    color=COND_COL["walking"], lw=2.5, zorder=3)
        ttl = (f"ratio {group['median_walk_over_stand_ratio']:.2f}×, "
               f"p={group['paired_ttest_log_p']:.3f}")
    else:
        ttl = f"n={len(paired)} paired (need ≥2 for test)"
    axB.set_xticks([0, 1]); axB.set_xticklabels(CONDITIONS)
    axB.set_xlim(-0.5, 1.5); axB.set_ylabel("75%-correct threshold (deg)")
    S.panel(axB, "b", f"Paired threshold shift  ({ttl})", fontsize=10)

    src = "slope from calibration refit" if slope_from_cal else "slope fixed (β=2.5, no calibration)"
    fig.suptitle(f"Discrimination threshold: standing vs walking (H5)  —  {src}",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Per-condition Weibull fit; the slope/lapse are the per-condition "
                   "calibration joint-refit estimates (or the QUEST+ design values if no "
                   "calibration). Pooled curve is illustrative; the paired per-participant "
                   "thresholds (each fitted to all of that participant's blocks of the "
                   "condition) are the test. Timeouts excluded.")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); S.save(fig, out)


def _print(report, paired, invariance):
    print("=== Discrimination threshold: standing vs walking (H5) ===")
    print(f"  slope source: {report['slope_source']}  "
          f"(per-condition slope used: {report['slope_per_condition_used']})")
    g = report["group_paired"]
    if g:
        print(f"  {g['n_participants']} paired participants")
        print(f"  median 75% threshold: standing {g['median_threshold_standing_deg']}° vs "
              f"walking {g['median_threshold_walking_deg']}°  "
              f"({g['median_walk_over_stand_ratio']:.2f}×, {g['direction']})")
        print(f"  paired t (log thr): p={g['paired_ttest_log_p']:.4f} "
              f"-> {'SIGNIFICANT' if g['significant_ttest'] else 'n.s.'}; "
              f"Wilcoxon p={g['wilcoxon_p']}")
    else:
        print(f"  only {len(paired)} participant(s) with both conditions well-fit — "
              "need >=2 for the paired test (machinery validated).")
    if invariance and invariance.get("ok"):
        print(f"  slope-invariance (calibration LRT): slopes {invariance['slope_per_condition']}, "
              f"chi2({invariance['df']})={invariance['lr_stat']}, p={invariance['p_value']} "
              f"-> slope {'DIFFERS' if invariance['slope_differs'] else 'invariant (supports Davidson)'}")
    else:
        reason = (invariance or {}).get("reason", "no calibration data")
        print(f"  NOTE: slope-invariance not tested ({reason}) -> slope held at design value "
              "(assumed from Davidson).")

    x = report.get("crossover") or {}
    print("\n  --- crossover model (condition x time-on-task) ---")
    if x.get("status") != "ok":
        print(f"  {x.get('status', 'not run')}"
              f"  [{x.get('n_complete', 0)} complete, {x.get('n_incomplete', 0)} partial]")
        print("  Until this is estimable the paired test above is NOT purged of "
              "time-on-task: with blocked conditions, whichever condition runs second "
              "is measured on a more fatigued participant.")
    else:
        print(f"  {x['n_complete']} participant(s) with all four cells "
              f"({x['n_incomplete']} incomplete)")
        for name, e in x["effects"].items():
            print(f"    {name:34s} {e['ratio']:.3f}x  t={e['t']:+.2f}  p={e['p']:.4f}"
                  f"  -> {'SIGNIFICANT' if e['significant'] else 'n.s.'}")
        print("  condition = H2 free of time-on-task; position = the fatigue effect; "
              "interaction = walking cost depends on fatigue.")


def main():
    """Compare discrimination thresholds between standing and walking (H5)."""
    analyse()


if __name__ == "__main__":
    main()
