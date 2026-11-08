"""Standing versus walking: performance, the 75 % threshold (H5) and calibration transfer.

* Hit rate and RT per condition and difficulty (Fig 2).
* Threshold by condition (H5): a fixed-slope Weibull per condition, compared paired.
* Calibration transfer: whether each calibrated threshold holds in the blocks run on it.

Usage::

    python -m Analysis.conditions


Standing vs walking: hit rate and RT
====================================

Standing vs walking comparison, after Davidson et al. (2024) Fig 2.

Aggregates behaviour per (participant, condition) and plots:
  * a) overall hit rate, standing vs walking (paired);
  * b) hit rate by difficulty (HARD / THRESHOLD / EASY, targets 65 / 75 / 85 %), the
       counterpart of Davidson's psychometric function over staircase levels;
  * c) overall reaction time, standing vs walking (paired);
  * d) reaction time by difficulty and condition.

Hit rates are over answered trials: a timeout is a missing response, not a wrong one.
If only one condition is present, the other side is left empty.


Threshold by condition (H5)
===========================

Discrimination threshold per condition (H5): standing vs walking.

Davidson et al. (2024) found that walking shifts the detection threshold, not the slope.
This tests the threshold for the 2-AFC tilt task: per participant and condition a Weibull
is fitted to the experiment trials (three tilt levels), the 75 %-correct threshold is
taken, and the conditions are compared with a paired test across participants (as in
Davidson Fig 2).

Three tilt levels do not identify the slope (cf. the QUEST+ review, Fig 6), so slope and
lapse are fixed and only alpha is fitted. The per-condition slope/lapse come from the
calibration joint refit (``calibration.refit_from_bids``), where the adaptive trials
identify the slope, and that module's slope-invariance LRT is reported, so equal slopes
are tested rather than assumed. Without calibration data the QUEST+ design values
(beta = 2.5, lambda = 0.05) are used. With a fixed slope per condition, alpha and the 75 %
threshold are monotonically related.

Timeouts and training trials are excluded, as elsewhere in the pipeline (qc.py,
descriptive.py).


Calibration transfer
====================

Calibration transfer: does the calibrated threshold hold in the blocks that use it?

Each condition is calibrated once per visit (QUEST+, ~40-70 trials); the following blocks
present the THR level at the calibrated 75 %-correct tilt. If the calibration transfers,
THR accuracy is near 75 % and a threshold refitted to the block trials is near the
calibrated one.

Computed per calibration (participant x condition x visit), pooling its blocks:

* Block threshold: 75 % threshold fitted to the blocks' answered trials (all three tilt
  levels) with QUEST+ slope, lapse and guess fixed (``conditions.fit_alpha``),
  with a 95 % profile-likelihood interval.
* Transfer ratio = block threshold / calibrated threshold. Above 1 the blocks were harder
  than planned, below 1 easier.
* Sampling null: responses are simulated at the blocks' tilts for a participant whose true
  threshold is the calibrated one, then refitted. The 2.5-97.5 % range of simulated ratios
  reflects sampling alone. With perfect calibration 5 % of calibrations fall outside, so
  the count is tested against 5 % (one-sided binomial). Spread of observed log ratios
  beyond the simulated spread estimates the calibration error.

Only blocks included in the behavioural analyses by the exclusion gate are used, so a
participant not doing the task does not look like a calibration failure. The left figure
panel shows all blocks, excluded ones hollow.

Outputs (``derivatives/calibration_transfer/``):
  * ``calibration_transfer_blocks.tsv``  per block: THR tilt, THR accuracy, gate decision
  * ``calibration_transfer_units.tsv``   per calibration: thresholds, ratio, null range
  * ``calibration_transfer.json``  summary and the test
  * ``figures/calibration_transfer.png`` / ``.pdf``
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
from scipy.stats import binomtest

try:
    from . import bids_io
    from . import calibration as cref
    from . import exclusion
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import calibration as cref
    import exclusion
    import plotstyle as S

S.apply()


# ###########################################################################
# Standing vs walking: hit rate and RT
# ###########################################################################
CONDITIONS = ["standing", "walking"]
COND_COL = S.COND
DIFF_ORDER = ["HARD", "THR", "EASY"]      # increasing target accuracy (65/75/85 %)


def _aggregate():
    """One row per (participant, condition) with overall and per-difficulty hit rate and RT."""
    rows = []
    for rec in bids_io.find_recordings():
        if rec.condition not in CONDITIONS:
            continue
        ev = bids_io.load_events(rec)
        if ev.empty:
            continue
        # hit rate over the answered trials: a timeout is a missing response
        ev = ev[(ev["is_training"] == 0) & ev["response"].notna()].copy()
        ev["acc"] = (ev["response_correct"] == 1).astype(float)
        ev["rt"] = pd.to_numeric(ev["response_time"], errors="coerce")
        row = {"sub": rec.sub, "condition": rec.condition,
               "hit_rate": float(ev["acc"].mean()),
               "rt": float(ev["rt"].mean())}
        for d in DIFF_ORDER:
            g = ev[ev["difficulty"] == d]
            row[f"hr_{d}"] = float(g["acc"].mean()) if len(g) else np.nan
            row[f"rt_{d}"] = float(g["rt"].mean()) if len(g) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Average over a participant's blocks so each participant counts once per condition.
    num = [c for c in df.columns if c not in ("sub", "condition")]
    return df.groupby(["sub", "condition"], as_index=False)[num].mean()


def _paired(ax, df, col, ylabel):
    # Seeded jitter, so the figure is the same on every run; the pairing lines run to the
    # jittered points, not to the column centres.
    rng = np.random.default_rng(0)
    xs = {}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        x = CONDITIONS.index(cond)
        jit = x + rng.uniform(-0.06, 0.06, len(sub))
        xs.update({(s, cond): j for s, j in zip(sub["sub"], jit)})
        ax.scatter(jit, sub[col], color=COND_COL[cond], alpha=0.75, zorder=2)
        if len(sub):
            ax.errorbar(x + 0.22, sub[col].mean(), yerr=sub[col].sem() if len(sub) > 1 else 0,
                        fmt="o", ms=9, mfc="none", mew=1.8, color=COND_COL[cond], zorder=3,
                        capsize=3)
    for s in df["sub"].unique():
        d = df[df["sub"] == s].set_index("condition")
        if set(d.index) == set(CONDITIONS):
            ax.plot([xs[(s, c)] for c in CONDITIONS], [d.loc[c, col] for c in CONDITIONS],
                    color=S.MIST, zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel(ylabel); ax.set_xlim(-0.5, 1.6)


def _within_sem(df, prefix):
    """Cousineau-Morey within-participant SEM per (condition, difficulty) cell.

    Each participant's cells are centred on their own mean and moved to the grand mean,
    which removes the between-participant level differences that a plain SEM carries
    (large for RT), then scaled by sqrt(J / (J - 1)) for the J cells (Morey 2008). Only
    participants with every cell enter.
    """
    cols = [f"{prefix}_{d}" for d in DIFF_ORDER]
    wide = df.pivot(index="sub", columns="condition", values=cols).dropna()
    if len(wide) < 2:
        return None
    vals = wide.to_numpy(float)
    norm = vals - vals.mean(axis=1, keepdims=True) + vals.mean()
    j = vals.shape[1]
    sem = norm.std(axis=0, ddof=1) / np.sqrt(len(wide)) * np.sqrt(j / (j - 1))
    return dict(zip(wide.columns, sem))


def _by_difficulty(ax, df, prefix, ylabel):
    x = np.arange(len(DIFF_ORDER))
    wsem = _within_sem(df, prefix)
    for k, cond in enumerate(CONDITIONS):
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        means = [sub[f"{prefix}_{d}"].mean() for d in DIFF_ORDER]
        sems = ([wsem[(f"{prefix}_{d}", cond)] for d in DIFF_ORDER] if wsem else
                [sub[f"{prefix}_{d}"].sem() if len(sub) > 1 else 0 for d in DIFF_ORDER])
        ax.errorbar(x + (k - 0.5) * 0.06, means, yerr=sems, fmt="-o",
                    color=COND_COL[cond], label=cond, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{d.lower()}\n({p})" for d, p in
                                          zip(DIFF_ORDER, ["65%", "75%", "85%"])])
    ax.set_xlabel("difficulty level (target % correct)"); ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def cond_main():
    """Compare standing against walking on hit rate and reaction time (Fig 2)."""
    df = _aggregate()
    bids_io.gate_report("conditions", condition=CONDITIONS,
                        n_analysed=int(df["sub"].nunique()) if not df.empty else 0)
    if df.empty:
        print("No standing/walking recordings found. Run Analysis/bidsify.py first.")
        return
    present = sorted(df["condition"].unique())
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    _paired(ax[0, 0], df, "hit_rate", "accuracy (P correct)"); S.panel(ax[0, 0], "a", "Accuracy")
    _by_difficulty(ax[0, 1], df, "hr", "accuracy (P correct)")
    S.panel(ax[0, 1], "b", "Accuracy by difficulty level")
    _paired(ax[1, 0], df, "rt", "mean reaction time (s)"); S.panel(ax[1, 0], "c", "Reaction time")
    _by_difficulty(ax[1, 1], df, "rt", "mean reaction time (s)")
    S.panel(ax[1, 1], "d", "Reaction time by difficulty level")

    n = int(df["sub"].nunique())
    fig.suptitle(f"Standing vs walking, n = {n} participants", fontsize=13,
                 fontweight="semibold", x=0.012, ha="left")
    note = ("Answered main trials; each participant's value is the mean over their blocks "
            "of that condition. a, c: one point per participant, lines pair them; open "
            "circle = mean ± SEM. b, d: mean ± within-participant SEM (Cousineau-Morey), "
            "which leaves out how much participants differ overall. Descriptive; the "
            "threshold contrast is tested by the H5 threshold comparison.")
    if set(present) != set(CONDITIONS):
        note = f"Only {', '.join(present)} present — standing block pending. " + note
    S.caption(fig, note)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = bids_io.fig_dir("conditions") / "conditions_standing_vs_walking.png"
    S.save(fig, out)

    print(f"Conditions present: {present}")
    for _, r in df.iterrows():
        print(f"  {r['sub']} {r['condition']:8s}: hit rate {r['hit_rate']:.3f}, RT {r['rt']:.3f}s")
    if set(present) != set(CONDITIONS):
        print("  NOTE: standing block not yet recorded — Fig 2 will populate both sides "
              "once a standing session (condition=standing) exists.")
    print(f"\nFigure -> {out}")


# ###########################################################################
# Threshold by condition (H5)
# ###########################################################################
BETA_FIX = 2.5          # QUEST+ design slope (StaircaseManager.cs)
LAPSE_FIX = 0.05        # QUEST+ design lapse
GUESS = 0.5             # 2-AFC chance
THRESHOLD_TARGET = 0.75           # 75 %-correct threshold
THRESHOLD_CONDITIONS = ("standing", "walking")
THRESHOLD_COND_COL = {"standing": S.ORANGE, "walking": S.TEAL}


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
            "thresh75_deg": round(float(cref._inverse(THRESHOLD_TARGET, a, beta, lapse, GUESS)), 4),
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
            if j.get("ok") and cond in THRESHOLD_CONDITIONS:
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
def threshold_analyse() -> dict | None:
    """Compare tilt-discrimination thresholds between standing and walking (H5).

    Weibull per participant and condition over the three difficulty levels, with slope and
    lapse from the calibration joint refit; slope invariance is tested with a nested LRT.

    With both crossover visits, a 2 (condition) x 2 (position) model on log10 threshold
    estimates the time-on-task effect directly. With one visit it is not estimable, and
    the output states that the paired test is still confounded with time on task.
    """
    recs = [r for r in bids_io.find_recordings() if r.condition in THRESHOLD_CONDITIONS]
    if not recs:
        print("No standing/walking recordings found. Run Analysis/bidsify.py first.")
        return None

    slopes, lapses, invariance = _calibration_slopes()

    def _bl(c):                                    # (slope, lapse) for condition c
        return slopes.get(c, BETA_FIX), lapses.get(c, LAPSE_FIX)

    block_trials: dict[str, dict] = {}             # sub -> {condition: [(tilt, correct) per block]}
    cells: dict[str, dict] = {}                    # sub -> {(condition, position): [thr75,...]}
    pooled = {c: ([], []) for c in THRESHOLD_CONDITIONS}     # condition -> (tilts, correct)
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
    for c in THRESHOLD_CONDITIONS:
        if pooled[c][0]:
            b, lam = _bl(c)
            pooled_fit[c] = fit_alpha(np.concatenate(pooled[c][0]),
                                      np.concatenate(pooled[c][1]), beta=b, lapse=lam)

    report = {
        "test": "H5 — discrimination threshold standing vs walking",
        "slope_source": ("calibration joint refit" if slopes else "QUEST+ design default"),
        "slope_per_condition_used": {c: round(_bl(c)[0], 4) for c in THRESHOLD_CONDITIONS},
        "lapse_per_condition_used": {c: round(_bl(c)[1], 4) for c in THRESHOLD_CONDITIONS},
        "slope_invariance_test": invariance,
        "per_participant": {s: {c: v.get(c) for c in THRESHOLD_CONDITIONS} for s, v in per.items()},
        "pooled_fit_illustrative": pooled_fit,
        "group_paired": group,
        "crossover": _crossover_model(cells),
    }
    out_dir = bids_io.deriv_dir("conditions")
    (out_dir / "threshold_by_condition.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    _threshold_figure(per, block_trials, pooled_fit, paired, group, bool(slopes),
            bids_io.fig_dir("conditions") / "threshold_by_condition.png")
    _print(report, paired, invariance)
    bids_io.gate_report("threshold", condition=THRESHOLD_CONDITIONS, n_analysed=len(paired))
    bids_io.gate_report("threshold", unit="calibrations")
    return report


# ===========================================================================
# Figure
# ===========================================================================
def _threshold_figure(per, block_trials, pooled_fit, paired, group, slope_from_cal, out: Path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.7),
                                   gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    # (a) accuracy against tilt in units of each participant's own threshold, with the
    # psychometric shape the thresholds were fitted with. Raw tilts pooled over people
    # whose thresholds differ ~4-fold would smear the curve flat and could not be
    # compared with it.
    xg = np.geomspace(0.1, 20, 200)
    edges = np.geomspace(0.25, 8, 11)
    cen = np.sqrt(edges[:-1] * edges[1:])
    for k, c in enumerate(THRESHOLD_CONDITIONS):
        rel, cor = [], []
        for s, conds in block_trials.items():
            f_s = (per.get(s) or {}).get(c)
            if not (f_s and f_s.get("ok")) or c not in conds:
                continue
            for tilt, correct in conds[c]:
                rel.append(np.asarray(tilt, float) / f_s["thresh75_deg"])
                cor.append(np.asarray(correct, float))
        if not rel:
            continue
        rel, cor = np.concatenate(rel), np.concatenate(cor)
        idx = np.digitize(rel, edges) - 1
        n = np.array([np.sum(idx == b) for b in range(len(cen))], float)
        acc = np.array([cor[idx == b].mean() if n[b] else np.nan for b in range(len(cen))])
        se = np.sqrt(acc * (1 - acc) / np.where(n > 0, n, np.nan))
        ok = n >= 20
        axA.errorbar(cen[ok] * (1 + (k - 0.5) * 0.04), 100 * acc[ok], yerr=100 * se[ok],
                     fmt="o", ms=5, color=THRESHOLD_COND_COL[c], capsize=0, zorder=3,
                     label=f"{c}: trials, binned")
        f = pooled_fit.get(c)
        if f:
            a_rel = f["alpha_deg"] / f["thresh75_deg"]   # Weibull scale at threshold 1
            p = cref.weibull(xg, a_rel, f["beta"], f["lapse"], GUESS)
            axA.plot(xg, 100 * p, color=THRESHOLD_COND_COL[c], lw=2.0, zorder=2,
                     label=f"{c}: model (β = {f['beta']:.2f}, λ = {f['lapse']:.3f})")
    axA.axhline(100 * THRESHOLD_TARGET, color=S.GREY, ls="--", lw=1.0, zorder=0)
    axA.axvline(1.0, color=S.GREY, ls=":", lw=1.0, zorder=0)
    axA.set_xscale("log"); axA.set_xlim(0.2, 10); axA.set_ylim(40, 102)
    axA.set_xticks([0.25, 0.5, 1, 2, 4, 8]); axA.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    axA.set_xlabel("tilt / participant's own 75 % threshold")
    axA.set_ylabel("accuracy (% correct)")
    S.panel(axA, "a", "Psychometric shape on a common scale", fontsize=10)
    axA.legend(loc="lower right", fontsize=8)

    # (b) paired 75 % thresholds
    for s, st, wk in paired:
        axB.plot([0, 1], [st, wk], color=S.MIST, lw=1.2, zorder=1)
        axB.scatter([0], [st], color=THRESHOLD_COND_COL["standing"], s=40, zorder=2)
        axB.scatter([1], [wk], color=THRESHOLD_COND_COL["walking"], s=40, zorder=2)
    if group:
        axB.scatter([0], [group["median_threshold_standing_deg"]], marker="_", s=600,
                    color=THRESHOLD_COND_COL["standing"], lw=2.5, zorder=3)
        axB.scatter([1], [group["median_threshold_walking_deg"]], marker="_", s=600,
                    color=THRESHOLD_COND_COL["walking"], lw=2.5, zorder=3)
        ttl = (f"ratio {group['median_walk_over_stand_ratio']:.2f}×, "
               f"p={group['paired_ttest_log_p']:.3f}")
    else:
        ttl = f"n={len(paired)} paired (need ≥2 for test)"
    axB.set_xticks([0, 1]); axB.set_xticklabels(THRESHOLD_CONDITIONS)
    axB.set_xlim(-0.5, 1.5); axB.set_ylabel("75%-correct threshold (deg)")
    S.panel(axB, "b", f"Paired threshold shift  ({ttl})", fontsize=10)

    src = "slope from calibration refit" if slope_from_cal else "slope fixed (β=2.5, no calibration)"
    fig.suptitle(f"Discrimination threshold: standing vs walking (H5)  —  {src}",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Each participant's 75 % threshold per condition is a Weibull fit to all "
                   "their blocks of it (answered trials), with slope β and lapse λ fixed at the "
                   "calibration refit. a: every trial's tilt divided by that threshold, "
                   "accuracy ± binomial SE in log bins with ≥ 20 trials; the line is the model "
                   "the thresholds assume. b: the thresholds, lines pair a participant, bars = "
                   "medians; p = paired t-test on log thresholds.")
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


def threshold_main():
    """Compare discrimination thresholds between standing and walking (H5)."""
    threshold_analyse()


# ###########################################################################
# Calibration transfer
# ###########################################################################
TRANSFER_TARGET = THRESHOLD_TARGET          # 75 % correct at THR
N_SIM = 400                  # simulated blocks per calibration
SEED = 20261007


# ===========================================================================
# Threshold fit with a profile-likelihood interval
# ===========================================================================
GRID = np.linspace(np.log(0.05), np.log(60.0), 700)     # log alpha search grid


def _nll_grid(tilt, correct):
    """Negative log-likelihood for every alpha on ``GRID``."""
    p = cref.weibull(tilt[None, :], np.exp(GRID)[:, None], BETA_FIX, LAPSE_FIX,
                     GUESS)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).sum(axis=1)


def _to_t75(alpha):
    return float(cref._inverse(TRANSFER_TARGET, alpha, BETA_FIX, LAPSE_FIX, GUESS))


def fit_threshold(tilt, correct) -> dict | None:
    """75 % threshold of answered trials (slope/lapse/guess fixed) with 95 % profile interval.

    ``censored`` is "high" or "low" if the fit hits a grid bound (accuracy at chance or
    ceiling); then only one side is informative."""
    tilt = np.asarray(tilt, float)
    correct = np.asarray(correct, float)
    ok = np.isfinite(tilt) & np.isfinite(correct)
    tilt, correct = tilt[ok], correct[ok]
    if tilt.size < 20:
        return None
    nll = _nll_grid(tilt, correct)
    i = int(np.argmin(nll))
    inside = nll <= nll[i] + 1.92                       # chi2(1) 95 % / 2
    lo, hi = GRID[inside].min(), GRID[inside].max()
    censored = "high" if i == len(GRID) - 1 else "low" if i == 0 else None
    return {"t75": _to_t75(np.exp(GRID[i])), "lo": _to_t75(np.exp(lo)),
            "hi": _to_t75(np.exp(hi)), "n": int(tilt.size), "censored": censored}


def sampling_null(tilt, t75_cal, rng, n_sim: int = N_SIM) -> np.ndarray:
    """Transfer ratios of a perfectly calibrated participant at these tilts."""
    tilt = np.asarray(tilt, float)
    alpha = t75_cal / _to_t75(1.0)                      # alpha with t75 == t75_cal
    p = cref.weibull(tilt, alpha, BETA_FIX, LAPSE_FIX, GUESS)
    # All simulated blocks are fitted at once; draws, order, filtering and grid match
    # calling fit_threshold per block.
    draws = rng.random((n_sim, tilt.size))
    fin = np.isfinite(tilt)
    if fin.sum() < 20:
        return np.asarray([])
    correct = (draws[:, fin] < p[fin]).astype(float)
    tilt = tilt[fin]
    pg = cref.weibull(tilt[None, :], np.exp(GRID)[:, None], BETA_FIX, LAPSE_FIX,
                      GUESS)
    pg = np.clip(pg, 1e-6, 1 - 1e-6)
    nll = -(correct @ np.log(pg).T + (1 - correct) @ np.log(1 - pg).T)    # (n_sim, grid)
    best = np.argmin(nll, axis=1)
    return np.array([_to_t75(np.exp(GRID[i])) for i in best]) / t75_cal


# ===========================================================================
# Data
# ===========================================================================
def _blocks() -> pd.DataFrame:
    """One row per block: calibration unit, THR tilt and accuracy, gate decision."""
    gate = bids_io.load_gate()
    beh = dict(zip(gate["recording"], gate["behaviour"]))
    rows = []
    for rec in bids_io.find_recordings(include_excluded=True):
        ev = bids_io.load_events(rec)
        if ev.empty:
            continue
        ev = ev[ev["is_training"] == 0]
        ans = ev[pd.to_numeric(ev["response_correct"], errors="coerce").isin([0, 1])]
        thr = ans[ans["difficulty"] == "THR"]
        tilt_thr = pd.to_numeric(ev.loc[ev["difficulty"] == "THR", "tilt_angle_deg"],
                                 errors="coerce").abs().median()
        visit = (bids_io.scan_meta(rec) or {}).get("visit")
        rows.append({
            "recording": rec.label(), "participant": rec.sub, "condition": rec.condition,
            "visit": int(visit) if visit is not None else np.nan,
            "run": rec.run,
            "thr_tilt_deg": float(tilt_thr),
            "n_thr_answered": int(len(thr)),
            "thr_accuracy": float((thr["response_correct"] == 1).mean()) if len(thr) else np.nan,
            "calibration_verdict": exclusion.calibration_verdict(rec.sub, rec.condition, visit),
            "behaviour": beh.get(rec.label(), "include"),
            "_tilt": pd.to_numeric(ans["tilt_angle_deg"], errors="coerce").abs().to_numpy(),
            "_correct": (ans["response_correct"] == 1).astype(float).to_numpy(),
        })
    return pd.DataFrame(rows)


def _units(blocks: pd.DataFrame, rng) -> pd.DataFrame:
    """Pool each calibration's included blocks, fit, and build the sampling null."""
    rows = []
    keep = blocks[blocks["behaviour"] == "include"]
    for (sub, cond, visit), g in keep.groupby(["participant", "condition", "visit"]):
        t_cal = float(g["thr_tilt_deg"].median())
        tilt = np.concatenate(g["_tilt"].to_list())
        corr = np.concatenate(g["_correct"].to_list())
        fit = fit_threshold(tilt, corr)
        if fit is None or not np.isfinite(t_cal) or t_cal <= 0:
            continue
        null = sampling_null(tilt, t_cal, rng)
        ratio = fit["t75"] / t_cal
        n_lo, n_hi = (np.quantile(null, [0.025, 0.975]) if len(null) else (np.nan, np.nan))
        rows.append({
            "participant": sub, "condition": cond, "visit": int(visit),
            "n_blocks": int(len(g)), "n_trials": fit["n"],
            "calibrated_t75_deg": round(t_cal, 4),
            "block_t75_deg": round(fit["t75"], 4),
            "block_t75_lo": round(fit["lo"], 4), "block_t75_hi": round(fit["hi"], 4),
            "censored": fit["censored"] or "-",
            "ratio": round(ratio, 3),
            "null_lo": round(float(n_lo), 3), "null_hi": round(float(n_hi), 3),
            "null_sd_log": round(float(np.std(np.log(null))), 4) if len(null) else np.nan,
            "outside_null": bool(ratio < n_lo or ratio > n_hi),
            "thr_accuracy": round(float(np.average(g["thr_accuracy"],
                                                   weights=g["n_thr_answered"])), 4),
            "calibration_verdict": g["calibration_verdict"].iloc[0],
        })
    return pd.DataFrame(rows)


def _summary(units: pd.DataFrame) -> dict:
    n = len(units)
    if not n:
        return {"n_calibrations": 0}
    k = int(units["outside_null"].sum())
    rel = units["calibration_verdict"].astype(str).str.startswith("RELIABLE")
    k_rel = int(units.loc[rel, "outside_null"].sum())
    # Censored fits give only a bound: counted as outside, excluded from the spread.
    unc = units[units["censored"] == "-"]
    logr = np.log(unc["ratio"].to_numpy())
    sd_obs = float(np.std(logr, ddof=1)) if len(unc) > 1 else float("nan")
    sd_null = float(np.sqrt(np.mean(unc["null_sd_log"] ** 2)))
    excess = float(np.sqrt(max(sd_obs ** 2 - sd_null ** 2, 0.0)))
    by_cond = {c: {"n": int(len(g)), "median_ratio": round(float(g["ratio"].median()), 3),
                   "median_thr_accuracy": round(float(g["thr_accuracy"].median()), 3)}
               for c, g in units.groupby("condition")}
    return {
        "n_calibrations": n, "n_participants": int(units["participant"].nunique()),
        "outside_sampling_range": k,
        "p_outside_vs_5pct": round(float(binomtest(k, n, 0.05, alternative="greater").pvalue), 6),
        "n_reliable": int(rel.sum()), "outside_sampling_range_reliable": k_rel,
        "censored": [f"{r.participant} {r.condition} visit {r.visit} ({r.censored})"
                     for r in units.itertuples() if r.censored != "-"],
        "n_for_spread": int(len(unc)),
        "sd_log_ratio_observed": round(sd_obs, 4),
        "sd_log_ratio_sampling": round(sd_null, 4),
        "calibration_error_factor": round(float(np.exp(excess)), 3),
        "thr_accuracy_range": [round(float(units["thr_accuracy"].min()), 3),
                               round(float(units["thr_accuracy"].max()), 3)],
        "ratio_range_uncensored": [round(float(unc["ratio"].min()), 3),
                                   round(float(unc["ratio"].max()), 3)],
        "by_condition": by_cond,
    }


# ===========================================================================
# Figure
# ===========================================================================
def _transfer_figure(blocks: pd.DataFrame, units: pd.DataFrame, summ: dict, stem: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(S.COL2_IN, 3.9),
                                     gridspec_kw={"width_ratios": [1.45, 1.0]})
    # Legends go under the panels: inside, either one would sit on data.
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.3, top=0.9, wspace=0.32)

    # a: THR accuracy per block and participant
    subs = sorted(blocks["participant"].unique())
    n_med = int(blocks.loc[blocks["behaviour"] == "include", "n_thr_answered"].median())
    lo, hi = (np.array([np.quantile(np.random.default_rng(SEED).binomial(n_med, TRANSFER_TARGET, 20000),
                                    q) for q in (0.025, 0.975)]) / n_med)
    ax_a.axhspan(lo, hi, color=S.GRID, zorder=0, lw=0)
    S.reference_line(ax_a, y=TRANSFER_TARGET)
    S.reference_line(ax_a, y=0.5, ls=":")
    for i, sub in enumerate(subs):
        g = blocks[blocks["participant"] == sub].sort_values(["condition", "visit", "run"])
        for j, (_, r) in enumerate(g.iterrows()):
            x = i + (j - (len(g) - 1) / 2) * min(0.08, 0.7 / max(len(g), 1))
            col = S.COND[r["condition"]]
            if r["behaviour"] == "include":
                ax_a.plot(x, r["thr_accuracy"], "o", ms=4.2, color=col, zorder=3)
            else:      # a cross, so open circles keep the one meaning they have in b
                ax_a.plot(x, r["thr_accuracy"], "x", ms=4.2, mew=1.1, color=col, zorder=3)
    ax_a.set_xticks(range(len(subs)))
    ax_a.set_xticklabels([s.replace("sub-", "") for s in subs])
    ax_a.set_xlabel("Participant")
    ax_a.set_ylabel("Accuracy at the calibrated tilt")
    ax_a.set_ylim(0.2, 1.02)
    ax_a.set_xlim(-0.6, len(subs) - 0.4)
    ax_a.grid(axis="x", visible=False)
    S.panel(ax_a, "a", "Blocks against the 75 % target")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4.5, mec=S.COND[c], mfc=S.COND[c],
                          label=c) for c in ("standing", "walking")]
    handles.append(plt.Line2D([], [], ls="", marker="x", ms=4.5, mew=1.1, color=S.GREY,
                              label="excluded block"))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=S.GRID,
                                 label="95 % range if on target"))
    ax_a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
                fontsize=8)

    # b: calibrated vs block threshold per calibration
    lim = (0.25, 8.0)
    xs = np.geomspace(*lim, 50)
    ax_b.plot(xs, xs, color=S.GREY, lw=1.0, zorder=1)
    for f in (0.5, 2.0):
        ax_b.plot(xs, xs * f, color=S.GREY, lw=0.8, ls=":", zorder=1)
    for _, u in units.iterrows():
        col = S.COND[u["condition"]]
        mk = "o" if str(u["calibration_verdict"]).startswith("RELIABLE") else "D"
        y = u["block_t75_deg"]
        if u["censored"] != "-":                    # bound only: marker at the axis edge
            y = lim[0] * 1.12 if u["censored"] == "low" else lim[1] / 1.12
            ax_b.plot(u["calibrated_t75_deg"], y, "v" if u["censored"] == "low" else "^",
                      ms=6, color=col, mfc=col, zorder=3)
        else:
            yerr = [[y - u["block_t75_lo"]], [u["block_t75_hi"] - y]]
            ax_b.errorbar(u["calibrated_t75_deg"], y, yerr=yerr, fmt=mk, ms=5, color=col,
                          mfc=col if u["outside_null"] else "white", mew=1.1,
                          elinewidth=0.9, capsize=0, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(*lim)
    ax_b.set_ylim(*lim)
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    for axis in (ax_b.xaxis, ax_b.yaxis):
        axis.set_major_locator(plt.FixedLocator(ticks))
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}°"))
        axis.set_minor_locator(plt.NullLocator())
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("Calibrated threshold")
    ax_b.set_ylabel("Threshold in the blocks")
    S.panel(ax_b, "b", "Calibration vs blocks")
    hb = [plt.Line2D([], [], ls="", marker="o", ms=5, mec=S.INK, mfc=S.INK,
                     label="missed the target"),
          plt.Line2D([], [], ls="", marker="o", ms=5, mec=S.INK, mfc="white",
                     label="transferred"),
          plt.Line2D([], [], ls="", marker="D", ms=4.5, mec=S.INK, mfc=S.INK,
                     label="unreliable calib."),
          plt.Line2D([], [], ls="", marker="v", ms=5, mec=S.INK, mfc=S.INK,
                     label="ceiling (bound)"),
          plt.Line2D([], [], color=S.GREY, lw=0.8, ls=":", label="factor 2")]
    ax_b.legend(handles=hb, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                fontsize=8, handletextpad=0.3, columnspacing=0.8)
    S.save_pub(fig, stem)


# ===========================================================================
# Main
# ===========================================================================
def transfer_analyse() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Blocks, per-calibration units and the summary of how the calibrations transfer."""
    rng = np.random.default_rng(SEED)
    blocks = _blocks()
    units = _units(blocks, rng) if not blocks.empty else pd.DataFrame()
    return blocks, units, _summary(units)


def transfer_main() -> dict:
    """Fit, test and plot the calibration transfer for every calibration."""
    blocks, units, summ = transfer_analyse()
    bids_io.gate_report("calibration_transfer", condition=("standing", "walking"),
                        analysed_subjects=sorted(units["participant"].unique())
                        if not units.empty else [])
    if units.empty:
        print("No blocks with a calibration to compare. Run the pipeline first.")
        return summ
    out = bids_io.deriv_dir("calibration_transfer")
    blocks.drop(columns=["_tilt", "_correct"]).to_csv(
        out / "calibration_transfer_blocks.tsv", sep="\t", index=False, na_rep="n/a",
        float_format="%.4f")
    units.to_csv(out / "calibration_transfer_units.tsv", sep="\t", index=False, na_rep="n/a")
    (out / "calibration_transfer.json").write_text(json.dumps({
        "Description": "Does the calibrated 75 % threshold hold in the blocks run on it? "
                       "Per calibration (participant x condition x visit), the blocks' "
                       "own threshold (slope/lapse/guess fixed at the QUEST+ design "
                       "values) against the calibrated one, with a simulated sampling "
                       "null (conditions.py).",
        "Model": {"beta": BETA_FIX, "lapse": LAPSE_FIX, "guess": GUESS,
                  "target": TRANSFER_TARGET, "n_sim": N_SIM, "seed": SEED},
        **summ}, indent=2), encoding="utf-8")
    _transfer_figure(blocks, units, summ, bids_io.fig_dir("calibration_transfer") / "calibration_transfer")

    print("=== Calibration transfer: calibrated vs block threshold ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(units[["participant", "condition", "visit", "n_blocks", "thr_accuracy",
                     "calibrated_t75_deg", "block_t75_deg", "ratio", "null_lo", "null_hi",
                     "outside_null", "censored", "calibration_verdict"]].to_string(index=False))
    print(f"  {summ['outside_sampling_range']}/{summ['n_calibrations']} calibrations outside "
          f"their sampling range (5 % expected; binomial p={summ['p_outside_vs_5pct']:.2g})")
    print(f"  SD of log ratio: observed {summ['sd_log_ratio_observed']:.3f} vs sampling "
          f"{summ['sd_log_ratio_sampling']:.3f} (uncensored n={summ['n_for_spread']}) -> "
          f"calibration error factor {summ['calibration_error_factor']:.2f}")
    print(f"  RELIABLE calibrations only: {summ['outside_sampling_range_reliable']}/"
          f"{summ['n_reliable']} outside; censored: {summ['censored'] or 'none'}")
    for c, d in summ["by_condition"].items():
        print(f"  {c}: n={d['n']}, median ratio {d['median_ratio']}, median THR accuracy "
              f"{d['median_thr_accuracy']}")
    print(f"Outputs -> {out}")
    return summ


# ###########################################################################
# Pipeline stage
# ###########################################################################
def main() -> None:
    """Pipeline stage ``conditions``: performance, H5 thresholds, calibration transfer."""
    cond_main()
    threshold_main()
    transfer_main()


if __name__ == "__main__":
    main()
