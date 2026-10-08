"""Descriptive overview of the whole dataset.

No inferential statistics. Summarises behaviour, stimulus, gait and tracking quality
per participant and condition for a first look at the data. Inferential analyses are in
`phase_analysis`, `glmm` and `population`.

Choices:
  * proportions (accuracy, response-side bias) use Wilson 95 % score intervals, which
    behave well at small n and near 0/1;
  * chance (0.5) and the target accuracies (HARD 65 % / THR 75 % / EASY 85 %) are drawn
    as reference lines;
  * each panel states its n, pooling across participants is labelled, and timeouts are
    excluded from accuracy denominators (as in `qc.py`).

Outputs (`derivatives/descriptive/`):
  * ``descriptive_behaviour.png``       accuracy, accuracy×difficulty, response bias, RT
  * ``descriptive_quality.png``         per-participant gait + tracking quality (walking)
  * ``descriptive_overview.tsv``        per-participant × condition table

Usage::

    python -m Analysis.descriptive
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
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import plotstyle as S

S.apply()

DIFF_ORDER = ["HARD", "THR", "EASY"]
DIFF_TARGET = {"HARD": 0.65, "THR": 0.75, "EASY": 0.85}   # intended % correct
COND_COLOR = S.COND


# ===========================================================================
# Stats helper
# ===========================================================================
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


# ===========================================================================
# Trial-level and per-recording data
# ===========================================================================
def _collect():
    """Return (trials_df, gait_by_rec, participants_df)."""
    recs = bids_io.find_recordings()
    frames = []
    gait = {}
    for rec in recs:
        ev = bids_io.load_events(rec)
        if not ev.empty:
            ev = ev[ev["is_training"] == 0].copy()
            ev["participant"] = rec.sub
            ev["condition"] = rec.condition
            ev["correct"] = pd.to_numeric(ev["response_correct"], errors="coerce")
            ev["rt"] = pd.to_numeric(ev["response_time"], errors="coerce")
            ev["tilt"] = pd.to_numeric(ev["tilt_angle_deg"], errors="coerce")
            frames.append(ev[["participant", "condition", "difficulty", "correct",
                              "response", "stimulus_side", "rt", "tilt"]])
        if rec.condition == "walking":
            gj = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.json"
            if gj.exists():
                gait[rec.sub] = json.loads(gj.read_text())
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ptsv = C.BIDS_DIR / "participants.tsv"
    participants = pd.read_csv(ptsv, sep="\t", na_values=["n/a"]) if ptsv.exists() else pd.DataFrame()
    return trials, gait, participants


def _per_cell(trials: pd.DataFrame) -> pd.DataFrame:
    """Behavioural summary per participant × condition, with Wilson CIs."""
    rows = []
    for (p, cond), g in trials.groupby(["participant", "condition"]):
        resp = g[g["response"].notna()]
        acc = g["correct"].dropna()
        k, n = int(acc.sum()), int(acc.notna().sum())
        pacc, acc_lo, acc_hi = wilson(k, n)
        kr = int((resp["response"] == "right").sum())
        nr = int(len(resp))
        pr, r_lo, r_hi = wilson(kr, nr)
        ks = int((g["stimulus_side"] == "right").sum())
        rows.append({
            "participant": p, "condition": cond, "n_trials": int(len(g)), "n_responded": nr,
            "accuracy": pacc, "acc_lo": acc_lo, "acc_hi": acc_hi,
            "p_right_resp": pr, "right_lo": r_lo, "right_hi": r_hi,
            "p_right_stim": ks / len(g) if len(g) else np.nan,
            "median_rt": float(resp["rt"].median()) if nr else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["participant", "condition"])


# ===========================================================================
# Figure 1: behaviour
# ===========================================================================
def _fig_behaviour(trials, cell, out: Path):
    parts = sorted(trials["participant"].unique())
    conds = [c for c in ("standing", "walking") if c in trials["condition"].unique()]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Descriptive behaviour overview", fontsize=13, fontweight="semibold",
                 x=0.012, ha="left")

    # (a) accuracy by condition, paired per participant
    a = ax[0, 0]
    xpos = {c: i for i, c in enumerate(conds)}
    for p in parts:
        sub = cell[cell["participant"] == p].set_index("condition")
        xs = [xpos[c] for c in conds if c in sub.index]
        ys = [sub.loc[c, "accuracy"] for c in conds if c in sub.index]
        yerr = [[sub.loc[c, "accuracy"] - sub.loc[c, "acc_lo"] for c in conds if c in sub.index],
                [sub.loc[c, "acc_hi"] - sub.loc[c, "accuracy"] for c in conds if c in sub.index]]
        a.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, lw=1, label=p)
    S.reference_line(a, y=0.5, label="chance")
    a.set_xticks(list(xpos.values())); a.set_xticklabels(conds)
    a.set_ylim(0.3, 1.0); a.set_ylabel("accuracy (P correct)")
    S.panel(a, "a", "Accuracy by condition (paired per participant)", fontsize=10)
    a.legend(fontsize=7, ncol=2)

    # (b) accuracy × difficulty × condition (mean ± SD across participants), with targets
    b = ax[0, 1]
    for cond in conds:
        means, sds, xs = [], [], []
        for i, d in enumerate(DIFF_ORDER):
            vals = []
            for p in parts:
                g = trials[(trials["participant"] == p) & (trials["condition"] == cond)
                           & (trials["difficulty"] == d)]["correct"].dropna()
                if len(g) >= 3:
                    vals.append(g.mean())
            if vals:
                means.append(np.mean(vals)); sds.append(np.std(vals)); xs.append(i)
        b.errorbar(xs, means, yerr=sds, marker="o", capsize=3, lw=1.5,
                   color=COND_COLOR.get(cond, S.INK), label=cond)
    for i, d in enumerate(DIFF_ORDER):
        b.hlines(DIFF_TARGET[d], i - 0.25, i + 0.25, color=S.GREY, ls=":", lw=1.4,
                 label="target" if i == 0 else None)
    S.reference_line(b, y=0.5)
    b.set_xticks(range(len(DIFF_ORDER)))
    b.set_xticklabels([f"{d}\n(target {int(DIFF_TARGET[d]*100)}%)" for d in DIFF_ORDER])
    b.set_ylim(0.3, 1.0); b.set_ylabel("accuracy (P correct)")
    S.panel(b, "b", "Accuracy × difficulty  (mean ± SD across participants)", fontsize=10)
    b.legend()

    # (c) response-side bias per participant × condition (Wilson CI) and stimulus balance
    c = ax[1, 0]
    labels, idx = [], 0
    for p in parts:
        for cond in conds:
            r = cell[(cell["participant"] == p) & (cell["condition"] == cond)]
            if r.empty:
                continue
            r = r.iloc[0]
            yerr = [[r["p_right_resp"] - r["right_lo"]], [r["right_hi"] - r["p_right_resp"]]]
            c.errorbar([idx], [r["p_right_resp"]], yerr=yerr, marker="o", capsize=3,
                       color=COND_COLOR.get(cond, S.INK))
            c.plot(idx, r["p_right_stim"], marker="_", ms=12, mew=2, color=S.GREY)
            labels.append(f"{p[-2:]}/{cond[0]}")
            idx += 1
    S.reference_line(c, y=0.5, label="unbiased (0.5)")
    c.set_xticks(range(len(labels))); c.set_xticklabels(labels, rotation=90, fontsize=7)
    c.set_ylim(0, 1); c.set_ylabel("P(respond 'right')")
    S.panel(c, "c", "Response-side bias  (colour = condition; grey ‘_’ = stimulus P(right))",
            fontsize=10)
    c.legend()

    # (d) RT distribution by condition (correct trials), per-participant medians overlaid
    d = ax[1, 1]
    data = [trials[(trials["condition"] == cond) & (trials["correct"] == 1)]["rt"].dropna().to_numpy()
            for cond in conds]
    vp = d.violinplot(data, showmedians=True)
    for j, body in enumerate(vp["bodies"]):
        body.set_facecolor(COND_COLOR.get(conds[j], S.GREY)); body.set_alpha(0.4)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if key in vp:
            vp[key].set_color(S.GREY)
    for i, cond in enumerate(conds, 1):
        meds = [cell[(cell["participant"] == p) & (cell["condition"] == cond)]["median_rt"].iloc[0]
                for p in parts
                if not cell[(cell["participant"] == p) & (cell["condition"] == cond)].empty]
        d.scatter(np.full(len(meds), i), meds, color=S.INK, s=18, zorder=3)
    d.set_xticks(range(1, len(conds) + 1)); d.set_xticklabels(conds)
    d.set_ylabel("reaction time (s), correct trials")
    S.panel(d, "d", "RT distribution by condition  (dots = per-participant medians)",
            fontsize=10)

    S.caption(fig, "Exploratory; error bars = Wilson 95% CI.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    S.save(fig, out)


# ===========================================================================
# Figure 2: gait and tracking quality (walking)
# ===========================================================================
def _fig_quality(gait, participants, out: Path):
    parts = sorted(gait.keys())
    if not parts:
        return
    def gv(p, k):
        v = gait[p].get(k)
        return float(v) if isinstance(v, (int, float)) else np.nan

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Descriptive gait & tracking quality (walking runs)", fontsize=13,
                 fontweight="semibold", x=0.012, ha="left")
    x = np.arange(len(parts))
    short = [p[-2:] for p in parts]

    # Twin-axis panels (a, c, d) use coloured y labels instead of legends.
    # a) cadence and step CoV
    a = ax[0, 0]
    a.bar(x - 0.2, [gv(p, "cadence_spm") for p in parts], 0.4, color=S.BLUE)
    a.grid(False)
    a2 = a.twinx(); a2.grid(False)
    a2.bar(x + 0.2, [gv(p, "step_cov_pct") for p in parts], 0.4, color=S.RED)
    a.set_xticks(x); a.set_xticklabels(short)
    a.set_ylabel("cadence (spm)", color=S.BLUE); a2.set_ylabel("step CoV (%)", color=S.RED)
    a2.spines["right"].set_visible(True)
    S.panel(a, "a", "Cadence & step-time variability", fontsize=10)

    # b) L/R alternation and Hilbert-trough agreement (reliability of the phase axis)
    b = ax[0, 1]
    b.bar(x - 0.2, [gv(p, "lr_alternation_pct") for p in parts], 0.4, color=S.GREEN, label="L/R alternation (%)")
    b.bar(x + 0.2, [100 * gv(p, "hilbert_trough_phase_corr") for p in parts], 0.4,
          color=S.PURPLE, label="Hilbert–trough agree (×100)")
    b.axhline(85, color=S.GREY, ls="--", lw=1, label="usable ≥ 85%")
    b.set_xticks(x); b.set_xticklabels(short); b.set_ylabel("%"); b.set_ylim(0, 105)
    S.panel(b, "b", "Stride-phase axis trustworthiness", fontsize=10)
    b.legend(loc="lower right")

    # c) clean coverage and dropped frames (tracking quality)
    c = ax[1, 0]
    c.bar(x - 0.2, [gv(p, "clean_coverage_pct") for p in parts], 0.4, color=S.TEAL)
    c.grid(False)
    c2 = c.twinx(); c2.grid(False)
    c2.bar(x + 0.2, [gv(p, "dropped_frame_pct") for p in parts], 0.4, color=S.ORANGE)
    c.set_xticks(x); c.set_xticklabels(short)
    c.set_ylabel("clean coverage (%)", color=S.TEAL); c.set_ylim(90, 100.5)
    c2.set_ylabel("dropped frames (%)", color=S.ORANGE)
    c2.spines["right"].set_visible(True)
    S.panel(c, "c", "Tracking quality", fontsize=10)

    # d) calibrated threshold and treadmill speed (from participants.tsv)
    d = ax[1, 1]; d.grid(False)
    d2 = d.twinx(); d2.grid(False)
    if not participants.empty:
        participants = participants.set_index("participant_id")
        thr = [participants.loc[p, "threshold_angle_deg"] if p in participants.index else np.nan for p in parts]
        spd = [participants.loc[p, "treadmill_speed_mps"] if p in participants.index else np.nan for p in parts]
        d.bar(x - 0.2, thr, 0.4, color=S.GOLD)
        d2.bar(x + 0.2, spd, 0.4, color=S.GREY)
        d.set_ylabel("threshold tilt (°)", color="#B9962E")
        d2.set_ylabel("treadmill speed (m/s)", color=S.GREY)
        d2.spines["right"].set_visible(True)
    d.set_xticks(x); d.set_xticklabels(short)
    S.panel(d, "d", "Calibrated threshold & belt speed", fontsize=10)

    S.caption(fig, "Exploratory.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    S.save(fig, out)


def main():
    """Write the descriptive behaviour and data-quality overview."""
    trials, gait, participants = _collect()
    bids_io.gate_report("descriptive", n_analysed=(
        int(trials["participant"].nunique()) if not trials.empty else 0))
    if trials.empty:
        print("No trial events found. Run the pipeline (bidsify) first.")
        return
    cell = _per_cell(trials)
    out_dir = bids_io.deriv_dir("descriptive")
    cell.to_csv(out_dir / "descriptive_overview.tsv", sep="\t", index=False,
                na_rep="n/a", float_format="%.4f")
    figs = bids_io.fig_dir("descriptive")
    _fig_behaviour(trials, cell, figs / "descriptive_behaviour.png")
    _fig_quality(gait, participants, figs / "descriptive_quality.png")

    print("=== Descriptive overview (exploratory) ===")
    print(f"  {trials['participant'].nunique()} participants, "
          f"{len(trials)} analysis trials across {sorted(trials['condition'].unique())}")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(cell[["participant", "condition", "n_trials", "accuracy",
                    "p_right_resp", "median_rt"]].to_string(index=False))
    print(f"\nDescriptive outputs -> {out_dir}")


if __name__ == "__main__":
    main()
