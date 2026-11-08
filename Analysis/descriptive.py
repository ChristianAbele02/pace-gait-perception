"""Descriptive overview of the whole dataset.

No inferential statistics. Summarises behaviour, stimulus, gait and tracking quality
per participant and condition for a first look at the data. Inferential analyses are in
`phase_analysis`, `population` and `bayes_glmm`.

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
  * ``sample_overview.png|pdf``         manuscript: trials kept per participant, each
                                        calibration's threshold, stride durations
  * ``sample_accuracy.png|pdf``         manuscript: accuracy per participant and condition,
                                        accuracy against tilt / calibrated threshold with the
                                        calibration's psychometric model, response bias
  * ``sample_reaction_times.png|pdf``   manuscript: RT distributions, per-participant medians,
                                        RT against difficulty (correct vs wrong), timeouts

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
    from . import conditions as tbc
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import plotstyle as S
    import conditions as tbc

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
                gait.setdefault(rec.sub, []).append(json.loads(gj.read_text()))
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
    # Grey pairing lines, condition-coloured points: nine participants do not need nine
    # colours here (panel c names them), and the colours stay those of the conditions.
    for p in parts:
        sub = cell[cell["participant"] == p].set_index("condition")
        have = [c for c in conds if c in sub.index]
        a.plot([xpos[c] for c in have], [sub.loc[c, "accuracy"] for c in have],
               color=S.MIST, lw=1.2, zorder=1)
        for c in have:
            a.errorbar([xpos[c]], [sub.loc[c, "accuracy"]],
                       yerr=[[sub.loc[c, "accuracy"] - sub.loc[c, "acc_lo"]],
                             [sub.loc[c, "acc_hi"] - sub.loc[c, "accuracy"]]],
                       fmt="o", ms=5, capsize=2, lw=1, color=COND_COLOR.get(c, S.INK), zorder=2)
    S.reference_line(a, y=0.5, label="chance")
    a.set_xticks(list(xpos.values())); a.set_xticklabels(conds)
    a.set_xlim(-0.4, len(conds) - 0.6)
    a.set_ylim(0.3, 1.0); a.set_ylabel("accuracy (P correct)")
    S.panel(a, "a", "Accuracy per participant (lines pair a participant)", fontsize=10)
    a.legend(loc="lower right")

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
                       color=COND_COLOR.get(cond, S.INK), label=cond if p == parts[0] else None)
            c.plot(idx, r["p_right_stim"], marker="_", ms=12, mew=2, color=S.GREY,
                   label="share of right tilts shown" if idx == 0 else None)
            labels.append(f"{p[-2:]}/{cond[0]}")
            idx += 1
    S.reference_line(c, y=0.5)
    c.set_xticks(range(len(labels))); c.set_xticklabels(labels, rotation=90, fontsize=8)
    c.set_xlabel("participant / condition (s = standing, w = walking)")
    c.set_ylim(0, 1); c.set_ylabel("P(respond 'right')")
    S.panel(c, "c", "Response-side bias", fontsize=10)
    c.legend(loc="upper left", ncol=3, fontsize=8)

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
        # Correct trials, like the violin they sit on (cell's median_rt is over all answers).
        cor = trials[(trials["condition"] == cond) & (trials["correct"] == 1)]
        meds = cor.groupby("participant")["rt"].median().dropna().to_numpy()
        d.scatter(np.full(len(meds), i), meds, color=S.INK, s=18, zorder=3)
    d.set_xticks(range(1, len(conds) + 1)); d.set_xticklabels(conds)
    d.set_ylabel("reaction time (s), correct trials")
    S.panel(d, "d", "RT distribution by condition  (dots = per-participant medians)",
            fontsize=10)

    S.caption(fig, "Answered main trials of the blocks that pass the exclusion gate. a, c: "
                   "Wilson 95 % CI per participant; b: mean ± SD across participants, dotted = "
                   "the accuracy each level targets; d: all correct trials (violin, line = "
                   "median) and each participant's median (dots).")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    S.save(fig, out)


# ===========================================================================
# Figure 2: gait and tracking quality (walking)
# ===========================================================================
def _fig_quality(gait, participants, out: Path):
    parts = sorted(gait.keys())
    if not parts:
        return
    def gv(p, k):
        """Mean of a gait-summary field over the participant's walking blocks."""
        vals = [float(g[k]) for g in gait[p] if isinstance(g.get(k), (int, float))]
        return float(np.mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    fig.suptitle("Gait and tracking quality per participant (walking)", fontsize=13,
                 fontweight="semibold", x=0.012, ha="left")
    x = np.arange(len(parts))
    short = [p[-2:] for p in parts]

    def bars(a, key, color, ylabel, tag, title, scale=1.0, ylim=None):
        a.bar(x, [scale * gv(p, key) for p in parts], 0.6, color=color)
        a.set_xticks(x); a.set_xticklabels(short); a.set_xlabel("participant")
        a.set_ylabel(ylabel)
        if ylim:
            a.set_ylim(*ylim)
        S.panel(a, tag, title, fontsize=10)

    bars(ax[0, 0], "cadence_spm", S.BLUE, "cadence (steps / min)", "a", "Cadence")
    bars(ax[0, 1], "step_cov_pct", S.RED, "step-time CoV (%)", "b", "Step-time variability")

    # c) L/R alternation + Hilbert–trough agreement (trust in the stride-phase axis)
    b = ax[0, 2]
    b.bar(x - 0.2, [gv(p, "lr_alternation_pct") for p in parts], 0.4, color=S.GREEN,
          label="foot labels alternate (%)")
    b.bar(x + 0.2, [100 * gv(p, "hilbert_trough_phase_corr") for p in parts], 0.4,
          color=S.PURPLE, label="Hilbert vs trough phase (r × 100)")
    b.axhline(85, color=S.GREY, ls="--", lw=1, label="usable ≥ 85")
    b.set_xticks(x); b.set_xticklabels(short); b.set_xlabel("participant")
    b.set_ylabel("%"); b.set_ylim(0, 135)
    S.panel(b, "c", "Stride-phase axis", fontsize=10)
    b.legend(loc="upper left", ncol=2, fontsize=8)

    bars(ax[1, 0], "clean_coverage_pct", S.TEAL, "clean head-tracking coverage (%)", "d",
         "Tracking coverage", ylim=(0, 105))
    bars(ax[1, 1], "dropped_frame_pct", S.ORANGE, "dropped frames (%)", "e", "Dropped frames")

    # f) calibrated thresholds per condition (participants.tsv)
    d = ax[1, 2]
    speeds = []
    if not participants.empty:
        participants = participants.set_index("participant_id")
        for k, cond in enumerate(("standing", "walking")):
            col = f"threshold_{cond}_deg"
            if col in participants.columns:
                vals = [participants.loc[p, col] if p in participants.index else np.nan
                        for p in parts]
                d.bar(x + (k - 0.5) * 0.4, vals, 0.4, color=S.COND[cond], label=cond)
        speeds = [participants.loc[p, "treadmill_speed_mps"] for p in parts
                  if p in participants.index]
        d.legend(loc="upper left", fontsize=8)
    d.set_xticks(x); d.set_xticklabels(short); d.set_xlabel("participant")
    d.set_ylabel("calibrated 75 % threshold (°)")
    S.panel(d, "f", "Calibrated thresholds", fontsize=10)

    spd = (f" Belt speed {np.nanmin(speeds):.2f}-{np.nanmax(speeds):.2f} m/s for everyone."
           if len(speeds) and np.isfinite(speeds).any() else "")
    S.caption(fig, "Per participant, the mean over their walking blocks' gait summaries. "
                   "c: share of steps whose foot label alternates with the previous step, and "
                   "the agreement of the Hilbert phase with the trough phase the analysis "
                   "uses. f: the threshold each condition's blocks were run at (latest "
                   "calibration)." + spd)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    S.save(fig, out)


# ===========================================================================
# Manuscript figures: the sample, its accuracy, its reaction times
# ===========================================================================
# These read the trial table (one row per trial, with the gate and presentation flags)
# and the calibration-transfer table (each calibration's 75 % threshold), so they show
# exactly the trials the analyses use. Main trials only; accuracy over answered trials.
_OFF = {"standing": -0.17, "walking": 0.17}          # row offset within a participant


def _sample_tables():
    """(main trials with their calibrated threshold, calibrations, walking blocks' gait)."""
    tt = C.DERIV_DIR / "trial_table" / "trials.tsv"
    ut = C.DERIV_DIR / "calibration_transfer" / "calibration_transfer_units.tsv"
    if not tt.exists():
        return None
    t = pd.read_csv(tt, sep="\t", na_values=["n/a"])
    t = t[t["is_training"] == 0].copy()
    units = pd.read_csv(ut, sep="\t", na_values=["n/a"]) if ut.exists() else pd.DataFrame()
    if len(units):
        t = t.merge(units[["participant", "condition", "visit", "calibrated_t75_deg"]],
                    on=["participant", "condition", "visit"], how="left")
        t["rel_tilt"] = t["tilt_deg"] / t["calibrated_t75_deg"]
    t["kept"] = (t["behaviour"] == "include") & (t["presentation_valid"] == 1)
    out = bids_io.excluded_recordings(scope="gait")
    gait = []
    for j in sorted((C.DERIV_DIR / "gait").glob("sub-*/ses-*/*_acq-walking_*_gait.json")):
        if j.name[: -len("_gait.json")] in out:
            continue
        d = json.loads(j.read_text(encoding="utf-8"))
        if d.get("stride_duration_s"):
            gait.append({"participant": j.name.split("_")[0],
                         "stride_s": float(d["stride_duration_s"]),
                         "cadence": d.get("cadence_spm")})
    return t, units, pd.DataFrame(gait)


def _rows(ax, subs):
    """Use ``ax``'s y axis for one row per participant, top to bottom."""
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([s.replace("sub-", "") for s in subs])
    ax.set_ylim(len(subs) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)


def _cond_handles(extra=()):
    """Legend handles for the two conditions, followed by ``extra``."""
    h = [plt.Line2D([], [], ls="", marker="o", ms=5, color=COND_COLOR[c], label=c)
         for c in ("standing", "walking")]
    return h + list(extra)


def _fig_sample(t, units, gait, stem):
    """Who contributed what: trials kept per participant, calibrated thresholds, strides."""
    subs = sorted(t["participant"].unique())
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2_IN, 3.1), sharey=True,
                           gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.27, wspace=0.12)

    # a: main trials kept per condition, and those the gate or the presentation rule left out
    for i, s in enumerate(subs):
        g = t[t["participant"] == s]
        left = 0
        for lab, n, col in (("standing", int((g["kept"] & (g["condition"] == "standing")).sum()),
                             COND_COLOR["standing"]),
                            ("walking", int((g["kept"] & (g["condition"] == "walking")).sum()),
                             COND_COLOR["walking"]),
                            ("excluded", int((~g["kept"]).sum()), S.MIST)):
            if n:
                ax[0].barh(i, n, left=left, height=0.62, color=col, edgecolor="white", lw=1.0)
                left += n
    ax[0].set_xlabel("main trials")
    ax[0].set_ylabel("participant")
    _rows(ax[0], subs)
    S.panel(ax[0], "a", "Trials")

    # b: each calibration's 75 % threshold (filled = reliable, open = flagged)
    if len(units):
        for _, u in units.iterrows():
            if u["participant"] not in subs:
                continue
            y = subs.index(u["participant"]) + _OFF[u["condition"]]
            col = COND_COLOR[u["condition"]]
            ok = str(u["calibration_verdict"]).startswith("RELIABLE")
            ax[1].plot(u["calibrated_t75_deg"], y, ls="",
                       marker="o" if int(u["visit"]) == 1 else "s", ms=5,
                       mfc=col if ok else "white", mec=col, mew=1.2)
        ax[1].set_xscale("log")
        ax[1].set_xticks([0.3, 0.5, 1, 2, 4])
        ax[1].set_xticklabels(["0.3", "0.5", "1", "2", "4"])
        ax[1].axvline(0.3, color=S.GREY, ls=":", lw=0.9)
    ax[1].set_xlabel("calibrated threshold (deg)")
    _rows(ax[1], subs)
    S.panel(ax[1], "b", "Thresholds")

    # c: stride duration of every walking block, and the participant's median
    if len(gait):
        for s, g in gait.groupby("participant"):
            if s in subs:
                y = subs.index(s)
                ax[2].plot(g["stride_s"], np.full(len(g), y), "|", ms=7, mew=1.2,
                           color=COND_COLOR["walking"], alpha=0.6)
                ax[2].plot(g["stride_s"].median(), y, "o", ms=5, color=S.INK)
    ax[2].set_xlabel("stride duration (s)")
    _rows(ax[2], subs)
    S.panel(ax[2], "c", "Gait")

    handles = _cond_handles([
        plt.Line2D([], [], ls="", marker="s", ms=6, color=S.MIST, label="excluded trials"),
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=S.INK, label="visit 1"),
        plt.Line2D([], [], ls="", marker="s", ms=5, mfc="white", mec=S.INK, label="visit 2"),
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=S.GREY,
                   label="open: flagged calibration"),
        plt.Line2D([], [], ls="", marker="|", ms=7, mew=1.2, color=COND_COLOR["walking"],
                   label="walking block"),
        plt.Line2D([], [], ls="", marker="o", ms=5, color=S.INK, label="median")])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.53, 0.0), columnspacing=1.2, handletextpad=0.3)
    S.save_pub(fig, stem)


def _wilson_df(g):
    """Wilson interval of the accuracy in ``g``."""
    k, n = int(g["correct"].sum()), int(g["correct"].notna().sum())
    return wilson(k, n)


def _psychometric(rel):
    """The calibration's assumed psychometric function at tilt / threshold ``rel``."""
    g, lam, b = tbc.GUESS, tbc.LAPSE_FIX, tbc.BETA_FIX
    a = (-np.log(1 - (0.75 - g) / (1 - g - lam))) ** (-1 / b)
    return g + (1 - g - lam) * (1 - np.exp(-(np.asarray(rel) / a) ** b))


def _fig_performance(t, stem):
    """Accuracy per participant and condition, against the calibrated threshold, and bias."""
    k = t[t["kept"] & (t["timeout"] == 0)].copy()
    subs = sorted(k["participant"].unique())
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2_IN, 2.9),
                           gridspec_kw={"width_ratios": [0.8, 1.1, 1.3]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.27, wspace=0.42)

    # a: accuracy per participant, paired across conditions
    acc = {c: {} for c in ("standing", "walking")}
    for (s, c), g in k.groupby(["participant", "condition"]):
        acc[c][s] = _wilson_df(g)
    for s in subs:
        if s in acc["standing"] and s in acc["walking"]:
            ax[0].plot([0, 1], [acc["standing"][s][0], acc["walking"][s][0]], color=S.MIST,
                       lw=1.0, zorder=1)
    for x, c in enumerate(("standing", "walking")):
        v = np.array([acc[c][s] for s in subs if s in acc[c]])
        ax[0].errorbar(np.full(len(v), x), v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                       fmt="o", ms=5, color=COND_COLOR[c], elinewidth=0.9, capsize=0, zorder=3)
        m, se = v[:, 0].mean(), v[:, 0].std(ddof=1) / np.sqrt(len(v))
        ax[0].errorbar(x + 0.22, m, yerr=1.96 * se, fmt="D", ms=6, color=S.INK, capsize=0,
                       elinewidth=1.4, zorder=4)
    ax[0].axhline(0.75, color=S.GREY, ls=":", lw=1.0)
    ax[0].axhline(0.5, color=S.GREY, ls="--", lw=0.8)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["standing", "walking"])
    ax[0].set_xlim(-0.4, 1.5)
    ax[0].set_ylim(0.45, 0.95)
    ax[0].set_ylabel("accuracy (answered trials)")
    ax[0].grid(axis="x", visible=False)
    S.panel(ax[0], "a", "Accuracy")

    # b: accuracy against tilt relative to the calibrated threshold
    xf = np.linspace(0.6, 1.45, 200)
    ax[1].plot(xf, _psychometric(xf), color=S.INK, lw=1.2, zorder=2)
    if "rel_tilt" in k:
        lev = k.groupby("difficulty")["rel_tilt"].median()
        for c in ("standing", "walking"):
            kc = k[k["condition"] == c]
            per = kc.groupby(["participant", "difficulty"])["correct"].agg(["mean", "size"])
            per = per[per["size"] >= 10].reset_index()
            per["x"] = per["difficulty"].map(lev) * (1 + (0.025 if c == "walking" else -0.025))
            ax[1].plot(per["x"], per["mean"], "o", ms=3.5, color=COND_COLOR[c], alpha=0.45,
                       mec="none", zorder=3)
            mean = per.groupby("difficulty").agg(x=("x", "first"), m=("mean", "mean"),
                                                 s=("mean", "std"), n=("mean", "size"))
            mean = mean.loc[[d for d in DIFF_ORDER if d in mean.index]]
            ax[1].errorbar(mean["x"], mean["m"], yerr=1.96 * mean["s"] / np.sqrt(mean["n"]),
                           fmt="-o", ms=5.5, lw=1.6, color=COND_COLOR[c], capsize=0, zorder=4)
        ax[1].set_xticks(list(lev.loc[[d for d in DIFF_ORDER if d in lev.index]]))
        ax[1].set_xticklabels([f"{lev[d]:.2f}\n{d.lower() if d != 'THR' else 'threshold'}"
                               for d in DIFF_ORDER if d in lev.index])
    ax[1].axhline(0.5, color=S.GREY, ls="--", lw=0.8)
    ax[1].set_xlim(0.6, 1.45)
    ax[1].set_ylim(0.45, 0.95)
    ax[1].set_xlabel("tilt / calibrated threshold")
    ax[1].set_ylabel("accuracy")
    S.panel(ax[1], "b", "Against threshold")

    # c: response bias, P("right") per participant and condition
    for s_i, s in enumerate(subs):
        for c in ("standing", "walking"):
            g = k[(k["participant"] == s) & (k["condition"] == c)]
            if len(g) < 20:
                continue
            p, lo, hi = wilson(int((g["response"] == "right").sum()), len(g))
            x = s_i + _OFF[c]
            ax[2].plot([x, x], [lo, hi], color=COND_COLOR[c], lw=1.0)
            ax[2].plot(x, p, "o", ms=4.5, color=COND_COLOR[c])
    ax[2].axhline(0.5, color=S.GREY, ls="--", lw=0.8)
    ax[2].set_xticks(range(len(subs)))
    ax[2].set_xticklabels([s.replace("sub-", "") for s in subs])
    ax[2].set_xlabel("participant")
    ax[2].set_ylabel('P(answer "right")')
    ax[2].set_ylim(0.2, 0.9)
    ax[2].grid(axis="x", visible=False)
    S.panel(ax[2], "c", "Response bias")

    fig.legend(handles=_cond_handles([
        plt.Line2D([], [], ls="", marker="D", ms=6, color=S.INK, label="mean ± 95 % CI"),
        plt.Line2D([], [], color=S.INK, lw=1.2, label="calibration model"),
        plt.Line2D([], [], ls=":", color=S.GREY, label="75 % target"),
        plt.Line2D([], [], ls="--", color=S.GREY, lw=0.8, label="chance")]),
        loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.53, 0.0),
        columnspacing=1.0, handletextpad=0.3)
    S.save_pub(fig, stem)


def _fig_rt(t, stem):
    """Reaction-time distributions, per-participant medians, RT against difficulty, timeouts."""
    k = t[t["kept"]].copy()
    r = k[(k["rt_valid"] == 1) & (k["timeout"] == 0)]
    subs = sorted(k["participant"].unique())
    fig, ax = plt.subplots(2, 2, figsize=(S.COL2_IN, 5.0))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.14, wspace=0.28, hspace=0.55)
    ax = ax.ravel()

    # a: distribution of all answered RTs, each participant weighted equally
    bins = np.arange(0.15, 2.0001, 0.05)
    for c in ("standing", "walking"):
        rc = r[r["condition"] == c]
        dens = [np.histogram(g["rt_s"], bins=bins, density=True)[0]
                for _, g in rc.groupby("participant")]
        if dens:
            d = np.mean(dens, 0)
            ax[0].fill_between(bins[:-1], d, step="post", color=COND_COLOR[c], alpha=0.18, lw=0)
            ax[0].step(bins[:-1], d, where="post", color=COND_COLOR[c], lw=1.5)
            med = float(np.median(rc.groupby("participant")["rt_s"].median()))
            ax[0].axvline(med, color=COND_COLOR[c], ls="--", lw=1.0)
            ax[0].text(1.95, 0.92 if c == "standing" else 0.78, f"{c}: median {med:.2f} s",
                       transform=ax[0].get_xaxis_transform(), ha="right", fontsize=8,
                       color=S.INK)
    ax[0].set_xlim(0.15, 2.0)
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel("reaction time (s), correct trials")
    ax[0].set_ylabel("density")
    S.panel(ax[0], "a", "Reaction times")

    # b: per-participant median RT, interquartile range, paired across conditions
    for s_i, s in enumerate(subs):
        meds = {}
        for c in ("standing", "walking"):
            g = r[(r["participant"] == s) & (r["condition"] == c)]["rt_s"]
            if len(g) < 20:
                continue
            q1, m, q3 = np.percentile(g, [25, 50, 75])
            x = s_i + _OFF[c]
            ax[1].plot([x, x], [q1, q3], color=COND_COLOR[c], lw=1.0)
            ax[1].plot(x, m, "o", ms=4.5, color=COND_COLOR[c])
            meds[c] = (x, m)
    ax[1].set_xticks(range(len(subs)))
    ax[1].set_xticklabels([s.replace("sub-", "") for s in subs])
    ax[1].set_xlabel("participant")
    ax[1].set_ylabel("RT (s), median and IQR")
    ax[1].grid(axis="x", visible=False)
    S.panel(ax[1], "b", "Per participant")

    # c: RT against difficulty, correct and wrong answers, relative to each participant's
    # own median so that between-participant speed differences drop out
    ans = k[(k["timeout"] == 0) & (k["rt_s"] > 0.15) & k["correct"].notna()].copy()
    if "rel_tilt" in ans:
        ans["rel_rt"] = 1000 * (ans["rt_s"] - ans.groupby(["participant", "condition"])["rt_s"]
                                .transform("median"))
        lev = ans.groupby("difficulty")["rel_tilt"].median()
        for c in ("standing", "walking"):
            for outcome, ls, mfc in ((1, "-", None), (0, "--", "white")):
                g = ans[(ans["condition"] == c) & (ans["correct"] == outcome)]
                per = g.groupby(["participant", "difficulty"])["rel_rt"].agg(["median", "size"])
                per = per[per["size"] >= 5].reset_index()
                m = per.groupby("difficulty")["median"].agg(["mean", "std", "size"])
                m = m.loc[[d for d in DIFF_ORDER if d in m.index]]
                x = lev.loc[m.index] * (1 + (0.02 if c == "walking" else -0.02))
                ax[2].errorbar(x, m["mean"], yerr=1.96 * m["std"] / np.sqrt(m["size"]),
                               fmt="o", ls=ls, ms=5, lw=1.4, color=COND_COLOR[c],
                               mfc=mfc or COND_COLOR[c], capsize=0)
        ax[2].set_xticks(list(lev.loc[[d for d in DIFF_ORDER if d in lev.index]]))
        ax[2].set_xticklabels([f"{lev[d]:.2f}" for d in DIFF_ORDER if d in lev.index])
    ax[2].axhline(0, color=S.GREY, lw=0.8)
    ax[2].set_xlabel("tilt / calibrated threshold")
    ax[2].set_ylabel("RT − own median (ms)")
    ax[2].legend(handles=[plt.Line2D([], [], color=S.INK, marker="o", ms=5, label="correct"),
                          plt.Line2D([], [], color=S.INK, ls="--", marker="o", ms=5,
                                     mfc="white", label="wrong")],
                 loc="lower right", frameon=False, handlelength=2.2, ncol=2)
    ax[2].set_ylim(bottom=ax[2].get_ylim()[0] - 35)
    S.panel(ax[2], "c", "Speed and difficulty")

    # d: timeouts per participant and condition
    for s_i, s in enumerate(subs):
        for c in ("standing", "walking"):
            g = k[(k["participant"] == s) & (k["condition"] == c)]
            if len(g) < 20:
                continue
            ax[3].bar(s_i + _OFF[c], 100 * g["timeout"].mean(), width=0.32,
                      color=COND_COLOR[c], edgecolor="white", lw=0.8)
    ax[3].set_xticks(range(len(subs)))
    ax[3].set_xticklabels([s.replace("sub-", "") for s in subs])
    ax[3].set_xlabel("participant")
    ax[3].set_ylabel("timeouts (% of trials)")
    ax[3].grid(axis="x", visible=False)
    S.panel(ax[3], "d", "Timeouts")

    fig.legend(handles=_cond_handles(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.53, 0.0), handletextpad=0.3)
    S.save_pub(fig, stem)


def sample_figures() -> None:
    """The three manuscript figures that introduce the sample."""
    tabs = _sample_tables()
    if tabs is None:
        print("  no trial table - run the tables stage first; sample figures skipped")
        return
    t, units, gait = tabs
    figs = bids_io.fig_dir("descriptive")
    _fig_sample(t, units, gait, figs / "sample_overview")
    _fig_performance(t, figs / "sample_accuracy")
    _fig_rt(t, figs / "sample_reaction_times")


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
    sample_figures()

    print("=== Descriptive overview (exploratory) ===")
    print(f"  {trials['participant'].nunique()} participants, "
          f"{len(trials)} analysis trials across {sorted(trials['condition'].unique())}")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(cell[["participant", "condition", "n_trials", "accuracy",
                    "p_right_resp", "median_rt"]].to_string(index=False))
    print(f"\nDescriptive outputs -> {out_dir}")


if __name__ == "__main__":
    main()
