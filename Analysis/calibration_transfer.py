"""Calibration transfer: does the calibrated threshold hold in the blocks that use it?

Each condition is calibrated once per visit (QUEST+, ~40-70 trials); the following blocks
present the THR level at the calibrated 75 %-correct tilt. If the calibration transfers,
THR accuracy is near 75 % and a threshold refitted to the block trials is near the
calibrated one.

Computed per calibration (participant x condition x visit), pooling its blocks:

* Block threshold: 75 % threshold fitted to the blocks' answered trials (all three tilt
  levels) with QUEST+ slope, lapse and guess fixed (``threshold_by_condition.fit_alpha``),
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
  * ``calibration_transfer.json``        summary and the test
  * ``figures/calibration_transfer.png`` / ``.pdf``

Usage::

    python -m Analysis.calibration_transfer
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
from scipy.stats import binomtest

try:
    from . import bids_io
    from . import plotstyle as S
    from . import calibration_refit as cref
    from . import threshold_by_condition as tbc
    from .block_rules import calibration_verdict
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import plotstyle as S
    import calibration_refit as cref
    import threshold_by_condition as tbc
    from block_rules import calibration_verdict

S.apply()

TARGET = tbc.TARGET          # 75 % correct at THR
N_SIM = 400                  # simulated blocks per calibration
SEED = 20261007


# ===========================================================================
# Threshold fit with a profile-likelihood interval
# ===========================================================================
GRID = np.linspace(np.log(0.05), np.log(60.0), 700)     # log alpha search grid


def _nll_grid(tilt, correct):
    """Negative log-likelihood for every alpha on ``GRID``."""
    p = cref.weibull(tilt[None, :], np.exp(GRID)[:, None], tbc.BETA_FIX, tbc.LAPSE_FIX,
                     tbc.GUESS)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).sum(axis=1)


def _to_t75(alpha):
    return float(cref._inverse(TARGET, alpha, tbc.BETA_FIX, tbc.LAPSE_FIX, tbc.GUESS))


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
    p = cref.weibull(tilt, alpha, tbc.BETA_FIX, tbc.LAPSE_FIX, tbc.GUESS)
    # All simulated blocks are fitted at once; draws, order, filtering and grid match
    # calling fit_threshold per block.
    draws = rng.random((n_sim, tilt.size))
    fin = np.isfinite(tilt)
    if fin.sum() < 20:
        return np.asarray([])
    correct = (draws[:, fin] < p[fin]).astype(float)
    tilt = tilt[fin]
    pg = cref.weibull(tilt[None, :], np.exp(GRID)[:, None], tbc.BETA_FIX, tbc.LAPSE_FIX,
                      tbc.GUESS)
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
            "calibration_verdict": calibration_verdict(rec.sub, rec.condition, visit),
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
def _figure(blocks: pd.DataFrame, units: pd.DataFrame, summ: dict, stem: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(S.COL2_IN, 3.3),
                                     gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88, wspace=0.32)

    # a: THR accuracy per block and participant
    subs = sorted(blocks["participant"].unique())
    n_med = int(blocks.loc[blocks["behaviour"] == "include", "n_thr_answered"].median())
    lo, hi = (np.array([np.quantile(np.random.default_rng(SEED).binomial(n_med, TARGET, 20000),
                                    q) for q in (0.025, 0.975)]) / n_med)
    ax_a.axhspan(lo, hi, color=S.GRID, zorder=0, lw=0)
    S.reference_line(ax_a, y=TARGET)
    S.reference_line(ax_a, y=0.5, ls=":")
    for i, sub in enumerate(subs):
        g = blocks[blocks["participant"] == sub].sort_values(["condition", "visit", "run"])
        for j, (_, r) in enumerate(g.iterrows()):
            x = i + (j - (len(g) - 1) / 2) * min(0.08, 0.7 / max(len(g), 1))
            col = S.COND[r["condition"]]
            filled = r["behaviour"] == "include"
            ax_a.plot(x, r["thr_accuracy"], "o", ms=4.2, mec=col, mew=1.0,
                      mfc=col if filled else "white", zorder=3)
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
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=4.5, mec=S.GREY, mfc="white",
                              label="excluded"))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=S.GRID, label="sampling range"))
    ax_a.legend(handles=handles, loc="lower left", ncol=2, fontsize=8)

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
                     label="outside sampling range"),
          plt.Line2D([], [], ls="", marker="o", ms=5, mec=S.INK, mfc="white",
                     label="within it"),
          plt.Line2D([], [], ls="", marker="D", ms=4.5, mec=S.INK, mfc=S.INK,
                     label="unreliable calibration"),
          plt.Line2D([], [], ls="", marker="v", ms=5, mec=S.INK, mfc=S.INK,
                     label="at ceiling (bound only)")]
    ax_b.legend(handles=hb, loc="upper left", fontsize=8, handletextpad=0.3)
    S.save_pub(fig, stem)


# ===========================================================================
# Main
# ===========================================================================
def analyse() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    blocks = _blocks()
    units = _units(blocks, rng) if not blocks.empty else pd.DataFrame()
    return blocks, units, _summary(units)


def main() -> dict:
    """Fit, test and plot the calibration transfer for every calibration."""
    blocks, units, summ = analyse()
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
                       "null (calibration_transfer.py).",
        "Model": {"beta": tbc.BETA_FIX, "lapse": tbc.LAPSE_FIX, "guess": tbc.GUESS,
                  "target": TARGET, "n_sim": N_SIM, "seed": SEED},
        **summ}, indent=2), encoding="utf-8")
    _figure(blocks, units, summ, bids_io.fig_dir("calibration_transfer") / "calibration_transfer")

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


if __name__ == "__main__":
    main()
