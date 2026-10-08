"""Exploratory: cyclic-spline GAM and mutual-information phase tests.

The confirmatory pipeline fits one cosine (2 or 4 cps) to binned hit rates. This works
well for a sinusoidal modulation, but (i) a narrow window or bimodal profile is only
partly captured, its energy spread across harmonics, and (ii) trial-level structure and
unequal bin counts are lost. Two less restrictive tests:

1. Cyclic-spline GAM: ``correct ~ C(sub) + s(phase, bs="cc")``, binomial, at trial level,
   with per-participant fixed intercepts; a GLMM with a flexible periodic curve in place of
   a cosine. The cyclic basis is continuous across 0/100 %. The smooth is tested with a
   likelihood-ratio χ² against the phase-free model on its effective df, and compared by
   AIC to a single-cosine GLM at the same frequency (``prefers_gam`` = non-sinusoidal
   shape fits better). statsmodels' GAM uses fixed, not random, participant intercepts, so
   it complements the random-slope GLMM in ``glmm.py``.

2. Mutual information: MI(phase; outcome) detects any dependence, not only sinusoidal,
   using circular phase bins and a permutation null (shuffled outcome). Reports p and the
   bias-corrected MI − E[MI_null]. It gives no preferred phase, so it supplements the
   Fourier/GAM fits. Used for accuracy (binary) and log-RT (quantile bins).

Confirmatory inference remains in ``phase_analysis`` / ``glmm`` / ``population``.

Usage::

    python -m Analysis.gam_phase
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import bids_io
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import gait as gait_mod
    import phase_analysis as pa
    import plotstyle as S

S.apply()

CONF_FREQS = (2.0, 4.0)
DF_SPLINE = 8               # cyclic cubic spline basis dimension
N_GRID = 200
RNG = np.random.default_rng(11)


# ===========================================================================
# Trial gathering (same phase assignment as the confirmatory analysis)
# ===========================================================================
def _walking_trials(rec) -> pd.DataFrame | None:
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        gait_mod.gait_recording(rec)
    if not npz.exists():
        return None
    g = np.load(npz)
    troughs = g["troughs"]
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    cont = pa.continuous_phase(g)
    tp = pa._trial_phases(rec, troughs, step_foot, tg, cont)
    if tp is None:
        return None
    ev, _ = tp
    rt = ev["rt"].to_numpy(float)
    acc = ev["acc"].to_numpy(float)
    logrt = np.where((ev["response"].notna()) & (acc == 1.0) & (rt > pa.RT_MIN_S) & np.isfinite(rt),
                     np.log10(np.clip(rt, 1e-3, None)), np.nan)
    return pd.DataFrame({"sub": rec.sub, "phase": ev["phase_target"].to_numpy(float),
                         "correct": acc, "logrt": logrt})


def gather_walking() -> pd.DataFrame:
    """All walking trials with columns ``sub``, ``phase`` (% stride at onset), ``correct``
    and ``logrt``. ``sub`` is needed for the per-participant intercepts.
    """
    frames = [f for rec in bids_io.find_recordings() if rec.condition == "walking"
              for f in [_walking_trials(rec)] if f is not None and not f.empty]
    cols = ["sub", "phase", "correct", "logrt"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True).dropna(subset=["phase"])


# ===========================================================================
# Cyclic-spline GAM
# ===========================================================================
def _best_freq(phase, y) -> float:
    """Frequency (2 or 4 cps) with the larger single-cosine amplitude, for the AIC baseline."""
    best, bamp = CONF_FREQS[0], -1.0
    ok = np.isfinite(phase) & np.isfinite(y)
    for f in CONF_FREQS:
        w = 2 * np.pi * f / 100.0
        X = np.column_stack([np.ones(ok.sum()), np.cos(w * phase[ok]), np.sin(w * phase[ok])])
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        amp = float(np.hypot(beta[1], beta[2]))
        if amp > bamp:
            best, bamp = f, amp
    return best


def fit_gam(df: pd.DataFrame, value: str = "correct", binomial: bool = True) -> dict | None:
    """Trial-level cyclic-spline GAM of ``value`` on stride phase, per-participant intercepts.

    Returns the smooth's LR χ² test, the AIC comparison with a single-cosine GLM and the
    population-average curve on a phase grid.
    """
    import statsmodels.api as sm
    from statsmodels.gam.api import GLMGam
    from statsmodels.gam.smooth_basis import CyclicCubicSplines

    d = df[["sub", "phase", value]].dropna()
    if d[value].nunique() < 2 or len(d) < 60:
        return None
    y = d[value].to_numpy(float)
    phase = d["phase"].to_numpy(float)
    fam = sm.families.Binomial() if binomial else sm.families.Gaussian()

    # parametric part: intercept and participant dummies
    dummies = pd.get_dummies(d["sub"], drop_first=True).astype(float)
    Xp = sm.add_constant(dummies, has_constant="add")
    Xp_arr = Xp.to_numpy(float)
    alpha = [max(1.0, len(d) / 200.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cc = CyclicCubicSplines(phase[:, None], df=[DF_SPLINE])
        gam = GLMGam(y, exog=Xp_arr, smoother=cc, alpha=alpha, family=fam)
        res = gam.fit()
        # phase-free and single-cosine models
        red = sm.GLM(y, Xp_arr, family=fam).fit()
        f0 = _best_freq(phase, y)
        w = 2 * np.pi * f0 / 100.0
        Xcos = np.column_stack([Xp_arr, np.cos(w * phase), np.sin(w * phase)])
        cosm = sm.GLM(y, Xcos, family=fam).fit()

    from scipy.stats import chi2
    lr = float(red.deviance - res.deviance)
    edf = float(res.df_model - red.df_model)
    p = float(chi2.sf(lr, max(edf, 1.0)))
    aic_gam = float(res.aic)
    aic_cos = float(cosm.aic)

    # population-average curve: participant dummies at their mean
    grid = np.linspace(0, 100, N_GRID)
    Xp_mean = np.tile(Xp_arr.mean(axis=0), (N_GRID, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = np.asarray(res.predict(Xp_mean, exog_smooth=grid[:, None]), float)
    peak_pct = float(grid[int(np.argmax(curve))])
    trough_pct = float(grid[int(np.argmin(curve))])
    # number of local maxima on the cyclic grid; > 1 means non-sinusoidal
    c = curve - curve.mean()
    nb = c.size
    modes = int(sum((c[i] > c[(i - 1) % nb]) and (c[i] >= c[(i + 1) % nb])
                    and (c[i] > 0.15 * np.ptp(c)) for i in range(nb)))
    return {
        "value": value, "n_trials": int(len(d)), "n_participants": int(d["sub"].nunique()),
        "smooth_lr_chi2": lr, "smooth_edf": edf, "p_value": p, "significant": bool(p < 0.05),
        "aic_gam": aic_gam, "aic_single_cosine": aic_cos, "cosine_freq_cps": float(f0),
        "prefers_gam_over_cosine": bool(aic_gam < aic_cos - 2.0),  # ΔAIC > 2
        "peak_pct": peak_pct, "trough_pct": trough_pct, "n_modes": modes,
        "curve_grid_pct": grid.tolist(), "curve_fit": curve.tolist(),
    }


# ===========================================================================
# Mutual information with permutation test
# ===========================================================================
def _mi_bits(xb: np.ndarray, yb: np.ndarray, nx: int, ny: int) -> float:
    """Plug-in mutual information (bits) from integer-coded samples."""
    c = np.histogram2d(xb, yb, bins=[nx, ny], range=[[0, nx], [0, ny]])[0]
    n = c.sum()
    if n == 0:
        return 0.0
    pij = c / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    nz = pij > 0
    return float(np.sum(pij[nz] * np.log2(pij[nz] / (pi @ pj)[nz])))


def mutual_info_perm(phase, y, *, y_binary=True, nbins_phase=18, nbins_y=6,
                     nperm=500, rng=None) -> dict | None:
    """MI between stride phase and outcome, with a permutation null.

    Phase uses ``nbins_phase`` equal bins; a continuous outcome is quantile-binned into
    ``nbins_y``. The null shuffles the outcome; the effect size is MI − mean null MI.
    """
    rng = rng or RNG
    phase = np.asarray(phase, float)
    y = np.asarray(y, float)
    ok = np.isfinite(phase) & np.isfinite(y)
    phase, y = phase[ok], y[ok]
    if phase.size < 80:
        return None
    xb = np.clip((phase / 100.0 * nbins_phase).astype(int), 0, nbins_phase - 1)
    if y_binary:
        yb = (y > 0.5).astype(int); ny = 2
    else:
        edges = np.quantile(y, np.linspace(0, 1, nbins_y + 1))
        edges[0] -= 1e-9
        yb = np.clip(np.digitize(y, edges[1:-1]), 0, nbins_y - 1); ny = nbins_y
    obs = _mi_bits(xb, yb, nbins_phase, ny)
    null = np.empty(nperm)
    for i in range(nperm):
        null[i] = _mi_bits(xb, rng.permutation(yb), nbins_phase, ny)
    p = float((1 + np.sum(null >= obs)) / (1 + nperm))
    return {"mi_bits": float(obs), "mi_null_mean_bits": float(null.mean()),
            "mi_debiased_bits": float(obs - null.mean()), "p_value": p,
            "significant": bool(p < 0.05), "n": int(phase.size),
            "nbins_phase": nbins_phase, "n_perm": nperm}


# ===========================================================================
# Figure
# ===========================================================================
def _figure(df, gam_acc, mi_acc, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7), gridspec_kw=dict(width_ratios=[1.5, 1.0]))
    fig.suptitle(f"Cyclic-spline GAM + mutual information (accuracy) — "
                 f"n = {gam_acc['n_participants']} participant(s), {gam_acc['n_trials']} trials",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    # (a) binned hit rate, GAM smooth and single cosine
    cen, binned, _ = pa._bin(df["phase"].to_numpy(float), df["correct"].to_numpy(float), 20, "mean")
    ax[0].scatter(cen, binned, s=20, color=S.GREY, label="observed (20 bins)")
    ax[0].plot(gam_acc["curve_grid_pct"], gam_acc["curve_fit"], color=S.PURPLE, lw=2.6,
               label=f"cyclic GAM (edf {gam_acc['smooth_edf']:.1f}, "
                     f"p={gam_acc['p_value']:.3f}{' *' if gam_acc['significant'] else ''})")
    # single-cosine fit for comparison
    f0 = gam_acc["cosine_freq_cps"]; w = 2 * np.pi * f0 / 100.0
    ok = np.isfinite(binned)
    X = np.column_stack([np.ones(ok.sum()), np.cos(w * cen[ok]), np.sin(w * cen[ok])])
    beta, *_ = np.linalg.lstsq(X, binned[ok], rcond=None)
    xf = np.linspace(0, 100, 200)
    ax[0].plot(xf, beta[0] + beta[1] * np.cos(w * xf) + beta[2] * np.sin(w * xf),
               color=S.ORANGE, lw=1.8, ls="--", label=f"single cosine @ {f0:.0f} cps")
    ax[0].set_xlabel("stride phase (% stride)"); ax[0].set_ylabel("hit rate")
    ax[0].legend(loc="upper right", fontsize=8)
    tag = "prefers GAM (non-sinusoidal)" if gam_acc["prefers_gam_over_cosine"] else "cosine adequate"
    S.panel(ax[0], "a", f"GAM vs cosine — ΔAIC {gam_acc['aic_single_cosine']-gam_acc['aic_gam']:+.1f} ({tag})",
            fontsize=9.5)
    # (b) MI observed vs null
    if mi_acc:
        ax[1].axvline(mi_acc["mi_bits"], color=S.PURPLE, lw=2.4,
                      label=f"MI {mi_acc['mi_bits']:.4f} bits (p={mi_acc['p_value']:.3f})")
        ax[1].axvline(mi_acc["mi_null_mean_bits"], color=S.RED, ls=":", lw=1.6, label="null mean")
        ax[1].set_xlabel("mutual information (bits)"); ax[1].set_yticks([])
        ax[1].legend(loc="upper right", fontsize=8)
    S.panel(ax[1], "b", "MI(phase; accuracy) vs permutation null", fontsize=9.5)
    S.caption(fig, "GAM: cyclic cubic spline, binomial, per-participant fixed intercepts; "
                   "p = LR χ² of the smooth. MI: circular-binned, permutation null. Both "
                   "EXPLORATORY (confirmatory = phase_analysis / glmm).")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    S.save(fig, out)


# ===========================================================================
# Driver
# ===========================================================================
def main():
    """Fit the cyclic-spline GAM and mutual-information analyses."""
    df = gather_walking()
    bids_io.gate_report("gam", condition="walking",
                        n_analysed=int(df["sub"].nunique()) if not df.empty else 0)
    if df.empty:
        bids_io.clear_stale("gam_phase", "gam_phase.json", "gam_phase.png")
        print("No walking trials found. Run the gait/phase stages first.")
        return
    out_dir = bids_io.deriv_dir("gam_phase")
    gam_acc = fit_gam(df, "correct", binomial=True)
    rt = df.dropna(subset=["logrt"])
    gam_rt = fit_gam(rt, "logrt", binomial=False) if len(rt) >= 60 else None
    mi_acc = mutual_info_perm(df["phase"].to_numpy(float), df["correct"].to_numpy(float),
                              y_binary=True, rng=np.random.default_rng(1))
    mi_rt = (mutual_info_perm(rt["phase"].to_numpy(float), rt["logrt"].to_numpy(float),
                              y_binary=False, rng=np.random.default_rng(2))
             if len(rt) >= 80 else None)

    payload = {"note": "EXPLORATORY cyclic-spline GAM + mutual information; confirmatory "
                       "tests remain phase_analysis / glmm / population.",
               "gam_accuracy": gam_acc, "gam_reaction_time": gam_rt,
               "mi_accuracy": mi_acc, "mi_reaction_time": mi_rt}
    # leave the fitted curves out of the JSON
    slim = json.loads(json.dumps(payload))
    for g in ("gam_accuracy", "gam_reaction_time"):
        if slim.get(g):
            slim[g].pop("curve_grid_pct", None); slim[g].pop("curve_fit", None)
    (out_dir / "gam_phase.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    if gam_acc:
        _figure(df, gam_acc, mi_acc, bids_io.fig_dir("gam_phase") / "gam_phase.png")

    print("=== Cyclic-spline GAM + mutual information (EXPLORATORY) ===")
    if gam_acc:
        print(f"  GAM accuracy : edf {gam_acc['smooth_edf']:.1f}, LR chi2={gam_acc['smooth_lr_chi2']:.1f}, "
              f"p={gam_acc['p_value']:.4f} -> {'SIG' if gam_acc['significant'] else 'n.s.'} | "
              f"peak {gam_acc['peak_pct']:.0f}%, {gam_acc['n_modes']} mode(s) | "
              f"dAIC vs cosine {gam_acc['aic_single_cosine']-gam_acc['aic_gam']:+.1f} "
              f"({'prefers GAM' if gam_acc['prefers_gam_over_cosine'] else 'cosine ok'})")
    if gam_rt:
        print(f"  GAM log-RT   : edf {gam_rt['smooth_edf']:.1f}, p={gam_rt['p_value']:.4f} -> "
              f"{'SIG' if gam_rt['significant'] else 'n.s.'}")
    if mi_acc:
        print(f"  MI accuracy  : {mi_acc['mi_bits']:.4f} bits (debiased {mi_acc['mi_debiased_bits']:+.4f}), "
              f"p={mi_acc['p_value']:.4f} -> {'SIG' if mi_acc['significant'] else 'n.s.'}")
    if mi_rt:
        print(f"  MI log-RT    : {mi_rt['mi_bits']:.4f} bits (debiased {mi_rt['mi_debiased_bits']:+.4f}), "
              f"p={mi_rt['p_value']:.4f} -> {'SIG' if mi_rt['significant'] else 'n.s.'}")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
