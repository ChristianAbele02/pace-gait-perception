"""Offline refit of the psychometric slope and lapse from calibration data (b8).

QUEST+ (``StaircaseManager``) estimates only the threshold, with Weibull slope and lapse
fixed at assumed values. With several participants' calibrations these can be checked by
fitting threshold, slope and lapse by maximum likelihood. The refined values can be fed
back into QUEST+ (with priors re-centred on the observed thresholds), making calibrations
faster and less prone to lapse-driven runaway.

Model (2-AFC Weibull on tilt magnitude, guess rate fixed at 0.5)::

    psi(x) = 0.5 + (0.5 - lambda) * (1 - exp(-(x/alpha)^beta))
    alpha = threshold (deg), beta = slope, lambda = lapse

``refit_from_bids()`` (``--bids``) fits the real calibrations: one shared slope and lapse per
condition, a threshold per calibration (``threshold_by_condition`` uses it). By default
``main`` simulates that refit at the rig's calibration design: calibrations drawn like the
real ones (thresholds, QUEST+ trial counts, catch trials) run through the same QUEST+
placement, repeated for 12 and 24 calibrations per condition, with the lapse free, fixed at
the QUEST+ value, or free with the catch trials added. The repetitions are cached
(``stage_cache``) and rerun only when this code or the real calibrations change.

Usage::

    python -m Analysis.calibration_refit            # simulation study + figure
    python -m Analysis.calibration_refit --bids     # refit pooled real ses-01 calibrations
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

try:
    from . import bids_io
    from . import plotstyle as S
    from . import stage_cache
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import plotstyle as S
    import stage_cache

S.apply()

GUESS = 0.5                 # 2-AFC chance level
LAPSE_MAX = 0.20

# Limits near the optimiser bounds that mark a fit as "railed": the data do not identify
# the parameter (e.g. near-ceiling accuracy at all tilts). Same alpha check as
# `threshold_by_condition.fit_alpha`, extended to beta and lapse.
ALPHA_RAIL = (0.12, 40.0)   # search bounds (0.1, 45)
BETA_RAIL = (0.6, 11.0)     # search bounds (0.5, 12)
LAPSE_RAIL_HI = 0.19        # search bound (0, LAPSE_MAX=0.20)


def _railed(alpha: float | None = None, beta: float | None = None,
           lapse: float | None = None) -> bool:
    r = False
    if alpha is not None:
        r = r or alpha <= ALPHA_RAIL[0] or alpha >= ALPHA_RAIL[1]
    if beta is not None:
        r = r or beta <= BETA_RAIL[0] or beta >= BETA_RAIL[1]
    if lapse is not None:
        r = r or lapse >= LAPSE_RAIL_HI
    return bool(r)


# ===========================================================================
# Psychometric function and MLE
# ===========================================================================
def weibull(x, alpha, beta, lapse, guess=GUESS):
    """Weibull psychometric function: P(correct) at stimulus level ``x``.

    ``alpha`` is the threshold (scale), ``beta`` the slope, ``lapse`` the gap below 1 at the
    upper asymptote and ``guess`` the floor (0.5 for 2-AFC). Same parameterisation as the
    rig's QUEST+, so offline and online fits are comparable.
    """
    x = np.asarray(x, float)
    f = 1.0 - np.exp(-np.power(np.clip(x, 1e-9, None) / alpha, beta))
    return guess + (1.0 - guess - lapse) * f


def _neg_log_lik(theta, x, correct):
    log_alpha, log_beta, lapse = theta
    p = weibull(x, np.exp(log_alpha), np.exp(log_beta), lapse)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def fit_psychometric(x, correct, alpha0=4.0, beta0=3.0) -> dict:
    """MLE of (threshold alpha, slope beta, lapse) for 2-AFC Weibull data."""
    x = np.asarray(x, float); correct = np.asarray(correct, float)
    if x.size < 8 or np.unique(correct).size < 2:
        return {"ok": False, "reason": "insufficient/degenerate data"}
    res = minimize(_neg_log_lik, x0=[np.log(alpha0), np.log(beta0), 0.03],
                   args=(x, correct), method="L-BFGS-B",
                   bounds=[(np.log(0.1), np.log(45)), (np.log(0.5), np.log(12)),
                           (0.0, LAPSE_MAX)])
    a, b, lam = np.exp(res.x[0]), np.exp(res.x[1]), res.x[2]
    # threshold at 75 % correct (QUEST+ target)
    thr75 = _inverse(0.75, a, b, lam)
    railed = _railed(a, b, lam)
    return {"ok": bool(res.success) and not railed, "railed": railed,
            "threshold_alpha_deg": round(float(a), 4),
            "slope_beta": round(float(b), 4), "lapse": round(float(lam), 4),
            "thresh75_deg": round(float(thr75), 4), "neg_log_lik": round(float(res.fun), 3),
            "n_trials": int(x.size)}


def _inverse(p_target, alpha, beta, lapse, guess=GUESS):
    """Tilt giving p_target correct."""
    num = (p_target - guess) / (1.0 - guess - lapse)
    num = np.clip(num, 1e-9, 1 - 1e-9)
    return alpha * (-np.log(1.0 - num)) ** (1.0 / beta)


def fit_joint(observers, lapse: float | None = None) -> dict:
    """MLE of one shared slope and lapse with per-participant thresholds.

    Pooling trials across observers with different thresholds flattens the slope; a
    joint fit with individual thresholds avoids this. ``observers`` is a list of
    ``(tilt, correct)`` arrays; ``lapse`` fixes the lapse instead of fitting it.
    """
    obs = [(np.asarray(x, float), np.asarray(c, float)) for x, c in observers
           if np.asarray(x).size >= 8 and np.unique(c).size >= 2]
    K = len(obs)
    if K == 0:
        return {"ok": False, "reason": "no usable observers"}
    x_all = np.concatenate([x for x, _ in obs])
    c_all = np.concatenate([c for _, c in obs])
    idx = np.concatenate([np.full(x.size, j) for j, (x, _) in enumerate(obs)])
    fixed = lapse is not None
    n_shared = 1 if fixed else 2

    def nll(theta):
        lam = lapse if fixed else theta[1]
        p = weibull(x_all, np.exp(theta[n_shared:])[idx], np.exp(theta[0]), lam)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -float(np.sum(c_all * np.log(p) + (1 - c_all) * np.log(1 - p)))

    x0 = ([np.log(3.0)] + ([] if fixed else [0.03])
          + [np.log(max(1.0, np.median(x))) for x, _ in obs])
    bounds = ([(np.log(0.5), np.log(12))] + ([] if fixed else [(0.0, LAPSE_MAX)])
              + [(np.log(0.1), np.log(45))] * K)
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)
    beta = float(np.exp(res.x[0]))
    lapse = float(lapse) if fixed else float(res.x[1])
    alphas = [float(np.exp(v)) for v in res.x[n_shared:]]
    thr75 = [float(_inverse(0.75, a, beta, lapse)) for a in alphas]
    # "ok" (used by threshold_by_condition.py) depends on the shared parameters only, so one
    # observer with a railed threshold does not invalidate the fit. Railed thresholds are
    # left out of thresh75_median_deg and counted in n_railed_observers.
    alpha_railed = [_railed(alpha=a) for a in alphas]
    shared_railed = _railed(beta=beta, lapse=lapse)
    thr75_clean = [t for t, r in zip(thr75, alpha_railed) if not r]
    return {"ok": bool(res.success) and not shared_railed, "railed": shared_railed,
            "n_railed_observers": int(sum(alpha_railed)),
            "slope_beta": round(beta, 4), "lapse": round(lapse, 4),
            "thresholds_alpha_deg": [round(a, 3) for a in alphas],
            "alpha_railed": alpha_railed,
            "thresh75_median_deg": round(float(np.median(thr75_clean or thr75)), 4),
            "n_participants": K, "n_trials": int(sum(x.size for x, _ in obs))}


# ===========================================================================
# Slope-invariance test across conditions (standing vs walking)
# ===========================================================================
def _fit_conditioned(obs, conds, share_slope, share_lapse) -> dict:
    """Joint MLE over conditions with optionally shared slope/lapse.

    ``obs`` maps condition -> list of ``(tilt, correct)``. Each observer in each condition
    has its own threshold; slope and lapse are shared (``share_*=True``) or per condition.
    Returns the negative log-likelihood, slope/lapse per condition and the number of free
    parameters, for a likelihood-ratio test.
    """
    thr_keys = [(c, j) for c in conds for j in range(len(obs[c]))]
    nthr = len(thr_keys)
    nbeta = 1 if share_slope else len(conds)
    nlapse = 1 if share_lapse else len(conds)
    cidx = {c: i for i, c in enumerate(conds)}

    def unpack(theta):
        bvec, lvec, avec = theta[:nbeta], theta[nbeta:nbeta + nlapse], theta[nbeta + nlapse:]
        betas = {c: float(np.exp(bvec[0 if share_slope else cidx[c]])) for c in conds}
        lapses = {c: float(lvec[0 if share_lapse else cidx[c]]) for c in conds}
        return betas, lapses, avec

    def nll(theta):
        betas, lapses, avec = unpack(theta)
        total = 0.0
        for k, (c, j) in enumerate(thr_keys):
            x, cc = obs[c][j]
            p = np.clip(weibull(x, np.exp(avec[k]), betas[c], lapses[c]), 1e-6, 1 - 1e-6)
            total -= np.sum(cc * np.log(p) + (1 - cc) * np.log(1 - p))
        return float(total)

    x0 = ([np.log(3.0)] * nbeta + [0.03] * nlapse
          + [np.log(max(1.0, np.median(obs[c][j][0]))) for c, j in thr_keys])
    bounds = ([(np.log(0.5), np.log(12))] * nbeta + [(0.0, LAPSE_MAX)] * nlapse
              + [(np.log(0.1), np.log(45))] * nthr)
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)
    betas, lapses, _ = unpack(res.x)
    return {"nll": float(res.fun), "betas": betas, "lapses": lapses,
            "n_params": nbeta + nlapse + nthr, "success": bool(res.success)}


def fit_slope_invariance(observers_by_cond) -> dict:
    """Test whether the slope differs between conditions (slope invariance, Davidson 2024).

    Two nested joint models, with per-participant thresholds:
      * ``free``: slope and lapse per condition;
      * ``shared_slope``: one slope, lapse per condition.
    The likelihood-ratio test (df = number of conditions - 1) tests equal slopes. A small p
    would undermine the fixed-slope comparison in ``threshold_by_condition``.
    ``observers_by_cond`` maps condition -> list of ``(tilt, correct)``.
    """
    obs = {c: [(np.asarray(x, float), np.asarray(cc, float)) for x, cc in v
               if np.asarray(x).size >= 8 and np.unique(cc).size >= 2]
           for c, v in observers_by_cond.items()}
    conds = [c for c in obs if obs[c]]
    if len(conds) < 2:
        return {"ok": False, "reason": "need >=2 conditions with usable data"}

    free = _fit_conditioned(obs, conds, share_slope=False, share_lapse=False)
    shared = _fit_conditioned(obs, conds, share_slope=True, share_lapse=False)
    df = len(conds) - 1
    lr = max(0.0, 2.0 * (shared["nll"] - free["nll"]))   # shared is the nested model
    p = float(chi2.sf(lr, df))
    return {"ok": bool(free["success"] and shared["success"]),
            "conditions": conds,
            "slope_per_condition": {c: round(free["betas"][c], 4) for c in conds},
            "lapse_per_condition": {c: round(free["lapses"][c], 4) for c in conds},
            "shared_slope": round(shared["betas"][conds[0]], 4),
            "lr_stat": round(lr, 4), "df": df, "p_value": round(p, 4),
            "slope_differs": bool(p < 0.05),
            "n_participants_per_condition": {c: len(obs[c]) for c in conds}}


# ===========================================================================
# Simulation study (parameter recovery at the rig's calibration design)
# ===========================================================================
# The rig's calibration, as the simulation reproduces it. Thresholds and trial counts
# come from the 23 calibrations the exclusion gate kept (2026-10-13): QUEST+ alpha has
# log10 mean 0.08 and SD 0.19 (about 1.2 deg; standing 0.03 / 0.16, walking 0.13 / 0.21);
# 34-66 QUEST+ trials per calibration (median 49); a catch trial at 12 deg every ~6
# trials, of which 2 of 166 were wrong. QUEST+ itself (StaircaseManager, mirrored in
# calibration_convergence_sim) assumes slope 2.5 and lapse 0.05 and places its trials on
# the 0.3-25 deg grid by minimum expected entropy; catch trials do not update it.
RIG_THR_LOG10 = (0.08, 0.19)
RIG_TRIALS = (34, 66)
CATCH_TILT_DEG, CATCH_EVERY = 12.0, 6
RIG_SLOPE, RIG_LAPSE = 2.5, 0.05
SIM_TRUE_SLOPE = 3.0        # between the QUEST+ design value and the standing refit
SIM_TRUE_LAPSE = 0.01       # the catch-trial error rate (2/166)
SIM_N_CALIBRATIONS = (12, 24)   # per condition: now, and 12 participants x 2 visits
SIM_REPS = 200
SIM_SEED = 2026
STRATEGIES = {
    "free": "lapse free, QUEST+ trials only",
    "rig": f"lapse fixed at {RIG_LAPSE} (QUEST+ value)",
    "catch": "lapse free, with catch trials",
}


def simulate_calibration(rng, alpha: float, beta: float, lapse: float, n_trials: int):
    """One calibration as the rig runs it: two interleaved QUEST+ trackers placing
    ``n_trials`` trials, plus a catch trial at ``CATCH_TILT_DEG`` every ``CATCH_EVERY``.

    Returns the QUEST+ trials ``(tilt, correct)``, the catch trials ``(tilt, correct)``
    and the pooled posterior threshold (deg), the calibration's estimate.
    """
    try:
        from . import calibration_convergence_sim as qsim
    except ImportError:
        import calibration_convergence_sim as qsim
    a_trk, b_trk = qsim.Quest(qsim.START_A, qsim.PRIOR_SD), qsim.Quest(qsim.START_B, qsim.PRIOR_SD)
    pool = qsim.Quest(qsim.START_A, 0.0)
    a_log = np.log10(alpha)
    x = np.empty(n_trials)
    c = np.empty(n_trials)
    for t in range(n_trials):
        trk = a_trk if t % 2 == 0 else b_trk
        x_log = trk.next_stim()
        ok = rng.random() < qsim._weibull(x_log, a_log, beta, lapse)
        trk.update(ok)
        pool.update_at(x_log, ok)
        x[t], c[t] = 10 ** x_log, float(ok)
    n_catch = n_trials // CATCH_EVERY
    xc = np.full(n_catch, CATCH_TILT_DEG)
    cc = (rng.random(n_catch) < weibull(xc, alpha, beta, lapse)).astype(float)
    return (x, c), (xc, cc), float(10 ** pool.thr_log())


def simulation_study(n_reps: int = SIM_REPS, n_calibrations=SIM_N_CALIBRATIONS,
                     true_slope: float = SIM_TRUE_SLOPE, true_lapse: float = SIM_TRUE_LAPSE,
                     seed: int = SIM_SEED) -> dict:
    """Repeat the rig's calibration ``n_reps`` times per cohort size and refit each.

    Every repetition draws thresholds and trial counts like the rig's, runs QUEST+,
    and fits the shared slope and lapse three ways (``STRATEGIES``). Returns the
    per-repetition estimates and their summary (median, quartiles, bias, RMSE).
    """
    rng = np.random.default_rng(seed)
    rows, placement = [], []
    for k in n_calibrations:
        for rep in range(n_reps):
            stair, catch = [], []
            for _ in range(k):
                alpha = float(10 ** rng.normal(*RIG_THR_LOG10))
                n = int(rng.integers(RIG_TRIALS[0], RIG_TRIALS[1] + 1))
                s, ct, est = simulate_calibration(rng, alpha, true_slope, true_lapse, n)
                stair.append(s)
                catch.append(ct)
                if rep == 0:
                    placement.append((s[0] / est, n))
            fits = {"free": fit_joint(stair),
                    "rig": fit_joint(stair, lapse=RIG_LAPSE),
                    "catch": fit_joint([(np.r_[x, xc], np.r_[c, cc])
                                        for (x, c), (xc, cc) in zip(stair, catch)])}
            for name, f in fits.items():
                rows.append({"n_calibrations": k, "rep": rep, "strategy": name,
                             "slope": f.get("slope_beta"), "lapse": f.get("lapse"),
                             "railed": bool(f.get("railed", True))})
    df = pd.DataFrame(rows)
    summary = []
    for (k, name), g in df.groupby(["n_calibrations", "strategy"], sort=False):
        row = {"n_calibrations": int(k), "strategy": name, "label": STRATEGIES[name],
               "n_reps": int(len(g)), "railed_pct": round(100 * float(g["railed"].mean()), 1)}
        for par, truth in (("slope", true_slope), ("lapse", true_lapse)):
            v = g[par].astype(float)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            row.update({f"{par}_median": round(float(med), 4),
                        f"{par}_iqr": [round(float(q1), 4), round(float(q3), 4)],
                        f"{par}_bias": round(float(med - truth), 4),
                        f"{par}_rmse": round(float(np.sqrt(np.mean((v - truth) ** 2))), 4)})
        summary.append(row)
    return {"design": {"thr_log10_mean_sd": list(RIG_THR_LOG10), "quest_trials": list(RIG_TRIALS),
                       "catch_tilt_deg": CATCH_TILT_DEG, "catch_every": CATCH_EVERY,
                       "quest_assumed": {"slope": RIG_SLOPE, "lapse": RIG_LAPSE},
                       "true_slope": true_slope, "true_lapse": true_lapse,
                       "n_calibrations": list(n_calibrations), "n_reps": n_reps, "seed": seed},
            "summary": summary, "_estimates": df, "_placement": placement}


def _real_placement() -> tuple[np.ndarray, list[int]] | None:
    """Presented tilt / final QUEST+ threshold, and QUEST+ trial counts, of the gated
    real calibrations (``None`` when there are none)."""
    gated = bids_io.excluded_calibrations()
    ratios, counts = [], []
    for f in bids_io.find_calibration_tables():
        if bids_io.calibration_label(f) in gated:
            continue
        df = pd.read_csv(f, sep="\t", na_values="n/a")
        if not {"presented_tilt_deg", "post_mean_threshold_deg", "tracker"} <= set(df):
            continue
        d = df[~df["tracker"].astype(str).isin(["catch", "warmup"])]
        thr = pd.to_numeric(d["post_mean_threshold_deg"], errors="coerce").dropna()
        if thr.empty:
            continue
        ratios.append(pd.to_numeric(d["presented_tilt_deg"], errors="coerce").dropna()
                      / float(thr.iloc[-1]))
        counts.append(len(d))
    return (np.concatenate(ratios), counts) if ratios else None


def _fig_recovery(rec: dict, out: Path, real=None):
    """Three panels: (a) the simulated calibrations against the real ones, (b) slope and
    (c) lapse recovered by each lapse strategy, for each cohort size."""
    des, est = rec["design"], rec["_estimates"]
    colors = {"free": S.BLUE, "rig": S.ORANGE, "catch": S.GREEN}
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2_IN, 3.1),
                           gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})

    # a) where QUEST+ puts its trials, relative to the calibrated threshold
    bins = np.geomspace(0.1, 30, 36)
    sim_n = [n for _, n in rec["_placement"]]
    ax[0].hist(np.concatenate([r for r, _ in rec["_placement"]]), bins=bins, density=True,
               color=S.MIST, label=f"simulated ({min(sim_n)}-{max(sim_n)} trials)")
    if real is not None:
        ax[0].hist(real[0], bins=bins, density=True, histtype="step", color=S.INK, lw=1.2,
                   label=f"rig ({min(real[1])}-{max(real[1])} trials)")
    catch_x = CATCH_TILT_DEG / 10 ** des["thr_log10_mean_sd"][0]
    ax[0].axvline(catch_x, color=S.GREY, ls=":", lw=1)
    ax[0].text(catch_x * 0.85, 0.5, f"catch trials ({CATCH_TILT_DEG:g}°)", rotation=90,
               ha="right", va="center", transform=ax[0].get_xaxis_transform(),
               fontsize=S.MIN_PT, color=S.GREY)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("tilt / calibrated threshold")
    ax[0].set_ylabel("density")
    ax[0].legend(loc="upper left", fontsize=S.MIN_PT, handlelength=1.2)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.3)
    S.panel(ax[0], "a", "Trial placement")

    # b, c) recovered slope and lapse per cohort size, one box per strategy
    ks = des["n_calibrations"]
    names = list(STRATEGIES)
    width = 0.26
    for a, par, truth in ((ax[1], "slope", des["true_slope"]), (ax[2], "lapse", des["true_lapse"])):
        for i, name in enumerate(names):
            pos = [g + (i - 1) * width for g in range(len(ks))]
            if par == "lapse" and name == "rig":          # fixed, not estimated
                for xpos in pos:
                    a.plot([xpos - width * 0.4, xpos + width * 0.4], [RIG_LAPSE] * 2,
                           color=colors[name], lw=2.5, solid_capstyle="butt")
                continue
            data = [est[(est["n_calibrations"] == k) & (est["strategy"] == name)][par].astype(float)
                    for k in ks]
            bp = a.boxplot(data, positions=pos, widths=width * 0.8, patch_artist=True,
                           showfliers=False, medianprops={"color": S.INK, "lw": 1.2},
                           whiskerprops={"color": colors[name]}, capprops={"color": colors[name]})
            for box in bp["boxes"]:
                box.set(facecolor=colors[name], edgecolor=colors[name], alpha=0.85)
        a.axhline(truth, color=S.INK, ls="--", lw=1, zorder=0)
        a.set_xticks(range(len(ks)), [str(k) for k in ks])
        a.set_xlim(-0.55, len(ks) - 0.45)
        a.set_xlabel("calibrations per condition")
    ax[1].axhline(RIG_SLOPE, color=S.GREY, ls=":", lw=1.2, zorder=0)
    ax[2].axhline(RIG_LAPSE, color=S.GREY, ls=":", lw=1.2, zorder=0)
    ax[1].set_ylabel("recovered slope β")
    ax[2].set_ylabel("recovered lapse λ")
    S.panel(ax[1], "b", "Slope")
    S.panel(ax[2], "c", "Lapse")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[n]) for n in names]
    labels = [STRATEGIES[n] for n in names]
    handles += [plt.Line2D([], [], color=S.INK, ls="--", lw=1),
                plt.Line2D([], [], color=S.GREY, ls=":", lw=1.2)]
    labels += [f"true value (β {des['true_slope']:g}, λ {des['true_lapse']:g})",
               f"QUEST+ assumption (β {RIG_SLOPE:g}, λ {RIG_LAPSE:g})"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=S.MIN_PT, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    S.save(fig, out)


# ===========================================================================
# Real data (BIDS calibration tables)
# ===========================================================================
def refit_from_bids() -> dict | None:
    """Refit threshold/slope/lapse from the ses-01 calibration _beh.tsv files.

    Each participant has a standing and a walking calibration (acq- entity). A joint fit is
    run per condition; legacy files without acq go under 'calibration'. A slope-invariance
    LRT compares the conditions. Returns ``{"by_condition": {condition: {joint,
    pooled_naive, n_calibrations}}, "slope_invariance": {...}}``.

    Calibrations rejected by the exclusion gate (UNRELIABLE, rule active) are left out: a
    participant at chance has no slope, and a flat calibration pushes the shared slope to
    its bound and creates a spurious standing-vs-walking difference.
    """
    beh = bids_io.find_calibration_tables()
    gated = bids_io.excluded_calibrations()
    by_cond: dict[str, list] = {}
    for f in beh:
        if bids_io.calibration_label(f) in gated:
            continue
        df = pd.read_csv(f, sep="\t")
        if "presented_tilt_deg" not in df or "correct" not in df:
            continue
        # Drop catch and warm-up trials: they use a large fixed tilt (~100 % correct), are
        # not QUEST+ trials, and would bias the threshold and flatten the slope.
        if "tracker" in df:
            df = df[~df["tracker"].astype(str).isin(["catch", "warmup"])]
        d = df[pd.to_numeric(df["correct"], errors="coerce").isin([0, 1])]
        x = pd.to_numeric(d["presented_tilt_deg"], errors="coerce").to_numpy()
        c = pd.to_numeric(d["correct"], errors="coerce").to_numpy()
        m = re.search(r"_acq-(standing|walking)", f.name)
        cond = m.group(1) if m else "calibration"
        by_cond.setdefault(cond, []).append((x, c))
    if not by_cond:
        return None
    out: dict[str, dict] = {}
    for cond, observers in by_cond.items():
        joint = fit_joint(observers)                                          # shared slope/lapse
        pooled = fit_psychometric(np.concatenate([x for x, _ in observers]),
                                  np.concatenate([c for _, c in observers]))  # naive pooling, for comparison
        out[cond] = {"joint": joint, "pooled_naive": pooled,
                     "n_calibrations": len(observers)}
    invariance = fit_slope_invariance(by_cond)   # tests Davidson's slope invariance
    return {"by_condition": out, "slope_invariance": invariance,
            "excluded_by_gate": sorted(gated)}


def _print_simulation(rec: dict) -> None:
    des = rec["design"]
    print("=== Calibration slope/lapse refit — simulation study at the rig's design ===")
    print(f"  {des['n_reps']} repetitions per cohort; true slope {des['true_slope']}, "
          f"true lapse {des['true_lapse']}; QUEST+ assumes slope {RIG_SLOPE}, "
          f"lapse {RIG_LAPSE}")
    for r in rec["summary"]:
        print(f"  {r['n_calibrations']:>2} calibrations | {r['label']:<34}: slope "
              f"{r['slope_median']:.2f} [{r['slope_iqr'][0]:.2f}, {r['slope_iqr'][1]:.2f}] "
              f"(bias {r['slope_bias']:+.2f}), lapse {r['lapse_median']:.3f} "
              f"(bias {r['lapse_bias']:+.3f}), railed {r['railed_pct']:.0f} %")


def stage_simulation(out_dir: Path) -> dict:
    """The pipeline step: run (or reuse) the simulation study and draw its figure."""
    out_json = out_dir / "calibration_refit_simulation.json"
    out_tsv = out_dir / "calibration_refit_simulation_estimates.tsv"
    out_png = bids_io.fig_dir("calibration") / "calibration_refit_simulation.png"
    real = _real_placement()
    fp = stage_cache.fingerprint(
        stage_cache.code_digest("calibration_refit.py", "calibration_convergence_sim.py",
                                "plotstyle.py"),
        np.__version__, [] if real is None else [real[0], real[1]])
    outputs = [out_json, out_tsv, out_png]
    if stage_cache.is_fresh("calib_refit", "simulation", fp, outputs):
        rec = json.loads(out_json.read_text(encoding="utf-8"))
        print("  code and real calibrations unchanged since the last run: reusing it")
        _print_simulation(rec)
        return rec
    rec = simulation_study()
    rec["_estimates"].to_csv(out_tsv, sep="\t", index=False)
    _fig_recovery(rec, out_png, real)
    pub = {k: v for k, v in rec.items() if not k.startswith("_")}
    out_json.write_text(json.dumps(pub, indent=2), encoding="utf-8")
    stage_cache.record("calib_refit", "simulation", fp, outputs)
    _print_simulation(rec)
    return rec


# ===========================================================================
def main():
    """Run the simulation study, or with ``--bids`` refit the real calibrations."""
    out_dir = bids_io.deriv_dir("calibration")
    if "--bids" in sys.argv:
        fit = refit_from_bids()
        if fit is None:
            print("No BIDS calibration tables found.")
            return
        bids_io.gate_report("calib_refit", unit="calibrations")
        (out_dir / "calibration_refit_real.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
        print("=== Real-calibration refit (joint shared slope/lapse, per condition) ===")
        for cond, res in fit["by_condition"].items():
            j, pn = res.get("joint", {}), res.get("pooled_naive", {})
            print(f"  [{cond}] {res.get('n_calibrations')} calibration(s), {j.get('n_trials')} trials")
            print(f"     JOINT : slope beta={j.get('slope_beta')}, lapse={j.get('lapse')}, "
                  f"median threshold(75%)={j.get('thresh75_median_deg')} deg"
                  + (" [RAILED - not trustworthy]" if j.get("railed") else ""))
            if j.get("n_railed_observers"):
                print(f"     ({j['n_railed_observers']}/{j.get('n_participants')} observer(s) had a "
                      "railed own threshold — excluded from the median above)")
            print(f"     (naive pooled, flattens slope: beta={pn.get('slope_beta')}, lapse={pn.get('lapse')})")
        inv = fit["slope_invariance"]
        if inv.get("ok"):
            print("\n  --- Slope-invariance test (Davidson claim: walking shifts threshold, NOT slope) ---")
            print(f"     per-condition slope: {inv['slope_per_condition']}")
            print(f"     shared-slope LRT: chi2({inv['df']})={inv['lr_stat']}, p={inv['p_value']} "
                  f"-> slope {'DIFFERS between conditions' if inv['slope_differs'] else 'invariant (n.s., supports Davidson)'}")
        elif inv.get("reason"):
            print(f"\n  Slope-invariance test skipped: {inv['reason']}")
        print(f"\nOutput -> {out_dir/'calibration_refit_real.json'}")
        return

    stage_simulation(out_dir)


if __name__ == "__main__":
    main()
