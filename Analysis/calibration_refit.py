"""Offline refit of the psychometric slope and lapse from calibration data (b8).

QUEST+ (``StaircaseManager``) estimates only the threshold, with Weibull slope and lapse
fixed at assumed values. With several participants' calibrations these can be checked by
fitting threshold, slope and lapse by maximum likelihood. The refined values can be fed
back into QUEST+ (with priors re-centred on the observed thresholds), making calibrations
faster and less prone to lapse-driven runaway.

Model (2-AFC Weibull on tilt magnitude, guess rate fixed at 0.5)::

    psi(x) = 0.5 + (0.5 - lambda) * (1 - exp(-(x/alpha)^beta))
    alpha = threshold (deg), beta = slope, lambda = lapse

By default ``main`` runs a simulation study: N observers with known threshold, slope and lapse are refitted individually and
pooled, and parameter recovery is reported with a figure. ``refit_from_bids()`` (``--bids``)
refits the real BIDS calibration tables.

Usage::

    python -m Analysis.calibration_refit            # simulation-study demo + figure
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
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import plotstyle as S

S.apply()

GUESS = 0.5                 # 2-AFC chance level
LAPSE_MAX = 0.20
RNG = np.random.default_rng(11)

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


def fit_joint(observers) -> dict:
    """MLE of one shared slope and lapse with per-participant thresholds.

    Pooling trials across observers with different thresholds flattens the slope; a
    joint fit with individual thresholds avoids this. ``observers`` is a list of
    ``(tilt, correct)`` arrays.
    """
    obs = [(np.asarray(x, float), np.asarray(c, float)) for x, c in observers
           if np.asarray(x).size >= 8 and np.unique(c).size >= 2]
    K = len(obs)
    if K == 0:
        return {"ok": False, "reason": "no usable observers"}

    def nll(theta):
        log_beta, lapse = theta[0], theta[1]
        beta = np.exp(log_beta)
        total = 0.0
        for j, (x, c) in enumerate(obs):
            p = weibull(x, np.exp(theta[2 + j]), beta, lapse)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            total -= np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))
        return float(total)

    x0 = [np.log(3.0), 0.03] + [np.log(max(1.0, np.median(x))) for x, _ in obs]
    bounds = [(np.log(0.5), np.log(12)), (0.0, LAPSE_MAX)] + [(np.log(0.1), np.log(45))] * K
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)
    beta, lapse = float(np.exp(res.x[0])), float(res.x[1])
    alphas = [float(np.exp(v)) for v in res.x[2:]]
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
# Simulation study (parameter recovery)
# ===========================================================================
def simulate_observer(n_trials, alpha, beta, lapse, levels=None):
    """Simulate (tilt, correct) for one observer at levels around its threshold."""
    if levels is None:
        levels = alpha * np.geomspace(0.4, 2.2, 7)     # levels around the threshold
    x = RNG.choice(levels, size=n_trials)
    p = weibull(x, alpha, beta, lapse)
    correct = (RNG.uniform(size=n_trials) < p).astype(float)
    return x, correct


def simulation_study(n_participants=10, n_trials=80,
                     true_slope=3.5, true_lapse=0.03,
                     thr_mean=5.0, thr_sd=1.2) -> dict:
    """Simulate observers with known parameters; refit individually, pooled and jointly."""
    truth, per, observers = [], [], []
    pooled_x, pooled_c = [], []
    for i in range(n_participants):
        alpha = float(max(1.0, RNG.normal(thr_mean, thr_sd)))
        x, c = simulate_observer(n_trials, alpha, true_slope, true_lapse)
        truth.append({"alpha": alpha, "beta": true_slope, "lapse": true_lapse})
        per.append(fit_psychometric(x, c))
        observers.append((x, c))
        pooled_x.append(x); pooled_c.append(c)
    pooled = fit_psychometric(np.concatenate(pooled_x), np.concatenate(pooled_c))  # naive pooling (flattens slope)
    joint = fit_joint(observers)                                                   # shared slope/lapse
    ok = [p for p in per if p.get("ok")]
    rec = {
        "n_participants": n_participants, "n_trials_each": n_trials,
        "true_slope": true_slope, "true_lapse": true_lapse,
        "true_threshold_mean": thr_mean,
        "per_participant_slope_median": round(float(np.median([p["slope_beta"] for p in ok])), 3) if ok else None,
        "per_participant_lapse_median": round(float(np.median([p["lapse"] for p in ok])), 4) if ok else None,
        "pooled_slope": pooled.get("slope_beta"), "pooled_lapse": pooled.get("lapse"),
        "pooled_threshold75": pooled.get("thresh75_deg"),
        # values to feed back into QUEST+:
        "joint_slope": joint.get("slope_beta"), "joint_lapse": joint.get("lapse"),
        "joint_threshold75_median": joint.get("thresh75_median_deg"),
        "_truth": truth, "_per": per, "_pooled": pooled, "_joint": joint,
    }
    return rec


def _fig_recovery(rec: dict, out: Path):
    per, truth = rec["_per"], rec["_truth"]
    ok = [(t, p) for t, p in zip(truth, per) if p.get("ok")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # a) joint vs naive pooled fit
    x = np.linspace(0.5, 14, 200)
    joint = rec["_joint"]
    if joint.get("ok"):
        a_med = float(np.median(joint["thresholds_alpha_deg"]))
        ax[0].plot(x, weibull(x, a_med, joint["slope_beta"], joint["lapse"]), color=S.GREEN, lw=2.5,
                   label=f"joint fit (β = {joint['slope_beta']}, λ = {joint['lapse']})")
    pooled = rec["_pooled"]
    if pooled.get("ok"):
        ax[0].plot(x, weibull(x, pooled["threshold_alpha_deg"], pooled["slope_beta"],
                              pooled["lapse"]), color=S.PURPLE, lw=1.8, ls="--",
                   label=f"naive pooled (β = {pooled['slope_beta']}, flattened)")
    ax[0].plot(x, weibull(x, rec["true_threshold_mean"], rec["true_slope"], rec["true_lapse"]),
               color=S.INK, ls=":", lw=1.6,
               label=f"truth (β = {rec['true_slope']}, λ = {rec['true_lapse']})")
    S.reference_line(ax[0], y=0.75, ls=":")
    ax[0].set_xlabel("tilt (°)"); ax[0].set_ylabel("P(correct)")
    S.panel(ax[0], "a", "Psychometric recovery", fontsize=10); ax[0].legend(loc="lower right")

    # b) recovered slope: per participant, pooled, joint
    ax[1].axhline(rec["true_slope"], color=S.INK, ls="--", label="true slope")
    ax[1].scatter(range(len(ok)), [p["slope_beta"] for _, p in ok], color=S.BLUE, label="per-participant")
    ax[1].axhline(rec["pooled_slope"], color=S.PURPLE, lw=1.8, ls="--", label="naive pooled")
    ax[1].axhline(rec["joint_slope"], color=S.GREEN, lw=2.5, label="joint")
    ax[1].set_xlabel("participant"); ax[1].set_ylabel("recovered slope β")
    S.panel(ax[1], "b", "Slope recovery", fontsize=10); ax[1].legend(loc="best")

    # c) recovered lapse
    ax[2].axhline(rec["true_lapse"], color=S.INK, ls="--", label="true lapse")
    ax[2].scatter(range(len(ok)), [p["lapse"] for _, p in ok], color=S.RED, label="per-participant")
    ax[2].axhline(rec["joint_lapse"], color=S.GREEN, lw=2.5, label="joint")
    ax[2].set_xlabel("participant"); ax[2].set_ylabel("recovered lapse λ")
    S.panel(ax[2], "c", "Lapse recovery", fontsize=10); ax[2].legend(loc="best")

    fig.suptitle(f"Calibration slope/lapse refit — {rec['n_participants']} simulated "
                 f"participants × {rec['n_trials_each']} trials", fontsize=12,
                 fontweight="semibold", x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); S.save(fig, out)


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

    rec = simulation_study()
    pub = {k: v for k, v in rec.items() if not k.startswith("_")}
    (out_dir / "calibration_refit_simulation.json").write_text(json.dumps(pub, indent=2), encoding="utf-8")
    _fig_recovery(rec, bids_io.fig_dir("calibration") / "calibration_refit_simulation.png")
    print("=== Calibration slope/lapse refit — simulation study ===")
    print(f"  {rec['n_participants']} participants x {rec['n_trials_each']} trials; "
          f"true slope {rec['true_slope']}, true lapse {rec['true_lapse']}")
    print(f"  per-participant median : slope {rec['per_participant_slope_median']} (noisy at low n)")
    print(f"  naive pooled           : slope {rec['pooled_slope']} (BIASED LOW - threshold "
          "heterogeneity flattens the aggregate slope)")
    print(f"  JOINT (recommended)    : slope {rec['joint_slope']}, lapse {rec['joint_lapse']}, "
          f"median threshold(75%) {rec['joint_threshold75_median']} deg")
    print("  -> a joint fit (shared slope/lapse, per-participant thresholds) recovers the truth; "
          "feed these back into QUEST+ and re-centre the threshold prior.")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
