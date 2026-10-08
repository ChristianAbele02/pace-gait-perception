"""Bayesian mixed models of stride phase on accuracy and reaction time (secondary).

Trial-level model complementing the confirmatory phase-agnostic test: one model per
measure over walking and standing trials (standing as control), fitted by MCMC (NumPyro
NUTS), which avoids the underestimated uncertainty of the variational fit in ``glmm.py``.

Model (trial j of participant i; W = 1 walking, 0 standing; f = 2 and 4 cps)::

    eta_ij = a + a_i + (b + b_i) W + d_HARD + d_EASY
           + W     * sum_f (mu_walk_f  + z_walk_if ) . (cos, sin)(f * phase_ij)
           + (1-W) * sum_f (mu_stand_f + z_stand_if) . (cos, sin)(f * phase_ij)
    z_*_if ~ N(0, tau_*_f^2 I_2)        # isotropic: no preferred phase built in

* accuracy: Bernoulli-logit on answered trials (timeouts are missing, as in the block
  rule and threshold fits);
* reaction time: Student-t (nu > 2) on log RT of the answered trials, RT >= ``RT_MIN_S``.

Walking trials use the stride phase at stimulus onset (gait.py, as in phase_analysis).
Standing trials (negative control) get a virtual stride phase: onset time on a clock at the
participant's median walking stride time, with a seeded random start. Stimulus timing (ISI
and jitter) is as in walking, so timing-only modulation would also appear in standing, and
no motion signal is used. The head-sway surrogate in glmm.py is not a clean null: as a
zero-phase filtered analytic signal, head movements evoked by stimulus or key press leak
back to the onset, and its onset phases can cluster at the 4-cps harmonic.

What it answers, per measure and frequency:

* Common phase ``mu_walk_f``: an oscillation at a phase shared across participants?
  Amplitude |mu| with 95 % interval and a Savage-Dickey Bayes factor for mu = 0 (2-D
  Gaussian approximation of the posterior at the origin).
* Individual modulation ``tau_walk_f``: do participants oscillate, each at their own phase?
  (the confirmatory question). Savage-Dickey Bayes factor for tau = 0 (reflected kernel
  density at the boundary).
* Walking specificity: the same for standing, ``mu_walk - mu_stand`` and
  P(tau_walk > tau_stand).
* Walking main effect ``b``: accuracy in percentage points, RT in percent. The accuracy
  effect is confounded with calibration transfer per condition (calibration_transfer.py)
  and with condition order; reported, not tested.

Priors follow Davidson et al. (2024): the per-participant RMS amplitude of 0.026 in hit
rate is ~0.14 on the logit scale at 75 % correct, so phase priors are N(0, 0.2) per
component and HalfNormal(0.2) for tau. Savage-Dickey Bayes factors depend on this scale,
which is stored in the output. ``--validate`` simulates the design under a null and under
individual-phase oscillators and reports how often each Bayes factor exceeds 3.

Outputs (``derivatives/bayes_glmm/``): ``bayes_glmm.json`` (estimates, Bayes factors,
sampler diagnostics), ``bayes_glmm_participants.tsv`` and ``figures/bayes_glmm.png|pdf``.

Usage::

    python -m Analysis.bayes_glmm
    python -m Analysis.bayes_glmm --validate      # slow: simulation check of the BFs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, multivariate_normal

try:
    from . import config as C
    from . import bids_io
    from . import phase_analysis as pa
    from . import plotstyle as S
    from . import stage_cache
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import phase_analysis as pa
    import plotstyle as S
    import stage_cache

S.apply()

FREQS = (2.0, 4.0)
RT_MIN_S = 0.15
DEFAULT_STRIDE_S = 1.2       # virtual stride for a standing-only participant
N_CHAINS, N_WARMUP, N_SAMPLES = 4, 1000, 1000
SEED = 20261007
BF_EVIDENCE = 3.0
DAVIDSON_RMS_HITRATE = 0.026
PRIORS = {
    "accuracy": {"a": (1.0, 1.5), "sd_a": 1.0, "b": 1.0, "sd_b": 1.0, "d": 1.0,
                 "mu": 0.2, "tau": 0.2},
    "rt": {"a": (np.log(0.85), 0.5), "sd_a": 0.5, "b": 0.2, "sd_b": 0.2, "d": 0.2,
           "mu": 0.05, "tau": 0.05, "sigma": 0.5},
}
MEASURES = ("accuracy", "rt")
# One XLA host device per chain for parallel chains; must be set before importing JAX.
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={N_CHAINS}")


def available() -> bool:
    """True if NumPyro and JAX are installed; otherwise the stage is skipped."""
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("numpyro", "jax"))


# ===========================================================================
# Data
# ===========================================================================
def _walking_rows(rec) -> pd.DataFrame | None:
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    tp = pa._trial_phases(rec, g["troughs"], step_foot, tg, pa.continuous_phase(g))
    if tp is None:
        return None
    ev = tp[0]
    return pd.DataFrame({"sub": rec.sub, "walk": 1, "phase": ev["phase_target"].to_numpy(float),
                         "difficulty": ev["difficulty"].to_numpy(),
                         "response_correct": ev["response_correct"].to_numpy(float),
                         "rt": ev["response_time"].to_numpy(float)})


def _stride_s(sub: str) -> float:
    """Median stride time over the participant's walking blocks (any gate scope)."""
    vals = []
    for j in (C.DERIV_DIR / "gait" / sub).glob("ses-*/*_acq-walking_*_gait.json"):
        v = json.loads(j.read_text(encoding="utf-8")).get("stride_duration_s")
        if v:
            vals.append(float(v))
    return float(np.median(vals)) if vals else DEFAULT_STRIDE_S


def _standing_rows(rec) -> pd.DataFrame | None:
    ev = bids_io.load_events(rec)
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna()]
    if ev.empty:
        return None
    stride = _stride_s(rec.sub)
    seed = SEED + int.from_bytes(rec.label().encode(), "little") % 100_003
    start = np.random.default_rng(seed).uniform(0, stride)
    phase = 100.0 * (((ev["onset"].to_numpy(float) + start) / stride) % 1.0)
    return pd.DataFrame({"sub": rec.sub, "walk": 0, "phase": phase,
                         "difficulty": ev["difficulty"].to_numpy(),
                         "response_correct": ev["response_correct"].to_numpy(float),
                         "rt": ev["response_time"].to_numpy(float)})


def gather_trials() -> pd.DataFrame:
    """Answered trials of all walking and standing blocks in the behaviour scope."""
    frames = []
    for rec in bids_io.find_recordings():
        f = _walking_rows(rec) if rec.condition == "walking" else (
            _standing_rows(rec) if rec.condition == "standing" else None)
        if f is not None and len(f):
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["response_correct"].isin([0, 1]) & df["phase"].notna()].copy()
    df["correct"] = df["response_correct"].astype(int)
    return df.reset_index(drop=True)


def _arrays(df: pd.DataFrame, measure: str) -> dict:
    if measure == "rt":
        df = df[np.isfinite(df["rt"]) & (df["rt"] >= RT_MIN_S)]
    subs = sorted(df["sub"].unique())
    phase = df["phase"].to_numpy(float)
    harm = np.stack([np.stack([np.cos(2 * np.pi * f * phase / 100.0),
                               np.sin(2 * np.pi * f * phase / 100.0)], -1) for f in FREQS], 1)
    return {"subs": subs,
            "sub": df["sub"].map({s: i for i, s in enumerate(subs)}).to_numpy(int),
            "walk": df["walk"].to_numpy(float),
            "hard": (df["difficulty"] == "HARD").to_numpy(float),
            "easy": (df["difficulty"] == "EASY").to_numpy(float),
            "harm": harm,                                        # (n, n_freq, 2)
            "y": (df["correct"].to_numpy(float) if measure == "accuracy"
                  else np.log(df["rt"].to_numpy(float)))}


# ===========================================================================
# Model
# ===========================================================================
def _model(sub, walk, hard, easy, harm, n_sub, measure, y=None):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    pr = PRIORS[measure]
    nf = harm.shape[1]
    a = numpyro.sample("a", dist.Normal(*pr["a"]))
    sd_a = numpyro.sample("sd_a", dist.HalfNormal(pr["sd_a"]))
    b = numpyro.sample("b", dist.Normal(0.0, pr["b"]))
    sd_b = numpyro.sample("sd_b", dist.HalfNormal(pr["sd_b"]))
    d = numpyro.sample("d", dist.Normal(jnp.zeros(2), pr["d"]))
    with numpyro.plate("participants", n_sub):
        a_i = numpyro.sample("a_raw", dist.Normal(0.0, 1.0)) * sd_a
        b_i = numpyro.sample("b_raw", dist.Normal(0.0, 1.0)) * sd_b
    eta = a + a_i[sub] + (b + b_i[sub]) * walk + d[0] * hard + d[1] * easy
    for cond, mask in (("walk", walk), ("stand", 1.0 - walk)):
        mu = numpyro.sample(f"mu_{cond}", dist.Normal(jnp.zeros((nf, 2)), pr["mu"]))
        tau = numpyro.sample(f"tau_{cond}", dist.HalfNormal(pr["tau"] * jnp.ones(nf)))
        z = numpyro.sample(f"z_{cond}_raw", dist.Normal(jnp.zeros((n_sub, nf, 2)), 1.0))
        coef = mu[None] + z * tau[None, :, None]                  # (n_sub, nf, 2)
        numpyro.deterministic(f"coef_{cond}", coef)
        eta = eta + mask * jnp.sum(coef[sub] * harm, axis=(1, 2))
    if measure == "accuracy":
        numpyro.sample("y", dist.Bernoulli(logits=eta), obs=y)
    else:
        sigma = numpyro.sample("sigma", dist.HalfNormal(pr["sigma"]))
        # nu > 2 keeps the variance finite; chains starting near nu = 0 otherwise fail to mix.
        nu = 2.0 + numpyro.sample("nu_minus_2", dist.Gamma(2.0, 0.1))
        numpyro.sample("y", dist.StudentT(nu, eta, sigma), obs=y)


def fit(arr: dict, measure: str, *, chains=N_CHAINS, warmup=N_WARMUP, samples=N_SAMPLES,
        seed=SEED) -> tuple[dict, dict]:
    """NUTS posterior draws (chains merged) and sampler diagnostics."""
    import jax
    from numpyro.infer import MCMC, NUTS, init_to_median
    from numpyro.diagnostics import summary

    kernel = NUTS(_model, target_accept_prob=0.95, init_strategy=init_to_median(num_samples=15))
    mcmc = MCMC(kernel, num_warmup=warmup,
                num_samples=samples, num_chains=chains, progress_bar=False,
                chain_method="parallel" if jax.local_device_count() >= chains else "sequential")
    mcmc.run(jax.random.PRNGKey(seed), sub=arr["sub"], walk=arr["walk"], hard=arr["hard"],
             easy=arr["easy"], harm=arr["harm"], n_sub=len(arr["subs"]), measure=measure,
             y=arr["y"])
    grouped = mcmc.get_samples(group_by_chain=True)
    keys = ["a", "b", "sd_a", "sd_b", "mu_walk", "tau_walk", "mu_stand", "tau_stand"]
    summ = summary({k: grouped[k] for k in keys}, group_by_chain=True)
    rhat = max(float(np.nanmax(summ[k]["r_hat"])) for k in keys)
    ess = min(float(np.nanmin(summ[k]["n_eff"])) for k in keys)
    div = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return post, {"max_r_hat": round(rhat, 4), "min_ess": round(ess, 1), "divergences": div,
                  "chains": chains, "warmup": warmup, "samples": samples}


# ===========================================================================
# Summaries
# ===========================================================================
def _hdi(x, prob=0.95):
    x = np.sort(np.asarray(x).ravel())
    n = len(x)
    k = int(np.floor(prob * n))
    i = int(np.argmin(x[k:] - x[:n - k]))
    return [float(x[i]), float(x[i + k])]


def bf10_vector_zero(draws, prior_sd) -> float:
    """Savage-Dickey BF10 for a 2-D coefficient = 0 (Gaussian posterior approximation)."""
    draws = np.asarray(draws)
    post0 = multivariate_normal(draws.mean(0), np.cov(draws.T)).pdf(np.zeros(2))
    prior0 = multivariate_normal(np.zeros(2), np.diag(np.square(prior_sd))).pdf(np.zeros(2))
    return float(prior0 / max(post0, 1e-300))


def bf10_scale_zero(draws, prior_sd) -> float:
    """Savage-Dickey BF10 for a half-normal scale = 0 (reflected KDE at the boundary)."""
    x = np.asarray(draws).ravel()
    post0 = 2.0 * float(gaussian_kde(np.concatenate([x, -x]))(0.0)[0])
    prior0 = 2.0 / (prior_sd * np.sqrt(2 * np.pi))
    return float(prior0 / max(post0, 1e-300))


def summarise(post: dict, measure: str, arr: dict) -> dict:
    """Effect sizes in natural units and Bayes factors."""
    pr = PRIORS[measure]
    if measure == "accuracy":
        p = float(np.mean(arr["y"]))
        scale, unit = p * (1 - p), "hit rate"          # logit to hit-rate slope at base rate
    else:
        scale, unit = 1.0, "log RT (~ proportion)"
    out = {"unit": unit, "base_rate": round(float(np.mean(arr["y"])), 4) if measure ==
           "accuracy" else None, "by_frequency": {}}
    b = post["b"]
    out["walking_main_effect"] = (
        {"pp": round(100 * scale * float(np.median(b)), 2),
         "hdi95_pp": [round(100 * scale * v, 2) for v in _hdi(b)]} if measure == "accuracy"
        else {"percent": round(100 * float(np.expm1(np.median(b))), 2),
              "hdi95_percent": [round(100 * float(np.expm1(v)), 2) for v in _hdi(b)]})
    for k, f in enumerate(FREQS):
        res = {}
        for cond in ("walk", "stand"):
            mu = post[f"mu_{cond}"][:, k, :]
            tau = post[f"tau_{cond}"][:, k]
            amp = np.hypot(mu[:, 0], mu[:, 1])
            rms = np.sqrt(amp ** 2 + 2 * tau ** 2)
            res[cond] = {
                "common_amplitude": round(scale * float(np.median(amp)), 5),
                "common_amplitude_hdi95": [round(scale * v, 5) for v in _hdi(amp)],
                "common_phase_deg": round(float(np.degrees(np.arctan2(
                    np.median(mu[:, 1]), np.median(mu[:, 0])))) % 360, 1),
                "bf10_common": round(bf10_vector_zero(mu, [pr["mu"], pr["mu"]]), 3),
                "tau": round(scale * float(np.median(tau)), 5),
                "tau_hdi95": [round(scale * v, 5) for v in _hdi(tau)],
                "bf10_individual": round(bf10_scale_zero(tau, pr["tau"]), 3),
                "rms_amplitude": round(scale * float(np.median(rms)), 5),
                "rms_amplitude_hdi95": [round(scale * v, 5) for v in _hdi(rms)],
            }
        diff = post["mu_walk"][:, k, :] - post["mu_stand"][:, k, :]
        res["walk_minus_stand"] = {
            "bf10_common_difference": round(bf10_vector_zero(
                diff, [pr["mu"] * np.sqrt(2)] * 2), 3),
            "p_tau_walk_gt_stand": round(float(np.mean(post["tau_walk"][:, k] >
                                                       post["tau_stand"][:, k])), 3)}
        out["by_frequency"][f"{f:g}cps"] = res
    return out


def participant_table(post: dict, arr: dict, measure: str) -> pd.DataFrame:
    scale = (np.mean(arr["y"]) * (1 - np.mean(arr["y"]))) if measure == "accuracy" else 1.0
    rows = []
    for i, s in enumerate(arr["subs"]):
        has_walk = bool(np.any(arr["walk"][arr["sub"] == i] == 1))
        has_stand = bool(np.any(arr["walk"][arr["sub"] == i] == 0))
        for k, f in enumerate(FREQS):
            for cond, present in (("walk", has_walk), ("stand", has_stand)):
                amp = scale * np.hypot(*np.moveaxis(post[f"coef_{cond}"][:, i, k, :], -1, 0))
                lo, hi = np.quantile(amp, [0.05, 0.95])
                rows.append({"measure": measure, "participant": s, "freq_cps": f,
                             "condition": "walking" if cond == "walk" else "standing",
                             "has_data": present, "amplitude_median": round(float(np.median(amp)), 5),
                             "amplitude_q05": round(float(lo), 5),
                             "amplitude_q95": round(float(hi), 5)})
    return pd.DataFrame(rows)


# ===========================================================================
# Figure
# ===========================================================================
def _figure(results: dict, posts: dict, stem: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(S.COL2_IN, 2.9),
                            gridspec_kw={"width_ratios": [1.0, 1.3, 0.75]})
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.2, top=0.86, wspace=0.62)

    # a: posterior of the common-phase coefficient, accuracy at 2 cps
    ax = axs[0]
    acc = posts["accuracy"]
    sc = results["accuracy"]["base_rate"] * (1 - results["accuracy"]["base_rate"])
    for cond, col in (("stand", S.COND["standing"]), ("walk", S.COND["walking"])):
        mu = acc[f"mu_{cond}"][::4, 0, :] * sc * 100
        ax.scatter(mu[:, 0], mu[:, 1], s=2, color=col, alpha=0.25, lw=0, rasterized=True)
    ax.axhline(0, color=S.GREY, lw=0.8)
    ax.axvline(0, color=S.GREY, lw=0.8)
    lim = max(4.0, float(np.quantile(np.abs(acc["mu_walk"][:, 0, :] * sc * 100), 0.995)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("cos (pp)")
    ax.set_ylabel("sin (pp)")
    S.panel(ax, "a", "Shared phase, 2 cps")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=4, color=S.COND[c], label=c)
                       for c in ("walking", "standing")], loc="upper left", fontsize=8,
              handletextpad=0.2)

    # b: individual modulation (RMS amplitude), walking vs standing, both measures
    ax = axs[1]
    labels, y = [], 0
    for meas, unit_scale, name in (("accuracy", 100, "acc"), ("rt", 100, "RT")):
        for f in FREQS:
            for cond in ("walk", "stand"):
                r = results[meas]["by_frequency"][f"{f:g}cps"][cond]
                lo, hi = r["rms_amplitude_hdi95"]
                col = S.COND["walking" if cond == "walk" else "standing"]
                ax.plot([lo * unit_scale, hi * unit_scale], [y, y], color=col, lw=1.6)
                ax.plot(r["rms_amplitude"] * unit_scale, y, "o", ms=4.5, color=col)
                y -= 1
            labels.append((y + 1.5, f"{name} {f:g} cps"))
            y -= 0.6
    ax.axvline(DAVIDSON_RMS_HITRATE * 100, color=S.GREY, ls="--", lw=0.9,
               label="Davidson (acc)")
    ax.set_ylim(y - 0.9, 0.5)                      # room for the legend
    ax.legend(loc="lower right", fontsize=8, handlelength=1.4)
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([t for _, t in labels])
    ax.set_xlabel("RMS amplitude (pp or %)")
    ax.grid(axis="y", visible=False)
    S.panel(ax, "b", "Per-person modulation")

    # c: walking main effect
    ax = axs[2]
    ra = results["accuracy"]["walking_main_effect"]
    rr = results["rt"]["walking_main_effect"]
    for yy, est, (lo, hi) in ((1, ra["pp"], ra["hdi95_pp"]), (0, rr["percent"], rr["hdi95_percent"])):
        ax.plot([lo, hi], [yy, yy], color=S.INK, lw=1.6)
        ax.plot(est, yy, "o", ms=4.5, color=S.INK)
    ax.axvline(0, color=S.GREY, lw=0.9)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["acc (pp)", "RT (%)"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("walking − standing")
    ax.grid(axis="y", visible=False)
    S.panel(ax, "c", "Walking")
    S.save_pub(fig, stem)


# ===========================================================================
# Validation by simulation
# ===========================================================================
def _simulate(arr: dict, rng, *, tau_logit: float, measure="accuracy") -> dict:
    """Simulate accuracy on the real design: base logit per participant and condition from
    the data, plus an individual-phase oscillation of scale ``tau_logit`` when walking."""
    n_sub = len(arr["subs"])
    base = np.zeros((n_sub, 2))
    for i in range(n_sub):
        for w in (0, 1):
            m = (arr["sub"] == i) & (arr["walk"] == w)
            p = np.clip(arr["y"][m].mean(), 0.55, 0.95) if m.any() else 0.75
            base[i, w] = np.log(p / (1 - p))
    z = rng.normal(0, tau_logit, size=(n_sub, 2))              # 2 cps only
    eta = base[arr["sub"], arr["walk"].astype(int)] + arr["walk"] * np.sum(
        z[arr["sub"]] * arr["harm"][:, 0, :], axis=1)
    sim = dict(arr)
    sim["y"] = (rng.random(len(eta)) < 1 / (1 + np.exp(-eta))).astype(float)
    return sim


def validate(n_rep: int = 20) -> dict:
    """Rate of Bayes factors above 3 under a null and under individual phases."""
    arr = _arrays(gather_trials(), "accuracy")
    rng = np.random.default_rng(SEED)
    p = float(np.mean(arr["y"]))
    tau_dav = DAVIDSON_RMS_HITRATE / (p * (1 - p)) / np.sqrt(2)  # logit, per component
    out = {}
    for name, tau in (("null", 0.0), ("idiosyncratic_davidson", tau_dav),
                      ("idiosyncratic_3x", 3 * tau_dav)):
        hits = {"common": 0, "individual": 0, "individual_null": 0}
        for r in range(n_rep):
            post, _ = fit(_simulate(arr, rng, tau_logit=tau), "accuracy", chains=2,
                          warmup=500, samples=500, seed=SEED + r)
            s = summarise(post, "accuracy", arr)["by_frequency"]["2cps"]["walk"]
            hits["common"] += s["bf10_common"] > BF_EVIDENCE
            hits["individual"] += s["bf10_individual"] > BF_EVIDENCE
            hits["individual_null"] += s["bf10_individual"] < 1 / BF_EVIDENCE
        out[name] = {"tau_logit": round(float(tau), 4), "n_rep": n_rep,
                     **{f"rate_bf10_{k}_gt3": round(v / n_rep, 3)
                        for k, v in hits.items() if k != "individual_null"},
                     "rate_bf10_individual_lt_1/3": round(hits["individual_null"] / n_rep, 3)}
        print(f"  {name}: {out[name]}")
    path = bids_io.deriv_dir("bayes_glmm") / "bayes_glmm_validation.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  -> {path}")
    return out


# ===========================================================================
# Main
# ===========================================================================
def _describe(results: dict) -> None:
    for meas in MEASURES:
        r = results[meas]
        print(f"  [{meas}] diagnostics: {r['diagnostics']}")
        we = r["walking_main_effect"]
        print("    walking main effect: " + (
            f"{we['pp']:+.2f} pp [{we['hdi95_pp'][0]:+.2f}, {we['hdi95_pp'][1]:+.2f}]"
            if meas == "accuracy" else
            f"{we['percent']:+.2f} % [{we['hdi95_percent'][0]:+.2f}, {we['hdi95_percent'][1]:+.2f}]"))
        u = 100
        for fk, fr in r["by_frequency"].items():
            for cond in ("walk", "stand"):
                c = fr[cond]
                print(f"    {fk} {cond:<5}: shared amp {u * c['common_amplitude']:.2f} "
                      f"[{u * c['common_amplitude_hdi95'][0]:.2f}, {u * c['common_amplitude_hdi95'][1]:.2f}] "
                      f"BF10={c['bf10_common']:.2g} | per-person tau {u * c['tau']:.2f} "
                      f"[{u * c['tau_hdi95'][0]:.2f}, {u * c['tau_hdi95'][1]:.2f}] "
                      f"BF10={c['bf10_individual']:.2g} | RMS {u * c['rms_amplitude']:.2f}")
            d = fr["walk_minus_stand"]
            print(f"    {fk} walk-stand: BF10(shared differs)={d['bf10_common_difference']:.2g}, "
                  f"P(tau_walk > tau_stand)={d['p_tau_walk_gt_stand']}")


def main(argv=None) -> dict | None:
    """Fit both measures, summarise, and write outputs and figure."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true", help="simulation check of the BFs")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from the saved results and draws")
    args = ap.parse_args(argv)
    if args.figure_only:
        out = bids_io.deriv_dir("bayes_glmm")
        results = json.loads((out / "bayes_glmm.json").read_text(encoding="utf-8"))
        d = np.load(out / "bayes_glmm_draws.npz")
        posts = {m: {k[len(m) + 1:]: d[k] for k in d.files if k.startswith(m + "_")}
                 for m in MEASURES}
        _figure(results, posts, bids_io.fig_dir("bayes_glmm") / "bayes_glmm")
        return results
    if not available():
        print("NumPyro/JAX not installed (pip install 'jax[cpu]' numpyro) - stage skipped.")
        return None
    if args.validate:
        return validate(args.reps)

    df = gather_trials()
    bids_io.gate_report("bayes_glmm", condition=("standing", "walking"),
                        analysed_subjects=sorted(df["sub"].unique()) if len(df) else [])
    if df.empty or df["sub"].nunique() < 2:
        print("Fewer than 2 participants with trials - nothing to fit.")
        return None
    arrs = {meas: _arrays(df, meas) for meas in MEASURES}
    out = bids_io.deriv_dir("bayes_glmm")
    outputs = [out / "bayes_glmm.json", out / "bayes_glmm_participants.tsv",
               out / "bayes_glmm_draws.npz"]
    # Reuse the fit if trial arrays, code, priors, sampler settings and library versions
    # are unchanged.
    import jax
    import numpyro
    fp = stage_cache.fingerprint(
        stage_cache.code_digest("bayes_glmm.py"), jax.__version__, numpyro.__version__,
        PRIORS, FREQS, RT_MIN_S, [N_CHAINS, N_WARMUP, N_SAMPLES, SEED],
        *[[arrs[m]["subs"], arrs[m]["sub"], arrs[m]["walk"], arrs[m]["hard"],
           arrs[m]["easy"], arrs[m]["harm"], arrs[m]["y"]] for m in MEASURES])
    if stage_cache.is_fresh("bayes_glmm", "fit", fp, outputs):
        print("=== Bayesian mixed models: stride phase x condition (SECONDARY) ===")
        print("  inputs, code and settings unchanged since the last fit: reusing it")
        results = json.loads(outputs[0].read_text(encoding="utf-8"))
        d = np.load(outputs[2])
        posts = {m: {k[len(m) + 1:]: d[k] for k in d.files if k.startswith(m + "_")}
                 for m in MEASURES}
        _figure(results, posts, bids_io.fig_dir("bayes_glmm") / "bayes_glmm")
    else:
        results, posts, tabs = {}, {}, []
        for meas in MEASURES:
            arr = arrs[meas]
            post, diag = fit(arr, meas)
            posts[meas] = post
            results[meas] = {"n_trials": int(len(arr["y"])), "n_participants": len(arr["subs"]),
                             "diagnostics": diag, **summarise(post, meas, arr)}
            tabs.append(participant_table(post, arr, meas))
        outputs[0].write_text(json.dumps({
            "Description": "Bayesian mixed models of stride phase (walking) and a virtual "
                           "stride phase (standing) on accuracy and log RT, NumPyro NUTS "
                           "(bayes_glmm.py). Secondary to the phase-agnostic confirmatory "
                           "test.",
            "Priors": PRIORS, "Frequencies_cps": list(FREQS), "BF_evidence": BF_EVIDENCE,
            "Units": "accuracy: hit-rate at the base rate; rt: log RT (about a proportion)",
            **results}, indent=2, default=float), encoding="utf-8")
        pd.concat(tabs).to_csv(outputs[1], sep="\t", index=False)
        np.savez_compressed(outputs[2],
                            **{f"{m}_{k}": posts[m][k] for m in MEASURES
                               for k in ("mu_walk", "mu_stand", "tau_walk", "tau_stand", "b")})
        stage_cache.record("bayes_glmm", "fit", fp, outputs)
        _figure(results, posts, bids_io.fig_dir("bayes_glmm") / "bayes_glmm")
        print("=== Bayesian mixed models: stride phase x condition (SECONDARY) ===")
    _describe(results)
    bad = [m for m in MEASURES if results[m]["diagnostics"]["max_r_hat"] > 1.01
           or results[m]["diagnostics"]["divergences"]]
    if bad:
        print(f"  ! sampler warnings for {bad}: check r_hat / divergences before reading")
    print(f"Outputs -> {out}")
    return results


if __name__ == "__main__":
    main()
