"""Hierarchical GLMM of the 2-cps stride oscillation (descriptive only, no tests).

Estimates the population oscillation: amplitude in logit and hit-rate units, preferred
phase, and their breakdown by tilt difficulty and condition. No p-values are reported.

Why no p-values
---------------
The model is fitted by variational inference (``BinomialBayesMixedGLM``). Mean-field
posteriors understate variance, and here the fixed-effect SEs on (b_cos, b_sin) cannot
absorb the random-slope variance on the same regressors. In null simulations the former
Wald test had a 60 % false-positive rate when participants oscillated at individual phases,
the expected case given Davidson's phase coherence of R = 0.33. The test was removed.

Inference
---------
* Confirmatory: ``group_agnostic.py``, the phase-agnostic statistic
  ``S = sum_i |z_i|^2 / s_i^2`` with a null from the participants' own permutations.
* Common phase (secondary): ``common_phase.py``, one-sample Hotelling T^2 on the
  per-participant coefficients (exact in small samples, same statistic as the power
  analysis; simulated false-positive rate 4 % without and 9 % with individual phases).
* Walking specificity: ``common_phase.interaction_test``, paired Hotelling T^2 on walking
  minus standing coefficients.

Model (accuracy)::

    correct_ij ~ Bernoulli(p_ij)
    logit(p_ij) = b0 + b_cos*cos(w*phase_ij) + b_sin*sin(w*phase_ij)
                + u_i + v_i*cos(w*phase_ij) + s_i*sin(w*phase_ij)
    u_i ~ N(0, tau0^2)          # participant random intercept
    v_i, s_i ~ N(0, tau^2)      # participant random slopes on cos / sin
    w = 2*pi*freq/100

Random slopes are needed even for point estimates: phases differ between participants, and
an intercept-only model would attribute that variance to the fixed effect. Both 2 cps and
4 cps are estimated, since responders peak at one or the other.

With >= 2 participants this is a Bayesian mixed GLM; with one participant the random effects
are not identifiable and a fixed-effects logistic regression is used (flagged in ``method``).

Usage::

    python -m Analysis.glmm
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
from scipy.signal import detrend

try:
    from . import bids_io
    from . import clean
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import clean
    import gait as gait_mod
    import phase_analysis as pa
    import plotstyle as S

S.apply()

FREQ_CPS = 2.0          # main frequency (2 cycles per stride)
# response likelihood removed (see config, outcome measures)
MEASURES = ("accuracy",)
DIFF_ORDER = ("HARD", "THR", "EASY")                      # increasing target accuracy
DIFF_COL = {"HARD": S.PURPLE, "THR": S.ORANGE, "EASY": S.TEAL}
DIFF_LABEL = {"HARD": "hard (65%)", "THR": "threshold (75%)", "EASY": "easy (85%)"}


# ===========================================================================
# Trial-level data
# ===========================================================================
def _walking_trial_phases(rec) -> pd.DataFrame | None:
    """One row per walking trial: sub, condition, stride phase at onset, correctness.

    Stride phase from gait.py (method per config.PHASE_METHOD), as in phase_analysis.
    """
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
    return pd.DataFrame({"sub": rec.label().split("_")[0], "condition": "walking",
                         "phase": ev["phase_target"].to_numpy(float),
                         "correct": ev["acc"].to_numpy(float),
                         "difficulty": ev["difficulty"].to_numpy()})


def _standing_trial_phases(rec) -> pd.DataFrame | None:
    """Surrogate-phase rows for standing trials (negative control).

    Standing has no stride phase. Each trial gets a Hilbert phase of the participant's
    vertical head signal (sway, breathing), computed as for walking (clean, resample,
    band-pass, analytic phase). This is not a locomotor rhythm, so the standing amplitude
    is expected to be ~0. The surrogate only makes the condition x phase model well-posed.
    Note: bayes_glmm.py explains why this surrogate is not a fully clean null.
    """
    motion, _, _ = bids_io.load_motion(rec)
    if "head_local_pos_y" not in motion:
        return None
    t = motion["t"].to_numpy()
    y = clean.clean_channel(t, motion["head_local_pos_y"].to_numpy(),
                            abs_radius_m=clean.HEAD_CLEAN_AXES["head_local_pos_y"])[0]
    fin = np.isfinite(y)
    if fin.sum() < int(2 * gait_mod.RESAMPLE_FS):
        return None
    tg, yg = bids_io.uniform_resample(t[fin], y[fin], fs=gait_mod.RESAMPLE_FS)
    yf = gait_mod._bandpass(detrend(yg, type="linear"), gait_mod.RESAMPLE_FS,
                            0.5, gait_mod.LOWPASS_HZ)
    cont, _ = gait_mod.continuous_stride_phase(tg, yf, None, None)

    ev = bids_io.load_events(rec)
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna()].copy()
    if ev.empty:
        return None
    _, pct_stride = gait_mod.phase_at_hilbert(ev["onset"].to_numpy(float), tg, cont)
    return pd.DataFrame({"sub": rec.label().split("_")[0], "condition": "standing",
                         "phase": pct_stride,
                         "correct": np.where(ev["response_correct"] == 1, 1.0, 0.0),
                         "difficulty": ev["difficulty"].to_numpy()})


def _gather_trials(conditions=("walking",)) -> pd.DataFrame:
    """One row per analysis trial across the requested conditions.

    Columns: sub, condition, phase, correct, difficulty. Default is walking only; pass
    ``("walking", "standing")`` for the condition x phase model.
    """
    frames = []
    for rec in bids_io.find_recordings():
        if rec.condition == "walking" and "walking" in conditions:
            f = _walking_trial_phases(rec)
        elif rec.condition == "standing" and "standing" in conditions:
            f = _standing_trial_phases(rec)
        else:
            continue
        if f is not None and not f.empty:
            frames.append(f)
    cols = ["sub", "condition", "phase", "correct", "difficulty"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True).dropna(subset=["phase", "correct"])


# ===========================================================================
# Fit
# ===========================================================================
def _design(df: pd.DataFrame, freq: float) -> pd.DataFrame:
    w = 2 * np.pi * freq / 100.0
    out = df.copy()
    out["cosph"] = np.cos(w * out["phase"])
    out["sinph"] = np.sin(w * out["phase"])
    return out


# The joint Wald test was removed (see module docstring); common-phase inference is in
# common_phase.py.


def fit_population(df: pd.DataFrame, freq: float = FREQ_CPS,
                   random_slopes: bool = True) -> dict | None:
    """Fit the population oscillation model at one frequency.

    Logistic mixed model of accuracy on cos/sin of stride phase with random participant
    intercepts and (by default) random slopes on both phase terms. With one participant a
    fixed-effects logit is used; ``method`` records which.

    Returns amplitude (logit and approximate hit rate) and preferred phase, without a
    p-value.
    """
    if df.empty or df["correct"].nunique() < 2:
        return None
    d = _design(df, freq)
    n_sub = d["sub"].nunique()
    base = float(d["correct"].mean())

    method, b, sd, note = None, {}, {}, ""
    if n_sub >= 2:
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            # Random intercept and (default) random slopes on cos/sin, since phase and
            # amplitude vary between participants.
            vc = {"sub": "0 + C(sub)"}
            if random_slopes:
                vc["cos_slope"] = "0 + C(sub):cosph"
                vc["sin_slope"] = "0 + C(sub):sinph"
            model = BinomialBayesMixedGLM.from_formula("correct ~ cosph + sinph", vc, d)
            res = model.fit_vb()
            names = list(res.model.exog_names)
            fe = np.asarray(res.fe_mean); fesd = np.asarray(res.fe_sd)
            idx = {nm: i for i, nm in enumerate(names)}
            for key in ("Intercept", "cosph", "sinph"):
                b[key], sd[key] = float(fe[idx[key]]), float(fesd[idx[key]])
            method = ("BinomialBayesMixedGLM (random participant intercept"
                      + (" + cos/sin slopes)" if random_slopes else ")"))
        except Exception as e:                          # pragma: no cover
            note = f"mixed-GLM fit failed ({type(e).__name__}: {e}); fell back to fixed-effects logit. "
            method = None
    if method is None:
        import statsmodels.api as sm
        X = sm.add_constant(d[["cosph", "sinph"]].to_numpy())
        res = sm.Logit(d["correct"].to_numpy(), X).fit(disp=0)
        b = {"Intercept": float(res.params[0]), "cosph": float(res.params[1]), "sinph": float(res.params[2])}
        sd = {"Intercept": float(res.bse[0]), "cosph": float(res.bse[1]), "sinph": float(res.bse[2])}
        method = (note + ("fixed-effects logistic regression (n=1 participant: random "
                          "effect not identifiable)" if n_sub < 2
                          else "fixed-effects logistic regression"))

    amp_logit = float(np.hypot(b["cosph"], b["sinph"]))
    pref_phase = float(np.arctan2(-b["sinph"], b["cosph"]))      # A*cos(w*x + phi)
    # delta-method hit-rate amplitude at the base rate
    amp_rate = float(amp_logit * base * (1 - base))
    return {
        "freq_cps": freq, "n_participants": int(n_sub), "n_trials": int(len(d)),
        "method": method, "base_rate": round(base, 4),
        "b_intercept": round(b["Intercept"], 4),
        "b_cos": round(b["cosph"], 4), "b_sin": round(b["sinph"], 4),
        "sd_cos": round(sd["cosph"], 4), "sd_sin": round(sd["sinph"], 4),
        "amp_logit": round(amp_logit, 4), "amp_hitrate": round(amp_rate, 4),
        "preferred_phase_rad": round(pref_phase, 4),
        # no p-value; inference in common_phase.hotelling_t2
        "inference": "descriptive only - see common_phase.py (Hotelling T2)",
    }


# ===========================================================================
# Amplitude by difficulty level (descriptive)
# ===========================================================================
def fit_difficulty_model(df: pd.DataFrame, freq: float = FREQ_CPS) -> dict | None:
    """2-cps amplitude per tilt level (descriptive, not a test).

    The hypothesis that modulation is largest at threshold (H3/H6) was dropped: with the
    0.85 / 0.10 / 0.05 mix, 1440 walking trials give only ~144 EASY and ~72 HARD trials, too
    few for a 2-cps fit over 40 bins. Testing it would need a balanced difficulty mix.

    EASY and HARD compress accuracy toward ceiling and floor, which lowers their amplitudes
    independently of the mechanism.
    """
    dw = df[df["condition"] == "walking"].copy()
    if dw.empty or "difficulty" not in dw.columns:
        return None
    keep = ("amp_logit", "amp_hitrate", "preferred_phase_rad",
            "n_participants", "n_trials", "base_rate",
            "b_intercept", "b_cos", "b_sin")
    by = {}
    for d in DIFF_ORDER:
        sub = dw[dw["difficulty"] == d]
        # Random intercept only: random slopes are unstable in these small subsets.
        f = fit_population(sub, freq, random_slopes=False) if len(sub) else None
        if f:
            by[d] = {k: f[k] for k in keep}
    if not by:
        return None
    amps = {d: by[d]["amp_logit"] for d in by}
    return {
        "freq_cps": freq, "by_difficulty": by,
        "amp_order_high_to_low": sorted(amps, key=amps.get, reverse=True),
        "thr_amplitude_largest": bool("THR" in amps and amps["THR"] >= max(amps.values())),
        "inference": "DESCRIPTIVE ONLY - no test. The 'largest at threshold' hypothesis "
                     "(H3) is retired: the 0.85/0.10/0.05 difficulty mix leaves ~108 EASY "
                     "and ~54 HARD trials, too few to fit a 2-cps oscillation.",
        "caveat": "EASY/HARD compress accuracy toward ceiling/floor, which alone lowers "
                  "measurable modulation; read the ordering, ideally on a sensitivity scale.",
    }


def _figure_difficulty(df: pd.DataFrame, dm: dict, out: Path):
    dw = df[df["condition"] == "walking"]
    w = 2 * np.pi * dm["freq_cps"] / 100.0
    xf = np.linspace(0, 100, 200)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.7),
                                   gridspec_kw=dict(width_ratios=[1.65, 1.0]))
    for d in DIFF_ORDER:
        m = dm["by_difficulty"].get(d)
        if not m:
            continue
        col = DIFF_COL[d]
        cen, obs = _binned_hitrate(dw[dw["difficulty"] == d])
        axA.scatter(cen, obs, s=14, color=col, alpha=0.40, zorder=1)
        logit = m["b_intercept"] + m["b_cos"] * np.cos(w * xf) + m["b_sin"] * np.sin(w * xf)
        axA.plot(xf, 1 / (1 + np.exp(-logit)), color=col, lw=2.4, zorder=3,
                 label=f"{DIFF_LABEL[d]}: amp {m['amp_hitrate']:.3f}")
    axA.set_xlabel("stride phase (% stride)"); axA.set_ylabel("hit rate")
    S.panel(axA, "a", "2-cps oscillation by difficulty", fontsize=10)
    axA.legend(loc="upper right", fontsize=8)

    ds = [d for d in DIFF_ORDER if d in dm["by_difficulty"]]
    amps = [dm["by_difficulty"][d]["amp_hitrate"] for d in ds]
    axB.bar(range(len(ds)), amps, color=[DIFF_COL[d] for d in ds], alpha=0.85,
            edgecolor="white", zorder=2)
    for i, d in enumerate(ds):
        axB.text(i, amps[i], f"{dm['by_difficulty'][d]['n_trials']}",
                 ha="center", va="bottom", fontsize=8, color=S.GREY)
    axB.set_xticks(range(len(ds))); axB.set_xticklabels([d.title() for d in ds])
    axB.set_ylabel("2-cps amplitude (hit-rate)")
    S.panel(axB, "b", "Amplitude by difficulty  (descriptive; n trials above bars)",
            fontsize=10)

    fig.suptitle(f"Difficulty-dependent stride-phase modulation (H6) — "
                 f"n = {dw['sub'].nunique()} participant(s), {len(dw)} walking trials",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "EASY/HARD compress accuracy toward ceiling/floor; read the ordering, "
                   "not raw amplitude. Per-level fit = random-intercept GLMM (fixed-effects "
                   "fallback at n=1).")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); S.save(fig, out)


# ===========================================================================
# Combined condition x phase model
# ===========================================================================
def _standardised(b: float, sd: float) -> float:
    """Coefficient divided by its SE, as a descriptive effect size (not converted to a p)."""
    return float(b / sd) if sd > 0 else 0.0


def _design_condition(df: pd.DataFrame, freq: float) -> pd.DataFrame:
    w = 2 * np.pi * freq / 100.0
    out = df.copy()
    out["cosph"] = np.cos(w * out["phase"])
    out["sinph"] = np.sin(w * out["phase"])
    out["walk"] = (out["condition"] == "walking").astype(float)   # standing = reference
    out["walk_cos"] = out["walk"] * out["cosph"]
    out["walk_sin"] = out["walk"] * out["sinph"]
    return out


def fit_condition_model(df: pd.DataFrame, freq: float = FREQ_CPS) -> dict | None:
    """GLMM over both conditions with a condition x phase interaction (descriptive).

        logit P(correct) = b0 + b_walk·walk
                          + b_cos·cos + b_sin·sin                  (standing oscillation, control)
                          + b_wcos·(walk·cos) + b_wsin·(walk·sin)  (additional walking oscillation)
                          + u_sub                                   (participant random intercept)

    Standing is the reference (walk = 0):
      * b_walk: accuracy difference walking - standing;
      * (b_cos, b_sin): standing oscillation, expected ~0 (control);
      * (b_wcos, b_wsin): walking-specific oscillation (interaction);
      * walking total: (b_cos+b_wcos, b_sin+b_wsin).

    Standing phase is a head-motion surrogate (``_standing_trial_phases``). Returns None
    unless both conditions are present.
    """
    if df.empty or df["correct"].nunique() < 2:
        return None
    if not {"walking", "standing"} <= set(df["condition"].unique()):
        return None
    d = _design_condition(df, freq)
    n_sub = d["sub"].nunique()
    base = float(d["correct"].mean())
    keys = ["Intercept", "walk", "cosph", "sinph", "walk_cos", "walk_sin"]
    method, b, sd, note = None, {}, {}, ""
    if n_sub >= 2:
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            model = BinomialBayesMixedGLM.from_formula(
                "correct ~ walk + cosph + sinph + walk_cos + walk_sin",
                {"sub": "0 + C(sub)"}, d)
            res = model.fit_vb()
            names = list(res.model.exog_names)
            fe, fesd = np.asarray(res.fe_mean), np.asarray(res.fe_sd)
            idx = {nm: i for i, nm in enumerate(names)}
            for k in keys:
                b[k], sd[k] = float(fe[idx[k]]), float(fesd[idx[k]])
            method = "BinomialBayesMixedGLM (random participant intercept)"
        except Exception as e:                          # pragma: no cover
            note = f"mixed-GLM fit failed ({type(e).__name__}: {e}); fixed-effects fallback. "
            method = None
    if method is None:
        import statsmodels.api as sm
        X = sm.add_constant(d[["walk", "cosph", "sinph", "walk_cos", "walk_sin"]].to_numpy())
        res = sm.Logit(d["correct"].to_numpy(), X).fit(disp=0)
        for i, k in enumerate(keys):
            b[k], sd[k] = float(res.params[i]), float(res.bse[i])
        method = note + ("fixed-effects logistic regression (n=1 participant: random "
                         "effect not identifiable)" if n_sub < 2
                         else "fixed-effects logistic regression")

    z_w = _standardised(b["walk"], sd["walk"])                   # walking main effect
    amp_s = float(np.hypot(b["cosph"], b["sinph"]))             # standing oscillation (control)
    amp_i = float(np.hypot(b["walk_cos"], b["walk_sin"]))       # walking-specific (interaction)
    bc_w, bs_w = b["cosph"] + b["walk_cos"], b["sinph"] + b["walk_sin"]   # walking total
    amp_w = float(np.hypot(bc_w, bs_w))
    pref_w = float(np.arctan2(-bs_w, bc_w))
    return {
        "freq_cps": freq, "n_participants": int(n_sub), "n_trials": int(len(d)),
        "n_walking": int((d["condition"] == "walking").sum()),
        "n_standing": int((d["condition"] == "standing").sum()),
        "method": method, "base_rate": round(base, 4),
        "locomotion_effect": {
            "b_logodds": round(b["walk"], 4), "se": round(sd["walk"], 4),
            "odds_ratio": round(float(np.exp(b["walk"])), 4),
            "approx_acc_delta": round(float(b["walk"] * base * (1 - base)), 4),
            "z_descriptive": round(z_w, 3)},
        "standing_oscillation_control": {"amp_logit": round(amp_s, 4)},
        "walking_specific_oscillation": {"amp_logit": round(amp_i, 4)},
        "inference": "descriptive only - the locomotion-specificity TEST is "
                     "common_phase.interaction_test (paired Hotelling T2)",
        "walking_total_oscillation": {
            "amp_logit": round(amp_w, 4), "preferred_phase_rad": round(pref_w, 4)},
        "_coef": {k: {"b": round(b[k], 4), "sd": round(sd[k], 4)} for k in keys},
    }


# ===========================================================================
# Figure
# ===========================================================================
def _figure(df: pd.DataFrame, fit: dict, out: Path):
    w = 2 * np.pi * fit["freq_cps"] / 100.0
    xf = np.linspace(0, 100, 200)
    logit = fit["b_intercept"] + fit["b_cos"] * np.cos(w * xf) + fit["b_sin"] * np.sin(w * xf)
    p = 1.0 / (1.0 + np.exp(-logit))
    # binned observed hit rate
    edges = np.linspace(0, 100, 21); cen = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(df["phase"], edges) - 1, 0, 19)
    obs = np.array([df["correct"].to_numpy()[idx == b].mean() if np.any(idx == b) else np.nan
                    for b in range(20)])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.scatter(cen, obs, s=20, color=S.GREY, label="observed (20 bins)")
    ax.plot(xf, p, color=S.PURPLE, lw=2.5,
            label=f"GLMM fit @ {fit['freq_cps']:.0f} cps "
                  f"(amp {fit['amp_hitrate']:.3f} hit-rate)")
    S.reference_line(ax, y=fit["base_rate"], ls=":", label="base rate")
    ax.set_xlabel("target onset (% stride)"); ax.set_ylabel("hit rate")
    ax.set_title(f"Hierarchical GLMM population oscillation — "
                 f"n = {fit['n_participants']}, {fit['n_trials']} trials", color=S.INK)
    ax.legend(loc="upper right")
    S.caption(fig, fit["method"])
    fig.tight_layout(); S.save(fig, out)


def _binned_hitrate(df: pd.DataFrame, nbins: int = 20):
    edges = np.linspace(0, 100, nbins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(df["phase"], edges) - 1, 0, nbins - 1)
    cor = df["correct"].to_numpy()
    obs = np.array([cor[idx == bn].mean() if np.any(idx == bn) else np.nan
                    for bn in range(nbins)])
    return cen, obs


def _figure_combined(df: pd.DataFrame, fit: dict, fc: dict, out: Path):
    """Plot the walking oscillation next to the standing surrogate-phase control, with the
    walking main effect and walking-specific amplitude in the title."""
    w = 2 * np.pi * FREQ_CPS / 100.0
    xf = np.linspace(0, 100, 200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    # left: walking stride-phase oscillation
    dw = df[df["condition"] == "walking"]
    cen, obs = _binned_hitrate(dw)
    logit_w = fit["b_intercept"] + fit["b_cos"] * np.cos(w * xf) + fit["b_sin"] * np.sin(w * xf)
    ax[0].scatter(cen, obs, s=20, color=S.GREY, label="observed (20 bins)")
    ax[0].plot(xf, 1 / (1 + np.exp(-logit_w)), color=S.TEAL, lw=2.5,
               label=f"fit @ 2 cps (amp {fit['amp_hitrate']:.3f})")
    S.reference_line(ax[0], y=fit["base_rate"], ls=":", label="base rate")
    ax[0].set_xlabel("stride phase (% stride)"); ax[0].set_ylabel("hit rate")
    S.panel(ax[0], "a", f"Walking — stride phase (n={fit['n_trials']})", fontsize=10)
    ax[0].legend(loc="upper right")

    # right: standing surrogate-phase control
    ds = df[df["condition"] == "standing"]
    c = fc["_coef"]
    cen2, obs2 = _binned_hitrate(ds)
    logit_s = c["Intercept"]["b"] + c["cosph"]["b"] * np.cos(w * xf) + c["sinph"]["b"] * np.sin(w * xf)
    ctrl_amp = fc["standing_oscillation_control"]["amp_logit"]
    ax[1].scatter(cen2, obs2, s=20, color=S.GREY, label="observed (20 bins)")
    ax[1].plot(xf, 1 / (1 + np.exp(-logit_s)), color=S.ORANGE, lw=2.5,
               label=f"control fit (amp {ctrl_amp:.3f} logit)")
    ax[1].set_xlabel("surrogate head-phase (% cycle)")
    S.panel(ax[1], "b", f"Standing — postural control (n={fc['n_standing']})", fontsize=10)
    ax[1].legend(loc="upper right")

    le, wi = fc["locomotion_effect"], fc["walking_specific_oscillation"]
    fig.suptitle(f"Locomotion × stride-phase GLMM (descriptive)  —  locomotion effect "
                 f"OR={le['odds_ratio']:.2f},  walking-specific amplitude "
                 f"{wi['amp_logit']:.3f} logit",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, fc["method"] + "  |  Estimates only: the locomotion-specificity TEST is "
                   "common_phase.interaction_test (paired Hotelling T2).")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); S.save(fig, out)


_GLMM_OUTPUTS = ("population_glmm.json", "population_glmm.png",
                 "population_glmm_difficulty.png")


def main():
    """Fit the population GLMMs (oscillation, difficulty, condition interaction) and plot them."""
    df = _gather_trials(conditions=("walking", "standing"))
    dw = df[df["condition"] == "walking"]
    if dw.empty:
        bids_io.gate_report("glmm", condition=("walking", "standing"), n_analysed=0)
        bids_io.clear_stale("population", *_GLMM_OUTPUTS)
        print("No walking trials found. Run the gait/phase stages first.")
        return
    fit = fit_population(dw, FREQ_CPS)              # 2 cps
    bids_io.gate_report("glmm", condition=("walking", "standing"),
                        n_analysed=int(df["sub"].nunique()))
    if fit is None:
        bids_io.clear_stale("population", *_GLMM_OUTPUTS)
        print("Not enough variation to fit the population GLMM.")
        return
    # 4 cps is also estimated (Davidson's oscillators peak at 2 or 4 cps). The 2/4 cps
    # tests with Holm correction are in group_agnostic and common_phase.
    fit4 = fit_population(dw, 4.0)
    fc = fit_condition_model(df, FREQ_CPS)          # None without standing data
    fd = fit_difficulty_model(df, FREQ_CPS)         # amplitude by tilt level
    out_dir = bids_io.deriv_dir("population")
    payload = dict(fit)
    payload["inference"] = ("DESCRIPTIVE ONLY. This model reports amplitudes and preferred "
                            "phases. Confirmatory inference is group_agnostic.py; the "
                            "common-phase test is common_phase.py (Hotelling T2).")
    if fit4 is not None:
        payload["fit_4cps"] = {k: fit4[k] for k in
                               ("freq_cps", "amp_logit", "amp_hitrate", "preferred_phase_rad",
                                "n_trials", "method")}
    if fc is not None:
        payload["condition_model"] = fc
    if fd is not None:
        payload["difficulty_model"] = fd
    (out_dir / "population_glmm.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    figs = bids_io.fig_dir("population")
    if fc is not None:
        _figure_combined(df, fit, fc, figs / "population_glmm.png")
    else:
        _figure(dw, fit, figs / "population_glmm.png")
    if fd is not None:
        _figure_difficulty(df, fd, figs / "population_glmm_difficulty.png")

    print("=== Hierarchical GLMM population oscillation (accuracy @ 2 cps) - DESCRIPTIVE ===")
    print("  No p-values: the variational Wald this model used to report had a 60% "
          "false-positive\n  rate under idiosyncratic phases. Inference lives in "
          "group_agnostic.py (confirmatory)\n  and common_phase.py (common-phase Hotelling T2).")
    print(f"  method      : {fit['method']}")
    print(f"  data        : {fit['n_participants']} participant(s), {fit['n_trials']} walking trials, "
          f"base rate {fit['base_rate']}")
    print(f"  oscillation : amp {fit['amp_logit']} logit (~{fit['amp_hitrate']} hit-rate), "
          f"preferred phase {fit['preferred_phase_rad']:.2f} rad")
    if fit4 is not None:
        print(f"  4 cps       : amp {fit4['amp_logit']} logit "
              f"(~{fit4['amp_hitrate']} hit-rate), "
              f"preferred phase {fit4['preferred_phase_rad']:.2f} rad")
    if fit["n_participants"] < 2:
        print("  NOTE: single participant -> fixed-effects fallback; random effects engage "
              "automatically with >=2 participants.")
    if fc is not None:
        le = fc["locomotion_effect"]
        sc = fc["standing_oscillation_control"]
        wi = fc["walking_specific_oscillation"]
        print("\n=== Locomotion × stride-phase (combined walking + standing) ===")
        print(f"  data        : {fc['n_walking']} walking + {fc['n_standing']} standing trials")
        print(f"  locomotion  : OR={le['odds_ratio']} (~{le['approx_acc_delta']:+.3f} hit-rate), "
              f"b/se={le['z_descriptive']}")
        print(f"  standing ctl: amp {sc['amp_logit']} logit  (surrogate phase; expected ~0)")
        print(f"  walk-specific: amp {wi['amp_logit']} logit")
        print("  NOTE: standing 'phase' is a head-motion surrogate (no stride), so its "
              "amplitude is a negative\n        control expected ~0. Estimates only - the "
              "locomotion-specificity TEST is\n        common_phase.interaction_test "
              "(paired Hotelling T2 on walking minus standing).")
    if fd is not None:
        print("\n=== Amplitude by difficulty level (DESCRIPTIVE - H3 retired) ===")
        for d in DIFF_ORDER:
            m = fd["by_difficulty"].get(d)
            if m:
                print(f"  {d:4s}: amp {m['amp_hitrate']:.3f} hit-rate, {m['n_trials']} trials")
        print(f"  amplitude order (high->low): {' > '.join(fd['amp_order_high_to_low'])}")
        print("  NOT a test. The 0.85/0.10/0.05 difficulty mix leaves too few EASY/HARD "
              "trials to fit a\n  2-cps oscillation, so the 'largest at threshold' "
              "hypothesis is not estimable under this\n  design and has been retired. "
              "EASY/HARD also compress accuracy toward ceiling/floor.")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
