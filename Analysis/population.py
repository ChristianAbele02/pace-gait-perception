r"""Population-level stride-phase tests on the per-participant Fourier coefficients.

* Phase-agnostic test (confirmatory): the studentised sum of squared coefficients at 2 and
  4 cps, Holm-corrected, on accuracy (primary), RT and the timeout rate.
* Common-phase test (secondary): one-sample Hotelling T^2 on the (cos, sin) coefficients,
  and the paired walking - standing test of locomotion specificity.
* Prevalence and preferred phase (Fig 4/5) and the stride-duration controls (Supp 7/8).

Usage::

    python -m Analysis.population


Phase-agnostic population test (confirmatory)
=============================================

Phase-agnostic population test, the confirmatory analysis.

Why this test
-------------
The pipeline has three population tests of stride-cycle oscillation. At the effect size
measured in Davidson et al. (2024)'s raw data (``davidson_power/effect.py``) their power
differs:

============================================  ===========================  ==========
test                                          what it tests                power*
============================================  ===========================  ==========
Hotelling T^2 (common-phase test, below)      a common-phase population     15 %
                                              mean oscillation
Bayesian prevalence (below)                   how many individuals clear    ~15 %
                                              a per-participant test
this test                                     do individuals oscillate      91 %
                                              at all, each at its own phase
============================================  ===========================  ==========

\* at the planned N = 12 with 1440 walking trials each (``davidson_power/power.py``).

Davidson's cross-participant phase coherence is only R = 0.33, so preferred phases are
individual. A common-phase test asks whether the vector mean of the coefficients is
non-zero, and averaging vectors with different directions cancels most of the signal.
Prevalence avoids this but depends on the per-participant test's sensitivity, ~0.15 at
this effect size, which is too low to estimate prevalence.

The statistic
-------------
For participant *i*, ``z_i = a_i + i·b_i`` is the Fourier coefficient (cos, sin) of the
binned stride-phase curve at frequency *f*, and ``s_i`` the scale of that participant's
permutation null. The statistic is::

    S(f) = sum_i |z_i|^2 / s_i^2

It is rotation-invariant in each ``z_i``: it measures oscillation strength regardless of
phase. It corresponds to testing the random-slope variance on (cos, sin) in a mixed model,
and its power is computed in ``power.population_power``.

Dividing by ``s_i`` makes the test efficient. Coefficients share a scale but not a
precision: a participant with fewer trials or accuracy further from ceiling has a wider
null and would dominate ``sum |z_i|^2`` through noise. After studentisation each
participant contributes 1 under H0 (so ``S ~ n_participants`` means no effect), as in
``response_bias.population_test``. ``agnostic_test`` explains why the test stays exact.

The null is exact. Each participant's trial-level permutations (the 1000 shuffles behind
their p-values in ``phase_analysis.analyse_measure``) give permuted coefficients
``z_i^(k)``. Participants are independent under the global null, so summing at a common
shuffle index *k* gives a null draw ``S^(k)``. No assumption about the distribution of
individual effects is needed (``power.robustness`` shows power is insensitive to it).

2 and 4 cycles per stride are tested (Davidson's responders peak at one or the other),
Holm-corrected over the pair as in the per-participant test.

One primary measure. Holm corrects over the two frequencies within a measure, not across
measures, so each additional confirmatory measure would inflate the family-wise error.
``config.PRIMARY_MEASURE`` (accuracy) is the pre-specified outcome, as in the power analysis
and Davidson's main result. Reaction time uses the same code and is reported as secondary
without family-wise control.

Inputs are the pooled walking results per participant from
``phase_analysis.subject_pooled_analysis`` (all walking runs concatenated), so each
participant counts once with all trials.


Common-phase population test (secondary)
========================================

Common-phase population test: Hotelling T^2 on the per-participant coefficients.

This test replaces the fixed-effect Wald test of a variational GLMM, which is not valid
inference; the trial-level model with valid uncertainty is ``bayes_glmm``.

Why not the Wald test
---------------------
The GLMM was a hierarchical logistic model fitted by variational inference
(``BinomialBayesMixedGLM``). Mean-field posteriors understate variance, and here the
fixed-effect SEs cannot absorb the random-slope variance on the same (cos, sin)
regressors. False-positive rates under the null (simulated):

=================================  ====================  ==========================
test                               null, no slope var    null, idiosyncratic phases
=================================  ====================  ==========================
GLMM variational Wald              10 %                  60 %
Hotelling T^2 (this module)        4 %                   9 %
=================================  ====================  ==========================

The Hotelling 9 % with scattered phases reflects asking a common-phase question of a
population whose phases differ.

Further advantages:

* exact in small samples (F reference, not asymptotic chi-square), relevant at N = 12-20;
* it is the statistic of the "coherent" arm in ``davidson_power/power.py``, which gives
  the 15 % power figure, so pipeline and power analysis match.

The statistic
-------------
For participant *i*, ``z_i = (a_i, b_i)`` is the fitted (cos, sin) pair at the tested
frequency, loaded with ``population.collect`` so both population tests use the same
coefficients::

    T^2 = N * zbar' S^-1 zbar ,      F = (N - 2) / (2 (N - 1)) * T^2  ~  F(2, N - 2)

Coefficients are not standardised, unlike in ``population``. Here ``S`` is the
between-participant covariance and already includes heterogeneity; a noisier participant
widens ``S``, making the test conservative. This also keeps the statistic identical to the
power simulation.

What it is and is not
---------------------
This is the secondary population analysis: does the population oscillate at a common
phase (non-zero vector mean)? The confirmatory test is ``population``. At Davidson's
phase coherence of R = 0.33 the vector mean cancels most of the signal (91 % vs 15 % power
at the planned design). A null result here with a positive one there is expected if
preferred phases are individual.

Locomotion specificity
----------------------
``interaction_test`` tests walking minus standing coefficients per participant against
the origin. Standing has no gait cycle; its phase is the Hilbert surrogate from
``_standing_trial_phases`` (same head signal and filters), a control axis. The paired
difference removes each participant's offset, so this tests whether an oscillation is
specific to walking rather than to the task.


Prevalence, preferred phase and stride-duration controls
========================================================

Population-level stride-phase results, after Davidson et al. (2024) Fig 4 and 5.

Aggregates the per-participant fits from ``phase_analysis.py`` over all walking
participants:

  * Fig 4 (a-b): each participant's R² spectrum per measure (accuracy, reaction time),
    coloured by the confirmatory verdict (significant at 2 cps, at 4 cps, or not).
  * Fig 4 (c-d): Bayesian population prevalence (Ince, Kay & Schyns 2022) of an
    oscillation at 2 cps, 4 cps and any frequency, MAP with 96 % HPDI. Positives come
    from the primary confirmatory test (2 & 4 cps, Holm; ``phase_analysis``), not the
    grid scan, and the sensitivity (``power.py``) refers to the same test, so α and
    sensitivity in the model are consistent.
  * Fig 5: every participant's fitted 2-cps cosine, the significant ones highlighted, and
    where in the step it peaks (polar: angle = peak position, radius = amplitude), with a
    Rayleigh test over all participants.
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
    from . import config as C
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import clean
    import config as C
    import gait as gait_mod
    import phase_analysis as pa
    import plotstyle as S

S.apply()


def _power():
    """The ``power`` module, imported on first use: it imports this module itself (through
    ``threshold_phase``), so a module-level import would be circular."""
    try:
        from . import power
    except ImportError:
        import power
    return power


# ###########################################################################
# Phase-agnostic population test (confirmatory)
# ###########################################################################
# Reporting order; the first measure is the primary one (``config.PRIMARY_MEASURE``), the
# others are secondary. Holm corrects over frequencies within a measure only; treating
# three measures as confirmatory would give a family-wise rate near 14 %, not 5 %.
PRIMARY = getattr(C, "PRIMARY_MEASURE", "accuracy")
MEASURES = (PRIMARY,) + tuple(
    m for m in getattr(C, "SECONDARY_MEASURES", ("reaction_time",))
    if m != PRIMARY)
FREQS = (2.0, 4.0)
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _subject_npz() -> list[tuple[str, Path]]:
    """Pooled walking ``*_phase.npz`` per subject (named by ``subject_walk_label``).

    The per-run files in the same folder are skipped so each participant counts once.
    """
    out = []
    for sub, ses in sorted({(r.sub, r.ses) for r in bids_io.find_recordings()
                            if r.condition == "walking"}):
        f = bids_io.deriv_dir("phase", sub, ses) / f"{pa.subject_walk_label(sub, ses)}_phase.npz"
        if f.exists():
            out.append((sub, f))
    return out


def collect(measure: str, freq: float):
    """``(observed z per participant, permuted z per participant, subject labels)``.

    Shapes ``(n_sub, 2)`` and ``(n_sub, n_perm, 2)``. Permutations are truncated to the
    smallest count so shuffle indices align.
    """
    obs, null, subs = [], [], []
    for sub, f in _subject_npz():
        with np.load(f) as z:
            ko, kn = f"{measure}_coef_{freq:.0f}cps", f"{measure}_null_coef_{freq:.0f}cps"
            if ko not in z or kn not in z:
                continue
            obs.append(np.asarray(z[ko], float))
            null.append(np.asarray(z[kn], float))
            subs.append(sub)
    if not obs:
        return None, None, []
    nperm = min(a.shape[0] for a in null)
    return np.array(obs), np.stack([a[:nperm] for a in null]), subs


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def agnostic_test(obs: np.ndarray, null: np.ndarray) -> dict:
    """Sum of squared, studentised per-participant coefficients vs its permutation null.

    Each coefficient is divided by the scale of the participant's own null,
    ``s_i = sqrt(mean_k |z_i^(k)|^2)``::

        S = sum_i |z_i|^2 / s_i^2

    Without this, the participant with the widest null (fewest trials, or accuracy
    furthest from ceiling; p(1-p) is 1.6x larger at 70 % than at 85 %) dominates the
    variance of the sum and power is lost. The raw sum is still valid but inefficient, and
    not interpretable as an effect size.

    After studentisation each term has null expectation 1, so ``S`` near
    ``n_participants`` means no effect. ``s_i`` is applied identically to the observed
    statistic and every null draw, so the test stays exact. Same convention as
    ``response_bias.population_test``.
    """
    # Per-participant null scale: RMS coefficient magnitude over the shuffles.
    s = np.sqrt((null ** 2).sum(axis=2).mean(axis=1))          # (n_sub,)
    good = np.isfinite(s) & (s > 0)
    if not good.any():                                          # no variance
        s = np.ones_like(s)
        good = np.ones_like(s, dtype=bool)
    obs_s, null_s, sc = obs[good], null[good], s[good][:, None]
    zo = obs_s / sc                                             # (n_sub, 2)
    zn = null_s / sc[:, None]                                   # (n_sub, n_perm, 2)

    stat = float((zo ** 2).sum())
    null_stat = (zn ** 2).sum(axis=(0, 2))              # (n_perm,), summed over participants
    p = float((1 + np.sum(null_stat >= stat)) / (1 + null_stat.size))
    return {
        "statistic": stat,
        "p_value": p,
        "n_participants": int(zo.shape[0]),
        "n_perm": int(null_stat.size),
        "null_mean": float(null_stat.mean()),
        "null_p95": float(np.percentile(null_stat, 95)),
        # Raw sum, for comparison with davidson_power's raw scale.
        "statistic_unstandardised": float((obs_s ** 2).sum()),
        # RMS per-participant amplitude on the raw coefficients, in hit-rate units as in
        # the power analysis (davidson_effect.json: 0.026 at 2 cps).
        "rms_amplitude": float(np.sqrt((obs_s ** 2).sum(axis=1).mean())),
        "n_excluded_zero_null": int((~good).sum()),
        "_null_stat": null_stat,
    }


def analyse_measure(measure: str) -> dict | None:
    """Both frequencies for one measure, Holm-corrected over the pair."""
    per_freq = {}
    for f in FREQS:
        obs, null, subs = collect(measure, f)
        if obs is None or obs.shape[0] < 2:
            return None
        r = agnostic_test(obs, null)
        r["subjects"] = subs
        per_freq[f] = r
    praw = [per_freq[f]["p_value"] for f in FREQS]
    padj = pa.holm_bonferroni(praw)
    for f, pa_ in zip(FREQS, padj):
        per_freq[f]["p_holm"] = float(pa_)
        per_freq[f]["significant"] = bool(pa_ < ALPHA)
    return {
        "measure": measure,
        "role": "PRIMARY (confirmatory)" if measure == PRIMARY else "secondary (supporting)",
        "is_primary": bool(measure == PRIMARY),
        "by_freq": per_freq,
        "significant_any": bool(any(per_freq[f]["significant"] for f in FREQS)),
        "n_participants": per_freq[FREQS[0]]["n_participants"],
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _figure(results: dict, out: Path) -> None:
    meas = [m for m in MEASURES if m in results]
    if not meas:
        return
    fig, axes = plt.subplots(2, len(meas), figsize=(4.3 * len(meas), 7.0), squeeze=False)
    fig.suptitle("Phase-agnostic population test", fontsize=13, fontweight="semibold",
                 x=0.012, ha="left")
    tags = "abcdef"
    for ci, m in enumerate(meas):
        res = results[m]
        for ri, f in enumerate(FREQS):
            ax = axes[ri][ci]
            r = res["by_freq"][f]
            ns = r["_null_stat"]
            ax.hist(ns, bins=40, color=S.MIST, edgecolor="white")
            ax.axvline(r["statistic"], color=S.MEASURE.get(m, S.INK), lw=2.2,
                       label=f"observed (Holm p={r['p_holm']:.3f}"
                             f"{'*' if r['significant'] else ''})")
            S.reference_line(ax, x=r["null_p95"], label="null 95th pct")
            ax.set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$")
            if ci == 0:
                ax.set_ylabel(f"permutations\n({f:.0f} cps)")
            # Legend on the side away from the observed line, so it never sits on it.
            right = r["statistic"] < r["null_p95"]
            ax.legend(loc="upper right" if right else "upper left", fontsize=8)
            role = "PRIMARY" if res.get("is_primary") else "secondary"
            S.panel(ax, tags[ri * len(meas) + ci],
                    f"{S.MEASURE_LABEL.get(m, m)} @ {f:.0f} cps  [{role}]", fontsize=10)
    n = results[meas[0]]["n_participants"]
    S.caption(fig, f"n = {n} participants. S sums each participant's 2- or 4-cps Fourier "
                   "amplitude, scaled by their own permutation null, so it detects an "
                   "oscillation at any preferred phase; under H0 its mean is n. Grey: S over "
                   "the participants' own shuffles. Holm over 2 and 4 cps within each measure.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    S.save(fig, out)


# ---------------------------------------------------------------------------
def agnostic_main() -> None:
    """Run the phase-agnostic population test; write figure and JSON."""
    out_dir = bids_io.deriv_dir("population")
    results, payload = {}, {}
    for m in MEASURES:
        r = analyse_measure(m)
        if r:
            results[m] = r
            payload[m] = {
                "role": r["role"],
                "is_primary": r["is_primary"],
                "significant_any": r["significant_any"],
                "n_participants": r["n_participants"],
                "by_freq": {f"{f:.0f}cps": {k: v for k, v in r["by_freq"][f].items()
                                            if not k.startswith("_") and k != "subjects"}
                            for f in FREQS},
            }
    bids_io.gate_report("agnostic", condition="walking", n_analysed=(
        results[PRIMARY]["n_participants"] if PRIMARY in results else 0))
    if not results:
        bids_io.clear_stale("population", "population_agnostic.json",
                            "population_agnostic.png")
        print("Phase-agnostic test needs >=2 participants with pooled walking phase "
              "results. Run Analysis.phase_analysis first.")
        return

    print("\n=== Phase-agnostic population test ===")
    print("  H0: no participant oscillates at this frequency (at any phase).")
    print(f"  PRIMARY outcome = {S.MEASURE_LABEL.get(PRIMARY, PRIMARY)}. Holm corrects over "
          "2 vs 4 cps WITHIN a measure; the secondary measures below are supporting "
          "and carry no family-wise claim.")
    for m, r in results.items():
        tag = "PRIMARY" if r["is_primary"] else "secondary"
        print(f"  {S.MEASURE_LABEL.get(m, m):<20} [{tag}]: n={r['n_participants']}")
        for f in FREQS:
            d = r["by_freq"][f]
            print(f"      {f:.0f} cps: S={d['statistic']:.5f} (null mean "
                  f"{d['null_mean']:.5f}, 95th {d['null_p95']:.5f}), "
                  f"RMS amp {d['rms_amplitude']:.4f}, "
                  f"p={d['p_value']:.4f} Holm p={d['p_holm']:.4f} "
                  f"-> {'SIGNIFICANT' if d['significant'] else 'n.s.'}")
        verdict = ("SIGNIFICANT (gait-locked oscillation at 2 or 4 cps)"
                   if r["significant_any"] else "n.s.")
        print(f"      => {verdict}"
              + ("" if r["is_primary"] else "   [secondary - supporting only]"))

    (out_dir / "population_agnostic.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    _figure(results, bids_io.fig_dir("population") / "population_agnostic.png")
    print(f"\nOutputs -> {out_dir}")


# ###########################################################################
# Common-phase population test (secondary)
# ###########################################################################
MIN_N = 4                 # minimum N for the F test
N_BINS = pa.N_BINS


# ===========================================================================
# The statistic
# ===========================================================================
def hotelling_t2(Z: np.ndarray) -> dict:
    """One-sample Hotelling T^2 of the rows of ``Z`` (N x 2) against the origin.

    Returns T^2, its exact F transform and p-value, and the mean vector as amplitude and
    preferred phase.

    ``N < MIN_N`` or a singular covariance gives ``p = 1`` instead of an error.
    """
    Z = np.asarray(Z, float)
    Z = Z[np.isfinite(Z).all(axis=1)]
    n = Z.shape[0]
    out = {"n_participants": int(n), "mean_cos": float("nan"), "mean_sin": float("nan"),
           "t2": float("nan"), "f": float("nan"), "df1": 2, "df2": max(0, n - 2),
           "p_value": 1.0, "amplitude": float("nan"),
           "preferred_phase_rad": float("nan"), "estimable": False}
    if n < MIN_N:
        return out
    mu = Z.mean(0)
    out["mean_cos"], out["mean_sin"] = float(mu[0]), float(mu[1])
    out["amplitude"] = float(np.hypot(*mu))
    out["preferred_phase_rad"] = float(np.arctan2(-mu[1], mu[0]))   # A cos(wx + phi)
    cov = np.cov(Z, rowvar=False)
    try:
        t2 = float(n * mu @ np.linalg.solve(cov, mu))
    except np.linalg.LinAlgError:
        return out
    if not np.isfinite(t2) or t2 < 0:
        return out
    f = (n - 2) / (2 * (n - 1)) * t2
    try:
        from scipy import stats
        p = float(stats.f.sf(f, 2, n - 2))
    except Exception:                                    # pragma: no cover
        return out
    out.update({"t2": t2, "f": float(f), "p_value": p, "estimable": True})
    return out


def paired_hotelling(z_a: np.ndarray, z_b: np.ndarray) -> dict:
    """Hotelling T^2 on the within-participant difference ``z_a - z_b``.

    Pairing removes each participant's offset.
    """
    r = hotelling_t2(np.asarray(z_a, float) - np.asarray(z_b, float))
    r["paired"] = True
    return r


# ===========================================================================
# Trial-level data, walking and standing (for the locomotion interaction)
# ===========================================================================
def _walking_trial_phases(rec) -> pd.DataFrame | None:
    """One row per walking trial: sub, condition, stride phase at onset, correctness.

    Stride phase is the real gait phase (trough-interpolated or Hilbert, per
    config.PHASE_METHOD) from gait.py's saved arrays — exactly as in the
    per-participant phase analysis.
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
    ev = ev[ev["acc"].notna()]                   # accuracy is over answered trials
    return pd.DataFrame({"sub": rec.label().split("_")[0], "condition": "walking",
                         "phase": ev["phase_target"].to_numpy(float),
                         "correct": ev["acc"].to_numpy(float),
                         "difficulty": ev["difficulty"].to_numpy()})


def _standing_trial_phases(rec) -> pd.DataFrame | None:
    """Surrogate-phase rows for STANDING trials — the locomotion negative control.

    Standing has no gait cycle, so there is *no* stride phase. To enter the
    condition × phase interaction we assign each standing trial a continuous Hilbert
    phase computed from the participant's own head-vertical signal (postural sway /
    breathing) with the SAME machinery used for walking (clean → resample → band-pass
    → analytic phase). Because that motion is not a locomotor rhythm, binning
    performance against it carries no 2-cps structure, so the standing oscillation
    amplitude is expected ≈ 0 — which is exactly the control the walking stride-phase
    modulation is tested against. The "phase" axis here is therefore a matched
    surrogate, NOT a stride; it makes the interaction well-posed, nothing more.
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
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna() & ev["response"].notna()].copy()
    if ev.empty:
        return None
    _, pct_stride = gait_mod.phase_at_hilbert(ev["onset"].to_numpy(float), tg, cont)
    return pd.DataFrame({"sub": rec.label().split("_")[0], "condition": "standing",
                         "phase": pct_stride,
                         "correct": np.where(ev["response_correct"] == 1, 1.0, 0.0),
                         "difficulty": ev["difficulty"].to_numpy()})


def _gather_trials(conditions=("walking",)) -> pd.DataFrame:
    """One row per analysis trial across the requested conditions.

    Columns: sub, condition, phase, correct. ``conditions=("walking",)`` (default)
    reproduces the original walking-only oscillation test; pass
    ``("walking", "standing")`` for the combined condition × phase model.
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
# Per-condition coefficients (for the locomotion interaction)
# ===========================================================================
def _condition_coefficients(freq: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-participant (cos, sin) at ``freq`` for walking and for standing, paired.

    ``phase_analysis`` skips standing (no stride), so both conditions are recomputed from
    ``_gather_trials`` with the same cleaning, filter and Hilbert phase. Different
    code paths for the two conditions would bias the paired test.
    """
    df = _gather_trials(("walking", "standing"))
    if df is None or df.empty:
        return np.empty((0, 2)), np.empty((0, 2)), []
    walk, stand, subs = [], [], []
    for sub, g in df.groupby("sub"):
        gw, gs = g[g["condition"] == "walking"], g[g["condition"] == "standing"]
        if len(gw) < 40 or len(gs) < 40:
            continue
        cw = _bin_and_fit(gw["phase"].to_numpy(float), gw["correct"].to_numpy(float), freq)
        cs = _bin_and_fit(gs["phase"].to_numpy(float), gs["correct"].to_numpy(float), freq)
        if cw is None or cs is None:
            continue
        walk.append(cw)
        stand.append(cs)
        subs.append(str(sub))
    return np.array(walk), np.array(stand), subs


def _bin_and_fit(phase: np.ndarray, y: np.ndarray, freq: float):
    """Bin to stride phase and return the (cos, sin) pair at ``freq``.

    Uses ``phase_analysis`` binning and the weighted solver, as in the confirmatory test.
    """
    centers, binned, _, n_bin = pa._bin(phase, y, N_BINS, "mean", with_counts=True)
    ok = np.isfinite(binned)
    if ok.sum() < 6:
        return None
    w = n_bin[ok] if getattr(C, "WEIGHT_BINS_BY_N", True) else None
    _, beta = pa._spectrum_with_coef(centers[ok], binned[ok], w)
    i = int(np.argmin(np.abs(pa.FREQS - freq)))
    return beta[i, 1:]


# ===========================================================================
# Tests
# ===========================================================================
def common_analyse_measure(measure: str) -> dict | None:
    """Common-phase test for one measure at 2 and 4 cps, Holm-corrected over the pair."""
    per_freq, subs = {}, []
    for f in FREQS:
        obs, _null, s = collect(measure, f)
        if obs is None or obs.shape[0] < MIN_N:
            return None
        per_freq[f] = hotelling_t2(obs)
        subs = s
    padj = pa.holm_bonferroni([per_freq[f]["p_value"] for f in FREQS])
    for f, adj in zip(FREQS, padj):
        per_freq[f]["p_holm"] = float(adj)
        per_freq[f]["significant"] = bool(adj < ALPHA)
    return {"measure": measure, "subjects": subs, "by_freq": per_freq,
            "is_primary": False,
            "significant_any": bool(any(per_freq[f]["significant"] for f in FREQS)),
            "n_participants": per_freq[FREQS[0]]["n_participants"]}


def interaction_test() -> dict | None:
    """Locomotion specificity: paired Hotelling on walking minus standing coefficients."""
    out = {}
    for f in FREQS:
        w, s, subs = _condition_coefficients(f)
        if w.shape[0] < MIN_N:
            return None
        out[f] = {"walking": hotelling_t2(w), "standing": hotelling_t2(s),
                  "difference": paired_hotelling(w, s), "subjects": subs}
    for part in ("walking", "standing", "difference"):
        padj = pa.holm_bonferroni([out[f][part]["p_value"] for f in FREQS])
        for f, adj in zip(FREQS, padj):
            out[f][part]["p_holm"] = float(adj)
            out[f][part]["significant"] = bool(adj < ALPHA)
    for f in FREQS:
        out[f]["difference"]["verdict"] = _locus(out[f])
    return out


def _locus(d: dict) -> str:
    """Name the condition that carries a walking - standing difference.

    The paired test only says that the two mean coefficients differ. Standing has no
    stride, so its phase comes from the head-sway surrogate; a difference produced by a
    shared standing "modulation" says nothing about walking. The difference is called
    walking-specific only when walking itself shares a phase and standing does not.
    """
    if not d["difference"]["significant"]:
        return "n.s."
    walk, stand = d["walking"]["significant"], d["standing"]["significant"]
    if walk and not stand:
        return "walking-specific"
    if stand and not walk:
        return "differs, carried by standing"
    if walk and stand:
        return "differs, both conditions share a phase"
    return "differs, neither condition alone"


# ===========================================================================
# Figure
# ===========================================================================
def _ellipse(cov, mu, n, conf=0.95, npts=200):
    """95 % confidence ellipse for the mean of a bivariate sample (F-based)."""
    try:
        from scipy import stats
        crit = (2 * (n - 1) / (n - 2)) * stats.f.ppf(conf, 2, n - 2) / n
    except Exception:                                    # pragma: no cover
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    th = np.linspace(0, 2 * np.pi, npts)
    circ = np.column_stack([np.cos(th), np.sin(th)])
    return mu + circ @ (vecs * np.sqrt(vals * crit)).T


def _common_figure(results: dict, inter: dict | None, out: Path) -> None:
    meas = [m for m in results if results[m]]
    if not meas:
        return
    ncol = len(meas) + (1 if inter else 0)
    fig, axes = plt.subplots(len(FREQS), ncol, figsize=(3.6 * ncol, 7.4), squeeze=False)
    fig.suptitle("Common-phase population test (Hotelling T²) — SECONDARY",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    tags = "abcdefgh"
    k = 0
    for ri, f in enumerate(FREQS):
        for ci, m in enumerate(meas):
            ax = axes[ri][ci]
            obs, _n, _s = collect(m, f)
            r = results[m]["by_freq"][f]
            _scatter_panel(ax, obs, r, S.MEASURE.get(m, S.BLUE))
            S.panel(ax, tags[k], f"{S.MEASURE_LABEL.get(m, m)} @ {f:.0f} cps", fontsize=10)
            k += 1
        if inter:
            ax = axes[ri][len(meas)]
            d = inter[f]
            w, s, _ = _condition_coefficients(f)
            _scatter_panel(ax, w - s, d["difference"], S.MEASURE["accuracy"])
            S.panel(ax, tags[k], f"accuracy, walking − standing @ {f:.0f} cps", fontsize=10)
            k += 1
    S.caption(fig, "Each point is one participant's (cos, sin) Fourier coefficient; X = their "
                   "mean, dashed ellipse = its 95 % confidence region. Hotelling T² asks whether "
                   "the mean differs from the origin, i.e. whether people share a phase. Holm "
                   "over 2 and 4 cps within each panel column. Secondary: with idiosyncratic "
                   "phases the mean cancels, so the confirmatory test is phase-agnostic. "
                   "Standing trials are phased on the head-sway surrogate.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    S.save(fig, out)


def _scatter_panel(ax, Z, r, color):
    Z = np.asarray(Z, float)
    ax.axhline(0, color=S.GRID, lw=1, zorder=0)
    ax.axvline(0, color=S.GRID, lw=1, zorder=0)
    ax.scatter(Z[:, 0], Z[:, 1], s=26, color=S.MIST, edgecolor=color, zorder=2,
               label=f"n = {Z.shape[0]} participants")
    mu = Z.mean(0)
    ax.scatter([mu[0]], [mu[1]], s=90, marker="X", color=color, zorder=4,
               label=f"mean (Holm p={r.get('p_holm', r['p_value']):.3f}"
                     f"{'*' if r.get('significant') else ''})")
    if r["estimable"]:
        e = _ellipse(np.cov(Z, rowvar=False), mu, Z.shape[0])
        if e is not None:
            ax.plot(e[:, 0], e[:, 1], color=color, lw=1.6, ls="--", zorder=3,
                    label="95 % CI for the mean")
    lim = float(np.nanmax(np.abs(Z))) * 1.25 or 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim * 1.45)                    # room for the legend
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("cos coefficient")
    ax.set_ylabel("sin coefficient")
    ax.legend(fontsize=7.5, loc="upper center")


# ===========================================================================
def common_main() -> None:
    """Run the common-phase test and the walking-vs-standing interaction."""
    out_dir = bids_io.deriv_dir("population")
    measures = (getattr(C, "PRIMARY_MEASURE", "accuracy"),) + tuple(
        getattr(C, "SECONDARY_MEASURES", ()))
    results = {m: common_analyse_measure(m) for m in measures}
    results = {m: r for m, r in results.items() if r}
    bids_io.gate_report("common_phase", condition="walking", n_analysed=max(
        (r["n_participants"] for r in results.values()), default=0))
    if not results:
        bids_io.clear_stale("population", "population_common_phase.json",
                            "population_common_phase.png")
        print(f"Common-phase test needs >={MIN_N} participants with pooled walking phase "
              "results. Run Analysis.phase_analysis first.")
        return
    inter = interaction_test()

    print("\n=== Common-phase population test — Hotelling T2 (SECONDARY) ===")
    print("  H0: the population mean (cos, sin) coefficient is the origin.")
    print("  Replaces the GLMM Wald, whose variational SEs gave a 60% false-positive rate.")
    print("  The CONFIRMATORY test is the phase-agnostic one; a null here beside a")
    print("  positive there is the expected pattern when preferred phases scatter.")
    for m, r in results.items():
        print(f"  {S.MEASURE_LABEL.get(m, m):<20}: n={r['n_participants']}")
        for f in FREQS:
            d = r["by_freq"][f]
            if not d["estimable"]:
                print(f"      {f:.0f} cps: not estimable at n={d['n_participants']}")
                continue
            print(f"      {f:.0f} cps: T2={d['t2']:.2f}, F(2,{d['df2']})={d['f']:.2f}, "
                  f"amp={d['amplitude']:.4f}, phase={d['preferred_phase_rad']:+.2f} rad, "
                  f"p={d['p_value']:.4f} Holm p={d['p_holm']:.4f} -> "
                  f"{'SIGNIFICANT' if d['significant'] else 'n.s.'}")

    payload = {"note": "SECONDARY common-phase population test (Hotelling T2). Replaces the "
                       "GLMM fixed-effect Wald, which is invalid. CONFIRMATORY inference is "
                       "population.py.",
               "measures": {m: {"n_participants": r["n_participants"],
                                "significant_any": r["significant_any"],
                                "by_freq": {f"{f:.0f}cps": r["by_freq"][f] for f in FREQS}}
                            for m, r in results.items()}}
    if inter:
        print("\n  Locomotion specificity (paired walking - standing):")
        for f in FREQS:
            d = inter[f]["difference"]
            if not d["estimable"]:
                print(f"      {f:.0f} cps: not estimable at n={d['n_participants']}")
                continue
            w, s = inter[f]["walking"], inter[f]["standing"]
            print(f"      {f:.0f} cps: n={d['n_participants']} paired, "
                  f"F(2,{d['df2']})={d['f']:.2f}, Holm p={d['p_holm']:.4f} -> "
                  f"{d['verdict']}  (walking amp {w['amplitude']:.4f}, Holm p="
                  f"{w['p_holm']:.4f}; standing amp {s['amplitude']:.4f}, Holm p="
                  f"{s['p_holm']:.4f})")
        payload["locomotion_interaction"] = {
            f"{f:.0f}cps": {k: inter[f][k] for k in ("walking", "standing", "difference")}
            for f in FREQS}

    (out_dir / "population_common_phase.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _common_figure(results, inter, bids_io.fig_dir("population") / "population_common_phase.png")
    print(f"\nOutputs -> {out_dir}")


# ###########################################################################
# Prevalence, preferred phase and stride-duration controls
# ###########################################################################
# response_counts removed (see config.SECONDARY_MEASURES).
PREVALENCE_MEASURES = ["accuracy", "reaction_time"]
MEASURE_COL = S.MEASURE
BAND_2 = (1.5, 2.5)
BAND_4 = (3.5, 4.5)
# Prevalence is identifiable only if sensitivity is clearly above the false-positive rate.
# Below this, theta(gamma) is nearly flat and the de-biased MAP drifts toward 1.0, so no
# de-biased estimate is reported.
IDENTIFIABLE_MIN_SENS = 0.10


# ===========================================================================
# Bayesian population prevalence (Ince, Kay & Schyns 2022)
# ===========================================================================
def bayesian_prevalence(k: int, n: int, alpha: float = ALPHA, sensitivity: float = 1.0):
    """Posterior over population prevalence γ given k of n significant participants.

    P(positive) = θ(γ) = γ·sensitivity + (1−γ)·alpha; k ~ Binomial(n, θ); uniform prior.
    Returns (grid, posterior, MAP, [lo, hi] 96 % HPDI).
    """
    g = np.linspace(0, 1, 2001)
    if n == 0:
        post = np.ones_like(g) / len(g)
        return g, post, float("nan"), (float("nan"), float("nan"))
    theta = g * sensitivity + (1 - g) * alpha
    from scipy.stats import binom
    post = binom.pmf(k, n, theta)
    # numpy 2 renamed np.trapz to np.trapezoid; a getattr default would evaluate np.trapz.
    trapz = getattr(np, "trapezoid", None) or np.trapz
    post /= trapz(post, g)
    m = g[int(np.argmax(post))]
    # 96 % HPDI: highest-density points up to 0.96 mass
    order = np.argsort(post)[::-1]
    cum = np.cumsum(post[order]) * (g[1] - g[0])
    inc = np.sort(order[cum <= 0.96])
    lo, hi = (g[inc[0]], g[inc[-1]]) if inc.size else (m, m)
    return g, post, float(m), (float(lo), float(hi))


# ===========================================================================
# Gather per-participant fits
# ===========================================================================
def _stride_duration(rec) -> float:
    """Mean stride duration (s) from the gait summary, NaN if missing (Supp Fig 7/8)."""
    js = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.json"
    if not js.exists():
        return float("nan")
    val = json.loads(js.read_text()).get("stride_duration_s")
    return float(val) if val is not None else float("nan")


def _gather():
    """One row per (participant, measure) from the pooled walking phase file.

    ``phase_analysis.subject_pooled_analysis`` concatenates all walking runs of a subject,
    so prevalence counts participants, not ~180-trial blocks."""
    rows = []
    groups: dict[tuple, list] = {}
    for rec in bids_io.find_recordings():
        if rec.condition == "walking":
            groups.setdefault((rec.sub, rec.ses), []).append(rec)
    for (sub, ses), recs in sorted(groups.items()):
        label = pa.subject_walk_label(sub, ses)
        d = bids_io.deriv_dir("phase", sub, ses)
        js, nz = d / f"{label}_phase.json", d / f"{label}_phase.npz"
        if not (js.exists() and nz.exists()):
            continue
        rep = json.loads(js.read_text())
        npz = np.load(nz)
        n_trials = int(rep.get("n_trials", 0))
        # mean stride duration over the participant's walking runs (Supp 7/8)
        sds = [_stride_duration(r) for r in recs]
        stride_dur = float(np.nanmean(sds)) if np.any(np.isfinite(sds)) else float("nan")
        for meas in PREVALENCE_MEASURES:
            if meas not in rep["measures"]:
                continue
            m = rep["measures"][meas]
            rows.append({
                "sub": sub, "measure": meas, "n_trials": n_trials,
                # primary confirmatory test (2 & 4 cps, Holm), used for prevalence:
                "significant": m["significant"],
                "group_freq": m.get("conf_best_freq_cps", m["best_freq_cps"]),
                # exploratory grid scan, for the spectrum panels and Supp 7/8:
                "best_freq": m["best_freq_cps"], "best_r2": m["best_r2"],
                "significant_grid": m.get("significant_grid", m["significant"]),
                "ci95": m["ci95"],
                "r2_at_2cps": m["r2_at_2cps"], "amp_at_2cps": m.get("amp_at_2cps"),
                "phase_2cps": m["preferred_phase_2cps_rad"],
                "stride_dur": stride_dur,
                "freqs": npz.get(f"{meas}_freqs"), "r2": npz.get(f"{meas}_r2"),
                "centers": npz.get(f"{meas}_centers"), "binned": npz.get(f"{meas}_binned"),
            })
    return rows


def _band_max_r2(freqs, r2, lo, hi) -> float:
    """Peak R² within [lo, hi] cps (2/4-cps band maxima for Supp Fig 7)."""
    if freqs is None or r2 is None:
        return float("nan")
    m = (np.asarray(freqs) >= lo) & (np.asarray(freqs) <= hi)
    return float(np.max(np.asarray(r2)[m])) if m.any() else float("nan")


def _corr(x, y):
    """Pearson (r, p) over finite pairs, or (nan, nan) at n<3."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    from scipy.stats import pearsonr
    r, p = pearsonr(x[ok], y[ok])
    return float(r), float(p)


def _sensitivity_for(mr) -> float:
    """Mean per-participant detection power for this measure.

    Based on each participant's trial count and power.REF_AMP. This ``s`` de-biases the
    prevalence (b1): with s < 1, the same k/n gives a higher prevalence than the
    sensitivity = 1 lower bound.
    """
    ns = [r.get("n_trials", 0) for r in mr]
    return _power().mean_sensitivity(ns, amp=_power().REF_AMP)


def _group(best_freq, sig):
    if not sig:
        return "none"
    if BAND_2[0] <= best_freq <= BAND_2[1]:
        return "2cps"
    if BAND_4[0] <= best_freq <= BAND_4[1]:
        return "4cps"
    return ">5" if best_freq > 5 else "other"


# ===========================================================================
# Figures
# ===========================================================================
_VERDICT_STYLE = {"2cps": (S.PURPLE, "significant at 2 cps"),
                  "4cps": (S.TEAL, "significant at 4 cps"),
                  "none": (S.MIST, "not significant")}


def _fig4(rows, out: Path):
    """Per measure: every participant's R² spectrum coloured by their confirmatory
    verdict (top), and the de-biased prevalence posteriors that verdict feeds (bottom).
    Spectra and counts use the same verdict, so a coloured line is always one of the k."""
    fig, ax = plt.subplots(2, len(PREVALENCE_MEASURES), figsize=(5.6 * len(PREVALENCE_MEASURES), 8.4), squeeze=False)
    for ci, meas in enumerate(PREVALENCE_MEASURES):
        mr = [r for r in rows if r["measure"] == meas]
        n = len(mr)
        a = ax[0, ci]
        for f in (2.0, 4.0):
            a.axvline(f, color=S.GREY, ls="--", lw=0.8, zorder=0)
        shown = set()
        # n.s. first, so the significant participants are drawn on top.
        for r in sorted(mr, key=lambda r: r["significant"]):
            if r["freqs"] is None:
                continue
            grp = _group(r["group_freq"], r["significant"])
            grp = grp if grp in _VERDICT_STYLE else "none"
            col, lab = _VERDICT_STYLE[grp]
            a.plot(r["freqs"], r["r2"], color=col, lw=1.8 if grp != "none" else 1.0,
                   zorder=3 if grp != "none" else 2,
                   label=lab if grp not in shown else None)
            shown.add(grp)
        a.set_xlim(0, 10); a.set_ylim(bottom=0)
        a.set_xlabel("frequency (cycles per stride)")
        if ci == 0:
            a.set_ylabel("R² of the Fourier fit")
        a.legend(loc="upper right", fontsize=8)
        S.panel(a, "ab"[ci], f"{S.MEASURE_LABEL[meas].capitalize()}: spectra (n = {n})",
                fontsize=10)

        # Bayesian prevalence, DE-BIASED with the measured per-participant sensitivity:
        # s < 1 lifts the estimate off the naive lower bound. The counts use the PRIMARY
        # confirmatory test (group_freq + significant), never the exploratory grid peak.
        s_hat = _sensitivity_for(mr)
        identifiable = s_hat >= IDENTIFIABLE_MIN_SENS
        s_eff = max(s_hat, ALPHA + 1e-3)   # keep s above alpha
        k2 = sum(1 for r in mr if _group(r["group_freq"], r["significant"]) == "2cps")
        k4 = sum(1 for r in mr if _group(r["group_freq"], r["significant"]) == "4cps")
        kany = sum(1 for r in mr if r["significant"])
        a = ax[1, ci]
        # Distinct dashes, so curves that coincide (equal k) all stay visible.
        for k, col, ls, lab in [(k2, S.PURPLE, "-", "2 cps"), (k4, S.TEAL, "--", "4 cps"),
                                (kany, S.INK, ":", "any")]:
            g, post, mapv, (lo, hi) = bayesian_prevalence(k, n, sensitivity=s_eff)
            a.plot(g, post, color=col, lw=2, ls=ls,
                   label=f"{lab}: k = {k}/{n}, MAP {mapv:.2f} [{lo:.2f}, {hi:.2f}]")
        a.set_xlabel("population prevalence"); a.set_xlim(0, 1)
        a.set_ylim(0, a.get_ylim()[1] * 1.35)     # headroom: the legend sits above the curves
        if ci == 0:
            a.set_ylabel("posterior density")
        a.legend(loc="upper right", fontsize=8)
        S.panel(a, "cd"[ci], f"{S.MEASURE_LABEL[meas].capitalize()}: prevalence "
                f"(ŝ = {s_hat:.2f})", fontsize=10)
        if not identifiable:
            a.text(0.5, 0.45, "prevalence unidentifiable\n(per-participant power too low:\n"
                   f"ŝ = {s_hat:.2f}; need more targets/pp)",
                   transform=a.transAxes, ha="center", va="center", fontsize=8,
                   color=S.RED, bbox=dict(boxstyle="round", fc="#FBEAEC", ec=S.RED))
    fig.suptitle("Stride-phase oscillations per participant and their population prevalence "
                 f"(n = {len({r['sub'] for r in rows})} walking participants)",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "a-b: one line per participant (all walking blocks pooled); colour = "
                   "the confirmatory per-participant test (2 and 4 cps, Holm); dashed guides "
                   "at 2 and 4 cps. c-d: posterior of the proportion of the population that "
                   "oscillates (Ince et al. 2022), MAP and 96 % HPDI, de-biased with ŝ, the "
                   "test's power for one participant at the reference amplitude.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); S.save(fig, out)


# Display units for a 2-cps amplitude: accuracy in percentage points, RT (fitted on
# log10 seconds) as a percentage change.
_AMP_UNIT = {"accuracy": (100.0, "pp"), "reaction_time": (100.0 * np.log(10.0), "%")}


def _rayleigh_p(theta) -> tuple[float, float]:
    """Mean resultant length and Rayleigh p (Zar 1999, eq. 27.4) for angles in rad."""
    theta = np.asarray(theta, float)
    n = theta.size
    if n < 3:
        return float("nan"), float("nan")
    R = float(np.abs(np.mean(np.exp(1j * theta))))
    Rn = n * R
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n)))
    return R, min(1.0, p)


def _fig5(rows, out: Path):
    """Every participant's 2-cps oscillation, drawn as the fitted (tested) cosine, and
    where in the step its peak falls. Significant participants are highlighted rather
    than shown alone, so the figure still says something when none of them is."""
    fig = plt.figure(figsize=(11, 4.4 * len(PREVALENCE_MEASURES)))
    gs = fig.add_gridspec(len(PREVALENCE_MEASURES), 2, width_ratios=[1.6, 1], hspace=0.55, wspace=0.25)
    xf = np.linspace(0, 100, 300)
    w = 2 * np.pi * 2.0 / 100.0
    for i, meas in enumerate(PREVALENCE_MEASURES):
        mr = [r for r in rows if r["measure"] == meas
              and r.get("amp_at_2cps") is not None and np.isfinite(r["phase_2cps"])]
        scale, unit = _AMP_UNIT[meas]
        col = MEASURE_COL[meas]
        sig = [_group(r["group_freq"], r["significant"]) == "2cps" for r in mr]
        axL = fig.add_subplot(gs[i, 0])
        for r, s in sorted(zip(mr, sig), key=lambda t: t[1]):
            y = scale * r["amp_at_2cps"] * np.cos(w * xf + r["phase_2cps"])
            axL.plot(xf, y, color=col if s else S.MIST, lw=2.0 if s else 1.1,
                     zorder=3 if s else 2)
        axL.axhline(0, color=S.GREY, lw=0.7, zorder=0)
        axL.axvline(50, color=S.GREY, ls=":", lw=0.9, zorder=0)
        axL.set_xlim(0, 100)
        axL.set_xlabel("target onset (% stride; 50 = second step)")
        axL.set_ylabel(f"{S.MEASURE_LABEL[meas]} change ({unit})")
        S.panel(axL, "ac"[i], f"{S.MEASURE_LABEL[meas].capitalize()}: 2-cps fit per "
                f"participant  ({sum(sig)}/{len(mr)} significant)", fontsize=10)

        # Peak position within the step: A·cos(wx + φ) peaks at x = -φ/w (mod 50 %).
        axP = fig.add_subplot(gs[i, 1], projection="polar")
        axP.set_theta_zero_location("N"); axP.set_theta_direction(-1)
        theta = np.array([(-r["phase_2cps"]) % (2 * np.pi) for r in mr])
        amp = np.array([scale * r["amp_at_2cps"] for r in mr])
        sig_a = np.array(sig, bool)
        axP.scatter(theta[~sig_a], amp[~sig_a], s=34, facecolor="white", edgecolor=col,
                    lw=1.3, zorder=3, label="not significant")
        if sig_a.any():
            axP.scatter(theta[sig_a], amp[sig_a], s=40, color=col, zorder=4,
                        label="significant")
        R, p = _rayleigh_p(theta)
        if np.isfinite(R):
            mu = float(np.angle(np.mean(np.exp(1j * theta))))
            axP.annotate("", xy=(mu, R * amp.max()), xytext=(0, 0), zorder=5,
                         arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.6))
        axP.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        axP.set_xticklabels(["0 %", "12.5 %", "25 %", "37.5 %"], fontsize=8)
        axP.tick_params(axis="y", labelsize=7.5)
        axP.set_rlabel_position(135)
        stat = (f"Rayleigh R = {R:.2f}, p = {p:.2f}" if np.isfinite(R) else "n < 3")
        S.polar_panel(axP, "bd"[i], f"Peak within the step ({unit})\n{stat}", pad=18,
                      fontsize=9.5)
        axP.legend(loc="upper left", bbox_to_anchor=(0.92, 1.12), fontsize=7.5)
    fig.suptitle("2-cps oscillation and its phase, per participant",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Left: the cosine fitted at 2 cycles per stride (once per step), as "
                   "tested; grey = not significant in the confirmatory test. Right: where in "
                   "the step that cosine peaks (angle, 0-50 % of the stride = one step) and "
                   "its amplitude (radius); for RT the peak is the slowest point. Arrow = "
                   "mean direction, length = R (Rayleigh test over all participants).",
              y=0.02)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.9)
    S.save(fig, out)


def _scatter_corr(ax, x, y, color, *, ncomp=1):
    """Scatter with least-squares line; returns Pearson r label (Bonferroni-flagged)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], s=22, color=color, alpha=0.75, zorder=2)
    r, p = _corr(x, y)
    if np.isfinite(r) and ok.sum() >= 2:
        xs = np.linspace(x[ok].min(), x[ok].max(), 2)
        b = np.polyfit(x[ok], y[ok], 1)
        sig = np.isfinite(p) and p < 0.05 / ncomp
        ax.plot(xs, np.polyval(b, xs), color=(S.INK if sig else S.GREY),
                lw=1.6 if sig else 1.1, ls="-" if sig else "--", zorder=1)
        return f"r={r:.2f}, p={p:.3f}" + ("*" if sig else " ns")
    return "n<3"


def _suppl7(rows, out: Path):
    """Supp Fig 7: stride duration vs 2-cps and 4-cps fit strength and best-fit frequency.

    Davidson found no relationship, i.e. the individual frequency is not a stride-duration
    artefact."""
    row_specs = [("peak R², 1.8-2.2 cps", lambda r: _band_max_r2(r["freqs"], r["r2"], 1.8, 2.2)),
                 ("peak R², 3.8-4.2 cps", lambda r: _band_max_r2(r["freqs"], r["r2"], 3.8, 4.2)),
                 ("best-fit frequency (cps)", lambda r: r["best_freq"])]
    fig, ax = plt.subplots(len(row_specs), len(PREVALENCE_MEASURES), figsize=(4.8 * len(PREVALENCE_MEASURES), 10),
                           squeeze=False, sharex=True)
    ncomp = len(row_specs) * len(PREVALENCE_MEASURES)
    tags = "abcdefghi"
    for ri, (ylab, getter) in enumerate(row_specs):
        for ci, meas in enumerate(PREVALENCE_MEASURES):
            a = ax[ri, ci]
            mr = [r for r in rows if r["measure"] == meas]
            x = [r["stride_dur"] for r in mr]
            y = [getter(r) for r in mr]
            lab = _scatter_corr(a, x, y, MEASURE_COL[meas], ncomp=ncomp)
            if ci == 0:
                a.set_ylabel(ylab)
            if ri == len(row_specs) - 1:
                a.set_xlabel("mean stride duration (s)")
            S.panel(a, tags[ri * len(PREVALENCE_MEASURES) + ci],
                    f"{S.MEASURE_LABEL[meas].capitalize()}  ({lab})", fontsize=10)
    fig.suptitle("Stride duration vs oscillation strength and frequency (Davidson Supp. 7), "
                 f"n = {len({r['sub'] for r in rows})} walking participants",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, f"One point per participant. Pearson r; solid line = significant after "
                   f"Bonferroni over the {ncomp} panels, dashed = not. Davidson found no "
                   "relation: an idiosyncratic frequency is not a by-product of stride length.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96)); S.save(fig, out)


def _suppl8(rows, out: Path):
    """Supp Fig 8: best-fit frequency in Hz (cps / stride duration) vs stride duration.

    Davidson predicts a negative correlation within the ~2 Hz and ~4 Hz clusters (shorter
    strides, faster oscillation), masked in the pooled data by Simpson's paradox."""
    fig, ax = plt.subplots(1, len(PREVALENCE_MEASURES), figsize=(5.2 * len(PREVALENCE_MEASURES), 4.6), squeeze=False)
    ax = ax[0]
    clusters = [("~2 Hz", 1.5, 2.5, S.PURPLE), ("~4 Hz", 3.5, 4.5, S.TEAL)]
    for ci, meas in enumerate(PREVALENCE_MEASURES):
        a = ax[ci]
        mr = [r for r in rows if r["measure"] == meas]
        sd = np.array([r["stride_dur"] for r in mr], float)
        hz = np.array([r["best_freq"] / r["stride_dur"]
                       if r["stride_dur"] and np.isfinite(r["stride_dur"]) else np.nan
                       for r in mr], float)
        a.scatter(sd, hz, s=26, color=MEASURE_COL[meas], alpha=0.75, zorder=2)
        r_all, p_all = _corr(sd, hz)
        for name, lo, hi, col in clusters:
            a.axhspan(lo, hi, color=col, alpha=0.08, lw=0, zorder=0)
            m = (hz >= lo) & (hz <= hi)
            if m.sum() >= 3:
                rc, pc = _corr(sd[m], hz[m])
                xs = np.linspace(sd[m].min(), sd[m].max(), 2)
                b = np.polyfit(sd[m][np.isfinite(sd[m])], hz[m][np.isfinite(sd[m])], 1)
                a.plot(xs, np.polyval(b, xs), color=col, lw=2, zorder=3,
                       label=f"{name}: r = {rc:.2f}, p = {pc:.3f}")
        if a.get_legend_handles_labels()[0]:
            a.legend(loc="upper right", fontsize=8)
        a.set_ylim(0, max(10.0, np.nanmax(hz) * 1.1) if np.isfinite(hz).any() else 10.0)
        a.set_xlabel("mean stride duration (s)"); a.set_ylabel("best-fit frequency (Hz)")
        allr = (f"all: r = {r_all:.2f}, p = {p_all:.2f}" if np.isfinite(r_all) else "n < 3")
        S.panel(a, "ab"[ci], f"{S.MEASURE_LABEL[meas].capitalize()}  ({allr})", fontsize=10)
    fig.suptitle("Oscillation frequency in Hz vs stride duration (Davidson Supp. 8)",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "One point per participant: best grid frequency (cycles per stride) "
                   "divided by stride duration. Shaded: the ~2 and ~4 Hz bands; a fit line "
                   "is drawn for a band holding 3 or more participants. Davidson's "
                   "prediction: within a band, shorter strides give a faster oscillation "
                   "(negative slope), i.e. the stride sets the period.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93)); S.save(fig, out)


def prevalence_main():
    """Build the prevalence and preferred-phase figures (Fig 4/5) and Supp Fig 7/8."""
    rows = _gather()
    bids_io.gate_report("population", condition="walking",
                        n_analysed=len({r["sub"] for r in rows}))
    if not rows:
        bids_io.clear_stale("population", "population_fig4_prevalence.png",
                            "population_fig5_phase.png",
                            "population_suppl7_stridedur_vs_fits.png",
                            "population_suppl8_hz_vs_stridedur.png")
        print("No per-participant phase fits found. Run Analysis/phase_analysis.py first.")
        return
    subs = sorted({r["sub"] for r in rows})
    out = bids_io.deriv_dir("population")
    figs = bids_io.fig_dir("population")
    _fig4(rows, figs / "population_fig4_prevalence.png")
    _fig5(rows, figs / "population_fig5_phase.png")
    _suppl7(rows, figs / "population_suppl7_stridedur_vs_fits.png")
    _suppl8(rows, figs / "population_suppl8_hz_vs_stridedur.png")
    print(f"Population analysis over {len(subs)} walking participant(s): {subs}")
    print(f"  (de-biased with measured per-participant sensitivity at amp={_power().REF_AMP}; "
          f"naive = sensitivity 1.0 = lower bound)")
    for meas in PREVALENCE_MEASURES:
        mr = [r for r in rows if r["measure"] == meas]
        k2 = sum(1 for r in mr if _group(r["group_freq"], r["significant"]) == "2cps")
        s_hat = _sensitivity_for(mr)
        med_n = int(np.median([r.get("n_trials", 0) for r in mr])) if mr else 0
        _, _, map_naive, _ = bayesian_prevalence(k2, len(mr))                       # s = 1 lower bound
        if s_hat >= IDENTIFIABLE_MIN_SENS:
            _, _, map_db, (lo, hi) = bayesian_prevalence(k2, len(mr), sensitivity=s_hat)
            verdict = f"prevalence MAP {map_db:.2f} [{lo:.2f},{hi:.2f}] (naive lower bound {map_naive:.2f})"
        else:
            verdict = (f"prevalence UNIDENTIFIABLE (s_hat too low); naive lower bound "
                       f"{map_naive:.2f}; need more targets/pp")
        print(f"  {meas:16s}: {k2}/{len(mr)} sig @ ~2 cps | median {med_n} targets/pp, "
              f"s_hat={s_hat:.2f} | {verdict}")
    if len(subs) < 5:
        print("  (n too small — figures validate the machinery only; pool participants.)")
    print(f"\nFigures -> {out}")


# ###########################################################################
# Pipeline stage
# ###########################################################################
def main() -> None:
    """Pipeline stage ``population``: confirmatory test, common phase, prevalence."""
    agnostic_main()
    common_main()
    prevalence_main()


if __name__ == "__main__":
    main()
