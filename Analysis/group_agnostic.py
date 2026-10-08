"""Phase-agnostic population test, the confirmatory analysis.

Why this test
-------------
The pipeline has three population tests of stride-cycle oscillation. At the effect size
measured in Davidson et al. (2024)'s raw data (``davidson_power/effect.py``) their power
differs:

============================================  ===========================  ==========
test                                          what it tests                power*
============================================  ===========================  ==========
``common_phase.py`` Hotelling T^2             a common-phase population     15 %
                                              mean oscillation
``population.py`` Bayesian prevalence         how many individuals clear    ~15 %
                                              a per-participant test
this module                                   do individuals oscillate      91 %
                                              at all, each at its own phase
============================================  ===========================  ==========

\\* at the planned N = 12 with 1440 walking trials each (``davidson_power/power.py``).

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
phase. It corresponds to testing the random-slope variance on (cos, sin) in ``glmm.py``,
and its power is computed in ``power.population_power``.

Dividing by ``s_i`` makes the test efficient. Coefficients share a scale but not a
precision: a participant with fewer trials or accuracy further from ceiling has a wider
null and would dominate ``sum |z_i|^2`` through noise. After studentisation each
participant contributes 1 under H0 (so ``S ~ n_participants`` means no effect), as in
``stance_foot.population_test``. ``agnostic_test`` explains why the test stays exact.

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

Usage::

    python -m Analysis.group_agnostic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import bids_io
    from . import config as C
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import phase_analysis as pa
    import plotstyle as S

S.apply()

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
    ``stance_foot.population_test``.
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
                       label=f"observed (p={r['p_holm']:.3f}"
                             f"{'*' if r['significant'] else ''})")
            S.reference_line(ax, x=r["null_p95"], label="null 95th pct")
            ax.set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$")
            if ci == 0:
                ax.set_ylabel(f"permutations\n({f:.0f} cps)")
            ax.legend(fontsize=8)
            role = "PRIMARY" if res.get("is_primary") else "secondary"
            S.panel(ax, tags[ri * len(meas) + ci],
                    f"{S.MEASURE_LABEL.get(m, m)} @ {f:.0f} cps  [{role}]", fontsize=10)
    n = results[meas[0]]["n_participants"]
    S.caption(fig, f"n = {n} participants. Each coefficient is studentised by the scale of "
                   "that participant's own null, so nobody dominates the sum by being noisy; "
                   "under H0 each contributes 1, so S ~ n. The statistic is rotation-invariant "
                   "per participant: it detects oscillation at ANY preferred phase. Null = the "
                   "participants' own trial-level permutations, summed at a common shuffle index.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    S.save(fig, out)


# ---------------------------------------------------------------------------
def main() -> None:
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


if __name__ == "__main__":
    main()
