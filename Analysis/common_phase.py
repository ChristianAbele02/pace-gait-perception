"""Common-phase population test: Hotelling T^2 on the per-participant coefficients.

Provides the common-phase p-values that the GLMM fixed-effect Wald test cannot give
validly. ``glmm.py`` still reports amplitude, preferred phase and the difficulty and
condition breakdowns.

Why not the Wald test
---------------------
``glmm.py`` fits a hierarchical logistic model by variational inference
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
frequency, loaded with ``group_agnostic.collect`` so both population tests use the same
coefficients::

    T^2 = N * zbar' S^-1 zbar ,      F = (N - 2) / (2 (N - 1)) * T^2  ~  F(2, N - 2)

Coefficients are not standardised, unlike in ``group_agnostic``. Here ``S`` is the
between-participant covariance and already includes heterogeneity; a noisier participant
widens ``S``, making the test conservative. This also keeps the statistic identical to the
power simulation.

What it is and is not
---------------------
This is the secondary population analysis: does the population oscillate at a common
phase (non-zero vector mean)? The confirmatory test is ``group_agnostic``. At Davidson's
phase coherence of R = 0.33 the vector mean cancels most of the signal (91 % vs 15 % power
at the planned design). A null result here with a positive one there is expected if
preferred phases are individual.

Locomotion specificity
----------------------
``interaction_test`` tests walking minus standing coefficients per participant against
the origin. Standing has no gait cycle; its phase is the Hilbert surrogate from
``glmm._standing_trial_phases`` (same head signal and filters), a control axis. The paired
difference removes each participant's offset, so this tests whether an oscillation is
specific to walking rather than to the task.

Usage::

    python -m Analysis.common_phase
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
    from . import group_agnostic as ga
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import group_agnostic as ga
    import phase_analysis as pa
    import plotstyle as S

S.apply()

FREQS = (2.0, 4.0)
ALPHA = 0.05
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
# Per-condition coefficients (for the locomotion interaction)
# ===========================================================================
def _condition_coefficients(freq: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-participant (cos, sin) at ``freq`` for walking and for standing, paired.

    ``phase_analysis`` skips standing (no stride), so both conditions are recomputed from
    ``glmm._gather_trials`` with the same cleaning, filter and Hilbert phase. Different
    code paths for the two conditions would bias the paired test.
    """
    try:
        from . import glmm as glmm_mod
    except ImportError:                                  # pragma: no cover
        import glmm as glmm_mod
    df = glmm_mod._gather_trials(("walking", "standing"))
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
def analyse_measure(measure: str) -> dict | None:
    """Common-phase test for one measure at 2 and 4 cps, Holm-corrected over the pair."""
    per_freq, subs = {}, []
    for f in FREQS:
        obs, _null, s = ga.collect(measure, f)
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
    padj = pa.holm_bonferroni([out[f]["difference"]["p_value"] for f in FREQS])
    for f, adj in zip(FREQS, padj):
        out[f]["difference"]["p_holm"] = float(adj)
        out[f]["difference"]["significant"] = bool(adj < ALPHA)
    return out


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


def _figure(results: dict, inter: dict | None, out: Path) -> None:
    meas = [m for m in results if results[m]]
    if not meas:
        return
    ncol = len(meas) + (1 if inter else 0)
    fig, axes = plt.subplots(len(FREQS), ncol, figsize=(4.6 * ncol, 7.4), squeeze=False)
    fig.suptitle("Common-phase population test (Hotelling T²) — SECONDARY",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    tags = "abcdefgh"
    k = 0
    for ri, f in enumerate(FREQS):
        for ci, m in enumerate(meas):
            ax = axes[ri][ci]
            obs, _n, _s = ga.collect(m, f)
            r = results[m]["by_freq"][f]
            _scatter_panel(ax, obs, r, S.MEASURE.get(m, S.BLUE))
            S.panel(ax, tags[k], f"{S.MEASURE_LABEL.get(m, m)} @ {f:.0f} cps", fontsize=10)
            k += 1
        if inter:
            ax = axes[ri][len(meas)]
            d = inter[f]
            w, s, _ = _condition_coefficients(f)
            _scatter_panel(ax, w - s, d["difference"], S.RED)
            S.panel(ax, tags[k], f"walking − standing @ {f:.0f} cps", fontsize=10)
            k += 1
    S.caption(fig, "Each point is one participant's (cos, sin) coefficient; the ellipse is the "
                   "95 % confidence region for their MEAN. The test asks whether that mean "
                   "differs from the origin, i.e. whether the population oscillates at a COMMON "
                   "phase. This is secondary: at Davidson's coherence of R = 0.33 the vector "
                   "mean cancels most of the signal, which is why the confirmatory test "
                   "(group_agnostic) is phase-agnostic instead.")
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
               label=f"mean (p={r.get('p_holm', r['p_value']):.3f}"
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
def main() -> None:
    """Run the common-phase test and the walking-vs-standing interaction."""
    out_dir = bids_io.deriv_dir("population")
    measures = (getattr(C, "PRIMARY_MEASURE", "accuracy"),) + tuple(
        getattr(C, "SECONDARY_MEASURES", ()))
    results = {m: analyse_measure(m) for m in measures}
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
    print("  The CONFIRMATORY test is group_agnostic (phase-agnostic); a null here beside a")
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
                       "group_agnostic.py.",
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
            print(f"      {f:.0f} cps: n={d['n_participants']} paired, "
                  f"F(2,{d['df2']})={d['f']:.2f}, Holm p={d['p_holm']:.4f} -> "
                  f"{'WALKING-SPECIFIC' if d['significant'] else 'n.s.'}")
        payload["locomotion_interaction"] = {
            f"{f:.0f}cps": {k: inter[f][k] for k in ("walking", "standing", "difference")}
            for f in FREQS}

    (out_dir / "population_common_phase.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _figure(results, inter, bids_io.fig_dir("population") / "population_common_phase.png")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
