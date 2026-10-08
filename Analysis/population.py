"""Population-level stride-phase results, after Davidson et al. (2024) Fig 4 and 5.

Aggregates the per-participant fits from ``phase_analysis.py`` over all walking
participants:

  * Fig 4 (a-i): each participant's R² spectrum, grouped for display by the exploratory
    grid peak (~2 cps, ~4 cps, >5 cps / none), per measure (accuracy, reaction time).
  * Fig 4 (j-l): Bayesian population prevalence (Ince, Kay & Schyns 2022) of an
    oscillation at 2 cps, 4 cps and any frequency, MAP with 96 % HPDI. Positives come
    from the primary confirmatory test (2 & 4 cps, Holm; ``phase_analysis``), not the
    grid scan, and the sensitivity (``power.py``) refers to the same test, so α and
    sensitivity in the model are consistent.
  * Fig 5 (a/c/e): 2-cps Fourier fits (relative change) of participants significant at
    ~2 cps; (b/d/f): polar histograms of their preferred phases with a Rayleigh test.

With only a few participants the output only checks that the code runs; prevalence and
phase clustering need the full sample.

Usage::

    python -m Analysis.population
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
    from . import power
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import power
    import phase_analysis as pa
    import plotstyle as S

S.apply()

# response_counts removed (see config.SECONDARY_MEASURES).
MEASURES = ["accuracy", "reaction_time"]
MEASURE_COL = S.MEASURE
BAND_2 = (1.5, 2.5)
BAND_4 = (3.5, 4.5)
ALPHA = 0.05   # within-participant false-positive rate (permutation test level)
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
        for meas in MEASURES:
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
    return power.mean_sensitivity(ns, amp=power.REF_AMP)


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
def _fig4(rows, out: Path):
    fig, ax = plt.subplots(4, 3, figsize=(14, 14))
    group_rows = {"2cps": 0, "4cps": 1, ">5": 2, "other": 2, "none": 2}
    group_col = {"2cps": S.PURPLE, "4cps": S.TEAL, ">5": S.RED, "other": S.GREY, "none": S.MIST}
    titles = ["Accuracy", "Reaction time"]
    for ci, meas in enumerate(MEASURES):
        ax[0, ci].set_title(titles[ci], fontsize=11, loc="left")
        mr = [r for r in rows if r["measure"] == meas]
        n_by = {"2cps": 0, "4cps": 0, ">5": 0, "none": 0}
        for r in mr:
            grp = _group(r["best_freq"], r["significant"])
            n_by[grp if grp in n_by else ">5"] = n_by.get(grp if grp in n_by else ">5", 0) + 1
            if r["freqs"] is not None:
                ax[group_rows[grp], ci].plot(r["freqs"], r["r2"], color=group_col[grp],
                                             alpha=0.6, lw=1)
        for gr, lab in [(0, "~2 cps"), (1, "~4 cps"), (2, ">5 cps / none")]:
            ax[gr, ci].set_ylim(0, 0.7); ax[gr, ci].set_xlim(0, 10)
            if ci == 0:
                ax[gr, ci].set_ylabel(f"R²\n({lab})")
            ax[gr, ci].set_xlabel("frequency (cps)")
        # j-l: prevalence de-biased with the estimated sensitivity (b1). Counts use the
        # confirmatory test (group_freq, significant), not the grid peak of the panels above.
        n = len(mr)
        s_hat = _sensitivity_for(mr)
        identifiable = s_hat >= IDENTIFIABLE_MIN_SENS
        s_eff = max(s_hat, ALPHA + 1e-3)   # keep s above alpha
        k2 = sum(1 for r in mr if _group(r["group_freq"], r["significant"]) == "2cps")
        k4 = sum(1 for r in mr if _group(r["group_freq"], r["significant"]) == "4cps")
        kany = sum(1 for r in mr if r["significant"])
        a = ax[3, ci]
        # Stagger the MAP labels vertically.
        for row_i, (k, col, lab) in enumerate(
                [(k2, S.PURPLE, "2 cps"), (k4, S.TEAL, "4 cps"), (kany, S.INK, "any")]):
            g, post, mapv, (lo, hi) = bayesian_prevalence(k, n, sensitivity=s_eff)
            a.plot(g, post, color=col, lw=2, label=lab)
            if np.isfinite(mapv):
                a.plot(mapv, post.max(), "o", color=col, ms=5)
                a.annotate(f"{lab}: {mapv:.2f} [{lo:.2f}, {hi:.2f}]",
                           xy=(0.97, 0.92 - 0.085 * row_i), xycoords="axes fraction",
                           fontsize=7.5, color=col, ha="right", va="top")
        a.set_xlabel("population prevalence proportion"); a.set_xlim(0, 1)
        a.set_ylim(bottom=0)   # start at 0 so flat posteriors look flat
        if ci == 0:
            a.set_ylabel("posterior density")
        a.legend(loc="upper left", title=f"n = {n},  ŝ = {s_hat:.2f}")
        if not identifiable:
            a.text(0.5, 0.45, "prevalence unidentifiable\n(per-participant power too low:\n"
                   f"ŝ = {s_hat:.2f}; need more targets/pp)",
                   transform=a.transAxes, ha="center", va="center", fontsize=8,
                   color=S.RED, bbox=dict(boxstyle="round", fc="#FBEAEC", ec=S.RED))
    fig.suptitle("Population stride-phase oscillations — "
                 f"{len({r['sub'] for r in rows})} walking participant(s)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98)); S.save(fig, out)


def _fig5(rows, out: Path):
    fig, ax = plt.subplots(3, 2, figsize=(11, 12),
                           subplot_kw=None)
    for i, meas in enumerate(MEASURES):
        sig2 = [r for r in rows if r["measure"] == meas
                and _group(r["group_freq"], r["significant"]) == "2cps"]
        # left: 2-cps fits (relative change)
        axL = ax[i, 0]
        xf = np.linspace(0, 100, 200)
        for r in sig2:
            ok = np.isfinite(r["binned"])
            if ok.sum() < 6:
                continue
            w = 2 * np.pi * 2.0 / 100.0
            Xo = np.column_stack([np.ones(ok.sum()), np.cos(w * r["centers"][ok]),
                                  np.sin(w * r["centers"][ok])])
            beta, *_ = np.linalg.lstsq(Xo, r["binned"][ok], rcond=None)
            X = np.column_stack([np.ones_like(xf), np.cos(w * xf), np.sin(w * xf)])
            yfit = X @ beta
            rel = (yfit - yfit.mean()) / (abs(yfit.mean()) + 1e-9)
            axL.plot(xf, rel, color=MEASURE_COL[meas], alpha=0.6)
        axL.axhline(0, color=S.GREY, lw=0.7, zorder=0)
        axL.set_xlabel("target onset (% stride)")
        axL.set_ylabel(f"{S.MEASURE_LABEL[meas]}\n(relative change)")
        n_meas = len({r['sub'] for r in rows if r['measure'] == meas})
        S.panel(axL, "ace"[i], f"{S.MEASURE_LABEL[meas].title()} — "
                f"{len(sig2)}/{n_meas} significant @ ~2 cps", fontsize=10)
        # right: polar histogram of preferred phases with Rayleigh test
        axR = fig.add_subplot(3, 2, 2 * i + 2, projection="polar")
        ax[i, 1].remove()
        phases = np.array([r["phase_2cps"] for r in sig2])
        if phases.size:
            axR.hist(phases % (2 * np.pi), bins=12, color=MEASURE_COL[meas], alpha=0.75)
            R = np.abs(np.mean(np.exp(1j * phases)))
            Z = phases.size * R ** 2
            axR.set_title(f"{'bdf'[i]}  preferred phase  (Rayleigh Z = {Z:.2f}, n = {phases.size})",
                          fontsize=9.5, pad=12)
        else:
            axR.set_title(f"{'bdf'[i]}  preferred phase  (no significant participants)",
                          fontsize=9.5, pad=12)
    fig.suptitle("Per-participant 2-cps oscillations & preferred phase",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); S.save(fig, out)


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
    fig, ax = plt.subplots(3, 3, figsize=(13, 12))
    row_specs = [("R² @ ~2 cps", lambda r: _band_max_r2(r["freqs"], r["r2"], 1.8, 2.2)),
                 ("R² @ ~4 cps", lambda r: _band_max_r2(r["freqs"], r["r2"], 3.8, 4.2)),
                 ("max-fit freq (cps)", lambda r: r["best_freq"])]
    ncomp = 9
    for ri, (ylab, getter) in enumerate(row_specs):
        for ci, meas in enumerate(MEASURES):
            a = ax[ri, ci]
            mr = [r for r in rows if r["measure"] == meas]
            x = [r["stride_dur"] for r in mr]
            y = [getter(r) for r in mr]
            lab = _scatter_corr(a, x, y, MEASURE_COL[meas], ncomp=ncomp)
            if ri == 0:
                a.set_title(["Accuracy", "Reaction time", "Response likelihood"][ci],
                            fontsize=11, loc="left")
            if ci == 0:
                a.set_ylabel(ylab)
            if ri == 2:
                a.set_xlabel("stride-cycle duration (s)")
            a.annotate(lab, xy=(0.04, 0.93), xycoords="axes fraction", fontsize=8,
                       color=S.INK, va="top")
    fig.suptitle("Stride-cycle duration vs oscillation strength (Davidson Supp Fig 7) — "
                 f"{len({r['sub'] for r in rows})} walking participant(s)",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Solid line = significant after Bonferroni (9 panels); dashed = ns. "
                   "Davidson: all ns — idiosyncratic frequency is not a stride-duration artefact.")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); S.save(fig, out)


def _suppl8(rows, out: Path):
    """Supp Fig 8: best-fit frequency in Hz (cps / stride duration) vs stride duration.

    Davidson predicts a negative correlation within the ~2 Hz and ~4 Hz clusters (shorter
    strides, faster oscillation), masked in the pooled data by Simpson's paradox."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    clusters = [("~2 Hz", 1.5, 2.5, S.PURPLE), ("~4 Hz", 3.5, 4.5, S.TEAL)]
    for ci, meas in enumerate(MEASURES):
        a = ax[ci]
        mr = [r for r in rows if r["measure"] == meas]
        sd = np.array([r["stride_dur"] for r in mr], float)
        hz = np.array([r["best_freq"] / r["stride_dur"]
                       if r["stride_dur"] and np.isfinite(r["stride_dur"]) else np.nan
                       for r in mr], float)
        a.scatter(sd, hz, s=24, color=MEASURE_COL[meas], alpha=0.55, zorder=2)
        r_all, p_all = _corr(sd, hz)
        notes = [f"all: r={r_all:.2f} p={p_all:.3f}"] if np.isfinite(r_all) else ["all: n<3"]
        for name, lo, hi, col in clusters:
            m = (hz >= lo) & (hz <= hi)
            if m.sum() >= 3:
                rc, pc = _corr(sd[m], hz[m])
                xs = np.linspace(sd[m].min(), sd[m].max(), 2)
                b = np.polyfit(sd[m][np.isfinite(sd[m])], hz[m][np.isfinite(sd[m])], 1)
                a.plot(xs, np.polyval(b, xs), color=col, lw=2, zorder=3)
                notes.append(f"{name}: r={rc:.2f} p={pc:.3f}")
        a.set_xlabel("stride-cycle duration (s)"); a.set_ylabel("best Fourier fit (Hz)")
        S.panel(a, "abc"[ci], S.MEASURE_LABEL[meas].title())
        a.annotate("\n".join(notes), xy=(0.04, 0.96), xycoords="axes fraction",
                   fontsize=7.6, va="top", color=S.INK)
    fig.suptitle("Perceptual oscillation frequency (Hz) vs stride duration "
                 "(Davidson Supp Fig 8)", fontsize=12, fontweight="semibold",
                 x=0.012, ha="left")
    S.caption(fig, "Within a frequency cluster, shorter strides → faster Hz oscillation "
                   "(negative slope) — the stride sets the period.")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); S.save(fig, out)


def main():
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
    print(f"  (de-biased with measured per-participant sensitivity at amp={power.REF_AMP}; "
          f"naive = sensitivity 1.0 = lower bound)")
    for meas in MEASURES:
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


if __name__ == "__main__":
    main()
