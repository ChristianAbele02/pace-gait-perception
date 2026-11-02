"""Population-level stride-phase results, after Davidson et al. (2024) Fig 4 and 5.

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
_VERDICT_STYLE = {"2cps": (S.PURPLE, "significant at 2 cps"),
                  "4cps": (S.TEAL, "significant at 4 cps"),
                  "none": (S.MIST, "not significant")}


def _fig4(rows, out: Path):
    """Per measure: every participant's R² spectrum coloured by their confirmatory
    verdict (top), and the de-biased prevalence posteriors that verdict feeds (bottom).
    Spectra and counts use the same verdict, so a coloured line is always one of the k."""
    fig, ax = plt.subplots(2, len(MEASURES), figsize=(5.6 * len(MEASURES), 8.4), squeeze=False)
    for ci, meas in enumerate(MEASURES):
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
    fig = plt.figure(figsize=(11, 4.4 * len(MEASURES)))
    gs = fig.add_gridspec(len(MEASURES), 2, width_ratios=[1.6, 1], hspace=0.55, wspace=0.25)
    xf = np.linspace(0, 100, 300)
    w = 2 * np.pi * 2.0 / 100.0
    for i, meas in enumerate(MEASURES):
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
    fig, ax = plt.subplots(len(row_specs), len(MEASURES), figsize=(4.8 * len(MEASURES), 10),
                           squeeze=False, sharex=True)
    ncomp = len(row_specs) * len(MEASURES)
    tags = "abcdefghi"
    for ri, (ylab, getter) in enumerate(row_specs):
        for ci, meas in enumerate(MEASURES):
            a = ax[ri, ci]
            mr = [r for r in rows if r["measure"] == meas]
            x = [r["stride_dur"] for r in mr]
            y = [getter(r) for r in mr]
            lab = _scatter_corr(a, x, y, MEASURE_COL[meas], ncomp=ncomp)
            if ci == 0:
                a.set_ylabel(ylab)
            if ri == len(row_specs) - 1:
                a.set_xlabel("mean stride duration (s)")
            S.panel(a, tags[ri * len(MEASURES) + ci],
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
    fig, ax = plt.subplots(1, len(MEASURES), figsize=(5.2 * len(MEASURES), 4.6), squeeze=False)
    ax = ax[0]
    clusters = [("~2 Hz", 1.5, 2.5, S.PURPLE), ("~4 Hz", 3.5, 4.5, S.TEAL)]
    for ci, meas in enumerate(MEASURES):
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
