"""Signal-detection decomposition of the stride-phase effect: d' vs criterion.

Question
--------
``phase_analysis`` tests whether accuracy oscillates over the stride, but not what
oscillates. In left/right tilt discrimination, accuracy depends on two separable
quantities:

  * sensitivity (d'): how well the tilt is seen;
  * criterion (c): offset of the decision boundary, i.e. a preference for "right" or
    "left".

A criterion that varies over the stride lowers accuracy just like a sensitivity dip, so an
accuracy oscillation can be perceptual, decisional or both. Davidson et al. (2024) did not
separate these. A response-side bias is exactly what ``c`` measures.

No extra data are needed: ``events.tsv`` has ``stimulus_side`` and ``response`` for every
trial, and tilt directions are balanced 50/50 (``CheckpointManager.ConstrainedShuffleNoLongRun``),
so both rates can be estimated at every phase.

Computation
-----------
"Tilt right" is the signal:

    H = P(said right | tilt right)      F = P(said right | tilt left)
    d' = z(H) - z(F)                    c = -0.5 * (z(H) + z(F))

Positive ``c`` = bias against answering right. Which side is the signal is a convention,
so |c| and its modulation are meaningful, not the resting sign.

Rates use the log-linear correction (Hautus 1995): +0.5 to each count and +1 to each
denominator, in every bin. This keeps d' finite in all-correct bins and, applied equally
to every permutation, does not bias the test.

Timeouts are excluded (unlike the accuracy analysis, where they count as incorrect): with
no response a trial is neither hit nor false alarm. The number excluded is reported per
participant. Whether timeouts depend on phase is a separate question, not tested here.

20 bins instead of 40, because each bin needs two rates. With ~1440 pooled walking trials
that is ~72 trials per bin, ~36 per cell. For the same reason only pooled trials per
participant are used; a single ~180-trial run would give ~4 trials per cell.

The test
--------
Same method as the confirmatory analysis: cosine fit at 2 and 4 cycles per stride,
Holm-corrected, against a trial-level permutation null.

The null shuffles the ``(stimulus_side, response)`` pair across trials with phases fixed.
Shuffling the pair keeps each trial's outcome and both marginals and only changes the phase
of each outcome; this is the exact randomisation for "neither d' nor c depends on phase".

The population test is ``group_agnostic.agnostic_test`` (studentised, rotation-invariant
``S = sum_i |z_i|^2 / s_i^2`` against the participants' own permutations), reused so both
modules compute the same statistic.

Exploratory. Confirmatory tests remain ``phase_analysis`` (per participant) and
``group_agnostic`` (population) on accuracy. This module decomposes an accuracy effect; it
does not test whether one exists.

Usage::

    python -m Analysis.sdt_phase
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
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

# 20 bins: each bin holds two rates (hit, false alarm), so needs about twice the trials.
N_BINS = 20
N_PERM = 1000
FREQS = (2.0, 4.0)
ALPHA = 0.05
# Bins with fewer answered trials are dropped (only relevant for short recordings).
MIN_BIN_N = 8
MIN_TRIALS = 200          # minimum answered trials per participant
RNG_SEED = 23
MEASURES = ("dprime", "criterion")
LABEL = {"dprime": "sensitivity d'", "criterion": "criterion c"}
COLOR = {"dprime": S.BLUE, "criterion": S.PURPLE}

_Z_CLIP = 5.0             # clip for degenerate cells


def _z(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF without scipy.

    Acklam's rational approximation with one Halley step using ``erfc``. Maximum absolute
    error vs ``scipy.stats.norm.ppf`` over 20 000 points in [1e-6, 1 - 1e-6] is 1e-11.
    """
    import math
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        x[lo] = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        x[hi] = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                  / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        x[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                  / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    # one Halley refinement
    e = 0.5 * np.array([math.erfc(-v / math.sqrt(2)) for v in x.ravel()]).reshape(x.shape) - p
    u = e * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return np.clip(x, -_Z_CLIP, _Z_CLIP)


# ===========================================================================
# Binned SDT
# ===========================================================================
def sdt_bins(phase, is_signal, said_signal, nbins=N_BINS, min_n=MIN_BIN_N):
    """Per-phase-bin ``(centers, d', c, hit rate, FA rate, n_bin, valid mask)``.

    Rates use the log-linear correction, so d' and c are always finite. ``valid`` marks
    bins with at least ``min_n`` answered trials and at least one trial per stimulus class.
    """
    phase = np.asarray(phase, float)
    sig = np.asarray(is_signal, bool)
    said = np.asarray(said_signal, bool)
    edges = np.linspace(0, 100, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)

    n_sig = np.bincount(idx[sig], minlength=nbins).astype(float)
    n_noi = np.bincount(idx[~sig], minlength=nbins).astype(float)
    n_hit = np.bincount(idx[sig & said], minlength=nbins).astype(float)
    n_fa = np.bincount(idx[~sig & said], minlength=nbins).astype(float)

    # Log-linear correction (Hautus 1995): +0.5 per count, +1 per denominator.
    hr = (n_hit + 0.5) / (n_sig + 1.0)
    fa = (n_fa + 0.5) / (n_noi + 1.0)
    zh, zf = _z(hr), _z(fa)
    dprime = zh - zf
    crit = -0.5 * (zh + zf)
    n_bin = n_sig + n_noi
    valid = (n_bin >= min_n) & (n_sig >= 1) & (n_noi >= 1)
    return centers, dprime, crit, hr, fa, n_bin, valid


def analyse_measure(phase, is_signal, said_signal, which: str, *,
                    nbins=N_BINS, nperm=N_PERM, rng=None) -> dict | None:
    """Confirmatory-style 2/4 cps test on the binned d' or criterion curve.

    As ``phase_analysis.analyse_measure``: weighted cosine fit, Holm over two frequencies,
    +1-corrected permutation p, and exported ``(cos, sin)`` coefficients for the population
    test.
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    phase = np.asarray(phase, float)
    sig = np.asarray(is_signal, bool)
    said = np.asarray(said_signal, bool)

    centers, dp, cr, hr, fa, n_bin, valid = sdt_bins(phase, sig, said, nbins)
    if valid.sum() < 6:
        return None
    x = centers[valid]
    y = (dp if which == "dprime" else cr)[valid]
    # Bin counts are fixed across permutations, so weights are constant. Total bin n is
    # used instead of the SDT variance: with balanced stimulus classes it tracks precision,
    # and a data-dependent weight would change with each shuffle.
    w = n_bin[valid] if getattr(C, "WEIGHT_BINS_BY_N", True) else None

    obs, obs_beta = pa._spectrum_with_coef(x, y, w)
    i2 = pa.IDX_2CPS
    i4 = pa.IDX_4CPS
    obs_2, obs_4 = float(obs[i2]), float(obs[i4])

    null_2 = np.empty(nperm)
    null_4 = np.empty(nperm)
    null_ab_2 = np.zeros((nperm, 2))
    null_ab_4 = np.zeros((nperm, 2))
    n = phase.size
    for k in range(nperm):
        # shuffle (stimulus, response) pairs against fixed phases
        perm = rng.permutation(n)
        _, dpp, crp, _, _, _, vp = sdt_bins(phase, sig[perm], said[perm], nbins)
        use = valid & vp
        if use.sum() < 6:
            null_2[k] = null_4[k] = 0.0
            continue
        yp = (dpp if which == "dprime" else crp)[use]
        wp = n_bin[use] if w is not None else None
        spec, beta = pa._spectrum_with_coef(centers[use], yp, wp)
        null_2[k], null_4[k] = float(spec[i2]), float(spec[i4])
        null_ab_2[k] = beta[i2, 1:]
        null_ab_4[k] = beta[i4, 1:]

    p_2 = float((1 + np.sum(null_2 >= obs_2)) / (1 + nperm))
    p_4 = float((1 + np.sum(null_4 >= obs_4)) / (1 + nperm))
    p_2h, p_4h = (float(v) for v in pa.holm_bonferroni([p_2, p_4]))
    r2_2, amp_2, phi_2, _ = pa.fit_one(x, y, 2.0, w)
    r2_4, amp_4, phi_4, _ = pa.fit_one(x, y, 4.0, w)
    return {
        "measure": which,
        "centers": centers, "binned": np.where(valid, dp if which == "dprime" else cr, np.nan),
        "hit_rate": hr, "fa_rate": fa, "bin_n": n_bin, "valid": valid,
        "mean": float(np.average(y, weights=w if w is not None else None)),
        "p_2cps": p_2, "p_4cps": p_4, "p_2cps_holm": p_2h, "p_4cps_holm": p_4h,
        "significant_2cps": bool(p_2h < ALPHA), "significant_4cps": bool(p_4h < ALPHA),
        "significant": bool(p_2h < ALPHA or p_4h < ALPHA),
        "r2_at_2cps": float(r2_2), "amp_at_2cps": float(amp_2),
        "preferred_phase_2cps_rad": float(phi_2),
        "r2_at_4cps": float(r2_4), "amp_at_4cps": float(amp_4),
        "preferred_phase_4cps_rad": float(phi_4),
        "conf_best_freq_cps": 2.0 if (p_2h, -obs_2) <= (p_4h, -obs_4) else 4.0,
        "coef_2cps": obs_beta[i2, 1:], "coef_4cps": obs_beta[i4, 1:],
        "null_coef_2cps": null_ab_2, "null_coef_4cps": null_ab_4,
        "_null_2": null_2, "_null_4": null_4,
    }


# ===========================================================================
# Trial extraction
# ===========================================================================
def _recording_trials(rec):
    """``(phase, is_signal, said_signal)`` for one walking recording, or ``None``.

    Uses the phase assignment of ``phase_analysis``, so SDT and accuracy curves share the
    same phases.
    """
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    cont = pa.continuous_phase(g)
    tp = pa._trial_phases(rec, g["troughs"], step_foot, tg, cont)
    if tp is None:
        return None
    ev, _ = tp
    if "stimulus_side" not in ev.columns or "response" not in ev.columns:
        return None
    side = ev["stimulus_side"].astype(object)
    resp = ev["response"].astype(object)
    # timeouts are neither hits nor false alarms
    keep = resp.notna() & side.notna() & ev["phase_target"].notna()
    if keep.sum() < 20:
        return None
    return {
        "phase": ev.loc[keep, "phase_target"].to_numpy(float),
        "is_signal": (side[keep].str.lower() == "right").to_numpy(bool),
        "said_signal": (resp[keep].str.lower() == "right").to_numpy(bool),
        "n_timeout": int((~resp.notna()).sum()),
    }


def _pool_subject(recs):
    """Concatenate all walking runs of one participant."""
    parts = [t for t in (_recording_trials(r) for r in recs) if t]
    if not parts:
        return None
    return {
        "phase": np.concatenate([p["phase"] for p in parts]),
        "is_signal": np.concatenate([p["is_signal"] for p in parts]),
        "said_signal": np.concatenate([p["said_signal"] for p in parts]),
        "n_timeout": int(sum(p["n_timeout"] for p in parts)),
        "n_runs": len(parts),
    }


def analyse_subject(tr, *, rng=None) -> dict | None:
    """Both SDT measures for one participant's pooled walking trials."""
    if tr["phase"].size < MIN_TRIALS:
        return None
    rng = rng or np.random.default_rng(RNG_SEED)
    out = {"n_trials": int(tr["phase"].size), "n_runs": int(tr["n_runs"]),
           "n_timeout_excluded": int(tr["n_timeout"]),
           "n_signal": int(tr["is_signal"].sum()),
           "n_said_signal": int(tr["said_signal"].sum())}
    # Overall (unbinned) d' and c; a large |c| is a response bias.
    _, dp0, cr0, hr0, fa0, _, _ = sdt_bins(tr["phase"], tr["is_signal"],
                                           tr["said_signal"], nbins=1, min_n=0)
    out["overall_dprime"] = float(dp0[0])
    out["overall_criterion"] = float(cr0[0])
    out["overall_hit_rate"] = float(hr0[0])
    out["overall_fa_rate"] = float(fa0[0])
    for m in MEASURES:
        out[m] = analyse_measure(tr["phase"], tr["is_signal"], tr["said_signal"], m,
                                 rng=rng)
    return out if any(out.get(m) for m in MEASURES) else None


# ===========================================================================
# Figures
# ===========================================================================
def _curve(ax, res, which, color):
    m = res[which]
    if not m:
        ax.text(0.5, 0.5, "not estimable", ha="center", va="center",
                transform=ax.transAxes, color=S.GREY)
        return
    ok = np.isfinite(m["binned"])
    ax.scatter(m["centers"][ok], m["binned"][ok], s=14, color=S.MIST, zorder=1)
    ax.plot(m["centers"][ok], m["binned"][ok], "-", color=S.INK, lw=1, zorder=2)
    f = m["conf_best_freq_cps"]
    w = 2 * np.pi * f / 100.0
    Xo = np.column_stack([np.ones(ok.sum()), np.cos(w * m["centers"][ok]),
                          np.sin(w * m["centers"][ok])])
    rt = np.sqrt(m["bin_n"][ok])
    beta, *_ = np.linalg.lstsq(Xo * rt[:, None], m["binned"][ok] * rt, rcond=None)
    xf = np.linspace(0, 100, 200)
    Xf = np.column_stack([np.ones_like(xf), np.cos(w * xf), np.sin(w * xf)])
    ph = m[f"p_{int(f)}cps_holm"]
    ax.plot(xf, Xf @ beta, color=color, lw=2.6,
            label=f"{f:.0f} cps, R²={m[f'r2_at_{int(f)}cps']:.2f}, "
                  f"Holm p={ph:.3f}" + ("*" if ph < ALPHA else ""))
    if which == "criterion":
        S.reference_line(ax, y=0.0, label="no bias")
    ax.set_xlabel("% stride cycle")
    ax.set_ylabel(LABEL[which])
    # Room for the legend above the noisy per-bin trace.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.28 * (hi - lo))
    ax.legend(loc="upper center", fontsize=8, ncol=1, framealpha=0.9)


def _figure(label, res, out: Path):
    fig, axg = plt.subplots(2, 2, figsize=(11.5, 7.6))
    ax = axg.ravel()
    fig.suptitle(f"Sensitivity vs criterion over the stride — {S.nice_label(label)}  "
                 f"(n = {res['n_trials']} responded trials)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")

    _curve(ax[0], res, "dprime", COLOR["dprime"])
    S.panel(ax[0], "a", "Sensitivity", fontsize=10)
    _curve(ax[1], res, "criterion", COLOR["criterion"])
    S.panel(ax[1], "b", "Criterion (decision bias)", fontsize=10)

    # c) hit and false-alarm rates
    m = res["dprime"] or res["criterion"]
    if m:
        ok = np.isfinite(m["binned"])
        ax[2].plot(m["centers"][ok], m["hit_rate"][ok], "-o", ms=3, color=S.GREEN,
                   lw=1.2, label="hit  P(say right | right)")
        ax[2].plot(m["centers"][ok], m["fa_rate"][ok], "-o", ms=3, color=S.RED,
                   lw=1.2, label="false alarm  P(say right | left)")
        ax[2].set_ylim(0, 1.22)              # room for the legend
        S.reference_line(ax[2], y=0.5)
        ax[2].set_xlabel("% stride cycle"); ax[2].set_ylabel("rate")
        ax[2].legend(loc="upper center", fontsize=8, ncol=2, framealpha=0.9)
    S.panel(ax[2], "c", "Underlying rates (log-linear corrected)", fontsize=10)

    # d) permutation nulls at the reported frequency
    drawn = False
    for which in MEASURES:
        mm = res[which]
        if not mm:
            continue
        f = int(mm["conf_best_freq_cps"])
        ns = mm[f"_null_{f}"]
        ax[3].hist(ns, bins=40, alpha=0.55, color=COLOR[which],
                   label=f"{LABEL[which]} null @ {f} cps")
        ax[3].axvline(mm[f"r2_at_{f}cps"], color=COLOR[which], lw=2.2, ls="-")
        drawn = True
    if drawn:
        ax[3].set_xlabel("R² under the stride-phase permutation null")
        ax[3].set_ylabel("permutations")
        ax[3].legend(fontsize=8)
    S.panel(ax[3], "d", "Permutation nulls (vertical line = observed)", fontsize=10)

    S.caption(fig, "EXPLORATORY. Signal = tilt right. Timeouts excluded (no response = "
                   "neither hit nor false alarm). Null shuffles the (stimulus, response) "
                   "pair against fixed stride phase, so both marginals and every trial "
                   "outcome are held; only when each outcome occurred is randomised. "
                   f"Overall d'={res['overall_dprime']:.2f}, c={res['overall_criterion']:+.3f}.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    S.save(fig, out)


def _population_figure(rows, pop, out: Path):
    fig, axg = plt.subplots(1, 3, figsize=(14.5, 4.6))
    subs = [r["sub"] for r in rows]
    ypos = np.arange(len(rows))

    # a) overall criterion per participant (response bias)
    crit = [r["overall_criterion"] for r in rows]
    axg[0].barh(ypos, crit, color=S.PURPLE, alpha=0.85)
    S.reference_line(axg[0], x=0.0, label="no bias")
    axg[0].set_yticks(ypos); axg[0].set_yticklabels(subs, fontsize=8)
    axg[0].set_xlabel("criterion c  (+ = biased against 'right')")
    axg[0].legend(fontsize=8)
    S.panel(axg[0], "a", "Resting response bias", fontsize=10)

    # b/c) population nulls per measure
    for i, which in enumerate(MEASURES):
        ax = axg[i + 1]
        d = pop.get(which)
        if not d:
            ax.text(0.5, 0.5, "not estimable", ha="center", va="center",
                    transform=ax.transAxes, color=S.GREY)
            S.panel(ax, "bc"[i], LABEL[which], fontsize=10)
            continue
        best = d["best_freq"]
        r = d["by_freq"][best]
        ax.hist(r["_null_stat"], bins=40, color=S.MIST, edgecolor="white")
        ax.axvline(r["statistic"], color=COLOR[which], lw=2.2,
                   label=f"observed (Holm p={r['p_holm']:.3f}"
                         f"{'*' if r['significant'] else ''})")
        S.reference_line(ax, x=r["null_p95"], label="null 95th pct")
        ax.set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$" + f"  @ {best:.0f} cps")
        ax.set_ylabel("permutations")
        ax.legend(fontsize=8)
        S.panel(ax, "bc"[i], f"Population: {LABEL[which]}", fontsize=10)

    fig.suptitle("Signal-detection decomposition — population",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, f"n = {len(rows)} participants. Population statistic and null are "
                   "group_agnostic.agnostic_test, unchanged — studentised per "
                   "participant, rotation-invariant, built from each participant's own "
                   "shuffles. EXPLORATORY: identifies what an accuracy effect is made "
                   "of, does not test whether one exists.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    S.save(fig, out)


# ===========================================================================
# Population
# ===========================================================================
def population_test(rows) -> dict:
    """Studentised phase-agnostic test per SDT measure, at 2 and 4 cps (Holm)."""
    out = {}
    for which in MEASURES:
        have = [r for r in rows if r.get(which)]
        if len(have) < 2:
            continue
        by_freq = {}
        for f in FREQS:
            key = f"{f:.0f}cps"
            obs = np.array([r[which][f"coef_{key}"] for r in have], float)
            nper = min(r[which][f"null_coef_{key}"].shape[0] for r in have)
            null = np.stack([r[which][f"null_coef_{key}"][:nper] for r in have])
            by_freq[f] = ga.agnostic_test(obs, null)
        padj = pa.holm_bonferroni([by_freq[f]["p_value"] for f in FREQS])
        for f, adj in zip(FREQS, padj):
            by_freq[f]["p_holm"] = float(adj)
            by_freq[f]["significant"] = bool(adj < ALPHA)
        best = min(FREQS, key=lambda f: (by_freq[f]["p_holm"], -by_freq[f]["statistic"]))
        out[which] = {
            "by_freq": by_freq, "best_freq": best,
            "n_participants": int(len(have)),
            "significant_any": bool(any(by_freq[f]["significant"] for f in FREQS)),
        }
    return out


def _json_safe(o):
    if isinstance(o, dict):
        # Keys may be float frequencies; convert to strings before filtering "_" keys.
        out = {}
        for k, v in o.items():
            ks = f"{k:.0f}cps" if isinstance(k, float) else str(k)
            if ks.startswith("_"):
                continue
            out[ks] = _json_safe(v)
        return out
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_json_safe(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


# ===========================================================================
def main():
    """Decompose the stride-phase effect into sensitivity and criterion."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    if not recs:
        print("No walking recordings found. Run the gait stage first.")
        return
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)

    out_dir = bids_io.deriv_dir("sdt")
    rows = []
    print("=== Sensitivity vs criterion over the stride (EXPLORATORY) ===")
    print("  Signal = tilt right. Timeouts excluded. 20 phase bins, log-linear rates.")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        tr = _pool_subject(subrecs)
        if tr is None:
            continue
        res = analyse_subject(tr, rng=np.random.default_rng(
            abs(hash((sub, ses, "sdt"))) % 2**32))
        if res is None:
            print(f"  {sub} {ses}: below {MIN_TRIALS} responded trials - skipped")
            continue
        res.update({"sub": sub, "ses": ses})
        rows.append(res)
        label = f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking"
        (out_dir / f"{label}_sdt.json").write_text(
            json.dumps(_json_safe(res), indent=2), encoding="utf-8")
        _figure(label, res, bids_io.fig_dir("sdt") / f"{label}_sdt_phase.png")
        bits = []
        for m in MEASURES:
            mm = res[m]
            if mm:
                f = int(mm["conf_best_freq_cps"])
                bits.append(f"{m} {f}cps R2={mm[f'r2_at_{f}cps']:.3f} "
                            f"p={mm[f'p_{f}cps_holm']:.3f}"
                            + ("*" if mm["significant"] else ""))
        print(f"  {sub} {ses}: n={res['n_trials']} (+{res['n_timeout_excluded']} timeouts), "
              f"d'={res['overall_dprime']:.2f}, c={res['overall_criterion']:+.3f}"
              + ("  |  " + "; ".join(bits) if bits else ""))

    bids_io.gate_report("sdt", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("sdt",))
    if len(rows) < 2:
        bids_io.clear_stale("sdt", "sdt_population.json", "sdt_population.png")
    if len(rows) >= 2:
        pop = population_test(rows)
        payload = {"n_participants": len(rows),
                   "note": "EXPLORATORY signal-detection decomposition. Confirmatory "
                           "tests remain phase_analysis and group_agnostic on accuracy.",
                   "measures": _json_safe(pop)}
        (out_dir / "sdt_population.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        _population_figure(rows, pop, bids_io.fig_dir("sdt") / "sdt_population.png")
        print(f"\n  POPULATION (n={len(rows)}):")
        for which in MEASURES:
            d = pop.get(which)
            if not d:
                continue
            f = d["best_freq"]
            r = d["by_freq"][f]
            print(f"    {LABEL[which]:<16}: S={r['statistic']:.2f} vs null mean "
                  f"{r['null_mean']:.2f} @ {f:.0f} cps, Holm p={r['p_holm']:.4f} -> "
                  f"{'MODULATED' if r['significant'] else 'n.s.'}")
        d_sig = pop.get("dprime", {}).get("significant_any")
        c_sig = pop.get("criterion", {}).get("significant_any")
        if d_sig and not c_sig:
            print("    => reads as SENSORY: sensitivity oscillates, criterion does not.")
        elif c_sig and not d_sig:
            print("    => reads as DECISIONAL: the criterion oscillates, sensitivity does not.")
        elif d_sig and c_sig:
            print("    => BOTH move; the accuracy effect is not attributable to one locus.")
        else:
            print("    => neither reaches significance; no decomposition claim.")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
