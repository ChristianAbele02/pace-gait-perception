"""Stride-phase modulation of performance, after Davidson et al. (2024) Fig 3.

For each non-training trial the stride phase at target onset is taken from gait.py, and
each measure is tested for oscillation over the stride. Measures: accuracy (primary) and
reaction time (secondary). Response likelihood was removed (see config, outcome measures).

  * 40 phase bins per stride (1 stride = 2 steps);
  * first-order Fourier fit ``A·cos(w·x+φ)+c`` at 0.1-10 cycles per stride (cps) in 0.1
    steps (Davidson's grid), keeping R² (linear least squares, exact for fixed frequency);
  * null from 1000 trial-level permutations: trial values are shuffled against fixed
    phases and re-binned, so the null keeps the real bin counts and within-bin variance.

Confirmatory per-participant test: R² at exactly 2 cps and 4 cps (where Davidson's
responders peak) against their permutation nulls, Holm-corrected over the pair
(FWER ≤ 0.05). Two targeted tests have much more power than a 100-frequency search.

Exploratory frequency scan: max R² over all frequencies per shuffle (95th percentile null),
reported as ``significant_grid`` / ``best_freq_cps`` and in the R² spectrum figure.

Also reports the preferred phase (φ) at 2 and 4 cps. Produces the Fig 3c-g panels per
block; per-participant results feed ``population.py`` (Fig 4/5). A pooled test over all
walking participants' trials is added as an exploratory supplement; it ignores
between-participant variance and is not a population test. Davidson's main effect is a
~2-cps oscillation (once per step) peaking near swing.

Usage::

    python -m Analysis.phase_analysis
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
    from . import config as C
    from . import bids_io
    from . import gait as gait_mod
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import gait as gait_mod
    import plotstyle as S

S.apply()

N_BINS = 40                                   # 40 bins per stride, as Davidson
N_ONSET_BINS = 20                             # coarser bins for the onset figure
FREQS = np.round(np.arange(0.1, 10.0001, 0.1), 1)   # cycles per stride (Davidson: 0.1-10 cps, 0.1 step)
N_PERM = 1000
RNG = np.random.default_rng(42)

# Confirmatory frequencies: 2 cps (step rate) and 4 cps. Davidson's responders peak at
# one of them. Holm-corrected over the pair; the full-grid scan is exploratory.
CONF_FREQS = (2.0, 4.0)
IDX_2CPS = int(np.argmin(np.abs(FREQS - 2.0)))
IDX_4CPS = int(np.argmin(np.abs(FREQS - 4.0)))


def holm_bonferroni(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values, in input order."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj

# --- Reaction-time filtering (b5) ---
RT_MIN_S = 0.15        # faster responses are anticipations
RT_MAD_K = 3.0         # reject |log RT - median| > k * (1.4826 * MAD)


def difficulty_adjust(y: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Subtract each difficulty level's mean from ``y`` and add the grand mean back.

    The three tilt levels (threshold / easy / hard, 0.85 / 0.10 / 0.05) are drawn
    independently of stride phase, so level differences only add noise to the bin means.
    Adding the grand mean back keeps the original units (hit rate, log10 RT).

    This only reduces variance: the adjustment depends on (y, level) alone and is applied
    identically in the permutation null, so the test stays exact. It recovers most of the
    ~5 points of power the difficulty mix costs (``davidson_power/``).

    A level with a single trial becomes the grand mean. Disabled by
    ``config.RESIDUALISE_DIFFICULTY = False``.
    """
    y = np.asarray(y, float)
    if not getattr(C, "RESIDUALISE_DIFFICULTY", True) or level is None:
        return y
    lev = np.asarray(level, dtype=object)
    if lev.shape != y.shape:
        return y
    out = y.copy()
    grand = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
    for name in {str(v) for v in lev}:
        m = np.array([str(v) == name for v in lev])
        if not m.any():
            continue
        sub = y[m]
        if not np.isfinite(sub).any():
            continue
        out[m] = sub - float(np.nanmean(sub)) + grand
    return out


def _clean_rt(ev):  # ev: events DataFrame
    """(phase, log10_rt, info) for the RT analysis: correct trials only, anticipations
    (< 150 ms) and MAD outliers on log RT removed, log10 RT (right-skewed RTs)."""
    m = (ev["response"].notna() & (ev["acc"] == 1.0) & (ev["rt"] > RT_MIN_S)
         & np.isfinite(ev["rt"]))
    sub = ev[m]
    if len(sub) < 6:
        return None
    lrt = np.log10(sub["rt"].to_numpy(float))
    med = np.median(lrt)
    mad = 1.4826 * np.median(np.abs(lrt - med)) or 1e-9
    keep = np.abs(lrt - med) <= RT_MAD_K * mad
    info = {"n_in": int(len(ev)), "n_correct": int((ev["acc"] == 1.0).sum()),
            "n_anticip": int((ev["rt"] <= RT_MIN_S).sum()),
            "n_rt_used": int(keep.sum()), "n_rt_outliers": int((~keep).sum())}
    # Difficulty affects RT more than accuracy, so the adjustment matters more here.
    lev = sub["difficulty"].to_numpy(dtype=object) if "difficulty" in sub.columns else None
    lrt_adj = difficulty_adjust(lrt, lev)
    info["difficulty_adjusted"] = bool(
        lev is not None and getattr(C, "RESIDUALISE_DIFFICULTY", True))
    return sub["phase_target"].to_numpy(float)[keep], lrt_adj[keep], info


# ===========================================================================
# Fourier fit (linear least squares per frequency)
# ===========================================================================
def fit_one(x_pct: np.ndarray, y: np.ndarray, f_cps: float, w_bin: np.ndarray | None = None):
    """Fit y ≈ c + a·cos(wx) + b·sin(wx), w = 2π·f/100. Returns (R2, amp, phase, fitted).

    ``w_bin``: optional per-bin trial counts as weights (a bin mean has variance
    sigma^2/n). ``None`` gives Davidson's unweighted fit.
    """
    w = 2 * np.pi * f_cps / 100.0
    X = np.column_stack([np.ones_like(x_pct), np.cos(w * x_pct), np.sin(w * x_pct)])
    if w_bin is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ybar = y.mean()
        wts = np.ones_like(y)
    else:
        wts = np.asarray(w_bin, float)
        rt = np.sqrt(wts)
        beta, *_ = np.linalg.lstsq(X * rt[:, None], y * rt, rcond=None)
        ybar = float(np.sum(wts * y) / np.sum(wts)) if np.sum(wts) > 0 else y.mean()
    yhat = X @ beta
    ss_res = float(np.sum(wts * (y - yhat) ** 2))
    ss_tot = float(np.sum(wts * (y - ybar) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    amp = float(np.hypot(beta[1], beta[2]))
    phase = float(np.arctan2(-beta[2], beta[1]))   # A·cos(wx+φ)
    return max(0.0, r2), amp, phase, yhat


_DESIGN_CACHE: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}


def _grid_design(x: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Design tensor and pseudo-inverse for all frequencies in ``FREQS`` at these x.

    With ``w`` (per-bin counts) the pseudo-inverse uses the whitened design ``X·sqrt(w)``,
    so ``pinv @ (y·sqrt(w))`` is the weighted LS solution. The returned ``X`` is
    unwhitened, for computing fitted values.

    Uses the same SVD truncation as ``np.linalg.lstsq(X, y, rcond=None)``
    (``max(M, N) * eps * s_max``), batched over frequencies, so results match lstsq even
    for rank-deficient designs with many empty bins.

    Cached on ``x`` (and ``w``): bin centres are fixed across the permutations of one
    ``analyse_measure`` call, so the SVD is computed once.
    """
    key = x.tobytes() + b"|" + (b"" if w is None else np.asarray(w, float).tobytes())
    hit = _DESIGN_CACHE.get(key)
    if hit is not None:
        return hit
    ang = 2 * np.pi * FREQS / 100.0
    ones = np.ones((FREQS.size, x.size))
    X = np.stack([ones, np.cos(np.outer(ang, x)), np.sin(np.outer(ang, x))], axis=2)
    Xw = X if w is None else X * np.sqrt(np.asarray(w, float))[None, :, None]
    u, s, vt = np.linalg.svd(Xw, full_matrices=False)
    cutoff = max(x.size, 3) * np.finfo(float).eps * s[:, :1]
    s_inv = np.where(s > cutoff, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    pinv = np.einsum("fkj,fk,fik->fji", vt, s_inv, u)      # (nfreq, 3, nbins)
    if len(_DESIGN_CACHE) > 32:                            # limit cache size
        _DESIGN_CACHE.clear()
    _DESIGN_CACHE[key] = (X, pinv)
    return X, pinv


def r2_spectrum(x_pct: np.ndarray, y: np.ndarray, w_bin: np.ndarray | None = None) -> np.ndarray:
    """R² of the single-frequency cosine fit at each frequency in ``FREQS``.

    Equal to ``[fit_one(x, y, f)[0] for f in FREQS]`` but faster, especially with the
    cached design in ``analyse_measure`` (1001 calls per measure and recording).

    Agreement with the loop over 8000 test configurations (sparse bins, aliased
    frequencies): 1e-14 for noise, oscillation, binomial-mean and binary y; 0 for constant
    y. For y constant to ~1e-9, R² is numerically undefined and results differ by up to
    3e-7; this cannot occur with real bin means. Both cases are tested in
    ``tests/test_pipeline.py``.
    """
    return _spectrum_with_coef(x_pct, y, w_bin)[0]


def _spectrum_with_coef(x_pct: np.ndarray, y: np.ndarray,
                        w_bin: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(R² per frequency, beta)``, with ``beta`` the (nfreq, 3) solution ``[c, a, b]`` of
    ``y ≈ c + a·cos(wx) + b·sin(wx)``.

    The (a, b) pair is the per-participant coefficient used by ``group_agnostic.py``, so
    the coefficients and the confirmatory R² come from the same computation.

    ``w_bin`` (per-bin counts) gives weighted LS and weighted R²,
    ``1 - sum(w·resid²) / sum(w·(y - ybar_w)²)``. Bin counts are fixed across the
    permutations of one ``analyse_measure`` call, so the null uses the same estimator.
    """
    x = np.asarray(x_pct, float)
    y = np.asarray(y, float)
    if w_bin is None:
        wts = None
        ss_tot = float(((y - y.mean()) ** 2).sum())
    else:
        wts = np.asarray(w_bin, float)
        wsum = float(wts.sum())
        ybar = float((wts * y).sum() / wsum) if wsum > 0 else float(y.mean())
        ss_tot = float((wts * (y - ybar) ** 2).sum())
    if ss_tot <= 0:
        return np.zeros(FREQS.size), np.zeros((FREQS.size, 3))
    X, pinv = _grid_design(x, wts)
    beta = pinv @ (y if wts is None else y * np.sqrt(wts))     # (nfreq, 3)
    resid = y - np.einsum("fij,fj->fi", X, beta)
    ss_res = (resid * resid).sum(1) if wts is None else (wts * resid * resid).sum(1)
    return np.maximum(1.0 - ss_res / ss_tot, 0.0), beta


def _bin(phase: np.ndarray, value: np.ndarray | None, nbins: int, how: str,
         with_counts: bool = False):
    """Bin by stride phase: how='mean' (with SEM) or 'count'.

    ``with_counts=True`` also returns per-bin trial counts, the weights of the weighted
    Fourier fit (``config.WEIGHT_BINS_BY_N``); the default 3-tuple is used elsewhere.
    """
    edges = np.linspace(0, 100, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    sem = np.full(nbins, np.nan)
    cnt = counts.astype(float)
    if how == "count":
        return (centers, cnt, sem, cnt) if with_counts else (centers, cnt, sem)
    v = np.asarray(value, float)
    nz = counts > 0
    n = np.where(nz, counts, 1)
    mean = np.bincount(idx, weights=v, minlength=nbins) / n
    # population SD (ddof=0), clamped at 0
    var = np.maximum(np.bincount(idx, weights=v * v, minlength=nbins) / n - mean ** 2, 0.0)
    out = np.where(nz, mean, np.nan)
    sem[nz] = np.sqrt(var[nz]) / np.sqrt(counts[nz])
    return (centers, out, sem, cnt) if with_counts else (centers, out, sem)


def analyse_measure(phase, value, how, nbins=N_BINS, nperm=N_PERM,
                    null_phase_pool=None) -> dict:
    """Confirmatory 2/4 cps test and exploratory grid scan for one measure.

    ``null_phase_pool`` (count mode only): all target-onset phases of the recording, from
    which the count null draws; without it ``phase`` itself is resampled."""
    phase = np.asarray(phase, float)
    value_arr = None if value is None else np.asarray(value, float)
    centers, binned, sem, bin_n = _bin(phase, value_arr, nbins, how, with_counts=True)
    ok = np.isfinite(binned)
    x, y = centers[ok], binned[ok]
    if x.size < 6:
        return None
    # Bin-count weights for mean measures only (in count mode the count is the outcome).
    # Counts are fixed across permutations, so observed and null use the same weights.
    w_bin = (bin_n[ok] if (how != "count" and getattr(C, "WEIGHT_BINS_BY_N", True)
                           and np.all(bin_n[ok] > 0)) else None)
    obs, obs_beta = _spectrum_with_coef(x, y, w_bin)
    obs_2, obs_4 = float(obs[IDX_2CPS]), float(obs[IDX_4CPS])
    # Coefficient scale: hit rate and log10 RT are comparable across participants; counts
    # are divided by the mean bin count to give relative modulation.
    coef_norm = 1.0
    if how == "count":
        mu = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
        coef_norm = 1.0 / mu if mu > 0 else 0.0
    # Trial-level permutation null, re-binned per shuffle:
    #   * mean measures: shuffle values against fixed phases;
    #   * counts: draw a same-size subset without replacement from the onset phase pool,
    #     which keeps any non-uniform target timing.
    # Per shuffle: max R² over the grid (exploratory), R² at 2 and 4 cps (confirmatory),
    # and the (cos, sin) coefficients at 2 and 4 cps, reused by group_agnostic.py so its
    # null uses the same shuffles.
    null_max = np.empty(nperm)
    null_2 = np.empty(nperm)
    null_4 = np.empty(nperm)
    null_ab_2 = np.zeros((nperm, 2))
    null_ab_4 = np.zeros((nperm, 2))
    if how == "count":
        pool = np.asarray(null_phase_pool, float) if (
            null_phase_pool is not None and np.size(null_phase_pool)) else phase
        pool = pool[np.isfinite(pool)]
        replace = pool.size < phase.size
    for i in range(nperm):
        if how == "count":
            null_phases = RNG.choice(pool, size=phase.size, replace=replace)
            _, b, _ = _bin(null_phases, None, nbins, "count")
        else:
            _, b, _ = _bin(phase, RNG.permutation(value_arr), nbins, "mean")
        bok = np.isfinite(b)
        if bok.sum() >= 6:
            # In mean mode bok == ok; slicing again keeps count mode correct.
            wp = bin_n[bok] if w_bin is not None else None
            spec, beta = _spectrum_with_coef(centers[bok], b[bok], wp)
            null_max[i] = spec.max(); null_2[i] = spec[IDX_2CPS]; null_4[i] = spec[IDX_4CPS]
            null_ab_2[i] = beta[IDX_2CPS, 1:]
            null_ab_4[i] = beta[IDX_4CPS, 1:]
        else:
            null_max[i] = null_2[i] = null_4[i] = 0.0

    # Confirmatory test at 2 and 4 cps, Holm-corrected; +1-corrected permutation p.
    p_2 = float((1 + np.sum(null_2 >= obs_2)) / (1 + nperm))
    p_4 = float((1 + np.sum(null_4 >= obs_4)) / (1 + nperm))
    p_2h, p_4h = (float(v) for v in holm_bonferroni([p_2, p_4]))
    sig_2, sig_4 = bool(p_2h < 0.05), bool(p_4h < 0.05)
    # Frequency for grouping/display: smaller Holm p (ties: larger R²) if significant,
    # else larger R².
    if sig_2 or sig_4:
        conf_best = 2.0 if (p_2h, -obs_2) <= (p_4h, -obs_4) else 4.0
    else:
        conf_best = 2.0 if obs_2 >= obs_4 else 4.0

    # Exploratory full-grid max-R² scan (Davidson's max statistic).
    ci95 = float(np.percentile(null_max, 95))
    bi = int(np.argmax(obs))

    r2_2, amp_2, phi_2, _ = fit_one(x, y, 2.0, w_bin)
    r2_4, amp_4, phi_4, _ = fit_one(x, y, 4.0, w_bin)
    return {
        "centers": centers, "binned": binned, "sem": sem, "bin_n": bin_n,
        "weighted": bool(w_bin is not None),
        "freqs": FREQS, "r2": obs, "ci95": ci95,
        # confirmatory 2 and 4 cps (Holm)
        "significant": bool(sig_2 or sig_4),
        "conf_best_freq_cps": float(conf_best),
        "p_2cps": p_2, "p_4cps": p_4, "p_2cps_holm": p_2h, "p_4cps_holm": p_4h,
        "significant_2cps": sig_2, "significant_4cps": sig_4,
        # exploratory full-grid scan
        "best_freq_cps": float(FREQS[bi]), "best_r2": float(obs[bi]),
        "significant_grid": bool(obs[bi] > ci95),
        # fit at the two confirmatory frequencies
        "r2_at_2cps": float(r2_2), "amp_at_2cps": float(amp_2),
        "preferred_phase_2cps_rad": float(phi_2),
        "r2_at_4cps": float(r2_4), "amp_at_4cps": float(amp_4),
        "preferred_phase_4cps_rad": float(phi_4),
        # observed and permuted (cos, sin) coefficients for group_agnostic.py (not in JSON)
        "coef_2cps": obs_beta[IDX_2CPS, 1:] * coef_norm,
        "coef_4cps": obs_beta[IDX_4CPS, 1:] * coef_norm,
        "null_coef_2cps": null_ab_2 * coef_norm,
        "null_coef_4cps": null_ab_4 * coef_norm,
    }


# ===========================================================================
# Per-recording analysis
# ===========================================================================
def continuous_phase(g) -> np.ndarray | None:
    """Continuous stride phase from a gait ``.npz`` for config.PHASE_METHOD, or None.

    "phaser" -> ``stride_phase_phaser`` (phaser.py); otherwise ``stride_phase_cont``
    (Hilbert), which callers use only when the method is "hilbert".
    """
    key = ("stride_phase_phaser" if getattr(C, "PHASE_METHOD", "trough") == "phaser"
           else "stride_phase_cont")
    return g[key] if key in g and np.size(g[key]) else None


def _phase_of(times, troughs, step_foot, tg=None, cont_phase=None):
    """Stride phase at event times using config.PHASE_METHOD.

    "hilbert" and "phaser" use the continuous phase from gait.py (fall back to troughs if
    missing); "trough" uses Davidson's trough interpolation.
    """
    if (getattr(C, "PHASE_METHOD", "trough") in ("hilbert", "phaser")
            and tg is not None and cont_phase is not None and np.size(cont_phase)):
        _, p = gait_mod.phase_at_hilbert(times, tg, cont_phase)
        return p
    _, p = gait_mod.phase_at(times, troughs, step_foot)
    return p


def _trial_phases(rec, troughs, step_foot=None, tg=None, cont_phase=None):
    ev = bids_io.load_events(rec)
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna()].copy()
    if ev.empty:
        return None
    onset = ev["onset"].to_numpy(float)
    ev["phase_target"] = _phase_of(onset, troughs, step_foot, tg, cont_phase)
    ev["acc"] = np.where(ev["response_correct"] == 1, 1.0, 0.0)   # timeout or incorrect = 0
    # Difficulty-adjusted accuracy for the Fourier fit; ``acc`` stays raw.
    ev["acc_adj"] = difficulty_adjust(
        ev["acc"].to_numpy(float),
        ev["difficulty"].to_numpy(dtype=object) if "difficulty" in ev.columns else None)
    ev["rt"] = ev["response_time"].astype(float)
    resp = ev[ev["response"].notna()].copy()
    press = resp["onset"].to_numpy(float) + resp["rt"].to_numpy(float)
    resp["phase_response"] = _phase_of(press, troughs, step_foot, tg, cont_phase)
    return ev.dropna(subset=["phase_target"]), resp.dropna(subset=["phase_response"])


def analyse(rec) -> dict | None:
    """Stride-phase analysis of one walking block.

    Runs the confirmatory 2/4 cps test and grid scan on accuracy and RT and writes the
    block's JSON, arrays and figure. Returns ``None`` for standing (no stride).

    Per-block results are for QC and figures. Population stages use the per-participant
    pooling in :func:`subject_pooled_analysis`, on which the design's power is based.
    """
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        gait_mod.gait_recording(rec)
    if not npz.exists():                # no usable steps
        print(f"  {rec.label()}: no gait output - skipped")
        return None
    g = np.load(npz)
    troughs = g["troughs"]
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    cont_phase = continuous_phase(g)
    tp = _trial_phases(rec, troughs, step_foot, tg, cont_phase)
    if tp is None:
        return None
    ev, resp = tp

    # RT: correct trials, anticipations and outliers removed, log10 (b5).
    rt_clean = _clean_rt(ev)
    rt_measure = analyse_measure(rt_clean[0], rt_clean[1], "mean") if rt_clean else None
    measures = {
        "accuracy": analyse_measure(ev["phase_target"].to_numpy(),
                                    ev["acc_adj"].to_numpy(), "mean"),
        "reaction_time": rt_measure,
    }
    rt_info = rt_clean[2] if rt_clean else {}
    if all(m is None for m in measures.values()):
        return None

    report = {"recording": rec.label(), "n_trials": int(len(ev)),
              "n_trials_no_phase": _n_unphased(rec, len(ev)),
              "n_responses": int(len(resp)), "rt_qc": rt_info,
              "onset_phase_coverage": _circ_stats(ev["phase_target"].to_numpy(float)),
              "measures": {k: _json_measure(v) for k, v in measures.items() if v}}
    out_dir = bids_io.deriv_dir("phase", rec.sub, rec.ses)
    (out_dir / f"{rec.label()}_phase.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # spectra and binned values for population.py
    np.savez(out_dir / f"{rec.label()}_phase.npz",
             **{f"{k}_{a}": np.asarray(measures[k][a])
                for k in measures if measures[k]
                for a in ("centers", "binned", "freqs", "r2")})
    _figure(rec, ev, measures, g, bids_io.fig_dir("phase", rec.sub, rec.ses) / f"{rec.label()}_phase.png")
    return report


def _json_measure(m: dict) -> dict:
    return {k: v for k, v in m.items()
            if k in ("significant", "conf_best_freq_cps",
                     "p_2cps", "p_4cps", "p_2cps_holm", "p_4cps_holm",
                     "significant_2cps", "significant_4cps",
                     "best_freq_cps", "best_r2", "significant_grid", "ci95",
                     "r2_at_2cps", "amp_at_2cps", "preferred_phase_2cps_rad",
                     "r2_at_4cps", "amp_at_4cps", "preferred_phase_4cps_rad")}


# ===========================================================================
# Figures: one sheet per block (Davidson Fig 3 c-g) and the pooled aggregate
# ===========================================================================
def _circ_stats(phase_pct: np.ndarray) -> dict | None:
    """Circular mean (% stride), resultant length R and large-n Rayleigh p of stride
    phases (R ~ 0 uniform, R -> 1 concentrated)."""
    ang = np.asarray(phase_pct, float)
    ang = ang[np.isfinite(ang)] / 100.0 * 2 * np.pi
    if ang.size == 0:
        return None
    Cb = np.mean(np.exp(1j * ang))
    return {"n": int(ang.size),
            "mean_pct": float((np.angle(Cb) % (2 * np.pi)) / (2 * np.pi) * 100.0),
            "R": float(np.abs(Cb)),
            "rayleigh_p": float(np.exp(-ang.size * np.abs(Cb) ** 2))}


def _wrapped_kde(phase: np.ndarray, xx: np.ndarray, bin_w: float, bw: float = 0.15):
    """Continuous onset density (count units), wrapped across the 0/100 % seam."""
    if phase.size < 5:
        return None
    from scipy.stats import gaussian_kde
    ext = np.concatenate([phase - 100, phase, phase + 100])
    return gaussian_kde(ext, bw_method=bw)(xx) * 3 * phase.size * bin_w


def _fit_curve(m: dict, f_cps: float, xf: np.ndarray) -> np.ndarray:
    """Cosine fit at ``f_cps`` as tested (weighted if the test was weighted)."""
    ok = np.isfinite(m["binned"])
    w = 2 * np.pi * f_cps / 100.0
    Xo = np.column_stack([np.ones(ok.sum()), np.cos(w * m["centers"][ok]), np.sin(w * m["centers"][ok])])
    yo = m["binned"][ok]
    if m.get("weighted") and "bin_n" in m:
        rt = np.sqrt(m["bin_n"][ok])
        Xo, yo = Xo * rt[:, None], yo * rt
    beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
    return np.column_stack([np.ones_like(xf), np.cos(w * xf), np.sin(w * xf)]) @ beta


def _panel(ax, m: dict, ylabel: str, color) -> None:
    """Binned measure (mean ± SEM) with the confirmatory fit (solid) and, if different,
    the best grid frequency (dashed)."""
    ax.scatter(m["centers"], m["binned"], s=9, color=S.MIST, zorder=1)
    ax.errorbar(m["centers"], m["binned"], yerr=m["sem"], fmt="-", color=S.INK, lw=1,
                ecolor=S.GREY, capsize=1, zorder=2)
    xf = np.linspace(0, 100, 200)
    fc = m.get("conf_best_freq_cps", 2.0)
    p = m.get(f"p_{fc:.0f}cps_holm")
    ax.plot(xf, _fit_curve(m, fc, xf), color=color, lw=2.6, zorder=3,
            label=f"{fc:.0f} cps (confirmatory)" + (f": Holm p = {p:.3f}" if p is not None else "")
                  + (" *" if m["significant"] else ""))
    if abs(m["best_freq_cps"] - fc) > 1e-9:
        ax.plot(xf, _fit_curve(m, m["best_freq_cps"], xf), color=S.GREY, lw=1.2, ls="--",
                zorder=2, label=f"grid best {m['best_freq_cps']:.1f} cps: R² = {m['best_r2']:.2f}"
                                + (" *" if m.get("significant_grid") else ""))
    ax.set_xlabel("% stride cycle"); ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8)


_MEASURE_PANELS = (("accuracy", "Accuracy", "hit rate"),
                   ("reaction_time", "Reaction time", "log₁₀ reaction time"))


def _draw_measures(axes, measures, tags) -> None:
    """Binned measures over the stride with the tested fits."""
    for a, tag, (key, title, ylabel) in zip(axes, tags, _MEASURE_PANELS):
        if measures.get(key):
            _panel(a, measures[key], ylabel, S.MEASURE[key])
        S.panel(a, tag, title, fontsize=10)


def _draw_spectrum(ax, measures, tag: str, title: str = "R² spectrum") -> None:
    """R² over the frequency grid; dotted = 95 % permutation CI, grey line = 2 cps."""
    ax.axvline(2.0, color=S.GREY, ls="--", lw=0.9, zorder=0, label="2 cps (gait)")
    for key in ("accuracy", "reaction_time"):
        m = measures.get(key)
        if m:
            col = S.MEASURE[key]
            ax.plot(m["freqs"], m["r2"], color=col, lw=2, label=S.MEASURE_LABEL[key])
            ax.axhline(m["ci95"], color=col, ls=":", lw=1)
    ax.set_xlabel("frequency (cycles per stride)"); ax.set_ylabel("R²")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)          # room for the legend above the CI
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    S.panel(ax, tag, f"{title}  (dotted = 95% CI)", fontsize=10)


def _draw_onsets(axL, axP, ev, g) -> None:
    """Onsets on one normalised stride, hits and misses separately: over the mean
    head-height template (a) and as a rose plot with circular means (b). Uses
    ev['phase_target'] and the 200-point stride template."""
    tg, yf, troughs = g["t"], g["y_filt"], g["troughs"]
    step_foot = g["step_foot"] if "step_foot" in g.files and np.size(g["step_foot"]) else None
    tmpl, tmpl_sd = gait_mod._gait_template(tg, yf, troughs, step_foot)
    phase = ev["phase_target"].to_numpy(float)
    acc = ev["acc"].to_numpy(float)
    ok = np.isfinite(phase)
    phase, acc = phase[ok], acc[ok]
    if tmpl is None or not np.size(tmpl) or phase.size < 6:
        axL.axis("off"); axP.axis("off")
        return
    groups = (("hit", phase[acc == 1.0], S.GREEN), ("miss", phase[acc != 1.0], S.RED))
    bin_w = 100.0 / N_ONSET_BINS
    centers = np.linspace(bin_w / 2, 100 - bin_w / 2, N_ONSET_BINS)
    xx = np.linspace(0, 100, 300)
    x_tmpl = np.linspace(0, 100, tmpl.size)

    axL.fill_between(x_tmpl, tmpl - tmpl_sd, tmpl + tmpl_sd, color=S.MIST, alpha=0.7,
                     label="head height ±1 SD", zorder=1)
    axL.plot(x_tmpl, tmpl, color=S.TEAL, lw=2.4, label="mean head height", zorder=3)
    for xv in (0, 50, 100):
        axL.axvline(xv, color=S.GREY, ls=":", lw=1.0, zorder=0)
    axL.set_xlabel("% stride cycle  (0 / 50 / 100 = double support)")
    axL.set_ylabel("detrended head height (m)")
    axL.set_xlim(0, 100)
    axR = axL.twinx()
    maxc = 1
    for name, ph, col in groups:
        counts, _ = np.histogram(ph, bins=N_ONSET_BINS, range=(0, 100))
        maxc = max(maxc, int(counts.max()) if counts.size else 1)
        axR.bar(centers, counts, width=bin_w * 0.9, color=col, alpha=0.30, edgecolor="white",
                linewidth=0.5, label=f"{name} onsets", zorder=2)
        dens = _wrapped_kde(ph, xx, bin_w)
        if dens is not None:
            axR.plot(xx, dens, color=col, lw=2.2, label=f"{name} density", zorder=4)
    axR.set_ylabel("target onset count"); axR.set_ylim(0, maxc * 1.5)
    axR.grid(False)
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6,
               fontsize=7.5, frameon=False)
    S.panel(axL, "a", f"Onsets over the head-height bob  (hit vs miss; n = {phase.size})",
            fontsize=10)

    axP.set_theta_zero_location("N"); axP.set_theta_direction(-1)
    theta = centers / 100.0 * 2 * np.pi
    bars_max, labels = 1, []
    for name, ph, col in groups:
        counts, _ = np.histogram(ph, bins=N_ONSET_BINS, range=(0, 100))
        bars_max = max(bars_max, int(counts.max()) if counts.size else 1)
        axP.bar(theta, counts, width=2 * np.pi / N_ONSET_BINS * 0.95, color=col, alpha=0.45,
                edgecolor="white", linewidth=0.5, zorder=2)
    for name, ph, col in groups:
        cs = _circ_stats(ph)
        if cs:
            axP.annotate("", xy=(cs["mean_pct"] / 100.0 * 2 * np.pi, cs["R"] * bars_max),
                         xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4),
                         zorder=5)
            labels.append(f"{name} {cs['mean_pct']:.0f}%, R={cs['R']:.2f}")
    axP.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    axP.set_xticklabels(["0%", "25%", "50%", "75%"], fontsize=8)
    axP.set_yticklabels([])
    S.polar_panel(axP, "b", "Mean onset phase\n" + "  ·  ".join(labels), pad=30, fontsize=9.5)


def _figure(rec, ev, measures, g, out: Path) -> None:
    """Stride-phase sheet per block: onset distribution (a-c), R² spectrum (d), accuracy
    and RT over the stride (e-f)."""
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], hspace=0.55, wspace=0.28)
    fig.suptitle(f"Stride-phase modulation — {S.nice_label(rec.label())}  (n = {len(ev)} trials)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    _draw_onsets(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], projection="polar"), ev, g)

    a = fig.add_subplot(gs[1, 0])
    centers, counts, _ = _bin(ev["phase_target"].to_numpy(), None, N_BINS, "count")
    a.plot(centers, counts, "-", color=S.INK, lw=1)
    S.reference_line(a, y=np.nansum(counts) / N_BINS, label="uniform")
    a.legend(loc="upper right")
    a.set_xlabel("target onset (% stride)"); a.set_ylabel("targets per bin")
    S.panel(a, "c", f"Target distribution  ({N_BINS} bins)", fontsize=10)
    _draw_spectrum(fig.add_subplot(gs[1, 1]), measures, "d")
    _draw_measures([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])], measures, "ef")
    S.caption(fig, "Onsets are placed in the stride by trough interpolation (Davidson 2024). "
                   "In e-f, solid = the confirmatory fit (2 or 4 cps, Holm), dashed = the best "
                   "exploratory grid frequency; * = significant. R = mean resultant length "
                   "(0 = uniform, 1 = concentrated).",
              wrap=170, y=0.03)
    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.06, right=0.95)
    S.save(fig, out)


def _print(r: dict):
    print(f"\n=== Stride-phase modulation: {r['recording']} ===")
    print(f"  {r['n_trials']} trials, {r['n_responses']} responses")
    for k, m in r["measures"].items():
        sig = "SIGNIFICANT" if m["significant"] else "n.s."
        grid = "sig" if m.get("significant_grid") else "ns"
        print(f"  {k:16s}: CONFIRMATORY {sig} (2cps Holm p={m['p_2cps_holm']:.3f}, "
              f"4cps Holm p={m['p_4cps_holm']:.3f}) | exploratory grid best "
              f"{m['best_freq_cps']:.2f} cps R²={m['best_r2']:.2f} ({grid})")


# ===========================================================================
def _n_unphased(rec, n_phased: int) -> int:
    """Main trials without stride phase (in a head-data gap, outside the walking segment,
    or in an over-long step; see gait.py)."""
    ev = bids_io.load_events(rec)
    if ev.empty:
        return 0
    return int((ev["is_training"] == 0).sum()) - int(n_phased)


# Pooled trials over all walking participants
# ===========================================================================
def _recording_trials(rec):
    """Trial-level (phase, value) per measure for one walking recording, or None.

    Same phase assignment and RT filtering as the per-block analysis.
    """
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    troughs = g["troughs"]
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    cont_phase = continuous_phase(g)
    tp = _trial_phases(rec, troughs, step_foot, tg, cont_phase)
    if tp is None:
        return None
    ev, resp = tp
    out = {"accuracy": (ev["phase_target"].to_numpy(float), ev["acc_adj"].to_numpy(float))}
    rt = _clean_rt(ev)
    if rt:
        out["reaction_time"] = (rt[0], rt[1])
    return out


def subject_walk_label(sub: str, ses: str) -> str:
    """Label of a participant's pooled walking phase result (all walking runs), without a
    run- entity. Read by ``population.py`` and ``group_agnostic.py``."""
    return f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking"


def subject_pooled_analysis(recs) -> None:
    """Confirmatory test per participant on all walking trials pooled.

    Per-participant power is defined on the pooled blocks, not on single ~180-trial runs.
    For each (subject, session) all walking runs are concatenated and ``analyse_measure``
    is run; the result is the participant-level input for the population stages.
    """
    from collections import defaultdict
    by_sub: dict[tuple, list] = defaultdict(list)
    for rec in recs:
        if rec.condition == "walking":
            by_sub[(rec.sub, rec.ses)].append(rec)
    for (sub, ses), subrecs in sorted(by_sub.items()):
        pools = {"accuracy": ([], []), "reaction_time": ([], [])}
        used = []
        for rec in subrecs:
            tr = _recording_trials(rec)
            if not tr:
                continue
            used.append(rec)
            for k, (ph, val) in tr.items():
                pools[k][0].append(ph)
                if val is not None:
                    pools[k][1].append(val)
        measures = {}
        for k, (phs, vals) in pools.items():
            if not phs:
                continue
            m = analyse_measure(np.concatenate(phs), np.concatenate(vals), "mean")
            if m:
                measures[k] = m
        label = subject_walk_label(sub, ses)
        out_dir = bids_io.deriv_dir("phase", sub, ses)
        if not measures:
            # Remove an earlier result so population tests do not count this participant.
            for ext in ("_phase.json", "_phase.npz"):
                (out_dir / f"{label}{ext}").unlink(missing_ok=True)
            print(f"  pooled {sub} {ses}: no analysable walking trials in "
                  f"{len(subrecs)} run(s) - participant left out")
            continue
        n_trials = int(sum(len(p) for p in pools["accuracy"][0]))
        rep = {"recording": label, "sub": sub, "session": ses, "n_runs": len(used),
               "n_trials": n_trials,
               "pooled_from_runs": [r.label() for r in used],
               "runs_without_trials": [r.label() for r in subrecs if r not in used],
               "onset_phase_coverage": _circ_stats(np.concatenate(pools["accuracy"][0])),
               "measures": {k: _json_measure(v) for k, v in measures.items()}}
        (out_dir / f"{label}_phase.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        np.savez(out_dir / f"{label}_phase.npz",
                 **{f"{k}_{a}": np.asarray(measures[k][a])
                    for k in measures if measures[k]
                    for a in ("centers", "binned", "freqs", "r2",
                              # coefficients for group_agnostic.py
                              "coef_2cps", "coef_4cps",
                              "null_coef_2cps", "null_coef_4cps")})
        flagged = [k for k, m in measures.items() if m["significant"]]
        print(f"  pooled {sub} {ses}: {len(used)} walking run(s), {n_trials} trials "
              f"-> {', '.join(flagged) if flagged else 'n.s.'}")


def pooled_analysis(recs) -> dict | None:
    """Exploratory: test all walking participants' trials pooled.

    Not a population test: it ignores between-participant variance, so one participant
    can drive the result. Population inference is in group_agnostic, common_phase and
    population.py.
    """
    pools = {"accuracy": ([], []), "reaction_time": ([], [])}
    subs = set()
    for rec in recs:
        tr = _recording_trials(rec)
        if not tr:
            continue
        subs.add(rec.sub)          # each participant counted once
        for k, (ph, val) in tr.items():
            pools[k][0].append(ph)
            if val is not None:
                pools[k][1].append(val)
    n_part = len(subs)
    if n_part < 2:
        return None

    results = {}
    for k, (phs, vals) in pools.items():
        if not phs:
            continue
        m = analyse_measure(np.concatenate(phs), np.concatenate(vals), "mean")
        if m:
            results[k] = m
    if not results:
        return None

    n_trials = int(sum(len(p) for p in pools["accuracy"][0]))
    out_dir = bids_io.deriv_dir("phase")
    _pooled_figure(results, n_part, n_trials)
    payload = {"n_participants": n_part, "n_trials_pooled": n_trials,
               "note": "EXPLORATORY pooled-trials test; ignores between-participant "
                       "variance. Use the GLMM / prevalence for population inference.",
               "measures": {k: _json_measure(v) for k, v in results.items()}}
    (out_dir / "pooled_phase.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== POOLED stride-phase (EXPLORATORY: {n_part} participants, "
          f"{n_trials} trials concatenated) ===")
    for k, m in results.items():
        sig = "SIGNIFICANT" if m["significant"] else "n.s."
        print(f"  {k:16s}: best {m['best_freq_cps']:.2f} cps R²={m['best_r2']:.2f} "
              f"(CI {m['ci95']:.2f}) -> {sig}; 2cps R²={m['r2_at_2cps']:.2f}")
    print("  (NOT a population test — ignores between-subject variance; see GLMM/prevalence.)")
    return results


def _pooled_figure(results, n_part, n_trials) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"Pooled stride-phase modulation — exploratory "
                 f"(n = {n_part} participants, {n_trials} trials)",
                 fontsize=12, fontweight="semibold", x=0.012, ha="left")
    _draw_measures(ax[:2], results, "ab")
    _draw_spectrum(ax[2], results, "c")
    S.caption(fig, "Trials concatenated across participants; ignores between-subject "
                   "variance, so not a population test (see group_agnostic / GLMM).")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    S.save(fig, bids_io.fig_dir("phase") / "pooled_phase.png")


def main():
    """Run the stride-phase analysis per block, per participant and pooled."""
    recs = bids_io.find_recordings()
    if not recs:
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return
    any_walk = False
    for rec in recs:
        r = analyse(rec)
        if r:
            any_walk = True
            _print(r)
    if any_walk:
        # confirmatory test per participant on pooled walking runs
        print("\n=== Per-participant pooled walking (all runs concatenated) ===")
        subject_pooled_analysis(recs)
        pooled_analysis(recs)   # exploratory pooled test
    walk = {(r.sub, r.ses) for r in recs if r.condition == "walking"}
    n_done = sum((bids_io.deriv_dir("phase", s, e) / f"{subject_walk_label(s, e)}_phase.json")
                 .exists() for s, e in walk)
    bids_io.gate_report("phase", condition="walking", n_analysed=n_done, prune=("phase",))
    if any_walk:
        print(f"\nStride-phase outputs -> {C.DERIV_DIR / 'phase'}")
        print("NOTE: a single short run is underpowered (Davidson used ~1300 targets / "
              "participant and 36 participants); the machinery is faithful but pool data "
              "before interpreting significance.")


if __name__ == "__main__":
    main()
