"""Exploratory: the group-average-curve analysis, run on our data and calibrated.

A common way to test stride-phase modulation in the walking literature fits one cosine to
the group-average curve and tests each frequency against a resampling null of that
curve's bins. This module runs that procedure on our walking trials, so its verdict can be
set beside ours, and measures how often it calls a modulation when there is none.

The procedure, per measure:

1. Every non-training trial gets the stride percentile (1-100) of its target onset (the
   press, for response likelihood).
2. Per participant, the measure is averaged within each percentile; the group value of a
   percentile is the mean over participants. Averaging into 20 equal bins gives the
   group curve.
3. ``a0 + a1 cos(w x) + b1 sin(w x)`` is fitted to the 20 bins at every frequency of the
   0.1-10 cps grid (step 0.1), keeping R^2.
4. Null: the 100 group percentile values are resampled with replacement, re-binned into
   20 and refitted, 1000 times. A frequency is significant when its R^2 exceeds the 95th
   percentile of its own null. There is no correction across frequencies or measures; a
   significant fit within 1.5-2.5 cps counts as a step-rate modulation, and the best
   fitting significant frequency is reported.
5. Individual level: the same fit and null on each participant's own curve; the
   participants significant at exactly 2 cps are counted and their peak phases averaged.

Measures: accuracy over all presented targets (a timeout scores 0, the definition this
procedure is usually run with), accuracy over answered trials (ours), response rate, RT in
seconds of the answered trials, and response likelihood (the share of presses falling in
each percentile). RT and accuracy are not difficulty-adjusted, as in the original procedure.

What differs from our tests, and what the calibration measures:

* The group curve can only show a modulation that participants share in phase;
  idiosyncratic phases cancel in the average. Our confirmatory ``S`` asks whether each
  participant oscillates at their own phase; the common-phase question is
  ``common_phase``'s Hotelling T^2.
* The null resamples bins of the averaged curve, not trials, so it carries no
  information about how many trials each bin holds.
* Each of 100 frequencies is tested at 5 % without correction, and an 11-frequency window
  is searched. ``calibrate`` measures the resulting false-positive rate on our own design:
  in each null dataset the outcomes are shuffled against the onset phases within every
  block (everything else kept), and the full procedure is rerun.
* Response likelihood is a raw press histogram: when the targets fell and how RTs spread
  shape it whatever the gait does. ``press_phase`` tests it against an RT-shuffle null.

The cosine fit and frequency grid are those of Davidson et al. (2024), the group-curve
resampling as described above. Not confirmatory: a side-by-side comparison only.

Usage::

    python -m Analysis.group_curve
"""
from __future__ import annotations

import json
import sys
import warnings
import zlib
from collections import defaultdict
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
    from . import press_phase as pp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import phase_analysis as pa
    import plotstyle as S
    import press_phase as pp

S.apply()

N_PCT, N_BINS = 100, 20
FREQS = np.round(np.arange(0.1, 10.0001, 0.1), 1)
WINDOW = (1.5, 2.5)
N_RES = 1000
N_CAL = 500
ALPHA = 0.05
MEASURES = {                  # name -> (label, our matching test in derivatives/population)
    "accuracy_all": ("accuracy, all targets (timeout = 0)", None),
    "accuracy": ("accuracy, answered trials", "accuracy"),
    "response_rate": ("response rate", "timeouts"),
    "rt": ("RT (s), answered trials", "reaction_time"),
    "response_likelihood": ("response likelihood (press histogram)", "press_phase"),
}
TRIAL_MEASURES = ("accuracy_all", "accuracy", "response_rate", "rt")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _block(rec, gait_out) -> dict | None:
    """Trial measures and press phases of one walking block in the gait scope, or None."""
    if rec.condition != "walking" or rec.label() in gait_out:
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if np.size(g["step_foot"]) else None
    tg = g["t"] if np.size(g["t"]) else None
    gait = (np.asarray(g["troughs"], float), step_foot, tg, pa.continuous_phase(g))
    tp = pa._trial_phases(rec, *gait)
    if tp is None:
        return None
    ev, _ = tp
    answered = ev["response"].notna().to_numpy()
    rt = ev["rt"].to_numpy(float)
    correct = (ev["response_correct"].to_numpy(float) == 1) & answered
    rt_ok = answered & np.isfinite(rt) & (rt > pa.RT_MIN_S)
    press = pp.press_phases(ev["onset"].to_numpy(float)[rt_ok], rt[rt_ok], gait)
    return {"phase": ev["phase_target"].to_numpy(float),
            "accuracy_all": correct.astype(float),
            "accuracy": np.where(answered, correct.astype(float), np.nan),
            "response_rate": answered.astype(float),
            "rt": np.where(rt_ok, rt, np.nan),
            "press_phase": press[np.isfinite(press)]}


def _pct(phase: np.ndarray) -> np.ndarray:
    """Stride percentile (0-99) of each phase in %."""
    return np.clip(np.floor(phase).astype(int), 0, N_PCT - 1)


def percentile_means(phase: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean of ``y`` per stride percentile (NaN for an empty one)."""
    ok = np.isfinite(phase) & np.isfinite(y)
    i = _pct(phase[ok])
    n = np.bincount(i, minlength=N_PCT)
    s = np.bincount(i, y[ok], N_PCT)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def press_shares(press_phase: np.ndarray) -> np.ndarray:
    """Share of a participant's presses falling in each stride percentile."""
    n = np.bincount(_pct(press_phase), minlength=N_PCT).astype(float)
    return n / max(n.sum(), 1.0)


def to_bins(pct: np.ndarray) -> np.ndarray:
    """``(..., 100)`` percentile values -> ``(..., 20)`` bin means (empty percentiles skipped)."""
    with warnings.catch_warnings():                 # an all-empty bin is filled below
        warnings.simplefilter("ignore", RuntimeWarning)
        b = np.nanmean(pct.reshape(*pct.shape[:-1], N_BINS, N_PCT // N_BINS), -1)
    m = np.nanmean(b, -1, keepdims=True)
    return np.where(np.isfinite(b), b, m)


# ---------------------------------------------------------------------------
# Fit and null
# ---------------------------------------------------------------------------
_X = (np.arange(N_BINS) + 0.5) * 100.0 / N_BINS


def _hats() -> np.ndarray:
    """``(n_freq, 20, 20)`` least-squares hat matrices of the raised cosine per frequency."""
    out = []
    for f in FREQS:
        w = 2 * np.pi * f / 100.0
        X = np.column_stack([np.ones(N_BINS), np.cos(w * _X), np.sin(w * _X)])
        out.append(X @ np.linalg.pinv(X))
    return np.stack(out)


HATS = _hats()


def r2_spectrum(curves: np.ndarray) -> np.ndarray:
    """R^2 of the raised cosine at every grid frequency, ``(..., 20) -> (..., n_freq)``."""
    y = np.asarray(curves, float)
    yc = y - y.mean(-1, keepdims=True)
    fit = np.einsum("fij,...j->...fi", HATS, y)
    ss_res = ((y[..., None, :] - fit) ** 2).sum(-1)
    ss_tot = (yc ** 2).sum(-1)[..., None]
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-300)


def peak_pct(curve: np.ndarray, freq: float) -> float:
    """Where in the stride (%) the fitted cosine at ``freq`` peaks, within its first cycle."""
    w = 2 * np.pi * freq / 100.0
    X = np.column_stack([np.ones(N_BINS), np.cos(w * _X), np.sin(w * _X)])
    _, a, b = np.linalg.lstsq(X, curve, rcond=None)[0]
    return float((np.arctan2(b, a) / w) % (100.0 / freq))


def bin_resample_null(pct: np.ndarray, rng, n_res: int = N_RES) -> np.ndarray:
    """R^2 spectra of ``n_res`` curves made from the percentile values drawn with
    replacement: ``(n_res, n_freq)``."""
    idx = rng.integers(0, N_PCT, (n_res, N_PCT))
    return r2_spectrum(to_bins(pct[idx]))


def verdict(pct: np.ndarray, rng, n_res: int = N_RES) -> dict:
    """The procedure on one percentile curve: significant frequencies, the best of them,
    whether one falls in the step-rate window, and the fit at exactly 2 cps."""
    curve = to_bins(pct)
    r2 = r2_spectrum(curve)
    crit = np.percentile(bin_resample_null(pct, rng, n_res), 95, axis=0)
    sig = r2 > crit
    win = (FREQS >= WINDOW[0] - 1e-9) & (FREQS <= WINDOW[1] + 1e-9)
    i2 = int(np.argmin(np.abs(FREQS - 2.0)))
    best = int(np.argmax(np.where(sig, r2, -np.inf))) if sig.any() else None
    return {"r2": r2, "crit": crit, "curve": curve,
            "n_significant_freqs": int(sig.sum()),
            "best_sig_freq": (float(FREQS[best]) if best is not None else None),
            "best_sig_r2": (float(r2[best]) if best is not None else None),
            "window_hit": bool((sig & win).any()),
            "window_freqs": [float(f) for f in FREQS[sig & win]],
            "r2_2cps": float(r2[i2]), "crit_2cps": float(crit[i2]), "sig_2cps": bool(sig[i2]),
            "peak_pct_2cps": peak_pct(curve, 2.0)}


# ---------------------------------------------------------------------------
# Participants -> group
# ---------------------------------------------------------------------------
def participant_tables(blocks: list[dict]) -> dict:
    """Percentile tables of every measure for one participant (blocks pooled)."""
    ph = np.concatenate([b["phase"] for b in blocks])
    out = {m: percentile_means(ph, np.concatenate([b[m] for b in blocks]))
           for m in TRIAL_MEASURES}
    out["response_likelihood"] = press_shares(np.concatenate([b["press_phase"] for b in blocks]))
    return out


def group_pct(tables: list[dict], measure: str) -> np.ndarray:
    """Group value per percentile: the mean of the participants' percentile means."""
    with warnings.catch_warnings():                 # a percentile empty in everyone stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack([t[measure] for t in tables]), 0)


def _shuffled(blocks: list[dict], measure: str, rng) -> list[dict]:
    """Blocks with ``measure`` permuted against the onset phases within each block, among
    the trials where it is defined (a true null with everything else kept)."""
    out = []
    for b in blocks:
        y = b[measure].copy()
        ok = np.flatnonzero(np.isfinite(y))
        y[ok] = y[rng.permutation(ok)]
        out.append({"phase": b["phase"], measure: y})
    return out


def calibrate(subject_blocks: list[list[dict]], measure: str, rng, *, n_cal: int = N_CAL,
              n_res: int = N_RES) -> dict:
    """False-positive rates of the procedure on null versions of our own data."""
    hits = {"window": 0, "at_2cps": 0, "any_freq": 0}
    n_sig = []
    for _ in range(n_cal):
        tabs = []
        for blocks in subject_blocks:
            sh = _shuffled(blocks, measure, rng)
            ph = np.concatenate([b["phase"] for b in sh])
            tabs.append({measure: percentile_means(ph, np.concatenate([b[measure] for b in sh]))})
        v = verdict(group_pct(tabs, measure), rng, n_res)
        hits["window"] += v["window_hit"]
        hits["at_2cps"] += v["sig_2cps"]
        hits["any_freq"] += v["n_significant_freqs"] > 0
        n_sig.append(v["n_significant_freqs"])
    return {"n_null_datasets": n_cal,
            "fpr_window_1p5_2p5": hits["window"] / n_cal,
            "fpr_at_2cps": hits["at_2cps"] / n_cal,
            "fpr_any_frequency": hits["any_freq"] / n_cal,
            "mean_significant_freqs": float(np.mean(n_sig))}


def individual(tables: list[dict], subs: list[str], rng) -> dict:
    """The individual-level step: per participant, accuracy curve against its own
    bin-resampling null; who is significant at exactly 2 cps, and where they peak."""
    rows = []
    for s, t in zip(subs, tables):
        v = verdict(t["accuracy_all"], rng)
        rows.append({"sub": s, "sig_2cps": v["sig_2cps"], "r2_2cps": v["r2_2cps"],
                     "crit_2cps": v["crit_2cps"], "peak_pct_2cps": v["peak_pct_2cps"]})
    sig = [r for r in rows if r["sig_2cps"]]
    out = {"participants": rows, "n_significant_2cps": len(sig)}
    if sig:
        ang = 2 * np.pi * np.array([r["peak_pct_2cps"] for r in sig]) / 50.0
        z = np.mean(np.exp(1j * ang))
        out["mean_peak_pct_of_stride_significant"] = float((np.angle(z) * 50.0 / (2 * np.pi)) % 50.0)
        out["resultant_R_significant"] = float(abs(z))
    return out


# ---------------------------------------------------------------------------
# Our results, for the side-by-side
# ---------------------------------------------------------------------------
def _ours() -> dict:
    """Holm p-values of our phase-agnostic S and common-phase T^2 at 2 and 4 cps."""
    out = {}
    try:
        ag = json.loads((C.DERIV_DIR / "population" / "population_agnostic.json").read_text("utf-8"))
        cp = json.loads((C.DERIV_DIR / "population" / "population_common_phase.json")
                        .read_text("utf-8"))["measures"]
        for m in ("accuracy", "reaction_time", "timeouts"):
            if m in ag:
                out[m] = {f: {"S_p_holm": ag[m]["by_freq"][f].get("p_holm"),
                              "T2_p_holm": cp.get(m, {}).get("by_freq", {}).get(f, {}).get("p_holm")}
                          for f in ("2cps", "4cps")}
    except (OSError, KeyError, ValueError):
        pass
    try:
        pr = json.loads((C.DERIV_DIR / "press_phase" / "press_phase_population.json")
                        .read_text("utf-8"))["population"]
        out["press_phase"] = {f: {"S_p_holm": pr[f]["agnostic"].get("p_holm"),
                                  "T2_p_holm": pr[f]["hotelling"].get("p_holm")}
                              for f in ("2cps", "4cps")}
    except (OSError, KeyError, ValueError):
        pass
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _figure(obs: dict, tables: list[dict], cal: dict, out: Path) -> None:
    """Figure: the group curve with its best significant fit, the R² spectra
    against their null percentiles, and the calibration on null data."""
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1.0]})
    m = "accuracy_all"
    v = obs[m]
    cur = 100 * np.vstack([to_bins(t[m]) for t in tables])
    se = cur.std(0, ddof=1) / np.sqrt(cur.shape[0])
    ax[0].bar(_X, 100 * v["curve"], width=100 / N_BINS * 0.9, color=S.MIST,
              yerr=se, error_kw={"lw": 0.8, "ecolor": S.GREY})
    lo = np.nanmin(100 * v["curve"] - se)
    ax[0].set_ylim(lo - 3, np.nanmax(100 * v["curve"] + se) + 3)
    if v["best_sig_freq"] is not None:
        f = v["best_sig_freq"]
        w = 2 * np.pi * f / 100.0
        X = np.column_stack([np.ones(N_BINS), np.cos(w * _X), np.sin(w * _X)])
        beta = np.linalg.lstsq(X, 100 * v["curve"], rcond=None)[0]
        xf = np.linspace(0, 100, 300)
        ax[0].plot(xf, beta[0] + beta[1] * np.cos(w * xf) + beta[2] * np.sin(w * xf),
                   color=S.RED if v["window_hit"] else S.INK, lw=2.0,
                   label=f"best significant fit, {f:.1f} cps, R² = {v['best_sig_r2']:.2f}")
        ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_xlabel("% of stride at target onset (0, 50 = double support)")
    ax[0].set_ylabel("accuracy, all targets (%)")
    S.panel(ax[0], "a", "Group-average curve, 20 bins", fontsize=10)

    for mm, col in (("accuracy_all", S.INK), ("rt", S.TEAL), ("response_likelihood", S.RED)):
        ax[1].plot(FREQS, obs[mm]["r2"], color=col, lw=1.6, label=MEASURES[mm][0])
        ax[1].plot(FREQS, obs[mm]["crit"], color=col, lw=0.9, ls="--")
    ax[1].axvspan(*WINDOW, color=S.MIST, alpha=0.5, lw=0)
    ax[1].set_xlabel("frequency (cycles per stride)")
    ax[1].set_ylabel("R² (solid) and null 95th pct (dashed)")
    ax[1].legend(fontsize=7.5, loc="upper right")
    S.panel(ax[1], "b", "Fit strength per frequency", fontsize=10)

    names = [k for k in TRIAL_MEASURES if k in cal]
    yv = np.arange(len(names))[::-1]
    for key, off, col, lab in (("fpr_at_2cps", 0.27, S.GREY, "significant at 2.0 cps"),
                               ("fpr_window_1p5_2p5", 0.0, S.RED, "significant in 1.5-2.5 cps"),
                               ("fpr_any_frequency", -0.27, S.INK, "significant anywhere")):
        ax[2].barh(yv + off, [100 * cal[n][key] for n in names], height=0.25, color=col, label=lab)
    ax[2].axvline(100 * ALPHA, color=S.GREY, ls="--", lw=1.0)
    ax[2].set_yticks(yv)
    ax[2].set_yticklabels([MEASURES[n][0].replace(", ", "\n", 1) for n in names], fontsize=8)
    ax[2].set_xlabel("false-positive rate on null data (%)")
    ax[2].set_xlim(0, 100)
    ax[2].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=1,
                 frameon=False)
    ax[2].grid(axis="y", visible=False)
    S.panel(ax[2], "c", "Calibration on our design", fontsize=10)

    fig.suptitle(f"Group-average-curve analysis on our walking data, n = {len(tables)}",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "EXPLORATORY comparison. a: group mean of the participants' percentile means "
                   "in 20 bins, ±1 SEM over participants; line, the best-fitting frequency whose "
                   "R² exceeds its own bin-resampling 95th percentile (red if in 1.5-2.5 cps). "
                   "b: R² of the raised cosine at every grid frequency and its null 95th "
                   "percentile; grey band, the 1.5-2.5 cps window. c: how often the procedure "
                   f"reports a modulation when outcomes are shuffled within blocks ({N_CAL} null "
                   "datasets); dashed, the nominal 5 %.")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    S.save(fig, out)


def _safe(o):
    """JSON-serialisable copy of a result, without the per-frequency arrays."""
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items() if k not in ("curve", "crit", "r2")}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main() -> dict | None:
    """Run the group-average-curve procedure, its calibration, and the side-by-side."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    gait_out = bids_io.excluded_recordings(scope="gait")
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[r.sub].append(r)
    subs, subject_blocks = [], []
    for sub, rr in sorted(by_sub.items()):
        bl = [b for b in (_block(r, gait_out) for r in rr) if b]
        if bl:
            subs.append(sub)
            subject_blocks.append(bl)
    bids_io.gate_report("group_curve", condition="walking", analysed_subjects=subs)
    out_dir = bids_io.deriv_dir("group_curve")
    if len(subs) < 2:
        bids_io.clear_stale("group_curve", "group_curve.json", "group_curve.png")
        return None
    tables = [participant_tables(bl) for bl in subject_blocks]
    rng = np.random.default_rng(zlib.crc32(b"group_curve"))
    obs = {m: verdict(group_pct(tables, m), rng) for m in MEASURES}
    print("=== Group-average-curve analysis (EXPLORATORY comparison) ===")
    for m, v in obs.items():
        print(f"  {MEASURES[m][0]:<40} best significant {v['best_sig_freq']} cps | "
              f"window hit {v['window_hit']} {v['window_freqs']} | 2 cps R2 {v['r2_2cps']:.3f} "
              f"vs crit {v['crit_2cps']:.3f} | {v['n_significant_freqs']} significant freqs")
    ind = individual(tables, subs, rng)
    print(f"  individual level (accuracy, all targets): {ind['n_significant_2cps']}/{len(subs)} "
          "significant at 2 cps")
    cal = {}
    for m in TRIAL_MEASURES:
        cal[m] = calibrate(subject_blocks, m, np.random.default_rng(zlib.crc32(f"cal_{m}".encode())))
        print(f"  calibration {m}: FPR 2 cps {cal[m]['fpr_at_2cps']:.3f}, window "
              f"{cal[m]['fpr_window_1p5_2p5']:.3f}, any frequency {cal[m]['fpr_any_frequency']:.3f}")
    ours = _ours()
    payload = {"note": "EXPLORATORY comparison: the group-average-curve analysis with a "
                       "bin-resampling null per frequency, run on our walking trials, its "
                       "false-positive rate on null versions of our data, and our tests' "
                       "Holm p-values beside it.",
               "n_participants": len(subs), "frequency_window_cps": list(WINDOW),
               "n_resamples": N_RES, "observed": _safe(obs), "individual_level": _safe(ind),
               "calibration": cal,
               "ours": ours,
               "measure_map": {m: MEASURES[m][1] for m in MEASURES}}
    (out_dir / "group_curve.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _figure(obs, tables, cal, bids_io.fig_dir("group_curve") / "group_curve.png")
    print(f"Outputs -> {out_dir}")
    return payload


if __name__ == "__main__":
    main()
