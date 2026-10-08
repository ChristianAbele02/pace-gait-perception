"""Exploratory: compare stride-phase estimates on held-out signals.

gait.py stores three stride phases per walking block: trough interpolation (Davidson's
method, confirmatory), the Hilbert phase of one channel, and a multi-channel Phaser-style
phase (phaser.py; Revzen & Guckenheimer 2008). None is ground truth, so each is scored on
signals none of them uses:

* held-out signals: head pitch (head_stability.py) and fore-aft head position, both
  gait-locked. A more accurate phase gives a phase-binned mean that explains more of their
  variance (R^2, 50 bins, same samples for all methods);
* stride timing: CV of stride times from each phase (trough pairs, or completed 2*pi cycles
  of a continuous phase). Detection noise and phase slips raise the CV.

On real head data the trough phase explained as much or more held-out variance than the
continuous methods in almost every block. The multi-channel phase gave noisier stride
times (slipped cycles), because lateral sway and head roll oscillate less cleanly than the
vertical bob. On clean synthetic channels it is 2-3x more accurate than any single channel,
so the head signals are the limiting factor. The confirmatory analysis uses troughs.

Usage::

    python -m Analysis.gait_phase_methods
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
    from . import gait as gait_mod
    from . import head_stability as hs
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import gait as gait_mod
    import head_stability as hs
    import plotstyle as S

S.apply()

N_BINS = 50
METHODS = ("trough", "hilbert", "phaser")


def phase_locked_r2(phase_pct: np.ndarray, x: np.ndarray, nbins: int = N_BINS) -> float:
    """Share of the variance of ``x`` explained by its phase-binned mean."""
    ok = np.isfinite(phase_pct) & np.isfinite(x)
    if ok.sum() < nbins * 10:
        return float("nan")
    p, v = phase_pct[ok], x[ok]
    idx = np.clip((p / 100.0 * nbins).astype(int), 0, nbins - 1)
    m = np.bincount(idx, v, nbins) / np.maximum(np.bincount(idx, None, nbins), 1)
    return float(1.0 - np.var(v - m[idx]) / np.var(v))


def cycle_times(tg: np.ndarray, phase_rad: np.ndarray) -> np.ndarray:
    """Times at which an unwrapped phase completes each cycle (crosses 2*pi*k)."""
    ok = np.isfinite(phase_rad)
    k = np.floor(np.where(ok, phase_rad, 0) / (2 * np.pi))
    out = []
    for i in np.flatnonzero(ok[1:] & ok[:-1] & (np.diff(k) == 1)):
        target = (k[i] + 1) * 2 * np.pi
        out.append(tg[i] + (target - phase_rad[i]) / (phase_rad[i + 1] - phase_rad[i])
                   * (tg[i + 1] - tg[i]))
    return np.asarray(out)


def _cv(intervals: np.ndarray) -> float:
    """CV (%) of intervals, leaving out those too long to be one stride."""
    st = intervals[intervals < gait_mod.MAX_STEP_RATIO * np.median(intervals)]
    return float(100 * st.std() / st.mean()) if st.size > 10 else float("nan")


def analyse_block(rec) -> dict | None:
    """Held-out R^2 and stride-time CV of the three phase methods for one walking block.

    Scored on samples where all methods have a phase and head pitch is valid.
    """
    b = hs.block_signals(rec)
    if b is None:
        return None
    g, tg, fs = b["g"], b["tg"], b["fs"]
    if not np.size(g["stride_phase_phaser"]):
        return None
    motion, _, _ = bids_io.load_motion(rec)
    fa = None
    if "head_local_pos_z" in motion:
        z = motion["head_local_pos_z"].interpolate(limit_direction="both").to_numpy(float)
        fa = gait_mod._bandpass(detrend(np.interp(tg, motion["t"].to_numpy(float), z)),
                                fs, 0.5, gait_mod.LOWPASS_HZ)
        fa[~np.isfinite(b["y"])] = np.nan
    sf = g["step_foot"] if np.size(g["step_foot"]) else None
    ph = {"trough": gait_mod.phase_at(tg, g["troughs"], sf)[1],
          "hilbert": gait_mod.phase_at_hilbert(tg, tg, g["stride_phase_cont"])[1],
          "phaser": gait_mod.phase_at_hilbert(tg, tg, g["stride_phase_phaser"])[1]}
    common = np.all([np.isfinite(v) for v in ph.values()], axis=0) & np.isfinite(b["pitch"])
    row = {"recording": rec.label(), "sub": rec.sub, "n_samples": int(common.sum())}
    for m, p in ph.items():
        pm = np.where(common, p, np.nan)
        row[f"r2_pitch_{m}"] = round(phase_locked_r2(pm, b["pitch"]), 4)
        row[f"r2_foreaft_{m}"] = round(phase_locked_r2(pm, fa), 4) if fa is not None else None
    tr = np.asarray(g["troughs"], float)
    row["stride_cv_trough"] = round(_cv(np.diff(tr[::2])), 3)
    for m, key in (("hilbert", "stride_phase_cont"), ("phaser", "stride_phase_phaser")):
        row[f"stride_cv_{m}"] = round(_cv(np.diff(cycle_times(tg, g[key]))), 3)
    return row


def _figure(df: pd.DataFrame, out: Path) -> None:
    """Plot each score across methods, one line per block."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(METHODS))
    col = {m: c for m, c in zip(METHODS, (S.INK, S.BLUE, S.ORANGE))}
    for a, (pref, lab, tag, title) in zip(ax, (
            ("r2_pitch_", "phase-locked R² of head pitch", "a", "Held out: head pitch"),
            ("r2_foreaft_", "phase-locked R² of fore-aft head", "b", "Held out: fore-aft"),
            ("stride_cv_", "stride-time CV (%)", "c", "Stride timing noise"))):
        for _, r in df.iterrows():
            v = [r.get(f"{pref}{m}") for m in METHODS]
            a.plot(x, v, color=S.MIST, lw=1, zorder=1)
        for i, m in enumerate(METHODS):
            a.scatter(np.full(len(df), i), df[f"{pref}{m}"], color=col[m], s=28, zorder=3)
        a.set_xticks(x)
        a.set_xticklabels(METHODS)
        a.set_ylabel(lab)
        S.panel(a, tag, title, fontsize=10)
    fig.suptitle("Stride-phase methods scored on signals none of them uses (exploratory)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "One line per walking block. Higher R² = the phase lines the held-out "
                   "signal up better across strides; lower stride CV = less detection noise "
                   "or fewer phase slips. trough = Davidson's method (confirmatory).")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    S.save(fig, out)


def main() -> pd.DataFrame:
    """Score all walking blocks; write a TSV, a summary JSON and a figure."""
    rows = [r for r in (analyse_block(rec) for rec in bids_io.find_recordings(scope="gait"))
            if r]
    bids_io.gate_report("phase_methods", "gait", condition="walking",
                        n_analysed=len({r["sub"] for r in rows}))
    if not rows:
        print("No walking blocks with gait output. Run gait.py first.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = bids_io.deriv_dir("gait_phase_methods")
    df.to_csv(out / "phase_methods.tsv", sep="\t", index=False, na_rep="n/a")
    _figure(df, bids_io.fig_dir("gait_phase_methods") / "phase_methods.png")
    wins = {m: int(sum(df[[f"r2_pitch_{k}" for k in METHODS]].idxmax(axis=1)
                       == f"r2_pitch_{m}")) for m in METHODS}
    print(df.to_string(index=False))
    print(f"\nBest held-out pitch R² per block: {wins}")
    (out / "phase_methods_summary.json").write_text(json.dumps(
        {"best_pitch_r2_counts": wins, "n_blocks": len(df)}, indent=2), encoding="utf-8")
    print(f"Phase-method comparison -> {out}")
    return df


if __name__ == "__main__":
    main()
