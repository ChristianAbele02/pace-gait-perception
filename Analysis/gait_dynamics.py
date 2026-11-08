"""Exploratory: the walking rhythm itself.

* Stride-to-stride dynamics: stride-time variability, DFA and local dynamic stability.
* Phase response to task events: whether onsets or button presses shift the gait rhythm.

Usage::

    python -m Analysis.gait_dynamics


Stride-to-stride gait dynamics
==============================

Exploratory: stride-to-stride dynamics of treadmill walking in the CAVE.

Three standard measures per walking block, from the head-derived gait events and head
motion (gait.py):

* Variability: mean stride time and its coefficient of variation (CV %).
* Long-range correlations: detrended fluctuation analysis (DFA; Peng et al. 1995) of the
  stride intervals. Healthy stride intervals show fractal correlations over hundreds of
  strides (Hausdorff et al. 1995): alpha above 0.5 (uncorrelated noise) and different from
  a shuffled series. Treadmill walking alters these dynamics compared with overground
  walking (Terrier & Dériaz 2011); here they describe treadmill walking in a CAVE during a
  visual task. Box sizes 16 .. N/4 strides, fixed because alpha depends on box range and
  series length (Damouras et al. 2010). Tracking gaps break the series: each unbroken run
  of at least ``MIN_RUN`` strides is integrated separately and all runs contribute boxes to
  one fluctuation function, so no stride spans a gap. A shuffled control (alpha ~ 0.5) is
  also reported.
* Local dynamic stability: short-term largest Lyapunov exponent lambda_S (Rosenstein et
  al. 1993), applied to gait by Dingwell & Cusumano (2000), reviewed by Bruijn et al.
  (2013). State space: vertical and lateral head velocity and vertical acceleration (no
  fore-aft axis in the gait file), normalised to 100 samples per stride, delay-embedded
  (m = 5, delay = 10 samples = 0.1 stride). lambda_S is the slope of mean log
  nearest-neighbour divergence over the first half stride, in 1/stride; larger means less
  locally stable. Head kinematics are not comparable in absolute terms to trunk studies;
  the measure is for comparing blocks and visits.

Blocks are ordered by ``acq_time`` to show time-on-task and visit trends.

Caveats:

* Stride times come from head-bob troughs, not foot contact. The white detection noise
  inflates CV and pulls alpha toward 0.5. Compare within participants, not with
  foot-sensor studies.
* A missed trough or tracking gap ends a run. Short runs make alpha unstable, and some
  blocks have no run long enough. Interpret alpha only if the shuffled control is near 0.5.
* lambda_S depends on the head data sampling rate; blocks logged at a few fps and at 60 Hz
  are not comparable.

References
----------
Peng C-K, Havlin S, Stanley HE, Goldberger AL (1995). Quantification of scaling exponents
    and crossover phenomena in nonstationary heartbeat time series. Chaos 5:82-87.
Hausdorff JM, Peng C-K, Ladin Z, Wei JY, Goldberger AL (1995). Is walking a random walk?
    Evidence for long-range correlations in stride interval of human gait.
    J Appl Physiol 78:349-358.
Terrier P, Dériaz O (2011). Kinematic variability, fractal dynamics and local dynamic
    stability of treadmill walking. J NeuroEng Rehabil 8:12.
Damouras S, Chang MD, Sejdić E, Chau T (2010). An empirical examination of detrended
    fluctuation analysis for gait data. Gait Posture 31:336-340.
Rosenstein MT, Collins JJ, De Luca CJ (1993). A practical method for calculating largest
    Lyapunov exponents from small data sets. Physica D 65:117-134.
Dingwell JB, Cusumano JP (2000). Nonlinear time series analysis of normal and
    pathological human walking. Chaos 10:848-863.
Bruijn SM, Meijer OG, Beek PJ, van Dieën JH (2013). Assessing the stability of human
    locomotion: a review of current measures. J R Soc Interface 10:20120999.


Gait phase response to task events
==================================

Exploratory: do stimulus onsets or button presses shift the walking rhythm?

The main hypothesis is that the gait cycle modulates perception; this tests the reverse.
Walking behaves like a limit-cycle oscillator, whose response to a brief input depends on
the phase at which it arrives (phase response curve, PRC). Funato et al. (2016) measured
the walking PRC with brief belt perturbations and found phase advances or delays depending
on cycle phase. Here the inputs are task events (Gabor onset, button press), not mechanical
perturbations. We found no prior study of the gait PRC to perceptual or manual events.

Per event (time ``te``, in step ``i`` between troughs ``t_i <= te < t_{i+1}``):

* pre-event step period ``T0 = (t_i - t_{i-4}) / 4`` (four steps, both feet, so
  left/right asymmetry cancels);
* phase shift after ``k`` steps, ``shift_k = (t_{i+k} - t_i) - k*T0``, in % of a step.
  Negative = later steps came early (advance), positive = late (delay). The reported
  shift uses ``k = 4`` (two strides). k must be even: with odd k the window has one more
  step of one foot, so left/right asymmetry makes the shift depend on stride phase even
  at random times. ``k = 1..6`` gives the time course: a persisting shift is a phase
  reset, a fading one is transient;
* the event's stride phase (trough method, 0-100 %), the PRC x axis.

Events whose window (4 steps before, 6 after) contains a gap or an over-long step
(``gait.MAX_STEP_RATIO``) are dropped.

Null: the same measure at ``N_SURR`` random times per event, drawn from the block's logged
frame times. This captures the stride-to-stride scatter of ``shift`` without input. Frame
times are used because events only occur on logged frames, and at low frame rates (e.g.
~6 fps) trough times are biased toward frame times; a continuous-time null would
differ from events for that reason alone. Tests:

* mean shift: event mean minus random-time mean, against random sets of the same size
  (two-sided);
* PRC shape: variance explained by a 2-harmonic fit of shift on stride phase, against the
  same fit on random sets of the same size.

Onset and press are not independent: the press follows ~0.8 s later, inside the onset's
window, so an onset effect may be a press effect and vice versa. Timeouts (onset without
press) are too few per block to separate them.

Both a continuous-time null and an odd-step window could fake an onset effect; both are
avoided above.

References
----------
Funato T, Yamamoto Y, Aoi S, et al. (2016). Evaluation of the phase-dependent rhythm
    control of human walking using phase response curves. PLoS Comput Biol 12:e1004950.
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
from scipy.spatial import cKDTree

try:
    from . import bids_io
    from . import gait as gait_mod
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import gait as gait_mod
    import plotstyle as S

S.apply()


# ###########################################################################
# Stride-to-stride gait dynamics
# ###########################################################################
MIN_RUN = 32               # strides; shorter runs are not used for DFA
DFA_MIN_BOX = 16           # strides
LYAP_POINTS = 100          # samples per stride after time normalisation
LYAP_EMBED, LYAP_DELAY = 5, 10
LYAP_FIT = (0, 50)         # samples (first half stride)
LYAP_MAX_STRIDES = 400     # cap per block


# ---------------------------------------------------------------------------
# Stride series
# ---------------------------------------------------------------------------
def stride_runs(troughs: np.ndarray) -> list[np.ndarray]:
    """Unbroken runs of stride intervals (s).

    A stride is two steps. A step longer than ``gait.MAX_STEP_RATIO`` x the median (gap or
    missed trough) ends a run. Strides use every other trough, so they share no step.
    """
    tr = np.asarray(troughs, float)
    if tr.size < 5:
        return []
    step = np.diff(tr)
    good = step <= gait_mod.MAX_STEP_RATIO * np.median(step)
    runs, start = [], 0
    for i in range(step.size + 1):
        if i == step.size or not good[i]:
            seg = tr[start:i + 1]                       # troughs of this run
            if seg.size >= 3:
                runs.append(np.diff(seg[::2]))
            start = i + 1
    return [r for r in runs if r.size >= 2]


def dfa(runs: list[np.ndarray], min_box: int = DFA_MIN_BOX) -> dict | None:
    """DFA-1 over several runs: each run integrated separately, boxes pooled per scale."""
    runs = [np.asarray(r, float) for r in runs if len(r) >= MIN_RUN]
    if not runs:
        return None
    n_max = max(len(r) for r in runs) // 4
    if n_max < min_box * 2:
        return None
    mu = np.mean(np.concatenate(runs))
    prof = [np.cumsum(r - mu) for r in runs]
    scales = np.unique(np.round(np.logspace(np.log10(min_box), np.log10(n_max), 12)).astype(int))
    F = []
    for n in scales:
        res = []
        for y in prof:
            for k in range(len(y) // n):
                seg = y[k * n:(k + 1) * n]
                x = np.arange(n)
                c = np.polyfit(x, seg, 1)
                res.append(np.mean((seg - np.polyval(c, x)) ** 2))
        F.append(np.sqrt(np.mean(res)) if res else np.nan)
    F = np.asarray(F)
    ok = np.isfinite(F) & (F > 0)
    if ok.sum() < 3:
        return None
    alpha = float(np.polyfit(np.log10(scales[ok]), np.log10(F[ok]), 1)[0])
    return {"alpha": alpha, "scales": scales.tolist(), "F": F.tolist(),
            "n_strides": int(sum(len(r) for r in runs)), "n_runs": len(runs)}


# ---------------------------------------------------------------------------
# Local dynamic stability
# ---------------------------------------------------------------------------
def _time_normalise(tg, sig, troughs, runs_idx):
    """Resample each stride of each unbroken run to LYAP_POINTS samples."""
    out = []
    for a, b in runs_idx:                       # first and last trough index of a run
        tr = troughs[a:b + 1:2]
        segs = []
        for s0, s1 in zip(tr[:-1], tr[1:]):
            m = (tg >= s0) & (tg <= s1)
            if m.sum() < 10 or not np.all(np.isfinite(sig[m])):
                segs = None
                break
            tt = np.linspace(s0, s1, LYAP_POINTS, endpoint=False)
            segs.append(np.column_stack([np.interp(tt, tg[m], sig[m, j])
                                         for j in range(sig.shape[1])]))
        if segs:
            out.append(np.vstack(segs))
    return out


def lyapunov_short(series: list[np.ndarray]) -> dict | None:
    """Rosenstein short-term exponent (1/stride) over several time-normalised runs."""
    emb, lab = [], []
    span = (LYAP_EMBED - 1) * LYAP_DELAY
    for k, x in enumerate(series):
        x = (x - x.mean(0)) / x.std(0)
        n = x.shape[0] - span
        if n <= 2 * LYAP_POINTS:
            continue
        e = np.hstack([x[i * LYAP_DELAY:i * LYAP_DELAY + n] for i in range(LYAP_EMBED)])
        emb.append(e)
        lab.append(np.column_stack([np.full(n, k), np.arange(n)]))
    if not emb:
        return None
    E, L = np.vstack(emb), np.vstack(lab)
    horizon = LYAP_POINTS
    n_all = E.shape[0]
    _, nn = cKDTree(E).query(E, k=12, workers=-1)   # all cores
    ii = np.arange(horizon)
    div = np.zeros(horizon)
    cnt = np.zeros(horizon)
    for i in range(n_all):
        run, pos = L[i]
        if i + horizon >= n_all or L[i + horizon - 1, 0] != run:
            continue                              # trajectory of i leaves its run
        for j in nn[i, 1:]:
            # nearest neighbour more than one stride away in time (or in another run)
            # whose trajectory stays within its run
            if L[j, 0] == run and abs(L[j, 1] - pos) < LYAP_POINTS:
                continue
            if j + horizon >= n_all or L[j + horizon - 1, 0] != L[j, 0]:
                continue
            d = np.linalg.norm(E[i + ii] - E[j + ii], axis=1)
            good = d > 0
            div[good] += np.log(d[good])
            cnt[good] += 1
            break
    ok = cnt > 0
    if ok.sum() < LYAP_FIT[1]:
        return None
    curve = np.where(ok, div / np.maximum(cnt, 1), np.nan)
    x = np.arange(horizon) / LYAP_POINTS          # in strides
    a, b = LYAP_FIT
    lam = float(np.polyfit(x[a:b], curve[a:b], 1)[0])
    return {"lambda_s_per_stride": lam, "curve": curve.tolist(), "n_points": int(E.shape[0])}


def _head_velocity(g) -> np.ndarray:
    """State-space signal on the gait grid from the band-passed channels.

    Columns: vertical and lateral head velocity, vertical acceleration. NaN where gait.py
    bridged a gap."""
    tg = g["t"]
    fs = 1.0 / np.median(np.diff(tg))
    y = np.asarray(g["y_filt"], float)
    x = np.asarray(g["x_filt"], float) if np.size(g["x_filt"]) else np.zeros_like(y)
    vy, vx = np.gradient(y) * fs, np.gradient(x) * fs
    cont = g["stride_phase_cont"] if np.size(g["stride_phase_cont"]) else np.zeros(tg.size)
    bad = ~np.isfinite(cont)
    sig = np.column_stack([vy, vx, np.gradient(vy) * fs])
    sig[bad] = np.nan
    return sig


# ---------------------------------------------------------------------------
# Per block
# ---------------------------------------------------------------------------
def analyse_block(rec) -> dict | None:
    """Variability, DFA (with shuffled control) and lambda_S for one walking block.

    Keys starting with ``_`` hold figure data and are not written to JSON.
    """
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    troughs = np.asarray(g["troughs"], float)
    runs = stride_runs(troughs)
    if not runs:
        return None
    allst = np.concatenate(runs)
    d = dfa(runs)
    rng = np.random.default_rng(0)
    d_shuf = dfa([rng.permutation(r) for r in runs])

    # Lyapunov: unbroken runs as trough-index spans (same rule as stride_runs, at least
    # 2*MIN_RUN steps), capped at LYAP_MAX_STRIDES strides per block for comparability.
    step = np.diff(troughs)
    good = step <= gait_mod.MAX_STEP_RATIO * np.median(step)
    spans, start = [], 0
    for i in range(step.size + 1):
        if i == step.size or not good[i]:
            if i - start >= 2 * MIN_RUN:
                spans.append((start, i))
            start = i + 1
    lyap = None
    if spans:
        series = _time_normalise(g["t"], _head_velocity(g), troughs, spans)
        total, capped = 0, []
        for s in series:
            take = min(len(s), (LYAP_MAX_STRIDES - total) * LYAP_POINTS)
            if take <= 2 * LYAP_POINTS:
                break
            capped.append(s[:take])
            total += take // LYAP_POINTS
        lyap = lyapunov_short(capped) if capped else None
    meta = bids_io.scan_meta(rec) or {}
    return {
        "recording": rec.label(), "sub": rec.sub, "run": rec.run,
        "visit": meta.get("visit"), "acq_time": meta.get("acq_time"),
        "n_strides": int(allst.size), "n_unbroken_runs": len(runs),
        "stride_time_mean_s": round(float(allst.mean()), 4),
        "stride_time_cv_pct": round(float(100 * allst.std() / allst.mean()), 3),
        "dfa_alpha": round(d["alpha"], 3) if d else None,
        "dfa_alpha_shuffled": round(d_shuf["alpha"], 3) if d_shuf else None,
        "dfa_n_strides": d["n_strides"] if d else 0,
        "lambda_s_per_stride": round(lyap["lambda_s_per_stride"], 3) if lyap else None,
        "_dfa": d, "_lyap_curve": lyap["curve"] if lyap else None,
        "_series": [r.tolist() for r in runs],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _figure(rows: list[dict], out: Path) -> None:
    """Plot stride series, DFA fluctuation functions, alpha per participant and divergence
    curves. Blocks are drawn thin and grey; one example block (median alpha) is highlighted."""
    subs = sorted({r["sub"] for r in rows})
    with_alpha = [r for r in rows if isinstance(r.get("dfa_alpha"), (int, float))
                  and np.isfinite(r["dfa_alpha"])]
    ex = (sorted(with_alpha, key=lambda r: r["dfa_alpha"])[len(with_alpha) // 2]
          if with_alpha else rows[0])
    ex_lab = f"{S.nice_label(ex['sub'])} {ex['recording'][-6:]}"
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    # a: stride times of the example block
    a = ax[0, 0]
    t0 = 0
    for seg in ex["_series"]:
        a.plot(np.arange(t0, t0 + len(seg)), np.asarray(seg) * 1000, color=S.BLUE, lw=0.8)
        t0 += len(seg) + 5
    a.set_xlabel("stride number (gaps break the series)")
    a.set_ylabel("stride time (ms)")
    S.panel(a, "a", f"Stride-interval series, example: {ex_lab}", fontsize=10)
    # b: DFA fluctuation functions
    a = ax[0, 1]
    for r in rows:
        if r["_dfa"] and r is not ex:
            sc, F = np.asarray(r["_dfa"]["scales"]), np.asarray(r["_dfa"]["F"])
            a.loglog(sc, F, "-", lw=0.8, color=S.MIST, zorder=1)
    if ex["_dfa"]:
        sc, F = np.asarray(ex["_dfa"]["scales"], float), np.asarray(ex["_dfa"]["F"], float)
        a.loglog(sc, F, "o-", ms=3, lw=1.4, color=S.BLUE, zorder=3, label=f"example ({ex_lab})")
        for al, ls in ((0.5, ":"), (1.0, "--")):
            a.loglog(sc, F[0] * (sc / sc[0]) ** al, color=S.INK, ls=ls, lw=1, zorder=2,
                     label=f"slope {al:g}" + (" (uncorrelated)" if al == 0.5 else " (1/f)"))
    a.loglog([], [], color=S.MIST, lw=0.8, label="other blocks")
    a.set_xlabel("box size (strides)")
    a.set_ylabel("fluctuation F(n) (s)")
    a.legend(loc="upper left", fontsize=8)
    S.panel(a, "b", "Detrended fluctuation analysis", fontsize=10)
    # c: alpha per participant, each block against its shuffled control
    a = ax[1, 0]
    for i, sub in enumerate(subs):
        rs = [r for r in rows if r["sub"] == sub]
        jit = np.linspace(-0.15, 0.15, len(rs)) if len(rs) > 1 else np.zeros(1)
        num = lambda v: float(v) if isinstance(v, (int, float)) else np.nan
        a.scatter(i + jit, [num(r["dfa_alpha_shuffled"]) for r in rs], marker="x", s=18,
                  color=S.GREY, zorder=2, label="shuffled control" if i == 0 else None)
        a.scatter(i + jit, [num(r["dfa_alpha"]) for r in rs], s=20, color=S.BLUE, zorder=3,
                  label="observed" if i == 0 else None)
    a.axhline(0.5, color=S.GREY, lw=0.8, ls=":")
    a.set_xticks(range(len(subs))); a.set_xticklabels([s.replace("sub-", "") for s in subs])
    a.set_xlabel("participant (one point per walking block)")
    a.set_ylabel("DFA α")
    a.legend(loc="upper right", fontsize=8)
    S.panel(a, "c", "Long-range correlation", fontsize=10)
    # d: Lyapunov divergence curves
    a = ax[1, 1]
    curves = []
    for r in rows:
        if r["_lyap_curve"]:
            c = np.asarray(r["_lyap_curve"], float)
            curves.append(c - c[0])
            a.plot(np.arange(c.size) / LYAP_POINTS, c - c[0], color=S.MIST, lw=0.8, zorder=1)
    if curves:
        n = min(len(c) for c in curves)
        mu = np.mean([c[:n] for c in curves], axis=0)
        a.plot(np.arange(n) / LYAP_POINTS, mu, color=S.BLUE, lw=2.2, zorder=3,
               label="mean over blocks")
    a.plot([], [], color=S.MIST, lw=0.8, label="one block")
    a.axvspan(LYAP_FIT[0] / LYAP_POINTS, LYAP_FIT[1] / LYAP_POINTS, color=S.MIST, alpha=0.3,
              lw=0, label="fit range (λS = slope)")
    a.set_xlabel("time since neighbour pair (strides)")
    a.set_ylabel("mean log divergence (rel.)")
    a.legend(loc="lower right", fontsize=8)
    S.panel(a, "d", "Local dynamic stability", fontsize=10)
    fig.suptitle(f"Stride-to-stride dynamics of treadmill walking (exploratory), "
                 f"{len(rows)} blocks of {len(subs)} participants",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Stride times from head-bob troughs; runs broken by tracking gaps are "
                   "kept apart. b-c: DFA-1 over boxes of 16 to N/4 strides (Peng 1995; "
                   "Hausdorff 1995); α 0.5 = no memory between strides, ~0.75-1 = healthy "
                   "long-range correlation; the shuffled control destroys the order and "
                   "should sit at 0.5. d: Rosenstein (1993) divergence of time-normalised "
                   "head velocity/acceleration; λS = its slope over the first half stride "
                   "(Dingwell & Cusumano 2000).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    S.save(fig, out)


def dynamics_main() -> list[dict]:
    """Gait dynamics per walking block; writes a JSON per block, one TSV and one figure."""
    rows = []
    for rec in bids_io.find_recordings(scope="gait"):
        r = analyse_block(rec)
        if r is None:
            continue
        rows.append(r)
        out = bids_io.deriv_dir("gait_dynamics", rec.sub, rec.ses)
        (out / f"{rec.label()}_gait_dynamics.json").write_text(
            json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, indent=2),
            encoding="utf-8")
        print(f"  {rec.label()}: {r['n_strides']} strides in {r['n_unbroken_runs']} run(s); "
              f"stride {r['stride_time_mean_s']} s, CV {r['stride_time_cv_pct']}%; "
              f"DFA alpha {r['dfa_alpha']} (shuffled {r['dfa_alpha_shuffled']}); "
              f"lambda_S {r['lambda_s_per_stride']} /stride")
    bids_io.gate_report("gait_dynamics", "gait", condition="walking",
                        n_analysed=len({r["sub"] for r in rows}))
    if not rows:
        print("No walking blocks with gait output. Run gait.py first.")
        return rows
    out = bids_io.deriv_dir("gait_dynamics")
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                 ).to_csv(out / "gait_dynamics.tsv", sep="\t", index=False, na_rep="n/a")
    _figure(rows, bids_io.fig_dir("gait_dynamics") / "gait_dynamics.png")
    print(f"Gait dynamics -> {out}")
    return rows


# ###########################################################################
# Gait phase response to task events
# ###########################################################################
PRE_STEPS, K_REPORT, K_MAX = 4, 4, 6     # K_REPORT must be even (see docstring)
N_SURR = 20                 # random times per real event
N_NULL = 2000               # random event sets for the tests
PRC_BINS = 10
RNG_SEED = 0


def _usable_steps(troughs: np.ndarray) -> np.ndarray:
    """Per step: False if too long to be one step (gap or missed trough)."""
    step = np.diff(troughs)
    return step <= gait_mod.MAX_STEP_RATIO * np.median(step)


def shifts(times: np.ndarray, troughs: np.ndarray, step_foot=None) -> dict:
    """Shift time course (% step, k = 1..K_MAX) and stride phase per time; NaN if unusable."""
    tr = np.asarray(troughs, float)
    times = np.asarray(times, float)
    n = times.size
    out = np.full((n, K_MAX), np.nan)
    if tr.size < PRE_STEPS + K_MAX + 2:
        return {"shift": out, "phase": np.full(n, np.nan)}
    good = _usable_steps(tr)
    idx = np.searchsorted(tr, times) - 1
    for j, i in enumerate(idx):
        if i - PRE_STEPS < 0 or i + K_MAX >= tr.size:
            continue
        if not good[i - PRE_STEPS:i + K_MAX].all():
            continue
        t0 = (tr[i] - tr[i - PRE_STEPS]) / PRE_STEPS
        k = np.arange(1, K_MAX + 1)
        out[j] = 100.0 * ((tr[i + k] - tr[i]) - k * t0) / t0
    _, phase = gait_mod.phase_at(times, tr, step_foot)
    return {"shift": out, "phase": phase}


def _harm_r2(phase_pct: np.ndarray, y: np.ndarray) -> float:
    """Variance of ``y`` explained by 2 harmonics of stride phase plus a constant."""
    ok = np.isfinite(phase_pct) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan")
    a = 2 * np.pi * phase_pct[ok] / 100.0
    X = np.column_stack([np.ones(ok.sum()), np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    res = y[ok] - X @ beta
    return float(1.0 - res.var() / y[ok].var()) if y[ok].var() > 0 else float("nan")


def _events(rec) -> dict[str, np.ndarray]:
    """Stimulus-onset and button-press times (s, recording clock) of one block."""
    ev = bids_io.load_events(rec)
    if ev.empty:
        return {"onset": np.array([]), "press": np.array([])}
    on = ev["onset"].to_numpy(float)
    rt = ev["response_time"].to_numpy(float)
    return {"onset": on[np.isfinite(on)],
            "press": (on + rt)[np.isfinite(on) & np.isfinite(rt)]}


def block_data(rec, rng) -> dict | None:
    """Event and random-time shifts for one walking block."""
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    tr = np.asarray(g["troughs"], float)
    if tr.size < 50:
        return None
    sf = g["step_foot"] if np.size(g["step_foot"]) else None
    out = {"sub": rec.sub, "recording": rec.label()}
    for kind, t in _events(rec).items():
        out[kind] = shifts(t, tr, sf)
    # Random times come from logged frame times (see module docstring: at ~6 fps trough
    # times are biased toward frames).
    motion, _, _ = bids_io.load_motion(rec)
    frames = motion["t"].to_numpy(float)
    frames = frames[(frames > tr[PRE_STEPS]) & (frames < tr[-K_MAX - 1])]
    n_surr = N_SURR * max(len(_events(rec)["onset"]), 1)
    surr_t = rng.choice(frames, size=n_surr, replace=frames.size < n_surr)
    out["random"] = shifts(surr_t, tr, sf)
    out["fs_logged"] = float(1.0 / np.median(np.diff(motion["t"].to_numpy(float))))
    return out


def _pool(blocks: list[dict], kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Usable shift time courses and stride phases of one event kind, pooled over blocks."""
    sh = np.vstack([b[kind]["shift"] for b in blocks])
    ph = np.concatenate([b[kind]["phase"] for b in blocks])
    ok = np.isfinite(sh[:, K_REPORT - 1]) & np.isfinite(ph)
    return sh[ok], ph[ok]


def participant_tests(blocks: list[dict], rng) -> dict:
    """Mean-shift and PRC-shape tests for one participant (all walking blocks)."""
    rs, rp = _pool(blocks, "random")
    res = {"n_random": int(len(rs))}
    for kind in ("onset", "press"):
        es, ep = _pool(blocks, kind)
        n = len(es)
        if n < 20 or len(rs) < 5 * n:
            res[kind] = {"n": n}
            continue
        y, yr = es[:, K_REPORT - 1], rs[:, K_REPORT - 1]
        diff = float(y.mean() - yr.mean())
        r2 = _harm_r2(ep, y)
        null_mean = np.empty(N_NULL)
        null_r2 = np.empty(N_NULL)
        for b in range(N_NULL):
            pick = rng.choice(len(yr), n, replace=False)
            null_mean[b] = yr[pick].mean() - yr.mean()
            null_r2[b] = _harm_r2(rp[pick], yr[pick])
        res[kind] = {
            "n": n,
            "mean_shift_pct_step": round(float(y.mean()), 3),
            "random_mean_shift_pct_step": round(float(yr.mean()), 3),
            "shift_vs_random_pct_step": round(diff, 3),
            "p_mean_shift": round(float((np.abs(null_mean) >= abs(diff)).mean()), 4),
            "prc_r2": round(r2, 4),
            "prc_r2_null_p95": round(float(np.nanpercentile(null_r2, 95)), 4),
            "p_prc_shape": round(float((null_r2 >= r2).mean()), 4),
            "time_course_pct_step": np.round(np.nanmean(es, 0), 3).tolist(),
            "random_time_course_pct_step": np.round(np.nanmean(rs, 0), 3).tolist(),
        }
    return res


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _binned(ph, y, nb=PRC_BINS):
    """Mean and standard error of ``y`` in ``nb`` stride-phase bins."""
    idx = np.clip((ph / 100 * nb).astype(int), 0, nb - 1)
    m = np.array([np.mean(y[idx == b]) if (idx == b).sum() else np.nan for b in range(nb)])
    se = np.array([np.std(y[idx == b]) / np.sqrt((idx == b).sum()) if (idx == b).sum() > 1
                   else np.nan for b in range(nb)])
    return m, se


def _prc_figure(by_sub: dict[str, list[dict]], tests: dict, out: Path, rng) -> None:
    """Plot group PRCs for onsets and presses against a random-time band, the time course
    and the pooled shift distributions. Participants thin and grey, their mean bold."""
    subs = sorted(by_sub)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    x = (np.arange(PRC_BINS) + 0.5) * 100 / PRC_BINS
    n_null = 300
    for col_i, kind in enumerate(("onset", "press")):
        a = ax[0, col_i]
        curves, nulls = [], []
        for s in subs:
            es, ep = _pool(by_sub[s], kind)
            rs, rp = _pool(by_sub[s], "random")
            if len(es) < 20:
                continue
            m, _ = _binned(ep, es[:, K_REPORT - 1])
            # null band from random sets of the same size
            null = np.array([_binned(rp[p], rs[p, K_REPORT - 1])[0] for p in
                             (rng.choice(len(rs), len(es), replace=False) for _ in range(n_null))])
            curves.append(m); nulls.append(null)
            a.plot(x, m, color=S.MIST, lw=0.9, zorder=1)
        if curves:
            cur = np.array(curves)
            mu = np.nanmean(cur, 0)
            se = np.nanstd(cur, 0, ddof=1) / np.sqrt(len(cur))
            # Group null: draw k of every participant's null, averaged across participants.
            gnull = np.nanmean(np.stack(nulls), axis=0)
            lo, hi = np.nanpercentile(gnull, [2.5, 97.5], axis=0)
            a.fill_between(x, lo, hi, color=S.GREY, alpha=0.18, lw=0, zorder=0,
                           label="random times, 95 %")
            a.errorbar(x, mu, yerr=se, color=S.BLUE, marker="o", ms=4, lw=2, capsize=2,
                       zorder=3, label=f"mean ± SEM, n = {len(cur)}")
        a.plot([], [], color=S.MIST, lw=0.9, label="one participant")
        a.axhline(0, color=S.GREY, lw=0.8)
        a.set_xlabel("stride phase of the event (%)")
        a.set_ylabel(f"shift after {K_REPORT} steps (% of a step)")
        a.legend(loc="upper right", ncol=3, fontsize=8)
        lo_, hi_ = a.get_ylim()
        a.set_ylim(lo_, hi_ + 0.2 * (hi_ - lo_))
        S.panel(a, "ab"[col_i], "Phase response curve: "
                + ("stimulus onset" if kind == "onset" else "button press"), fontsize=10)
    a = ax[1, 0]
    k = np.arange(1, K_MAX + 1)
    for kind, ls, col in (("onset", "-", S.BLUE), ("press", "--", S.ORANGE)):
        rows = []
        for s in subs:
            t = tests[s].get(kind, {})
            if "time_course_pct_step" in t:
                d = (np.asarray(t["time_course_pct_step"])
                     - np.asarray(t["random_time_course_pct_step"]))
                rows.append(d)
                a.plot(k, d, ls, color=S.MIST, lw=0.9, zorder=1)
        if rows:
            r = np.array(rows)
            a.errorbar(k, r.mean(0), yerr=r.std(0, ddof=1) / np.sqrt(len(r)), fmt=ls + "o",
                       color=col, ms=4, lw=2, capsize=2, zorder=3,
                       label=f"{'stimulus onset' if kind == 'onset' else 'button press'}, mean ± SEM")
    a.axhline(0, color=S.GREY, lw=0.8)
    a.set_xlabel("steps after the event")
    a.set_ylabel("shift vs random times (% of a step)")
    a.legend(loc="upper left", fontsize=8)
    S.panel(a, "c", "Does the shift persist?", fontsize=10)
    a = ax[1, 1]
    bins = np.linspace(-30, 30, 41)
    pooled = {kind: [] for kind in ("random", "onset", "press")}
    for s in subs:
        for kind in pooled:
            es, _ = _pool(by_sub[s], kind)
            if len(es):
                pooled[kind].append(es[:, K_REPORT - 1])
    for kind, col, ls, lw in (("random", S.GREY, "-", 2.4), ("onset", S.BLUE, "-", 1.5),
                              ("press", S.ORANGE, "--", 1.5)):
        if pooled[kind]:
            v = np.concatenate(pooled[kind])
            a.hist(v, bins=bins, density=True, histtype="step", color=col, ls=ls, lw=lw,
                   label={"random": "random times", "onset": "stimulus onsets",
                          "press": "button presses"}[kind])
    a.set_xlabel(f"shift after {K_REPORT} steps (% of a step)")
    a.set_ylabel("density")
    a.legend(loc="upper right", fontsize=8)
    S.panel(a, "d", "Event shifts vs background, all participants", fontsize=10)
    fig.suptitle(f"Do the task's events reset the gait rhythm? (exploratory), n = {len(subs)} "
                 "participants", fontsize=13, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Shift = how much earlier (−) or later (+) the next steps came than the "
                   "pre-event rhythm predicts. a-b: each participant's mean shift per phase "
                   "bin (grey) and their mean; band = 95 % range of that mean for random "
                   "times, i.e. the rhythm's own fluctuation. c: shift minus the random-time "
                   "shift over the following steps. Onset and press are ~0.8 s apart and not "
                   "independent. Method after Funato et al. (2016).")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    S.save(fig, out)


def prc_main() -> dict:
    """Gait phase response per participant; writes tests to JSON and a figure."""
    rng = np.random.default_rng(RNG_SEED)
    by_sub: dict[str, list[dict]] = {}
    for rec in bids_io.find_recordings(scope="gait"):
        b = block_data(rec, rng)
        if b is not None:
            by_sub.setdefault(b["sub"], []).append(b)
    bids_io.gate_report("gait_prc", "gait", condition="walking", n_analysed=len(by_sub))
    if not by_sub:
        print("No walking blocks with gait output. Run gait.py first.")
        return {}
    tests = {s: participant_tests(bl, rng) for s, bl in sorted(by_sub.items())}
    out = bids_io.deriv_dir("gait_prc")
    (out / "gait_prc.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    _prc_figure(by_sub, tests, bids_io.fig_dir("gait_prc") / "gait_prc.png", rng)
    for s, t in tests.items():
        for kind in ("onset", "press"):
            r = t.get(kind, {})
            if "p_mean_shift" in r:
                print(f"  {s} {kind:5s}: n={r['n']}, shift vs random {r['shift_vs_random_pct_step']:+.2f}% "
                      f"of a step (p={r['p_mean_shift']}); PRC shape R2 {r['prc_r2']} "
                      f"(null 95% {r['prc_r2_null_p95']}, p={r['p_prc_shape']})")
            else:
                print(f"  {s} {kind:5s}: too few usable events ({r.get('n', 0)})")
    print(f"Gait PRC -> {out}")
    return tests


# ###########################################################################
# Pipeline stage
# ###########################################################################
def main() -> None:
    """Pipeline stage ``gait_dynamics``: stride dynamics, then the phase response curve."""
    dynamics_main()
    prc_main()


if __name__ == "__main__":
    main()
