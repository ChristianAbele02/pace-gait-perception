"""Exploratory: do stimulus onsets or button presses shift the walking rhythm?

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

Usage::

    python -m Analysis.gait_prc
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
    from . import gait as gait_mod
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import gait as gait_mod
    import plotstyle as S

S.apply()

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


def _figure(by_sub: dict[str, list[dict]], tests: dict, out: Path, rng) -> None:
    """Plot PRCs with random-time bands, shift time course and shift distributions."""
    subs = sorted(by_sub)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    x = (np.arange(PRC_BINS) + 0.5) * 100 / PRC_BINS
    for col_i, kind in enumerate(("onset", "press")):
        a = ax[0, col_i]
        for i, s in enumerate(subs):
            c = S.CYCLE[i % len(S.CYCLE)]
            es, ep = _pool(by_sub[s], kind)
            rs, rp = _pool(by_sub[s], "random")
            if len(es) < 20:
                continue
            m, se = _binned(ep, es[:, K_REPORT - 1])
            # null band from random sets of the same size
            null = np.array([_binned(rp[p], rs[p, K_REPORT - 1])[0] for p in
                             (rng.choice(len(rs), len(es), replace=False) for _ in range(300))])
            lo, hi = np.nanpercentile(null, [2.5, 97.5], axis=0)
            a.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0)
            a.errorbar(x, m, yerr=se, color=c, marker="o", ms=4, lw=1.5, capsize=2,
                       label=f"{S.nice_label(s)} (n={len(es)})")
        a.axhline(0, color=S.GREY, lw=0.8)
        a.set_xlabel("stride phase of the event (%)")
        a.set_ylabel(f"shift after {K_REPORT} steps (% of a step)")
        a.legend(frameon=False, fontsize=8)
        S.panel(a, "ab"[col_i], f"Phase response curve: {'stimulus onset' if kind == 'onset' else 'button press'}",
                fontsize=10)
    a = ax[1, 0]
    k = np.arange(1, K_MAX + 1)
    for i, s in enumerate(subs):
        c = S.CYCLE[i % len(S.CYCLE)]
        t = tests[s]
        for kind, ls in (("onset", "-"), ("press", "--")):
            if "time_course_pct_step" in t.get(kind, {}):
                a.plot(k, np.asarray(t[kind]["time_course_pct_step"])
                       - np.asarray(t[kind]["random_time_course_pct_step"]),
                       ls, color=c, marker="o", ms=3,
                       label=f"{S.nice_label(s)} {kind}")
    a.axhline(0, color=S.GREY, lw=0.8)
    a.set_xlabel("steps after the event")
    a.set_ylabel("shift vs random times (% of a step)")
    a.legend(frameon=False, fontsize=8)
    S.panel(a, "c", "Does the shift persist?", fontsize=10)
    a = ax[1, 1]
    for i, s in enumerate(subs):
        c = S.CYCLE[i % len(S.CYCLE)]
        rs, _ = _pool(by_sub[s], "random")
        es, _ = _pool(by_sub[s], "onset")
        bins = np.linspace(-30, 30, 41)
        a.hist(rs[:, K_REPORT - 1], bins=bins, density=True, histtype="step", color=c, ls=":")
        if len(es):
            a.hist(es[:, K_REPORT - 1], bins=bins, density=True, histtype="step", color=c, lw=1.8,
                   label=f"{S.nice_label(s)} onsets")
    a.plot([], [], color=S.GREY, ls=":", label="random times")
    a.set_xlabel(f"shift after {K_REPORT} steps (% of a step)")
    a.set_ylabel("density")
    a.legend(frameon=False, fontsize=8)
    S.panel(a, "d", "Event shifts vs background", fontsize=10)
    fig.suptitle("Do the task's events reset the gait rhythm? (exploratory)", fontsize=13,
                 fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Shift = how much earlier (−) or later (+) the next steps came than the "
                   "pre-event rhythm predicts. Bands: 95 % range of the same curve for random "
                   "times (the rhythm's own fluctuation). Onset and press are ~0.8 s apart and "
                   "not independent. PRC method after Funato et al. (2016).")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    S.save(fig, out)


def main() -> dict:
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
    _figure(by_sub, tests, bids_io.fig_dir("gait_prc") / "gait_prc.png", rng)
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


if __name__ == "__main__":
    main()
