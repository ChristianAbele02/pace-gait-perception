"""Exploratory: drift-diffusion decomposition of the stride modulation.

The Fourier analysis shows that accuracy rises and RT falls together near the preferred
stride phase, but not why. A drift-diffusion model maps the joint (accuracy, RT) pattern
onto latent parameters:

  * drift rate v: quality of sensory evidence. A v rhythm raises accuracy and shortens RT
    together, i.e. a perceptual sensitivity oscillation.
  * boundary a: response caution. An a rhythm trades speed against accuracy, i.e. a
    criterion or urgency effect.
  * non-decision time Ter: sensory and motor delay. A Ter rhythm shifts RT without
    changing accuracy, i.e. motor timing (the button press within the step).

Uses EZ-diffusion (Wagenmakers, van der Maas & Grasman 2007), a closed-form map from
(P_correct, mean and variance of correct RT) to (v, a, Ter), so no HDDM/pyddm is needed.
Phase is binned; each parameter gets a first-order Fourier fit at 2 or 4 cps and a
phase-label permutation test. The significant parameter with the largest relative
amplitude is reported as the carrier. EZ is well-posed for this 2-AFC design (correct and
error trials, near threshold) but is a screening method; a hierarchical Bayesian
regression DDM (v, a, t as circular functions of phase) would be the confirmatory test on
the full cohort.

Needs many trials per bin (Davidson: ≈ 885 walking targets per participant), so the pooled
result is more informative than single participants.

Usage::

    python -m Analysis.ddm_phase
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
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import gait as gait_mod
    import phase_analysis as pa
    import plotstyle as S

S.apply()

CONF_FREQS = (2.0, 4.0)
N_BINS = 8                   # phase bins for EZ
N_PERM = 1000
S_SCALE = 0.1               # EZ scaling constant (Wagenmakers 2007 default)
PARAMS = ("v", "a", "Ter")
PARAM_LABEL = {"v": "drift rate v (sensory)", "a": "boundary a (caution)",
               "Ter": "non-decision Ter (motor)"}
RNG = np.random.default_rng(23)


# ===========================================================================
# EZ-diffusion closed form
# ===========================================================================
def ez_diffusion(pc: float, vrt: float, mrt: float, s: float = S_SCALE):
    """EZ-diffusion (Wagenmakers et al. 2007): (P_correct, var and mean of correct RT) to
    (v, a, Ter). None if degenerate (no RT variance or accuracy at chance)."""
    if not (np.isfinite(pc) and np.isfinite(vrt) and np.isfinite(mrt)) or vrt <= 0:
        return None
    pc = min(max(pc, 1e-4), 1 - 1e-4)
    if abs(pc - 0.5) < 1e-3:                     # v is undefined at exactly chance
        pc = 0.5 + (1e-3 if pc >= 0.5 else -1e-3)
    L = np.log(pc / (1 - pc))                    # logit
    x = L * (L * pc ** 2 - L * pc + pc - 0.5) / vrt
    v = np.sign(pc - 0.5) * s * abs(x) ** 0.25
    if v == 0:
        return None
    a = s ** 2 * L / v
    mdt = (a / (2 * v)) * (2 * pc - 1)           # = (a/2v)(1-e^{-La})/(1+e^{-La})
    ter = mrt - mdt
    return {"v": float(v), "a": float(a), "Ter": float(ter)}


def ez_from_trials(correct: np.ndarray, rt: np.ndarray, s: float = S_SCALE):
    """EZ parameters from 2-AFC trials (rt in s, correct in {0,1})."""
    correct = np.asarray(correct, float)
    rt = np.asarray(rt, float)
    n = correct.size
    if n < 8:
        return None
    pc = float(correct.mean())
    # edge correction for 0 % or 100 % correct (Wagenmakers 2007)
    if pc >= 1.0:
        pc = 1 - 1.0 / (2 * n)
    elif pc <= 0.0:
        pc = 1.0 / (2 * n)
    rc = rt[(correct == 1.0) & np.isfinite(rt) & (rt > pa.RT_MIN_S)]
    if rc.size < 3:
        return None
    return ez_diffusion(pc, float(np.var(rc, ddof=1)), float(np.mean(rc)), s)


# ===========================================================================
# Phase-resolved EZ and oscillation test
# ===========================================================================
def _fit_osc(centers, y, freq):
    """LS fit y ≈ c + a·cos(wx) + b·sin(wx); return (amp, psi, R²) or None."""
    ok = np.isfinite(centers) & np.isfinite(y)
    if ok.sum() < 5:
        return None
    x, yy = centers[ok], y[ok]
    w = 2 * np.pi * freq / 100.0
    X = np.column_stack([np.ones_like(x), np.cos(w * x), np.sin(w * x)])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    yhat = X @ beta
    ss = np.sum((yy - yhat) ** 2); st = np.sum((yy - yy.mean()) ** 2)
    return {"amp": float(np.hypot(beta[1], beta[2])),
            "psi": float(np.arctan2(beta[2], beta[1])),
            "r2": float(1 - ss / st) if st > 0 else 0.0,
            "mean": float(yy.mean())}


def _binned_ez(phase, correct, rt, nbins=N_BINS):
    """EZ parameters per phase bin. Returns (centers, {param: values})."""
    edges = np.linspace(0, 100, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    out = {p: np.full(nbins, np.nan) for p in PARAMS}
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 8:
            ez = ez_from_trials(correct[m], rt[m])
            if ez:
                for p in PARAMS:
                    out[p][b] = ez[p]
    return centers, out


def decompose(phase, correct, rt, freq=None, nbins=N_BINS, nperm=N_PERM, rng=None) -> dict | None:
    """Find the DDM parameter carrying the stride oscillation.

    Per-bin EZ, first-order Fourier fit at ``freq`` (default: 2 or 4 cps, whichever has the
    larger accuracy oscillation) and a phase-label permutation null. Returns amplitude,
    relative amplitude and p per parameter, and the carrier."""
    rng = rng or RNG
    phase = np.asarray(phase, float); correct = np.asarray(correct, float); rt = np.asarray(rt, float)
    ok = np.isfinite(phase) & np.isfinite(correct)
    phase, correct, rt = phase[ok], correct[ok], rt[ok]
    if phase.size < nbins * 12:
        return None
    if freq is None:
        # frequency with the largest hit-rate oscillation
        cen, hr, _ = pa._bin(phase, correct, nbins, "mean")
        freq = max(CONF_FREQS, key=lambda f: (_fit_osc(cen, hr, f) or {"amp": 0})["amp"])

    centers, binp = _binned_ez(phase, correct, rt, nbins)
    obs = {p: _fit_osc(centers, binp[p], freq) for p in PARAMS}
    if all(o is None for o in obs.values()):
        return None

    # permutation null: shuffle phase labels, re-bin, refit
    null_amp = {p: np.zeros(nperm) for p in PARAMS}
    for i in range(nperm):
        perm = rng.permutation(phase.size)
        _, bp = _binned_ez(phase[perm], correct, rt, nbins)
        for p in PARAMS:
            fo = _fit_osc(centers, bp[p], freq)
            null_amp[p][i] = fo["amp"] if fo else 0.0

    results = {}
    for p in PARAMS:
        o = obs[p]
        if o is None:
            continue
        pval = float((1 + np.sum(null_amp[p] >= o["amp"])) / (1 + nperm))
        mean_abs = abs(np.nanmean(binp[p])) or 1e-9
        results[p] = {"amp": o["amp"], "rel_amp": float(o["amp"] / mean_abs),
                      "psi_rad": o["psi"], "pref_pct": float((o["psi"] / (2 * np.pi * freq / 100.0))
                                                             % (100.0 / freq)),
                      "r2": o["r2"], "mean": o["mean"], "p_value": pval,
                      "significant": bool(pval < 0.05)}
    if not results:
        return None
    # carrier: significant parameter with the largest relative amplitude, else smallest p
    sig = {p: r for p, r in results.items() if r["significant"]}
    if sig:
        carrier = max(sig, key=lambda p: sig[p]["rel_amp"])
    else:
        carrier = min(results, key=lambda p: results[p]["p_value"])
    return {"freq_cps": float(freq), "n_trials": int(phase.size), "n_bins": nbins,
            "params": results, "carrier": carrier,
            "carrier_significant": bool(results[carrier]["significant"]),
            "centers_pct": centers.tolist(),
            "binned": {p: binp[p].tolist() for p in PARAMS},
            "interpretation": {
                "v": "sensory sensitivity", "a": "response caution / speed-accuracy",
                "Ter": "motor / non-decision timing"}[carrier]}


# ===========================================================================
# Trial gathering
# ===========================================================================
def _recording_trials(rec):
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        gait_mod.gait_recording(rec)
    if not npz.exists():
        return None
    g = np.load(npz)
    troughs = g["troughs"]
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    cont = pa.continuous_phase(g)
    tp = pa._trial_phases(rec, troughs, step_foot, tg, cont)
    if tp is None:
        return None
    ev, _ = tp
    return (ev["phase_target"].to_numpy(float), ev["acc"].to_numpy(float), ev["rt"].to_numpy(float))


def _pool(recs):
    ph, co, rt = [], [], []
    for rec in recs:
        tr = _recording_trials(rec)
        if tr:
            ph.append(tr[0]); co.append(tr[1]); rt.append(tr[2])
    if not ph:
        return None
    return np.concatenate(ph), np.concatenate(co), np.concatenate(rt)


# ===========================================================================
# Figure
# ===========================================================================
def _figure(label, res, out: Path):
    centers = np.asarray(res["centers_pct"])
    freq = res["freq_cps"]; w = 2 * np.pi * freq / 100.0
    xf = np.linspace(0, 100, 200)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    fig.suptitle(f"Drift-diffusion decomposition — {S.nice_label(label)}  "
                 f"({res['n_trials']} trials, fit @ {freq:.0f} cps; carrier: "
                 f"{PARAM_LABEL[res['carrier']]})",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    cols = {"v": S.TEAL, "a": S.PURPLE, "Ter": S.ORANGE}
    for a_, p in zip(ax, PARAMS):
        yb = np.asarray(res["binned"][p], float)
        a_.scatter(centers, yb, s=26, color=cols[p], zorder=3)
        r = res["params"].get(p)
        if r:
            fo = _fit_osc(centers, yb, freq)
            if fo:
                a_.plot(xf, fo["mean"] + r["amp"] * np.cos(w * xf - r["psi_rad"]),
                        color=cols[p], lw=2.4,
                        label=f"amp {r['amp']:.3g} (p={r['p_value']:.3f}"
                              f"{' *' if r['significant'] else ''})")
            a_.legend(loc="upper right", fontsize=8)
        a_.set_xlabel("stride phase (% stride)"); a_.set_ylabel(PARAM_LABEL[p])
        S.panel(a_, "abc"[PARAMS.index(p)], PARAM_LABEL[p], fontsize=9.5)
    S.caption(fig, "Per-bin EZ-diffusion (Wagenmakers 2007); first-order Fourier at the "
                   "gait frequency, phase-label permutation null. A drift (v) rhythm ⇒ "
                   "sensory; boundary (a) ⇒ caution; non-decision (Ter) ⇒ motor. EXPLORATORY.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    S.save(fig, out)


# ===========================================================================
# Driver
# ===========================================================================
def main():
    """Fit the EZ decomposition per participant and pooled."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    if not recs:
        print("No walking recordings found. Run the gait/phase stages first.")
        return
    out_dir = bids_io.deriv_dir("ddm_phase")
    by_sub = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)

    print("=== Drift-diffusion decomposition (EZ; EXPLORATORY) ===")
    rows = []
    for (sub, ses), subrecs in sorted(by_sub.items()):
        pooled = _pool(subrecs)
        if pooled is None:
            continue
        res = decompose(*pooled, rng=np.random.default_rng(abs(hash((sub, ses, "ddm"))) % 2**32))
        if res is None:
            continue
        res.update({"sub": sub, "ses": ses})
        rows.append(res)
        label = f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking"
        (out_dir / f"{label}_ddm_phase.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        _figure(label, res, bids_io.fig_dir("ddm_phase") / f"{label}_ddm_phase.png")
        pr = res["params"]
        print(f"  {sub} {ses}: carrier = {res['carrier']} ({res['interpretation']}"
              f"{', sig' if res['carrier_significant'] else ', n.s.'}) | "
              + " ".join(f"{p}:p={pr[p]['p_value']:.3f}" for p in PARAMS if p in pr))

    bids_io.gate_report("ddm", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("ddm_phase",))
    # pooled decomposition over all walking participants
    pooled_all = _pool(recs)
    pres = None
    if pooled_all is not None and pooled_all[0].size >= N_BINS * 12:
        pres = decompose(*pooled_all, rng=np.random.default_rng(999))
    if pres:
        pres["note"] = ("EXPLORATORY pooled decomposition (ignores between-participant "
                        "variance). Confirmatory locus test = hierarchical regression DDM "
                        "on the full cohort.")
        (out_dir / "ddm_phase_pooled.json").write_text(json.dumps(pres, indent=2), encoding="utf-8")
        _figure("pooled (all walking)", pres, bids_io.fig_dir("ddm_phase") / "ddm_phase_pooled.png")
        pr = pres["params"]
        print(f"\n  POOLED (n={len({r['sub'] for r in rows})} participants, "
              f"{pres['n_trials']} trials): carrier = {pres['carrier']} "
              f"({pres['interpretation']}{', sig' if pres['carrier_significant'] else ', n.s.'})")
        for p in PARAMS:
            if p in pr:
                print(f"    {PARAM_LABEL[p]:28s}: amp {pr[p]['amp']:.3g} "
                      f"(rel {pr[p]['rel_amp']:.3f}), p={pr[p]['p_value']:.3f}")
    else:
        bids_io.clear_stale("ddm_phase", "ddm_phase_pooled.json", "ddm_phase_pooled.png")
    print("\n  NOTE: EZ is a screening decomposition; a hierarchical Bayesian regression DDM "
          "on the full cohort is the confirmatory locus test.")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
