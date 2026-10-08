"""Exploratory: dissociation of stimulus-onset and response phase.

Davidson et al. (2024) analyse accuracy and RT against stride phase at target onset
(sensory) but response likelihood against phase at the button press (motor), and find
different peaks (accuracy/RT near swing, responses near heel strike). This module tests
the dissociation directly and controls the confound that response phase = onset phase +
RT, so a response-locked measure is a shifted copy of the onset-locked one even without a
sensory-motor difference.

Primary test (no RT shift). Both measures are referenced to target onset:
  * accuracy vs onset phase: preferred phase ψ_acc (sensory);
  * P(respond | target) vs onset phase: preferred phase ψ_resp (1 − omission rate; a
    property of the target, so no RT correction is needed).
The dissociation is Δψ = ψ_resp − ψ_acc with a trial-bootstrap CI. A CI excluding 0 means
best sensory performance and most likely responding occur at different stride phases.

Secondary (as in Davidson): the response-likelihood peak referenced to response time, with
the implied RT phase shift (2π·k·medianRT/stride), to compare with ψ_resp.

Also reported: whether the RT oscillation is stronger locked to onset or to the response.

Confirmatory tests remain ``phase_analysis`` (Fourier 2/4 cps) and ``glmm`` / ``population``.

Usage::

    python -m Analysis.onset_response
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

CONF_FREQS = (2.0, 4.0)      # as in phase_analysis
N_BOOT = 1000
RNG = np.random.default_rng(7)


# ===========================================================================
# Circular helpers
# ===========================================================================
def _wrap(a):
    """Wrap angle(s) to (-π, π]."""
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def _circ_mean(angles, weights=None):
    """Circular mean direction and resultant length R of a set of angles."""
    ang = np.asarray(angles, float)
    ok = np.isfinite(ang)
    ang = ang[ok]
    if ang.size == 0:
        return float("nan"), 0.0
    w = np.ones_like(ang) if weights is None else np.asarray(weights, float)[ok]
    z = np.sum(w * np.exp(1j * ang)) / max(np.sum(w), 1e-12)
    return float(np.angle(z)), float(np.abs(z))


def _rayleigh_p(R, n):
    """Large-sample Rayleigh p for a mean resultant length R over n angles."""
    if n == 0:
        return 1.0
    Z = n * R * R
    return float(np.exp(-Z) * (1 + (2 * Z - Z * Z) / (4 * n)))


# ===========================================================================
# Oscillation vector (trial-level LS fit at a fixed frequency)
# ===========================================================================
def _fit_vec(phase_pct: np.ndarray, y: np.ndarray, freq: float) -> dict | None:
    """Trial-level LS fit y ≈ c + a·cos(wx) + b·sin(wx) at ``freq`` cps (w = 2π·freq/100).

    Returns amplitude and peak angle ψ = atan2(b, a) (model A·cos(wx − ψ) peaks at wx = ψ);
    ``pref_pct`` is the peak in % stride.
    """
    phase_pct = np.asarray(phase_pct, float)
    y = np.asarray(y, float)
    ok = np.isfinite(phase_pct) & np.isfinite(y)
    x, yy = phase_pct[ok], y[ok]
    if x.size < 10 or np.ptp(yy) == 0:
        return None
    w = 2 * np.pi * freq / 100.0
    X = np.column_stack([np.ones_like(x), np.cos(w * x), np.sin(w * x)])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    a, b = float(beta[1]), float(beta[2])
    amp = float(np.hypot(a, b))
    psi = float(np.arctan2(b, a))
    pref_pct = float((psi / w) % (100.0 / freq))
    return {"a": a, "b": b, "amp": amp, "psi": psi, "pref_pct": pref_pct, "n": int(x.size)}


def _best_freq(phase, acc) -> float:
    """2 or 4 cps, whichever has the larger accuracy oscillation."""
    best, bamp = CONF_FREQS[0], -1.0
    for f in CONF_FREQS:
        v = _fit_vec(phase, acc, f)
        if v and v["amp"] > bamp:
            best, bamp = f, v["amp"]
    return best


# ===========================================================================
# Trial extraction per recording and subject
# ===========================================================================
def _gait_arrays(rec):
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
    stride_period = float(np.median(np.diff(troughs))) if np.size(troughs) > 1 else float("nan")
    return g, troughs, step_foot, tg, cont, stride_period


def _recording_trials(rec) -> dict | None:
    """Onset-referenced trial arrays of one walking recording, response phases of answered
    trials, and the median stride period."""
    if rec.condition != "walking":
        return None
    ga = _gait_arrays(rec)
    if ga is None:
        return None
    _, troughs, step_foot, tg, cont, stride_period = ga
    tp = pa._trial_phases(rec, troughs, step_foot, tg, cont)
    if tp is None:
        return None
    ev, resp = tp
    onset_phase = ev["phase_target"].to_numpy(float)
    acc = ev["acc"].to_numpy(float)
    responded = ev["response"].notna().to_numpy()
    # log10 RT of answered correct trials (same RT filter as phase_analysis)
    rt = ev["rt"].to_numpy(float)
    logrt = np.full(ev.shape[0], np.nan)
    good_rt = responded & (ev["acc"].to_numpy(float) == 1.0) & (rt > pa.RT_MIN_S) & np.isfinite(rt)
    logrt[good_rt] = np.log10(rt[good_rt])
    return {
        "onset_phase": onset_phase, "acc": acc, "responded": responded.astype(float),
        "logrt_onset": logrt,
        "resp_onset_phase": resp["phase_target"].to_numpy(float),
        "resp_response_phase": resp["phase_response"].to_numpy(float),
        "resp_logrt": np.log10(np.clip(resp["rt"].to_numpy(float), 10 ** (np.log10(pa.RT_MIN_S)), None)),
        "stride_period": stride_period,
    }


def _pool_subject(recs) -> dict | None:
    """Concatenate all walking runs of one subject."""
    keys = ("onset_phase", "acc", "responded", "logrt_onset",
            "resp_onset_phase", "resp_response_phase", "resp_logrt")
    pools = {k: [] for k in keys}
    periods = []
    for rec in recs:
        tr = _recording_trials(rec)
        if not tr:
            continue
        for k in keys:
            pools[k].append(tr[k])
        if np.isfinite(tr["stride_period"]):
            periods.append(tr["stride_period"])
    if not pools["onset_phase"]:
        return None
    out = {k: np.concatenate(v) for k, v in pools.items()}
    out["stride_period"] = float(np.median(periods)) if periods else float("nan")
    return out


# ===========================================================================
# Dissociation statistic
# ===========================================================================
def dissociation(tr: dict, freq: float | None = None, nboot: int = N_BOOT,
                 rng: np.random.Generator | None = None) -> dict | None:
    """ψ_acc, ψ_resp (onset-referenced) and the bootstrap CI of Δψ = ψ_resp − ψ_acc.

    Also returns the response-time-referenced response-likelihood peak with the RT phase
    shift, and which reference gives the stronger RT oscillation.
    """
    rng = rng or RNG
    ph = tr["onset_phase"]
    acc = tr["acc"]
    resp = tr["responded"]
    freq = freq or _best_freq(ph, acc)

    v_acc = _fit_vec(ph, acc, freq)
    v_resp = _fit_vec(ph, resp, freq)          # P(respond | target), onset-referenced
    if v_acc is None or v_resp is None:
        return None
    dpsi = float(_wrap(v_resp["psi"] - v_acc["psi"]))

    # bootstrap Δψ over trials
    n = ph.size
    boot = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        va = _fit_vec(ph[idx], acc[idx], freq)
        vr = _fit_vec(ph[idx], resp[idx], freq)
        boot[i] = _wrap(vr["psi"] - va["psi"]) if (va and vr) else np.nan
    boot = boot[np.isfinite(boot)]
    mu, R = _circ_mean(boot)
    # circular percentile CI around the bootstrap mean
    centred = _wrap(boot - mu)
    lo, hi = np.percentile(centred, [2.5, 97.5])
    ci = (_wrap(mu + lo), _wrap(mu + hi))
    excludes_zero = not (_wrap(0 - mu) >= lo and _wrap(0 - mu) <= hi)

    # response-likelihood peak in response-time reference, with RT shift (Davidson)
    rl = _response_time_peak(tr, freq)

    # RT oscillation locked to onset vs response
    rt_on = _fit_vec(tr["resp_onset_phase"], tr["resp_logrt"], freq)
    rt_rs = _fit_vec(tr["resp_response_phase"], tr["resp_logrt"], freq)

    return {
        "freq_cps": float(freq), "n_trials": int(n),
        "psi_acc_rad": v_acc["psi"], "acc_pref_pct": v_acc["pref_pct"], "acc_amp": v_acc["amp"],
        "psi_respond_rad": v_resp["psi"], "respond_pref_pct": v_resp["pref_pct"],
        "respond_amp": v_resp["amp"],
        "delta_psi_rad": dpsi, "delta_psi_boot_mean_rad": float(mu),
        "delta_psi_ci_rad": [float(ci[0]), float(ci[1])], "boot_R": float(R),
        "dissociation": bool(excludes_zero),
        "response_time_referenced": rl,
        "rt_onset_locked_amp": (rt_on["amp"] if rt_on else None),
        "rt_response_locked_amp": (rt_rs["amp"] if rt_rs else None),
        "rt_cleaner_reference": (None if not (rt_on and rt_rs)
                                 else ("onset" if rt_on["amp"] >= rt_rs["amp"] else "response")),
    }


def _response_time_peak(tr: dict, freq: float) -> dict | None:
    """Response-likelihood peak in response-time reference (Davidson's measure).

    Fits the per-bin ratio of responses to onsets, then maps the peak to onset reference
    by removing the RT shift 2π·k·medianRT/stride."""
    nb = 20
    edges = np.linspace(0, 100, nb + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    onset_counts, _ = np.histogram(tr["onset_phase"][np.isfinite(tr["onset_phase"])], bins=edges)
    resp_counts, _ = np.histogram(tr["resp_response_phase"][np.isfinite(tr["resp_response_phase"])], bins=edges)
    ok = onset_counts > 0
    if ok.sum() < 10:
        return None
    ratio = np.divide(resp_counts, onset_counts, out=np.zeros_like(resp_counts, float), where=ok)
    v = _fit_vec(cen[ok], ratio[ok], freq)
    if v is None:
        return None
    med_rt = 10 ** np.nanmedian(tr["resp_logrt"]) if np.isfinite(np.nanmedian(tr["resp_logrt"])) else np.nan
    shift = (2 * np.pi * freq * med_rt / tr["stride_period"]) if (
        np.isfinite(med_rt) and np.isfinite(tr["stride_period"]) and tr["stride_period"] > 0) else np.nan
    psi_onsetref = float(_wrap(v["psi"] - shift)) if np.isfinite(shift) else None
    return {"psi_resptime_rad": v["psi"], "pref_pct": v["pref_pct"], "amp": v["amp"],
            "rt_shift_rad": (float(shift) if np.isfinite(shift) else None),
            "psi_onset_referenced_rad": psi_onsetref}


# ===========================================================================
# Figures
# ===========================================================================
def _figure(label, tr, res, out: Path):
    freq = res["freq_cps"]
    w = 2 * np.pi * freq / 100.0
    xf = np.linspace(0, 100, 200)
    fig = plt.figure(figsize=(12.4, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.32)
    fig.suptitle(f"Onset vs response locus — {S.nice_label(label)}  "
                 f"({res['n_trials']} trials, fit @ {freq:.0f} cps)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")

    # (a) accuracy and P(respond) vs onset phase, with fits
    axL = fig.add_subplot(gs[0, 0])
    for y, tag, col in [(tr["acc"], "accuracy", S.TEAL), (tr["responded"], "P(respond)", S.ORANGE)]:
        v = _fit_vec(tr["onset_phase"], y, freq)
        cen, binned, _ = pa._bin(tr["onset_phase"], y, 20, "mean")
        axL.scatter(cen, binned, s=14, color=col, alpha=0.4)
        if v:
            base = float(np.nanmean(y))     # cos/sin are ~orthogonal to the constant
            axL.plot(xf, base + v["a"] * np.cos(w * xf) + v["b"] * np.sin(w * xf),
                     color=col, lw=2.4,
                     label=f"{tag}: peak {v['pref_pct']:.0f}% (amp {v['amp']:.3f})")
    axL.set_xlabel("target onset (% stride)"); axL.set_ylabel("rate")
    axL.legend(loc="upper right", fontsize=8)
    S.panel(axL, "a", "Onset-referenced: sensory vs response locus", fontsize=10)

    # (b) preferred-phase vectors
    axP = fig.add_subplot(gs[0, 1], projection="polar")
    axP.set_theta_zero_location("N"); axP.set_theta_direction(-1)
    for psi, amp, tag, col in [
        (res["psi_acc_rad"], res["acc_amp"], "accuracy (onset)", S.TEAL),
        (res["psi_respond_rad"], res["respond_amp"], "P(respond) (onset)", S.ORANGE)]:
        axP.annotate("", xy=(psi, amp), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
    rl = res.get("response_time_referenced")
    if rl:
        axP.annotate("", xy=(rl["psi_resptime_rad"], rl["amp"]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=S.PURPLE, lw=2.0, ls="--"))
    axP.set_yticklabels([])
    dpsi = np.degrees(res["delta_psi_rad"])
    axP.set_title(f"b  Preferred phase  ·  Δψ={dpsi:+.0f}°"
                  + ("  *" if res["dissociation"] else ""), loc="left", fontsize=9.5)

    S.caption(fig, "Δψ = P(respond) − accuracy preferred phase (both onset-referenced; "
                   "shift-free). * = bootstrap 95% CI excludes 0. Dashed purple = "
                   "response-time-referenced response likelihood (Davidson), RT-shifted.")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.07, right=0.95)
    S.save(fig, out)


def _population_figure(rows, out: Path):
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    fig.suptitle(f"Onset vs response locus — population (n = {len(rows)} participants)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    # (a) preferred phases per participant
    axP = fig.add_subplot(gs[0, 0], projection="polar")
    axP.set_theta_zero_location("N"); axP.set_theta_direction(-1)
    for r in rows:
        axP.plot([r["psi_acc_rad"]], [1], "o", color=S.TEAL, ms=6)
        axP.plot([r["psi_respond_rad"]], [1], "s", color=S.ORANGE, ms=6)
    for ang, col in [(_circ_mean([r["psi_acc_rad"] for r in rows])[0], S.TEAL),
                     (_circ_mean([r["psi_respond_rad"] for r in rows])[0], S.ORANGE)]:
        axP.annotate("", xy=(ang, 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
    axP.set_ylim(0, 1.15); axP.set_yticklabels([])
    S.polar_panel(axP, "a", "Preferred phase (● acc, ■ respond)", fontsize=10)
    # (b) distribution of Δψ
    axH = fig.add_subplot(gs[0, 1])
    d = np.degrees([r["delta_psi_rad"] for r in rows])
    axH.hist(d, bins=np.linspace(-180, 180, 13), color=S.MIST, edgecolor="white")
    mu, R = _circ_mean([r["delta_psi_rad"] for r in rows])
    axH.axvline(np.degrees(mu), color=S.INK, lw=2, label=f"circ mean {np.degrees(mu):+.0f}°")
    axH.axvline(0, color=S.RED, ls=":", lw=1.2, label="no dissociation")
    axH.set_xlabel("Δψ = respond − accuracy (deg)"); axH.set_ylabel("participants")
    axH.legend(fontsize=8)
    S.panel(axH, "b", "Sensory–motor phase gap", fontsize=10)
    S.caption(fig, "Δψ clustered away from 0 (Rayleigh) ⇒ population-level sensory-vs-"
                   "response phase dissociation.")
    # tight_layout does not support polar axes; use explicit margins.
    fig.subplots_adjust(left=0.05, right=0.97, top=0.82, bottom=0.15, wspace=0.3)
    S.save(fig, out)


# ===========================================================================
# Driver
# ===========================================================================
def main():
    """Test the onset-vs-response phase dissociation per participant and across participants."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    if not recs:
        print("No walking recordings found. Run the gait/phase stages first.")
        return
    from collections import defaultdict
    by_sub = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)

    out_dir = bids_io.deriv_dir("onset_response")
    rows = []
    print("=== Onset vs response locus (EXPLORATORY) ===")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        tr = _pool_subject(subrecs)
        if tr is None:
            continue
        res = dissociation(tr, rng=np.random.default_rng(abs(hash((sub, ses))) % 2**32))
        if res is None:
            continue
        res.update({"sub": sub, "ses": ses})
        rows.append(res)
        label = f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking"
        (out_dir / f"{label}_onset_response.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        _figure(label, tr, res, bids_io.fig_dir("onset_response") / f"{label}_onset_response.png")
        flag = "DISSOCIATED" if res["dissociation"] else "n.s."
        print(f"  {sub} {ses}: fit @ {res['freq_cps']:.0f} cps | acc peak "
              f"{res['acc_pref_pct']:.0f}%, respond peak {res['respond_pref_pct']:.0f}% | "
              f"dpsi={np.degrees(res['delta_psi_rad']):+.0f} deg "
              f"[{np.degrees(res['delta_psi_ci_rad'][0]):+.0f},"
              f"{np.degrees(res['delta_psi_ci_rad'][1]):+.0f}] -> {flag}")

    bids_io.gate_report("onset_response", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("onset_response",))
    if len(rows) < 2:
        bids_io.clear_stale("onset_response", "onset_response_population.json",
                            "onset_response_population.png")
    if len(rows) >= 2:
        mu_d, R_d = _circ_mean([r["delta_psi_rad"] for r in rows])
        p_d = _rayleigh_p(R_d, len(rows))
        mu_a, R_a = _circ_mean([r["psi_acc_rad"] for r in rows])
        mu_r, R_r = _circ_mean([r["psi_respond_rad"] for r in rows])
        pop = {"n_participants": len(rows),
               "delta_psi_circmean_deg": float(np.degrees(mu_d)), "delta_psi_R": float(R_d),
               "delta_psi_rayleigh_p": float(p_d),
               "population_dissociation": bool(p_d < 0.05 and R_d > 0),
               "accuracy_meanphase_deg": float(np.degrees(mu_a)), "accuracy_R": float(R_a),
               "respond_meanphase_deg": float(np.degrees(mu_r)), "respond_R": float(R_r),
               "note": "EXPLORATORY. Δψ is onset-referenced (RT-shift-free). Confirmatory "
                       "tests remain phase_analysis / glmm / population."}
        (out_dir / "onset_response_population.json").write_text(json.dumps(pop, indent=2), encoding="utf-8")
        _population_figure(rows, bids_io.fig_dir("onset_response") / "onset_response_population.png")
        print(f"\n  POPULATION: dpsi circ-mean {np.degrees(mu_d):+.0f} deg (R={R_d:.2f}, "
              f"Rayleigh p={p_d:.3f}) -> "
              f"{'DISSOCIATED' if pop['population_dissociation'] else 'n.s.'}")
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
