"""How large a SHARED stride-phase oscillation would the analysed sample have detected?

The power analysis (``davidson_power``) is built on Davidson et al.'s effect, whose
participants oscillate at their own phases (cross-participant phase coherence R = 0.33).
Group-level reports of stride-phase effects are often of the other kind: one curve
averaged over participants, which only shows a modulation if participants share its
phase. This stage asks how large such a shared oscillation would have to be for the
pipeline's tests to catch it in the sample actually analysed.

Effect model: every participant's accuracy oscillates at 2 cycles per stride with the
same amplitude ``A`` and the same phase (drawn once per simulated study). Each participant
contributes the number of answered walking trials they actually contributed. The
simulator and the three tests are those of ``davidson_power/power.py``, validated against
the pipeline's own estimators:

* phase-agnostic ``S`` (the confirmatory test, ``group_agnostic``), Holm over 2 and
  4 cps, against a null simulated at the same trial counts;
* common phase, Hotelling T2 on the participants' coefficients (``common_phase``),
  Holm over 2 and 4 cps;
* pooled, every trial pooled into one curve (``phase_analysis.pooled_analysis``),
  approximated by the trial-weighted mean coefficient against its sampling variance:
  ``|z|^2 / Var z`` is exponential with mean 1 under H0, Holm over 2 and 4 cps.

For each test the result is the power curve over ``A`` and the smallest shared amplitude
detected with 80 % power (MDA). Marked for comparison: the shared 2-cps amplitude the
data show (``common_phase``: the length of the participants' mean coefficient), the
amplitude of the pooled curve (``phase_analysis`` pooled), and Davidson et al.'s
coherent group amplitude. The same is computed for the planned design (N = 12 and 20 at
1440 trials) so the interim result can be read against the final one.

The model assumes every block runs at its calibrated operating point; with the observed
calibration error (``calibration_cost``) the power is lower, so the MDAs are lower bounds.

Outputs (``derivatives/shared_power/``): ``shared_power.json``,
``figures/shared_power.png``.

Usage::

    python -m Analysis.shared_power
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import bids_io
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import phase_analysis as pa
    import plotstyle as S

S.apply()


def _simulator():
    """``davidson_power/power.py``, loaded by path (the folder is not a package)."""
    path = Path(__file__).resolve().parent / "davidson_power" / "power.py"
    spec = importlib.util.spec_from_file_location("davidson_power_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dp = _simulator()

AMPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]) / 100.0
N_SIM, N_NULL = 2000, 5000
PLANNED_N = (12, 20)
SEED = 20261028
TESTS = ("agnostic", "common", "pooled")
LABEL = {"agnostic": "phase-agnostic S (confirmatory)", "common": "common phase (Hotelling)",
         "pooled": "pooled trials"}
COLOR = {"agnostic": S.BLUE, "common": S.PURPLE, "pooled": S.ORANGE}


def study_coefficients(rng, ntrials, amp: float, nsim: int):
    """Per simulated study and participant: fitted coefficients ``z`` (nsim, N, 2) and
    their null variances ``s2``, for a shared 2-cps oscillation of amplitude ``amp``."""
    phi = rng.uniform(0, 2 * np.pi, nsim)
    zs, s2s = [], []
    for n in ntrials:
        coef = np.zeros((nsim, 2), complex)
        coef[:, 0] = amp * np.exp(1j * phi)
        est = dp.estimate(dp.simulate_tables(rng, coef, int(n)))
        zs.append(est.z)
        s2s.append(est.s2)
    return np.stack(zs, 1), np.stack(s2s, 1)


def statistics(z: np.ndarray, s2: np.ndarray, ntrials) -> dict:
    """The three tests' statistics per study (each ``(nsim, 2)`` over 2 and 4 cps)."""
    w = np.asarray(ntrials, float)[None, :, None]
    zp = (w * z).sum(1) / w.sum(1)
    vp = (w ** 2 * s2).sum(1) / w.sum(1) ** 2
    return {"S": (np.abs(z) ** 2 / s2).sum(1),
            "p_common": dp.hotelling_p(np.moveaxis(z, 2, 1)),
            "p_pooled": np.exp(-np.abs(zp) ** 2 / vp)}


def power_curve(rng, ntrials, *, amps=AMPS, nsim=N_SIM, nnull=N_NULL) -> dict:
    """Power of each test over the shared amplitudes, for one set of trial counts."""
    null = statistics(*study_coefficients(rng, ntrials, 0.0, nnull), ntrials)
    null_s = np.sort(null["S"], 0)
    out = {t: [] for t in TESTS}
    for a in amps:
        st = statistics(*study_coefficients(rng, ntrials, float(a), nsim), ntrials)
        exceed = np.stack([nnull - np.searchsorted(null_s[:, f], st["S"][:, f])
                           for f in range(2)], 1)
        p_agn = (1 + exceed) / (1 + nnull)
        # Holm over the pair rejects anything iff the smaller p is below alpha / 2
        out["agnostic"].append(float((p_agn.min(1) < dp.ALPHA_ANY).mean()))
        out["common"].append(float((st["p_common"].min(1) < dp.ALPHA_ANY).mean()))
        out["pooled"].append(float((st["p_pooled"].min(1) < dp.ALPHA_ANY).mean()))
    return {"amps": [float(a) for a in amps], **out,
            "mda80": {t: mda(amps, out[t]) for t in TESTS}}


def mda(amps, power, target: float = 0.80) -> float | None:
    """Smallest amplitude reaching ``target`` power, by linear interpolation."""
    amps, power = np.asarray(amps, float), np.asarray(power, float)
    above = np.flatnonzero(power >= target)
    if not above.size:
        return None
    i = int(above[0])
    if i == 0:
        return float(amps[0])
    a0, a1, p0, p1 = amps[i - 1], amps[i], power[i - 1], power[i]
    return float(a0 + (target - p0) * (a1 - a0) / max(p1 - p0, 1e-12))


def _trial_counts() -> dict[str, int]:
    """Answered walking trials per analysed participant (pooled phase outputs)."""
    out = {}
    for sub, ses in sorted({(r.sub, r.ses) for r in bids_io.find_recordings()
                            if r.condition == "walking"}):
        f = bids_io.deriv_dir("phase", sub, ses) / f"{pa.subject_walk_label(sub, ses)}_phase.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("n_answered"):
                out[sub] = int(d["n_answered"])
    return out


def _observed() -> dict:
    """The shared and pooled 2-cps accuracy amplitudes the data show."""
    obs = {"davidson_coherent": float(dp.COHERENT[0])}
    f = bids_io.deriv_dir("population") / "population_common_phase.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        obs["common_phase"] = d["measures"]["accuracy"]["by_freq"]["2cps"]["amplitude"]
    f = bids_io.deriv_dir("phase") / "pooled_phase.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        obs["pooled"] = d["measures"]["accuracy"]["amp_at_2cps"]
    return obs


def _figure(res: dict, out: Path) -> None:
    """Figure: power against the shared amplitude for the analysed sample and the
    planned design, with the observed and reference amplitudes marked."""
    fig, axes = plt.subplots(1, 2, figsize=(S.COL2_IN * 1.5, 4.3), sharey=True)
    panels = [("analysed", f"Analysed sample (N = {res['analysed']['n_participants']}, "
                           f"{res['analysed']['median_trials']} trials median)"),
              (f"planned_N{PLANNED_N[-1]}", f"Planned design (N = {PLANNED_N[-1]}, "
                                            f"{dp.ADOPTED_TRIALS} trials)")]
    obs = res["observed"]
    marks = [("common_phase", "observed shared amplitude", ":"),
             ("pooled", "observed pooled amplitude", "-."),
             ("davidson_coherent", "Davidson et al., coherent", "--")]
    for ax, (key, title), tag in zip(axes, panels, "ab"):
        cur = res[key]
        x = 100 * np.asarray(cur["amps"])
        for t in TESTS:
            ax.plot(x, cur[t], "o-", color=COLOR[t], lw=1.8, ms=4, label=LABEL[t])
        ax.axhline(0.8, color=S.GREY, lw=0.8, ls="--")
        ax.axhline(0.05, color=S.GREY, lw=0.8, ls=":")
        for k, lab, ls in marks:
            if obs.get(k) is not None:
                ax.axvline(100 * obs[k], color=S.INK, lw=1.0, ls=ls, label=lab)
        ax.set_xlabel("shared 2-cps amplitude (accuracy, pp)")
        S.panel(ax, tag, title, fontsize=10)
    axes[0].set_ylabel("power")
    axes[0].set_ylim(0, 1.02)
    # lower right of b is empty above the 5 % line: the curves have all left it by then
    axes[1].legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.suptitle("Power for a stride-phase oscillation shared by all participants",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Every participant oscillates at 2 cycles per stride with the same amplitude "
                   "and phase; simulated with the validated davidson_power simulator at each "
                   "participant's answered walking trials, blocks at their calibrated operating "
                   "point. Holm over 2 and 4 cps for every test. Dashed line, 80 % power; dotted, "
                   "the 5 % level.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    S.save(fig, out)


def main() -> dict | None:
    """Power curves for a shared oscillation at the analysed and the planned sample."""
    counts = _trial_counts()
    if len(counts) < 4:
        print("Fewer than 4 participants with pooled phase outputs. Run the phase stage first.")
        return None
    rng = np.random.default_rng(SEED)
    print("=== Power for a shared stride-phase oscillation ===")
    print(f"  {len(counts)} participants, answered walking trials {min(counts.values())}-"
          f"{max(counts.values())}")
    res = {"note": "Shared-phase 2-cps accuracy oscillation; davidson_power simulator; "
                   "blocks at their calibrated operating point (MDAs are lower bounds).",
           "observed": _observed()}
    res["analysed"] = {**power_curve(rng, list(counts.values())),
                       "n_participants": len(counts),
                       "median_trials": int(np.median(list(counts.values()))),
                       "trials": counts}
    for n in PLANNED_N:
        res[f"planned_N{n}"] = {**power_curve(rng, [dp.ADOPTED_TRIALS] * n),
                                "n_participants": n, "median_trials": dp.ADOPTED_TRIALS}
    for key in ["analysed"] + [f"planned_N{n}" for n in PLANNED_N]:
        m = res[key]["mda80"]
        print(f"  {key:<14} MDA80: " + ", ".join(
            f"{t} {100 * m[t]:.2f} pp" if m[t] is not None else f"{t} > {100 * AMPS[-1]:.1f} pp"
            for t in TESTS) + f" | level at A=0: agnostic {res[key]['agnostic'][0]:.3f}, "
              f"common {res[key]['common'][0]:.3f}, pooled {res[key]['pooled'][0]:.3f}")
    o = res["observed"]
    print("  observed shared amplitude " + ", ".join(f"{k} {100 * v:.2f} pp" for k, v in o.items()))
    out = bids_io.deriv_dir("shared_power")
    (out / "shared_power.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _figure(res, bids_io.fig_dir("shared_power") / "shared_power.png")
    print(f"Shared-phase power -> {out}")
    return res


if __name__ == "__main__":
    main()
