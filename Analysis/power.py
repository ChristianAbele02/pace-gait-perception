"""Power of the design's tests at the data actually collected.

* Per-participant sensitivity: Monte-Carlo detection power of the confirmatory test,
  cached on disk; used to de-bias the prevalence estimate.
* Calibration cost: what the observed calibration error costs in information and power.
* Shared oscillation: power for an oscillation with the same amplitude and phase in every
  participant.

Usage::

    python -m Analysis.power                # calibration cost + shared oscillation
    python -m Analysis.power sensitivity    # (re)measure the per-participant sensitivity


Per-participant sensitivity
===========================

Per-participant detection power (sensitivity) for the 2-cps stride oscillation.

The prevalence model in ``population.py`` (Ince, Kay & Schyns 2022) needs the
within-participant sensitivity ``s``: the probability that a true oscillator is flagged
significant by the permutation test. With the default ``sensitivity=1.0`` the prevalence
MAP is a lower bound, as the planning simulations showed. This module estimates ``s`` to
de-bias it (b1) and reports achieved power per participant (b7).

``s`` is a Monte-Carlo estimate using ``phase_analysis.analyse_measure``, the confirmatory
2 & 4 cps Holm test that ``population.py`` also counts, so both use the same criterion.
Simulated observers have ``n_targets`` trials whose hit probability oscillates at 2 cps
with a reference amplitude; ``s`` is the fraction flagged significant. Results are cached in
``derivatives/planning/sensitivity_cache.json``, keyed by bucketed ``n_targets`` and amplitude.


Power for a shared oscillation
==============================

How large a SHARED stride-phase oscillation would the analysed sample have detected?

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

* phase-agnostic ``S`` (the confirmatory test, ``population``), Holm over 2 and
  4 cps, against a null simulated at the same trial counts;
* common phase, Hotelling T2 on the participants' coefficients (``population``),
  Holm over 2 and 4 cps;
* pooled, every trial pooled into one curve (``phase_analysis.pooled_analysis``),
  approximated by the trial-weighted mean coefficient against its sampling variance:
  ``|z|^2 / Var z`` is exponential with mean 1 under H0, Holm over 2 and 4 cps.

For each test the result is the power curve over ``A`` and the smallest shared amplitude
detected with 80 % power (MDA). Marked for comparison: the shared 2-cps amplitude the
data show (``population``: the length of the participants' mean coefficient), the
amplitude of the pooled curve (``phase_analysis`` pooled), and Davidson et al.'s
coherent group amplitude. The same is computed for the planned design (N = 12 and 20 at
1440 trials) so the interim result can be read against the final one.

The model assumes every block runs at its calibrated operating point; with the observed
calibration error (see the calibration cost) the power is lower, so the MDAs are lower bounds.

Outputs (``derivatives/shared_power/``): ``shared_power.json``,
``figures/shared_power.png``.


The cost of the calibration error
=================================

What the calibration error costs the confirmatory test, at the planned sample size.

``conditions.py`` shows that the blocks rarely run at the 75 % the calibration
aimed for. This stage asks what that does to the confirmatory phase-agnostic test, and
what a better calibration or the information-weighted test (``threshold_phase.py``)
would win back. It has three parts.

1. Where the error comes from. Per calibration, the transfer ratio (block threshold /
calibrated threshold) against the calibrated threshold:

* If the calibration only carries estimation error, extreme calibrations regress toward
  the participant's real threshold in the blocks. Then ``ln ratio`` falls with
  ``ln calibrated``, and the slope of ``ln block`` on ``ln calibrated`` is below 1.
* The spread of ``ln ratio``, with the block sampling removed, is compared with the
  posterior SD at which QUEST+ stops (``QUEST_STOP_SD_LOG10``). If the two agree, the
  calibration is as precise as its stopping rule makes it, and no more.

2. Information retained. A stride-phase modulation of the threshold moves accuracy
by ``dP/dln(alpha)``, and a trial at accuracy ``P`` carries Fisher information
``(dP/dln alpha)^2 / (P (1 - P))`` about it (``threshold_phase.information``). A block at
ratio ``r`` puts its three tilt levels at other points of the function than planned.
The information retained is the design-weighted mean over the levels, relative to a
perfectly calibrated block, as a function of the calibration error: ``ln r ~ N(mu, sigma)``.

3. Power. The ``davidson_power`` simulator (validated against the pipeline's own
estimators by ``python -m Analysis.davidson_power.power validate``) draws each simulated
participant's walking trials. Here every visit gets its own transfer ratio, which sets
the accuracy of each tilt level and how far a threshold oscillation moves it. Davidson's
effect is defined at 75 %. Both tests run on the same simulated studies:

* the confirmatory accuracy test (cells centred, unweighted), and
* the information-weighted test (each visit's threshold fitted, residuals weighted by
  ``h(P)``).

Each is the studentised phase-agnostic sum against a null simulated under the same
calibration error, Holm over 2 and 4 cps. The scenarios:

* perfect calibration (what the power analysis assumed),
* the observed walking calibrations (ratios resampled from ``calibration_transfer``), and
* a projection for QUEST+ stopping at ``PROJECTED_STOP_SD_LOG10``: the observed bias, and
  the observed spread with the stopping-rule part narrowed accordingly.

At N = 20 and Davidson's full effect the power stays high in every scenario, so the cost
shows mainly in the power curve over effect size, in the smallest effect detected with
80 % power (MDA), and in power against N.

The simulation assumes the rig's psychometric model (Weibull, slope 2.5, lapse 0.05) is
true. The resampled ratios also carry each block's sampling error (SD about 0.04 in ln for
four blocks), so they overstate the calibration error slightly.

Outputs (``derivatives/calibration_cost/``):
  * ``calibration_cost.json`` diagnosis, information retained, power and MDA
  * ``figures/calibration_cost.png`` / ``.pdf``
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import bids_io
    from . import config as C
    from . import phase_analysis as pa
    from . import plotstyle as S
    from . import stage_cache
    from . import threshold_phase as tp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import phase_analysis as pa
    import plotstyle as S
    import stage_cache
    import threshold_phase as tp

S.apply()


# ###########################################################################
# Per-participant sensitivity
# ###########################################################################
# Reference 2-cps amplitude on the 0..1 hit-rate scale that defines a true oscillation.
# 0.05 is a modest modulation in line with Davidson; override per call for curves.
REF_AMP = 0.05
BASE_RATE = 0.75            # threshold-level base hit rate
SIM_N = 160                 # Monte-Carlo observers per cell
SIM_NPERM = 300             # permutations per simulated test (real analysis: 1000)
RNG = np.random.default_rng(7)

_CACHE_PATH = C.DERIV_DIR / "planning" / "sensitivity_cache.json"


def _bucket(n_targets: int) -> int:
    """Bucket trial counts so the cache is reusable across nearby Ns."""
    if n_targets <= 0:
        return 0
    step = 50 if n_targets < 400 else 100
    return int(round(n_targets / step) * step)


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except (ValueError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _simulate_sensitivity(n_targets: int, amp: float, base: float = BASE_RATE,
                          n_sim: int = SIM_N, nperm: int = SIM_NPERM) -> float:
    """Fraction of simulated 2-cps oscillators flagged significant (= sensitivity)."""
    if n_targets < 12:
        return 0.0
    w = 2 * np.pi * 2.0 / 100.0     # 2 cps on the 0..100 % stride axis
    hits = 0
    for _ in range(n_sim):
        phase = RNG.uniform(0, 100, n_targets)
        phi0 = RNG.uniform(0, 2 * np.pi)
        p = np.clip(base + amp * np.cos(w * phase + phi0), 0.01, 0.99)
        correct = (RNG.uniform(size=n_targets) < p).astype(float)
        res = pa.analyse_measure(phase, correct, "mean", nperm=nperm)
        if res is not None and res["significant"]:
            hits += 1
    return hits / n_sim


def sensitivity(n_targets: int, amp: float = REF_AMP, *, use_cache: bool = True) -> float:
    """Per-participant detection power for a 2-cps oscillation of amplitude ``amp``.

    Cached by bucketed ``n_targets`` and amplitude. This is the ``s`` that de-biases
    the population prevalence estimate.
    """
    # Version tag: bump it when the significance criterion changes. v2 = 2 & 4 cps Holm.
    key = f"v2_n{_bucket(n_targets)}_a{amp:.3f}"
    cache = _load_cache() if use_cache else {}
    if key in cache:
        return float(cache[key])
    s = _simulate_sensitivity(_bucket(n_targets), amp)
    if use_cache:
        cache[key] = s
        _save_cache(cache)
    return s


def mean_sensitivity(n_targets_list, amp: float = REF_AMP) -> float:
    """Mean detection power across participants given each one's trial count."""
    vals = [sensitivity(int(n), amp) for n in n_targets_list if n and n > 0]
    return float(np.mean(vals)) if vals else 1.0


# ===========================================================================
# CLI: print a sensitivity table (and warm the cache)
# ===========================================================================
def sensitivity_main():
    """Measure and cache the per-participant detection sensitivity used to de-bias prevalence."""
    print("Per-participant detection power (sensitivity) for a 2-cps oscillation")
    print(f"  base hit rate {BASE_RATE}, {SIM_N} sims x {SIM_NPERM} perms, "
          f"test = phase_analysis.analyse_measure\n")
    amps = [0.03, 0.05, 0.07]
    ns = [80, 200, 400, 800, 1300, 2000]
    header = "  n_targets | " + " | ".join(f"amp={a:.2f}" for a in amps)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in ns:
        cells = " | ".join(f"{sensitivity(n, a)*100:5.0f}%" for a in amps)
        print(f"  {n:9d} | {cells}")
    print(f"\nCache -> {_CACHE_PATH}")


# ###########################################################################
# Power for a shared oscillation
# ###########################################################################
def _simulator():
    """``davidson_power/power.py``, loaded by path (the folder is not a package)."""
    path = Path(__file__).resolve().parent / "davidson_power" / "power.py"
    spec = importlib.util.spec_from_file_location("davidson_power_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dp = _simulator()

AMPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]) / 100.0
SHARED_N_SIM, SHARED_N_NULL = 2000, 5000
PLANNED_N = (12, 20)
SHARED_SEED = 20261028
SHARED_TESTS = ("agnostic", "common", "pooled")
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


def shared_power_curve(rng, ntrials, *, amps=AMPS, nsim=SHARED_N_SIM, nnull=SHARED_N_NULL) -> dict:
    """Power of each test over the shared amplitudes, for one set of trial counts."""
    null = statistics(*study_coefficients(rng, ntrials, 0.0, nnull), ntrials)
    null_s = np.sort(null["S"], 0)
    out = {t: [] for t in SHARED_TESTS}
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
            "mda80": {t: shared_mda(amps, out[t]) for t in SHARED_TESTS}}


def shared_mda(amps, power, target: float = 0.80) -> float | None:
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


def _shared_figure(res: dict, out: Path) -> None:
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
        for t in SHARED_TESTS:
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


def shared_main() -> dict | None:
    """Power curves for a shared oscillation at the analysed and the planned sample."""
    counts = _trial_counts()
    if len(counts) < 4:
        print("Fewer than 4 participants with pooled phase outputs. Run the phase stage first.")
        return None
    rng = np.random.default_rng(SHARED_SEED)
    print("=== Power for a shared stride-phase oscillation ===")
    print(f"  {len(counts)} participants, answered walking trials {min(counts.values())}-"
          f"{max(counts.values())}")
    res = {"note": "Shared-phase 2-cps accuracy oscillation; davidson_power simulator; "
                   "blocks at their calibrated operating point (MDAs are lower bounds).",
           "observed": _observed()}
    res["analysed"] = {**shared_power_curve(rng, list(counts.values())),
                       "n_participants": len(counts),
                       "median_trials": int(np.median(list(counts.values()))),
                       "trials": counts}
    for n in PLANNED_N:
        res[f"planned_N{n}"] = {**shared_power_curve(rng, [dp.ADOPTED_TRIALS] * n),
                                "n_participants": n, "median_trials": dp.ADOPTED_TRIALS}
    for key in ["analysed"] + [f"planned_N{n}" for n in PLANNED_N]:
        m = res[key]["mda80"]
        print(f"  {key:<14} MDA80: " + ", ".join(
            f"{t} {100 * m[t]:.2f} pp" if m[t] is not None else f"{t} > {100 * AMPS[-1]:.1f} pp"
            for t in SHARED_TESTS) + f" | level at A=0: agnostic {res[key]['agnostic'][0]:.3f}, "
              f"common {res[key]['common'][0]:.3f}, pooled {res[key]['pooled'][0]:.3f}")
    o = res["observed"]
    print("  observed shared amplitude " + ", ".join(f"{k} {100 * v:.2f} pp" for k, v in o.items()))
    out = bids_io.deriv_dir("shared_power")
    (out / "shared_power.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _shared_figure(res, bids_io.fig_dir("shared_power") / "shared_power.png")
    print(f"Shared-phase power -> {out}")
    return res


# ###########################################################################
# The cost of the calibration error
# ###########################################################################
QUEST_STOP_SD_LOG10 = 0.10         # StaircaseManager: calibration stops at this pooled posterior SD
PROJECTED_STOP_SD_LOG10 = 0.07     # the tighter stopping rule projected
N_PLANNED = 20
VISITS = 2
TRIALS = dp.ADOPTED_TRIALS         # walking trials per participant over both visits
LEVEL_PC = dp.MIX.base             # planned accuracy of the THR / EASY / HARD levels
LEVEL_W = dp.MIX.weight
U0 = tp._u(LEVEL_PC)               # Weibull u of each level at the calibrated threshold
SCALES = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2]      # effect, as a multiple of Davidson's
N_GRID = [8, 11, 14, 17, 20, 24]
N_SCALES = [0.7, 1.0]
COST_N_SIM, COST_N_NULL = 2000, 3000
COST_SEED = 20261027
COST_TESTS = ("accuracy", "weighted")
LOG_ALPHA = np.linspace(-2.5, 2.5, 241)       # per-visit threshold fit, relative to calibrated


# ===========================================================================
# 1. Diagnosis
# ===========================================================================
def ratio_from_accuracy(acc):
    """Transfer ratio at which the THR level lands at accuracy ``acc``."""
    pc = np.clip(np.asarray(acc, float), tp.P_MIN, tp.P_MAX)
    return (tp._u(tp.DESIGN_PC) / tp._u(pc)) ** (1 / tp.BETA)


def log_ratios(units: pd.DataFrame) -> np.ndarray:
    """ln transfer ratio per calibration; a fit censored at ceiling or chance is placed
    where its THR accuracy puts it."""
    r = units["ratio"].astype(float).to_numpy()
    cens = units["censored"].astype(str).to_numpy() != "-"
    r[cens] = ratio_from_accuracy(units.loc[cens, "thr_accuracy"].astype(float))
    return np.log(r)


def diagnosis(units: pd.DataFrame) -> dict:
    """Structure of the calibration error from the transfer table: how extreme
    calibrations regress in the blocks, the spread of ln(block / calibrated) beyond block
    sampling against the QUEST+ stopping rule, and the walking ratios."""
    unc = units[units["censored"].astype(str) == "-"]
    lc = np.log(unc["calibrated_t75_deg"].astype(float).to_numpy())
    lb = np.log(unc["block_t75_deg"].astype(float).to_numpy())
    lr = lb - lc
    sd_obs = float(np.std(lr, ddof=1))
    sd_samp = float(np.sqrt(np.mean(unc["null_sd_log"].astype(float) ** 2)))
    walk = units[units["condition"] == "walking"]
    wl = log_ratios(walk)
    wunc = walk["censored"].astype(str) == "-"
    w_samp = float(np.sqrt(np.mean(walk.loc[wunc, "null_sd_log"].astype(float) ** 2)))
    return {
        "n_calibrations": int(len(unc)),
        "corr_log_calibrated_log_ratio": round(float(np.corrcoef(lc, lr)[0, 1]), 3),
        "slope_log_block_on_log_calibrated": round(float(np.polyfit(lc, lb, 1)[0]), 3),
        "sd_log_ratio_observed": round(sd_obs, 4),
        "sd_log_ratio_sampling": round(sd_samp, 4),
        "sd_log_ratio_calibration": round(float(np.sqrt(max(sd_obs ** 2 - sd_samp ** 2, 0))), 4),
        "sd_log_quest_stop": round(QUEST_STOP_SD_LOG10 * np.log(10), 4),
        "median_ratio": {c: round(float(g["ratio"].astype(float).median()), 3)
                         for c, g in units.groupby("condition")},
        "walking": {"n": int(len(walk)), "n_participants": int(walk["participant"].nunique()),
                    "mean_log_ratio": round(float(wl.mean()), 4),
                    "sd_log_ratio": round(float(wl.std(ddof=1)), 4),
                    "sd_log_ratio_calibration": round(float(np.sqrt(max(
                        wl.std(ddof=1) ** 2 - w_samp ** 2, 0))), 4),
                    "log_ratios": [round(float(v), 4) for v in wl]},
    }


# ===========================================================================
# 2. Information retained
# ===========================================================================
def level_accuracy(log_r):
    """Accuracy of each tilt level, ``(..., 3)``, in a block whose true threshold is
    ``exp(log_r)`` x the calibrated one."""
    u = U0 / np.exp(np.asarray(log_r, float))[..., None] ** tp.BETA
    return tp.GUESS + (1 - tp.GUESS - tp.LAPSE) * -np.expm1(-u)


def information_retained(log_r) -> np.ndarray:
    """Design-weighted information of a block at ratio ``exp(log_r)``, relative to a
    perfectly calibrated block."""
    pc = np.clip(level_accuracy(log_r), 0.5 + 1e-9, 1 - tp.LAPSE - 1e-9)
    return (tp.efficiency(pc) @ LEVEL_W) / float(tp.efficiency(LEVEL_PC) @ LEVEL_W)


def expected_information(mu: float, sigma: float, n: int = 801) -> float:
    """``E[information_retained]`` over ``ln r ~ N(mu, sigma)`` (normal quantiles)."""
    if sigma <= 0:
        return float(information_retained(mu))
    from scipy.stats import norm
    q = norm.ppf((np.arange(n) + 0.5) / n)
    return float(information_retained(mu + sigma * q).mean())


# ===========================================================================
# 3. Power
# ===========================================================================
def visit_mix(log_r: np.ndarray):
    """The simulator's difficulty mix for ``log_r`` ``(m, VISITS)``: one cell per visit x
    level, each with its own accuracy and its own share of the threshold oscillation."""
    pc = level_accuracy(log_r)                                    # (m, V, 3)
    gain = tp.slope(pc) / tp.slope(tp.DESIGN_PC)
    m = pc.shape[0]
    return dp.Mix(np.tile(LEVEL_W, VISITS) / VISITS, pc.reshape(m, 1, -1),
                  gain.reshape(m, 1, -1))


def estimate_values(t, vhit: np.ndarray, vmiss: np.ndarray):
    """``dp.estimate`` for any per-cell trial values: a hit in cell k enters as
    ``vhit[k]``, a miss as ``vmiss[k]``. Same occupancy-weighted Fourier fit and the
    same closed-form permutation scale ``s^2``."""
    vh, vm = vhit[:, None, :], vmiss[:, None, :]
    sums = (t.hits * vh + (t.n - t.hits) * vm).sum(2)             # (M, bins)
    count = t.n.sum(2)
    total = count.sum(1)
    mean = sums.sum(1) / total
    sq = (t.hits * vh ** 2 + (t.n - t.hits) * vm ** 2).sum((1, 2))
    var = sq / total - mean ** 2
    gram_inv = np.linalg.inv(np.einsum("mb,fbi,fbj->mfij", count, dp.DESIGN, dp.DESIGN))
    beta = np.einsum("mfij,fbj,mb->mfi", gram_inv, dp.DESIGN, sums)
    s2 = (total / (total - 1) * var)[:, None] * (gram_inv[..., 1, 1] + gram_inv[..., 2, 2])
    return beta[..., 1] + 1j * beta[..., 2], s2


def accuracy_values(t):
    """The confirmatory test's values: each cell centred on its mean, unweighted."""
    n, h = t.n.sum(1), t.hits.sum(1)
    shift = (h.sum(1) / n.sum(1))[:, None] - h / np.maximum(n, 1)
    return 1 + shift, shift


def weighted_values(t):
    """The information-weighted test's values: per visit, the threshold is fitted to the
    three levels (fixed model, grid ML), and every cell's residual is weighted by ``h(P)``."""
    m = t.n.shape[0]
    n = t.n.sum(1).reshape(m * VISITS, -1)
    h = t.hits.sum(1).reshape(m * VISITS, -1)
    pg = np.clip(level_accuracy(LOG_ALPHA), 1e-6, 1 - 1e-6)      # (grid, 3)
    ll = h @ np.log(pg).T + (n - h) @ np.log1p(-pg).T
    i = np.clip(ll.argmax(1), 1, LOG_ALPHA.size - 2)
    y0, y1, y2 = (np.take_along_axis(ll, (i + d)[:, None], 1)[:, 0] for d in (-1, 0, 1))
    curv = y0 - 2 * y1 + y2
    off = np.divide(0.5 * (y0 - y2), curv, out=np.zeros_like(curv), where=curv < 0)
    p_hat = level_accuracy(LOG_ALPHA[i] + (LOG_ALPHA[1] - LOG_ALPHA[0]) * off)
    w = tp.score_weight(np.clip(p_hat, tp.P_MIN, tp.P_MAX))
    cell = h / np.maximum(n, 1)
    return (w * (1 - cell)).reshape(m, -1), (-w * cell).reshape(m, -1)


def study_stats(rng, npp: int, nsim: int, draw, scale: float, rows: int = 20_000) -> np.ndarray:
    """Studentised phase-agnostic sums ``(nsim, test, freq)`` over simulated studies."""
    chunk = max(1, rows // npp)
    out = []
    for start in range(0, nsim, chunk):
        k = min(chunk, nsim - start)
        coef = dp.draw_coefficients(rng, (k * npp,), scale)
        t = dp.simulate_tables(rng, coef, TRIALS, visit_mix(draw(rng, (k * npp, VISITS))))
        terms = []
        for values in (accuracy_values, weighted_values):
            z, s2 = estimate_values(t, *values(t))
            terms.append(np.abs(z) ** 2 / s2)
        out.append(np.stack(terms, 1).reshape(k, npp, 2, 2).sum(1))
    return np.concatenate(out)


def cost_power_curve(rng, npp: int, draw, scales, nsim: int = COST_N_SIM, nnull: int = COST_N_NULL) -> dict:
    """Power of both tests at each effect scale, against one null simulated under the
    same calibration error."""
    null = np.sort(study_stats(rng, npp, nnull, draw, 0.0), axis=0)
    out = {t: [] for t in COST_TESTS}
    for sc in scales:
        s = study_stats(rng, npp, nsim, draw, sc)
        exceed = nnull - np.stack([[np.searchsorted(null[:, a, f], s[:, a, f]) for f in range(2)]
                                   for a in range(2)], 0)          # (test, freq, nsim)
        p = (1 + exceed) / (1 + nnull)
        for a, name in enumerate(COST_TESTS):
            out[name].append(round(float((p[a].min(0) < dp.ALPHA_ANY).mean()), 4))
    return out


def cost_mda(scales, power, target: float = 0.8) -> float | None:
    """Effect (x Davidson) detected with ``target`` power, by interpolation."""
    mono = np.maximum.accumulate(power)
    return round(float(np.interp(target, mono, scales)), 3) if mono[-1] >= target else None


def scenarios(diag: dict) -> dict:
    """Ratio distributions to simulate: perfect calibration, the observed walking
    ratios, and the spread a stricter QUEST+ stopping rule would leave."""
    w = diag["walking"]
    obs = np.asarray(w["log_ratios"], float)
    mu = float(w["mean_log_ratio"])
    s_cal = float(w["sd_log_ratio_calibration"])
    ln10 = np.log(10)
    s_proj = float(np.sqrt(max(s_cal ** 2 - (QUEST_STOP_SD_LOG10 * ln10) ** 2
                               + (PROJECTED_STOP_SD_LOG10 * ln10) ** 2, 0.0)))
    return {
        "perfect calibration": {"draw": lambda rng, shape: np.zeros(shape), "mu": 0.0, "sigma": 0.0},
        "observed (walking)": {"draw": lambda rng, shape: rng.choice(obs, shape), "mu": mu,
                               "sigma": float(obs.std(ddof=1))},
        f"projected: QUEST+ stop at {PROJECTED_STOP_SD_LOG10:.2f} log10": {
            "draw": lambda rng, shape: mu + s_proj * rng.standard_normal(shape), "mu": mu,
            "sigma": s_proj},
    }


def simulate(diag: dict) -> dict:
    """Power of the confirmatory and information-weighted tests against effect size
    for each scenario, and against N at the reference effect sizes."""
    rng = np.random.default_rng(COST_SEED)
    scen = scenarios(diag)
    obs = np.asarray(diag["walking"]["log_ratios"], float)
    out = {"scales": SCALES, "n_planned": N_PLANNED, "trials": TRIALS, "n_sim": COST_N_SIM,
           "n_null": COST_N_NULL, "scenarios": {}, "by_n": {}}
    for name, sc in scen.items():
        pw = cost_power_curve(rng, N_PLANNED, sc["draw"], SCALES)
        info = (float(information_retained(obs).mean()) if name.startswith("observed")
                else expected_information(sc["mu"], sc["sigma"]))
        out["scenarios"][name] = {
            "mu_log_ratio": round(sc["mu"], 4), "sd_log_ratio": round(sc["sigma"], 4),
            "information_retained": round(info, 4),
            "power": pw, "mda": {t: cost_mda(SCALES, pw[t]) for t in COST_TESTS}}
    for name in ("perfect calibration", "observed (walking)"):
        draw = scen[name]["draw"]
        planned = out["scenarios"][name]["power"]
        out["by_n"][name] = {
            str(n): ({t: [planned[t][SCALES.index(s)] for s in N_SCALES] for t in COST_TESTS}
                     if n == N_PLANNED else cost_power_curve(rng, n, draw, N_SCALES))
            for n in N_GRID}
    out["by_n_grid"], out["by_n_scales"] = N_GRID, N_SCALES
    return out


def _print(sim: dict) -> None:
    """Print information retained, power and smallest detected effect per scenario."""
    for name, sc in sim["scenarios"].items():
        p = {t: sc["power"][t][SCALES.index(1.0)] for t in COST_TESTS}
        print(f"  {name}: information {100 * sc['information_retained']:.0f} %; at N = "
              f"{sim['n_planned']} MDA {sc['mda']['accuracy']} x Davidson (weighted "
              f"{sc['mda']['weighted']}), power at 1x {100 * p['accuracy']:.0f} % "
              f"(weighted {100 * p['weighted']:.0f} %)")


def info_curve(mu: float, sigmas) -> list[float]:
    """Expected information retained at shift ``mu`` for each spread in ``sigmas``."""
    return [round(expected_information(mu, s), 4) for s in sigmas]


# ===========================================================================
# Figure
# ===========================================================================
SCEN_STYLE = {"perfect calibration": (S.INK, "-"), "observed (walking)": (S.COND["walking"], "-")}


def _cost_figure(units: pd.DataFrame, diag: dict, sim: dict, curves: dict, stem: Path) -> None:
    """Figure: calibration error against the calibrated threshold, information
    retained, power at the planned N, and power against N."""
    fig, axes = plt.subplots(2, 2, figsize=(S.COL2_IN, 6.2))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.94, wspace=0.34, hspace=0.48)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    proj = next(k for k in sim["scenarios"] if k.startswith("projected"))
    style = {**SCEN_STYLE, proj: (S.GREY, "-.")}

    # a: transfer ratio against the calibrated threshold
    band = 10 ** QUEST_STOP_SD_LOG10
    ax_a.axhspan(1 / band, band, color=S.GRID, lw=0, zorder=0)
    S.reference_line(ax_a, y=1.0)
    unc = units["censored"].astype(str) == "-"
    shown_bound = False
    for cond, g in units.groupby("condition"):
        u = g[unc[g.index]]
        ax_a.plot(u["calibrated_t75_deg"], u["ratio"], "o", ms=4.5, color=S.COND[cond],
                  mec="white", mew=0.6, zorder=3, label=cond)
        c = g[~unc[g.index]]
        if len(c):
            ax_a.plot(c["calibrated_t75_deg"], ratio_from_accuracy(c["thr_accuracy"]), "v",
                      ms=5, color=S.COND[cond], zorder=3,
                      label=None if shown_bound else "ceiling (bound)")
            shown_bound = True
    lc = np.log(units.loc[unc, "calibrated_t75_deg"].astype(float))
    lr = np.log(units.loc[unc, "ratio"].astype(float))
    b1, b0 = np.polyfit(lc, lr, 1)
    xs = np.geomspace(units["calibrated_t75_deg"].min() * 0.9, units["calibrated_t75_deg"].max() * 1.1, 50)
    ax_a.plot(xs, np.exp(b0 + b1 * np.log(xs)), color=S.INK, lw=1.1, zorder=2,
              label=f"fit, r = {diag['corr_log_calibrated_log_ratio']:.2f}")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    for axis in (ax_a.xaxis, ax_a.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(plt.NullFormatter())
    ax_a.set_xticks([0.25, 0.5, 1, 2, 4])
    ax_a.set_yticks([0.5, 0.75, 1, 1.5, 2])
    ax_a.set_xlabel("Calibrated threshold (deg)")
    ax_a.set_ylabel("Block / calibrated threshold")
    ax_a.plot([], [], "s", ms=7, color=S.GRID, label="QUEST+ stop, ±1 SD")
    ax_a.legend(loc="upper right", fontsize=8, ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax_a.set_ylim(0.4, 3.2)
    S.panel(ax_a, "a", "Calibration error")

    # b: information retained against the calibration error
    sig = np.asarray(curves["sigmas"])
    ax_b.plot(sig, curves["unbiased"], color=S.INK, lw=1.5, label="no bias")
    ax_b.plot(sig, curves["biased"], color=S.INK, lw=1.5, ls="--",
              label=f"walking bias ×{np.exp(diag['walking']['mean_log_ratio']):.2f}")
    for name, sc in sim["scenarios"].items():
        col, _ = style[name]
        ax_b.plot(sc["sd_log_ratio"], sc["information_retained"], "o", ms=6, color=col,
                  mec="white", mew=0.7, zorder=4,
                  label=("stop at 0.07" if name.startswith("projected") else name.split(" (")[0]))
    for sd, lab in ((QUEST_STOP_SD_LOG10, "stop now"), (PROJECTED_STOP_SD_LOG10, "stop 0.07")):
        x = sd * np.log(10)
        S.reference_line(ax_b, x=x, ls=":")
        ax_b.text(x, 0.27, lab, rotation=90, ha="right", va="bottom", fontsize=8, color=S.GREY)
    ax_b.set_xlim(0, sig.max())
    ax_b.set_ylim(0.25, 1.05)                  # room for the legend under the curves
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax_b.set_xlabel("Calibration error (SD of ln ratio)")
    ax_b.set_ylabel("Information retained")
    ax_b.legend(loc="lower right", fontsize=8, labelspacing=0.25)
    S.panel(ax_b, "b", "Information retained")

    # c: power against effect size at the planned N
    for name, sc in sim["scenarios"].items():
        col, ls = style[name]
        ax_c.plot(sim["scales"], sc["power"]["accuracy"], color=col, ls=ls, lw=1.5,
                  marker="o", ms=3.5, label=name.replace("projected: ", ""))
    obs = sim["scenarios"]["observed (walking)"]["power"]["weighted"]
    ax_c.plot(sim["scales"], obs, color=S.COND["walking"], ls=":", lw=1.5, marker="s", ms=3,
              label="observed, information-weighted")
    S.reference_line(ax_c, y=0.8, ls=":")
    S.reference_line(ax_c, x=1.0)
    ax_c.set_ylim(0, 1.02)
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax_c.set_xlabel("Effect (× Davidson's RMS amplitude)")
    ax_c.set_ylabel("Power")
    ax_c.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)   # hides the reference line behind it
    S.panel(ax_c, "c", f"Power at N = {sim['n_planned']}")

    # d: power against N
    ns = sim["by_n_grid"]
    for name in ("perfect calibration", "observed (walking)"):
        col, _ = style[name]
        for j, sc in enumerate(sim["by_n_scales"]):
            pw = [sim["by_n"][name][str(n)]["accuracy"][j] for n in ns]
            ax_d.plot(ns, pw, color=col, lw=1.5 if sc < 1 else 0.9, ls="-" if sc < 1 else "--",
                      marker="o", ms=3.5 if sc < 1 else 2.5,
                      label=f"{name.split(' (')[0]}, effect {sc:g}×")
    S.reference_line(ax_d, y=0.8, ls=":")
    S.reference_line(ax_d, x=sim["n_planned"])
    S.reference_line(ax_d, x=diag["walking"]["n_participants"], ls=":")
    ax_d.text(diag["walking"]["n_participants"], 0.03, " now", fontsize=8, color=S.GREY)
    ax_d.set_ylim(0, 1.02)
    ax_d.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax_d.set_xlabel("Participants")
    ax_d.set_ylabel("Power")
    ax_d.legend(loc="lower right", fontsize=8, frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)   # hides the reference line behind it
    S.panel(ax_d, "d", "Power against N")
    S.save_pub(fig, stem)


# ===========================================================================
# Main
# ===========================================================================
OUTPUTS = ("calibration_cost.json", "calibration_cost.png", "calibration_cost.pdf")


def cost_main() -> dict:
    """Diagnose the calibration error and price it in information and power."""
    f = C.DERIV_DIR / "calibration_transfer" / "calibration_transfer_units.tsv"
    units = pd.read_csv(f, sep="\t") if f.exists() else pd.DataFrame()
    walking = units[units["condition"] == "walking"] if not units.empty else units
    bids_io.gate_report("calibration_cost", condition=("standing", "walking"),
                        analysed_subjects=sorted(units["participant"].unique())
                        if not units.empty else [])
    if len(walking) < 3:
        bids_io.clear_stale("calibration_cost", *OUTPUTS)
        print("Needs calibration_transfer results for at least 3 walking calibrations. "
              "Run the calib_transfer stage first.")
        return {}
    diag = diagnosis(units)
    print("=== Calibration cost: information and power at the observed calibration error ===")
    print(f"  corr(ln calibrated, ln ratio) = {diag['corr_log_calibrated_log_ratio']}, slope of "
          f"ln block on ln calibrated = {diag['slope_log_block_on_log_calibrated']}")
    print(f"  calibration error SD (ln) {diag['sd_log_ratio_calibration']} vs QUEST+ stopping "
          f"SD {diag['sd_log_quest_stop']}")
    out = bids_io.deriv_dir("calibration_cost")
    # The simulation is reused while the observed ratios, this code and the simulator's
    # code are unchanged.
    fp = stage_cache.fingerprint(
        stage_cache.code_digest("power.py", "threshold_phase.py",
                                "davidson_power/power.py"),
        diag["walking"]["log_ratios"], [COST_N_SIM, COST_N_NULL, COST_SEED, N_PLANNED, TRIALS])
    if stage_cache.is_fresh("calibration_cost", "simulation", fp, [out / OUTPUTS[0]]):
        print("  ratios and code unchanged since the last simulation: reusing it")
        saved = json.loads((out / OUTPUTS[0]).read_text(encoding="utf-8"))
        _print(saved["simulation"])
        _cost_figure(units, diag, saved["simulation"], saved["information_curves"],
                bids_io.fig_dir("calibration_cost") / "calibration_cost")
        return saved["simulation"]
    sim = simulate(diag)
    _print(sim)
    sig = np.linspace(0, 0.5, 26)
    curves = {"sigmas": sig.round(3).tolist(), "unbiased": info_curve(0.0, sig),
              "biased": info_curve(diag["walking"]["mean_log_ratio"], sig)}
    (out / OUTPUTS[0]).write_text(json.dumps({
        "Description": "What the calibration error costs: the transfer ratio's error "
                       "structure, the information the blocks keep and the power of the "
                       "confirmatory and the information-weighted phase-agnostic tests at "
                       "the planned N (power.py).",
        "Model": {"beta": tp.BETA, "lapse": tp.LAPSE, "guess": tp.GUESS,
                  "levels_accuracy": LEVEL_PC.tolist(), "levels_weight": LEVEL_W.tolist(),
                  "quest_stop_sd_log10": QUEST_STOP_SD_LOG10,
                  "projected_stop_sd_log10": PROJECTED_STOP_SD_LOG10, "seed": COST_SEED},
        "diagnosis": diag, "information_curves": curves, "simulation": sim}, indent=2),
        encoding="utf-8")
    stage_cache.record("calibration_cost", "simulation", fp, [out / OUTPUTS[0]])
    _cost_figure(units, diag, sim, curves, bids_io.fig_dir("calibration_cost") / "calibration_cost")
    print(f"Outputs -> {out}")
    return sim


# ###########################################################################
# Pipeline stage
# ###########################################################################
PARTS = {"cost": cost_main, "shared": shared_main, "sensitivity": sensitivity_main}


def main() -> None:
    """Pipeline stage ``power``: the calibration cost, then the shared oscillation."""
    cost_main()
    shared_main()


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    if names:
        for n in names:
            PARTS[n]()
    else:
        main()
