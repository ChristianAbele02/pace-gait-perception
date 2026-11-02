"""What the calibration error costs the confirmatory test, at the planned sample size.

``calibration_transfer.py`` shows that the blocks rarely run at the 75 % the calibration
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
  * ``calibration_cost.json``           diagnosis, information retained, power and MDA
  * ``figures/calibration_cost.png`` / ``.pdf``

Usage::

    python -m Analysis.calibration_cost
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
import pandas as pd

try:
    from . import bids_io
    from . import config as C
    from . import stage_cache
    from . import plotstyle as S
    from . import threshold_phase as tp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import plotstyle as S
    import stage_cache
    import threshold_phase as tp

S.apply()


def _simulator():
    """``davidson_power/power.py``, loaded by path (the folder is not a package)."""
    path = Path(__file__).resolve().parent / "davidson_power" / "power.py"
    spec = importlib.util.spec_from_file_location("davidson_power_sim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dp = _simulator()

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
N_SIM, N_NULL = 2000, 3000
SEED = 20261027
TESTS = ("accuracy", "weighted")
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


def power_curve(rng, npp: int, draw, scales, nsim: int = N_SIM, nnull: int = N_NULL) -> dict:
    """Power of both tests at each effect scale, against one null simulated under the
    same calibration error."""
    null = np.sort(study_stats(rng, npp, nnull, draw, 0.0), axis=0)
    out = {t: [] for t in TESTS}
    for sc in scales:
        s = study_stats(rng, npp, nsim, draw, sc)
        exceed = nnull - np.stack([[np.searchsorted(null[:, a, f], s[:, a, f]) for f in range(2)]
                                   for a in range(2)], 0)          # (test, freq, nsim)
        p = (1 + exceed) / (1 + nnull)
        for a, name in enumerate(TESTS):
            out[name].append(round(float((p[a].min(0) < dp.ALPHA_ANY).mean()), 4))
    return out


def mda(scales, power, target: float = 0.8) -> float | None:
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
    rng = np.random.default_rng(SEED)
    scen = scenarios(diag)
    obs = np.asarray(diag["walking"]["log_ratios"], float)
    out = {"scales": SCALES, "n_planned": N_PLANNED, "trials": TRIALS, "n_sim": N_SIM,
           "n_null": N_NULL, "scenarios": {}, "by_n": {}}
    for name, sc in scen.items():
        pw = power_curve(rng, N_PLANNED, sc["draw"], SCALES)
        info = (float(information_retained(obs).mean()) if name.startswith("observed")
                else expected_information(sc["mu"], sc["sigma"]))
        out["scenarios"][name] = {
            "mu_log_ratio": round(sc["mu"], 4), "sd_log_ratio": round(sc["sigma"], 4),
            "information_retained": round(info, 4),
            "power": pw, "mda": {t: mda(SCALES, pw[t]) for t in TESTS}}
    for name in ("perfect calibration", "observed (walking)"):
        draw = scen[name]["draw"]
        planned = out["scenarios"][name]["power"]
        out["by_n"][name] = {
            str(n): ({t: [planned[t][SCALES.index(s)] for s in N_SCALES] for t in TESTS}
                     if n == N_PLANNED else power_curve(rng, n, draw, N_SCALES))
            for n in N_GRID}
    out["by_n_grid"], out["by_n_scales"] = N_GRID, N_SCALES
    return out


def _print(sim: dict) -> None:
    """Print information retained, power and smallest detected effect per scenario."""
    for name, sc in sim["scenarios"].items():
        p = {t: sc["power"][t][SCALES.index(1.0)] for t in TESTS}
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


def _figure(units: pd.DataFrame, diag: dict, sim: dict, curves: dict, stem: Path) -> None:
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


def main() -> dict:
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
        stage_cache.code_digest("calibration_cost.py", "threshold_phase.py",
                                "davidson_power/power.py"),
        diag["walking"]["log_ratios"], [N_SIM, N_NULL, SEED, N_PLANNED, TRIALS])
    if stage_cache.is_fresh("calibration_cost", "simulation", fp, [out / OUTPUTS[0]]):
        print("  ratios and code unchanged since the last simulation: reusing it")
        saved = json.loads((out / OUTPUTS[0]).read_text(encoding="utf-8"))
        _print(saved["simulation"])
        _figure(units, diag, saved["simulation"], saved["information_curves"],
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
                       "the planned N (calibration_cost.py).",
        "Model": {"beta": tp.BETA, "lapse": tp.LAPSE, "guess": tp.GUESS,
                  "levels_accuracy": LEVEL_PC.tolist(), "levels_weight": LEVEL_W.tolist(),
                  "quest_stop_sd_log10": QUEST_STOP_SD_LOG10,
                  "projected_stop_sd_log10": PROJECTED_STOP_SD_LOG10, "seed": SEED},
        "diagnosis": diag, "information_curves": curves, "simulation": sim}, indent=2),
        encoding="utf-8")
    stage_cache.record("calibration_cost", "simulation", fp, [out / OUTPUTS[0]])
    _figure(units, diag, sim, curves, bids_io.fig_dir("calibration_cost") / "calibration_cost")
    print(f"Outputs -> {out}")
    return sim


if __name__ == "__main__":
    main()
