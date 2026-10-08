"""Precision of the Bayesian population-prevalence estimate as a function of N.

Planning tool, not part of the pipeline. Uses ``bayesian_prevalence`` from
``population.py`` (Ince, Kay & Schyns 2022; Davidson Fig 4 j-l). Given N participants, a
per-participant test sensitivity ``s`` (detection power from the target-count power
simulation) and false-positive rate ``alpha``, how wide is the 96 % HPDI on prevalence?

Model (as in the pipeline):
    a participant tests positive with prob  theta = gamma*s + (1-gamma)*alpha
    k ~ Binomial(N, theta);  uniform prior on gamma;  posterior over gamma.

k is drawn by Monte-Carlo for each scenario; the table reports median MAP and median
96 % HPDI width. The figure shows posteriors at N = 10 vs 36 and HPDI width vs N.

Usage::

    python -m Analysis.prevalence_planning
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .population import bayesian_prevalence, ALPHA
    from . import bids_io
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from population import bayesian_prevalence, ALPHA
    import bids_io
    import plotstyle as S

S.apply()

RNG = np.random.default_rng(0)
N_MC = 2000  # Monte-Carlo draws of k per scenario

# Sensitivities from the target-count power simulation at Davidson's ~1300 targets:
# 2-cps amplitude 0.04 -> ~0.22, 0.06 -> ~0.45, strong/long run -> ~0.70.
# alpha is the permutation test level.
SENS = {"weak (s=0.22)": 0.22, "moderate (s=0.45)": 0.45, "strong (s=0.70)": 0.70}
TRUE_PREV = [0.25, 0.50, 0.75]      # ground-truth population prevalence
N_GRID = [5, 10, 15, 20, 30, 36, 50]


def _hpdi_width_for_k(k: int, n: int) -> tuple[float, float, float]:
    _, _, mapv, (lo, hi) = bayesian_prevalence(k, n, alpha=ALPHA)
    return mapv, lo, hi


def _simulate(true_prev: float, sens: float, n: int):
    """Draw k, compute the posterior, return median MAP and median HPDI bounds."""
    theta = true_prev * sens + (1 - true_prev) * ALPHA
    ks = RNG.binomial(n, theta, size=N_MC)
    maps, los, his = [], [], []
    for k in np.unique(ks):
        m, lo, hi = _hpdi_width_for_k(int(k), n)
        w = np.sum(ks == k)
        maps += [m] * w
        los += [lo] * w
        his += [hi] * w
    maps, los, his = np.array(maps), np.array(los), np.array(his)
    width = his - los
    return {
        "theta": theta,
        "map_med": float(np.median(maps)),
        "lo_med": float(np.median(los)),
        "hi_med": float(np.median(his)),
        "width_med": float(np.median(width)),
        "width_lo": float(np.percentile(width, 25)),
        "width_hi": float(np.percentile(width, 75)),
    }


def _print_table():
    print("=" * 78)
    print("Bayesian population-prevalence precision (96% HPDI width) by N")
    print(f"  posterior: population.bayesian_prevalence | alpha (FPR) = {ALPHA}")
    print("  width = hi - lo of the 96% HPDI; smaller = more precise. Median over")
    print(f"  {N_MC} Monte-Carlo draws of k ~ Binomial(N, theta).")
    print("=" * 78)
    for slab, s in SENS.items():
        print(f"\nper-participant sensitivity = {slab}")
        for tp in TRUE_PREV:
            print(f"  true prevalence = {tp:.2f}")
            print(f"    {'N':>4} | {'med MAP':>8} | {'median 96% HPDI':>20} | {'width':>6}")
            for n in N_GRID:
                r = _simulate(tp, s, n)
                hpdi = f"[{r['lo_med']:.2f}, {r['hi_med']:.2f}]"
                print(f"    {n:>4} | {r['map_med']:>8.2f} | {hpdi:>20} | {r['width_med']:>6.2f}")


def _fig(out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # (A) posterior curves: N=10 vs 36, moderate sensitivity, true prev = 0.5
    s, tp = SENS["moderate (s=0.45)"], 0.50
    theta = tp * s + (1 - tp) * ALPHA
    for n, col in [(10, S.ORANGE), (36, S.BLUE)]:
        k = int(round(n * theta))
        g, post, mapv, (lo, hi) = bayesian_prevalence(k, n, alpha=ALPHA)
        ax[0].plot(g, post, color=col, lw=2, label=f"N = {n} (k = {k}), HPDI width {hi-lo:.2f}")
        ax[0].axvspan(lo, hi, color=col, alpha=0.12)
        ax[0].plot(mapv, post.max(), "o", color=col)
    S.reference_line(ax[0], x=tp, label=f"true prevalence {tp:.2f}")
    ax[0].set_xlabel("population prevalence proportion")
    ax[0].set_ylabel("posterior density")
    S.panel(ax[0], "a", "Posterior: N = 10 vs 36", fontsize=10)
    ax[0].legend(loc="upper right"); ax[0].set_xlim(0, 1)

    # (B) HPDI width vs N, by sensitivity (true prev = 0.5)
    for (slab, sv), col in zip(SENS.items(), (S.RED, S.PURPLE, S.GREEN)):
        widths = [_simulate(0.50, sv, n)["width_med"] for n in N_GRID]
        ax[1].plot(N_GRID, widths, "o-", color=col, label=slab)
    ax[1].axvline(10, color=S.GREY, ls=":", lw=1)
    ax[1].annotate("N = 10", (10.5, 0.96), xycoords=("data", "axes fraction"),
                   fontsize=8.5, color=S.GREY, ha="left", va="top")
    ax[1].set_xlabel("number of participants N")
    ax[1].set_ylabel("median 96% HPDI width (smaller = more precise)")
    S.panel(ax[1], "b", "Prevalence-estimate precision vs N", fontsize=10)
    ax[1].legend(loc="upper right"); ax[1].set_ylim(0, 1)

    fig.suptitle("Population-prevalence planning", fontsize=12, fontweight="semibold",
                 x=0.012, ha="left")
    S.caption(fig, "Moderate per-participant power (s = 0.45), true prevalence 0.5.")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    S.save(fig, out)


def main():
    """Plot how precisely prevalence can be estimated as a function of sample size."""
    _print_table()
    out = bids_io.fig_dir("planning") / "prevalence_precision_vs_N.png"
    _fig(out)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
