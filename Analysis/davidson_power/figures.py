"""Figures 1-4 and 6 of the Davidson-based power analysis.

Drawn at :data:`plotstyle.COL2_IN` (183 mm), saved as 300 dpi PNG and vector PDF. All
numbers are read from the result JSONs. Tables are in ``README.md`` and captions in
``../figures/figure_captions.md``. Figure 5 (adopted design) is in ``../figures``.

Run::  python -m Analysis.davidson_power.figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plotstyle as S  # noqa: E402

S.apply()
HERE = Path(__file__).resolve().parent
load = lambda name: json.loads((HERE / name).read_text(encoding="utf-8"))
EFFECT, SWEEP, POP, MDA = map(load, ("davidson_effect.json", "sweep_results.json",
                                     "group_power_results.json", "design_compare_results.json"))
SUMMARY, PARTS = EFFECT["summary"], EFFECT["participants"]

TRIALS_PER_BLOCK = 180
ADOPTED_TRIALS = 1440               # 8 walking blocks over two visits
PLANNED_N = (12, 20)
SUPERSEDED = 0.071                  # earlier planning value, shown for comparison
LEGEND = dict(fontsize=8, frameon=True, framealpha=0.96, edgecolor="#DDE1E6", borderpad=0.45,
              labelspacing=0.3, handlelength=1.6)


def rule80(ax, x, ha="right", y=80):
    """Dashed 80 % power line, labelled at ``x``."""
    ax.axhline(y, color=S.GREY, ls=(0, (5, 3)), lw=1.0, zorder=1)
    ax.annotate(f"{y:.0f} % power", (x, y - 2.5), fontsize=8, color=S.GREY, ha=ha, va="top")


def curve(ax, x, p, n, color, **kw):
    """Power curve in % with its 95 % Monte-Carlo band."""
    p = np.asarray(p, float)
    half = 196 * np.sqrt(p * (1 - p) / n)
    ax.fill_between(x, 100 * p - half, 100 * p + half, color=color, alpha=0.18, lw=0, zorder=2)
    return ax.plot(x, 100 * p, color=color, lw=1.8, ms=3.8, zorder=3, **{"marker": "o", **kw})


def percent_axis(ax, label, top=100):
    ax.set_ylabel(label)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, 101, 20))


def crossing(x, y, level):
    """x where a rising curve first reaches ``level`` (NaN if never); dips are treated as MC noise."""
    y = np.maximum.accumulate(np.asarray(y, float))
    return float(np.interp(level, y, x)) if y[0] <= level <= y[-1] else float("nan")


def save(fig, name):
    S.save_pub(fig, HERE / name)
    print(f"wrote {name}.{{png,pdf}}")


def fig1_effect_size():
    """Effect size distribution and the three corrections to the earlier estimate."""
    fig, ax = plt.subplots(1, 2, figsize=(S.COL2_IN, 2.9), layout="constrained", width_ratios=[1, 1.1])
    best = np.array([max(p["amp_2cps_debiased"], p["amp_4cps_debiased"]) for p in PARTS])
    flagged = np.array([p["davidson_significant"] for p in PARTS])
    rms = SUMMARY["rms_amp_2cps"]

    bins = np.linspace(0, 0.10, 21)
    ax[0].hist([best[~flagged], best[flagged]], bins=bins, stacked=True, color=[S.MIST, S.TEAL],
               edgecolor="white", lw=0.6,
               label=[f"not flagged (n = {(~flagged).sum()})", f"flagged (n = {flagged.sum()})"])
    # Lines end below the legend so they are not mistaken for ticks.
    ax[0].axvline(rms, ymax=0.7, color=S.BLUE, ls="--", lw=1.4, label=f"RMS at 2 cps {rms:.3f}")
    ax[0].axvline(SUPERSEDED, ymax=0.7, color=S.INK, ls=":", lw=1.4, label=f"superseded {SUPERSEDED}")
    ax[0].set(xlabel="de-biased amplitude, better of 2 / 4 cps", ylabel="participants",
              xlim=(0, 0.105), ylim=(0, 15), yticks=[0, 5, 10, 15])
    ax[0].legend(loc="upper right", handlelength=1.3, **{k: v for k, v in LEGEND.items() if k != "handlelength"})
    S.panel(ax[0], "A", f"Per-participant amplitude (N = {len(PARTS)})", fontsize=9.5)

    # Each bar removes one inflation, in the order they were found.
    steps = [("as previously used", SUPERSEDED, S.INK),
             ("correct data slice", SUMMARY["amp_max24_davidson_responders"], S.RED),
             ("all participants, not the flagged 8", SUMMARY["amp_max24_median"], S.ORANGE),
             ("fixed 2 cps, not the better of 2 / 4", rms, S.BLUE)]
    y = np.arange(len(steps))[::-1]
    labels, values, colours = zip(*steps)
    ax[1].barh(y, values, color=colours, height=0.6)
    for yi, v in zip(y, values):
        ax[1].annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                       fontsize=8.5, fontweight="semibold")
    ax[1].axvline(SUPERSEDED, color=S.INK, ls=":", lw=0.9, alpha=0.5, zorder=1)
    ax[1].annotate("", xy=(rms, -0.78), xytext=(SUPERSEDED, -0.78),
                   arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.0, shrinkA=0, shrinkB=0))
    ax[1].annotate(f"÷ {SUPERSEDED / rms:.1f}", ((SUPERSEDED + rms) / 2, -0.72), ha="center", va="bottom",
                   fontsize=9, fontweight="bold")
    ax[1].set_yticks(y, labels=labels, fontsize=8)
    ax[1].set(xlabel="amplitude [accuracy units]", xlim=(0, 0.088), ylim=(-1.2, 3.55))
    ax[1].grid(axis="y", visible=False)
    S.panel(ax[1], "B", "Inflations removed in turn", fontsize=9.5)

    fig.suptitle("Davidson (2024) stride-phase effect size", fontsize=11, fontweight="semibold",
                 x=0.008, ha="left")
    save(fig, "fig1_effect_size")


def fig2_per_participant_power():
    """Per-participant detection power against trial count."""
    fig, ax = plt.subplots(figsize=(S.COL2_IN, 2.9), layout="constrained")
    ns = np.array(SWEEP["ns"])
    for (label, amp), c in zip(SWEEP["amps"].items(), (S.BLUE, S.ORANGE, S.RED)):
        s = [SWEEP["res"][str(n)][label]["s"] for n in ns]
        curve(ax, ns, s, SWEEP["nsim"], c)
        # Labels at the line ends; a legend would overlap the curves.
        ax.annotate(f"{amp:.3f}", (ns[-1], 100 * s[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=c, fontweight="semibold")
    ax.annotate("amplitude", (ns[-1] + 40, 112), fontsize=8, color=S.GREY)
    rule80(ax, 2150)
    for x, text, color, ha in ((ADOPTED_TRIALS, f"adopted\n{ADOPTED_TRIALS}", S.TEAL, "center"),
                               (836, "Davidson\n836", S.GREY, "right")):
        ax.axvline(x, color=color, lw=1.3 if color == S.TEAL else 1.0, ls="-" if color == S.TEAL else ":")
        ax.annotate(text, (x - (16 if ha == "right" else 0), 102), fontsize=8, color=color, ha=ha, va="bottom",
                    fontweight="semibold" if color == S.TEAL else "normal", linespacing=1.2)
    ax.set_xlabel("walking trials per participant")
    ax.set_xlim(150, 2450)
    percent_axis(ax, "per-participant\ndetection power (%)", top=128)
    top = ax.secondary_xaxis("top", functions=(lambda x: x / TRIALS_PER_BLOCK, lambda b: b * TRIALS_PER_BLOCK))
    top.set_xlabel(f"walking blocks (× {TRIALS_PER_BLOCK} trials)", fontsize=8.5, labelpad=2)
    top.tick_params(labelsize=8)
    ax.set_title("Per-participant detection is out of reach", loc="left", fontsize=11, fontweight="semibold",
                 pad=26)
    save(fig, "fig2_per_participant_power")


def fig3_population_power():
    """Davidson phase scatter and population power of the agnostic vs common-phase tests."""
    fig = plt.figure(figsize=(S.COL2_IN, 3.0), layout="constrained")
    ax_z, ax_p = fig.subplots(1, 2, width_ratios=[1, 2.1])

    z = np.array([p["coef_2cps"] for p in PARTS])
    lim = 0.105
    ax_z.add_patch(plt.Circle((0, 0), SUMMARY["rms_amp_2cps"], fill=False, color=S.BLUE, ls="--", lw=1.0))
    ax_z.scatter(*z.T, s=14, color=S.GREY, edgecolor="white", lw=0.4, zorder=3)
    mean = z.mean(0)
    ax_z.annotate("", xy=mean, xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=S.RED, lw=2.0, mutation_scale=12), zorder=4)
    # Labels in the empty corners; the arrow sits among the points.
    coh = SUMMARY["group_level"]["2.01"]
    ax_z.text(0.98, 0.03, f"mean vector\nR = {coh['phase_coherence']:.2f}", transform=ax_z.transAxes,
              fontsize=8, color=S.RED, fontweight="semibold", ha="right", va="bottom", linespacing=1.15)
    ax_z.text(0.98, 0.97, f"RMS {SUMMARY['rms_amp_2cps']:.3f}", transform=ax_z.transAxes, fontsize=8,
              color=S.BLUE, ha="right", va="top")
    ax_z.axhline(0, color=S.GRID, lw=0.8, zorder=0)
    ax_z.axvline(0, color=S.GRID, lw=0.8, zorder=0)
    ax_z.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[-0.1, 0, 0.1], yticks=[-0.1, 0, 0.1],
             xlabel="cos coefficient", ylabel="sin coefficient")
    ax_z.grid(False)
    S.panel(ax_z, "A", "Davidson, 2 cps (N = 36)", fontsize=9.5)

    npp = np.array(POP["npp"])
    for key, color, label, marker in (("agnostic", S.BLUE, "phase-agnostic (confirmatory)", "o"),
                                      ("coherent", S.RED, "common-phase", "s")):
        p = [POP["power"][str(n)][str(ADOPTED_TRIALS)][key] for n in npp]
        curve(ax_p, npp, p, POP["nsim"], color, marker=marker, label=label)
    ax_p.axvspan(*PLANNED_N, color=S.MIST, alpha=0.4, lw=0, zorder=0)
    ax_p.annotate("planned N", (np.mean(PLANNED_N), 3), fontsize=8, color=S.GREY, ha="center")
    rule80(ax_p, npp[-1])
    n0 = PLANNED_N[0]
    a0, c0 = (100 * POP["power"][str(n0)][str(ADOPTED_TRIALS)][k] for k in ("agnostic", "coherent"))
    ax_p.annotate("", xy=(n0, a0 - 2), xytext=(n0, c0 + 2), arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.1))
    ax_p.annotate(f"{a0:.0f} % vs {c0:.0f} %\nat N = {n0}", (n0 + 0.7, (a0 + c0) / 2), fontsize=8.5, va="center",
                  fontweight="semibold", linespacing=1.25)
    ax_p.set_xlabel("participants (N)")
    percent_axis(ax_p, "population power (%)")
    ax_p.legend(loc="lower right", **LEGEND)
    S.panel(ax_p, "B", f"Power at {ADOPTED_TRIALS} walking trials", fontsize=9.5)
    fig.suptitle("Idiosyncratic phases: the population test decides the study", fontsize=11,
                 fontweight="semibold", x=0.008, ha="left")
    save(fig, "fig3_population_power")


def fig4_trials_vs_participants():
    """Power grid over trials and participants with the 80 % boundary."""
    fig, ax = plt.subplots(figsize=(S.COL2_IN, 2.9), layout="constrained")
    npp, trials = POP["npp"], POP["ntrials"]
    grid = 100 * np.array([[POP["power"][str(n)][str(t)]["agnostic"] for n in npp] for t in trials])
    cmap = LinearSegmentedColormap.from_list("power", ["#FFFFFF", "#CFE0F1", S.BLUE])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto", origin="lower")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v >= 80 else S.INK, fontweight="semibold" if v >= 80 else "normal")
    # 80 % boundary drawn along cell edges, without interpolating between cells.
    above = grid >= 80
    for i, j in zip(*np.nonzero(above)):
        for di, dj, seg in ((-1, 0, ([j - .5, j + .5], [i - .5] * 2)), (0, -1, ([j - .5] * 2, [i - .5, i + .5]))):
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or not above[ni, nj]:
                ax.plot(*seg, color=S.INK, lw=1.6, solid_capstyle="projecting")
    row = trials.index(ADOPTED_TRIALS)
    cols = [npp.index(n) for n in PLANNED_N]
    ax.add_patch(plt.Rectangle((cols[0] - 0.5, row - 0.5), cols[1] - cols[0] + 1, 1, fill=False,
                               ec=S.ORANGE, lw=2.0, zorder=5))
    ax.set_xticks(range(len(npp)), labels=npp)
    ax.set_yticks(range(len(trials)), labels=[f"{t // TRIALS_PER_BLOCK}  ({t})" for t in trials])
    ax.set(xlabel="participants (N)", ylabel="walking blocks (trials)")
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Confirmatory power (%)  ·  line: 80 %  ·  box: adopted design", loc="left",
                 fontsize=11, fontweight="semibold", pad=8)
    save(fig, "fig4_trials_vs_participants")


def fig6_one_vs_two_visits():
    """Power and detectable effect for one vs two visits."""
    fig, ax = plt.subplots(1, 2, figsize=(S.COL2_IN, 3.05), layout="constrained")
    npp = np.array(MDA["npp"], float)
    ref = MDA["scales"].index(MDA["ref_scale"])
    designs = [("two_visits", "two visits", S.BLUE), ("one_visit", "one visit", S.ORANGE)]
    need, xlim = {}, (npp[0] - 1, npp[-1] + 9)
    for key, name, c in designs:
        trials = MDA["designs"][key]
        p = [MDA["power"][key][str(int(n))][ref] for n in npp]
        curve(ax[0], npp, p, MDA["nsim_ref"], c, label=f"{name} · {trials} trials")
        need[key] = crossing(npp, p, MDA["target"])
        if np.isfinite(need[key]):
            # Vertical stem marking the required N.
            ax[0].plot([need[key]] * 2, [0, 100 * MDA["target"]], color=c, lw=1.0, ls=":")
            ax[0].annotate(f"N = {need[key]:.0f}", (need[key], 3), color=c, fontsize=8.5, fontweight="semibold",
                           ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
        m = np.array([np.nan if v is None else v for v in (MDA["mda"][key][str(int(n))] for n in npp)])
        ax[1].plot(npp, m, "o-", color=c, lw=1.8, ms=3.8)
        last = np.flatnonzero(np.isfinite(m))[-1]
        ax[1].annotate(name, (npp[last], m[last]), xytext=(5, 0), textcoords="offset points", va="center",
                       fontsize=8, color=c, fontweight="semibold")
    rule80(ax[0], xlim[1] - 0.5)
    ax[0].set(xlabel="participants (N)", xlim=xlim)
    percent_axis(ax[0], "population power,\nphase-agnostic test (%)", top=104)
    ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.09), **LEGEND)
    S.panel(ax[0], "A", "Power against sample size", fontsize=9.5)

    rms = MDA["rms_2cps"]
    ax[1].axhline(rms, color=S.INK, ls=":", lw=1.1, zorder=1)
    # Right-aligned to stay clear of the curves.
    ax[1].annotate(f"measured effect {rms:.3f}", (xlim[1] - 0.5, rms), xytext=(0, 4), textcoords="offset points",
                   fontsize=8, ha="right", va="bottom")
    ax[1].set(xlabel="participants (N)", ylabel="detectable amplitude at\n80 % power [accuracy units]", xlim=xlim)
    ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.3f}")
    S.panel(ax[1], "B", "Smallest detectable effect", fontsize=9.5)

    fig.suptitle(f"One visit or two: 80 % power at N = {need['two_visits']:.0f} "
                 f"instead of N = {need['one_visit']:.0f}", fontsize=11, fontweight="semibold", x=0.008, ha="left")
    save(fig, "fig6_one_vs_two_visits")


if __name__ == "__main__":
    for make in (fig1_effect_size, fig2_per_participant_power, fig3_population_power,
                 fig4_trials_vs_participants, fig6_one_vs_two_visits):
        make()
