"""Methods figure: why the walking/standing split changed from 6/4 to 8/2 blocks.

Moving two standing blocks to walking gives a large gain in confirmatory power and a small
loss on the standing comparison, which was already over-powered.

Panel a (gain): power of the confirmatory phase-agnostic population test against walking
trials per participant, for three cohort sizes, old and new designs marked.
Panel b (cost): precision of the standing 75 % threshold against standing trials. The H2
walking-versus-standing comparison is built on this threshold.

Values come from ``davidson_power/design_tradeoff.json`` (regenerate with the simulations
documented in that file), so the figure matches the READMEs. There is no
in-figure caption because ``S.caption`` falls outside a constrained-layout canvas; the
caption is in ``figure_captions.md``.

Run::  python Analysis/figures/make_design_tradeoff.py
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

import diagram_kit as D
from diagram_kit import S

DATA = D.FIG_DIR.parent / "davidson_power" / "design_tradeoff.json"
N_COLOUR = {"12": S.BLUE, "15": S.TEAL, "20": S.PURPLE}
CURVE = dict(marker="o", ms=4, lw=2.0, zorder=3)
AXIS_FONT = dict(fontsize=D.FLOW_PT["body"], color=D.FLOW_INK)


def mark_design(ax, x, tag, color, y_top) -> None:
    """Draw a dashed line for one design, labelled above the data."""
    ax.axvline(x, color=color, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.text(x, y_top, tag, ha="center", va="top", linespacing=1.15, fontsize=D.FLOW_PT["small"],
            color=color, fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.85))


def mark_designs(ax, adopted, arm, y_top) -> None:
    """Mark the old design (muted) and the adopted one."""
    for when, design, color in (("was", adopted["old"], D.FLOW_MUTED), ("now", adopted["new"], D.FLOW_INK)):
        mark_design(ax, design[f"{arm}_trials"], f"{when}\n{design[f'{arm}_blocks']} blocks", color, y_top)


def change(ax, start, end, *, rad, color, text, text_xy, **text_kw) -> None:
    """Draw a curved arrow from old to new design with the change in a box."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=12))
    ax.text(*text_xy, text, fontsize=D.FLOW_PT["small"], color=color, fontweight="semibold",
            ha="center", va="center", **text_kw,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color,
                      linewidth=0.9, alpha=0.95))


def trials_axis(ax, trials, blocks, arm) -> None:
    """Tick every simulated budget, labelled with trials and blocks."""
    ax.set_xticks(trials, labels=[f"{t}\n{b}" for t, b in zip(trials, blocks)],
                  fontsize=D.FLOW_PT["small"])
    ax.set_xlabel(f"{arm} trials / participant  ·  blocks", **AXIS_FONT)


def draw_gain(ax, gain, adopted) -> None:
    trials = gain["walking_trials"]
    for n, color in N_COLOUR.items():
        ax.plot(trials, gain["power"][n], color=color, label=f"N = {n}", **CURVE)
    ax.axhline(0.80, color=D.FLOW_MUTED, ls="--", lw=1.0, zorder=1)
    ax.text(724, 0.812, "80 %", fontsize=D.FLOW_PT["small"], color=D.FLOW_MUTED, va="bottom")

    ax.set_ylim(0.44, 1.12)
    mark_designs(ax, adopted, "walking", 1.105)
    x_old, x_new = adopted["old"]["walking_trials"], adopted["new"]["walking_trials"]
    p_old, p_new = (gain["power"]["12"][trials.index(x)] for x in (x_old, x_new))
    change(ax, (x_old + 8, p_old - 0.012), (x_new - 8, p_new - 0.012), rad=0.32, color=S.BLUE,
           text=f"N = 12:\n{p_old:.2f} → {p_new:.2f}", text_xy=(1258, 0.700), linespacing=1.25)

    trials_axis(ax, trials, gain["walking_blocks"], "walking")
    ax.set_ylabel("power, confirmatory test", **AXIS_FONT)
    # Kept left of the "now" line so the line does not cross the legend.
    ax.legend(fontsize=D.FLOW_PT["small"], loc="lower right", bbox_to_anchor=(0.94, 0.01),
              frameon=False, labelcolor=D.FLOW_INK)
    S.panel(ax, "a", "Gain — confirmatory power", fontsize=10)


def draw_cost(ax, cost, adopted) -> None:
    trials, sd = cost["standing_trials"], cost["sd_pct_of_threshold"]
    ax.plot(trials, sd, color=S.ORANGE, **CURVE)

    ax.set_ylim(2.6, 10.4)
    mark_designs(ax, adopted, "standing", 10.25)
    x_old, x_new = adopted["old"]["standing_trials"], adopted["new"]["standing_trials"]
    sd_old, sd_new = (sd[trials.index(x)] for x in (x_old, x_new))
    change(ax, (x_old - 12, sd_old - 0.10), (x_new + 12, sd_new - 0.10), rad=-0.34, color=S.ORANGE,
           text=f"{sd_old:.1f} % → {sd_new:.1f} %", text_xy=(540, 3.05))

    trials_axis(ax, trials, cost["standing_blocks"], "standing")
    ax.set_ylabel("standing threshold SD (% of threshold)", **AXIS_FONT)
    S.panel(ax, "b", "Cost — standing precision", fontsize=10)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    fig, (ax_gain, ax_cost) = plt.subplots(1, 2, figsize=(S.COL2_IN, 3.35), width_ratios=[1.3, 1],
                                           layout="constrained")
    draw_gain(ax_gain, data["gain"], data["adopted"])
    draw_cost(ax_cost, data["cost_threshold"], data["adopted"])
    D.save(fig, "design_tradeoff")


if __name__ == "__main__":
    main()
