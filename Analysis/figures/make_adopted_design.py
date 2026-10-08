"""Methods figure: the adopted collection design.

Each condition is calibrated once and its blocks run back-to-back; the session then switches
once to the other condition. The leading condition comes from a permuted-block allocation
(``ExperimentConductor.ALLOCATION``) and crosses over between the two visits. This is why
there are two visits: each participant fills all four condition x session-half cells, so
the standing-versus-walking threshold contrast is within-participant and free of
time-on-task (``Analysis/threshold_by_condition._crossover_model``).

Styling is shared with ``make_trial_flow`` via ``diagram_kit.FLOW``. Trial budget and power
are in the ``davidson_power`` figures.

Run::  python Analysis/figures/make_adopted_design.py
"""
from __future__ import annotations

import math
from typing import NamedTuple

import matplotlib.pyplot as plt

import diagram_kit as D
import paradigm as P
from diagram_kit import S


class Segment(NamedTuple):
    kind: str          # key into TONE
    minutes: float
    label: str = ""


TONE = {"cal": "blue", "walk": "walk", "stand": "green", "rest": "rest"}
KEY = {"cal": "Calibration", "walk": "Walking block", "stand": "Standing block",
       "rest": "Baseline / break"}

# Lanes: two allocation arms x two visits. The slot letter in ALLOCATION ('S' or 'W') is the
# condition leading visit 1; visit 2 starts with the other. (lane label, standing first?)
LANES = [("slot S · visit 1", True), ("slot S · visit 2", False),
         ("slot W · visit 1", False), ("slot W · visit 2", True)]
LANE_H = 0.58


def stand_group() -> list[Segment]:
    """Standing calibration, then the standing block(s)."""
    return [Segment("cal", P.CAL_MIN, "Cal-S"), *[Segment("stand", P.BLOCK_MIN, "Stand")] * P.N_STAND]


def walk_group() -> list[Segment]:
    """Walking calibration, then the walking blocks with a break after every second one.

    Four blocks are 48 min of walking per visit, so there is a 5 min break every two blocks
    instead of one 3 min break at the midpoint. ``qc.py`` monitors fatigue per block with
    the easy-trial vigilance flag.
    """
    group = [Segment("cal", P.CAL_MIN, "Cal-W")]
    for i in range(1, P.N_WALK + 1):
        group.append(Segment("walk", P.BLOCK_MIN, "Walk"))
        if i % 2 == 0 and i < P.N_WALK:
            group.append(Segment("rest", P.WALK_BREAK_MIN))
    return group


def build_session(standing_first: bool) -> list[Segment]:
    """One visit: baseline, first condition, switch, second condition, cool-down."""
    first, second = stand_group(), walk_group()
    if not standing_first:
        first, second = second, first
    return [Segment("rest", 1), *first, Segment("rest", 3), *second, Segment("rest", 1)]


SESSION_MIN = sum(seg.minutes for seg in build_session(True))
XAXIS_MAX = 20 * math.ceil(SESSION_MIN / 20)    # next multiple of 20 min
XMAX = SESSION_MIN + 32                         # space for the crossover arrows


def draw_lane(ax, y, session, radius, aspect) -> None:
    """Draw one visit as a row of chips, to scale in minutes."""
    x = 0
    for seg in session:
        ax.add_patch(D.chip(x, y, seg.minutes, LANE_H, TONE[seg.kind],
                            radius=min(radius, seg.minutes / 2.2), aspect=aspect, zorder=2))
        # Label only chips wide enough for the text; short ones are identified by colour.
        if seg.minutes >= 10:
            ax.text(x + seg.minutes / 2, y + LANE_H / 2, seg.label, ha="center", va="center",
                    fontsize=D.FLOW_PT["small"], color=D.FLOW_INK, fontweight="semibold", zorder=3)
        x += seg.minutes


def draw_crossover(ax, x, y_top, y_bottom) -> None:
    """Draw a double arrow joining one participant's two visits."""
    ax.annotate("", xy=(x, y_bottom + 0.26), xytext=(x, y_top - 0.26),
                arrowprops=dict(arrowstyle="<->", color=D.FLOW_INK, lw=D.FLOW_LW["arrow"],
                                mutation_scale=13))
    # No icon: draw_icon scales in data units and would be distorted on these axes.
    ax.text(x + 2.6, (y_top + y_bottom) / 2, "order\ncrosses over", ha="left", va="center",
            fontsize=D.FLOW_PT["small"], color=D.FLOW_INK, fontweight="semibold",
            linespacing=1.2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(S.COL2_IN, 2.75), layout="constrained")
    n = len(LANES)
    lane_y = [n - 1 - i for i in range(n)]          # first lane on top

    ax.set(xlim=(0, XMAX), ylim=(-0.7, n - 0.1))
    radius, aspect = D.corner_radius(ax)
    for y, (_, standing_first) in zip(lane_y, LANES):
        draw_lane(ax, y, build_session(standing_first), radius, aspect)
    for top in (0, 2):                              # each participant's visit pair
        draw_crossover(ax, SESSION_MIN + 8, lane_y[top] + LANE_H, lane_y[top + 1])

    ax.set_yticks([y + LANE_H / 2 for y in lane_y], labels=[lab for lab, _ in LANES],
                  fontsize=D.FLOW_PT["body"], fontweight="semibold")
    ax.tick_params(axis="y", length=0, labelcolor=D.FLOW_INK)
    ax.set_xticks(range(0, XAXIS_MAX + 1, 20))
    D.time_axis(ax, 0, XAXIS_MAX, label=f"minutes into the session  ·  ~{SESSION_MIN} min per visit",
                label_y=-0.19)

    handles = [D.chip(0, 0, 1, 1, TONE[k], radius=0.25) for k in TONE]
    ax.legend(handles, [KEY[k] for k in TONE], loc="upper left", bbox_to_anchor=(0, -0.26),
              ncol=len(TONE), fontsize=D.FLOW_PT["body"], frameon=False, handlelength=1.5,
              handleheight=0.9, columnspacing=1.6, labelcolor=D.FLOW_INK)

    D.save(fig, "adopted_design")


if __name__ == "__main__":
    main()
