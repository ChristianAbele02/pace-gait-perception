"""Methods figure: session and trial flow.

Shows the software flow: setup scene, per-condition QUEST+ calibration, experiment block
and its trial loop. The visit structure is in ``make_adopted_design.py``. Paradigm numbers
come from :mod:`paradigm`.

Blue scene boxes, green pills, orange decisions and Tabler icons, drawn at
:data:`plotstyle.COL2_IN` (183 mm) with no text below 8 pt.

Run::  python Analysis/figures/make_trial_flow.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

import diagram_kit as D
import paradigm as P
from diagram_kit import S
from icons_svg import draw_icon

# 100 canvas units = 1 inch, so the axes fill the COL2_IN-wide figure without letterboxing.
# The y range is cropped to the drawing.
W, Y0, Y1 = 720, 80, 910
Z_PANEL, Z_ARROW, Z_NODE, Z_TEXT = 1, 2, 3, 5
GREEN_TEXT = "#4A6B34"
TOGGLE_W, TOGGLE_H, TOGGLE_GAP = 28, 12, 8


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


class FlowCanvas:
    """Equal-aspect canvas with flowchart drawing methods."""

    fig: Figure
    ax: Axes

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(S.COL2_IN, (Y1 - Y0) / 100))
        self.ax.set_position([0, 0, 1, 1])
        self.ax.set(xlim=(0, W), ylim=(Y0, Y1), aspect="equal")
        self.ax.axis("off")
        self.ax.grid(False)
        self.fig.canvas.draw()           # fix layout so text_width is accurate

    def box(self, cx, cy, w, h, tone="blue", *, lw=D.FLOW_LW["box"], rounding=0.06, z=Z_NODE) -> None:
        """Rounded box centred on ``(cx, cy)``; corner radius is a fraction of the short side."""
        self.ax.add_patch(D.chip(cx - w / 2, cy - h / 2, w, h, tone, radius=rounding * min(w, h),
                                 lw=lw, zorder=z))

    def pill(self, cx, cy, w=250, h=36) -> None:
        self.box(cx, cy, w, h, "green", lw=D.FLOW_LW["pill"], rounding=0.5)

    def decision(self, cx, cy, hw, hh, text) -> None:
        """Orange decision diamond."""
        fill, edge = D.FLOW["orange"]
        self.ax.add_patch(Polygon([(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
                                  facecolor=fill, edgecolor=edge, linewidth=D.FLOW_LW["box"], zorder=Z_NODE))
        self.text(cx, cy, text)

    def text(self, x, y, s, size=D.FLOW_PT["body"], weight="normal", color=D.FLOW_INK, **kw):
        kw = {"ha": "center", "va": "center", "linespacing": 1.25} | kw
        return self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, zorder=Z_TEXT, **kw)

    def title(self, x, y, s) -> None:
        self.text(x, y, s, size=D.FLOW_PT["title"], weight="bold")

    def text_width(self, s, size=D.FLOW_PT["body"]) -> float:
        return D.text_width(self.ax, s, fontsize=size)

    def arrow(self, *pts, label=None, at=None) -> None:
        """Polyline through ``pts`` with an arrowhead at the end and an optional label."""
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=D.FLOW_INK, lw=D.FLOW_LW["arrow"], solid_capstyle="round",
                     solid_joinstyle="round", zorder=Z_ARROW)
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=Z_ARROW,
                         arrowprops=dict(arrowstyle="-|>", color=D.FLOW_INK, lw=D.FLOW_LW["arrow"],
                                         shrinkA=0, shrinkB=0, mutation_scale=13))
        if label:
            self.text(*at, label, size=D.FLOW_PT["small"], weight="bold")

    def icon(self, name, cx, cy, size=15.0) -> None:
        draw_icon(self.ax, name, cx, cy, size=size, lw=D.FLOW_LW["icon"], color=D.FLOW_INK, zorder=Z_TEXT)

    def icon_row(self, icon_x, y, name, text, gap=20) -> None:
        """Icon at ``icon_x`` with its label left-aligned ``gap`` to the right."""
        self.icon(name, icon_x, y)
        self.text(icon_x + gap, y, text, size=D.FLOW_PT["small"], ha="left")

    def gabor_glyph(self, cx, cy, r=7.5) -> None:
        """Grating in a circle (no stock icon for a Gabor)."""
        self.ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=D.FLOW_INK, lw=D.FLOW_LW["icon"],
                                 zorder=Z_TEXT))
        for dx in (-5, -2.5, 0, 2.5, 5):
            half = 0.92 * (r ** 2 - dx ** 2) ** 0.5
            self.ax.plot([cx + dx] * 2, [cy - half, cy + half], color=D.FLOW_INK, lw=1.0,
                         solid_capstyle="round", zorder=Z_TEXT)

    def toggle(self, x, y) -> None:
        """Toggle switch in the off position."""
        self.ax.add_patch(D.rounded_box(x, y - TOGGLE_H / 2, TOGGLE_W, TOGGLE_H, radius=TOGGLE_H / 2,
                                        facecolor="#3A3A3A", edgecolor="none", zorder=Z_TEXT - 1))
        self.ax.add_patch(Circle((x + TOGGLE_H / 2 + 1, y), TOGGLE_H / 2 - 1.5, facecolor="white",
                                 edgecolor="none", zorder=Z_TEXT))


def draw_setup(c: FlowCanvas) -> None:
    c.box(215, 832, 390, 96, z=Z_PANEL)
    c.title(215, 861, "Scene 0: Setup")
    c.text(215, 840, "Participant  ·  Visit  ·  Counterbalance slot")
    c.text(215, 823, "Block  ·  Trial  ·  Treadmill speed")
    debug = "Debug mode"
    x = 215 - (TOGGLE_W + TOGGLE_GAP + c.text_width(debug)) / 2      # centre toggle and label
    c.toggle(x, 803)
    c.text(x + TOGGLE_W + TOGGLE_GAP, 803, debug, ha="left")
    c.decision(215, 690, 92, 38, "Calibration\nneeded?")


def draw_calibration(c: FlowCanvas) -> None:
    c.box(580, 745, 230, 270, z=Z_PANEL)
    c.title(580, 862, "Scene 2: Calibration")
    c.text(580, 845, "QUEST+ (Bayesian adaptive)", size=D.FLOW_PT["small"], weight="bold",
           color=D.FLOW_DARKBLUE)
    steps = [("flag", f"{P.N_WARMUP} warm-up trials\n({P.WARM_HI}° → {P.WARM_LO}°, excluded)"),
             ("target-arrow", "Min-entropy tilt\nplacement"),
             ("ripple", "Dual trackers A / B\n+ catch trials"),
             ("chart-line", "Precision stop ·\npooled posterior"),
             ("eye", "Threshold → easy /\nhard tilts")]
    for i, (name, text) in enumerate(steps):
        c.icon_row(492, 816 - 40 * i, name, text)
    c.text(580, 626, "→ BIDS  ses-01  _beh.tsv", size=D.FLOW_PT["small"], weight="bold",
           color=D.FLOW_DARKBLUE)


def draw_experiment(c: FlowCanvas) -> None:
    c.box(240, 375, 440, 490, "panel", rounding=0.03, z=Z_PANEL)
    c.title(240, 602, "Scene 1: Experiment")
    c.text(240, 585, f"2-AFC Gabor tilt  ·  difficulty {' / '.join(map(str, P.MIX))}  (THR / EASY / HARD)",
           size=D.FLOW_PT["small"])
    c.text(240, 570, f"ISI {P.ISI_S:.1f} ± {P.ISI_JIT:.2f} s  ·  "
                     f"{P.TRIALS_PER_BLOCK} trials per block, first {P.N_TRAIN} training",
           size=D.FLOW_PT["small"], color=D.FLOW_MUTED)

    blocks = [(132, "Block 0: Standing", [("stopwatch", "Timer-triggered"), ("man", "Treadmill stopped")],
               f"acq-standing  ·  {plural(P.N_STAND, 'block')} / visit"),
              (348, "Block 1: Walking", [("map-pin", "Checkpoint-triggered"), ("treadmill", "Treadmill active")],
               f"acq-walking  ·  {plural(P.N_WALK, 'block')} / visit")]
    for cx, title, rows, acq in blocks:
        c.box(cx, 508, 200, 94, "green", lw=D.FLOW_LW["pill"])
        c.text(cx, 539, title, size=9, weight="bold")
        for i, (name, text) in enumerate(rows):
            c.icon_row(cx - 86, 518 - 19 * i, name, text, gap=18)
        c.text(cx, 476, acq, size=D.FLOW_PT["small"], color=GREEN_TEXT)

    # Trial loop. Icons share one x, set by the widest label so rows stay centred.
    tasks = [("gabor", f"Gabor task  ·  {P.STIM_MS} ms", 362),
             ("keyboard", f"Response window  ·  {P.RESP_S:.0f} s", 308),
             ("database", "Log data → Motion-BIDS", 254)]
    icon_x = 240 - (18 + max(c.text_width(text) for _, text, _ in tasks)) / 2
    c.pill(240, 416)
    c.text(240, 416, "Checkpoint triggered")
    for name, text, cy in tasks:
        c.pill(240, cy)
        if name == "gabor":
            c.gabor_glyph(icon_x, cy)
        else:
            c.icon(name, icon_x, cy)
        c.text(icon_x + 18, cy, text, ha="left")
    c.decision(240, 190, 80, 30, "More\ncheckpoints?")


def draw_arrows(c: FlowCanvas) -> None:
    c.arrow((215, 784), (215, 730))                                          # setup -> decision
    c.arrow((307, 690), (463, 690), label="Yes", at=(385, 701))              # -> calibration
    c.arrow((215, 650), (215, 622), label="No", at=(233, 636))               # -> experiment
    for cx in (132, 348):                                                    # block -> loop
        c.arrow((cx, 461), (cx, 448), (240, 448), (240, 434))
    for y_from, y_to in ((398, 380), (344, 326), (290, 272), (236, 222)):   # down the loop
        c.arrow((240, y_from), (240, y_to))
    c.arrow((160, 190), (34, 190), (34, 416), (113, 416), label="Yes", at=(62, 204))
    c.arrow((240, 160), (240, 100), (10, 100), (10, 820), (18, 820),         # block done -> setup
            label="No", at=(258, 146))
    # Routed over the top to avoid crossing Scene 2.
    c.arrow((580, 882), (580, 900), (215, 900), (215, 882))                  # calibration -> setup


def main() -> None:
    canvas = FlowCanvas()
    for draw in (draw_setup, draw_calibration, draw_experiment, draw_arrows):
        draw(canvas)
    D.save(canvas.fig, "trial_flow")


if __name__ == "__main__":
    main()
