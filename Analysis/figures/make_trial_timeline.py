"""Methods figure: the trial timeline.

Two consecutive trials to scale in seconds. The stimulus is the Gabor rendered from the
shader equations, tilted left on the first trial and right on the second. Each response
window shows the response given (correct, then an error). Timings come from :mod:`paradigm`.

Run::  python Analysis/figures/make_trial_timeline.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

import diagram_kit as D
import paradigm as P
from diagram_kit import S

# Real threshold tilts (a few degrees) are invisible at this size, so the tilt is exaggerated
# (stated in the caption). Spatial frequency, envelope and contrast are the shader values.
DEMO_TILT = 20.0
GABOR_ZOOM = 0.145             # printed large; the patch is the focus

# (tilt, response, correct?). The second trial is an error so the figure shows that the
# window records the response, not the correct answer.
TRIALS = [(-DEMO_TILT, "left", True), (DEMO_TILT, "left", False)]
T_END = len(TRIALS) * P.ISI_S

Y_STIM, Y_RESP, BAR_H, Y_ISI = 1.95, 0.30, 0.6, 2.94


def place_gabor(ax, t, y, tilt) -> AnnotationBbox:
    """Place a Gabor thumbnail at time ``t``.

    ``AnnotationBbox`` keeps a fixed size on the page; ``imshow`` would be stretched because
    the x axis is seconds and the y axis is lanes.
    """
    image = OffsetImage(D.gabor_rgba(tilt), zoom=GABOR_ZOOM, interpolation="bilinear")
    return ax.add_artist(AnnotationBbox(image, (t, y), frameon=False, box_alignment=(0.5, 0.5),
                                        pad=0.0, zorder=4))


def draw_trial(ax, t0, tilt, given, correct, radius, aspect) -> None:
    """Draw one trial: onset line, stimulus and response window."""
    ax.axvline(t0, color=D.FLOW["rest"].edge, lw=0.9, zorder=0)
    place_gabor(ax, t0, Y_STIM, tilt)
    ax.add_patch(D.chip(t0, Y_RESP, P.RESP_S, BAR_H, "green", radius=radius, aspect=aspect, zorder=2))
    # Tick/cross in addition to green/red, which is lost in greyscale and for colour-blind readers.
    ax.text(t0 + P.RESP_S / 2, Y_RESP + BAR_H / 2, f"“{given}”  {'✓' if correct else '✗'}",
            fontsize=D.FLOW_PT["body"], ha="center", va="center", fontweight="semibold", zorder=3,
            color=D.FLOW["green" if correct else "red"].edge)


def draw_isi(ax) -> None:
    """Draw the onset-to-onset ISI and the jitter range of the next onset."""
    ax.annotate("", xy=(P.ISI_S, Y_ISI), xytext=(0, Y_ISI),
                arrowprops=dict(arrowstyle="<->", color=D.FLOW_INK, lw=D.FLOW_LW["arrow"],
                                mutation_scale=13))
    ax.text(P.ISI_S / 2, Y_ISI + 0.07, f"ISI {P.ISI_S:.1f} s", fontsize=D.FLOW_PT["body"],
            ha="center", va="bottom", fontweight="semibold", color=D.FLOW_INK)
    ax.add_patch(Rectangle((P.ISI_S - P.ISI_JIT, Y_ISI - 0.17), 2 * P.ISI_JIT, 0.09,
                           facecolor=D.FLOW["rest"].edge, edgecolor="none"))
    ax.text(P.ISI_S + P.ISI_JIT + 0.14, Y_ISI - 0.13, f"± {P.ISI_JIT:.2f} s jitter",
            fontsize=D.FLOW_PT["small"], ha="left", va="center", color=D.FLOW_MUTED)


def main() -> None:
    fig, ax = plt.subplots(figsize=(S.COL2_IN, 2.30), layout="constrained")
    ax.set(xlim=(-3.05, T_END + 0.55), ylim=(-0.08, 3.15), yticks=[],
           xticks=np.arange(0, T_END + 0.1, 1.0))
    D.time_axis(ax, 0, T_END, label="time from first stimulus onset (s)", label_y=-0.14)
    radius, aspect = D.corner_radius(ax, points=3.0)

    # The window duration goes in the lane label, the response in the box.
    for text, y in (("stimulus", Y_STIM), (f"response window\n{P.RESP_S:.1f} s", Y_RESP + BAR_H / 2)):
        ax.text(-0.72, y, text, fontsize=D.FLOW_PT["body"], color=D.FLOW_MUTED, ha="right",
                va="center", linespacing=1.3)
    for k, trial in enumerate(TRIALS):
        draw_trial(ax, k * P.ISI_S, *trial, radius, aspect)

    # White background: the onset line runs through this label and hides digits in print.
    ax.text(0, Y_STIM + 0.62, f"{P.STIM_MS} ms", fontsize=D.FLOW_PT["body"], color=D.FLOW["blue"].edge,
            fontweight="semibold", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
    draw_isi(ax)
    D.save(fig, "trial_timeline")


if __name__ == "__main__":
    main()
