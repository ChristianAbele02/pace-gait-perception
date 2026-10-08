"""Shared drawing helpers for the methods schematics.

Defines the common style (``FLOW`` palette, line weights, font sizes) and shared
primitives: rounded chips, a blank drawing surface, a minimal time axis, text measurement,
the rendered Gabor and a save helper. Tables go in ``figure_captions.md``, not in figures.

Everything is drawn with matplotlib primitives, so figures are reproducible and vector in
the PDF. Importing the module adds ``Analysis/`` to the path and applies :mod:`plotstyle`,
so a figure script starts with::

    import diagram_kit as D
    from diagram_kit import S
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parent))
import plotstyle as S  # noqa: E402

S.apply()


# --- flowchart palette ----------------------------------------------------------------
class Tone(NamedTuple):
    fill: str
    edge: str


# Taken from the original ILL flow diagram: light fill, edge in the same hue, dark text.
# White text on saturated fills becomes unreadable when printed small.
FLOW = {"blue": Tone("#D9E8F5", "#2E6CA8"),
        "green": Tone("#DCEBD0", "#6BA644"),
        "walk": Tone("#C6E0CE", "#4F9C77"),
        "orange": Tone("#F6CB8E", "#E0902C"),
        "red": Tone("#F8DAD6", "#C4453C"),
        "rest": Tone("#E9ECEF", "#A6AFB8"),
        "panel": Tone("#EAF1F8", "#2E6CA8")}
FLOW_INK, FLOW_MUTED, FLOW_DARKBLUE = "#1F1F1F", "#566270", "#274E78"

# Line weights and font sizes for the 183 mm print width.
FLOW_LW = {"box": 1.4, "pill": 1.1, "arrow": 1.2, "icon": 1.15}
FLOW_PT = {"title": 10.5, "body": 8.5, "small": 8.0}

Z_BG, Z_LINK, Z_INK = 0, 1, 3


# --- shapes and axes --------------------------------------------------------------------
def rounded_box(x, y, w, h, *, radius, aspect=1.0, **kw) -> FancyBboxPatch:
    """Rectangle with lower-left corner ``(x, y)`` and rounded corners."""
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                          mutation_aspect=aspect, **kw)


def chip(x, y, w, h, tone, *, radius, aspect=1.0, lw=FLOW_LW["pill"], **kw) -> FancyBboxPatch:
    """Rounded box in one ``FLOW`` tone."""
    fill, edge = FLOW[tone]
    return rounded_box(x, y, w, h, radius=radius, aspect=aspect, facecolor=fill,
                       edgecolor=edge, linewidth=lw, **kw)


def corner_radius(ax, points=2.2) -> tuple[float, float]:
    """``(radius, aspect)`` that keep chip corners circular on non-equal-aspect axes.

    ``FancyBboxPatch`` rounds in data units, so on axes with different units (e.g. seconds
    vs lanes) corners become ellipses. The returned ``mutation_aspect`` corrects this.
    Call after the axis limits are set.
    """
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    x_per_pt = np.ptp(ax.get_xlim()) / bb.width
    y_per_pt = np.ptp(ax.get_ylim()) / bb.height
    return points * x_per_pt, y_per_pt / x_per_pt


def blank_axes(ax, width=100.0, height=68.0) -> None:
    """Blank equal-aspect drawing surface of ``width`` x ``height`` units.

    Choose ``height`` so ``width/height`` is slightly larger than the panel aspect. The
    drawing then fills the panel width; otherwise matplotlib shrinks it to fit the height
    and labels no longer fit their boxes.
    """
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    # Equal-aspect panels can differ in height; anchoring to the top aligns their titles.
    ax.set_anchor("N")
    ax.axis("off")
    ax.grid(False)


def time_axis(ax, start, end, *, label, label_y) -> None:
    """Reduce an axes to a muted bottom spine over ``[start, end]``.

    The x label is centred under that span, not under the whole (wider) axes.
    """
    ax.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set(color=FLOW_MUTED, bounds=(start, end))
    ax.tick_params(axis="x", labelsize=FLOW_PT["small"], labelcolor=FLOW_INK, color=FLOW_MUTED)
    ax.set_xlabel(label, fontsize=FLOW_PT["body"], color=FLOW_INK)
    lo, hi = ax.get_xlim()
    ax.xaxis.set_label_coords(((start + end) / 2 - lo) / (hi - lo), label_y)


def text_width(ax, text, **font) -> float:
    """Rendered width of ``text`` in data units."""
    t = ax.text(0, 0, text, **font)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    t.remove()
    (x0, _), (x1, _) = ax.transData.inverted().transform([(bb.x0, 0), (bb.x1, 0)])
    return x1 - x0


def label(ax, x, y, text, *, size=FLOW_PT["body"], weight="normal", color=S.INK, **kw):
    """Centred text at ink z-order. Keep sizes at 8 pt or above for print."""
    kw = {"ha": "center", "va": "center"} | kw
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, zorder=Z_INK, **kw)


def save(fig, name: str) -> None:
    """Save ``figures/<name>.png`` and ``.pdf`` at print size and close the figure."""
    S.save_pub(fig, FIG_DIR / name)
    print(f"wrote {name}.{{png,pdf}}")


# --- Gabor stimulus, from the shader -------------------------------------------------
# Values mirror FollowerPlane / GaborShader, via Analysis/gabor_preview.py.
GABOR_FREQ = 59.6              # cycles across the quad (= 3 cpd at 19.85 deg)
GABOR_SIGMA = 0.0353           # Gaussian SD in quad units (= 0.70 deg)
GABOR_DISK = 0.22              # FollowerPlane.gaborDiskRadius, overriding the shader's 0.151 (~8.7 deg patch)
GABOR_CONTRAST = 0.7


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gabor_rgba(tilt_deg=0.0, *, res=260, contrast=GABOR_CONTRAST, phase=0.0) -> np.ndarray:
    """Render the stimulus as an RGBA array using the shader equations.

    ``val = 0.5 + 0.5 * gaussian * grating * contrast``. The aperture edge is smoothed
    (smoothstep) so it gives no orientation cue, and is used as alpha.

    The image is cropped just outside the disk; the full 19.85 deg quad would make the
    8.7 deg patch tiny.
    """
    u = np.linspace(-1.12 * GABOR_DISK, 1.12 * GABOR_DISK, res)
    ux, uy = np.meshgrid(u, -u)
    rad = np.deg2rad(tilt_deg)
    x = ux * np.cos(rad) - uy * np.sin(rad)        # coordinate across the tilted grating
    r2 = ux * ux + uy * uy                         # radius is rotation-invariant
    val = 0.5 + 0.5 * contrast * np.exp(-r2 / (2 * GABOR_SIGMA ** 2)) \
        * np.cos(2 * np.pi * GABOR_FREQ * x + phase)
    alpha = 1.0 - _smoothstep(GABOR_DISK - 0.02, GABOR_DISK, np.sqrt(r2))
    val = np.clip(val, 0.0, 1.0)
    return np.dstack([val, val, val, alpha])
