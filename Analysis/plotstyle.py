"""Shared figure style for the analysis pipeline: fonts, colours and layout helpers.

Import it like the other shared modules::

    try:
        from . import plotstyle as S
    except ImportError:
        import plotstyle as S

    S.apply()                       # once, at import time
    ax.plot(..., color=S.MEASURE["accuracy"])
    fig.suptitle(S.nice_label(rec.label()))
    S.save(fig, out_png)

Colours are colour-blind aware and assigned by role, titles avoid raw BIDS stems, and
PNGs are saved at 300 dpi.
"""
from __future__ import annotations

import re
import textwrap as _textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette (colour-blind aware). Use via the role maps below.
# ---------------------------------------------------------------------------
INK = "#22272E"      # primary text / main traces
BLUE = "#3B6FB6"
RED = "#D1495B"
GREEN = "#2E8B57"
ORANGE = "#E08E3C"
PURPLE = "#7B5EA7"
TEAL = "#2A9D8F"
GOLD = "#E9C46A"
GREY = "#8A929B"     # reference lines / neutral marks
MIST = "#C8CDD4"     # raw signal / fills / de-emphasised
GRID = "#E6E8EB"

# Default colour cycle for series without a fixed role.
CYCLE = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GOLD, GREY]

# ---------------------------------------------------------------------------
# Journal column widths. Create figures at their final print size: font sizes are in
# points and only hold at 1:1 scale (an 18 in figure shrunk to 183 mm turns 8 pt into ~3 pt).
# ---------------------------------------------------------------------------
COL1_IN = 3.50      # single column: 89 mm (Nature/Science), 85 mm (Cell), 3.5 in (PLoS)
COL2_IN = 7.20      # double column: 183 mm (Nature/Science), 174 mm (Cell), 7 in (PLoS)
PAGE_H_IN = 9.40    # max height of a full-page figure (~239 mm)

# Minimum font size in any figure.
MIN_PT = 8.0

# --- Colour by role ------------------------------------------------------
MEASURE = {            # stride-phase measures
    "accuracy": BLUE,
    "reaction_time": RED,
    "timeouts": GREY,
    "response_counts": PURPLE,
    "response_likelihood": PURPLE,
}
COND = {"standing": ORANGE, "walking": TEAL}      # experiment condition
TRACKER = {"A": BLUE, "B": RED}                    # QUEST+ interleaved trackers
OUTCOME = {                                        # trial outcome
    "correct": GREEN, "hit": GREEN,
    "wrong": RED, "miss": RED,
    "timeout": GREY, "none": GREY,
    "excised": ORANGE, "flag": ORANGE,
}
FREQGRP = {"2cps": PURPLE, "4cps": TEAL, ">5": GREY, "other": GREY, "none": MIST}

# Display names for the stride-phase measures.
MEASURE_LABEL = {
    "accuracy": "accuracy",
    "reaction_time": "reaction time",
    "timeouts": "timeout rate",
    "response_counts": "response likelihood",
    "response_likelihood": "response likelihood",
}


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
def apply() -> None:
    """Set the rcParams. Safe to call more than once."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6.0,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,

        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,

        "lines.linewidth": 1.6,
        "lines.markersize": 5,

        "axes.prop_cycle": plt.cycler(color=CYCLE),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nice_label(rec_label: str) -> str:
    """Human-readable subject label from a BIDS stem.

    ``sub-01_ses-02_task-ILL_acq-walking``        -> ``sub-01 · walking``;
    ``sub-01_ses-02_task-ILL_acq-walking_run-05`` -> ``sub-01 · walking · run-05``;
    ``sub-01_ses-01_task-ILL_beh``                -> ``sub-01 · ses-01``.
    """
    if not rec_label:
        return ""
    sub = re.search(r"sub-\d+", rec_label)
    acq = re.search(r"acq-(walking|standing)", rec_label)
    ses = re.search(r"ses-\d+", rec_label)
    run = re.search(r"run-\d+", rec_label)
    head = sub.group(0) if sub else rec_label
    if acq:
        return f"{head} · {acq.group(1)}" + (f" · {run.group(0)}" if run else "")
    if ses:
        return f"{head} · {ses.group(0)}"
    return head


def panel(ax, tag: str, title: str = "", **kw) -> None:
    """Left-aligned panel title of the form ``a  Title``."""
    txt = f"{tag}  {title}".rstrip()
    ax.set_title(txt, loc="left", **kw)


def polar_panel(ax, tag: str, title: str = "", *, pad: float = 22.0, **kw) -> None:
    """``panel()`` for polar axes.

    With the global left title location, a polar title is placed far left and can overlap
    the suptitle (``tight_layout`` does not fix polar axes). Centring and padding avoids this.
    """
    ax.set_title(f"{tag}  {title}".rstrip(), loc="center", pad=pad, **kw)


def caption(fig, text: str, *, y: float = -0.01, color: str = GREY,
            size: float = 8.0, wrap: int | None = None, x: float = 0.5,
            ha: str = "center") -> None:
    """Small grey note below a figure (method notes, parameters).

    ``wrap`` sets the line length in characters; use it for anything longer than a short
    phrase, otherwise the text is one line across the whole canvas. Keep notes to 1-3
    lines; full captions belong in ``figure_captions.md``.
    """
    if wrap:
        text = _textwrap.fill(text, wrap)
    fig.text(x, y, text, ha=ha, va="top", fontsize=size, color=color,
             linespacing=1.45)


def legend_outside(ax, **kw):
    """Legend to the right of the axes."""
    return ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), **kw)


def reference_line(ax, *, y=None, x=None, label=None, ls="--"):
    """Grey reference line (chance level, targets, means)."""
    if y is not None:
        return ax.axhline(y, color=GREY, ls=ls, lw=1.0, label=label, zorder=0)
    return ax.axvline(x, color=GREY, ls=ls, lw=1.0, label=label, zorder=0)


def style_image_axis(ax) -> None:
    """Remove ticks, spines and grid from an imshow panel."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, path, *, constrained: bool = False) -> None:
    """Save (300 dpi, tight bbox from rcParams) and close the figure.

    ``constrained`` has no effect; layout is left to the caller.
    """
    fig.savefig(path)
    plt.close(fig)


def save_pub(fig, stem, *, formats=("png", "pdf"), audit: bool = True):
    """Save a manuscript figure as PNG and PDF, then close it.

    ``stem`` is a path without suffix: ``fig4_adopted_design`` gives
    ``fig4_adopted_design.png`` (300 dpi) and ``fig4_adopted_design.pdf`` (vector).

    With ``audit``, text smaller than :data:`MIN_PT` at print size is reported; undersized
    text is easy to miss on screen.
    """
    from pathlib import Path

    stem = Path(stem)
    if audit:
        for bad in _undersized_text(fig):
            print(f"  ! {stem.name}: {bad}")
    out = [stem.with_suffix(f".{ext}") for ext in formats]
    # Use "standard" instead of the global "tight" so the saved width stays exactly COL2_IN.
    # bbox_inches=None would fall back to the rcParam ("tight").
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        for path in out:
            # No PDF creation date, so unchanged figures are byte-identical.
            fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    plt.close(fig)
    return out


def _undersized_text(fig):
    """Report text artists set below :data:`MIN_PT`, as ``(size, excerpt)`` strings."""
    bad = []
    for t in fig.findobj(plt.Text):
        s = (t.get_text() or "").strip()
        if not s:
            continue
        pt = t.get_fontsize()
        if pt < MIN_PT - 1e-6:
            bad.append(f"{pt:.1f} pt (< {MIN_PT:.0f} pt): {s[:48]!r}")
    return bad
