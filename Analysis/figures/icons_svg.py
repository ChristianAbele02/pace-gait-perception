"""Tabler icons (MIT licence) as matplotlib vector paths.

The icons are stroke-only line art on a 24 x 24 viewBox (no fill, round caps), matching the
line style of the schematics. The SVGs are cached in ``icons/`` so figures build offline::

    from icons_svg import draw_icon
    draw_icon(ax, "treadmill", cx, cy, size=30)     # size = icon width in data units
"""
from __future__ import annotations

import re
from functools import cache
from pathlib import Path

from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

from diagram_kit import FLOW_INK

ICON_DIR = Path(__file__).resolve().parent / "icons"
_VIEWBOX = re.compile(r'viewBox="([\d.\s-]+)"')
_PATH_DATA = re.compile(r'\bd="([^"]+)"')


@cache
def _load(name: str) -> tuple[MplPath, float, float]:
    """Return the icon as a single compound path plus its viewBox width and height."""
    svg = (ICON_DIR / f"tabler-{name}.svg").read_text(encoding="utf-8")
    *_, width, height = map(float, _VIEWBOX.search(svg)[1].split())
    return MplPath.make_compound_path(*map(parse_path, _PATH_DATA.findall(svg))), width, height


def draw_icon(ax, name, cx, cy, size=28.0, lw=2.0, color=FLOW_INK, zorder=5) -> PathPatch:
    """Draw a cached icon centred on ``(cx, cy)``, scaled to ``size`` data units wide."""
    path, width, height = _load(name)
    scale = size / width
    # SVG y points down: centre on the origin, flip y, then translate.
    to_data = Affine2D().translate(-width / 2, -height / 2).scale(scale, -scale).translate(cx, cy)
    return ax.add_patch(PathPatch(path.transformed(to_data), facecolor="none", edgecolor=color,
                                  lw=lw, joinstyle="round", capstyle="round", zorder=zorder))
