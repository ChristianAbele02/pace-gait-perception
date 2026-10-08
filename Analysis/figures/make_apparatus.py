"""Methods figure: the CAVE apparatus.

Three-quarter view into the CAVE: three walls of two stacked display panels (numbered
1-6), the treadmill and participant in front of the front wall, and the Gabor target on
the front wall at eye height.

The Gabor is rendered from the shader equations (``diagram_kit.gabor_rgba``) with its real
spatial frequency and contrast. Dimensions and machine topology are listed in
``figure_captions.md`` and the READMEs.

Run::  python Analysis/figures/make_apparatus.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

import diagram_kit as D
from diagram_kit import S

# --- room geometry, in diagram units ----------------------------------------------------
# The front wall is the rectangle at the back; the side walls widen toward the viewer to
# give perspective.
FX0, FX1, FY0, FY1 = 33, 67, 15, 45        # front wall
NX0, NX1, NY0, NY1 = 5, 95, 5, 53          # near edges of the side walls
MID_F = (FY0 + FY1) / 2                    # panel seam, front wall
MID_N = (NY0 + NY1) / 2                    # panel seam, near edges
CENTRE = (FX0 + FX1) / 2
TREADMILL_AT = (CENTRE, 10.0)              # camera aiming point

# Light grey-blue walls read as screens and contrast with the Gabor and the participant.
# Side walls are darker for depth.
WALLS = [([(NX0, NY0), (FX0, FY0), (FX0, FY1), (NX0, NY1)], "#B6C2CD"),      # left
         ([(NX1, NY0), (FX1, FY0), (FX1, FY1), (NX1, NY1)], "#C3CDD7"),      # right
         ([(FX0, FY0), (FX1, FY0), (FX1, FY1), (FX0, FY1)], "#CFD8E0")]      # front
FLOOR = [(NX0, NY0), (FX0, FY0), (FX1, FY0), (NX1, NY0)]
SEAMS = [((FX0, MID_F), (FX1, MID_F)), ((NX0, MID_N), (FX0, MID_F)), ((NX1, MID_N), (FX1, MID_F))]
SEAM, WALL_EDGE = "#93A2B0", "#A6B2BD"
FLOOR_FILL, FLOOR_EDGE = "#F3F5F6", "#DDE1E4"
BODY = "#2B3138"

# The six half-screens as (bottom-left, bottom-right, top-right, top-left) corners in figure
# space (left = smaller x). Quads let each number sit at the same relative position.
HALF_SCREENS = {
    1: [(NX0, MID_N), (FX0, MID_F), (FX0, FY1), (NX0, NY1)],          # left, top
    2: [(NX0, NY0), (FX0, FY0), (FX0, MID_F), (NX0, MID_N)],          # left, bottom
    3: [(FX0, MID_F), (FX1, MID_F), (FX1, FY1), (FX0, FY1)],          # front, top
    4: [(FX0, FY0), (FX1, FY0), (FX1, MID_F), (FX0, MID_F)],          # front, bottom
    5: [(FX1, MID_F), (NX1, MID_N), (NX1, NY1), (FX1, FY1)],          # right, top
    6: [(FX1, FY0), (NX1, NY0), (NX1, MID_N), (FX1, MID_F)],          # right, bottom
}

# Two outer corners and the two wall junctions, all aimed at the treadmill.
CAMERAS = [(NX0 + 4, NY1 - 3), (NX1 - 4, NY1 - 3), (FX0, FY1), (FX1, FY1)]


def quad_point(corners, u, v) -> tuple[float, float]:
    """Bilinear point at fraction ``u`` along and ``v`` up a quad.

    Follows the sloped edges of the side walls, unlike a fixed x/y offset.
    """
    bl, br, tr, tl = np.asarray(corners, float)
    bottom, top = bl + u * (br - bl), tl + u * (tr - tl)
    return tuple(bottom + v * (top - bottom))


def draw_room(ax) -> None:
    """Draw floor, walls and the seam between each wall's two panels."""
    ax.add_patch(Polygon(FLOOR, facecolor=FLOOR_FILL, edgecolor=FLOOR_EDGE, lw=0.8, zorder=D.Z_BG))
    for corners, fill in WALLS:
        ax.add_patch(Polygon(corners, facecolor=fill, edgecolor=WALL_EDGE, lw=0.9, zorder=D.Z_BG + 1))
    for (x0, y0), (x1, y1) in SEAMS:
        ax.plot([x0, x1], [y0, y1], color=SEAM, lw=1.0, zorder=D.Z_BG + 2)


def screen_number(ax, x, y, n) -> None:
    """Numbered marker for one half-screen."""
    ax.add_patch(Circle((x, y), 2.3, facecolor="white", edgecolor=WALL_EDGE, lw=0.8, zorder=D.Z_INK))
    ax.text(x, y, str(n), fontsize=8, fontweight="bold", color=S.INK, ha="center", va="center",
            zorder=D.Z_INK + 1)


def draw_treadmill(ax, cx) -> None:
    """Treadmill belt in perspective, with side rails."""
    ax.add_patch(Polygon([(cx - 13, 7.0), (cx + 13, 7.0), (cx + 10, 12.5), (cx - 10, 12.5)],
                         facecolor="#4A545E", edgecolor="#39424B", lw=0.9, zorder=D.Z_LINK))
    ax.add_patch(Polygon([(cx - 9.5, 8.2), (cx + 9.5, 8.2), (cx + 7.4, 11.6), (cx - 7.4, 11.6)],
                         facecolor="#6B7681", edgecolor="none", zorder=D.Z_LINK + 1))
    for side in (-1, 1):
        ax.plot([cx + side * 12.4, cx + side * 10.6], [7.4, 12.2], color="#8A939C", lw=1.2,
                solid_capstyle="round", zorder=D.Z_LINK + 1)


def draw_walker(ax, cx, feet=11.0) -> None:
    """Participant seen from behind, mid-stride, as a filled silhouette.

    An outline would have too little contrast against the light walls.
    """
    head = feet + 21.0
    limb = dict(color=BODY, lw=2.0, solid_capstyle="round", zorder=D.Z_INK)
    ax.add_patch(Circle((cx, head), 2.4, facecolor=BODY, edgecolor="none", zorder=D.Z_INK))
    ax.add_patch(Polygon([(cx - 3.5, head - 3.3), (cx + 3.5, head - 3.3),
                          (cx + 2.5, head - 11.2), (cx - 2.5, head - 11.2)],
                         facecolor=BODY, edgecolor="none", zorder=D.Z_INK))
    for side, swing in ((-1, -0.5), (1, 0.9)):         # arms, counter-swinging
        ax.plot([cx + side * 3.6, cx + side * 4.5 + swing], [head - 3.6, head - 11.3], **limb)
    for side, swing in ((-1, -1.4), (1, 1.8)):         # legs, mid-stride
        ax.plot([cx + side * 1.5, cx + side * 1.6 + swing], [head - 11.2, feet], **limb)


def draw_camera(ax, cx, cy, target=TREADMILL_AT) -> None:
    """Tracking camera as a wedge aimed at ``target``."""
    a = np.arctan2(target[1] - cy, target[0] - cx)
    tips = [(2.6, a + 2.5), (2.6, a - 2.5), (3.6, a)]           # (radius, angle)
    ax.add_patch(Polygon([(cx + r * np.cos(t), cy + r * np.sin(t)) for r, t in tips],
                         facecolor=S.RED, edgecolor="none", zorder=D.Z_INK))


def leader(ax, start, end, color=S.GREY) -> None:
    """Thin line from a label to its target."""
    (x0, y0), (x1, y1) = start, end
    ax.plot([x0, x1], [y0, y1], color=color, lw=0.7, zorder=D.Z_LINK)


def main() -> None:
    fig, ax = plt.subplots(figsize=(S.COL2_IN, 4.45), layout="constrained")
    D.blank_axes(ax, 100, 58)

    draw_room(ax)
    draw_treadmill(ax, CENTRE)
    draw_walker(ax, CENTRE)

    # Gabor at eye height on the front wall. The head overlaps only the grey surround below
    # the grating. The tilt is exaggerated; real threshold tilts would be invisible here.
    gy, gs = 36.3, 7.6
    ax.imshow(D.gabor_rgba(18.0), extent=(CENTRE - gs / 2, CENTRE + gs / 2, gy - gs / 2, gy + gs / 2),
              interpolation="bilinear", zorder=D.Z_LINK, aspect="auto")

    for cam in CAMERAS:
        draw_camera(ax, *cam)
    for n, quad in HALF_SCREENS.items():            # number in the bottom-left corner
        screen_number(ax, *quad_point(quad, 0.10, 0.16), n)

    # Labels above the room, connected to their targets.
    y_lab = 56.5
    D.label(ax, 12, y_lab, "tracking cameras", weight="semibold", color=S.RED)
    leader(ax, (12, y_lab - 1.9), (NX0 + 4, NY1 - 1.2), color=S.RED)
    for x, y_wall, text in [(27, 46.7, "left wall"), (CENTRE, FY1, "front wall"), (76, 47.6, "right wall")]:
        D.label(ax, x, y_lab, text, weight="semibold")
        leader(ax, (x, y_lab - 1.9), (x, y_wall + 0.8))
    D.label(ax, CENTRE + 16, 10.0, "treadmill", color=S.GREY, ha="left")
    leader(ax, (CENTRE + 11, 10.2), (CENTRE + 15, 10.0))

    ax.set_title("CAVE apparatus", loc="left", fontsize=11, fontweight="semibold", pad=8)
    D.save(fig, "apparatus")


if __name__ == "__main__":
    main()
