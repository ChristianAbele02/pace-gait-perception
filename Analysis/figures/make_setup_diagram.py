"""Methods figure: the rig's computers and how they communicate.

The tracking PC streams head pose to the head node. The head node runs the experiment, drives
the treadmill and synchronises three render nodes, each drawing one CAVE wall (two stacked
panels). Machines are labelled by role; hostnames are installation-specific.

Run::  python Analysis/figures/make_setup_diagram.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import diagram_kit as D
from diagram_kit import FLOW, FLOW_DARKBLUE, FLOW_LW, FLOW_MUTED, FLOW_PT, S
from icons_svg import draw_icon

W, H = 100.0, 64.0
COLS = (17.0, 50.0, 83.0)              # left, middle, right column centres
BOX_W, BOX_H = 24.0, 11.0
ROWS = {"inputs": 52.0, "render": 31.0, "walls": 10.0}


def card(ax, cx, cy, tone, icon, title, sub):
    """Draw one machine: rounded box with icon, name and role."""
    r, a = D.corner_radius(ax)
    ax.add_patch(D.chip(cx - BOX_W / 2, cy - BOX_H / 2, BOX_W, BOX_H, tone, radius=r, aspect=a,
                        lw=FLOW_LW["box"], zorder=D.Z_LINK + 1))
    draw_icon(ax, icon, cx, cy + 2.6, size=4.2, lw=FLOW_LW["icon"], zorder=D.Z_INK)
    D.label(ax, cx, cy - 1.3, title, size=FLOW_PT["body"], weight="bold")
    D.label(ax, cx, cy - 3.6, sub, size=FLOW_PT["small"], color=FLOW_MUTED)


def arrow(ax, x0, y0, x1, y1, tone):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=D.Z_LINK,
                arrowprops=dict(arrowstyle="-|>", color=FLOW[tone].edge, lw=FLOW_LW["arrow"],
                                shrinkA=0, shrinkB=0, mutation_scale=9))


def main() -> None:
    fig, ax = plt.subplots(figsize=(S.COL2_IN, S.COL2_IN * H / W))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    D.blank_axes(ax, W, H)

    top, mid, low = ROWS["inputs"], ROWS["render"], ROWS["walls"]
    card(ax, COLS[0], top, "green", "device-cctv", "Tracking PC", "OptiTrack Motive, VRPN")
    card(ax, COLS[1], top, "blue", "server", "Head node", "Linux, runs the experiment")
    card(ax, COLS[2], top, "green", "treadmill", "Treadmill", "Modbus server")
    for i, cx in enumerate(COLS, start=1):
        card(ax, cx, mid, "blue", "server", f"Render node {i}", "Windows")
    for cx, name in zip(COLS, ("Left wall", "Front wall", "Right wall")):
        card(ax, cx, low, "orange", "device-desktop", name, "2 panels, 1920 x 2160")

    half = BOX_W / 2
    arrow(ax, COLS[0] + half, top, COLS[1] - half, top, "green")      # head pose
    arrow(ax, COLS[1] + half, top, COLS[2] - half, top, "green")      # belt speed
    bus = (top + mid) / 2
    edge = FLOW["blue"].edge
    ax.plot([COLS[1], COLS[1]], [top - BOX_H / 2, bus], color=edge, lw=FLOW_LW["arrow"], zorder=D.Z_LINK)
    ax.plot([COLS[0], COLS[2]], [bus, bus], color=edge, lw=FLOW_LW["arrow"], zorder=D.Z_LINK)
    for cx in COLS:
        arrow(ax, cx, bus, cx, mid + BOX_H / 2, "blue")               # sync + start/stop
        arrow(ax, cx, mid - BOX_H / 2, cx, low + BOX_H / 2, "orange")  # video
    D.label(ax, (COLS[0] + COLS[1]) / 2, top + 2.8, "head\npose", size=FLOW_PT["small"], color=FLOW_MUTED)
    D.label(ax, (COLS[1] + COLS[2]) / 2, top + 2.8, "belt\nspeed", size=FLOW_PT["small"], color=FLOW_MUTED)
    D.label(ax, COLS[1] + 9.5, bus + 1.5, "scene sync, start/stop", size=FLOW_PT["small"], color=FLOW_MUTED)
    D.label(ax, W / 2, 1.2, "Unity 2017.4  ·  UniCave off-axis projection  ·  OptiTrack over VRPN",
            size=FLOW_PT["small"], color=FLOW_DARKBLUE)
    D.save(fig, "setup_diagram")


if __name__ == "__main__":
    main()
