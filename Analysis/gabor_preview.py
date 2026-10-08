"""Render the CAVE Gabor patch as the shader draws it, for reference.

Reimplements Assets/Materials/GaborShader.shader (UVs centred at 0, range [-0.5, 0.5]
across the quad):

    x =  uv.x*cos - uv.y*sin ;  y = uv.x*sin + uv.y*cos        (rotation)
    gaussian = exp(-(x^2+y^2)/(2*sigma^2))                     (Gaussian envelope)
    grating  = cos(2*pi*frequency*x + phase)                   (carrier)
    val      = 0.5 + 0.5 * gaussian*grating*contrast           (luminance)
    alpha    = 1 - smoothstep(DiskRadius-0.02, DiskRadius, |uv|)(small circular patch)

Quad geometry (README "Stimulus revision"): a 0.7 m quad at 2 m spans ~19.85 deg, so
cpd = frequency / 19.85, sigma_deg = sigma * 19.85, patch_diameter_deg = (DiskRadius / 0.5) * 19.85.

Current FollowerPlane values: frequency 59.6 (3 cpd), sigma 0.0353 (0.70 deg Gaussian SD),
DiskRadius 0.22 (~8.7 deg patch; FollowerPlane.gaborDiskRadius overrides the shader default
0.151), contrast 0.7. This is a small Gabor comparable to Davidson's, replacing the old
19.85 deg disk.

    python gabor_preview.py   # writes gabor_current.png, gabor_parameters.png,
                              # gabor_difficulty_proposals.png
"""
from __future__ import annotations

import os.path as _osp
import sys
sys.path.insert(0, _osp.dirname(_osp.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
import plotstyle as S

S.apply()

# Output folder: <project root>/gabor/
OUT_DIR = _osp.join(_osp.dirname(_osp.dirname(_osp.abspath(__file__))), "gabor")
_os.makedirs(OUT_DIR, exist_ok=True)

QUAD_DEG = 19.85                      # visual angle of the full quad (1.0 in UV)
BG = (0.18, 0.26, 0.20)              # typical CAVE background colour (muted green)

# current shader / FollowerPlane parameters
CUR_FREQ = 59.6                      # cycles across the quad   (= 3 cpd)
CUR_SIGMA = 0.0353                   # Gaussian SD in quad units (= 0.70 deg)
CUR_DISK = 0.22                      # visible disk radius (UV) (~8.7 deg patch)
CUR_CONTRAST = 0.7

UV_PER_DEG = 0.5 / (QUAD_DEG / 2)    # = 1/19.85


def smoothstep(a, b, x):
    """Hermite smoothstep from 0 at ``a`` to 1 at ``b``; softens the aperture edge."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def render(freq, sigma, contrast, rotation_deg=0.0, phase=0.0,
           disk_radius=CUR_DISK, res=900, noise_sd=0.0, noise_seed=0):
    """Return (val, alpha) images as computed by the shader.

    ``noise_sd`` > 0 adds seeded Gaussian luminance noise (0..1 units), previewing a
    proposed shader noise term. In the shader the seed would have to be broadcast per trial
    so all six half-walls show the same noise; here a fixed seed stands in for that."""
    u = np.linspace(-0.5, 0.5, res)
    ux, uy = np.meshgrid(u, u)
    rad = np.deg2rad(rotation_deg)
    x = ux * np.cos(rad) - uy * np.sin(rad)
    y = ux * np.sin(rad) + uy * np.cos(rad)
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    grating = np.cos(2 * np.pi * freq * x + phase)
    val = 0.5 + 0.5 * gaussian * grating * contrast
    if noise_sd > 0:
        rng = np.random.default_rng(noise_seed)
        val = np.clip(val + rng.normal(0.0, noise_sd, val.shape), 0.0, 1.0)
    dist = np.sqrt(ux * ux + uy * uy)
    alpha = 1.0 - smoothstep(disk_radius - 0.02, disk_radius, dist)
    return val, alpha


def composite(val, alpha):
    """Alpha-composite a rendered Gabor onto the CAVE background colour.

    Perceived contrast depends on the background, so the preview uses the real one.
    """
    img = np.repeat(val[..., None], 3, axis=2)
    bg = np.ones_like(img) * np.array(BG)
    a = alpha[..., None]
    return np.clip(img * a + bg * (1 - a), 0, 1)


def _show(ax, val, alpha, title, zoom_deg=None):
    ax.imshow(composite(val, alpha), extent=[-QUAD_DEG / 2, QUAD_DEG / 2,
                                             -QUAD_DEG / 2, QUAD_DEG / 2])
    if zoom_deg:
        ax.set_xlim(-zoom_deg, zoom_deg); ax.set_ylim(-zoom_deg, zoom_deg)
    ax.set_title(title, fontsize=9.5, loc="center")
    S.style_image_axis(ax)


def _disk_deg(disk_radius):
    return (disk_radius / 0.5) * QUAD_DEG


# ---------------------------------------------------------------------------
# 1) current patch
# ---------------------------------------------------------------------------
def fig_current():
    """Render the current Gabor at true visual angle."""
    val, alpha = render(CUR_FREQ, CUR_SIGMA, CUR_CONTRAST, rotation_deg=0.0)
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.imshow(composite(val, alpha), extent=[-QUAD_DEG / 2, QUAD_DEG / 2,
                                             -QUAD_DEG / 2, QUAD_DEG / 2])
    z = _disk_deg(CUR_DISK) / 2 + 1.5
    ax.set_xlim(-z, z); ax.set_ylim(-z, z)
    ax.grid(False)
    ax.set_xlabel("degrees of visual angle"); ax.set_ylabel("degrees of visual angle")
    ax.set_title("ILL CAVE Gabor — as rendered", fontsize=12, loc="left")
    S.caption(fig, f"3.0 cpd · Gaussian σ = 0.70° · contrast 0.7 · "
                   f"patch ≈ {_disk_deg(CUR_DISK):.1f}° disk", y=0.02)
    fig.tight_layout(); fig.savefig(_osp.join(OUT_DIR, "gabor_current.png")); plt.close(fig)
    print("wrote", _osp.join(OUT_DIR, "gabor_current.png"))


# proposed harder stimulus (decision #1)
PROP_CONTRAST = 0.4
PROP_NOISE_SD = 0.12     # static luminance noise, one frozen pattern per trial


def fig_proposed():
    """Render the proposed harder patch in the style of fig_current().

    Same 3 cpd / σ 0.70° / ~8.7° patch, with contrast 0.4 and static noise (SD 0.12).
    The fixed seed gives one frozen pattern (in the shader it would be broadcast per trial)."""
    val, alpha = render(CUR_FREQ, CUR_SIGMA, PROP_CONTRAST, rotation_deg=0.0,
                        noise_sd=PROP_NOISE_SD, noise_seed=12345)
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.imshow(composite(val, alpha), extent=[-QUAD_DEG / 2, QUAD_DEG / 2,
                                             -QUAD_DEG / 2, QUAD_DEG / 2])
    z = _disk_deg(CUR_DISK) / 2 + 1.5
    ax.set_xlim(-z, z); ax.set_ylim(-z, z)
    ax.grid(False)
    ax.set_xlabel("degrees of visual angle"); ax.set_ylabel("degrees of visual angle")
    ax.set_title("ILL CAVE Gabor — proposed (lower contrast + static noise)",
                 fontsize=12, loc="left")
    S.caption(fig, f"3.0 cpd · Gaussian σ = 0.70° · contrast {PROP_CONTRAST} · static "
                   f"noise SD {PROP_NOISE_SD} · patch ≈ {_disk_deg(CUR_DISK):.1f}° disk", y=0.02)
    fig.tight_layout(); fig.savefig(_osp.join(OUT_DIR, "gabor_proposed.png")); plt.close(fig)
    print("wrote", _osp.join(OUT_DIR, "gabor_proposed.png"))


# ---------------------------------------------------------------------------
# 2) effect of each parameter
# ---------------------------------------------------------------------------
def fig_parameters():
    """Vary each Gabor parameter separately to show its effect (design aid, not pipeline)."""
    fig, ax = plt.subplots(5, 3, figsize=(12, 19))
    fig.suptitle("Gabor parameters — what each one changes", fontsize=13,
                 fontweight="semibold", x=0.012, ha="left")
    Z = 4.0   # zoom (deg)

    # Row 0: spatial frequency (cpd)
    for j, cpd in enumerate([1.5, 3.0, 6.0]):
        v, a = render(cpd * QUAD_DEG, CUR_SIGMA, 0.9, disk_radius=0.25)
        _show(ax[0, j], v, a, f"frequency = {cpd:.1f} cpd", zoom_deg=Z)
    ax[0, 0].set_ylabel("FREQUENCY\n(stripe spacing)", fontsize=10)

    # Row 1: sigma, Gaussian SD in degrees
    for j, sdeg in enumerate([0.35, 0.70, 1.4]):
        v, a = render(CUR_FREQ, sdeg / QUAD_DEG, 0.9, disk_radius=0.25)
        _show(ax[1, j], v, a, f"σ = {sdeg:.2f}°" + (" (current)" if sdeg == 0.70 else ""),
              zoom_deg=Z)
    ax[1, 0].set_ylabel("SIGMA\n(Gabor size:\nGaussian SD)", fontsize=10)

    # Row 2: disk radius (visible patch size)
    for j, dr in enumerate([0.08, CUR_DISK, 0.40]):
        v, a = render(CUR_FREQ, CUR_SIGMA, 0.9, disk_radius=dr)
        tag = f"patch ≈ {_disk_deg(dr):.1f}°" + (" (current)" if dr == CUR_DISK else "")
        _show(ax[2, j], v, a, tag, zoom_deg=_disk_deg(0.40) / 2 + 1)
    ax[2, 0].set_ylabel("DISK RADIUS\n(visible patch size;\nframes the Gabor)", fontsize=10)

    # Row 3: contrast (stripe amplitude)
    for j, c in enumerate([0.25, 0.6, 1.0]):
        v, a = render(CUR_FREQ, CUR_SIGMA, c, disk_radius=0.25)
        _show(ax[3, j], v, a, f"contrast = {c:.2f}", zoom_deg=Z)
    ax[3, 0].set_ylabel("CONTRAST\n(stripe amplitude)", fontsize=10)

    # Row 4: rotation (the task: left vs right tilt)
    for j, rot in enumerate([-10, 0, 10]):
        v, a = render(CUR_FREQ, CUR_SIGMA, 0.9, rotation_deg=rot, disk_radius=0.25)
        _show(ax[4, j], v, a, f"tilt = {rot:+d}°", zoom_deg=Z)
    ax[4, 0].set_ylabel("ROTATION\n(the task:\nleft vs right tilt)", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(_osp.join(OUT_DIR, "gabor_parameters.png"), dpi=120); plt.close(fig)
    print("wrote", _osp.join(OUT_DIR, "gabor_parameters.png"))


# ---------------------------------------------------------------------------
# 3) difficulty proposals: contrast x noise (decision #1)
# ---------------------------------------------------------------------------
def fig_difficulty_proposals(tilt_deg=8.0):
    """Grid of contrast (columns) x noise (rows) at a fixed visible tilt.

    For choosing a harder stimulus that moves the threshold off the grid floor (~0.4 deg)
    to ~1-3 deg. Top-left is the current stimulus (contrast 0.7, no noise). Frequency,
    sigma and patch size are the current values."""
    contrasts = [0.7, 0.4, 0.25, 0.15]
    noises = [0.0, 0.10, 0.20]
    Z = 4.0
    fig, ax = plt.subplots(len(noises), len(contrasts),
                           figsize=(3.0 * len(contrasts), 3.0 * len(noises)))
    fig.suptitle(f"Harder-Gabor proposals — contrast x noise  (shown at a {tilt_deg:.0f}° tilt; "
                 "current = top-left)", fontsize=12, fontweight="semibold", x=0.012, ha="left")
    for r, nsd in enumerate(noises):
        for c, con in enumerate(contrasts):
            v, a = render(CUR_FREQ, CUR_SIGMA, con, rotation_deg=tilt_deg,
                          disk_radius=0.25, noise_sd=nsd, noise_seed=7)
            cur = " (CURRENT)" if (con == 0.7 and nsd == 0.0) else ""
            _show(ax[r, c], v, a, f"contrast {con:.2f} · noise {nsd:.2f}{cur}", zoom_deg=Z)
        ax[r, 0].set_ylabel(f"noise SD = {nsd:.2f}", fontsize=10)
    S.caption(fig, "Lower contrast and/or added noise reduce the signal-to-noise ratio, "
                   "raising the orientation threshold AND making it less dependent on "
                   "sub-pixel tilt rendering. Pick a starting point, then pilot to land the "
                   "threshold at ~1-3°. Noise must be seeded identically on all CAVE walls.",
              y=0.01)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = _osp.join(OUT_DIR, "gabor_difficulty_proposals.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_current()
    fig_parameters()
    fig_difficulty_proposals()
