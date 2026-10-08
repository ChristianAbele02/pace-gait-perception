"""Phaser-style stride phase combined from several head signals.

The trough method samples one signal (vertical head bob) a few times per stride and
interpolates linearly; the Hilbert method uses one signal continuously. Both carry that
signal's noise. Revzen & Guckenheimer (2008) showed that combining several signals of the
same oscillator, weighted by how precisely each tracks the common phase, lowers the
variance. Steps:

1. Protophase per channel: analytic-signal (Hilbert) phase of each band-passed channel.
2. Protophase to phase (Kralemann et al. 2008). For non-sinusoidal waveforms the
   protophase advances unevenly within a cycle, differently per channel. The Fourier
   correction maps it to a uniformly advancing phase so channels are comparable.
3. Step-rate channels to stride rate. The vertical bob oscillates twice per stride, so its
   phase is halved. The 0/pi ambiguity is resolved per continuous segment against the
   stride-rate channels.
4. Weighted circular mean. Channels are aligned and averaged, weighted by their
   concentration around the common phase (von Mises precision), then corrected again.
5. Origin. 0 is set to the double-support trough starting a foot-A step, as in
   ``gait.phase_at``, so all three methods share one axis. The offset is set per unbroken
   segment because the unwrapped phase can resume with any offset after a gap.

Simplified relative to Phaser: the published method combines channels by PCA and a Fourier
series fit; here it is a precision-weighted circular mean. The protophase correction and
precision weighting are the same.

Status: on clean synthetic channels this is 2-3x more accurate than any single channel
(``tests/test_pipeline.py``). On real head data it explains no more held-out phase-locked
variance than the trough phase and slips cycles (``gait_phase_methods.py``), because
lateral sway and head roll oscillate less cleanly than the vertical bob. Available via
``config.PHASE_METHOD = "phaser"``, but not the default.

References
----------
Fisher NI (1993). Statistical Analysis of Circular Data. Cambridge University Press.
    (von Mises concentration from a resultant length, used for the channel weights.)
Revzen S, Guckenheimer JM (2008). Estimating the phase of synchronized oscillators.
    Phys Rev E 78:051907. doi:10.1103/PhysRevE.78.051907
Kralemann B, Cimponeriu L, Rosenblum M, Pikovsky A, Mrowka R (2008). Phase dynamics of
    coupled oscillators reconstructed from data. Phys Rev E 77:066205.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

N_HARMONICS = 10          # Fourier terms in the protophase -> phase correction


def protophase(x: np.ndarray) -> np.ndarray:
    """Unwrapped analytic-signal protophase (rad) of a zero-mean oscillatory signal."""
    x = np.asarray(x, float)
    return np.unwrap(np.angle(hilbert(x - np.mean(x))))


def to_phase(theta: np.ndarray, valid: np.ndarray | None = None,
             n_harm: int = N_HARMONICS) -> np.ndarray:
    """Kralemann et al. (2008) protophase-to-phase transform of an unwrapped protophase.

    phi = theta + sum_{n != 0} S_n / (i n) (exp(i n theta) - 1),
    S_n = < exp(-i n theta) >, estimated on the ``valid`` samples only.
    """
    theta = np.asarray(theta, float)
    ok = np.ones(theta.shape, bool) if valid is None else np.asarray(valid, bool)
    if ok.sum() < 10:
        return theta.copy()
    phi = theta.copy()
    e_ok = np.exp(1j * theta[ok])
    for n in range(1, n_harm + 1):
        s_n = np.mean(np.conj(e_ok) ** n)                # S_n; S_-n is its conjugate
        term = s_n / (1j * n) * (np.exp(1j * n * theta) - 1.0)
        phi += 2.0 * term.real                            # n and -n terms combined
    return phi


def _circ_offset(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Circular mean of (a - b) and its resultant length R (0..1)."""
    z = np.exp(1j * (a - b))
    z = z[np.isfinite(z)]
    if not z.size:
        return 0.0, 0.0
    m = z.mean()
    return float(np.angle(m)), float(np.abs(m))


def _kappa_weight(r: float) -> float:
    """Precision weight from a resultant length: kappa of a von Mises (Fisher 1993 approx)."""
    r = min(max(r, 1e-6), 0.999)
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1.0 / (r ** 3 - 4 * r ** 2 + 3 * r)


def _segments(valid: np.ndarray) -> list[slice]:
    """Contiguous runs of True in ``valid``."""
    v = np.concatenate([[False], np.asarray(valid, bool), [False]])
    d = np.diff(v.astype(int))
    return [slice(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def stride_phase(tg: np.ndarray, stride_channels: dict[str, np.ndarray],
                 step_channels: dict[str, np.ndarray], valid: np.ndarray,
                 troughs: np.ndarray | None = None,
                 step_foot: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Combined stride phase (rad, unwrapped, 2*pi per stride; NaN where not ``valid``).

    ``stride_channels`` oscillate once per stride (lateral sway, head roll),
    ``step_channels`` twice (vertical bob). All are band-passed and sampled on ``tg``.
    Returns the phase and a dict with each channel's weight and concentration.
    """
    valid = np.asarray(valid, bool)
    phases: dict[str, np.ndarray] = {}
    for name, x in stride_channels.items():
        if x is not None and np.size(x) == tg.size and np.nanstd(x) > 0:
            phases[name] = to_phase(protophase(x), valid)

    # Halve step-rate phases to stride rate; choose 0 or pi per segment against the
    # stride-rate reference, if any.
    ref = next(iter(phases.values()), None)
    for name, x in step_channels.items():
        if x is None or np.size(x) != tg.size or not np.nanstd(x) > 0:
            continue
        half = to_phase(protophase(x), valid) / 2.0
        if ref is not None:
            # A constant +pi is absorbed by the alignment below; what matters is that all
            # segments make the same choice. The longest segment sets the reference.
            segs = sorted(_segments(valid), key=lambda s: s.stop - s.start, reverse=True)
            if segs:
                off_ref, _ = _circ_offset(half[segs[0]], ref[segs[0]])
                for seg in segs[1:]:
                    off, _ = _circ_offset(half[seg], ref[seg])
                    if abs(np.angle(np.exp(1j * (off - off_ref)))) > np.pi / 2:
                        half[seg] = half[seg] + np.pi
        phases[f"{name}(step/2)"] = half

    if not phases:
        return np.full(tg.shape, np.nan), {"channels": {}}

    # Align channels to the first and weight by concentration around the mean.
    names = list(phases)
    base = phases[names[0]]
    aligned = {}
    for n in names:
        off, _ = _circ_offset(phases[n][valid], base[valid])
        aligned[n] = phases[n] - off
    z = np.mean([np.exp(1j * aligned[n]) for n in names], axis=0)
    prelim = np.angle(z)
    weights, conc = {}, {}
    for n in names:
        _, r = _circ_offset(aligned[n][valid], prelim[valid])
        conc[n] = round(r, 4)
        weights[n] = _kappa_weight(r)
    wsum = sum(weights.values())
    z = sum(weights[n] * np.exp(1j * aligned[n]) for n in names) / wsum
    phi = np.unwrap(np.angle(z))
    phi = to_phase(phi, valid)

    # Origin: 0 = trough starting a foot-A step (pi = foot B), as in gait.phase_at. Set per
    # unbroken segment, since the phase can resume with any offset after a gap.
    if troughs is not None and np.size(troughs) >= 2:
        tr = np.asarray(troughs, float)
        foot = (np.asarray(step_foot, int)[: tr.size] if step_foot is not None
                and np.size(step_foot) else np.arange(tr.size) % 2)
        tr = tr[: foot.size]
        at = np.interp(tr, tg, phi, left=np.nan, right=np.nan)
        target = np.pi * foot
        glob_off, _ = _circ_offset(at, target)
        for seg in _segments(valid):
            inside = (tr >= tg[seg.start]) & (tr <= tg[seg.stop - 1])
            off = (_circ_offset(at[inside], target[inside])[0]
                   if inside.sum() >= 4 else glob_off)
            phi[seg] = phi[seg] - off

    phi = np.where(valid, phi, np.nan)
    return phi, {"channels": {n: {"weight": round(weights[n] / wsum, 4),
                                  "concentration": conc[n]} for n in names}}
