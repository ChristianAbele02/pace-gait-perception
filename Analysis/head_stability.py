"""Exploratory: head stabilisation over the stride (compensatory head pitch).

During walking the head pitches against its vertical translation to keep the line of sight
on a point ahead. Pozzo et al. (1990) showed pitch stabilisation in walking, running and
hopping; Hirasaki et al. (1999) described the pitch as aiming the head at a head fixation
point (HFP) ahead of the walker, roughly independent of speed. If performance varies with
stride phase, phases of weakest stabilisation are a likely mechanism.

Per walking block, from the logged 6-DOF head pose:

* vertical head translation ``y`` (gait band of ``head_local_pos_y``, as in gait.py) and
  head pitch ``p`` (nose down positive; see Pitch axis);
* step-band coupling: coherence, phase and gain of pitch vs the bob at the step frequency
  (coherence and gain do not depend on the pitch sign);
* head fixation point ``D* = cov(y, tan p) / var(tan p)`` (m) on the step band, the
  distance at which pitch cancels the translation for a target straight ahead. The step
  band is used because the compensatory pitch follows the twice-per-stride bob, while
  broadband pitch includes slower movements. Positive = compensatory (head looks down when
  high), negative = anti-compensatory;
* compensation at the stimulus distance (2 m), step band: ``1 - RMS(e) / RMS(e0)`` with
  ``e0 = atan(y / D)`` the angular error from translation alone and ``e = e0 - p`` the
  error after pitch. 1 = perfect, 0 = none, < 0 = worse. Also given with mirrored pitch
  sign until the sign is verified;
* stride-phase profiles (50 bins) of translation, pitch, both gaze errors and angular
  speed. Angular speed uses the gait-band rotation, since frame-to-frame differences of
  the raw pose mostly reflect tracking jitter.

Pitch axis. The rigid body is not aligned with the room and Euler angles wrap, so pitch is
not taken from an Euler angle. Rotation relative to the block's median pose is expressed as
a room-frame rotation vector. Pitch is the horizontal axis with most power at the step
frequency (pitch follows the bob twice per stride; roll and yaw once), oriented toward room
+x, the participant's right when facing +z (see ``config``). Positive rotation about +x
turns +z down (nose down). This relies on the same room-axis assumption as the stance-foot
labels (``config.LATERAL_SIGN_VERIFIED``); a mirrored frame swaps compensatory and
anti-compensatory. The measured step-band pitch lies 125-170 deg from the bob, i.e.
anti-compensatory, contradicting the literature, so a mirrored rotation is more
likely. Until verified (``config.HEAD_PITCH_SIGN_VERIFIED``) the sign of fixation point and
compensation is not interpreted.

Not measured: retinal motion of the Gabor. The patch is head-locked and the render eye only
follows head motion below 1.8 Hz (``ProjectionPlane.swayGain = 0``), so the step-rate bob
hardly reaches the image. How much residual motion reaches the stimulus depends on
rendering and eye movements (not recorded).

References
----------
Pozzo T, Berthoz A, Lefort L (1990). Head stabilization during various locomotor tasks
    in humans. I. Normal subjects. Exp Brain Res 82:97-106.
Hirasaki E, Moore ST, Raphan T, Cohen B (1999). Effects of walking velocity on vertical
    head and body movements during locomotion. Exp Brain Res 127:117-130.

Usage::

    python -m Analysis.head_stability
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, detrend, welch
from scipy.spatial.transform import Rotation, Slerp

try:
    from . import bids_io
    from . import config as C
    from . import gait as gait_mod
    from . import phaser
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import gait as gait_mod
    import phaser
    import plotstyle as S

S.apply()

STIM_DISTANCE_M = 2.0       # head-locked Gabor plane 2 m ahead (FollowerPlane)
N_BINS = 50                 # stride-phase bins
STEP_BAND_REL = 0.25        # step band = step frequency +-25 %
MAX_POSE_DEV_DEG = 45.0     # larger deviation from the median pose = tracking error
QUAT = ["head_mocap_quat_x", "head_mocap_quat_y", "head_mocap_quat_z", "head_mocap_quat_w"]


# ---------------------------------------------------------------------------
# Head rotation on the gait grid
# ---------------------------------------------------------------------------
def _good_pose_frames(motion) -> np.ndarray:
    """Frames with a fresh, valid, finite mocap quaternion."""
    ok = np.ones(len(motion), bool)
    for c in ("head_mocap_valid", "mocap_frame_advanced"):
        if c in motion:
            ok &= motion[c].to_numpy() == 1
    for c in QUAT:
        ok &= np.isfinite(motion[c].to_numpy(float))
    return ok


def rotation_on_grid(motion, tg: np.ndarray, valid_grid: np.ndarray) -> dict | None:
    """Head rotation relative to the block's median pose, interpolated onto ``tg``.

    Returns ``rotvec`` (n, 3; deg, room axes), ``valid`` (grid mask) and
    ``pose_outlier_pct``. The reference pose is the mean of frames within
    ``MAX_POSE_DEV_DEG`` of a first mean, so tracking flips do not bias it. Rotations are
    slerped between good frames; grid points in head-data gaps (``valid_grid`` False) or
    far from a pose sample are invalid.
    """
    if not all(c in motion for c in QUAT):
        return None
    t = motion["t"].to_numpy(float)
    ok = _good_pose_frames(motion)
    if ok.sum() < 100:
        return None
    rot = Rotation.from_quat(motion.loc[ok, QUAT].to_numpy(float))
    r0 = rot.mean()
    dev = (rot * r0.inv()).magnitude()
    keep = np.degrees(dev) < MAX_POSE_DEV_DEG
    r0 = rot[keep].mean()                       # mean pose without outliers
    tt, rr = t[ok][keep], rot[keep]
    uniq = np.concatenate([[True], np.diff(tt) > 0])
    tt, rr = tt[uniq], rr[uniq]
    inside = (tg >= tt[0]) & (tg <= tt[-1])
    rv = np.full((tg.size, 3), np.nan)
    if inside.sum() < 10:
        return None
    rg = Slerp(tt, rr)(tg[inside])
    rv[inside] = np.degrees((rg * r0.inv()).as_rotvec())
    # require a nearby real pose sample
    near = np.abs(tg - tt[np.clip(np.searchsorted(tt, tg), 0, tt.size - 1)])
    valid = inside & valid_grid & (near < gait_mod.GAP_S)
    rv[~valid] = np.nan
    return {"rotvec": rv, "valid": valid,
            "pose_outlier_pct": round(100.0 * (1 - keep.mean()), 2)}


def pitch_axis(rv: np.ndarray, fs: float, step_hz: float) -> tuple[np.ndarray, float]:
    """Horizontal axis (unit, room frame) whose rotation has most power at ``step_hz``.

    Oriented toward +x. Returns (axis, share of its power within +-15 % of the step
    frequency).
    """
    ok = np.all(np.isfinite(rv), axis=1)
    # The spectrum along u = (cos a, 0, sin a) is cos² P_xx + sin² P_zz + 2 cos sin Re P_xz,
    # so three spectra cover all candidate angles.
    x = detrend(rv[ok], axis=0)
    nper = min(x.shape[0], int(8 * fs))
    f, p_xx = welch(x[:, 0], fs=fs, nperseg=nper)
    _, p_zz = welch(x[:, 2], fs=fs, nperseg=nper)
    _, p_xz = csd(x[:, 0], x[:, 2], fs=fs, nperseg=nper)
    band = (f > 0.3) & (f < 5.0)
    near = band & (np.abs(f - step_hz) < 0.15 * step_hz)
    best, best_share = np.array([1.0, 0.0, 0.0]), -1.0
    for a in np.radians(np.arange(0, 180, 5)):
        c, s = np.cos(a), np.sin(a)
        pxx = c * c * p_xx + s * s * p_zz + 2 * c * s * p_xz.real
        share = pxx[near].sum() / max(pxx[band].sum(), 1e-12)
        if share > best_share:
            best, best_share = np.array([c, 0.0, s]), share
    if best[0] < 0:
        best = -best
    return best, float(best_share)


def _bin_by_phase(phase_pct: np.ndarray, x: np.ndarray, nbins: int = N_BINS):
    """Mean, SD and count of ``x`` in ``nbins`` stride-phase bins (NaN for empty bins)."""
    ok = np.isfinite(phase_pct) & np.isfinite(x)
    idx = np.clip((phase_pct[ok] / 100.0 * nbins).astype(int), 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins)
    mean = np.bincount(idx, x[ok], minlength=nbins) / np.maximum(cnt, 1)
    sq = np.bincount(idx, x[ok] ** 2, minlength=nbins) / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(sq - mean ** 2, 0))
    mean[cnt == 0] = np.nan
    sd[cnt == 0] = np.nan
    return mean, sd, cnt


def block_signals(rec) -> dict | None:
    """Vertical translation, pitch, angular speed and trough stride phase of one walking
    block on the gait grid. Also used by gait_phase_methods (pitch as held-out signal)."""
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    tg, y, troughs = g["t"], g["y_filt"], g["troughs"]
    if tg.size < 100 or troughs.size < 10:
        return None
    step_foot = g["step_foot"] if np.size(g["step_foot"]) else None
    fs = 1.0 / np.median(np.diff(tg))
    motion, _, _ = bids_io.load_motion(rec)
    # grid is valid where gait.py did not bridge a gap (Hilbert phase not NaN)
    cont = g["stride_phase_cont"] if np.size(g["stride_phase_cont"]) else np.zeros(tg.size)
    rot = rotation_on_grid(motion, tg, np.isfinite(cont))
    if rot is None:
        return None
    step_hz = 1.0 / float(np.median(np.diff(troughs)))
    axis, share = pitch_axis(rot["rotvec"], fs, step_hz)
    pitch = gait_mod._bandpass(np.nan_to_num(detrend(np.nan_to_num(rot["rotvec"] @ axis))),
                               fs, 0.5, gait_mod.LOWPASS_HZ)
    pitch[~rot["valid"]] = np.nan
    # Angular speed from the gait-band rotation; raw frame differences amplify jitter
    # (1 deg per frame is 60 deg/s at 60 Hz).
    rvf = np.column_stack([gait_mod._bandpass(np.nan_to_num(detrend(np.nan_to_num(rot["rotvec"][:, j]))),
                                              fs, 0.5, gait_mod.LOWPASS_HZ) for j in range(3)])
    speed = np.linalg.norm(np.gradient(rvf, axis=0), axis=1) * fs
    speed[~rot["valid"]] = np.nan
    _, phase = gait_mod.phase_at(tg, troughs, step_foot)
    return {"g": g, "tg": tg, "fs": fs, "y": np.where(rot["valid"], y, np.nan),
            "pitch": pitch, "speed": speed, "phase": phase,
            "axis": axis, "axis_step_share": share, "step_hz": step_hz,
            "pose_outlier_pct": rot["pose_outlier_pct"]}


def _step_band(x: np.ndarray, fs: float, step_hz: float) -> np.ndarray:
    """``x`` band-passed to +-STEP_BAND_REL around the step frequency (NaN kept as NaN)."""
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if ok.sum() < 100:
        return out
    filt = gait_mod._bandpass(np.where(ok, x, 0.0), fs,
                              step_hz * (1 - STEP_BAND_REL), step_hz * (1 + STEP_BAND_REL))
    out[ok] = filt[ok]
    return out


def _step_coupling(y, p, fs, step_hz) -> tuple[float, float, float]:
    """Coherence, phase (deg, pitch vs vertical) and gain (deg/mm) at the step frequency,
    from the longest gap-free stretch."""
    ok = np.isfinite(y) & np.isfinite(p)
    segs = phaser._segments(ok)
    seg = max(segs, key=lambda s: s.stop - s.start) if segs else slice(0, 0)
    if seg.stop - seg.start < int(16 * fs):
        return float("nan"), float("nan"), float("nan")
    ys, ps = y[seg], p[seg]
    n = min(ys.size, int(8 * fs))
    f, coh = coherence(ys, ps, fs=fs, nperseg=n)
    _, pyp = csd(ys, ps, fs=fs, nperseg=n)
    _, pyy = welch(ys, fs=fs, nperseg=n)
    i = int(np.argmin(np.abs(f - step_hz)))
    return (float(coh[i]), float(np.degrees(np.angle(pyp[i]))),
            float(np.abs(pyp[i]) / max(pyy[i], 1e-18) / 1000.0))


def analyse_block(rec) -> dict | None:
    """Head-stabilisation metrics and stride-phase profiles for one walking block.

    Fixation point and compensation use the step band, where the compensatory pitch acts;
    broadband pitch includes slower unrelated movement. Profiles are broadband.
    """
    b = block_signals(rec)
    if b is None:
        return None
    y, p, D, fs, sh = b["y"], b["pitch"], STIM_DISTANCE_M, b["fs"], b["step_hz"]
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 500:
        return None
    ys, ps = _step_band(y, fs, sh), _step_band(p, fs, sh)
    oks = np.isfinite(ys) & np.isfinite(ps)
    tp = np.tan(np.radians(ps[oks]))
    hfp = float(np.cov(ys[oks], tp, bias=True)[0, 1] / np.var(tp)) if np.var(tp) > 0 else float("nan")
    e0s = np.degrees(np.arctan(ys[oks] / D))
    comp = 1.0 - np.sqrt(np.mean((e0s - ps[oks]) ** 2)) / np.sqrt(np.mean(e0s ** 2))
    # same with mirrored pitch sign (config.HEAD_PITCH_SIGN_VERIFIED)
    comp_m = 1.0 - np.sqrt(np.mean((e0s + ps[oks]) ** 2)) / np.sqrt(np.mean(e0s ** 2))
    coh, phase, gain = _step_coupling(y, p, fs, sh)
    r = float(np.corrcoef(y[ok], p[ok])[0, 1])
    e0 = np.degrees(np.arctan(y / D))          # error from translation only
    e = e0 - p                                  # error after head pitch
    prof = {k: _bin_by_phase(b["phase"], v) for k, v in
            (("y_mm", 1000 * y), ("pitch_deg", p), ("err0_deg", e0), ("err_deg", e),
             ("speed_dps", b["speed"]))}
    worst = int(np.nanargmax(np.abs(prof["err_deg"][0])))
    return {
        "recording": rec.label(), "sub": rec.sub,
        "pitch_sign_verified": bool(getattr(C, "HEAD_PITCH_SIGN_VERIFIED", False)),
        "vertical_rms_mm": round(float(1000 * np.nanstd(y[ok])), 2),
        "pitch_rms_deg": round(float(np.nanstd(p[ok])), 3),
        "pitch_step_band_rms_deg": round(float(np.nanstd(ps[oks])), 3),
        "pitch_axis_room": [round(float(v), 3) for v in b["axis"]],
        "pitch_axis_step_power_share": round(b["axis_step_share"], 3),
        "step_coherence": round(coh, 3),
        "step_phase_deg": round(phase, 1),
        "step_gain_deg_per_mm": round(gain, 4),
        "corr_vertical_pitch": round(r, 3),
        "head_fixation_point_m": round(hfp, 3),
        "compensation_at_2m": round(float(comp), 3),
        "compensation_at_2m_if_sign_mirrored": round(float(comp_m), 3),
        "median_angular_speed_dps": round(float(np.nanmedian(b["speed"])), 2),
        "worst_gaze_error_phase_pct": round((worst + 0.5) * 100 / N_BINS, 1),
        "pose_outlier_pct": b["pose_outlier_pct"],
        "n_samples": int(ok.sum()),
        "_profiles": {k: {"mean": v[0].tolist(), "sd": v[1].tolist(), "n": v[2].tolist()}
                      for k, v in prof.items()},
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pool_profiles(blocks: list[dict]) -> dict:
    """Sample-weighted mean of block profiles."""
    out = {}
    for k in blocks[0]["_profiles"]:
        m = np.array([b["_profiles"][k]["mean"] for b in blocks], float)
        n = np.array([b["_profiles"][k]["n"] for b in blocks], float)
        s = np.array([b["_profiles"][k]["sd"] for b in blocks], float)
        w = np.where(np.isfinite(m), n, 0)
        mean = np.nansum(np.nan_to_num(m) * w, 0) / np.maximum(w.sum(0), 1)
        sd = np.sqrt(np.nansum(np.nan_to_num(s) ** 2 * w, 0) / np.maximum(w.sum(0), 1))
        out[k] = (mean, sd)
    return out


def _figure(by_sub: dict[str, list[dict]], out: Path) -> None:
    """Plot stride-phase profiles per participant: translation, pitch, gaze error, speed."""
    x = (np.arange(N_BINS) + 0.5) * 100 / N_BINS
    subs = sorted(by_sub)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    specs = [(ax[0, 0], "y_mm", "vertical head position (mm)", "a", "Vertical translation"),
             (ax[0, 1], "pitch_deg", "head pitch, nose down (°)", "b", "Head pitch"),
             (ax[1, 0], None, "gaze error at 2 m (°)", "c", "Gaze error, target at 2 m"),
             (ax[1, 1], "speed_dps", "head angular speed (°/s)", "d", "Head angular speed")]
    for i, sub in enumerate(subs):
        col = S.CYCLE[i % len(S.CYCLE)]
        prof = _pool_profiles(by_sub[sub])
        for a, key, _, _, _ in specs:
            if key is None:
                m0, _ = prof["err0_deg"]
                m, sd = prof["err_deg"]
                a.plot(x, m0, color=col, ls="--", lw=1.2)
                a.plot(x, m, color=col, lw=2, label=S.nice_label(sub))
                a.fill_between(x, m - sd, m + sd, color=col, alpha=0.12, lw=0)
            else:
                m, sd = prof[key]
                a.plot(x, m, color=col, lw=2, label=S.nice_label(sub))
                a.fill_between(x, m - sd, m + sd, color=col, alpha=0.12, lw=0)
    for a, _, ylab, tag, title in specs:
        a.set_ylabel(ylab)
        S.panel(a, tag, title, fontsize=10)
        for xv in (0, 50, 100):
            a.axvline(xv, color=S.GREY, lw=0.8, ls=":")
    for a in ax[1]:
        a.set_xlabel("stride phase (%; 0 and 50 = double support)")
    ax[1, 0].plot([], [], color=S.GREY, ls="--", label="translation only")
    ax[1, 0].plot([], [], color=S.GREY, lw=2, label="with head pitch")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Head stabilisation over the stride (exploratory)", fontsize=13,
                 fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Mean ± SD over strides, all walking blocks of a participant. Panel c: "
                   "angular error of a target 2 m ahead from vertical translation alone "
                   "(dashed) and after the head's pitch (solid); closer to 0 = better "
                   "gaze stabilisation. Pitch sign assumes the room axes (see module).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    S.save(fig, out)


def _summary_figure(rows: list[dict], out: Path) -> None:
    """Plot fixation point, compensation and coupling per walking block."""
    subs = sorted({r["sub"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    for j, (key, lab) in enumerate((("head_fixation_point_m", "head fixation point (m)"),
                                    ("compensation_at_2m", "compensation at 2 m"),
                                    ("corr_vertical_pitch", "r (vertical, pitch)"))):
        for i, sub in enumerate(subs):
            v = [r[key] for r in rows if r["sub"] == sub]
            ax[j].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v,
                          color=S.CYCLE[i % len(S.CYCLE)], s=30, zorder=3)
        ax[j].axhline(0, color=S.GREY, lw=0.8)
        ax[j].set_xticks(range(len(subs)))
        ax[j].set_xticklabels([S.nice_label(s) for s in subs])
        ax[j].set_ylabel(lab)
        ax[j].set_xlim(-0.6, len(subs) - 0.4)
    ax[0].axhline(STIM_DISTANCE_M, color=S.RED, lw=0.8, ls="--")
    S.panel(ax[0], "a", "Head fixation point", fontsize=10)
    S.panel(ax[1], "b", "Compensation", fontsize=10)
    S.panel(ax[2], "c", "Translation–pitch coupling", fontsize=10)
    S.caption(fig, "One point per walking block. Fixation point > 0 = compensatory pitch "
                   "(red dashed: the 2 m stimulus distance). Compensation 1 = perfect, "
                   "0 = none, < 0 = pitch adds to the error.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, out)


def main() -> list[dict]:
    """Head-stabilisation metrics per walking block and profiles per participant."""
    rows = []
    for rec in bids_io.find_recordings(scope="gait"):
        r = analyse_block(rec)
        if r is None:
            continue
        rows.append(r)
        out = bids_io.deriv_dir("head_stability", rec.sub, rec.ses)
        (out / f"{rec.label()}_head_stability.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8")
        print(f"  {rec.label()}: step coherence {r['step_coherence']}, phase "
              f"{r['step_phase_deg']} deg; fixation point {r['head_fixation_point_m']} m; "
              f"compensation at 2 m {r['compensation_at_2m']} "
              f"(mirrored sign {r['compensation_at_2m_if_sign_mirrored']}); "
              f"vertical {r['vertical_rms_mm']} mm, step-band pitch "
              f"{r['pitch_step_band_rms_deg']} deg RMS")
    bids_io.gate_report("head_stability", "gait", condition="walking",
                        n_analysed=len({r["sub"] for r in rows}))
    if not rows:
        print("No walking blocks with gait + head pose. Run gait.py first.")
        return rows
    by_sub: dict[str, list[dict]] = {}
    for r in rows:
        by_sub.setdefault(r["sub"], []).append(r)
    out = bids_io.deriv_dir("head_stability")
    figs = bids_io.fig_dir("head_stability")
    _figure(by_sub, figs / "head_stability_profiles.png")
    _summary_figure(rows, figs / "head_stability_summary.png")
    print(f"Head stability -> {out}")
    return rows


if __name__ == "__main__":
    main()
