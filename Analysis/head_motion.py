"""Exploratory: how the head moves over the stride.

* Head stabilisation: head pitch against the vertical bob; fixation point and gaze
  compensation at the stimulus distance.
* Head kinematics: translation and rotation over the stride, by stance foot.

Usage::

    python -m Analysis.head_motion


Head stabilisation
==================

Exploratory: head stabilisation over the stride (gaze-stabilising head pitch).

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
* head fixation point ``D = 1 mm / tan(gain)`` (m) from the step-frequency gain, the ratio
  of pitch to vertical amplitude (Hirasaki et al., 1999): the distance at which a pitch of
  that size would cancel the translation for a target straight ahead, if it were in time;
* pitch lag behind the bob at the step frequency (deg; 0 = in time, 180 =
  anti-compensatory). A lagging pitch of the right size still leaves an error of about
  ``2 sin(lag / 2)`` of the translation's;
* in-phase fixation point ``cov(y, tan p) / var(tan p)`` (m) on the step band: the
  least-squares distance, which the lag shrinks towards 0 (about ``D cos(lag)``).
  Positive = compensatory (head looks down when high), negative = anti-compensatory;
* stream offset: how far the vertical position (``head_local_pos_y``, Unity's transform)
  lags the mocap client's own position, the stream the quaternion comes from. A latency
  difference between the two would show up as pitch lag;
* compensation at the stimulus distance (2 m), step band: ``1 - RMS(e) / RMS(e0)`` with
  ``e0 = atan(y / D)`` the angular error from translation alone and ``e = e0 - p`` the
  error after the head's rotation. 1 = perfect, 0 = none, < 0 = worse;
* stride-phase profiles (50 bins) of translation, pitch, both gaze errors and angular
  speed. Angular speed uses the gait-band rotation, since frame-to-frame differences of
  the raw pose mostly reflect tracking jitter.

Pitch axis. The rigid body is not aligned with the room and Euler angles wrap, so pitch is
not taken from an Euler angle. Rotation relative to the block's median pose is expressed as
a rotation vector in the frame of the logged Motive quaternion (x right, y up, z backward;
see ``config``). Pitch is the horizontal axis with most power at the step frequency (pitch
follows the bob twice per stride; roll and yaw once), oriented toward -x, so positive
rotation turns the forward direction (-z) down: nose down. The rig check measured that sign
with a chin-down nod (``rig_check.py``, ``config.HEAD_PITCH_SIGN_VERIFIED``).

Not measured: retinal motion of the Gabor. The patch follows the head's position but not
its rotation (head roll tilts it on the retina; see ``response_bias.py``), and the render eye
only follows head motion below 1.8 Hz (``ProjectionPlane.swayGain = 0``), so the step-rate
bob hardly reaches the image. How much residual motion reaches the stimulus depends on
rendering and eye movements (not recorded).

References
----------
Pozzo T, Berthoz A, Lefort L (1990). Head stabilization during various locomotor tasks
    in humans. I. Normal subjects. Exp Brain Res 82:97-106.
Hirasaki E, Moore ST, Raphan T, Cohen B (1999). Effects of walking velocity on vertical
    head and body movements during locomotion. Exp Brain Res 127:117-130.


Head kinematics
===============

Exploratory: head kinematics over the stride, by stance foot.

What the head does during one stride, channel by channel, so that the stride-phase
results can be read against the motion that produces them: vertical, lateral and
fore-aft translation, and roll, pitch and yaw. Head height traces a near-sinusoid with
two cycles per stride (Hwang & Effenberg, 2021); lateral sway, roll and yaw alternate
with the stance foot and so repeat once per stride; pitch follows the vertical bob
(Pozzo et al., 1990). With the stance-foot labels from ``gait.label_steps`` fixing the
stride origin, the first half of every profile is the right foot's step and the second
half the left foot's, so the once-per-stride components are visible instead of averaging
out.

Rotation
--------
The rigid body's axes are not aligned with the head, so no Euler channel is roll or pitch
by construction. Rotation is taken relative to the block's mean head pose as a rotation
vector in the frame of the logged Motive quaternion (x right, y up, z backward;
``head_motion.rotation_on_grid``) and projected onto room axes: roll about +z
(positive = right ear up), yaw about -y (positive = turning right), and pitch about the
horizontal axis with most power at the step rate (``head_motion.pitch_axis``, close to
x; positive = nose down). The mocap frame is the same in every session, and the rig check
measured each sign (``rig_check.py``, ``config``).

Roll is the channel that matters most for the task. The Gabor plane follows the tracked
head POSITION but not its rotation (``CameraHolder`` tracks position only), so the
grating stays upright in the room and a head roll of r degrees rotates it on the retina
by r, less whatever ocular counter-roll compensates. ``roll_series`` is the roll this
module computes, shared with ``response_bias.py``.

Per walking block every channel is band-passed to 0.3-5 Hz (drift out, stride and step
rates in), binned in 50 stride-phase bins and averaged per participant over blocks. The
amplitude of each participant's mean profile at 1 and 2 cycles per stride is reported.

Coupling to the sway. Where each participant's once-per-stride yaw and roll peak
relative to the head's sideways excursion, in % of the stride:

* ``yaw_lag_pct``: the turn to the right after the head's leftmost point (and the turn to
  the left after its rightmost). Near 0, the head turns back towards the midline as it is
  carried out, as a target straight ahead requires; near +-50, it turns outwards with the
  sway. ``yaw_fixation_m`` is the distance at which a yaw of the observed size would hold
  a target straight ahead against a sway of the observed size (lateral / tan yaw), if in
  time;
* ``roll_lead_pct``: the left-ear-down roll peak (right ear up) relative to the head's
  leftmost point; negative = the head tilts towards the side it is moving to before it
  gets there.

Outputs (``derivatives/head_kinematics/``): ``head_kinematics.json`` (per-participant
amplitudes, coupling and their medians), ``figures/head_kinematics.png``.

References
----------
Hwang T-H, Effenberg AO (2021). Head trajectory diagrams for gait symmetry analysis
    using a single head-worn IMU. Sensors 21:6621.
Pozzo T, Berthoz A, Lefort L (1990). Head stabilization during various locomotor tasks
    in humans. I. Normal subjects. Exp Brain Res 82:97-106.
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
    from . import clean
    from . import config as C
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import clean
    import config as C
    import gait as gait_mod
    import phase_analysis as pa
    import plotstyle as S

S.apply()


# ###########################################################################
# Head stabilisation
# ###########################################################################
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
    frames = np.zeros(len(motion), bool)
    frames[np.flatnonzero(ok)[keep]] = True
    return {"rotvec": rv, "valid": valid, "pose_frames": frames,
            "pose_outlier_pct": round(100.0 * (1 - keep.mean()), 2)}


def stream_offset(motion, pose_frames: np.ndarray) -> tuple[float, float]:
    """``(ms, r)``: lag of ``head_local_pos_y`` behind the mocap client's vertical
    position, from the peak of their cross-correlation in the gait band (parabolic
    refinement on a 240 Hz grid), and the correlation at that peak. ``pose_frames`` are
    the mocap rows ``rotation_on_grid`` kept (fresh, valid, not a foreign rigid body)."""
    if not {"head_mocap_pos_y", "head_local_pos_y"} <= set(motion.columns):
        return float("nan"), float("nan")
    t = motion["t"].to_numpy(float)
    series = []
    for col, ok in (("head_local_pos_y", np.ones(len(t), bool)),
                    ("head_mocap_pos_y", pose_frames)):
        v = motion[col].to_numpy(float)
        ok = ok & np.isfinite(v)
        if ok.sum():                       # head height stays within 15 cm on the treadmill
            ok &= np.abs(v - np.median(v[ok])) < 0.15
        tt, vv = t[ok], v[ok]
        uniq = np.concatenate([[True], np.diff(tt) > 0])
        series.append((tt[uniq], vv[uniq]))
    if min(s[0].size for s in series) < 1000:
        return float("nan"), float("nan")
    fs = 240.0
    tg = np.arange(max(s[0][0] for s in series), min(s[0][-1] for s in series), 1 / fs)
    a, b = (gait_mod._bandpass(np.interp(tg, tt, vv), fs, 0.5, gait_mod.LOWPASS_HZ)
            for tt, vv in series)
    m = int(0.15 * fs)
    a, b = a[m:-m], b[m:-m]
    lags = np.arange(-m, m + 1)
    cc = np.array([np.corrcoef(a[m + L: a.size - m + L], b[m: b.size - m])[0, 1]
                   for L in lags])
    i = int(np.argmax(cc))
    shift = 0.0
    if 0 < i < lags.size - 1:
        den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
        shift = 0.5 * (cc[i - 1] - cc[i + 1]) / den if den else 0.0
    return float((lags[i] + shift) / fs * 1000.0), float(cc[i])


def pitch_axis(rv: np.ndarray, fs: float, step_hz: float) -> tuple[np.ndarray, float]:
    """Horizontal axis (unit, mocap frame) whose rotation has most power at ``step_hz``.

    Oriented toward -x, so positive rotation about it is nose down (the mocap frame's z
    points backward; ``config``, ``rig_check``). Returns (axis, share of its power within
    +-15 % of the step frequency).
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
    return pitch_axis_orientation(best), float(best_share)


def pitch_axis_orientation(axis: np.ndarray) -> np.ndarray:
    """``axis`` pointing towards -x, so that positive rotation about it is nose down."""
    return -axis if axis[0] > 0 else axis


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
    block on the gait grid."""
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
            "pose_outlier_pct": rot["pose_outlier_pct"],
            "stream_offset": stream_offset(motion, rot["pose_frames"])}


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


def _segments(valid: np.ndarray) -> list[slice]:
    """Contiguous runs of True in ``valid``."""
    v = np.concatenate([[False], np.asarray(valid, bool), [False]])
    d = np.diff(v.astype(int))
    return [slice(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def _step_coupling(y, p, fs, step_hz) -> tuple[float, float, float]:
    """Coherence, phase (deg, pitch vs vertical) and gain (deg/mm) at the step frequency,
    from the longest gap-free stretch."""
    ok = np.isfinite(y) & np.isfinite(p)
    segs = _segments(ok)
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


def fixation_point_m(gain_deg_per_mm: float) -> float:
    """Distance (m) at which a pitch of ``gain`` degrees per mm of vertical translation,
    in time with it, would hold a target straight ahead."""
    return float(1e-3 / np.tan(np.radians(gain_deg_per_mm))) if gain_deg_per_mm > 0         else float("nan")


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
    coh, phase, gain = _step_coupling(y, p, fs, sh)
    offset_ms, offset_r = b["stream_offset"]
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
        "pitch_lag_deg": round(-phase, 1),
        "corr_vertical_pitch": round(r, 3),
        "head_fixation_point_m": round(fixation_point_m(gain), 3),
        "inphase_fixation_point_m": round(hfp, 3),
        "compensation_at_2m": round(float(comp), 3),
        "median_angular_speed_dps": round(float(np.nanmedian(b["speed"])), 2),
        "worst_gaze_error_phase_pct": round((worst + 0.5) * 100 / N_BINS, 1),
        "pose_outlier_pct": b["pose_outlier_pct"],
        "stream_offset_ms": round(offset_ms, 1),
        "stream_offset_r": round(offset_r, 3),
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
    """Plot stride-phase profiles: each participant thin and grey, the group mean and SD bold."""
    x = (np.arange(N_BINS) + 0.5) * 100 / N_BINS
    subs = sorted(by_sub)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    specs = [(ax[0, 0], "y_mm", "vertical head position (mm)", "a", "Vertical translation"),
             (ax[0, 1], "pitch_deg", "head pitch, nose down (°)", "b", "Head pitch"),
             (ax[1, 0], "err_deg", "gaze error at 2 m (°)", "c", "Gaze error, target at 2 m"),
             (ax[1, 1], "speed_dps", "head angular speed (°/s)", "d", "Head angular speed")]
    profs = {sub: _pool_profiles(by_sub[sub]) for sub in subs}

    def group(a, key, color, ls="-", label="mean ± SD across participants"):
        m = np.array([profs[s][key][0] for s in subs], float)
        for row in m:
            a.plot(x, row, color=S.MIST, lw=0.9, ls=ls, zorder=1)
        mu, sd = np.nanmean(m, 0), np.nanstd(m, 0, ddof=1) if len(m) > 1 else np.zeros(len(x))
        a.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, lw=0, zorder=2)
        a.plot(x, mu, color=color, lw=2.4, ls=ls, zorder=3, label=label)

    for a, key, ylab, tag, title in specs:
        if key == "err_deg":
            group(a, "err0_deg", S.GREY, ls="--", label="translation only, mean")
            group(a, key, S.RED, label="with head pitch, mean")
        else:
            group(a, key, S.BLUE)
        a.set_ylabel(ylab)
        S.panel(a, tag, title, fontsize=10)
        for xv in (0, 50, 100):
            a.axvline(xv, color=S.GREY, lw=0.8, ls=":")
    for a in ax[1]:
        a.set_xlabel("stride phase (%; 0 and 50 = double support)")
    ax[0, 0].plot([], [], color=S.MIST, lw=0.9, label="one participant")
    for a in (ax[0, 0], ax[1, 0]):
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi + 0.3 * (hi - lo))          # headroom: the legend sits above
        a.legend(loc="upper center", ncol=2, fontsize=8)
    fig.suptitle(f"Head stabilisation over the stride (exploratory), n = {len(subs)} "
                 "participants", fontsize=13, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Grey: each participant's mean profile over the strides of all their "
                   "walking blocks. Bold: the mean of those profiles ± SD across "
                   "participants. c: angular error of a target 2 m ahead caused by the "
                   "head's vertical translation alone (dashed) and after its pitch (solid); "
                   "closer to 0 = better gaze stabilisation. Positive pitch = nose down "
                   "(rig check).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    S.save(fig, out)


def _summary_figure(rows: list[dict], out: Path) -> None:
    """Plot fixation point, pitch lag and compensation, one point per walking block."""
    subs = sorted({r["sub"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
    for j, (key, lab) in enumerate((("head_fixation_point_m", "head fixation point (m)"),
                                    ("pitch_lag_deg", "pitch lag behind the bob (°)"),
                                    ("compensation_at_2m", "compensation at 2 m"))):
        for i, sub in enumerate(subs):
            v = [r[key] for r in rows if r["sub"] == sub]
            ax[j].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v,
                          color=S.BLUE, alpha=0.75, s=24, zorder=3)
        ax[j].axhline(0, color=S.GREY, lw=0.8)
        ax[j].set_xticks(range(len(subs)))
        ax[j].set_xticklabels([s.replace("sub-", "") for s in subs])
        ax[j].set_xlabel("participant")
        ax[j].set_ylabel(lab)
        ax[j].set_xlim(-0.6, len(subs) - 0.4)
    # Labelled in place, just under each line: a legend box would sit on the line.
    ax[0].axhline(STIM_DISTANCE_M, color=S.RED, lw=0.8, ls="--")
    ax[0].text(-0.5, STIM_DISTANCE_M, "stimulus distance", color=S.RED, fontsize=8,
               ha="left", va="top")
    ax[2].axhline(1.0, color=S.GREEN, lw=0.8, ls="--")
    ax[2].text(-0.5, 1.0, "perfect", color=S.GREEN, fontsize=8, ha="left", va="top")
    S.panel(ax[0], "a", "Head fixation point", fontsize=10)
    S.panel(ax[1], "b", "Pitch lag", fontsize=10)
    S.panel(ax[2], "c", "Compensation", fontsize=10)
    S.caption(fig, "One point per walking block. a: distance at which a pitch of the "
                   "observed size, in time with the vertical bob, would keep a point "
                   "straight ahead steady (step frequency). b: lag of the pitch behind the "
                   "bob (0 = nose lowest when the head is highest; |lag| > 90 = anti-"
                   "compensatory). c: share of the translation-induced gaze error at 2 m "
                   "the pitch removes (1 = all, 0 = none, < 0 = pitch adds to it). "
                   "Positive pitch = nose down (rig check).")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    S.save(fig, out)


def population_summary(rows: list[dict]) -> dict:
    """Median and IQR over walking blocks of the per-block metrics. The stream offset is
    summarised over blocks whose two position streams correlate at r >= 0.95; below that
    one of them carries tracking errors and its peak is not a latency."""
    def q(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if not v.size:
            return {"median": None, "q25": None, "q75": None, "min": None, "max": None, "n": 0}
        return {"median": round(float(np.median(v)), 3),
                "q25": round(float(np.percentile(v, 25)), 3),
                "q75": round(float(np.percentile(v, 75)), 3),
                "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
                "n": int(v.size)}
    keys = ("head_fixation_point_m", "pitch_lag_deg", "inphase_fixation_point_m",
            "compensation_at_2m", "step_coherence", "pitch_step_band_rms_deg",
            "vertical_rms_mm")
    clean = [r for r in rows if (r.get("stream_offset_r") or 0) >= 0.95]
    lag = np.array([r["pitch_lag_deg"] for r in rows], float)
    return {
        "note": "EXPLORATORY. Per walking block, step band. Positive pitch = nose down "
                "(rig check); positive lag = pitch after the bob; positive stream offset "
                "= head_local_pos_y after the mocap client's position.",
        "n_blocks": len(rows), "n_participants": len({r["sub"] for r in rows}),
        "n_compensatory": int(np.sum(np.abs(lag) < 90)),
        "n_compensation_at_2m_negative": int(sum(r["compensation_at_2m"] < 0 for r in rows)),
        **{k: q([r[k] for r in rows]) for k in keys},
        "stream_offset_ms": q([r["stream_offset_ms"] for r in clean]),
        "n_blocks_stream_offset": len(clean),
    }


def stability_main() -> list[dict]:
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
        print(f"  {rec.label()}: step coherence {r['step_coherence']}, pitch lag "
              f"{r['pitch_lag_deg']} deg; fixation point {r['head_fixation_point_m']} m; "
              f"compensation at 2 m {r['compensation_at_2m']}; "
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
    summary = population_summary(rows)
    (out / "head_stability_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    print(f"  {summary['n_compensatory']} of {summary['n_blocks']} blocks compensatory; "
          f"fixation point {summary['head_fixation_point_m']['median']} m, pitch lag "
          f"{summary['pitch_lag_deg']['median']} deg, compensation at 2 m "
          f"{summary['compensation_at_2m']['median']}; stream offset "
          f"{summary['stream_offset_ms']['median']} ms")
    print(f"Head stability -> {out}")
    return rows


# ###########################################################################
# Head kinematics
# ###########################################################################
BAND_HZ = (0.3, 5.0)          # drift out; stride (~0.85 Hz) and step (~1.7 Hz) rates in
# rotation axes in the mocap frame (x right, y up, z backward; config, rig_check)
ROLL_AXIS = np.array([0.0, 0.0, 1.0])   # positive = right ear up
YAW_AXIS = np.array([0.0, -1.0, 0.0])   # positive = turning right
POSITION = {"vertical": "head_local_pos_y", "lateral": "head_local_pos_x",
            "fore_aft": "head_local_pos_z"}
CHANNELS = ("vertical", "lateral", "fore_aft", "roll", "pitch", "yaw")
UNIT = {"vertical": "mm", "lateral": "mm", "fore_aft": "mm",
        "roll": "deg", "pitch": "deg", "yaw": "deg"}
TITLE = {"vertical": "Vertical", "lateral": "Lateral (sway)", "fore_aft": "Fore-aft",
         "roll": "Roll", "pitch": "Pitch", "yaw": "Yaw"}


# ---------------------------------------------------------------------------
# Signals on the gait grid
# ---------------------------------------------------------------------------
def _gait(rec):
    """The block's gait arrays, or None if gait has not been run or found too few steps."""
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    if g["t"].size < 100 or g["troughs"].size < 10:
        return None
    return g


def _valid_grid(g) -> np.ndarray:
    """Grid points gait did not bridge across a gap (its Hilbert phase is finite there)."""
    cont = g["stride_phase_cont"] if np.size(g["stride_phase_cont"]) else np.zeros(g["t"].size)
    return np.isfinite(cont)


def _band(x: np.ndarray, fs: float) -> np.ndarray:
    """``x`` detrended and band-passed to ``BAND_HZ``; NaN stays NaN."""
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if ok.sum() < 100:
        return out
    v = np.where(ok, x, 0.0)
    v[ok] = detrend(v[ok])
    out[ok] = gait_mod._bandpass(v, fs, *BAND_HZ)[ok]
    return out


def rotation_channels(motion, g) -> dict | None:
    """Roll, pitch and yaw (deg) on the gait grid, relative to the block's mean pose.

    Unfiltered apart from a linear detrend per channel, so slow changes of posture are kept
    (they tilt the retinal image as much as the gait-rate roll does). NaN where the grid
    bridges a gap or no real pose sample is near.
    """
    tg = g["t"]
    rot = rotation_on_grid(motion, tg, _valid_grid(g))
    if rot is None:
        return None
    rv = rot["rotvec"]
    fs = 1.0 / float(np.median(np.diff(tg)))
    step_hz = 1.0 / float(np.median(np.diff(g["troughs"])))
    pitch_ax, _ = pitch_axis(rv, fs, step_hz)
    out = {"roll": rv @ ROLL_AXIS, "pitch": rv @ pitch_ax, "yaw": rv @ YAW_AXIS}
    for k, v in out.items():
        ok = np.isfinite(v)
        if ok.sum() > 10:
            v = v.copy()
            v[ok] = detrend(v[ok])
            out[k] = v
    out["pitch_axis"] = pitch_ax
    out["pose_outlier_pct"] = rot["pose_outlier_pct"]
    return out


def roll_series(rec) -> tuple[np.ndarray, np.ndarray] | None:
    """``(t grid, roll deg)`` of one walking block, NaN where unmeasured (see
    ``rotation_channels``). Shared with ``response_bias.py``."""
    g = _gait(rec)
    if g is None:
        return None
    motion, _, _ = bids_io.load_motion(rec)
    rot = rotation_channels(motion, g)
    return None if rot is None else (g["t"], rot["roll"])


def _positions(motion, g) -> dict:
    """Head translation (mm) on the gait grid: spike-cleaned, frozen poses removed."""
    t = motion["t"].to_numpy(float)
    stale, _ = clean.stale_frames(motion)
    valid = _valid_grid(g)
    out = {}
    for name, col in POSITION.items():
        if col not in motion:
            continue
        v, *_ = clean.clean_channel(t, motion[col].to_numpy(float),
                                    abs_radius_m=clean.HEAD_CLEAN_AXES[col])
        v = np.asarray(v, float).copy()
        if stale is not None:
            v[stale] = np.nan
        vg = gait_mod._resample_to(g["t"], t, v)
        if vg is not None:
            out[name] = np.where(valid, 1000.0 * vg, np.nan)
    return out


# ---------------------------------------------------------------------------
# Profiles and amplitudes
# ---------------------------------------------------------------------------
def harmonic(profile: np.ndarray, f_cps: float) -> tuple[float, float]:
    """``(amplitude, preferred phase in % stride)`` of a binned profile at ``f_cps``.

    The phase is where the fitted cosine peaks within its first cycle.
    """
    centers = (np.arange(N_BINS) + 0.5) * 100.0 / N_BINS
    ok = np.isfinite(profile)
    if ok.sum() < N_BINS // 2:
        return float("nan"), float("nan")
    _, amp, phi, _ = pa.fit_one(centers[ok], profile[ok], f_cps)
    peak = (-phi / (2 * np.pi * f_cps / 100.0)) % (100.0 / f_cps)
    return float(amp), float(peak)


def block_profiles(rec) -> dict | None:
    """Stride-phase profiles of every channel for one walking block, or None.

    Blocks whose stance-foot labels are unreliable are skipped: their stride origin is
    arbitrary, so the once-per-stride components would be averaged with random signs.
    """
    if rec.condition != "walking":
        return None
    g = _gait(rec)
    if g is None:
        return None
    gj = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.json"
    verdict = json.loads(gj.read_text(encoding="utf-8")).get("foot_label_verdict", "") \
        if gj.exists() else ""
    if str(verdict).startswith("unreliable") or verdict == "unusable":
        return None
    step_foot = g["step_foot"] if np.size(g["step_foot"]) else None
    if step_foot is None:
        return None
    motion, _, _ = bids_io.load_motion(rec)
    rot = rotation_channels(motion, g)
    if rot is None:
        return None
    tg = g["t"]
    fs = 1.0 / float(np.median(np.diff(tg)))
    _, phase = gait_mod.phase_at(tg, g["troughs"], step_foot)
    sig = {**_positions(motion, g), **{k: rot[k] for k in ("roll", "pitch", "yaw")}}
    prof = {k: _bin_by_phase(phase, _band(v, fs), N_BINS)[0] for k, v in sig.items()}
    return {"recording": rec.label(), "sub": rec.sub, "profiles": prof,
            "pitch_axis_room": [round(float(v), 3) for v in rot["pitch_axis"]],
            "pose_outlier_pct": rot["pose_outlier_pct"]}


def participant_summary(blocks: list[dict]) -> dict:
    """Mean profile per channel over a participant's blocks, and its 1 / 2 cps harmonics."""
    out = {"n_blocks": len(blocks), "profiles": {}, "amplitude": {}}
    for ch in CHANNELS:
        stack = [b["profiles"][ch] for b in blocks if ch in b["profiles"]]
        if not stack:
            continue
        with np.errstate(invalid="ignore"):
            m = np.nanmean(np.vstack(stack), axis=0)
        out["profiles"][ch] = m
        a1, p1 = harmonic(m, 1.0)
        a2, p2 = harmonic(m, 2.0)
        out["amplitude"][ch] = {"amp_1cps": a1, "peak_1cps_pct": p1,
                                "amp_2cps": a2, "peak_2cps_pct": p2}
    out["coupling"] = coupling(out["amplitude"])
    return out


def _wrap_pct(d: float) -> float:
    """A difference of stride phases wrapped to [-50, 50)."""
    return float((d + 50.0) % 100.0 - 50.0)


def coupling(amplitude: dict) -> dict:
    """Yaw and roll relative to the lateral sway at 1 cycle per stride (module docstring)."""
    lat, yaw, roll = (amplitude.get(k) for k in ("lateral", "yaw", "roll"))
    if not (lat and yaw and roll):
        return {}
    leftmost = lat["peak_1cps_pct"] + 50.0      # +x = right, so the trough is leftmost
    return {"yaw_lag_pct": _wrap_pct(yaw["peak_1cps_pct"] - leftmost),
            "yaw_fixation_m": float(lat["amp_1cps"] / 1000.0
                                    / np.tan(np.radians(yaw["amp_1cps"]))),
            "roll_lead_pct": _wrap_pct(roll["peak_1cps_pct"] - leftmost)}


def _median_range(vals) -> dict:
    """Median, range and count of the finite values in ``vals``."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not v.size:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": round(float(np.median(v)), 3), "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3), "n": int(v.size)}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _kinematics_figure(subs: dict, summary: dict, out: Path) -> None:
    """Figure: stride-phase profile of each channel, participants thin, mean bold."""
    x = (np.arange(N_BINS) + 0.5) * 100.0 / N_BINS
    fig, axes = plt.subplots(2, 3, figsize=(S.COL2_IN * 1.55, 6.4), sharex=True)
    for ax, ch, tag in zip(axes.ravel(), CHANNELS, "abcdef"):
        rows = [s["profiles"][ch] for s in subs.values() if ch in s["profiles"]]
        if not rows:
            ax.set_visible(False)
            continue
        arr = np.vstack(rows)
        for r in arr:
            ax.plot(x, r, color=S.MIST, lw=0.9)
        m = np.nanmean(arr, 0)
        se = np.nanstd(arr, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(arr), 0).clip(1))
        ax.fill_between(x, m - se, m + se, color=S.BLUE, alpha=0.22, lw=0)
        ax.plot(x, m, color=S.BLUE, lw=2.0)
        ax.axvspan(0, 50, color=S.GRID, zorder=-1, lw=0)
        for xs in (0, 50, 100):
            ax.axvline(xs, color=S.GREY, ls=":", lw=0.8)
        a = summary[ch]
        ax.text(0.98, 0.04, f"1 cps {a['amp_1cps']['median']:.2g} {UNIT[ch]}\n"
                            f"2 cps {a['amp_2cps']['median']:.2g} {UNIT[ch]}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=S.INK)
        ax.set_ylabel(f"{TITLE[ch].split(' ')[0].lower()} ({UNIT[ch]})")
        S.panel(ax, tag, TITLE[ch], fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("% of stride (0, 50 = double support)")
    fig.suptitle(f"Head motion over the stride, n = {len(subs)} participants",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Band-passed 0.3-5 Hz, 50 phase bins; grey, participant means; blue, mean "
                   f"± SEM. Shaded half: {gait_mod.foot_name(0)} step, unshaded: "
                   f"{gait_mod.foot_name(1)}. Text: median amplitude of the participant means "
                   "at 1 and 2 cycles per stride. Positive: up, right, forward; roll right "
                   "ear up, pitch nose down, yaw turning right (rotation relative to the "
                   "mean pose; signs measured by the rig check).")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    S.save(fig, out)


# ---------------------------------------------------------------------------
def _json_safe(o):
    """JSON-serialisable copy of ``o``; non-finite floats become None."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return [None if not np.isfinite(v) else round(float(v), 4) for v in o.ravel()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def kinematics_main() -> dict | None:
    """Head kinematics over the stride per participant, and the group figure."""
    print("=== Head kinematics over the stride (EXPLORATORY) ===")
    by_sub: dict[str, list[dict]] = {}
    for rec in bids_io.find_recordings(scope="gait"):
        b = block_profiles(rec)
        if b is not None:
            by_sub.setdefault(rec.sub, []).append(b)
    bids_io.gate_report("head_kinematics", "gait", condition="walking",
                        n_analysed=len(by_sub))
    if not by_sub:
        print("No walking blocks with gait, foot labels and head pose. Run gait.py first.")
        return None
    subs = {s: participant_summary(bl) for s, bl in sorted(by_sub.items())}
    summary = {ch: {f"amp_{f}cps": _median_range([s["amplitude"][ch][f"amp_{f}cps"]
                                                  for s in subs.values() if ch in s["amplitude"]])
                    for f in (1, 2)} for ch in CHANNELS}
    cpl = {k: [s["coupling"][k] for s in subs.values() if k in s["coupling"]]
           for k in ("yaw_lag_pct", "yaw_fixation_m", "roll_lead_pct")}
    coupling_summary = {k: _median_range(v) for k, v in cpl.items()}
    coupling_summary["n_yaw_lag_positive"] = int(sum(v > 0 for v in cpl["yaw_lag_pct"]))
    coupling_summary["n_yaw_towards_midline"] = int(sum(abs(v) < 25 for v in cpl["yaw_lag_pct"]))
    coupling_summary["n_roll_leads"] = int(sum(v < 0 for v in cpl["roll_lead_pct"]))
    for s, d in subs.items():
        a = d["amplitude"]
        print(f"  {s}: {d['n_blocks']} blocks | vertical 2 cps {a['vertical']['amp_2cps']:.1f} mm, "
              f"lateral 1 cps {a['lateral']['amp_1cps']:.1f} mm, roll 1 cps "
              f"{a['roll']['amp_1cps']:.2f} / 2 cps {a['roll']['amp_2cps']:.2f} deg, "
              f"pitch 2 cps {a['pitch']['amp_2cps']:.2f} deg, yaw 1 cps {a['yaw']['amp_1cps']:.2f} deg")
    out = bids_io.deriv_dir("head_kinematics")
    payload = {
        "note": "EXPLORATORY, descriptive. Band-passed 0.3-5 Hz; amplitude of each "
                "participant's mean stride profile. Rotation relative to the mean pose; "
                "positive roll = right ear up, pitch = nose down, yaw = turning right "
                "(rig check).",
        "roll_sign_verified": bool(getattr(C, "HEAD_ROLL_SIGN_VERIFIED", False)),
        "n_participants": len(subs),
        "summary": summary,
        "coupling": coupling_summary,
        "participants": {s: {"n_blocks": d["n_blocks"], "amplitude": d["amplitude"],
                             "coupling": d["coupling"]}
                         for s, d in subs.items()},
    }
    (out / "head_kinematics.json").write_text(json.dumps(_json_safe(payload), indent=2),
                                              encoding="utf-8")
    _kinematics_figure(subs, summary, bids_io.fig_dir("head_kinematics") / "head_kinematics.png")
    print(f"\n  median roll amplitude: {summary['roll']['amp_1cps']['median']} deg at 1 cps, "
          f"{summary['roll']['amp_2cps']['median']} deg at 2 cps")
    print(f"  yaw after the sway: {coupling_summary['yaw_lag_pct']['median']} % of the stride "
          f"({coupling_summary['n_yaw_lag_positive']}/{len(cpl['yaw_lag_pct'])} positive), "
          f"fits a target at {coupling_summary['yaw_fixation_m']['median']} m; roll leads the "
          f"sway by {coupling_summary['roll_lead_pct']['median']} % "
          f"({coupling_summary['n_roll_leads']}/{len(cpl['roll_lead_pct'])})")
    print(f"Head kinematics -> {out}")
    return payload


# ###########################################################################
# Pipeline stage
# ###########################################################################
def main() -> None:
    """Pipeline stage ``head_motion``: stabilisation, then kinematics."""
    stability_main()
    kinematics_main()


if __name__ == "__main__":
    main()
