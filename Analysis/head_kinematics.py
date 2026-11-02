"""Exploratory: head kinematics over the stride, by stance foot.

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
``head_stability.rotation_on_grid``) and projected onto room axes: roll about +z
(positive = right ear up), yaw about -y (positive = turning right), and pitch about the
horizontal axis with most power at the step rate (``head_stability.pitch_axis``, close to
x; positive = nose down). The mocap frame is the same in every session, and the rig check
measured each sign (``rig_check.py``, ``config``).

Roll is the channel that matters most for the task. The Gabor plane follows the tracked
head POSITION but not its rotation (``CameraHolder`` tracks position only), so the
grating stays upright in the room and a head roll of r degrees rotates it on the retina
by r, less whatever ocular counter-roll compensates. ``roll_series`` is the roll this
module computes, shared with ``roll_bias.py``.

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

Usage::

    python -m Analysis.head_kinematics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

try:
    from . import bids_io
    from . import clean
    from . import config as C
    from . import gait as gait_mod
    from . import head_stability as hs
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import clean
    import config as C
    import gait as gait_mod
    import head_stability as hs
    import phase_analysis as pa
    import plotstyle as S

S.apply()

N_BINS = 50
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
    rot = hs.rotation_on_grid(motion, tg, _valid_grid(g))
    if rot is None:
        return None
    rv = rot["rotvec"]
    fs = 1.0 / float(np.median(np.diff(tg)))
    step_hz = 1.0 / float(np.median(np.diff(g["troughs"])))
    pitch_ax, _ = hs.pitch_axis(rv, fs, step_hz)
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
    ``rotation_channels``). Shared with ``roll_bias.py``."""
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
    prof = {k: hs._bin_by_phase(phase, _band(v, fs), N_BINS)[0] for k, v in sig.items()}
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
def _figure(subs: dict, summary: dict, out: Path) -> None:
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


def main() -> dict | None:
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
    _figure(subs, summary, bids_io.fig_dir("head_kinematics") / "head_kinematics.png")
    print(f"\n  median roll amplitude: {summary['roll']['amp_1cps']['median']} deg at 1 cps, "
          f"{summary['roll']['amp_2cps']['median']} deg at 2 cps")
    print(f"  yaw after the sway: {coupling_summary['yaw_lag_pct']['median']} % of the stride "
          f"({coupling_summary['n_yaw_lag_positive']}/{len(cpl['yaw_lag_pct'])} positive), "
          f"fits a target at {coupling_summary['yaw_fixation_m']['median']} m; roll leads the "
          f"sway by {coupling_summary['roll_lead_pct']['median']} % "
          f"({coupling_summary['n_roll_leads']}/{len(cpl['roll_lead_pct'])})")
    print(f"Head kinematics -> {out}")
    return payload


if __name__ == "__main__":
    main()
