"""Rig check: the sign conventions of the logged head pose, measured on a known movement script.

Which way is right, which way is forward, and which rotation is a nod, an ear-to-shoulder
tilt or a head turn are properties of how the rig is mounted and how Motive streams, not
of the data. This module reads them off two recordings of a scripted performer
(``rigcheck/``, participant 101, never part of the dataset) and reports each convention
the analysis relies on:

* ``config.LATERAL_POSITIVE_IS_RIGHT``: a whole-body shift to the right, standing on the
  right leg, and the first single support of a walk started with the right foot all move
  ``head_local_pos_x`` the same way;
* forward: a step forward moves ``head_local_pos_z`` (and the raw Motive position);
* head rotation, as the analysis computes it (``head_motion.rotation_on_grid``: the
  rotation from the mean pose, as a rotation vector of the logged Motive quaternion):
  pitch about x as ``head_motion.pitch_axis`` orients it, roll and yaw about
  ``head_motion``'s axes. A chin-down nod, the right ear to the right shoulder and a
  head turn to the right must come out positive pitch, negative roll and positive yaw.
  The raw rotation-vector components are kept in the JSON as the record of what Motive
  delivers.

Standing script (``p101-v1-b0-t01``), each segment followed by ~10 s still: baseline;
shift right x3; shift left x3; step forward (right foot, then left); stand on the right
leg x3; on the left leg x3; nod (chin down) x3; right ear to shoulder x3; left ear x3;
turn right x3; turn left x3; still. Times were not noted. The 29 held movements are found
on the pose itself and assigned to the segments in order by count; each segment must have
moved its own channel in every movement, with one sign, before that sign is read.

Walking (``p101-v1-b1-t01``): the belt starts after a still baseline and the performer
steps off with the RIGHT foot, so the first single support is on the LEFT foot. The
first sideways head movement after the belt starts gives the side of that stance; the
gait decoder's labels (``gait.extract_gait``) are compared with it.

Frames in which Motive solved the head rigid body onto something else (the head
position jumps to floor height with an unrelated orientation) are removed before any
rotation is read: a frame further than ``SPIKE_DEG`` from the running median pose.

Outputs (``rigcheck/``): ``rig_check.json`` (per segment: channel, movement count, signed
peaks, verdict; walking start; the conventions), ``figures/rig_check.png``.

Usage::

    python -m Analysis.rig_check
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

try:
    from . import bids_io
    from . import bidsify
    from . import clean
    from . import config as C
    from . import gait as gait_mod
    from . import head_motion
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import bidsify
    import clean
    import config as C
    import gait as gait_mod
    import head_motion
    import plotstyle as S

S.apply()

RIG_DIR = C.PROJECT_ROOT / "rigcheck"
GRID_FS = 30.0              # analysis grid (Hz); the movements are ~1 s long
SPIKE_DEG = 30.0            # frame this far from the running median pose = another body
SMOOTH_S = 0.3              # median smoothing of every channel on the grid
STILL_S = 10.0              # the baseline: first stretch this long without movement
MOVE_MIN_X = 2.0            # a held movement reaches at least this many thresholds ...
MOVE_MIN_S = 1.5            # ... and lasts this long (the holds were 2-5 s)
MERGE_GAP_S = 0.8           # parts of one movement are closer than this (repeats: >1.3 s)
# movement thresholds per channel, and the unit each is reported in
THRESH = {"x": 0.04, "z": 0.08, "pitch": 8.0, "roll": 8.0, "yaw": 12.0}
UNIT = {"x": "m", "z": "m", "pitch": "deg", "roll": "deg", "yaw": "deg"}
RAW = {"pitch": "rv_x", "roll": "rv_z", "yaw": "rv_y"}   # Motive rotation-vector component
SHORT = {"shift_right": "shift R", "shift_left": "shift L", "step_forward": "step forward",
         "stand_right_leg": "R leg", "stand_left_leg": "L leg", "nod_chin_down": "chin down",
         "right_ear_to_shoulder": "R ear down", "left_ear_to_shoulder": "L ear down",
         "turn_right": "turn R", "turn_left": "turn L"}
# The script: (segment, channel it must move, movements, anatomical direction)
SCRIPT = [
    ("shift_right", "x", 3, "right"), ("shift_left", "x", 3, "left"),
    ("step_forward", "z", 2, "forward"),
    ("stand_right_leg", "x", 3, "right"), ("stand_left_leg", "x", 3, "left"),
    ("nod_chin_down", "pitch", 3, "nose down"),
    ("right_ear_to_shoulder", "roll", 3, "right ear down"),
    ("left_ear_to_shoulder", "roll", 3, "left ear down"),
    ("turn_right", "yaw", 3, "turn right"), ("turn_left", "yaw", 3, "turn left"),
]


# ---------------------------------------------------------------------------
# Recordings: the rig logs converted exactly as the dataset's are
# ---------------------------------------------------------------------------
def convert(rig_dir: Path, work: Path) -> dict:
    """Convert the rig logs into a throwaway BIDS dataset under ``work`` and return
    ``{condition: Recording}``. ``config``'s dataset paths point there only meanwhile."""
    raw, bids = work / "raw", work / "bids"
    raw.mkdir(parents=True)
    for f in sorted(rig_dir.glob("p*-v*-b*-t*_*.csv*")):
        shutil.copy2(f, raw / f.name)
    saved = (C.BIDS_DIR, C.RAW_DIR, C.SOURCEDATA_DIR, C.DERIV_DIR)
    try:
        C.BIDS_DIR, C.RAW_DIR = bids, raw
        C.SOURCEDATA_DIR, C.DERIV_DIR = bids / "sourcedata", bids / "derivatives"
        bidsify.convert_all(raw_dir=raw, bids_dir=bids)
        recs = bids_io.find_recordings(bids, include_excluded=True)
    finally:
        C.BIDS_DIR, C.RAW_DIR, C.SOURCEDATA_DIR, C.DERIV_DIR = saved
    return {r.condition: r for r in recs}


def pose_on_grid(motion) -> dict:
    """Head position (m) and rotation (deg) on a uniform grid, without foreign frames.

    Rotation is the analysis's own: the rotation from the mean pose of the clean frames,
    as a rotation vector in the frame of the logged quaternion (``rotation_on_grid``).
    """
    t = motion["t"].to_numpy(float)
    t = t - t[0]
    tg = np.arange(0.0, t[-1], 1.0 / GRID_FS)
    stale, _ = clean.stale_frames(motion)
    out = {"t": tg}
    for name, col in (("x", "head_local_pos_x"), ("z", "head_local_pos_z"),
                      ("mocap_z", "head_mocap_pos_z")):
        v, *_ = clean.clean_channel(t, motion[col].to_numpy(float),
                                    abs_radius_m=clean.HEAD_CLEAN_AXES.get(col, 0.5))
        v = np.asarray(v, float).copy()
        if stale is not None:
            v[stale] = np.nan
        ok = np.isfinite(v)
        out[name] = np.interp(tg, t[ok], v[ok])
    ok = head_motion._good_pose_frames(motion)
    if stale is not None:
        ok &= ~stale
    rot = Rotation.from_quat(motion.loc[ok, head_motion.QUAT].to_numpy(float))
    rv = np.degrees(rot.as_rotvec())
    med = median_filter(rv, size=(31, 1), mode="nearest")
    keep = np.linalg.norm(rv - med, axis=1) < SPIKE_DEG
    r0 = rot[keep].mean()
    rel = np.degrees((rot[keep] * r0.inv()).as_rotvec())
    tk = t[ok][keep]
    for j, name in enumerate(("rv_x", "rv_y", "rv_z")):
        out[name] = np.interp(tg, tk, rel[:, j])
    out["foreign_frame_pct"] = round(float(100 * (1 - keep.mean())), 2)
    k = max(1, int(SMOOTH_S * GRID_FS))
    for name in ("x", "z", "mocap_z", "rv_x", "rv_y", "rv_z"):
        out[name] = median_filter(out[name], size=k, mode="nearest")
    # the analysis's channels: pitch about x oriented as head_motion.pitch_axis orients
    # it, roll and yaw about head_motion's axes
    rv = np.column_stack([out["rv_x"], out["rv_y"], out["rv_z"]])
    out["pitch"] = rv @ head_motion.pitch_axis_orientation(np.array([1.0, 0.0, 0.0]))
    out["roll"], out["yaw"] = rv @ head_motion.ROLL_AXIS, rv @ head_motion.YAW_AXIS
    return out


# ---------------------------------------------------------------------------
# Standing: segments of the script
# ---------------------------------------------------------------------------
def _activity(p: dict, ref: dict) -> np.ndarray:
    """Largest movement across channels, in units of each channel's threshold."""
    return np.max([np.abs(p[c] - ref[c]) / THRESH[c] for c in THRESH], axis=0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end indices of each run of True in ``mask``."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def movements(p: dict) -> list[tuple[int, int]]:
    """Grid spans of the script's movements, in order.

    A movement is a stretch in which one channel leaves the baseline pose by more than its
    threshold in one direction, so going straight from +x to -x (one leg to the other) is
    two. It counts when it is a held movement, not a brush past the threshold (the
    overshoot on returning to centre): at least ``MOVE_MIN_X`` thresholds and
    ``MOVE_MIN_S`` long. A movement also shifts other channels (an ear-to-shoulder tilt
    moves the head sideways; a step forward is a weight shift onto the other leg, the step,
    and the shift again on the way back), so held deviations closer than ``MERGE_GAP_S``
    are one movement, unless the same channel reverses (one leg straight to the other).
    The script starts with the first still stretch of ``STILL_S``; before it the
    performer was getting ready.
    """
    ref = {c: float(np.median(p[c])) for c in THRESH}
    quiet = [(a, b) for a, b in _runs(_activity(p, ref) <= 1.0)
             if (b - a) / GRID_FS >= STILL_S]
    start = quiet[0][0] if quiet else 0
    if quiet:
        ref = {c: float(np.median(p[c][quiet[0][0]:quiet[0][1]])) for c in THRESH}
    cands = []
    for c in THRESH:
        z = (p[c] - ref[c]) / THRESH[c]
        for s in (1, -1):
            for a, b in _runs(s * z > 1.0):
                if a >= start and (s * z[a:b]).max() >= MOVE_MIN_X and (b - a) / GRID_FS >= MOVE_MIN_S:
                    cands.append((a, b, c, s))
    gap = int(MERGE_GAP_S * GRID_FS)
    merged: list[list] = []                       # [start, end, {channel: sign}]
    for a, b, c, s in sorted(cands):
        last = merged[-1] if merged else None
        if last and a < last[1] + gap and last[2].get(c, s) == s:
            last[1] = max(last[1], b)
            last[2][c] = s
        else:
            merged.append([a, b, {c: s}])
    return [(a, b) for a, b, _ in merged]


def standing_segments(p: dict) -> list[dict]:
    """The script's ten segments, each checked against its channel and signed.

    The movements are assigned to the segments in order by count (3, 3, 2, 3, ...), so the
    count must come out exactly; each segment must then have moved the channel it is
    meant to move in every one of its movements.
    """
    tg = p["t"]
    moves = movements(p)
    n_script = sum(n for _, _, n, _ in SCRIPT)
    if len(moves) != n_script:
        raise RuntimeError(f"{len(moves)} movements found, the script has {n_script}: "
                           + ", ".join(f"{tg[a]:.0f}-{tg[b - 1]:.0f}s" for a, b in moves))
    rows, i = [], 0
    for name, chan, n_exp, direction in SCRIPT:
        seg = moves[i:i + n_exp]
        i += n_exp
        a0 = seg[0][0]
        pre = slice(max(0, a0 - int(3 * GRID_FS)), a0)            # the pause before it
        base = {c: float(np.median(p[c][pre])) for c in THRESH}
        peaks, moved, dominant, mocap_z, raw = [], [], [], [], []
        for a, b in seg:
            dev = {c: p[c][a:b] - base[c] for c in THRESH}
            k = int(np.argmax(np.abs(dev[chan])))
            peaks.append(round(float(dev[chan][k]), 3))
            if RAW.get(chan) in p:
                rc = p[RAW[chan]]
                raw.append(round(float(rc[a + k] - np.median(rc[pre])), 1))
            moved.append(abs(dev[chan][k]) >= MOVE_MIN_X * THRESH[chan])
            dominant.append(max(THRESH, key=lambda c: np.max(np.abs(dev[c])) / THRESH[c]))
            mocap_z.append(round(float(p["mocap_z"][a + k] - np.median(p["mocap_z"][pre])), 3))
        sign = int(np.sign(np.median(peaks)))
        rows.append({
            "segment": name, "direction": direction, "channel": chan, "unit": UNIT[chan],
            "start_s": round(float(tg[a0]), 1), "end_s": round(float(tg[seg[-1][1] - 1]), 1),
            "n_movements": n_exp, "peaks": peaks, "sign": sign,
            "dominant_channel": dominant,
            "same_sign_all": all(np.sign(peaks) == sign),
            "channel_moved_every_time": all(moved),
            "mocap_z_peak_m": mocap_z if chan == "z" else None,
            "raw_rotvec": ({"component": RAW[chan], "peaks_deg": raw} if raw else None),
        })
    return rows


# ---------------------------------------------------------------------------
# Walking: the first single support against the decoder's labels
# ---------------------------------------------------------------------------
def walking_start(rec, motion) -> dict:
    """Lateral sway of the first single support after the belt starts, and the
    stance feet the gait stage decodes for the first steps."""
    t = motion["t"].to_numpy(float)
    t0 = t[0]
    spd = motion["treadmill_speed"].to_numpy(float)
    on = float(t[np.flatnonzero(spd > 0.05)[0]])
    x, *_ = clean.clean_channel(t, motion["head_local_pos_x"].to_numpy(float),
                                abs_radius_m=clean.HEAD_CLEAN_AXES["head_local_pos_x"])
    pre = (t > on - 5) & (t < on)
    dx = x - np.nanmedian(x[pre])
    after = np.flatnonzero((t >= on) & (np.abs(dx) > 0.02))
    first = int(after[0])
    # the extreme of that first excursion, before it comes back through zero
    back = np.flatnonzero((t > t[first]) & (np.sign(dx) != np.sign(dx[first])))
    end = int(back[0]) if back.size else first + 1
    k = first + int(np.nanargmax(np.abs(dx[first:end])))
    res = gait_mod.extract_gait(rec)
    tr, foot, mg = res.trough_times, res.step_foot, res.step_margin
    # where the decoded foot's sway extreme falls in its step (single support is mid-step)
    peak_pct = []
    for i in range(len(foot)):
        if tr[i + 1] - tr[i] > 1.5 * np.median(np.diff(tr)):
            continue
        w = (res.t_grid >= tr[i]) & (res.t_grid < tr[i + 1])
        j = np.argmax((1 if foot[i] == 0 else -1) * res.x_filt[w])
        peak_pct.append(100 * (res.t_grid[w][j] - tr[i]) / (tr[i + 1] - tr[i]))
    return {
        "sway_extreme_pct_of_step": [round(float(v), 1) for v in np.percentile(peak_pct, [25, 50, 75])],
        "belt_start_s": round(on - t0, 2),
        "first_sway_s": round(float(t[k] - t0), 2),
        "first_sway_m": round(float(dx[k]), 3),
        "first_stance_foot": "left",            # the walk was started with the right foot
        "decoded_steps": [{"start_s": round(float(tr[i] - t0), 2),
                           "end_s": round(float(tr[i + 1] - t0), 2),
                           "foot_code": int(foot[i]), "margin": round(float(mg[i]), 2)}
                          for i in range(min(6, len(foot)))],
        "foot_code_0_sway_sign": +1,            # gait: code 0 = the positive-sway foot
        "n_steps": int(res.n_steps), "lr_alternation_pct": res.lr_alternation_pct,
        "foot_label_verdict": res.foot_label_verdict,
        "data_gap_total_s": res.data_gap_total_s,
        "_x": (t - t0, dx), "_on": on - t0, "_troughs": tr - t0,
        "_filt": (res.t_grid - t0, res.x_filt, res.y_filt),   # what the decoder sees
        "_foot": np.asarray(foot),
    }


def walking_pitch_axis(rec, motion) -> list[float]:
    """The pitch axis ``head_motion`` would pick for this walk (room frame)."""
    res = gait_mod.extract_gait(rec)
    valid = np.isfinite(res.stride_phase_cont)
    rot = head_motion.rotation_on_grid(motion, res.t_grid, valid)
    fs = 1.0 / float(np.median(np.diff(res.t_grid)))
    step_hz = 1.0 / float(np.median(np.diff(res.trough_times)))
    rv = rot["rotvec"].copy()
    rv[~np.all(np.isfinite(rv), axis=1)] = 0.0
    axis, _ = head_motion.pitch_axis(rv, fs, step_hz)
    return [round(float(v), 3) for v in axis]


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
def conventions(rows: list[dict], walk: dict) -> dict:
    """Sign conventions implied by the segments and the walk start, each with
    whether every repetition agreed."""
    s = {r["segment"]: r for r in rows}
    lateral = [s["shift_right"]["sign"], -s["shift_left"]["sign"],
               s["stand_right_leg"]["sign"], -s["stand_left_leg"]["sign"],
               -int(np.sign(walk["first_sway_m"]))]       # first stance = LEFT
    roll = [-s["right_ear_to_shoulder"]["sign"], s["left_ear_to_shoulder"]["sign"]]
    yaw = [s["turn_right"]["sign"], -s["turn_left"]["sign"]]
    first_code = walk["decoded_steps"][0]["foot_code"] if walk["decoded_steps"] else None
    return {
        "lateral_positive_is_right": all(v == 1 for v in lateral),
        "lateral_evidence": lateral,
        "forward_is_positive_local_z": s["step_forward"]["sign"] == 1,
        "mocap_z_points_backward": all(v < 0 for v in s["step_forward"]["mocap_z_peak_m"]),
        "analysis_pitch_positive_is_nose_down": s["nod_chin_down"]["sign"] == 1,
        "analysis_roll_positive_is_right_ear_up": all(v == 1 for v in roll),
        "analysis_yaw_positive_is_turn_right": all(v == 1 for v in yaw),
        "first_decoded_step_foot_code": first_code,
        "all_segments_match_script": all(r["channel_moved_every_time"] and r["same_sign_all"]
                                         for r in rows),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _figure(p: dict, rows: list[dict], walk: dict, out: Path) -> None:
    """Figure: the pose channels over the standing script with each segment shaded,
    and the start of the walk with its decoded steps."""
    chans = [("x", "lateral x (cm)", 100), ("z", "forward z (cm)", 100),
             ("pitch", "pitch (deg)", 1), ("roll", "roll (deg)", 1), ("yaw", "yaw (deg)", 1)]
    fig = plt.figure(figsize=(16, 13))
    rows_gs = fig.add_gridspec(2, 1, height_ratios=[len(chans), 1.3], hspace=0.22)
    gs = rows_gs[0].subgridspec(len(chans), 1, hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(len(chans))]
    colour = {"x": S.BLUE, "z": S.GREEN, "pitch": S.ORANGE, "roll": S.PURPLE, "yaw": S.TEAL}
    for ax, (c, lab, k) in zip(axes, chans):
        ax.plot(p["t"], k * (p[c] - np.median(p[c])), color=S.INK, lw=0.8)
        for r in rows:
            if r["channel"] == c:
                ax.axvspan(r["start_s"], r["end_s"], color=colour[c], alpha=0.15, lw=0)
                ax.text(0.5 * (r["start_s"] + r["end_s"]), 0.97,
                        SHORT[r["segment"]], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8, color=S.INK)
        ax.axhline(0, color=S.GREY, lw=0.6)
        ax.set_ylabel(lab)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("time in the standing recording (s)")
    S.panel(axes[0], "a", "Standing script: each segment shaded on the channel it must move",
            fontsize=10)
    ax = fig.add_subplot(rows_gs[1])
    t, dx = walk["_x"]
    w = (t > walk["_on"] - 3) & (t < walk["_on"] + 8)
    ax.plot(t[w], 100 * dx[w], color=S.MIST, lw=1.0, label="lateral, raw")
    tf, xf, yf = walk["_filt"]
    wf = tf < walk["_on"] + 8
    ax.plot(tf[wf], 100 * xf[wf], color=S.INK, lw=1.3, label="lateral, gait band (decoded)")
    ax.plot(tf[wf], 500 * yf[wf], color=S.GREEN, lw=1.0,
            label="vertical ×5, gait band (troughs = double support)")
    ax.axvline(walk["_on"], color=S.RED, lw=1.0, ls="--", label="belt starts")
    ax.scatter([walk["first_sway_s"]], [100 * walk["first_sway_m"]], color=S.RED, zorder=5,
               label="first single support (left foot)")
    tr = walk["_troughs"]
    for i, f in enumerate(walk["_foot"]):
        if tr[i] < walk["_on"] + 8:
            ax.axvspan(tr[i], min(tr[i + 1], walk["_on"] + 8),
                       color=gait_mod.FOOT_COLOUR[int(f)], alpha=0.16, lw=0)
    for tr in walk["_troughs"][(walk["_troughs"] > walk["_on"] - 3)
                               & (walk["_troughs"] < walk["_on"] + 8)]:
        ax.axvline(tr, color=S.GREY, ls=":", lw=0.7)
    ax.axhline(0, color=S.GREY, lw=0.6)
    ax.set_xlabel("time in the walking recording (s)")
    ax.set_ylabel("cm")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(facecolor=gait_mod.FOOT_COLOUR[k], alpha=0.35,
                      label=f"decoded stance: {gait_mod.foot_name(k)}") for k in (0, 1)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              frameon=False)
    S.panel(ax, "b", "Walk started with the right foot: first sway and the decoded steps, "
                     "each from one double support to the next", fontsize=10)
    S.caption(fig, "Rig check with a scripted performer. Rotations are the analysis's own "
                   "(rotation vector of the logged Motive quaternion relative to the mean "
                   "pose); foreign rigid-body frames removed. Signs are read relative to the "
                   "pause before each segment.", wrap=170, y=0.02)
    fig.subplots_adjust(top=0.96, bottom=0.07, left=0.06, right=0.82)
    S.save(fig, out)


def _json_safe(o):
    """JSON-serialisable copy of ``o``, without private keys."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(rig_dir: Path = RIG_DIR) -> dict:
    """Convert the rig-check recordings, measure every convention and write
    ``rig_check.json`` and the figure."""
    with tempfile.TemporaryDirectory() as tmp:
        recs = convert(rig_dir, Path(tmp))
        m_stand, _, _ = bids_io.load_motion(recs["standing"])
        m_walk, _, _ = bids_io.load_motion(recs["walking"])
        p = pose_on_grid(m_stand)
        rows = standing_segments(p)
        walk = walking_start(recs["walking"], m_walk)
        walk["pitch_axis_walking"] = walking_pitch_axis(recs["walking"], m_walk)
    conv = conventions(rows, walk)
    out = {"recordings": sorted(f.name for f in rig_dir.glob("p*-v*-b*-t*_*.csv*")),
           "foreign_frame_pct_standing": p["foreign_frame_pct"],
           "segments": rows, "walking": walk, "conventions": conv}
    (rig_dir / "figures").mkdir(parents=True, exist_ok=True)
    _figure(p, rows, walk, rig_dir / "figures" / "rig_check.png")
    (rig_dir / "rig_check.json").write_text(json.dumps(_json_safe(out), indent=2),
                                            encoding="utf-8")
    print("\n=== Rig check ===")
    for r in rows:
        ok = r["channel_moved_every_time"] and r["same_sign_all"]
        print(f"  {r['segment']:<22} {r['start_s']:6.1f}-{r['end_s']:6.1f}s  "
              f"{r['channel']:<5} peaks {r['peaks']} {r['unit']}  (dominant "
              f"{'/'.join(r['dominant_channel'])}){'' if ok else '  <-- CHECK'}")
    print(f"  walking: belt at {walk['belt_start_s']} s, first sway "
          f"{100 * walk['first_sway_m']:+.1f} cm at {walk['first_sway_s']} s; decoded "
          f"steps {[s['foot_code'] for s in walk['decoded_steps']]}; walking pitch axis "
          f"{walk['pitch_axis_walking']}")
    for k, v in conv.items():
        print(f"  {k}: {v}")
    print(f"Outputs -> {rig_dir}")
    return out


if __name__ == "__main__":
    main()
