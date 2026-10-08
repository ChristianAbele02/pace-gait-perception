"""Convert raw Unity per-frame CSVs into a Motion-BIDS compliant dataset.

The Unity logger (``ILLLoggerController``) writes one semicolon-separated row per frame,
combining continuous head tracking with trial columns that are filled only on the frame a
trial ends. BIDS keeps these separate.

Output layout (under ``data/``; details in the generated ``data/README.md``)::

    sourcedata/raw/                  p01-v1-b1-t01_walking.csv.gz      (original, untouched)
    sourcedata/sub-01/ses-02/motion/ sub-01_ses-02_task-ILL_tracksys-optitrack_acq-walking_run-01_motion.csv.gz
    sub-01/sub-01_sessions.tsv, sub-01_scans.tsv
    sub-01/ses-01/beh/    sub-01_ses-01_task-ILL_acq-walking_run-01_beh.tsv  (+ .json)  QUEST+ trials
    sub-01/ses-02/motion/ sub-01_ses-02_task-ILL_tracksys-optitrack_acq-walking_run-01_motion.tsv
                          (+ _channels.tsv/.json, _motion.json, _events.tsv/.json)

Each raw log is archived gzipped and BIDS-named under ``sourcedata/`` (~6x smaller).
After conversion the original is moved unchanged into ``sourcedata/raw/``
(``prune_loose=True`` deletes it instead). ``.csv.gz`` logs are copied as they are; plain
``.csv`` is gzipped. Later runs read from the ``sourcedata`` archives.

Sampling: Unity logs are frame-timed (nominal 60 Hz, uneven spacing). The ``t`` channel
holds the true sample time, zeroed at the first sample like all event onsets.
``SamplingFrequency`` is nominal, ``SamplingFrequencyEffective`` the median actual rate.

Usage::

    python -m Analysis.bidsify              # convert every raw CSV (loose or archived)
    python Analysis/bidsify.py p01-b1-t01-1.csv
"""
from __future__ import annotations

import filecmp
import gzip
import io
import json
import re
import shutil
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import bids_io, config as C, stage_cache
except ImportError:  # run as a script, not a module
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import stage_cache

BIDS_VERSION = "1.11.1"

# ---------------------------------------------------------------------------
# Channel definitions: raw column to Motion-BIDS channel.
# (raw_col, bids_name, tracked_point, ch_type, component, units, reference_frame)
# ---------------------------------------------------------------------------
_CHANNELS = [
    ("timestamp",            "t",                  "n/a",  "LATENCY", "n/a", "s",   "n/a"),
    ("head-global-x",        "head_global_pos_x",  "Head", "POS",     "x",   "m",   "global"),
    ("head-global-y",        "head_global_pos_y",  "Head", "POS",     "y",   "m",   "global"),
    ("head-global-z",        "head_global_pos_z",  "Head", "POS",     "z",   "m",   "global"),
    ("head-local-x",         "head_local_pos_x",   "Head", "POS",     "x",   "m",   "local"),
    ("head-local-y",         "head_local_pos_y",   "Head", "POS",     "y",   "m",   "local"),
    ("head-local-z",         "head_local_pos_z",   "Head", "POS",     "z",   "m",   "local"),
    ("head-global-rot-x",    "head_global_ori_x",  "Head", "ORNT",    "x",   "deg", "global"),
    ("head-global-rot-y",    "head_global_ori_y",  "Head", "ORNT",    "y",   "deg", "global"),
    ("head-global-rot-z",    "head_global_ori_z",  "Head", "ORNT",    "z",   "deg", "global"),
    ("head-local-rot-x",     "head_local_ori_x",   "Head", "ORNT",    "x",   "deg", "local"),
    ("head-local-rot-y",     "head_local_ori_y",   "Head", "ORNT",    "y",   "deg", "local"),
    ("head-local-rot-z",     "head_local_ori_z",   "Head", "ORNT",    "z",   "deg", "local"),
    # 6-DOF mocap pose from the streaming client (a2/a3), independent of the Unity head
    # Transform.
    ("head-mocap-x",         "head_mocap_pos_x",   "Head", "POS",     "x",   "m",   "mocap"),
    ("head-mocap-y",         "head_mocap_pos_y",   "Head", "POS",     "y",   "m",   "mocap"),
    ("head-mocap-z",         "head_mocap_pos_z",   "Head", "POS",     "z",   "m",   "mocap"),
    ("head-mocap-qx",        "head_mocap_quat_x",  "Head", "ORNT",    "quat_x", "n/a", "mocap"),
    ("head-mocap-qy",        "head_mocap_quat_y",  "Head", "ORNT",    "quat_y", "n/a", "mocap"),
    ("head-mocap-qz",        "head_mocap_quat_z",  "Head", "ORNT",    "quat_z", "n/a", "mocap"),
    ("head-mocap-qw",        "head_mocap_quat_w",  "Head", "ORNT",    "quat_w", "n/a", "mocap"),
    ("head-mocap-rot-x",     "head_mocap_ori_x",   "Head", "ORNT",    "x",   "deg", "mocap"),
    ("head-mocap-rot-y",     "head_mocap_ori_y",   "Head", "ORNT",    "y",   "deg", "mocap"),
    ("head-mocap-rot-z",     "head_mocap_ori_z",   "Head", "ORNT",    "z",   "deg", "mocap"),
    ("head-mocap-valid",     "head_mocap_valid",   "Head", "MISC",    "n/a", "n/a", "mocap"),
    ("treadmill-speed",      "treadmill_speed",    "n/a",  "VEL",     "n/a", "m/s", "n/a"),
    ("distance-on-path",     "distance_on_path",   "n/a",  "MISC",    "n/a", "m",   "global"),
    ("task-visible",         "stim_visible",       "n/a",  "MISC",    "n/a", "n/a", "n/a"),
    ("stimulus-onset",       "stimulus_onset",     "n/a",  "MISC",    "n/a", "n/a", "n/a"),
    ("response-onset",       "response_onset",     "n/a",  "MISC",    "n/a", "n/a", "n/a"),
    ("mocap-frame-advanced", "mocap_frame_advanced", "n/a", "MISC",   "n/a", "n/a", "n/a"),
    ("lateral-sway-flag",    "lateral_sway_flag",  "Head", "MISC",    "n/a", "n/a", "local"),
    ("tracking-framenumber", "tracking_frame",     "n/a",  "MISC",    "n/a", "n/a", "n/a"),
    # mocap device clock, not a latency from recording onset, so MISC
    ("tracking-timestamp",   "tracking_time",      "n/a",  "MISC",    "n/a", "s",   "n/a"),
    ("clock-hires",          "clock_hires",        "n/a",  "LATENCY", "n/a", "s",   "n/a"),
]

_CHANNEL_DESCRIPTIONS = {
    "t": "Time of this sample in seconds from the first sample of the recording (Unity frame "
         "clock). Samples are frame-timed and NOT evenly spaced: always use this channel, "
         "never the row index, for timing. Event onsets use the same zero.",
    "head_global_pos_x": "Head position in Unity world space, X.",
    "head_global_pos_y": "Head position in Unity world space, Y (vertical / up).",
    "head_global_pos_z": "Head position in Unity world space, Z (forward along path).",
    "head_local_pos_x": "Head position relative to the moving CAVE rig, X (lateral sway).",
    "head_local_pos_y": "Head position relative to the moving CAVE rig, Y (vertical bob).",
    "head_local_pos_z": "Head position relative to the moving CAVE rig, Z (fore-aft).",
    "head_global_ori_x": "Head orientation (Euler) world, X / pitch.",
    "head_global_ori_y": "Head orientation (Euler) world, Y / yaw.",
    "head_global_ori_z": "Head orientation (Euler) world, Z / roll.",
    "head_local_ori_x": "Head orientation (Euler) rig-local, X / pitch.",
    "head_local_ori_y": "Head orientation (Euler) rig-local, Y / yaw.",
    "head_local_ori_z": "Head orientation (Euler) rig-local, Z / roll.",
    "head_mocap_pos_x": "Head rigid-body position from the mocap streaming client, X (mocap space).",
    "head_mocap_pos_y": "Head rigid-body position from the mocap streaming client, Y (mocap space).",
    "head_mocap_pos_z": "Head rigid-body position from the mocap streaming client, Z (mocap space).",
    "head_mocap_quat_x": "Head rigid-body orientation quaternion X (unambiguous; from mocap).",
    "head_mocap_quat_y": "Head rigid-body orientation quaternion Y (unambiguous; from mocap).",
    "head_mocap_quat_z": "Head rigid-body orientation quaternion Z (unambiguous; from mocap).",
    "head_mocap_quat_w": "Head rigid-body orientation quaternion W (unambiguous; from mocap).",
    "head_mocap_ori_x": "Head rigid-body orientation (Euler) X / pitch, from mocap.",
    "head_mocap_ori_y": "Head rigid-body orientation (Euler) Y / yaw, from mocap.",
    "head_mocap_ori_z": "Head rigid-body orientation (Euler) Z / roll, from mocap.",
    "head_mocap_valid": "Mocap rigid-body tracking-valid flag (1 = tracked this frame).",
    "treadmill_speed": "Treadmill belt speed reported over Modbus.",
    "distance_on_path": "Cumulative distance travelled along the spline path.",
    "stim_visible": "Gabor patch visibility flag (1 = on screen).",
    "stimulus_onset": "Pulse (1) on the frame the Gabor is first rendered (task-visible rising edge).",
    "response_onset": "Pulse (1) on the frame the response button is detected.",
    "mocap_frame_advanced": "1 when this row sampled a NEW mocap frame (for effective-rate / dropped-frame QC).",
    "lateral_sway_flag": "Lateral-sway guard level from local head X: 0 OK, 1 warn (re-center), "
                         "2 critical (OptiTrack dropout risk). Use to mask suspect head-position frames.",
    "tracking_frame": "OptiTrack frame number of the mocap sample logged in this row "
                      "(OptiTrack runs at 120 Hz, faster than the logger).",
    "tracking_time": "OptiTrack device timestamp (s) of the mocap sample logged in this row; "
                     "its own clock, not zeroed.",
    "clock_hires": "Monotonic high-resolution clock (Stopwatch) started with the logger, "
                   "seconds; the reference clock for the precise reaction times.",
}


# ===========================================================================
# Raw I/O (loose CSV/CSV.GZ or sourcedata archive)
# ===========================================================================
class UnusableRawLog(ValueError):
    """A raw log that holds no analysable frames (empty, header-only, unreadable)."""


def _salvage_text(path: Path) -> str:
    """Readable text of a truncated log, up to its last complete line.

    A crash or disconnect can cut a ``.csv`` mid-row or a ``.csv.gz`` mid-stream; frames
    before the break are recovered.
    """
    if path.name.endswith(".gz"):
        # zlib returns data up to the cut; gzip.open raises without returning it
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            raw = d.decompress(path.read_bytes())
        except zlib.error:
            raw = b""
    else:
        raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    cut = text.rfind("\n")
    return text[:cut + 1] if cut >= 0 else ""


def _read_raw(path: Path) -> tuple[pd.DataFrame, str]:
    """Read one per-frame rig log as ``(frames, log_status)``.

    ``log_status`` is ``"complete"`` or ``"truncated"``. For truncated logs (crash,
    disconnect, damaged file) all complete frames are kept and a partial last row is
    dropped. Rows with a non-numeric timestamp are dropped. Raises :class:`UnusableRawLog`
    if fewer than two frames remain.
    """
    status = "complete"
    try:
        if not path.name.endswith((".gz", ".zip")):
            with open(path, "rb") as f:
                if f.seek(0, 2) > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        raise EOFError("last row is incomplete")
        # compression inferred from the extension
        df = pd.read_csv(path, sep=";", low_memory=False)
    except (EOFError, OSError, zlib.error, pd.errors.ParserError):
        status = "truncated"
        text = _salvage_text(path)
        if not text.strip():
            raise UnusableRawLog(f"{path.name}: nothing readable")
        df = pd.read_csv(io.StringIO(text), sep=";", low_memory=False)
    except pd.errors.EmptyDataError:
        raise UnusableRawLog(f"{path.name}: empty file") from None
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "timestamp" not in df.columns:
        raise UnusableRawLog(f"{path.name}: no timestamp column")
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts.isna().any():
        df = df[ts.notna()].reset_index(drop=True)
    if len(df) < 2:
        raise UnusableRawLog(f"{path.name}: {len(df)} frame(s)")
    return df, status


def _read_text(path: Path) -> str:
    """Read a text file, gzipped or not (used for the calibration '#' header)."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="ignore")


def _archive_to_sourcedata(src: Path, stem: str, sub: str, ses: str,
                           datatype: str, bids_dir: Path) -> Path:
    """Store the raw log gzipped and BIDS-named in sourcedata/<sub>/<ses>/<datatype>/.

    ``.csv.gz`` is copied unchanged; ``.csv`` is gzipped.
    """
    dest_dir = bids_dir / "sourcedata" / sub / ses / datatype
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem}.csv.gz"
    if src.name.endswith(".csv.gz"):
        shutil.copyfile(src, dest)
    else:
        with open(src, "rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
            shutil.copyfileobj(fin, fout)
    return dest


def _move_original(src: Path, bids_dir: Path) -> Path | None:
    """Move an original rig log unchanged into sourcedata/raw/.

    ``raw/`` is outside ``sub-*/ses-*/`` so originals are not mistaken for archives. Never
    overwrites: if a file of that name exists, the loose copy is deleted only if identical,
    otherwise kept with a warning.
    """
    dest_dir = bids_dir / "sourcedata" / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        if filecmp.cmp(src, dest, shallow=False):
            src.unlink()
            return dest
        print(f"  WARNING: {dest} exists with different content; left {src} in place")
        return None
    shutil.move(str(src), str(dest))
    return dest


# ===========================================================================
# Motion / events / channels tables
# ===========================================================================
def _effective_fs(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else C.NOMINAL_FS


def _motion_base(sub: str, ses: str, condition: str | None, run: str | None = None) -> str:
    """BIDS motion file base in schema order: ``task-``, ``tracksys-``, ``acq-``, ``run-``."""
    acq = C.acq_for(condition)
    acq_tag = f"_acq-{acq}" if acq else ""
    run_tag = f"_{run}" if run else ""
    return f"{sub}_{ses}_task-{C.TASK_LABEL}_tracksys-{C.TRACKING_SYSTEM}{acq_tag}{run_tag}"


def _old_order_motion_base(sub: str, ses: str, condition: str | None,
                           run: str | None = None) -> str:
    """Base name used before 2026-09-21 (``acq-`` before ``tracksys-``, rejected by the
    validator); only used to remove or rename such files."""
    acq = C.acq_for(condition)
    acq_tag = f"_acq-{acq}" if acq else ""
    run_tag = f"_{run}" if run else ""
    return f"{sub}_{ses}_task-{C.TASK_LABEL}{acq_tag}_tracksys-{C.TRACKING_SYSTEM}{run_tag}"


def _peek_condition(src: Path) -> str | None:
    """Condition from the first block-id of a raw log."""
    try:
        d = pd.read_csv(src, sep=";", usecols=["block-id"], nrows=1)
        return C.block_to_condition(int(d["block-id"].iloc[0]))
    except Exception:
        return None


_HEAD_PLACEMENT = ("head: rigid body of asymmetrically arranged reflective markers fixed "
                   "to the participant's head")


def _build_motion_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fs = _effective_fs(df["timestamp"].to_numpy())
    t0 = float(pd.to_numeric(df["timestamp"], errors="coerce").iloc[0])
    motion = pd.DataFrame()
    chan_rows = []
    for raw_col, name, point, ctype, comp, units, ref in _CHANNELS:
        if raw_col not in df.columns:
            continue
        motion[name] = pd.to_numeric(df[raw_col], errors="coerce")
        if raw_col == "timestamp":
            # BIDS LATENCY: seconds from recording onset (first sample = 0)
            motion[name] = motion[name] - t0
        # BIDS Motion requires name, component, type, tracked_point, units first.
        chan_rows.append({
            "name": name, "component": comp, "type": ctype, "tracked_point": point,
            "units": units,
            "placement": _HEAD_PLACEMENT if point == "Head" else "n/a",
            "reference_frame": ref, "sampling_frequency": round(fs, 4),
            "description": _CHANNEL_DESCRIPTIONS.get(name, "n/a"),
        })
    return motion, pd.DataFrame(chan_rows)


def _build_events_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each stimulus onset with the row where the trial ended.

    Onset is the ``stimulus-onset`` pulse (a1), or in older logs the rising edge of
    ``task-visible`` (same definition). Times are seconds from the first sample, as ``t``.
    """
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy()
    t = t - t[0]
    if "stimulus-onset" in df.columns and pd.to_numeric(df["stimulus-onset"], errors="coerce").fillna(0).max() == 1:
        onset_mask = pd.to_numeric(df["stimulus-onset"], errors="coerce").fillna(0).to_numpy() == 1
        rising = np.flatnonzero(onset_mask)
    else:
        vis0 = pd.to_numeric(df["task-visible"], errors="coerce").fillna(0).to_numpy()
        rising = np.flatnonzero((vis0[1:] == 1) & (vis0[:-1] == 0)) + 1
    vis = pd.to_numeric(df["task-visible"], errors="coerce").fillna(0).to_numpy()
    falling = np.flatnonzero((vis[1:] == 0) & (vis[:-1] == 1)) + 1
    onset_times = t[rising]
    # A trial ends on one frame, where RecordTrial sets checkpoint-index >= 0 (else -1).
    # Select on checkpoint-index, not response-correct: timeouts have response-correct
    # == -1, the same as the per-frame default, and would be lost. Timeouts are kept with
    # response/correct/RT = "n/a".
    resolved = df[pd.to_numeric(df["checkpoint-index"], errors="coerce").fillna(-1) != -1].copy()

    def onset_for(ts: float):
        prior = onset_times[onset_times <= ts + 1e-6]
        if prior.size == 0:
            return np.nan, np.nan
        on = prior[-1]
        after = falling[t[falling] > on]
        dur = (t[after[0]] - on) if after.size else C.STIM_DURATION_S
        return on, dur

    side = {0: "left", 1: "right"}
    has_precise = "reaction-time-precise" in df.columns
    rows = []
    for i, r in resolved.iterrows():
        res_t = float(t[df.index.get_loc(i)])
        on, dur = onset_for(res_t)
        resp, corr = int(r["checkpoint-response"]), int(r["response-correct"])
        # Use the high-resolution RT (a9) if valid, else the frame-clock RT.
        rt = float(r["checkpoint-reaction-time"])
        if has_precise:
            rtp = pd.to_numeric(pd.Series([r["reaction-time-precise"]]), errors="coerce").iloc[0]
            if pd.notna(rtp) and rtp > 0:
                rt = float(rtp)
        rows.append({
            "onset": round(on, 6) if np.isfinite(on) else "n/a",
            "duration": round(dur, 6) if np.isfinite(dur) else "n/a",
            "trial_type": str(r["checkpoint-type"]),
            "difficulty": str(r["difficulty-level"]),
            "is_training": int(r["is-training"]),
            "checkpoint_index": int(r["checkpoint-index"]),
            "stimulus_side": side.get(int(r["visual-tilt"]), "n/a"),
            "tilt_angle_deg": round(float(r["visual-angle"]), 4),
            "expected_response": side.get(int(r["expected-response"]), "n/a"),
            "response": side.get(resp, "n/a") if resp != -1 else "n/a",
            "response_correct": corr if corr != -1 else "n/a",
            "response_time": round(rt, 4) if rt > 0 else "n/a",
            "resolution_time": round(res_t, 6),
            "distance_on_path": round(float(r["distance-on-path"]), 4),
        })
    ev = pd.DataFrame(rows)
    return ev.sort_values("resolution_time").reset_index(drop=True) if not ev.empty else ev


def _motion_json(df: pd.DataFrame, channels: pd.DataFrame) -> dict:
    # Only fields defined for _motion.json in BIDS 1.11.1. Sampling and coordinate frames
    # are documented in the dataset README and channels.json.
    t = df["timestamp"].to_numpy()
    fs = _effective_fs(t)
    counts = channels["type"].value_counts().to_dict()
    tracked = {p for p in channels["tracked_point"] if p not in ("n/a", "", None)}
    meta = {
        "TaskName": C.TASK_LABEL,
        "TaskDescription": "2-AFC Gabor tilt discrimination while walking on a treadmill "
                           "or standing in a CAVE: a Gabor is briefly tilted left or right "
                           "and the participant reports the direction. OptiTrack tracks the "
                           "head at 120 Hz; the log keeps the latest sample on every rendered "
                           "frame (nominal 60 Hz, variable spacing), so use the 't' channel "
                           "for timing. SamplingFrequencyEffective is the median rate this "
                           "recording actually achieved.",
        "Instructions": "Report whether the briefly presented Gabor was tilted left or right.",
        "TrackingSystemName": "OptiTrack optical motion capture (120 Hz) of a head rigid "
                              "body in the Interactive Locomotion Lab CAVE",
        "Manufacturer": C.MANUFACTURER,
        "ManufacturersModelName": C.MOCAP_MODEL,
        "SoftwareVersions": f"{C.MOCAP_SOFTWARE}; Unity 2017.4.25f1 with UniCave; unityVrpn "
                            "(VRPN) for the Unity head transform; OptiTrack NatNet client "
                            "for the head_mocap_* channels",
        "SamplingFrequency": C.NOMINAL_FS,
        "SamplingFrequencyEffective": round(fs, 4),
        "TrackedPointsCount": int(len(tracked)),
        "MotionChannelCount": int(len(channels)),
        "SubjectArtefactDescription": "n/a",
        "MissingValues": "n/a",
    }
    for kind in ("ACCEL", "ANGACCEL", "GYRO", "JNTANG", "LATENCY", "MAGN", "MISC",
                 "ORNT", "POS", "VEL"):
        meta[f"{kind}ChannelCount"] = int(counts.get(kind, 0))
    meta.update(_institution())
    return meta


def _institution() -> dict:
    """Institution fields for beh and motion sidecars (from config)."""
    fields = {"InstitutionName": C.INSTITUTION_NAME,
              "InstitutionAddress": C.INSTITUTION_ADDRESS,
              "InstitutionalDepartmentName": C.INSTITUTION_DEPARTMENT}
    return {k: v for k, v in fields.items() if v}


# The three reference frames of channels.tsv, described in *_channels.json. Unity is
# left-handed (X right, Y up, Z forward) with Euler order Z, X, Y.
_REFERENCE_FRAMES = {
    "global": {"SpatialAxes": "RUF", "RotationRule": "left-hand", "RotationOrder": "ZXY",
               "Description": "Unity world space. The virtual rig travels along the path, "
                              "so this frame includes the forward progress through the "
                              "scene."},
    "local": {"SpatialAxes": "RUF", "RotationRule": "left-hand", "RotationOrder": "ZXY",
              "Description": "Relative to the moving CAVE rig (Unity axes). Within-CAVE "
                             "head movement: X lateral sway, Y vertical bob, Z fore-aft. "
                             "The gait analysis uses these channels."},
    "mocap": {"SpatialAxes": "RUB", "RotationRule": "right-hand", "RotationOrder": "ZXY",
              "Description": "OptiTrack/Motive tracking volume as streamed (right-handed, "
                             "Y up), read straight from the head rigid body. Same "
                             "measurement as 'local' up to a fixed offset, with Z pointing "
                             "backwards. Orientation is the quaternion head_mocap_quat_*; "
                             "head_mocap_ori_* are Unity's ZXY Euler decomposition of it."},
}


def _channels_json() -> dict:
    return {"reference_frame": {
        "Description": "Coordinate frame in which each channel is expressed.",
        "Levels": _REFERENCE_FRAMES}}


def _events_json() -> dict:
    return {
        "StimulusPresentation": {
            "OperatingSystem": "Linux (Ubuntu 24.04) on the head node; Windows on the "
                               "three render nodes",
            "SoftwareName": "Unity with UniCave (distributed CAVE rendering)",
            "SoftwareVersion": "2017.4.25f1",
            "ScreenRefreshRate": 60,
            "ScreenResolution": [3840, 2160],
            "ScreenSize": [1.42, 0.80],
            "ScreenDistance": 1.0,
            "HeadStabilization": "none",
        },
        "onset": {"Description": "Gabor onset: the frame on which the head node first "
                                 "rendered it, in seconds from the first sample of the "
                                 "matching _motion.tsv (same zero as its 't' channel). The "
                                 "image reaches the wall panels about 40 ms later (the "
                                 "panels' input lag; Hofmann 2023, App. C). See "
                                 "task-ILL_stimulus.json for the display.", "Units": "s"},
        "duration": {"Description": "Measured on-screen time, from the frame the Gabor "
                                    "appeared to the frame it disappeared. Nominally 0.1 s; "
                                    "longer when the frame rate dropped.", "Units": "s"},
        "trial_type": {"Description": "Locomotion condition of the block this trial "
                                      "belongs to.",
                       "Levels": {"walking": "walking on the treadmill",
                                  "standing": "standing still"}},
        "difficulty": {"Description": "Tilt level, set per participant from the QUEST+ "
                                      "calibration (target proportion correct).",
                       "Levels": {"HARD": "about 65 % correct",
                                  "THR": "about 75 % correct (threshold)",
                                  "EASY": "about 85 % correct"}},
        "is_training": {"Description": "1 = training trial at the start of the block "
                                       "(excluded from analysis), 0 = main trial."},
        "checkpoint_index": {"Description": "Trial number within the block, from 0."},
        "stimulus_side": {"Description": "Direction the Gabor was tilted.",
                          "Levels": {"left": "tilted left", "right": "tilted right"}},
        "tilt_angle_deg": {"Description": "Tilt magnitude presented, from vertical.",
                           "Units": "degree"},
        "expected_response": {"Description": "The correct response (same as "
                                             "stimulus_side).",
                              "Levels": {"left": "left button", "right": "right button"}},
        "response": {"Description": "Button pressed; n/a = no response within the 2 s "
                                    "response window.",
                     "Levels": {"left": "left button", "right": "right button"}},
        "response_correct": {"Description": "1 = correct, 0 = incorrect, n/a = no response."},
        "response_time": {"Description": "Reaction time from Gabor onset (high-resolution "
                                         "clock when available).", "Units": "s"},
        "resolution_time": {"Description": "When the trial ended (response or timeout), on "
                                           "the same zero as onset.", "Units": "s"},
        "distance_on_path": {"Description": "Distance travelled along the virtual path when "
                                            "the trial ended (0 while standing).", "Units": "m"},
    }


# ===========================================================================
# Counterbalance
# ===========================================================================
def _setup_fix(sub: str, visit) -> dict:
    """Corrected setup-screen values for this subject and visit (``config.SETUP_CORRECTIONS``)."""
    return {k: v for k, v in C.SETUP_CORRECTIONS.get((sub, visit), {}).items() if k != "reason"}


def _counterbalance(df: pd.DataFrame, sub: str) -> dict:
    """Condition order, visit and start time from the raw log.

    Logged by ``ExperimentConductor`` since 2026-08-18. Older recordings (old
    ``participant % 2`` rule, no crossover) get ``n/a``. Wrongly logged values are replaced
    from ``config.SETUP_CORRECTIONS``.
    """
    def last(col, cast=None, default="n/a"):
        if col not in df.columns:
            return default
        s = df[col].dropna()
        if not len(s):
            return default
        v = s.iloc[-1]
        if cast is None:
            return default if str(v) in ("", "n/a", "-1") else str(v)
        try:
            v = cast(v)
        except (TypeError, ValueError):
            return default
        return default if v < 0 else v

    cb = {"visit": last("visit", int),
          "counterbalance_slot": last("counterbalance-slot", int),
          "condition_order_first": last("condition-order-first"),
          "acq_time": last("session-start-iso")}
    return cb | {k: v for k, v in _setup_fix(sub, cb["visit"]).items() if k in cb}


def _condition_position(condition: str, order_first) -> str:
    """Whether this condition ran first or second in its visit (time-on-task covariate for
    the H2 threshold contrast)."""
    if order_first in ("standing", "walking"):
        return "first" if condition == order_first else "second"
    return "n/a"


# ===========================================================================
# Per-file conversion
# ===========================================================================
def convert_motion_file(csv_path: Path, sub: str, ses: str, bids_dir: Path,
                        run: str | None = None) -> tuple[dict, dict]:
    """Convert one raw block CSV into a BIDS motion recording.

    Writes ``_motion.tsv`` (per frame), ``_channels.tsv``, ``_motion.json`` and
    ``_events.tsv`` (one row per stimulus). ``run`` is the block index within the
    condition (``run-`` entity).

    Returns ``(participant_row, session_row)``; the session row also feeds ``scans.tsv``.
    """
    df, log_status = _read_raw(csv_path)
    # condition from the data, valid for loose and archived sources
    condition = C.block_to_condition(int(df["block-id"].iloc[0]))
    motion_dir = bids_dir / sub / ses / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    base = _motion_base(sub, ses, condition, run)
    # Remove files of this recording written under older names (no acq-, old entity order).
    legacy = f"{sub}_{ses}_task-{C.TASK_LABEL}_tracksys-{C.TRACKING_SYSTEM}"
    for old in {legacy, _old_order_motion_base(sub, ses, condition, run)} - {base}:
        for ext in ("_motion.tsv", "_channels.tsv", "_channels.json", "_motion.json",
                    "_events.tsv", "_events.json"):
            (motion_dir / f"{old}{ext}").unlink(missing_ok=True)
    motion, channels = _build_motion_table(df)
    events = _build_events_table(df)

    # _motion.tsv has no header; columns are the rows of _channels.tsv. Writing ~50 000
    # rows is the slowest step, so it is skipped if table, code and file are unchanged
    # (stage_cache).
    motion_tsv = motion_dir / f"{base}_motion.tsv"
    fp = stage_cache.fingerprint(stage_cache.code_digest("bidsify.py"), motion)
    unit = str(motion_tsv.resolve())
    if not stage_cache.is_fresh("bidsify_motion", unit, fp, [motion_tsv], root=bids_dir):
        motion.to_csv(motion_tsv, sep="\t", index=False, header=False,
                      na_rep="n/a", float_format="%.6f")
        stage_cache.record("bidsify_motion", unit, fp, [motion_tsv], root=bids_dir)
    channels.to_csv(motion_dir / f"{base}_channels.tsv", sep="\t", index=False, na_rep="n/a")
    _dump_json(motion_dir / f"{base}_channels.json", _channels_json())
    _dump_json(motion_dir / f"{base}_motion.json", _motion_json(df, channels))

    # events use the same base (BIDS Motion recommendation)
    events.to_csv(motion_dir / f"{base}_events.tsv", sep="\t", index=False, na_rep="n/a")
    _dump_json(motion_dir / f"{base}_events.json", _events_json())

    speed = pd.to_numeric(df["treadmill-speed"], errors="coerce")
    speed = speed[(speed > 0.05) & (speed < C.MAX_PLAUSIBLE_SPEED_MPS)]
    # Thresholds used in this block. Generic columns are kept for older readers;
    # convert_all overrides them from the calibration files.
    thr_deg = round(float(df["threshold-angle"].iloc[-1]), 4)
    easy_deg = round(float(df["easy-angle"].iloc[-1]), 4)
    hard_deg = round(float(df["hard-angle"].iloc[-1]), 4)
    cb = _counterbalance(df, sub)
    prow = {
        "participant_id": sub,
        "threshold_angle_deg": thr_deg,
        "easy_angle_deg": easy_deg,
        "hard_angle_deg": hard_deg,
        f"threshold_{condition}_deg": thr_deg,
        f"easy_{condition}_deg": easy_deg,
        f"hard_{condition}_deg": hard_deg,
        "treadmill_speed_mps": round(float(speed.median()), 4) if len(speed) else "n/a",
        "counterbalance_slot": cb["counterbalance_slot"],
    }
    srow = {"session_id": ses, "datatype": "motion", "condition": condition,
            "n_trials": int(len(events)), "duration_s": round(float(df["timestamp"].iloc[-1]
                                                                     - df["timestamp"].iloc[0]), 1),
            "visit": cb["visit"], "condition_order_first": cb["condition_order_first"],
            "condition_position": _condition_position(condition, cb["condition_order_first"]),
            "acq_time": cb["acq_time"],
            "effective_sampling_hz": round(_effective_fs(df["timestamp"].to_numpy()), 2),
            "log_status": log_status,
            # relative path, matched in _write_scans_tsv
            "filename": f"{ses}/motion/{base}_motion.tsv"}
    note = ("" if log_status == "complete" else
            f"  [WARNING: log {log_status}; kept the {len(motion)} whole frames before the break]")
    print(f"  {base}: {len(motion)} samples, {len(events)} events -> {motion_dir}{note}")
    return prow, srow


def convert_calibration(csv_path: Path, sub: str, ses: str, bids_dir: Path,
                        condition: str | None = None, run: str | None = None) -> dict:
    """Convert one QUEST+ calibration log into a BIDS ``_beh.tsv`` table.

    Calibrations have no motion stream and go to ``beh/`` in ses-01. The condition is the
    ``acq-`` entity and ``run`` is the visit number.
    """
    df = pd.read_csv(csv_path, sep=";", comment="#")
    # Visit and condition order on every row since 2026-09-16; passed to scans.tsv.
    # Calibration logs have no start time.
    cb = _counterbalance(df.rename(columns={"condition_order_first": "condition-order-first"}),
                         sub)
    df = df.assign(**{k: v for k, v in _setup_fix(sub, cb["visit"]).items() if k in df.columns})
    beh_dir = bids_dir / sub / ses / "beh"
    beh_dir.mkdir(parents=True, exist_ok=True)
    # acq- for the condition (legacy single calibration: none), run- for the visit.
    acq = C.acq_for(condition)
    base = _stem_for(sub, ses, "beh", condition, run)
    df.to_csv(beh_dir / f"{base}.tsv", sep="\t", index=False, na_rep="n/a")
    _dump_json(beh_dir / f"{base}.json", {
        # _beh.tsv, not _events.tsv: the QUEST+ log has no onset/duration (BIDS rule).
        "TaskName": C.TASK_LABEL,
        "TaskDescription": "QUEST+ adaptive calibration (Watson 2017) of the 2-AFC "
                           "Gabor tilt-discrimination threshold; two interleaved trackers "
                           "(A/B). Precedes the main locomotion experiment (ses-02).",
        "CalibrationCondition": acq or "n/a",
        "Instructions": "Report whether the briefly presented Gabor was tilted left or right.",
        **_institution(),
        "tracker": {"Description": "Which trial stream this row belongs to.",
                    "Levels": {"A": "QUEST+ tracker A (interleaved with B)",
                               "B": "QUEST+ tracker B (interleaved with A)",
                               "catch": "easy attentiveness probe, not fed to QUEST+",
                               "warmup": "easy practice trial at the start, not fed to "
                                         "QUEST+"}},
        "global_trial": {"Description": "Trial number across the whole calibration."},
        "tracker_trial": {"Description": "Trial number within this tracker."},
        "presented_tilt_deg": {"Description": "Tilt magnitude presented, from vertical.",
                               "Units": "degree"},
        "tilt_direction": {"Description": "Direction the Gabor was tilted.",
                           "Levels": {"0": "left", "1": "right"}},
        "response": {"Description": "Button pressed.",
                     "Levels": {"0": "left", "1": "right", "-1": "no response"}},
        "correct": {"Description": "Whether the response was correct.",
                    "Levels": {"1": "correct", "0": "incorrect",
                               "-1": "no response (not fed to QUEST+)"}},
        "post_mean_threshold_deg": {"Description": "QUEST+ posterior-mean threshold after "
                                                   "this trial.", "Units": "degree"},
        "post_sd_log": {"Description": "Posterior SD of the threshold after this trial, in "
                                       "log10 units."},
        # identity columns, logged on every row since 2026-09-16
        "participant": {"Description": "Participant number typed on the setup screen."},
        "visit": {"Description": "Which of the participant's two visits (1 or 2)."},
        "counterbalance_slot": {"Description": "Slot in the permuted-block allocation "
                                               "that set the condition order."},
        "block": {"Description": "Block field at launch (0 = standing, 1 = walking)."},
        "trial": {"Description": "Trial field at launch; for a calibration, the visit "
                                 "number by protocol."},
        "condition": {"Description": "Locomotor condition this calibration measured "
                                     "(standing / walking)."},
        "condition_order_first": {"Description": "Condition run FIRST in this visit."},
    })
    thr = _calibration_thresholds(csv_path)
    print(f"  {base}: {len(df)} trials -> {beh_dir}")
    return {"session_id": ses, "datatype": "beh", "condition": condition or "calibration",
            "n_trials": int(len(df)), "duration_s": "n/a",
            "visit": cb["visit"], "condition_order_first": cb["condition_order_first"],
            "condition_position": _condition_position(condition, cb["condition_order_first"]),
            "filename": f"{ses}/beh/{base}.tsv",
            "_thresholds": thr}


def _calibration_thresholds(path: Path) -> dict:
    out = {}
    for line in _read_text(path).splitlines():
        if line.startswith("# threshold_deg"):
            for kv in line.lstrip("# ").split(";"):
                k, _, v = kv.partition("=")
                try:
                    out[k.strip()] = float(v)
                except ValueError:
                    pass
            break
    return out


# ===========================================================================
# Dataset-level files
# ===========================================================================
def _dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


_SESSION_DESCRIPTIONS = {
    C.CALIBRATION_SESSION: "QUEST+ tilt-threshold calibrations (standing and walking), "
                           "from every visit",
    C.EXPERIMENT_SESSION: "Gabor task blocks while walking and standing, with head "
                          "tracking, from every visit",
}


def _write_sessions_tsv(bids_dir: Path, sub: str, sessions: list[dict]) -> None:
    # One row per session. Sessions are parts of the protocol, not visits; scans.tsv has
    # the visit per file.
    rows = [{"session_id": s["session_id"], "datatype": s["datatype"],
             "description": _SESSION_DESCRIPTIONS.get(s["session_id"], "n/a")}
            for s in sessions]
    sdf = pd.DataFrame(rows).drop_duplicates("session_id").sort_values("session_id")
    (bids_dir / sub).mkdir(parents=True, exist_ok=True)
    sdf.to_csv(bids_dir / sub / f"{sub}_sessions.tsv", sep="\t", index=False, na_rep="n/a")
    _dump_json(bids_dir / sub / f"{sub}_sessions.json", {
        "session_id": {"Description": "BIDS session label. Sessions split the protocol "
                                      "into calibration (ses-01) and experiment (ses-02); "
                                      "they are NOT visits (see the visit column of "
                                      "scans.tsv)."},
        "datatype": {"Description": "BIDS datatype folder the session contains.",
                     "Levels": {"beh": "behavioural tables", "motion": "motion recordings"}},
        "description": {"Description": "What the session contains."},
    })


def _write_stim_sidecar(bids_dir: Path) -> None:
    """Dataset-level Gabor stimulus description (`task-ILL_stimulus.json`).

    Parameters not in the per-frame CSV. The suffix ``_stimulus`` is custom (in
    ``.bidsignore``); ``_stim`` is reserved for continuous stimulus recordings. Values are
    the ``FollowerPlane`` settings of the ILL-UniCave prefab in visual degrees (0.70 m quad
    at 2.0 m = 19.85 deg).
    """
    _dump_json(bids_dir / f"task-{C.TASK_LABEL}_stimulus.json", {
        "StimulusType": "Gabor patch (sinusoidal grating in a Gaussian envelope)",
        "Task": "2-AFC orientation discrimination: tilted left or right of vertical",
        "SpatialFrequency": {"Value": 3.0, "Units": "cycles/degree"},
        "EnvelopeSigma": {"Value": 0.70, "Units": "degree",
                          "Description": "SD of the Gaussian envelope."},
        "Contrast": {"Value": 0.70, "Units": "Michelson",
                     "Description": "Fixed for all trials; difficulty is set by tilt only."},
        "ApertureDiameter": {"Value": 8.7, "Units": "degree",
                             "Description": "Visible disc around the Gabor, on a uniform "
                                            "grey quad of 19.85 deg."},
        "EccentricityDeg": {"Value": 0.0, "Units": "degree",
                            "Description": "Head-locked at eye level, straight ahead."},
        "ViewingDistance": {"Value": 2.0, "Units": "m",
                            "Description": "Virtual distance of the head-locked quad."},
        "NominalDuration": {"Value": C.STIM_DURATION_S, "Units": "s",
                            "Description": "Intended on-screen time. The measured time of "
                                           "every trial is the duration column of "
                                           "*_events.tsv."},
        "TiltLevels": "Per participant and condition from the QUEST+ calibration: EASY "
                      "(~85 % correct), THR (~75 %), HARD (~65 %). Values in "
                      "participants.tsv; the tilt of each trial in *_events.tsv.",
        "Display": {
            "Description": "Interactive Locomotion Lab CAVE (Hofmann 2023, ch. 2): six 65-inch "
                           "Sony Bravia X85 panels in three columns by two rows, arranged as a "
                           "half hexagon around the treadmill. The centre column faces along "
                           "the walking direction; the side columns are rotated by 60 degrees. "
                           "The upper panels are mounted upside down so the gap between rows "
                           "is 2 cm. The lower edge of the array is 0.5 m above the belt. "
                           "From 1.70 m eye height the array covers about 180 x 64 degrees.",
            "PanelResolution": [3840, 2160],
            "PanelActiveArea": {"Value": [1.42, 0.80], "Units": "m"},
            "RefreshRate": {"Value": 60, "Units": "Hz"},
            "ViewingDistanceFrontWall": {"Value": 1.0, "Units": "m",
                                         "Description": "Approximate distance from the "
                                                        "participant to the front wall."},
            "Rendering": "One Linux head node runs the experiment and three Windows render "
                         "nodes draw one panel column each (UniCave, off-axis projection "
                         "from the tracked head position).",
        },
        "DisplayLatency": {
            "Value": 0.04, "Units": "s",
            "Description": "The panels' own input lag: the time between the frame on which a "
                           "stimulus is rendered (the logged onset) and light on the panels. "
                           "The full chain from a tracked movement to the picture measured "
                           "127.0 +- 2.5 ms (pendulum test, Hofmann 2023, App. C; not "
                           "re-measured for this study).",
        },
        "Apparatus": "Treadmill 2.5 x 1.5 m (Ingenieurbüro Achilles) on a Moog "
                     "170-140B-2-D-1 six-axis motion platform, which is held level and "
                     "still in this study. Participants wear a full-body harness on a "
                     "retracting lifeline.",
        "Source": "FollowerPlane component of Assets/ILL-UniCave 1 1.prefab "
                  "(gaborFrequency 59.6, gaborSigma 0.0353, gaborDiskRadius 0.22, "
                  "gaborContrast 0.7).",
    })


def _scan_condition(name: str) -> str:
    """Condition of a data file from its acq- entity, or 'calibration' for beh."""
    m = re.search(r"_acq-(walking|standing)", name)
    if m:
        return m.group(1)
    return "calibration" if "_beh" in name else "unknown"


def _exclusion_reason(name: str, bids_dir: Path | None = None) -> str:
    """Reason from ``config.EXCLUDED_RECORDINGS``, or n/a.

    Automatic rule verdicts are in ``derivatives/qc/gate.tsv``, which the raw dataset does
    not depend on. Keys are recording labels (file name without ``tracksys-`` and suffix).
    """
    label = re.sub(r"_tracksys-[a-zA-Z0-9]+", "", name).removesuffix("_motion.tsv")
    reason = bids_io.excluded_recordings(bids_dir, apply_block_rule=False).get(label)
    return reason.removeprefix("manual: ") if reason else "n/a"


def _write_scans_tsv(bids_dir: Path, sub: str, sessions: list[dict] | None = None) -> None:
    """Per-subject `sub-XX_scans.tsv`: data files, condition and counterbalance.

    Counterbalance columns (``visit``, ``condition_order_first``, ``condition_position``)
    vary within participants (crossover), so they are here, not in participants.tsv.
    """
    sub_dir = bids_dir / sub
    files = sorted(list(sub_dir.glob("ses-*/motion/*_motion.tsv"))
                   + list(sub_dir.glob("ses-*/beh/*_beh.tsv")))
    if not files:
        return
    meta = {s["filename"]: s for s in (sessions or []) if s.get("filename")}
    rows = []
    for f in files:
        rel = f.relative_to(sub_dir).as_posix()
        m = meta.get(rel, {})
        rows.append({"filename": rel, "condition": _scan_condition(f.name),
                     "visit": m.get("visit", "n/a"),
                     "acq_time": m.get("acq_time", "n/a"),
                     "condition_order_first": m.get("condition_order_first", "n/a"),
                     "condition_position": m.get("condition_position", "n/a"),
                     "n_trials": m.get("n_trials", "n/a"),
                     "duration_s": m.get("duration_s", "n/a"),
                     "effective_sampling_hz": m.get("effective_sampling_hz", "n/a"),
                     "log_status": m.get("log_status", "n/a"),
                     "exclusion_reason": _exclusion_reason(f.name, bids_dir)})
    # Sort by visit, calibrations first, then blocks in recording order.
    rows.sort(key=lambda r: (str(r["visit"]), "/beh/" not in r["filename"],
                             str(r["acq_time"]), r["filename"]))
    pd.DataFrame(rows).to_csv(sub_dir / f"{sub}_scans.tsv", sep="\t", index=False, na_rep="n/a")
    _dump_json(sub_dir / f"{sub}_scans.json", {
        "filename": {"Description": "Path of the data file, relative to the subject directory."},
        "condition": {"Description": "Locomotion condition of this file.",
                      "Levels": {"walking": "walking on the treadmill (calibration or block)",
                                 "standing": "standing still (calibration or block)",
                                 "calibration": "single legacy calibration without a "
                                                "condition"}},
        "acq_time": {"Description": "Wall-clock start of the recording (ISO-8601, local), "
                                    "from the rig's `session-start-iso` column. Used for "
                                    "time-on-task within a session and for the interval "
                                    "between a participant's two visits. n/a for "
                                    "calibrations, which log no start time, and for "
                                    "recordings made before 2026-08-18."},
        "visit": {"Description": "Which of the participant's two visits this file "
                                 "belongs to (1 or 2)."},
        "n_trials": {"Description": "Rows in the file's trial table: every Gabor trial of "
                                    "a block including training, or every calibration "
                                    "trial including warm-up and catch trials."},
        "duration_s": {"Description": "Length of the motion recording (n/a for "
                                      "calibrations).", "Units": "s"},
        "effective_sampling_hz": {"Description": "Median rate at which the rig actually "
                                                 "logged frames (nominal 60 Hz). A value far "
                                                 "below 60 means the rendering slowed down, "
                                                 "so timing is coarser and stimuli stayed on "
                                                 "screen longer than intended: check such "
                                                 "files before analysis.", "Units": "Hz"},
        "log_status": {"Description": "Whether the raw block log was read to its end.",
                       "Levels": {"complete": "the log ends on a whole row",
                                  "truncated": "the log stops part-way (crash, disconnect "
                                               "or a damaged file); every whole frame "
                                               "before the break was kept"}},
        "exclusion_reason": {"Description": "Why this recording is left out of the analysis "
                                            "(n/a = included). The data are kept unchanged."},
        "condition_order_first": {"Description": "Condition run FIRST in this visit. The "
                                                 "order crosses over between visits, so a "
                                                 "participant has both values across their "
                                                 "two visits.",
                                  "Levels": {"standing": "standing block(s) first",
                                             "walking": "walking block(s) first"}},
        "condition_position": {"Description": "Whether THIS file's condition ran first or "
                                              "second in its visit — the time-on-task "
                                              "covariate for the standing-vs-walking "
                                              "threshold contrast (H2).",
                               "Levels": {"first": "early half of the visit",
                                          "second": "late half of the visit"}},
    })


def _write_derivatives_description(bids_dir: Path) -> None:
    """Write ``derivatives/README.md`` and a ``dataset_description.json`` in each existing
    stage folder (new stages add theirs via ``bids_io.deriv_dir``)."""
    deriv = bids_dir / "derivatives"
    deriv.mkdir(parents=True, exist_ok=True)
    # BIDS-Derivatives describes each stage folder, not derivatives/ itself.
    (deriv / "dataset_description.json").unlink(missing_ok=True)
    for d in sorted(p for p in deriv.iterdir() if p.is_dir()):
        bids_io.describe_derivative(d.name, deriv)
    lines = ["# Analysis outputs", "",
             "Everything here is regenerated from the raw dataset one level up by "
             "`python -m Analysis.run_pipeline`; nothing in this folder is original data.",
             "Each folder is one analysis stage and carries its own `dataset_description.json`.",
             "", "**Layout.** Inside a folder, data files follow the raw dataset's layout "
             "(`sub-XX/ses-YY/`) and are named after the recording they come from. Figures "
             "sit in a `figures/` folder next to the data they show. A block gets one sheet "
             "per stage, `<recording>_<stage>.png` (for example "
             "`gait/sub-01/ses-02/figures/sub-01_ses-02_task-ILL_acq-walking_run-05_gait.png`); "
             "results across participants sit in `<stage>/figures/`.", "",
             "`derivatives/summary/report.html` shows the block sheets and every figure across "
             "participants on one page.", ""]
    for group, folders in bids_io.DERIVATIVE_GROUPS.items():
        lines += [f"## {group}", "", "| Folder | Contents |", "|---|---|"]
        lines += [f"| `{name}/` | {desc} |" for name, desc in folders.items()]
        lines.append("")
    lines += ["A folder appears once its stage has run.", ""]
    (deriv / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _consolidate_participants(participants: list[dict]) -> pd.DataFrame:
    """Merge the per-recording rows into one row per participant.

    Each numeric field is the mean over rows with a value, so the walking belt speed is
    kept (standing rows have n/a) and the calibration angles are preserved.
    """
    df = pd.DataFrame(participants)
    if df.empty or "participant_id" not in df.columns:
        return df
    rows = []
    for pid, g in df.groupby("participant_id"):
        row = {"participant_id": pid}
        for col in df.columns:
            if col == "participant_id":
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if col == "counterbalance_slot" and vals.nunique() > 1:
                raise ValueError(f"{pid}: files disagree on counterbalance_slot "
                                 f"{sorted(vals.unique())}; add a config.SETUP_CORRECTIONS entry")
            if not len(vals):
                row[col] = "n/a"
                continue
            m = float(vals.mean())
            # keep integer columns as int
            row[col] = int(m) if float(m).is_integer() else round(m, 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("participant_id")


_README = """\
# ILL: visual perception while walking (Motion-BIDS dataset)

Participants judge whether a briefly shown Gabor patch is tilted left or right
(two-alternative forced choice) while walking on a treadmill or standing still in a
CAVE, with their head tracked by OptiTrack. The study replicates and extends Davidson,
Verstraten & Alais (2024), Nat. Commun. 15:2027.

This file is written by `Analysis/bidsify.py`. Change the generator, not this file.

## The lab

Recorded in the Interactive Locomotion Lab (ILL) at Bielefeld University, described in
Hofmann (2023, PhD thesis, ch. 2):

- **Treadmill:** 2.5 x 1.5 m belt (Ingenieurbüro Achilles) on a Moog six-axis motion
  platform, which stays level and still in this study.
- **Display:** six 65-inch 4K panels (Sony Bravia X85, 60 Hz) in a half hexagon, three
  columns by two rows, covering about 180 x 64 degrees. Rendered by Unity 2017.4 with
  UniCave on one Linux head node and three Windows render nodes, with off-axis projection
  from the tracked head.
- **Tracking:** 16 OptiTrack Flex 13 cameras at 120 Hz (Motive), following a rigid body of
  reflective markers on the participant's head.
- **Timing:** a stimulus appears on the panels about 40 ms after the frame on which it is
  logged, which is the panels' input lag (Hofmann 2023, App. C).

The full stimulus and display specification is in `task-ILL_stimulus.json`.

## What is where

```
data/
├── README.md                  this overview
├── dataset_description.json   dataset metadata
├── participants.tsv / .json   one row per participant: thresholds, belt speed, counterbalance slot
├── task-ILL_stimulus.json     the Gabor stimulus in retinal units (custom file, see .bidsignore)
├── sub-XX/
│   ├── sub-XX_sessions.tsv    what each session contains
│   ├── sub-XX_scans.tsv       one row per data file: condition, visit, start time,
│   │                          order, trials, duration, effective sampling rate
│   ├── ses-01/beh/            QUEST+ calibrations, one table per condition and visit
│   └── ses-02/motion/         experiment blocks, one set of files per block:
│         *_motion.tsv           per-frame head tracking and task signals (no header row)
│         *_channels.tsv/.json   the columns of *_motion.tsv, in order, and their frames
│         *_motion.json          recording metadata (nominal and effective sampling rate)
│         *_events.tsv/.json     one row per Gabor trial
├── sourcedata/
│   ├── raw/                   the original rig logs, byte for byte, original names
│   └── sub-XX/ses-YY/...      the same logs gzipped and BIDS-named (read by the pipeline)
└── derivatives/               analysis outputs, one folder per stage (see its README.md)
```

## File names

`sub-01_ses-02_task-ILL_tracksys-optitrack_acq-walking_run-03_motion.tsv`

| Part | Meaning |
|---|---|
| `sub-01` | participant (`p01` on the rig) |
| `ses-01` / `ses-02` | calibration / experiment. Sessions are parts of the protocol, **not visits** |
| `task-ILL` | the only task |
| `tracksys-optitrack` | tracking system (motion files only) |
| `acq-walking` / `acq-standing` | condition |
| `run-03` | block number within that condition, counted across both visits; for a calibration, the visit number |

The rig's name for the same block is `p01-v1-b1-t03_walking.csv.gz`: participant 01,
visit 1, block field 1 (walking), trial field 3, which becomes `run-03`.

## Visits

Every participant comes twice, with the order of standing and walking reversed on the
second visit. Both visits' calibrations are in `ses-01` and both visits' blocks in
`ses-02`; the `visit` column of `sub-XX_scans.tsv` says which visit every file belongs
to, and `condition_position` whether that condition came first or second in the visit.

## Time and sampling

The rig logs one row per rendered frame, nominally 60 Hz. Frames are not evenly spaced,
so always use the `t` channel for timing. `t` and all event times count seconds from
the first sample of the recording. `SamplingFrequency` in `*_motion.json` is the nominal
60 Hz and `SamplingFrequencyEffective` the rate the recording actually reached; the same
effective rate is listed per file in `sub-XX_scans.tsv`. A recording far below 60 Hz was
slowed down by rendering: its timing is coarser and its stimuli stayed on screen longer
than the nominal 0.1 s (the measured time is in the `duration` column of `*_events.tsv`).
Check such recordings before analysis.

## Which head channels to use

| Channels | What they are |
|---|---|
| `head_local_pos_*` | head relative to the CAVE rig: `x` lateral sway, `y` vertical bob, `z` fore-aft. Used for gait |
| `head_mocap_*` | the same head rigid body read straight from OptiTrack, including its orientation (`head_mocap_quat_*`) |
| `head_global_*` | Unity world space, which includes the travel along the virtual path |
| `head_local_ori_*`, `head_global_ori_*` | Unity transform angles. They do not carry head rotation (only the path heading shows in `head_global_ori_y`); use `head_mocap_quat_*` |

The OptiTrack stream has been seen to return, for single samples, a different tracked
object under the head's ID (the position drops by about 1.4 m and the orientation jumps
by more than 100 degrees). Filter `head_mocap_*` for such jumps before use. The
reference frames (`global`, `local`, `mocap`) are defined in every `*_channels.json`.

Missing values are written as `n/a`.
"""

# participants.tsv column order: identity, per-condition thresholds, belt speed, generic
# (walking) thresholds for older readers.
_PARTICIPANT_COLUMNS = [
    "participant_id", "counterbalance_slot",
    "threshold_standing_deg", "easy_standing_deg", "hard_standing_deg",
    "threshold_walking_deg", "easy_walking_deg", "hard_walking_deg",
    "treadmill_speed_mps",
    "threshold_angle_deg", "easy_angle_deg", "hard_angle_deg",
]


def _write_dataset_level(bids_dir: Path, participants: list[dict]) -> None:
    bids_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(bids_dir / "dataset_description.json", {
        "Name": "ILL: visual perception while walking in a CAVE",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "raw",
        "Authors": list(C.AUTHORS),
        "Acknowledgements": (f"Supervised by {' and '.join(C.SUPERVISORS)}. "
                             if C.SUPERVISORS else "")
                            + "Replicates and extends Davidson, Verstraten & Alais (2024), "
                              "Nat. Commun. 15:2027.",
        "ReferencesAndLinks": [
            "https://doi.org/10.1038/s41467-024-45780-4",
            "Hofmann, F. (2023). Human Motion Planning and -Adjustment during Obstacle "
            "Traversal in Virtual Reality. PhD thesis, Faculty of Biology, Bielefeld "
            "University. (Apparatus: the Interactive Locomotion Lab, ch. 2 and App. C.)",
            "https://doi.org/10.1007/s00221-022-06443-2",
        ],
        "GeneratedBy": [{"Name": "Analysis/bidsify.py",
                         "Description": "Converts the Unity rig's per-frame CSV logs and "
                                        "QUEST+ calibration logs to BIDS."}],
        "SourceDatasets": [{"URL": "bids::sourcedata/"}],
    })

    (bids_dir / "README").unlink(missing_ok=True)      # replaced by README.md
    where = ", ".join(x for x in (C.INSTITUTION_DEPARTMENT, C.INSTITUTION_NAME) if x)
    credits = f"\n## Credits\n\nDataset by {', '.join(C.AUTHORS)}"
    credits += f" ({where})" if where else ""
    credits += f", supervised by {' and '.join(C.SUPERVISORS)}.\n" if C.SUPERVISORS else ".\n"
    (bids_dir / "README.md").write_text(_README + credits, encoding="utf-8")

    # custom files for the validator to skip
    (bids_dir / ".bidsignore").write_text(
        "# Custom (non-schema) files included on purpose:\n"
        f"task-{C.TASK_LABEL}_stimulus.json\n", encoding="utf-8")

    pdf = _consolidate_participants(participants)
    if not pdf.empty:
        order = [c for c in _PARTICIPANT_COLUMNS if c in pdf.columns]
        pdf = pdf[order + [c for c in pdf.columns if c not in order]]
    pdf.to_csv(bids_dir / "participants.tsv", sep="\t", index=False, na_rep="n/a")
    _dump_json(bids_dir / "participants.json", {
        "participant_id": {"Description": "Participant identifier (p## on the rig)."},
        "counterbalance_slot": {"Description": "Slot in the pre-generated permuted-block "
                                "condition-order allocation (ExperimentConductor.ALLOCATION). "
                                "Usually the participant number, but re-assigned to re-fill "
                                "the slot of an excluded participant. The per-visit order "
                                "itself is in scans.tsv (it crosses over between visits)."},
        "threshold_standing_deg": {"Description": "Tilt giving ~75 % correct, from the "
                                   "standing calibration of the latest visit (each visit's "
                                   "calibration is in ses-01/beh).", "Units": "degree"},
        "easy_standing_deg": {"Description": "Tilt giving ~85 % correct, standing.", "Units": "degree"},
        "hard_standing_deg": {"Description": "Tilt giving ~65 % correct, standing.", "Units": "degree"},
        "threshold_walking_deg": {"Description": "Tilt giving ~75 % correct, from the "
                                  "walking calibration of the latest visit (each visit's "
                                  "calibration is in ses-01/beh).", "Units": "degree"},
        "easy_walking_deg": {"Description": "Tilt giving ~85 % correct, walking.", "Units": "degree"},
        "hard_walking_deg": {"Description": "Tilt giving ~65 % correct, walking.", "Units": "degree"},
        "treadmill_speed_mps": {"Description": "Median belt speed while walking.", "Units": "m/s"},
        "threshold_angle_deg": {"Description": "Same as threshold_walking_deg (standing if "
                                "there is no walking calibration); kept for older "
                                "analysis scripts.", "Units": "degree"},
        "easy_angle_deg": {"Description": "Same as easy_walking_deg; kept for older scripts.",
                           "Units": "degree"},
        "hard_angle_deg": {"Description": "Same as hard_walking_deg; kept for older scripts.",
                           "Units": "degree"},
    })


# ===========================================================================
# Discovery and conversion
# ===========================================================================
def _cond_from_name(name: str) -> str:
    """Condition of an archived file from its acq- entity or _beh."""
    m = re.search(r"_acq-(walking|standing)", name)
    if m:
        return m.group(1)
    return "calibration" if "_beh" in name else "motion"


def _run_from_name(name: str) -> str | None:
    """``run-<index>`` entity of an archived file, or None if it carries no run-."""
    m = re.search(r"_(run-\d+)", name)
    return m.group(1) if m else None


def _discover_inputs(raw_dir: Path, bids_dir: Path) -> list[dict]:
    """Find raw inputs: loose CSVs (preferred) or sourcedata archives.

    Items are keyed by (sub, ses, datatype, condition, run), so walking and standing and
    all blocks of a condition stay separate. A loose CSV replaces the matching archive.
    """
    found: dict[tuple, dict] = {}
    sd = bids_dir / "sourcedata"
    # Archives first (sub/ses/datatype from the path, run from the name; .csv.gz or legacy
    # .csv.zip).
    archives = sorted(list(sd.glob("sub-*/ses-*/*/*.csv.gz"))
                      + list(sd.glob("sub-*/ses-*/*/*.csv.zip")))
    for a in archives:
        sub, ses, datatype = a.parts[-4], a.parts[-3], a.parts[-2]
        cond = _cond_from_name(a.name)
        run = _run_from_name(a.name)
        found[(sub, ses, datatype, cond, run)] = {"source": a, "sub": sub, "ses": ses,
                                                  "datatype": datatype, "archived": True,
                                                  "condition": cond, "run": run}
    # loose logs (.csv or .csv.gz) override archives
    loose = []
    for raw in list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.csv.gz")):
        ids = C.parse_raw_name(raw.name)
        if ids:
            loose.append((raw, ids))
    # Process originals before re-runs (no -N suffix first), visit 1 before visit 2.
    loose.sort(key=lambda it: (it[1]["participant"], it[1]["visit"] or 0, it[1]["block"],
                               it[1]["trial"], it[1]["dup"] or 0, it[0].name))
    from_loose: set[tuple] = set()
    for raw, ids in loose:
        sub, ses = C.subject_for(ids), C.session_for(ids)
        datatype = "beh" if ids["is_calibration"] else "motion"
        cond = ids["condition"]
        run = C.run_for(ids)
        key = (sub, ses, datatype, cond, run)
        prev = found.get(key)
        if prev is not None and (key in from_loose
                                 or not _same_content(raw, prev["source"])):
            # Another block already has this run (reused Trial number, rig wrote a "-1"
            # file; or visit 2 repeated a visit-1 number). Assign a separate run index
            # outside 1..8 so neither block is lost.
            run = _free_run(found, key[:4], ids["trial"])
            print(f"  WARNING: {raw.name} maps to the same {key[4] or 'run'} as "
                  f"{prev['source'].name}; stored as {run} instead. Check the session "
                  "log for which block this is.")
            key = (sub, ses, datatype, cond, run)
        found[key] = {"source": raw, "sub": sub, "ses": ses, "datatype": datatype,
                      "archived": False, "condition": cond, "run": run}
        from_loose.add(key)
    return list(found.values())


def _same_content(a: Path, b: Path) -> bool:
    """True if two raw logs hold the same frames, whether or not either is gzipped."""
    if filecmp.cmp(a, b, shallow=False):
        return True

    def payload(p: Path) -> bytes | None:
        try:
            return gzip.decompress(p.read_bytes()) if p.name.endswith(".gz") else p.read_bytes()
        except (EOFError, OSError, zlib.error):
            return None

    pa = payload(a)
    return pa is not None and pa == payload(b)


def _free_run(found: dict, key4: tuple, trial: int) -> str:
    """First unused run label of the form ``run-<100k + trial>`` for this key."""
    k = 1
    while (*key4, f"run-{100 * k + trial:03d}") in found:
        k += 1
    return f"run-{100 * k + trial:03d}"


def _stem_for(sub: str, ses: str, datatype: str, condition: str | None = None,
              run: str | None = None) -> str:
    if datatype == "beh":
        # acq- for per-condition calibrations (none for legacy), run- per visit
        acq = C.acq_for(condition)
        ent = f"_acq-{acq}" if acq else ""
        run_tag = f"_{run}" if run else ""
        return f"{sub}_{ses}_task-{C.TASK_LABEL}{ent}{run_tag}_beh"
    # motion archive name = BIDS motion base
    return f"{_motion_base(sub, ses, condition, run)}_motion"


def convert_all(raw_dir: Path | None = None, bids_dir: Path | None = None,
                prune_loose: bool = False) -> None:
    """Convert all raw inputs into the BIDS dataset (pipeline entry point).

    Reads loose CSVs in ``raw_dir`` and archives in ``sourcedata/``, so re-running is
    idempotent and the dataset can be rebuilt from its archive. New logs are archived
    (gzipped, BIDS-named); afterwards loose originals are moved to ``sourcedata/raw/``
    (``_move_original``) or deleted with ``prune_loose=True``.

    Also writes ``dataset_description.json``, ``README.md``, ``participants.tsv``/``.json``,
    per-subject ``sessions.tsv`` and ``scans.tsv``, and the derivatives description.
    """
    raw_dir = raw_dir or C.RAW_DIR
    bids_dir = bids_dir or C.BIDS_DIR
    inputs = _discover_inputs(raw_dir, bids_dir)
    if not inputs:
        print(f"No raw inputs found in {raw_dir} or {bids_dir/'sourcedata'}")
        return

    print(f"Converting {len(inputs)} input(s) -> {bids_dir}")
    participants: list[dict] = []
    sessions: dict[str, list[dict]] = defaultdict(list)
    loose_to_remove: list[Path] = []
    skipped: list[str] = []

    for item in inputs:
        sub, ses, datatype = item["sub"], item["ses"], item["datatype"]
        src = item["source"]               # .csv / .csv.gz / .csv.zip
        run = item.get("run")              # run- index; None for single/legacy
        # condition from the data for motion, from the file name for calibrations
        cond = ((_peek_condition(src) or item.get("condition")) if datatype == "motion"
                else item.get("condition"))
        stem = _stem_for(sub, ses, datatype, cond, run)
        if item["archived"]:
            tag = "converted (from sourcedata)"
            # rename archives with old names (content unchanged)
            canonical = src.with_name(f"{stem}.csv.gz")
            if src.name.endswith(".csv.gz") and src != canonical and not canonical.exists():
                src = src.rename(canonical)
                tag += ", renamed to current naming"
        else:
            tag = "archived+converted"
        print(f"- {src.name}  [{tag}]")
        try:
            if datatype == "beh":
                srow = convert_calibration(src, sub, ses, bids_dir, cond, run)
            else:
                prow, srow = convert_motion_file(src, sub, ses, bids_dir, run)
                participants.append(prow)
        except (UnusableRawLog, EOFError, OSError, zlib.error,
                pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            # Skip broken logs; the original is still moved to sourcedata/raw/.
            print(f"  SKIPPED - no usable data: {e}")
            skipped.append(f"{src.name}: {e}")
            if not item["archived"]:
                loose_to_remove.append(src)
            continue
        sessions[sub].append(srow)
        if not item["archived"]:           # archive new loose log
            _archive_to_sourcedata(src, stem, sub, ses, datatype, bids_dir)
            loose_to_remove.append(src)

    # Add calibration thresholds to the participant rows: per-condition columns and a
    # generic threshold_angle_deg (walking, else standing/legacy) for older readers such as
    # descriptive.py.
    for sub, sess in sessions.items():
        cal = {s.get("condition"): s["_thresholds"] for s in sess
               if s.get("datatype") == "beh" and s.get("_thresholds")
               and "threshold_deg" in s["_thresholds"]}
        generic = (cal.get("walking") or cal.get("standing")
                   or cal.get("calibration") or (next(iter(cal.values())) if cal else None))
        for prow in participants:
            if prow["participant_id"] != sub:
                continue
            for c, thr in cal.items():
                if c in ("standing", "walking"):
                    prow[f"threshold_{c}_deg"] = round(thr["threshold_deg"], 4)
                    prow[f"easy_{c}_deg"] = round(thr.get("easy_deg", float("nan")), 4)
                    prow[f"hard_{c}_deg"] = round(thr.get("hard_deg", float("nan")), 4)
            if generic:
                prow["threshold_angle_deg"] = round(generic["threshold_deg"], 4)
                prow["easy_angle_deg"] = round(generic.get("easy_deg", float("nan")), 4)
                prow["hard_angle_deg"] = round(generic.get("hard_deg", float("nan")), 4)

    for sub, sess in sessions.items():
        _write_sessions_tsv(bids_dir, sub, sess)
        _write_scans_tsv(bids_dir, sub, sess)
    if not participants:
        participants = [{"participant_id": sub} for sub in sessions]
    _write_dataset_level(bids_dir, participants)
    _write_stim_sidecar(bids_dir)
    _write_derivatives_description(bids_dir)

    if prune_loose and loose_to_remove:
        for p in loose_to_remove:
            p.unlink()
        print(f"Archived to sourcedata and removed {len(loose_to_remove)} loose CSV(s) "
              f"from {raw_dir}")
    elif loose_to_remove:
        moved = [d for d in (_move_original(p, bids_dir) for p in loose_to_remove) if d]
        print(f"Moved {len(moved)} original log(s) unchanged into "
              f"{bids_dir / 'sourcedata' / 'raw'}")
    if skipped:
        print(f"WARNING: {len(skipped)} log(s) held no usable data and were not converted:")
        for msg in skipped:
            print(f"  - {msg}")
    print(f"Done. Dataset root: {bids_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg)
            ids = C.parse_raw_name(p.name)
            if ids is None:
                print(f"Skip (unrecognised name): {p}")
                continue
            C.BIDS_DIR.mkdir(parents=True, exist_ok=True)
            sub, ses = C.subject_for(ids), C.session_for(ids)
            datatype = "beh" if ids["is_calibration"] else "motion"
            cond = ids["condition"]
            run = C.run_for(ids)
            _archive_to_sourcedata(p, _stem_for(sub, ses, datatype, cond, run), sub, ses, datatype, C.BIDS_DIR)
            if ids["is_calibration"]:
                convert_calibration(p, sub, ses, C.BIDS_DIR, cond, run)
            else:
                convert_motion_file(p, sub, ses, C.BIDS_DIR, run)
    else:
        convert_all()
