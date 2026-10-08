"""Detection and repair of head-position spikes and tracking dropouts.

Optical tracking sometimes loses the head (marker swap, leaving the capture volume) and
produces sharp spikes or jumps. A spike has energy at all frequencies, so band-passing
spreads it as ringing across the gait band. Spikes must therefore be detected and
repaired before filtering.

Method
------
* Detect. A sample is bad if any of these hold:
    - non-finite (NaN/Inf);
    - Hampel outlier: far from the rolling median relative to the rolling MAD (no
      assumption about the gait waveform);
    - beyond an optional absolute radius from the global median (dropout values, e.g. the
      head jumping tens of cm);
    - isolated velocity spike: the jump into and out of the sample both exceed a
      plausible head speed.
* Repair. Bad runs shorter than ``max_interp_s`` are linearly interpolated so filtering
  does not ring around them. Longer runs are set to NaN, so no stride phase is invented
  across a real dropout.
* Report. Fraction flagged, fraction excised, clean coverage and longest clean span, to
  decide whether a recording is usable.

Use in the gait analysis and in QC
----------------------------------
``gait.py`` (``_walking_segment``) cleans ``head_local_pos_y`` / ``_x`` before resampling
and band-passing; ``qc.py`` reports the result per recording. Brief dropouts can lie inside
the 2 m radius of ``qc._valid_pos_mask``, and once band-passed (0.5-5 Hz) such a spike
rings at many times the bob amplitude, which the trough detector would follow.

Safeguards:
  * Thresholds are far outside physiological values: >0.30 m vertical excursion or
    >5 m/s head speed, against a bob of ~0.02 m at ~0.2 m/s (a 15-75x margin). The Hampel
    test also needs a deviation above a ~0.10 m floor, so the bob is never flagged however
    small the local MAD. Unit tests (``tests/test_pipeline.py``) check that a clean bob is
    unchanged and injected spikes are removed.
  * Only long dropouts are excised. A bridged long gap is flat (no troughs), so data is
    lost rather than oscillation invented. A ``severe`` verdict (clean coverage < 60 %)
    is a warning in report.py, not an automatic exclusion.

Cleaning repairs the gait signal only; it says nothing about the usability of a block's
responses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C


# Head channels cleaned by default, with the maximum plausible distance (m) from the
# median per axis. The bob is a few cm and sway slightly more; larger values are artefacts.
HEAD_CLEAN_AXES = {
    "head_local_pos_y": 0.30,   # vertical bob
    "head_local_pos_x": 0.40,   # lateral sway
    "head_local_pos_z": 0.40,   # fore-aft
}


def _hampel_mask(v: np.ndarray, win: int, k: float, floor_m: float) -> np.ndarray:
    """Conservative Hampel outlier mask.

    A sample is an outlier if its deviation from the rolling median exceeds both
    k * 1.4826 * rolling MAD and ``floor_m``. Without the floor, a small local MAD on a
    smooth signal would flag the bob peaks; with it, the ~2 cm bob is never flagged while
    spikes of tens of cm are.
    """
    s = pd.Series(v)
    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median()
    sigma = 1.4826 * mad.to_numpy()
    dev = np.abs(s.to_numpy() - med.to_numpy())
    return (dev > k * sigma) & (dev > floor_m)


def detect_outliers(t: np.ndarray, v: np.ndarray, *, med_win_s: float = 0.4,
                    mad_k: float = 6.0, abs_radius_m: float | None = None,
                    local_excursion_m: float | None = None,
                    max_speed_mps: float = C.MAX_PLAUSIBLE_SPEED_MPS) -> np.ndarray:
    """Boolean mask of implausible samples in one position channel.

    ``local_excursion_m`` is the Hampel floor (see ``_hampel_mask``); default a third of
    ``abs_radius_m`` (~0.10 m for the 0.30 m vertical radius), above the bob and below spikes.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    n = v.size
    bad = ~np.isfinite(v)
    finite = np.isfinite(v)
    if finite.sum() >= 3:
        floor = local_excursion_m
        if floor is None:
            floor = 0.33 * abs_radius_m if abs_radius_m is not None else 0.10
        dt = np.diff(t)
        pos_dt = dt[dt > 0]
        fs = (1.0 / np.median(pos_dt)) if pos_dt.size else C.NOMINAL_FS
        win = max(3, int(round(med_win_s * fs)))
        if win % 2 == 0:
            win += 1
        hm = np.zeros(n, bool)
        hm[finite] = _hampel_mask(v[finite], win, mad_k, floor)
        bad |= hm
        if abs_radius_m is not None:
            bad |= np.abs(v - np.nanmedian(v)) > abs_radius_m
        # Isolated velocity spike: fast jump into and out of one sample.
        with np.errstate(invalid="ignore", divide="ignore"):
            sp = np.abs(np.diff(v)) / np.where(dt > 0, dt, np.nan)
        sp = np.nan_to_num(sp, nan=0.0, posinf=np.inf)
        left = np.concatenate([[0.0], sp])      # speed from previous sample
        right = np.concatenate([sp, [0.0]])     # speed to next sample
        bad |= (left > max_speed_mps) & (right > max_speed_mps)
    return bad


def repair(t: np.ndarray, v: np.ndarray, bad: np.ndarray, *,
           max_interp_s: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate short bad runs, excise (NaN) long ones.

    Returns (v_clean, repaired_mask, excised_mask).
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float).copy()
    n = v.size
    repaired = np.zeros(n, bool)
    excised = np.zeros(n, bool)
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        left, right = i - 1, j                       # good samples on either side
        bounded = left >= 0 and right < n
        gap_s = (t[right] - t[left]) if bounded else np.inf
        if bounded and gap_s <= max_interp_s:
            v[i:j] = np.interp(t[i:j], [t[left], t[right]], [v[left], v[right]])
            repaired[i:j] = True
        else:
            v[i:j] = np.nan                          # real dropout: leave NaN
            excised[i:j] = True
        i = j
    return v, repaired, excised


def _longest_clean_run_s(t: np.ndarray, clean_mask: np.ndarray) -> float:
    best, i, n = 0.0, 0, len(clean_mask)
    while i < n:
        if not clean_mask[i]:
            i += 1
            continue
        j = i
        while j < n and clean_mask[j]:
            j += 1
        best = max(best, float(t[j - 1] - t[i]))
        i = j
    return best


def clean_channel(t: np.ndarray, v: np.ndarray, *, abs_radius_m: float | None = None,
                  max_interp_s: float = 0.20, **detect_kw):
    """Detect and repair one channel. Returns (v_clean, bad, repaired, excised, info)."""
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    bad = detect_outliers(t, v, abs_radius_m=abs_radius_m, **detect_kw)
    v_clean, repaired, excised = repair(t, v, bad, max_interp_s=max_interp_s)
    n = v.size
    clean_mask = np.isfinite(v_clean)                # good or interpolated
    med = float(np.nanmedian(v))
    exc = np.abs(v - med)
    info = {
        "n": int(n),
        "n_flagged": int(bad.sum()),
        "pct_flagged": round(100.0 * bad.sum() / n, 2) if n else 0.0,
        "n_interpolated": int(repaired.sum()),
        "n_excised": int(excised.sum()),
        "clean_coverage_pct": round(100.0 * clean_mask.sum() / n, 2) if n else 0.0,
        "longest_clean_run_s": round(_longest_clean_run_s(t, clean_mask), 2),
        "max_excursion_m": round(float(np.nanmax(exc[bad])), 3) if bad.any() else 0.0,
    }
    return v_clean, bad, repaired, excised, info


def _verdict(coverage: float, excised_pct: float) -> str:
    """Usability verdict from clean coverage (good + repaired) and excised fraction.

    The raw flag count is not used: interpolated jitter is harmless, and clean runs with
    light OptiTrack jitter still flag a few %.
    """
    if coverage >= 99.0 and excised_pct < 1.0:
        return "clean (jitter repaired)"
    if coverage >= 90.0:
        return "salvageable (spikes repaired/excised)"
    if coverage >= 60.0:
        return "degraded but salvageable"
    return "severe - too little clean data"


def clean_motion_positions(motion: pd.DataFrame, axes: dict | None = None):
    """Clean the head-position channels of a motion table.

    Returns (cleaned_cols, per_channel_info, aggregate, bad_any, excised_any). The
    aggregate uses the worst axis (max flagged, min coverage).
    """
    axes = axes or HEAD_CLEAN_AXES
    t = motion["t"].to_numpy()
    n = len(t)
    cleaned: dict[str, np.ndarray] = {}
    per: dict[str, dict] = {}
    bad_any = np.zeros(n, bool)
    excised_any = np.zeros(n, bool)
    for ch, radius in axes.items():
        if ch not in motion.columns:
            continue
        vc, bad, rep, exc, info = clean_channel(t, motion[ch].to_numpy(), abs_radius_m=radius)
        cleaned[ch] = vc
        per[ch] = info
        bad_any |= bad
        excised_any |= exc
    if per:
        pct = max(i["pct_flagged"] for i in per.values())
        cov = min(i["clean_coverage_pct"] for i in per.values())
        longest = min(i["longest_clean_run_s"] for i in per.values())
        max_exc = max(i["max_excursion_m"] for i in per.values())
    else:
        pct, cov, longest, max_exc = 0.0, 100.0, 0.0, 0.0
    excised_pct = round(100.0 * excised_any.sum() / n, 2) if n else 0.0
    aggregate = {
        "pct_flagged": pct,
        "clean_coverage_pct": cov,
        "excised_pct": excised_pct,
        "n_excised": int(excised_any.sum()),
        "longest_clean_run_s": longest,
        "max_excursion_m": max_exc,
        "verdict": _verdict(cov, excised_pct),
    }
    return cleaned, per, aggregate, bad_any, excised_any


# Fallback freeze detection: minimum duration of repeated head positions counted as frozen.
# Positions are logged to 0.1 mm, so single repeats also occur with a still head.
STALE_MIN_RUN_S = 0.10
# mocap_frame_advanced is trusted only if at least this share of rows is fresh. Working
# sessions have 96-99 %, an unrecorded flag ~0 %. A stall long enough to drop below this
# is still caught by the repeated-pose fallback.
MIN_FRESH_FLAG_FRAC = 0.5
HEAD_POS = ("head_local_pos_x", "head_local_pos_y", "head_local_pos_z")


def stale_frames(motion: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    """Rows with a frozen head pose, and the method used to find them.

    Normally ``mocap_frame_advanced == 0``. Some sessions log the mocap channels as zeros
    (the pose reached the render nodes, but the OptiTrack channels were not recorded);
    taken literally, the whole block would be stale. In that case runs of
    identical head positions of at least ``STALE_MIN_RUN_S`` are used, which on sessions
    with a working flag finds every flagged frame. Returns ``(None, "none")`` if neither
    source exists.
    """
    adv = motion.get("mocap_frame_advanced")
    if adv is not None and np.mean(adv.to_numpy() == 1) >= MIN_FRESH_FLAG_FRAC:
        return adv.to_numpy() == 0, "mocap_frame_advanced"
    cols = [c for c in HEAD_POS if c in motion]
    if not cols or "t" not in motion:
        return None, "none"
    p = motion[cols].to_numpy(float)
    t = motion["t"].to_numpy(float)
    rep = np.r_[False, np.all(np.diff(p, axis=0) == 0, axis=1)]
    stale = np.zeros(rep.size, bool)
    # Repeated rows i..j-1 share the pose of row i-1 (last fresh), so the frozen span is
    # t[i-1]..t[j-1] plus one frame.
    edges = np.diff(np.r_[0, rep.astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if t[min(b, t.size - 1)] - t[a - 1] >= STALE_MIN_RUN_S:
            stale[a:b] = True
    return stale, "repeated_position"
