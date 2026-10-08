"""Stride-cycle extraction from vertical head motion, after Davidson et al. (2024).

Follows the gait preprocessing of Davidson, Verstraten & Alais (2024, Nat Commun
15:2027):

  1. Vertical head position (``head_local_pos_y``) over the walking segment is resampled
     onto a uniform grid (true timestamps) and band-passed (0.5-5 Hz zero-phase
     Butterworth; Davidson used raw HMD position at 90 Hz, the filter reduces OptiTrack
     jitter and drift before trough detection).
  2. Troughs in vertical position mark double support. Consecutive troughs bound a step,
     two steps a stride, so head height oscillates twice per stride (2 cps).
  3. Each step is normalised to 50 % of the stride (Davidson: 200 points per stride);
     an event's phase is its position in the step (% step) and stride (% stride).
  4. A stride template (mean detrended head height) and stride durations are produced
     (Fig 1d / 3b).

Phase origin: 0 % = a double-support trough. Head height alone does not show which foot
is in stance, leaving a half-stride ambiguity. This does not affect the confirmatory test
(a half-stride shift is one cycle at 2 cps and two at 4 cps), but it does affect the odd
harmonics (1, 3 cps), which carry left/right asymmetry, and any foot comparison.

Foot identity therefore comes from lateral sway, which oscillates once per stride as the
head and trunk shift toward the stance limb (:func:`label_steps`). The per-step evidence
is decoded under an alternation constraint, head roll serves as an independent check,
and the mapping to anatomical left/right requires ``config.LATERAL_SIGN_VERIFIED``. Used
by ``stance_foot.py``.

Usage::

    python -m Analysis.gait
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
import numpy as np
from scipy.signal import butter, filtfilt, detrend, find_peaks

try:
    from . import config as C
    from . import bids_io
    from . import clean
    from . import phaser
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import clean
    import phaser
    import plotstyle as S

S.apply()

RESAMPLE_FS = 90.0          # uniform grid (Davidson's HMD rate)
LOWPASS_HZ = 5.0            # upper band-pass edge before trough detection
GAIT_BAND_HZ = (0.5, 4.0)   # plausible step-rate band (dominant frequency)
MIN_STEP_INTERVAL_S = 0.30  # minimum interval between troughs
STRIDE_POINTS = 200         # points per stride, as Davidson (0.5 %)

# --- data missing inside a block ---------------------------------------------
# Head data can be missing (excised dropout, frozen frames, logger hitch, belt stop). The
# resampler bridges gaps with a straight line, so no phase is read there. Troughs and
# Hilbert phase within GAP_PAD_S of a gap are discarded (filter settling), and a step longer
# than MAX_STEP_RATIO x the median (missed trough, or troughs on both sides of a gap) gets
# no phase. Treadmill step CoV is a few %, so 1.6x only catches such cases.
GAP_S = 0.25                # larger sample spacing = gap
GAP_PAD_S = 0.5             # discarded on each side of a gap
MAX_STEP_RATIO = 1.6        # longest valid step relative to the median

# --- stance-foot labelling ---------------------------------------------------
SWAY_BAND_HZ = (0.3, 3.0)      # around the ~0.9 Hz once-per-stride sway
ROLL_BAND_HZ = (0.3, 3.0)      # head roll, same cycle
# Candidate orientation channels for the roll check, in order of preference. Mocap Euler
# angles first: the Unity head Transform may be yaw-only (known issue in
# ILLLoggerController), so the 6-DOF pose is logged from the streaming client. Which mocap
# axis is roll depends on the Motive rigid-body definition, so _pick_roll selects it by
# frequency content.
ROLL_CHANNELS = ("head_mocap_ori_z", "head_mocap_ori_x", "head_local_ori_z")
ROLL_CLEAN_DEG = 25.0          # larger |roll - median| is an artefact
ROLL_MAX_RATE_DPS = 600.0      # velocity limit for roll spikes
# Decoder settings (see _viterbi_alternate): no single step can break the alternation,
# but a sustained parity shift is cheap. An isolated flip costs 2*log(p/(1-p)) = 15.2,
# while one step can contribute at most 2*EMISSION_GAIN*MARGIN_CLIP = 12; a sustained
# shift pays 7.6 once.
ALT_PRIOR = 0.9995             # prior P(next step is the other foot)
EMISSION_GAIN = 3.0            # weight of a unit-margin step
MARGIN_CLIP = 2.0              # cap on |per-step evidence| (MAD units)
AMBIGUOUS_MARGIN = 0.25        # |margin| below this = weak evidence
ROLL_FUSE_MIN_AGREE = 0.70     # min agreement to fuse roll into the evidence


# ===========================================================================
# Phase helper (also used by phase_analysis.py)
# ===========================================================================
def phase_at(times: np.ndarray, troughs: np.ndarray,
             step_foot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Map event times to (% step, % stride) given double-support trough times.

    Each step (trough[i] to trough[i+1]) is normalised to 50 % of the stride, as in
    Davidson.

    ``step_foot`` (0/1 per step, from lateral sway) sets which half of the stride a step
    maps to; without it the step index parity (i % 2) is used, leaving the half-stride
    ambiguity. NaN outside the trough range and inside over-long steps
    (``MAX_STEP_RATIO``).
    """
    times = np.asarray(times, float)
    troughs = np.asarray(troughs, float)
    pct_step = np.full(times.shape, np.nan)
    pct_stride = np.full(times.shape, np.nan)
    if troughs.size < 2:
        return pct_step, pct_stride
    max_step = MAX_STEP_RATIO * float(np.median(np.diff(troughs)))
    idx = np.searchsorted(troughs, times) - 1   # step i: troughs[i] <= t < troughs[i+1]
    inside = (idx >= 0) & (idx < troughs.size - 1)
    i = idx[inside]
    width = troughs[i + 1] - troughs[i]
    keep = ~(width > max_step)
    j = np.flatnonzero(inside)[keep]
    i, width = i[keep], width[keep]
    frac = (times[j] - troughs[i]) / width
    half = i % 2
    if step_foot is not None and len(step_foot):
        sf = np.asarray(step_foot)
        labelled = i < len(sf)
        half = np.where(labelled, sf[np.minimum(i, len(sf) - 1)].astype(int), half)
    pct_step[j] = 100.0 * frac
    pct_stride[j] = 50.0 * half + 50.0 * frac
    return pct_step, pct_stride


# ===========================================================================
# Hilbert phase (b3)
# ===========================================================================
def continuous_stride_phase(tg: np.ndarray, y_filt: np.ndarray,
                            x_filt: np.ndarray | None = None,
                            troughs: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Continuous (unwrapped) stride phase in radians, 2π per stride, via Hilbert.

    Alternative to ``phase_at`` that does not depend on trough detection.

      * With lateral sway (once per stride) its analytic phase is used directly, with no
        half-stride ambiguity.
      * Otherwise the vertical signal (twice per stride) is used with its phase halved,
        which reintroduces the ambiguity.

    With ``troughs`` the first trough is set to 0, as in ``phase_at``. Returns
    (continuous_phase_rad, source).
    """
    from scipy.signal import hilbert
    y_filt = np.asarray(y_filt, float)
    if x_filt is not None and np.size(x_filt) and np.any(np.asarray(x_filt)):
        cont = np.unwrap(np.angle(hilbert(np.asarray(x_filt, float))))     # 1 cycle per stride
        source = "lateral"
    else:
        cont = np.unwrap(np.angle(hilbert(y_filt))) / 2.0                  # 2 cycles per stride, halved
        source = "vertical"
    if troughs is not None and np.size(troughs):
        i0 = int(np.clip(np.searchsorted(tg, troughs[0]), 0, cont.size - 1))
        cont = cont - cont[i0]
    return cont, source


def phase_at_hilbert(times: np.ndarray, tg: np.ndarray,
                     cont_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map event times to (% step, % stride) from the continuous Hilbert phase.

    Interpolates the unwrapped phase and wraps afterwards. % step runs twice as fast as
    % stride.
    """
    times = np.asarray(times, float)
    uph = np.interp(times, np.asarray(tg, float), np.asarray(cont_phase, float),
                    left=np.nan, right=np.nan)
    pct_stride = np.mod(uph, 2 * np.pi) / (2 * np.pi) * 100.0
    pct_step = np.mod(2 * uph, 2 * np.pi) / (2 * np.pi) * 100.0
    return pct_step, pct_stride


@dataclass
class GaitResult:
    """Gait results of one recording.

    Scalars (``summary()``) go to JSON; arrays are saved as ``.npz`` for the phase
    analyses. Both ``trough_times`` (trough phase) and ``stride_phase_cont`` (Hilbert
    phase) are kept; ``hilbert_trough_phase_corr`` gives their agreement.
    """
    recording: str
    walking: bool
    dominant_freq_hz: float | None
    n_steps: int
    step_duration_s: float | None
    step_cov_pct: float | None
    stride_duration_s: float | None
    cadence_spm: float | None        # steps per minute
    lateral_sway_amp_m: float | None = None   # mean |per-step lateral sway| (m)
    lr_alternation_pct: float | None = None    # % alternating consecutive steps (raw)
    # --- stance-foot labels (lateral sway, head roll, alternation constraint) ---
    lr_alternation_constrained_pct: float | None = None  # alternation after HMM decoding
    n_steps_relabelled: int | None = None      # steps changed by the decoder
    ambiguous_step_pct: float | None = None    # % steps with |margin| below threshold
    roll_channel: str | None = None            # orientation channel used as roll
    roll_stride_step_ratio: float | None = None  # its stride / step power ratio
    roll_agreement_pct: float | None = None    # roll vs sway label agreement (%)
    roll_polarity: int | None = None           # sign between roll and sway
    roll_fused: bool | None = None             # roll fused into the evidence
    foot_label_verdict: str | None = None      # reliable / usable (check) / unreliable
    foot_label_anatomical: bool | None = None  # left/right naming verified
    # --- sampling / filter QC (b4) ---
    effective_fs_hz: float | None = None       # median 1/dt of the raw walking segment
    dropped_frame_pct: float | None = None     # % sample intervals > 1.5x nominal
    raw_segment_samples: int | None = None
    hilbert_phase_source: str | None = None    # 'lateral' or 'vertical'
    hilbert_trough_phase_corr: float | None = None  # Hilbert vs trough phase agreement
    phaser_trough_phase_corr: float | None = None   # multi-channel vs trough phase agreement
    phaser_channels: dict | None = None        # per-channel weight and concentration (phaser.py)
    # --- spike cleaning (before resampling and band-pass) ---
    clean_flagged_pct: float | None = None     # % vertical samples flagged
    clean_excised_pct: float | None = None     # % excised (dropouts, not interpolated)
    clean_coverage_pct: float | None = None    # % good or interpolated
    clean_max_spike_m: float | None = None     # largest excursion removed (m)
    # --- gaps in the head data within the walking segment (GAP_S) ---
    n_data_gaps: int | None = None             # number of gaps
    data_gap_total_s: float | None = None      # total gap length (s)
    # arrays (not in JSON):
    t_grid: np.ndarray = None
    y_filt: np.ndarray = None        # band-passed vertical head height (m)
    x_filt: np.ndarray = None        # band-passed lateral sway (m)
    trough_times: np.ndarray = None
    step_foot: np.ndarray = None     # stance foot per step (0 = +sway foot)
    step_margin: np.ndarray = None   # signed per-step evidence (> 0: reference foot)
    roll_filt: np.ndarray = None     # band-passed head roll (deg)
    stride_template: np.ndarray = None   # mean head height over the stride (200 points)
    stride_template_sd: np.ndarray = None
    stride_durations: np.ndarray = None
    stride_phase_cont: np.ndarray = None  # Hilbert stride phase (rad) on t_grid (b3)
    stride_phase_phaser: np.ndarray = None  # multi-channel stride phase (rad) on t_grid

    def summary(self) -> dict:
        """Scalar fields for JSON (arrays dropped)."""
        return {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray)}


# ===========================================================================
# Preprocessing and trough detection
# ===========================================================================
def _stale_run_stats(t: np.ndarray, stale: np.ndarray) -> dict:
    """Count, percentage and longest run of frozen-pose samples."""
    n = int(stale.sum())
    out = {"n_stale": n, "pct_stale": round(100.0 * n / max(1, stale.size), 2),
           "longest_run_s": 0.0, "longest_run_start_s": None}
    best_len = best_start = best_n = 0
    i = 0
    while i < stale.size:
        if stale[i]:
            j = i
            while j < stale.size and stale[j]:
                j += 1
            dur = float(t[min(j, t.size - 1)] - t[i])
            if dur > best_len:
                best_len, best_start, best_n = dur, float(t[i] - t[0]), j - i
            i = j
        else:
            i += 1
    out["longest_run_s"] = round(best_len, 2)
    out["longest_run_start_s"] = round(best_start, 1) if best_n else None
    out["longest_run_n"] = best_n
    return out


def _walking_segment(rec):
    """Walking-segment (t, vertical_y, lateral_x, roll candidates, is_walking, motion,
    clean_info).

    Head position is spike-cleaned (``clean.py``) before filtering; band-passed spikes
    become ringing that the trough detector would follow. Thresholds are 15-75x above
    the real bob.

    Cleaning runs on the full series so the rolling median has neighbours; then the moving
    segment is selected. Short spikes are interpolated; long dropouts become NaN and are
    dropped by the resampler (a bridged gap has no troughs). ``clean_info`` holds
    per-channel statistics.
    """
    motion, _, _ = bids_io.load_motion(rec)
    t = motion["t"].to_numpy()
    speed = motion.get("treadmill_speed")
    if speed is not None:
        s = speed.to_numpy()
        moving = (s > 0.05) & (s < C.MAX_PLAUSIBLE_SPEED_MPS)
    else:
        moving = np.ones(len(motion), bool)
    is_walking = bool(moving.mean() > 0.2)

    clean_info: dict = {}
    yc = xc = None
    roll_cands: dict[str, np.ndarray] = {}
    if "head_local_pos_y" in motion:
        yc, _, _, _, clean_info["y"] = clean.clean_channel(
            t, motion["head_local_pos_y"].to_numpy(),
            abs_radius_m=clean.HEAD_CLEAN_AXES["head_local_pos_y"])
    if "head_local_pos_x" in motion:
        xc, _, _, _, clean_info["x"] = clean.clean_channel(
            t, motion["head_local_pos_x"].to_numpy(),
            abs_radius_m=clean.HEAD_CLEAN_AXES["head_local_pos_x"])
    for name in ROLL_CHANNELS:
        if name not in motion:
            continue
        # Roll candidates are cleaned with thresholds in degrees; clean.py defaults are in
        # metres and would flag normal roll as spikes.
        roll_cands[name], _, _, _, clean_info[f"roll:{name}"] = clean.clean_channel(
            t, _signed_deg(motion[name].to_numpy()),
            abs_radius_m=ROLL_CLEAN_DEG, local_excursion_m=0.33 * ROLL_CLEAN_DEG,
            max_speed_mps=ROLL_MAX_RATE_DPS)

    # Frozen frames (mocap_frame_advanced == 0, e.g. a ~9 s OptiTrack stall) repeat the
    # previous pose while valid=1. The spike detector misses them, so they are excised
    # here. Without the flag, repeated poses are used (clean.stale_frames).
    stale, stale_source = clean.stale_frames(motion)
    if stale is not None:
        clean_info["stale"] = {**_stale_run_stats(t, stale), "source": stale_source}
        if yc is not None:
            yc = np.asarray(yc, float).copy(); yc[stale] = np.nan
        if xc is not None:
            xc = np.asarray(xc, float).copy(); xc[stale] = np.nan
        for name, v in roll_cands.items():
            v = np.asarray(v, float).copy(); v[stale] = np.nan
            roll_cands[name] = v

    # Keep moving samples with finite cleaned vertical position.
    finite_y = np.isfinite(yc) if yc is not None else np.ones(len(motion), bool)
    use = moving & finite_y
    y = yc[use] if yc is not None else None
    x = xc[use] if xc is not None else None   # may contain NaNs; dropped when resampling
    rolls = {k: v[use] for k, v in roll_cands.items()}
    return t[use], y, x, rolls, is_walking, motion, clean_info


def _signed_deg(a: np.ndarray) -> np.ndarray:
    """Unity Euler angles in [0, 360) as a signed series centred on neutral.

    Wrapping to (-180, 180] alone fails if neutral is near ±180 (the series jumps by 360).
    The series is therefore re-centred on its circular mean before the final wrap.
    """
    a = np.asarray(a, float)
    w = (a + 180.0) % 360.0 - 180.0
    ok = np.isfinite(w)
    if ok.sum():
        mid = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(w[ok])))))
        w = (w - mid + 180.0) % 360.0 - 180.0
    return w


ROLL_BAND_REL = 0.25       # ±25 % band around a target frequency


def _band_power(sig: np.ndarray, fs: float, f0: float, rel: float = ROLL_BAND_REL) -> float:
    """Signal power in a band around ``f0``.

    A band instead of a single DFT bin, because gait frequency drifts by a few percent and
    the step rate is only approximately known. The compared frequencies differ by 2x, so
    ±25 % bands do not overlap.
    """
    from scipy.signal import welch
    if sig is None or not np.size(sig) or not np.isfinite(f0) or f0 <= 0:
        return 0.0
    v = np.asarray(sig, float)
    v = v[np.isfinite(v)]
    if v.size < int(4 * fs):
        return 0.0
    f, pxx = welch(v, fs=fs, nperseg=min(v.size, int(16 * fs)))
    m = (f >= f0 * (1 - rel)) & (f <= f0 * (1 + rel))
    return float(pxx[m].sum()) if m.any() else 0.0


def _pick_roll(tg: np.ndarray, cands: dict,
               troughs: np.ndarray) -> tuple[str | None, np.ndarray | None, float | None]:
    """Choose the orientation channel used as the roll check.

    Roll oscillates once per stride; pitch and components coupled to the bob oscillate
    twice. Candidates are ranked by stride-rate / step-rate power and accepted only if the
    stride component dominates. Selection does not use agreement with sway, so the check
    stays independent.

    The step rate comes from the troughs, not the Welch peak, whose 0.125 Hz resolution
    is too coarse.
    """
    troughs = np.asarray(troughs, float)
    if not cands or troughs.size < 3:
        return None, None, None
    step_s = float(np.median(np.diff(troughs)))
    if not np.isfinite(step_s) or step_s <= 0:
        return None, None, None
    step_hz = 1.0 / step_s
    best = (None, None, 0.0)
    for name, sig in cands.items():
        if sig is None or not np.size(sig):
            continue
        num = _band_power(sig, RESAMPLE_FS, step_hz / 2.0)
        den = _band_power(sig, RESAMPLE_FS, step_hz)
        ratio = num / den if den > 0 else (np.inf if num > 0 else 0.0)
        if ratio > best[2]:
            best = (name, sig, float(ratio))
    if best[0] is None or best[2] <= 1.0:      # no roll-like channel
        return None, None, (round(best[2], 2) if best[0] else None)
    return best[0], best[1], round(min(best[2], 1e6), 2)


def _resample_to(tg: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray | None:
    """Interpolate a channel onto an existing grid, ignoring its NaN samples.

    Resampling channels separately could give grids of different lengths, since their
    NaNs differ; interpolating onto the vertical grid keeps them aligned.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 2:
        return None
    return np.interp(tg, t[ok], v[ok])


def _bandpass(y: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    # Low edge removes slow drift that would mask trough prominence; high edge removes
    # OptiTrack jitter. Output is also the detrended head height of the stride template.
    ny = 0.5 * fs
    b, a = butter(4, [lo / ny, min(hi, 0.99 * ny) / ny], btype="band")
    return filtfilt(b, a, y)


def _gap_spans(t: np.ndarray) -> np.ndarray:
    """``(n, 2)`` start/end times of intervals without samples for more than ``GAP_S``.

    The threshold is at least 3 median frame intervals, so low-fps sessions are not read
    as continuous gaps.
    """
    t = np.asarray(t, float)
    if t.size < 2:
        return np.empty((0, 2))
    dt = np.diff(t)
    i = np.flatnonzero(dt > max(GAP_S, 3.0 * float(np.median(dt))))
    return np.column_stack([t[i], t[i + 1]])


def _in_gaps(x: np.ndarray, spans: np.ndarray, pad: float = GAP_PAD_S) -> np.ndarray:
    """True where ``x`` lies in a gap widened by ``pad`` on each side."""
    x = np.asarray(x, float)
    bad = np.zeros(x.shape, bool)
    for a, b in spans:
        bad |= (x > a - pad) & (x < b + pad)
    return bad


def _detect_troughs(tg: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Trough (double-support) times and dominant frequency of the filtered signal."""
    from scipy.signal import welch
    f, pxx = welch(y, fs=RESAMPLE_FS, nperseg=min(len(y), int(8 * RESAMPLE_FS)))
    band = (f >= GAIT_BAND_HZ[0]) & (f <= GAIT_BAND_HZ[1])
    dom = float(f[band][np.argmax(pxx[band])]) if band.any() else None
    dist = max(1, int(MIN_STEP_INTERVAL_S * RESAMPLE_FS))
    if dom:
        dist = max(dist, int(0.6 / dom * RESAMPLE_FS))
    prom = 0.5 * np.std(y)
    troughs, _ = find_peaks(-y, distance=dist, prominence=prom)
    return tg[troughs], dom


# ===========================================================================
# Stride template (mean head height over the stride)
# ===========================================================================
def _gait_template(tg: np.ndarray, y: np.ndarray, troughs: np.ndarray,
                   step_foot: np.ndarray | None = None):
    """200-point stride template: each detrended step resampled to 100 points, first- and
    second-half steps concatenated. ``step_foot`` assigns the halves; without it the
    step index parity is used."""
    idx = np.searchsorted(tg, troughs)
    steps_a, steps_b = [], []
    dur = np.diff(troughs)
    max_step = MAX_STEP_RATIO * float(np.median(dur)) if dur.size else np.inf
    for i in range(len(idx) - 1):
        seg = y[idx[i]:idx[i + 1]]
        if seg.size < 4 or dur[i] > max_step:
            continue
        seg = detrend(seg, type="linear")
        rs = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, seg.size), seg)
        half = int(step_foot[i]) if (step_foot is not None and i < len(step_foot)) else (i % 2)
        (steps_a if half == 0 else steps_b).append(rs)
    if not steps_a or not steps_b:
        return None, None
    n = min(len(steps_a), len(steps_b))
    stride = np.vstack([np.hstack([steps_a[k], steps_b[k]]) for k in range(n)])
    return stride.mean(0), stride.std(0)


# ===========================================================================
# Stance-foot labels (lateral sway, alternation constraint, roll check)
# ===========================================================================
def foot_name(value) -> str:
    """Name of a ``step_foot`` code (0 = positive-sway foot).

    "right"/"left" only if ``config.LATERAL_SIGN_VERIFIED``; otherwise "foot A"/"foot B".
    """
    if value is None or not np.isfinite(value):
        return "?"
    if not getattr(C, "LATERAL_SIGN_VERIFIED", False):
        return "foot A" if int(value) == 0 else "foot B"
    ref_is_right = bool(getattr(C, "LATERAL_POSITIVE_IS_RIGHT", True))
    if int(value) == 0:
        return "right" if ref_is_right else "left"
    return "left" if ref_is_right else "right"


def _step_means(tg: np.ndarray, sig: np.ndarray, troughs: np.ndarray) -> np.ndarray:
    """Mean of a band-passed channel over each trough-to-trough step (NaN if empty)."""
    idx = np.searchsorted(tg, troughs)
    out = np.full(max(0, len(troughs) - 1), np.nan)
    for i in range(out.size):
        a, b = idx[i], min(idx[i + 1], sig.size)
        if b > a:
            seg = sig[a:b]
            out[i] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
    return out


def _normalise(v: np.ndarray) -> np.ndarray:
    """Divide per-step means by their MAD so the HMM gain means the same for everyone
    (sway amplitude varies several-fold between participants and speeds)."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.zeros_like(v)
    scale = 1.4826 * np.median(np.abs(v[ok] - np.median(v[ok])))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(v[ok])) or 1.0
    return np.nan_to_num(v / scale, nan=0.0)


def _viterbi_alternate(margin: np.ndarray, *, alt_prior: float = ALT_PRIOR,
                       gain: float = EMISSION_GAIN) -> np.ndarray:
    """Most likely alternating 0/1 step sequence given signed per-step evidence.

    Gait alternates, but labelling each step by its sway sign alone lets single noisy
    steps flip, which corrupts the odd harmonics without showing in summary statistics.

    Two-state HMM (state 0 = positive-sway foot) with a strong alternation prior and the
    scaled per-step margin as emission; Viterbi uses the whole sequence, so a weak step
    follows its neighbours.

    The prior is not absolute: two same-foot steps remain possible at a fixed cost, since
    a missed trough produces a double step and a hard constraint would flip all later
    labels.

    ``ALT_PRIOR``, ``EMISSION_GAIN`` and ``MARGIN_CLIP`` are set so that:

      * an isolated flip is impossible: it costs ``2*log(p/(1-p))`` = 15.2, while one step
        contributes at most ``2*EMISSION_GAIN*MARGIN_CLIP`` = 12;
      * a sustained parity shift (missed trough) pays 7.6 once and is then supported by
        the following steps.

    The margin is clipped beforehand because filter edge transients can inflate the first
    and last step by an order of magnitude.
    """
    m = np.asarray(margin, float)
    n = m.size
    if n == 0:
        return np.zeros(0, int)
    log_alt = np.log(alt_prior)
    log_stay = np.log1p(-alt_prior)
    emit = np.column_stack([gain * m, -gain * m])   # state 0: +margin, state 1: -margin
    delta = emit[0].copy()
    back = np.zeros((n, 2), int)
    for i in range(1, n):
        # rows: destination state; columns: (stay, alternate)
        cand = np.array([[delta[0] + log_stay, delta[1] + log_alt],
                         [delta[1] + log_stay, delta[0] + log_alt]])
        src = np.array([[0, 1], [1, 0]])
        best = np.argmax(cand, axis=1)
        back[i] = src[[0, 1], best]
        delta = cand[[0, 1], best] + emit[i]
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _foot_verdict(d: dict) -> str:
    """Reliability of the stance-foot labels.

    Based on the raw alternation; the decoded alternation is ~100 % by construction.
    """
    alt, amb, roll = (d.get("lr_alternation_pct"), d.get("ambiguous_step_pct"),
                      d.get("roll_agreement_pct"))
    if alt is None:
        return "unusable"
    if alt >= 85 and (amb is None or amb <= 20) and (roll is None or roll >= 70):
        return "reliable"
    if alt >= 70 and (amb is None or amb <= 40):
        return "usable (check)"
    return "unreliable - do not use foot identity"


def label_steps(tg, xf, troughs, roll_f=None) -> dict:
    """Per-step stance-foot labels from lateral sway, with alternation constraint and roll
    check.

    The head and trunk shift toward the stance limb in single support, so mean lateral
    sway over a step indicates the stance foot. In addition:

      1. per-step means are scaled to comparable margins (``_normalise``);
      2. head roll, an independent degree of freedom, is processed the same way and
         averaged in if it agrees. Its sign relative to sway is estimated from the overall
         agreement; sway remains the reference, so roll cannot invert the labels;
      3. the sequence is decoded under an alternation prior (``_viterbi_alternate``).

    Returns ``step_foot`` (0 = positive-sway foot), per-step margins and diagnostics.
    ``lr_alternation_pct`` is the raw alternation rate used by QC and the report.
    """
    n_steps = max(0, len(troughs) - 1)
    out = {
        "step_foot": np.zeros(n_steps, int), "step_margin": np.zeros(n_steps),
        "lr_alternation_pct": None, "lr_alternation_constrained_pct": None,
        "n_steps_relabelled": 0, "ambiguous_step_pct": None,
        "lateral_sway_amp_m": None, "roll_agreement_pct": None,
        "roll_polarity": None, "roll_fused": False, "foot_label_verdict": "unusable",
    }
    if xf is None or n_steps < 2:
        return out
    sway = _step_means(tg, xf, troughs)
    if np.isfinite(sway).sum() < 2:
        return out
    margin = _normalise(sway)

    if roll_f is not None and np.size(roll_f) == np.size(xf):
        roll = _normalise(_step_means(tg, roll_f, troughs))
        both = np.isfinite(roll) & np.isfinite(margin) & (roll != 0) & (margin != 0)
        if both.sum() >= 4:
            agree = float(np.mean(np.sign(roll[both]) == np.sign(margin[both])))
            polarity = 1 if agree >= 0.5 else -1
            aligned = max(agree, 1.0 - agree)      # agreement up to the unknown sign
            out["roll_agreement_pct"] = round(100.0 * aligned, 1)
            out["roll_polarity"] = polarity
            if aligned >= ROLL_FUSE_MIN_AGREE:
                margin = 0.5 * (margin + polarity * roll)
                out["roll_fused"] = True

    # Clip before decoding: values beyond MARGIN_CLIP MADs are artefacts (filter edge
    # transients inflate the first and last step ~6x the 99th percentile).
    margin = np.clip(margin, -MARGIN_CLIP, MARGIN_CLIP)

    raw = (np.sign(np.where(margin == 0, 1.0, margin)) < 0).astype(int)
    foot = _viterbi_alternate(margin)
    # A fully inverted decode is the same labelling; align before counting changes.
    if raw.size and np.mean(raw != foot) > 0.5:
        raw = 1 - raw

    conf = np.abs(margin)
    out.update(
        step_foot=foot, step_margin=margin,
        lr_alternation_pct=round(float(100 * np.mean(raw[:-1] != raw[1:])), 1),
        lr_alternation_constrained_pct=round(float(100 * np.mean(foot[:-1] != foot[1:])), 1),
        n_steps_relabelled=int(np.sum(raw != foot)),
        ambiguous_step_pct=round(float(100 * np.mean(conf < AMBIGUOUS_MARGIN)), 1),
        lateral_sway_amp_m=round(float(np.nanmean(np.abs(sway))), 4),
    )
    out["foot_label_verdict"] = _foot_verdict(out)
    return out


def extract_gait(rec) -> GaitResult:
    """Stride cycles and stride phase from the head trace of one recording.

    Steps: select the walking segment, clean spikes and dropouts, resample to a uniform
    grid (Unity frames are uneven), detrend, band-pass, detect vertical troughs (double
    support). A stride spans two steps.

    Steps are labelled by foot from lateral sway. Hilbert and multi-channel phases are
    computed for comparison.
    """
    tt, yy, xx, roll_raw, is_walking, motion, clean_info = _walking_segment(rec)
    yci = clean_info.get("y", {})
    if yy is None or tt.size < int(2 * RESAMPLE_FS):
        return GaitResult(rec.label(), is_walking, None, 0, None, None, None, None)

    tg, yg = bids_io.uniform_resample(tt, yy, fs=RESAMPLE_FS)
    yg = detrend(yg, type="linear")
    yf = _bandpass(yg, RESAMPLE_FS, 0.5, LOWPASS_HZ)
    troughs, dom = _detect_troughs(tg, yf)
    # Discard troughs in or next to data gaps (GAP_S).
    gaps = _gap_spans(tt)
    troughs = troughs[~_in_gaps(troughs, gaps)]

    # Lateral sway and roll on the vertical grid (_resample_to), band-passed, then decoded
    # into stance-foot labels.
    xf = roll_f = None
    step_foot = None
    lab: dict = {}
    if xx is not None:
        xg = _resample_to(tg, tt, xx)
        if xg is not None:
            xf = _bandpass(detrend(xg, type="linear"), RESAMPLE_FS, *SWAY_BAND_HZ)
            cands = {}
            for name, raw in (roll_raw or {}).items():
                rg = _resample_to(tg, tt, raw)
                if rg is not None:
                    cands[name] = _bandpass(detrend(rg, type="linear"), RESAMPLE_FS,
                                            *ROLL_BAND_HZ)
            roll_name, roll_f, roll_ratio = _pick_roll(tg, cands, troughs)
            lab = label_steps(tg, xf, troughs, roll_f)
            lab["roll_channel"] = roll_name
            lab["roll_stride_step_ratio"] = roll_ratio
            step_foot = lab["step_foot"]

    # Step and stride statistics without over-long intervals (gaps, missed troughs).
    step_int = np.diff(troughs)
    if step_int.size:
        step_int = step_int[step_int <= MAX_STEP_RATIO * np.median(step_int)]
    stride_int = np.diff(troughs[::2]) if troughs.size >= 3 else np.array([])
    if stride_int.size:
        stride_int = stride_int[stride_int <= MAX_STEP_RATIO * np.median(stride_int)]
    tmpl, tmpl_sd = _gait_template(tg, yf, troughs, step_foot)

    # --- sampling QC on the raw walking-segment timestamps (b4) ---
    dt = np.diff(tt)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    eff_fs = round(float(1.0 / np.median(dt)), 2) if dt.size else None
    drop_pct = round(float(100 * np.mean(dt > 1.5 / C.NOMINAL_FS)), 2) if dt.size else None

    # --- Hilbert stride phase (b3) and agreement with the trough phase ---
    cont_phase, hsource = continuous_stride_phase(tg, yf, xf, troughs)
    # No Hilbert phase across gaps.
    cont_phase = np.where(_in_gaps(tg, gaps), np.nan, cont_phase)
    h_corr = None
    if troughs.size >= 4:
        # circular agreement of both phases at step midpoints
        mids = 0.5 * (troughs[:-1] + troughs[1:])
        _, ps_trough = phase_at(mids, troughs, step_foot)
        _, ps_hilb = phase_at_hilbert(mids, tg, cont_phase)
        ok = np.isfinite(ps_trough) & np.isfinite(ps_hilb)
        if ok.sum() >= 4:
            a = np.exp(1j * 2 * np.pi * ps_trough[ok] / 100.0)
            b = np.exp(1j * 2 * np.pi * ps_hilb[ok] / 100.0)
            h_corr = round(float(np.abs(np.mean(a * np.conj(b)))), 3)  # 0..1

    # Multi-channel stride phase (phaser.py); used if config.PHASE_METHOD = "phaser".
    ph_phase, ph_info = phaser.stride_phase(
        tg, {"lateral": xf, "roll": roll_f}, {"vertical": yf},
        ~_in_gaps(tg, gaps), troughs, step_foot)
    p_corr = None
    if troughs.size >= 4 and np.isfinite(ph_phase).any():
        mids = 0.5 * (troughs[:-1] + troughs[1:])
        _, ps_trough = phase_at(mids, troughs, step_foot)
        _, ps_ph = phase_at_hilbert(mids, tg, ph_phase)
        ok = np.isfinite(ps_trough) & np.isfinite(ps_ph)
        if ok.sum() >= 4:
            a = np.exp(1j * 2 * np.pi * ps_trough[ok] / 100.0)
            b = np.exp(1j * 2 * np.pi * ps_ph[ok] / 100.0)
            p_corr = round(float(np.abs(np.mean(a * np.conj(b)))), 3)

    res = GaitResult(
        recording=rec.label(), walking=is_walking,
        dominant_freq_hz=round(dom, 3) if dom else None,
        n_steps=int(troughs.size),
        step_duration_s=round(float(step_int.mean()), 4) if step_int.size else None,
        step_cov_pct=round(float(100 * step_int.std() / step_int.mean()), 2) if step_int.size else None,
        stride_duration_s=round(float(stride_int.mean()), 4) if stride_int.size else None,
        cadence_spm=round(float(60.0 / step_int.mean()), 1) if step_int.size else None,
        lateral_sway_amp_m=lab.get("lateral_sway_amp_m"),
        lr_alternation_pct=lab.get("lr_alternation_pct"),
        lr_alternation_constrained_pct=lab.get("lr_alternation_constrained_pct"),
        n_steps_relabelled=lab.get("n_steps_relabelled"),
        ambiguous_step_pct=lab.get("ambiguous_step_pct"),
        roll_channel=lab.get("roll_channel"),
        roll_stride_step_ratio=lab.get("roll_stride_step_ratio"),
        roll_agreement_pct=lab.get("roll_agreement_pct"),
        roll_polarity=lab.get("roll_polarity"),
        roll_fused=lab.get("roll_fused"),
        foot_label_verdict=lab.get("foot_label_verdict"),
        foot_label_anatomical=bool(getattr(C, "LATERAL_SIGN_VERIFIED", False)),
        effective_fs_hz=eff_fs, dropped_frame_pct=drop_pct,
        raw_segment_samples=int(tt.size),
        hilbert_phase_source=hsource, hilbert_trough_phase_corr=h_corr,
        phaser_trough_phase_corr=p_corr, phaser_channels=ph_info.get("channels"),
        clean_flagged_pct=yci.get("pct_flagged"),
        clean_excised_pct=(round(100.0 * yci["n_excised"] / yci["n"], 2)
                           if yci.get("n") else None),
        clean_coverage_pct=yci.get("clean_coverage_pct"),
        clean_max_spike_m=yci.get("max_excursion_m"),
        n_data_gaps=int(len(gaps)),
        data_gap_total_s=round(float((gaps[:, 1] - gaps[:, 0]).sum()), 2),
        t_grid=tg, y_filt=yf, x_filt=xf, roll_filt=roll_f,
        trough_times=troughs, step_foot=step_foot, step_margin=lab.get("step_margin"),
        stride_template=tmpl, stride_template_sd=tmpl_sd,
        stride_durations=stride_int, stride_phase_cont=cont_phase,
        stride_phase_phaser=ph_phase,
    )
    return res


# ===========================================================================
# Figures
# ===========================================================================
def _draw_template(ax, res: GaitResult) -> None:
    """Davidson Fig 1d / 3b: mean detrended head height over the stride."""
    x = np.linspace(0, 100, STRIDE_POINTS)
    if res.stride_template is not None:
        m, sd = res.stride_template, res.stride_template_sd
        ax.fill_between(x, m - sd, m + sd, color=S.MIST, label="±1 SD")
        ax.plot(x, m, color=S.INK, lw=2, label="mean")
        for xs in (0, 50, 100):                  # double support
            ax.axvline(xs, color=S.GREY, ls=":", lw=0.8)
        ax.text(25, m.max(), "step 1", ha="center", fontsize=8.5, color=S.GREY)
        ax.text(75, m.max(), "step 2", ha="center", fontsize=8.5, color=S.GREY)
        ax.legend(loc="lower center", ncol=2)
    ax.set_xlabel("% stride cycle"); ax.set_ylabel("detrended head height (m)")
    S.panel(ax, "a", "Stride template", fontsize=10)


def _draw_durations(ax, res: GaitResult) -> None:
    if res.stride_durations is not None and res.stride_durations.size > 1:
        ax.hist(res.stride_durations, bins=20, color=S.BLUE, alpha=0.85)
        ax.axvline(res.stride_duration_s, color=S.INK, ls="--",
                   label=f"mean = {res.stride_duration_s:.2f} s")
        ax.legend()
    ax.set_xlabel("stride duration (s)"); ax.set_ylabel("stride count")
    S.panel(ax, "b", f"Stride duration  (step CoV {res.step_cov_pct}%)", fontsize=10)


def _draw_trace(ax3d, ax, res: GaitResult, events, window_s: float = 12.0) -> None:
    """Davidson Fig 1c / 3a: 3D head path (walking direction × sway × height) and head
    height trace with onsets (filled = hit, open = miss), centred on the median onset."""
    tg, yf, xf = res.t_grid, res.y_filt, res.x_filt
    ev = events[events["onset"].notna()] if not events.empty else events
    t0 = max(tg[0], float(ev["onset"].median() - window_s / 2) if len(ev) else tg[0])
    t1 = min(tg[-1], t0 + window_s)
    win = (tg >= t0) & (tg <= t1)
    lat = xf if xf is not None else np.zeros_like(tg)
    fwd, hgt = tg[win] - t0, yf[win] + 1.7
    onsets = [(r["onset"] - t0, np.interp(r["onset"], tg, lat), np.interp(r["onset"], tg, yf) + 1.7,
               r.get("response_correct") == 1)
              for _, r in ev[(ev["onset"] >= t0) & (ev["onset"] <= t1)].iterrows()] if len(ev) else []

    ax3d.plot(fwd, lat[win], hgt, color=S.TEAL, lw=1.6)
    for f, lv, hv, hit in onsets:
        ax3d.scatter(f, lv, hv, s=40, facecolor=S.GREEN if hit else "white",
                     edgecolor=S.GREEN, depthshade=False, zorder=5)
    ax3d.set_xlabel("walking direction (s)"); ax3d.set_ylabel("lateral sway (m)")
    ax3d.set_zlabel("head height (m)")
    ax3d.set_title("c  3D head path", loc="left", fontsize=10)
    ax3d.view_init(elev=18, azim=-72)

    ax.plot(fwd, hgt, color=S.INK, lw=1.1)
    for f, _, hv, hit in onsets:
        ax.axvline(f, color=S.MIST, lw=0.8, zorder=0)
        ax.scatter(f, hv, s=40, facecolor=S.GREEN if hit else "white", edgecolor=S.GREEN, zorder=3)
    ax.set_xlabel(f"time from {t0:.0f} s into the block (s)"); ax.set_ylabel("head height (m)")
    S.panel(ax, "d", "Head height and target onsets  (filled = hit, open = miss)", fontsize=10)


FOOT_COLOUR = (S.BLUE, S.ORANGE)      # by step_foot (0 = +sway foot)


def _draw_feet(ax, res: GaitResult, window_s: float = 12.0) -> None:
    """QC of the stance-foot labels: evidence, decoded labels and roll agreement.

    Also used to verify the sway sign: with a known first step, check the first shaded
    step in panel (e); if inverted, flip ``config.LATERAL_POSITIVE_IS_RIGHT`` and set
    ``LATERAL_SIGN_VERIFIED``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    tg, xf, troughs, foot = res.t_grid, res.x_filt, res.trough_times, res.step_foot
    if xf is None or foot is None or np.size(troughs) < 3 or not len(foot):
        for a in ax:
            a.axis("off")
        ax[0].text(0, 0.5, "stance-foot labels unavailable (no lateral sway channel)",
                   transform=ax[0].transAxes, color=S.GREY, fontsize=10)
        return
    margin = res.step_margin if res.step_margin is not None else np.zeros(len(foot))
    names = (foot_name(0), foot_name(1))

    # (e) sway trace, steps shaded by decoded foot
    t0 = float(tg[0])
    t1 = float(min(tg[-1], t0 + window_s))
    win = (tg >= t0) & (tg <= t1)
    ax[0].plot(tg[win] - t0, xf[win] * 100.0, color=S.INK, lw=1.3, zorder=3)
    if res.roll_filt is not None:
        ax[0].plot(tg[win] - t0, (res.roll_polarity or 1) * res.roll_filt[win], color=S.PURPLE,
                   lw=1.0, alpha=0.75, zorder=2)
    for i in range(len(foot)):
        a, b = troughs[i], troughs[i + 1]
        if b >= t0 and a <= t1:
            ax[0].axvspan(max(a, t0) - t0, min(b, t1) - t0, color=FOOT_COLOUR[int(foot[i])],
                          alpha=0.16, lw=0, zorder=0)
    for tr in troughs[(troughs >= t0) & (troughs <= t1)]:
        ax[0].axvline(tr - t0, color=S.GREY, ls=":", lw=0.7, zorder=1)
    leg = [Patch(facecolor=FOOT_COLOUR[k], alpha=0.35, label=f"stance: {names[k]}") for k in (0, 1)]
    leg.append(Line2D([], [], color=S.INK, lw=1.3, label="lateral sway (cm)"))
    if res.roll_filt is not None:
        leg.append(Line2D([], [], color=S.PURPLE, lw=1.0, label="head roll (deg, aligned)"))
    ax[0].legend(handles=leg, loc="upper right", ncol=2, fontsize=7.5)
    ax[0].set_xlabel("time from start of walking segment (s)")
    ax[0].set_ylabel("sway (cm) / roll (deg)")
    S.panel(ax[0], "e", "Decoded steps  (double support = dotted)", fontsize=10)

    # (f) per-step evidence
    raw = (np.asarray(margin) < 0).astype(int)
    if raw.size and np.mean(raw != foot) > 0.5:
        raw = 1 - raw                     # same labelling, inverted names
    moved = np.where(raw != np.asarray(foot))[0]
    ax[1].axhspan(-AMBIGUOUS_MARGIN, AMBIGUOUS_MARGIN, color=S.MIST, alpha=0.55, lw=0,
                  zorder=0, label=f"low evidence (|m| < {AMBIGUOUS_MARGIN})")
    ax[1].bar(np.arange(len(margin)), margin, width=0.9, lw=0, zorder=2,
              color=[FOOT_COLOUR[int(f)] for f in foot])
    ax[1].axhline(0, color=S.GREY, lw=0.8, zorder=3)
    if moved.size:
        ax[1].scatter(moved, np.zeros(moved.size), marker="v", s=28, color=S.RED,
                      zorder=5, label=f"relabelled by alternation ({moved.size})")
    ax[1].set_ylim(-1.15 * MARGIN_CLIP, 1.15 * MARGIN_CLIP)
    ax[1].set_xlabel("step index"); ax[1].set_ylabel("signed evidence (MAD units, clipped)")
    ax[1].legend(loc="upper right", fontsize=7.5)
    S.panel(ax[1], "f", f"Per-step evidence  ({res.ambiguous_step_pct}% low)", fontsize=10)

    # (g) roll agreement
    if res.roll_filt is not None and res.roll_agreement_pct is not None:
        sway_m = _normalise(_step_means(tg, xf, troughs))
        roll_m = _normalise(_step_means(tg, res.roll_filt, troughs)) * (res.roll_polarity or 1)
        ok = np.isfinite(sway_m) & np.isfinite(roll_m)
        agree = np.sign(sway_m) == np.sign(roll_m)
        for m, col, lab in ((ok & agree, S.TEAL, "agree"), (ok & ~agree, S.RED, "disagree")):
            if m.any():
                ax[2].scatter(sway_m[m], roll_m[m], s=16, color=col, alpha=0.75, label=lab)
        ax[2].axhline(0, color=S.GREY, lw=0.8); ax[2].axvline(0, color=S.GREY, lw=0.8)
        both = np.concatenate([sway_m[ok], roll_m[ok]]) if ok.any() else np.array([1.0])
        lim = float(np.nanpercentile(np.abs(both), 99)) * 1.25 or 1.0
        ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim)
        ax[2].legend(loc="upper left", fontsize=7.5)
        ax[2].set_xlabel("sway evidence"); ax[2].set_ylabel("roll evidence (aligned)")
        S.panel(ax[2], "g", f"Roll cross-check ({res.roll_channel}): {res.roll_agreement_pct}%"
                            f"{', fused' if res.roll_fused else ', not fused'}", fontsize=9.5)
    else:
        ax[2].text(0.5, 0.5, "no roll-like orientation channel:\ncross-check unavailable",
                   ha="center", va="center", transform=ax[2].transAxes, color=S.GREY, fontsize=9)
        ax[2].set_xticks([]); ax[2].set_yticks([])
        S.panel(ax[2], "g", "Roll cross-check", fontsize=10)


def _figure(rec, res: GaitResult, events, out: Path) -> None:
    """Gait sheet per block: stride (a-b), head path with onsets (c-d), stance-foot
    labels and roll check (e-g)."""
    fig = plt.figure(figsize=(16, 14))
    rows = fig.add_gridspec(3, 1, height_ratios=[1, 0.85, 1], hspace=0.38)
    top = rows[0].subgridspec(1, 3, wspace=0.28)
    feet = rows[2].subgridspec(1, 3, width_ratios=[1.8, 1.2, 1.0], wspace=0.3)
    fig.suptitle(f"Gait — {S.nice_label(rec.label())}  ·  stance-foot labels "
                 f"[{res.foot_label_verdict}]", fontsize=13, fontweight="semibold",
                 x=0.012, ha="left")
    _draw_template(fig.add_subplot(top[0]), res)
    _draw_durations(fig.add_subplot(top[1]), res)
    _draw_trace(fig.add_subplot(top[2], projection="3d"), fig.add_subplot(rows[1]), res, events)
    _draw_feet([fig.add_subplot(feet[i]) for i in range(3)], res)
    S.caption(fig, "Foot identity comes from lateral sway (the head leans toward the stance "
                   "limb), decoded under an alternation prior and cross-checked against head "
                   "roll. " + ("Sway sign has been verified against this rig, so left/right is "
                               "anatomical." if res.foot_label_anatomical else
                               "The sway-sign convention is NOT yet verified for this rig, so "
                               "the two feet are named A/B: they are told apart reliably, but "
                               "which one is anatomically left is still unknown."),
              wrap=170, y=0.035)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.98)
    S.save(fig, out)


def gait_recording(rec) -> GaitResult:
    """Run :func:`extract_gait` on one recording and persist its outputs.

    Writes the summary JSON, the time series ``.npz`` (for the phase analyses) and the
    gait sheet.
    """
    res = extract_gait(rec)
    events = bids_io.load_events(rec)
    out_dir = bids_io.deriv_dir("gait", rec.sub, rec.ses)
    (out_dir / f"{rec.label()}_gait.json").write_text(json.dumps(res.summary(), indent=2),
                                                      encoding="utf-8")
    if res.t_grid is not None:
        np.savez(out_dir / f"{rec.label()}_gait.npz",
                 t=res.t_grid, y_filt=res.y_filt,
                 x_filt=res.x_filt if res.x_filt is not None else [],
                 troughs=res.trough_times,
                 step_foot=res.step_foot if res.step_foot is not None else [],
                 step_margin=res.step_margin if res.step_margin is not None else [],
                 roll_filt=res.roll_filt if res.roll_filt is not None else [],
                 stride_template=res.stride_template if res.stride_template is not None else [],
                 stride_durations=res.stride_durations if res.stride_durations is not None else [],
                 stride_phase_cont=res.stride_phase_cont if res.stride_phase_cont is not None else [],
                 stride_phase_phaser=(res.stride_phase_phaser
                                      if res.stride_phase_phaser is not None else []))
        _figure(rec, res, events, bids_io.fig_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.png")
    return res


def _print(res: GaitResult):
    print(f"\n=== Gait: {res.recording} ===")
    if not res.walking:
        print("  standing block - no gait extraction.")
        return
    print(f"  spike cleaning (vertical)   : {res.clean_flagged_pct}% flagged, "
          f"{res.clean_excised_pct}% excised, coverage {res.clean_coverage_pct}%, "
          f"max spike {res.clean_max_spike_m} m removed BEFORE band-pass")
    print(f"  dominant vertical-head freq : {res.dominant_freq_hz} Hz (step rate)")
    print(f"  steps detected              : {res.n_steps}")
    print(f"  step / stride duration      : {res.step_duration_s} s / {res.stride_duration_s} s "
          f"(cadence {res.cadence_spm} spm)")
    print(f"  gait variability (step CoV) : {res.step_cov_pct}%")
    print(f"  sampling (raw segment)      : {res.effective_fs_hz} Hz effective, "
          f"{res.dropped_frame_pct}% dropped frames ({res.raw_segment_samples} samples)")
    print(f"  Hilbert phase ({res.hilbert_phase_source})       : "
          f"circular agreement with trough phase = {res.hilbert_trough_phase_corr}")
    print(f"  lateral sway (gait band)    : {res.lateral_sway_amp_m} m/step")
    print(f"  stance-foot labels          : alternation {res.lr_alternation_pct}% raw -> "
          f"{res.lr_alternation_constrained_pct}% decoded "
          f"({res.n_steps_relabelled} steps relabelled, "
          f"{res.ambiguous_step_pct}% low-evidence)")
    print("  head-roll cross-check       : "
          + (f"{res.roll_channel} ({res.roll_stride_step_ratio}x stride/step), "
             f"{res.roll_agreement_pct}% agreement"
             f"{' (fused into the evidence)' if res.roll_fused else ' (below threshold, not fused)'}"
             if res.roll_agreement_pct is not None else "channel unavailable"))
    print(f"  foot identity               : {res.foot_label_verdict} - labels named "
          f"{foot_name(0)}/{foot_name(1)}"
          + ("" if res.foot_label_anatomical
             else "  (sway sign NOT yet verified against the rig)"))


def main():
    """Extract stride cycles and stride phase for every walking recording."""
    # Gait scope: only manual exclusions apply.
    recs = bids_io.find_recordings(scope="gait")
    if not recs:
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return
    for rec in recs:
        _print(gait_recording(rec))
    walk = [r for r in recs if r.condition == "walking"]
    n_done = len({r.sub for r in walk
                  if (bids_io.deriv_dir("gait", r.sub, r.ses) / f"{r.label()}_gait.npz").exists()})
    bids_io.gate_report("gait", "gait", condition="walking", n_analysed=n_done)
    print(f"\nGait outputs -> {C.DERIV_DIR / 'gait'}")


if __name__ == "__main__":
    main()
