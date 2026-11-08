"""Exploratory: what the left/right response depends on over the step.

* Stance-foot contrast: whether performance, or the response direction, depends on which
  foot is in stance.
* Sensitivity vs criterion (SDT): whether an accuracy oscillation is sensory or decisional.
* Head roll and the left/right response: the Gabor follows head position but not head
  rotation, so roll tilts it on the retina.

None of this is pre-registered. The confirmatory tests are ``phase_analysis`` (per
participant) and ``population`` (phase-agnostic test).

Usage::

    python -m Analysis.response_bias


Stance-foot contrast
====================

Exploratory: does performance depend on which foot is in stance?

The confirmatory analysis (``phase_analysis``, ``population``) tests 2 and 4 cycles per
stride, which repeat every step. A half-stride shift is one cycle at 2 cps and two at
4 cps, so these results are identical with the foot labels swapped. Left/right asymmetry
appears only in the odd harmonics (1, 3 cps) and needs a stride origin that distinguishes
the feet.

``gait.label_steps`` provides that origin from lateral sway. This module tests:

    Does accuracy (or RT) differ between targets shown during one foot's stance and
    the other's?

One degree of freedom, no Fourier fit, and it pools easily across participants (unlike a
1 cps R² fit).

The test
--------
Statistic: ``Δ = mean(y | stance foot A) − mean(y | stance foot B)`` over the
participant's pooled walking trials (accuracy primary, log10 RT secondary).

Null: randomly flip the foot labels of whole strides. Under H0 the two half-strides are
exchangeable, so this is an exact randomisation. It keeps each trial's within-step phase,
timing and the trial count, changing only the foot label, so a 2 cps oscillation, slow
drift or block structure cannot create an effect.

Population level, as in ``population`` (when the direction differs between
participants, same-direction tests are weak):

  * direction-agnostic (primary): ``S = Σ_i z_i²``, ``z_i = Δ_i / sd(null_i)``. Tests for
    an asymmetry in any direction per participant. Valid even with the sway-sign convention
    unverified, since it only needs consistent foot labels within a participant.
  * direction-specific (secondary): ``T = mean_i z_i``. Tests for the same direction in
    everyone. This is an anatomical claim, reported as provisional until
    ``config.LATERAL_SIGN_VERIFIED`` is set; until then only its magnitude is meaningful.

Both nulls are built from the participants' own permutations.

Response direction
------------------
The accuracy contrast cannot see a lean of the response towards one side: with tilts
balanced, a bias towards "right" on foot A's steps raises accuracy for right tilts and
lowers it for left tilts by the same amount. Three further contrasts on the same
stride-flip null ask about the response itself:

  * criterion ``Δc = c(foot A) - c(foot B)``, with ``c`` the signal-detection criterion
    (tilt right = signal, log-linear rates as in ``response_bias``). A lean towards the side
    of the stance foot makes Δc non-zero.
  * tilt towards vs away from the stance foot, on accuracy and on log RT. A trial is
    "towards" when the tilt points to the side of the foot in stance. A stride flip swaps
    the foot of every trial in the stride, and so its pairing, so the same null applies.
    In accuracy this is the criterion contrast again, in percentage points; in RT it asks
    whether a tilt towards the stance side is answered faster or slower, which a criterion
    shift alone does not predict.

Until ``config.LATERAL_SIGN_VERIFIED`` is set, "towards" means towards the side the
lateral-sway convention assigns to the stance foot. The direction-agnostic S is valid
either way.

Confirmatory tests remain ``phase_analysis`` (per participant, 2/4 cps) and
``population`` (population).


Sensitivity vs criterion (SDT)
==============================

Signal-detection decomposition of the stride-phase effect: d' vs criterion.

Question
--------
``phase_analysis`` tests whether accuracy oscillates over the stride, but not what
oscillates. In left/right tilt discrimination, accuracy depends on two separable
quantities:

  * sensitivity (d'): how well the tilt is seen;
  * criterion (c): offset of the decision boundary, i.e. a preference for "right" or
    "left".

A criterion that varies over the stride lowers accuracy just like a sensitivity dip, so an
accuracy oscillation can be perceptual, decisional or both. Davidson et al. (2024) did not
separate these. A response-side bias is exactly what ``c`` measures.

No extra data are needed: ``events.tsv`` has ``stimulus_side`` and ``response`` for every
trial, and tilt directions are balanced 50/50 (``CheckpointManager.ConstrainedShuffleNoLongRun``),
so both rates can be estimated at every phase.

Computation
-----------
"Tilt right" is the signal:

    H = P(said right | tilt right)      F = P(said right | tilt left)
    d' = z(H) - z(F)                    c = -0.5 * (z(H) + z(F))

Positive ``c`` = bias against answering right. Which side is the signal is a convention,
so |c| and its modulation are meaningful, not the resting sign.

Rates use the log-linear correction (Hautus 1995): +0.5 to each count and +1 to each
denominator, in every bin. This keeps d' finite in all-correct bins and, applied equally
to every permutation, does not bias the test.

Timeouts are excluded, as in the accuracy analysis: with no response a trial is neither
hit nor false alarm. The number excluded is reported per participant. Whether timeouts
depend on phase is the timeout measure of ``phase_analysis``.

20 bins instead of 40, because each bin needs two rates. With ~1360 pooled walking trials
that is ~68 trials per bin, ~34 per cell. For the same reason only pooled trials per
participant are used; a single ~170-trial run would give ~4 trials per cell.

The test
--------
Same method as the confirmatory analysis: cosine fit at 2 and 4 cycles per stride,
Holm-corrected, against a trial-level permutation null.

The null shuffles the ``(stimulus_side, response)`` pair across trials with phases fixed.
Shuffling the pair keeps each trial's outcome and both marginals and only changes the phase
of each outcome; this is the exact randomisation for "neither d' nor c depends on phase".

The population test is ``population.agnostic_test`` (studentised, rotation-invariant
``S = sum_i |z_i|^2 / s_i^2`` against the participants' own permutations), reused so both
modules compute the same statistic.

Stride rate (1 cps). The 2 and 4 cps family repeats once per step and cannot see a bias
that flips with the stance foot. A head roll that reaches the retina (see the head-roll
analysis below), or a lean of the response towards the stance side, alternates with the
foot and sits at 1 cycle per stride. The same fit and shuffles are therefore also evaluated at 1 cps, as a single
separate test outside the Holm family. At 1 cps the stride origin is fixed by the
stance-foot labels (foot A first), which are consistent across participants, so a
common-phase Hotelling test (``population.hotelling_t2``) is reported alongside S.

Exploratory. Confirmatory tests remain ``phase_analysis`` (per participant) and
``population`` (phase-agnostic test) on accuracy. This module decomposes an accuracy effect; it
does not test whether one exists.


Head roll and the left/right response
=====================================

Exploratory: does head roll at stimulus onset bias the tilt judgement?

The Gabor plane follows the tracked head POSITION but not its rotation (``CameraHolder``
tracks position only), so the grating stays upright in the room. A head roll of r degrees
therefore rotates the grating on the retina by r, less whatever ocular counter-roll
compensates; counter-roll covers only a fraction of a dynamic head roll (Collewijn et al.,
1985). Walking rolls the head once per stride (``head_motion.py`` measures it), and near
threshold the tilts judged are small. Retinal-image accounts of stride-phase effects in
orientation tasks (Matthis et al., 2022) can usually be set aside when the tilts judged
are large; for near-threshold tilts the roll has to be tested.

A retinal rotation adds a signed offset to the tilt that is seen. That moves the
left/right decision boundary (a criterion shift in signal-detection terms) without
changing how well the tilt is resolved, so it shows up in WHICH response is given rather
than in accuracy over balanced tilts. The test is therefore on the response:

    logit P(say right) = block_b + beta_tilt * tilt + beta_roll * roll

per participant over the pooled walking trials. ``tilt`` is the signed stimulus tilt in
degrees (+ right), ``roll`` the head roll averaged over the stimulus's measured on-screen
time, relative to the block's mean pose. The effect is reported as the equivalent
tilt ``g = beta_roll / beta_tilt``, degrees of stimulus tilt per degree of head roll.
An uncompensated retinal rotation predicts ``|g| = 1``; full compensation (by counter-roll
and by the observer's knowledge of their own head tilt) predicts ``g = 0``.

Two further models guard against the obvious confounds:

* adjusted: adds the stride phase (cos and sin at 1 and 2 cycles per stride) and a
  quadratic trend of time within each block. It asks whether roll predicts the response
  beyond the average stride pattern and beyond a slow drift of the response bias over
  the block, which a slow change of posture or a slipping head marker could share.
* fast: replaces roll by its gait-band part (0.3-5 Hz, ``head_motion.BAND_HZ``),
  so only the roll that changes from step to step is used; no slow drift reaches it.

Null: the roll values are permuted within blocks after removing what the nuisance terms
(block, and the phase and time terms of the adjusted model) explain, then added back to
those fitted values (permutation of residuals; Kennedy, 1995). Each permutation refits
the model. The per-participant ``z = beta_roll / SD(null)`` enters the population tests of
``response_bias.population_test``: the direction-agnostic ``S = sum z^2`` and the
same-direction ``T = mean z``. The mocap frame is the same in every session, so roll has
one sign convention across participants and T is a valid test of a common direction.
Positive roll is right ear up (measured by the rig check, ``config.HEAD_ROLL_SIGN_VERIFIED``)
and predicts more "right" responses.

References
----------
Collewijn H, Van der Steen J, Ferman L, Jansen TC (1985). Human ocular counterroll:
    assessment of static and dynamic properties from electromagnetic scleral coil
    recordings. Exp Brain Res 59:185-196.
Kennedy PE (1995). Randomization tests in econometrics. J Bus Econ Stat 13:85-94.
Matthis JS, Muller KS, Bonnen KL, Hayhoe MM (2022). Retinal optic flow during natural
    locomotion. PLoS Comput Biol 18:e1009575.
"""
from __future__ import annotations

import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

try:
    from . import bids_io
    from . import config as C
    from . import gait as gait_mod
    from . import head_motion as hk
    from . import phase_analysis as pa
    from . import plotstyle as S
    from . import population
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import gait as gait_mod
    import head_motion as hk
    import phase_analysis as pa
    import plotstyle as S
    import population

S.apply()


# ###########################################################################
# Stance-foot contrast
# ###########################################################################
N_PERM = 2000            # per-participant stride-flip permutations
MIN_TRIALS_PER_FOOT = 20  # minimum trials per foot
N_STEP_BINS = 16         # bins for the accuracy-by-step panel
RNG_SEED = 11


# ===========================================================================
# Statistical helpers
# ===========================================================================
def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (more accurate than Wald here)."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def _two_sided_p(obs: float, null: np.ndarray) -> float:
    """Two-sided permutation p with +1 correction."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs):
        return float("nan")
    return float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + null.size))


def _stride_flip_null(y: np.ndarray, foot: np.ndarray, stride_pos: np.ndarray,
                      n_strides: int, n_perm: int, rng: np.random.Generator,
                      chunk: int = 256) -> np.ndarray:
    """Stride-flip null distribution of Δ, computed from per-stride sums.

    A flip swaps a stride's two per-foot sums between groups, so with (sum, count) per
    stride and foot each permutation is one dot product. Permutations are processed in
    chunks to limit memory; float64 is kept because effects of ~1e-3 are within float32
    error over thousands of log-RTs.
    """
    is0 = foot == 0
    a_sum = np.bincount(stride_pos[is0], weights=y[is0], minlength=n_strides)
    b_sum = np.bincount(stride_pos[~is0], weights=y[~is0], minlength=n_strides)
    a_n = np.bincount(stride_pos[is0], minlength=n_strides).astype(float)
    b_n = np.bincount(stride_pos[~is0], minlength=n_strides).astype(float)
    d_sum, d_n = b_sum - a_sum, b_n - a_n            # change in group 0 when flipped
    base_sum, base_n = a_sum.sum(), a_n.sum()
    tot_sum, tot_n = base_sum + b_sum.sum(), base_n + b_n.sum()

    out = np.empty(n_perm)
    for i in range(0, n_perm, chunk):
        f = rng.integers(0, 2, size=(min(chunk, n_perm - i), n_strides)).astype(np.float64)
        s0 = base_sum + f @ d_sum
        c0 = base_n + f @ d_n
        s1, c1 = tot_sum - s0, tot_n - c0
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i:i + f.shape[0]] = np.where((c0 > 0) & (c1 > 0),
                                             s0 / np.where(c0 > 0, c0, 1)
                                             - s1 / np.where(c1 > 0, c1, 1), np.nan)
    return out


def stance_contrast(y: np.ndarray, foot: np.ndarray, stride: np.ndarray, *,
                    n_perm: int = N_PERM,
                    rng: np.random.Generator | None = None) -> dict | None:
    """Foot contrast on one measure, with the stride-flip permutation null.

    ``y`` values, ``foot`` 0/1 stance labels and ``stride`` ids, one per trial. Non-finite
    ``y`` are dropped (RT exists only for clean correct trials).

    The stride-flip null (see module docstring) needs no adjustment for within-step phase
    or time on task.
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    y = np.asarray(y, float)
    foot = np.asarray(foot, int)
    stride = np.asarray(stride)
    ok = np.isfinite(y) & np.isin(foot, (0, 1))
    y, foot, stride = y[ok], foot[ok], stride[ok]
    n0, n1 = int((foot == 0).sum()), int((foot == 1).sum())
    if min(n0, n1) < MIN_TRIALS_PER_FOOT:
        return None

    # compact stride ids to column indices
    _, stride_pos = np.unique(stride, return_inverse=True)
    n_strides = int(stride_pos.max()) + 1

    obs = float(np.mean(y[foot == 0]) - np.mean(y[foot == 1]))
    null = _stride_flip_null(y, foot, stride_pos, n_strides, n_perm, rng)
    sd = float(np.nanstd(null))
    return {
        "delta": obs, "p": _two_sided_p(obs, null),
        "null_sd": sd, "z": float(obs / sd) if sd > 0 else float("nan"),
        "n_foot0": n0, "n_foot1": n1,
        "mean_foot0": float(np.mean(y[foot == 0])), "mean_foot1": float(np.mean(y[foot == 1])),
        "_null": null,
    }


def _criterion(counts: np.ndarray) -> np.ndarray:
    """SDT criterion from ``(..., [n_signal, n_hit, n_noise, n_false_alarm])`` counts,
    with the log-linear correction ``response_bias`` uses."""
    from scipy.special import ndtri
    hr = (counts[..., 1] + 0.5) / (counts[..., 0] + 1.0)
    fa = (counts[..., 3] + 0.5) / (counts[..., 2] + 1.0)
    return -0.5 * (ndtri(hr) + ndtri(fa))


def criterion_contrast(is_right: np.ndarray, said_right: np.ndarray, foot: np.ndarray,
                       stride: np.ndarray, *, n_perm: int = N_PERM,
                       rng: np.random.Generator | None = None, chunk: int = 256) -> dict | None:
    """``Δc = c(foot 0) − c(foot 1)`` with the stride-flip null.

    As in ``_stride_flip_null``, a flip can only move a stride's per-foot counts between
    the two groups, so the four SDT counts are aggregated per stride and foot once and
    each permutation is a matrix product. ``said_right`` NaN (a timeout) is dropped.
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    said = np.asarray(said_right, float)
    foot = np.asarray(foot, int)
    ok = np.isfinite(said) & np.isin(foot, (0, 1))
    sig = np.asarray(is_right, bool)[ok]
    said, foot, stride = said[ok] > 0.5, foot[ok], np.asarray(stride)[ok]
    n0, n1 = int((foot == 0).sum()), int((foot == 1).sum())
    if min(n0, n1) < MIN_TRIALS_PER_FOOT:
        return None
    _, sp = np.unique(stride, return_inverse=True)
    ns = int(sp.max()) + 1
    cnt = lambda m: np.bincount(sp[m], minlength=ns).astype(float)
    per = {k: np.stack([cnt((foot == k) & sig), cnt((foot == k) & sig & said),
                        cnt((foot == k) & ~sig), cnt((foot == k) & ~sig & said)])
           for k in (0, 1)}                                   # (4, n_strides)
    base, d = per[0].sum(1), (per[1] - per[0]).T              # (4,), (n_strides, 4)
    tot = per[0].sum(1) + per[1].sum(1)
    c0, c1 = float(_criterion(per[0].sum(1))), float(_criterion(per[1].sum(1)))
    null = np.empty(n_perm)
    for i in range(0, n_perm, chunk):
        f = rng.integers(0, 2, size=(min(chunk, n_perm - i), ns)).astype(np.float64)
        g0 = base + f @ d
        null[i:i + f.shape[0]] = _criterion(g0) - _criterion(tot - g0)
    obs = c0 - c1
    sd = float(np.std(null))
    return {"delta": obs, "p": _two_sided_p(obs, null), "null_sd": sd,
            "z": float(obs / sd) if sd > 0 else float("nan"),
            "n_foot0": n0, "n_foot1": n1, "mean_foot0": c0, "mean_foot1": c1,
            "_null": null}


def towards_label(foot: np.ndarray, is_right: np.ndarray) -> np.ndarray:
    """0 where the tilt points to the stance foot's side, 1 where it points away.

    Foot 0 is the +x-sway foot, the right one under ``config.LATERAL_POSITIVE_IS_RIGHT``.
    """
    right_foot = 0 if getattr(C, "LATERAL_POSITIVE_IS_RIGHT", True) else 1
    on_right = np.asarray(foot, int) == right_foot
    return np.where(on_right == np.asarray(is_right, bool), 0, 1)


# ===========================================================================
# Trial extraction
# ===========================================================================
def _recording_trials(rec) -> dict | None:
    """Per-trial stance foot, stride id, within-step phase, accuracy and log RT of one
    walking recording.

    A stride is a reference-foot step plus the following step. Ids come from
    ``cumsum(step_foot == 0)`` rather than pairing steps, so a non-alternating step (e.g.
    from a missed trough) shifts only one stride boundary.
    """
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        gait_mod.gait_recording(rec)
    if not npz.exists():
        return None
    g = np.load(npz)
    troughs = g["troughs"]
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    if step_foot is None or np.size(troughs) < 3:
        return None
    step_foot = np.asarray(step_foot, int)
    margin = (np.abs(g["step_margin"]) if "step_margin" in g and np.size(g["step_margin"])
              else np.full(step_foot.size, np.inf))

    gait_json = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.json"
    verdict = "unknown"
    if gait_json.exists():
        verdict = json.loads(gait_json.read_text(encoding="utf-8")).get(
            "foot_label_verdict", "unknown")
    if str(verdict).startswith("unreliable") or str(verdict) == "unusable":
        return {"excluded": verdict}

    ev = bids_io.load_events(rec)
    if ev.empty:
        return None
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna()].copy()
    if ev.empty:
        return None
    onset = ev["onset"].to_numpy(float)

    i = np.searchsorted(troughs, onset) - 1        # step containing each onset
    inside = (i >= 0) & (i < step_foot.size)
    i_safe = np.clip(i, 0, step_foot.size - 1)
    foot = np.where(inside, step_foot[i_safe], -1)
    conf = np.where(inside, margin[i_safe], np.nan)
    # a new stride starts at each reference-foot (label 0) step
    stride_of_step = np.cumsum(step_foot == 0) - 1
    stride = np.where(inside, stride_of_step[i_safe], -1)
    span = np.where(inside, troughs[np.clip(i_safe + 1, 0, troughs.size - 1)] - troughs[i_safe],
                    np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_step = np.where(inside & (span > 0), 100.0 * (onset - troughs[i_safe]) / span, np.nan)

    responded = ev["response"].notna().to_numpy()
    # accuracy over the answered trials: a timeout is NaN and stance_contrast drops it
    acc = np.where(responded, np.where(ev["response_correct"].to_numpy() == 1, 1.0, 0.0),
                   np.nan)
    rt = ev["response_time"].to_numpy(float)
    logrt = np.full(acc.size, np.nan)
    good = responded & (acc == 1.0) & (rt > pa.RT_MIN_S) & np.isfinite(rt)
    if good.any():
        lr = np.log10(rt[good])
        med = np.median(lr)
        mad = 1.4826 * np.median(np.abs(lr - med)) or 1e-9
        keep = np.abs(lr - med) <= pa.RT_MAD_K * mad
        idx = np.where(good)[0][keep]
        logrt[idx] = np.log10(rt[idx])

    is_right = (ev["stimulus_side"].astype(str).str.lower() == "right").to_numpy(bool)
    said_right = np.where(responded,
                          (ev["response"].astype(str).str.lower() == "right").to_numpy(float),
                          np.nan)

    keep = inside & (foot >= 0)
    return {k: v[keep] for k, v in
            {"foot": foot, "stride": stride, "pct_step": pct_step, "acc": acc,
             "logrt": logrt, "confidence": conf, "is_right": is_right,
             "said_right": said_right}.items()}


def _pool_subject(recs) -> dict | None:
    """Concatenate a subject's walking runs.

    Stride ids are offset per run so strides from different runs stay distinct.
    """
    keys = ("foot", "stride", "pct_step", "acc", "logrt", "confidence", "is_right",
            "said_right")
    pools: dict[str, list] = {k: [] for k in keys}
    offset = 0
    excluded = []
    for rec in recs:
        tr = _recording_trials(rec)
        if tr is None:
            continue
        if "excluded" in tr:
            excluded.append((rec.label(), tr["excluded"]))
            continue
        shifted = tr["stride"] + offset
        for k in keys:
            pools[k].append(shifted if k == "stride" else tr[k])
        if shifted.size:
            offset = int(shifted.max()) + 2
    if not pools["foot"]:
        return None
    out = {k: np.concatenate(v) for k, v in pools.items()}
    out["excluded_runs"] = excluded
    return out


# ===========================================================================
# Per-participant analysis
# ===========================================================================
def analyse_subject(tr: dict, *, n_perm: int = N_PERM,
                    rng: np.random.Generator | None = None) -> dict | None:
    """Accuracy (primary) and log-RT (secondary) stance-foot contrasts for one subject, with
    balance checks and a sensitivity analysis on confidently labelled steps."""
    rng = rng or np.random.default_rng(RNG_SEED)
    acc = stance_contrast(tr["acc"], tr["foot"], tr["stride"], n_perm=n_perm, rng=rng)
    if acc is None:
        return None
    rt = stance_contrast(tr["logrt"], tr["foot"], tr["stride"], n_perm=n_perm, rng=rng)

    # Sensitivity: same test on confidently labelled steps only. A real effect should
    # remain; one that disappears may come from mislabelled steps.
    conf_ok = np.asarray(tr["confidence"], float) >= gait_mod.AMBIGUOUS_MARGIN
    acc_conf = (stance_contrast(tr["acc"][conf_ok], tr["foot"][conf_ok], tr["stride"][conf_ok],
                                n_perm=n_perm, rng=rng) if conf_ok.sum() > 2 * MIN_TRIALS_PER_FOOT
                else None)

    # Balance check: within-step phase coverage per foot. The permutation null already
    # protects the p-value; reported for transparency.
    bal = {}
    for k in (0, 1):
        m = (np.asarray(tr["foot"]) == k) & np.isfinite(tr["pct_step"])
        ang = 2 * np.pi * np.asarray(tr["pct_step"], float)[m] / 100.0
        z = np.mean(np.exp(1j * ang)) if ang.size else np.nan
        bal[f"foot{k}_step_phase_mean_pct"] = (float((np.angle(z) % (2 * np.pi))
                                                     / (2 * np.pi) * 100.0)
                                               if ang.size else None)
        bal[f"foot{k}_step_phase_R"] = float(np.abs(z)) if ang.size else None

    # Response direction: the criterion by stance foot, and tilt towards vs away from it.
    towards = towards_label(tr["foot"], tr["is_right"])
    response = {
        "criterion": criterion_contrast(tr["is_right"], tr["said_right"], tr["foot"],
                                        tr["stride"], n_perm=n_perm, rng=rng),
        "towards_accuracy": stance_contrast(tr["acc"], towards, tr["stride"],
                                            n_perm=n_perm, rng=rng),
        "towards_logrt": stance_contrast(tr["logrt"], towards, tr["stride"],
                                         n_perm=n_perm, rng=rng),
    }

    return {
        "accuracy": acc, "reaction_time": rt, "accuracy_confident_steps": acc_conf,
        "response": response,
        "balance": bal, "n_trials": int(np.size(tr["foot"])),
        "pct_confident_steps": round(float(100 * np.mean(conf_ok)), 1),
        "foot_names": [gait_mod.foot_name(0), gait_mod.foot_name(1)],
        "anatomical": bool(getattr(C, "LATERAL_SIGN_VERIFIED", False)),
    }


# ===========================================================================
# Population tests
# ===========================================================================
def population_test(nulls: list[np.ndarray], deltas: list[float]) -> dict:
    """Direction-agnostic and direction-specific population tests against nulls from the
    participants' own permutations.

    Each contrast is divided by the SD of its own null so no participant dominates, as in
    ``population``.
    """
    k = min(len(n) for n in nulls)
    z = np.array([d / (np.nanstd(n) or np.nan) for d, n in zip(deltas, nulls)])
    zn = np.vstack([np.asarray(n[:k], float) / (np.nanstd(n) or np.nan) for n in nulls])
    good = np.isfinite(z)
    z, zn = z[good], zn[good]

    s_obs = float(np.nansum(z ** 2))
    s_null = np.nansum(zn ** 2, axis=0)
    t_obs = float(np.nanmean(z))
    t_null = np.nanmean(zn, axis=0)

    p_agn = float((1 + np.sum(s_null >= s_obs)) / (1 + s_null.size))
    p_com = _two_sided_p(t_obs, t_null)
    try:
        from scipy import stats
        p_t = float(stats.ttest_1samp(np.asarray(deltas, float), 0.0).pvalue)
    except Exception:
        p_t = float("nan")
    return {
        "n_participants": int(z.size),
        "agnostic_S": s_obs, "agnostic_p": p_agn,
        "agnostic_significant": bool(p_agn < 0.05),
        "common_T": t_obs, "common_p": p_com,
        "common_significant": bool(p_com < 0.05),
        "common_ttest_p_on_raw_delta": p_t,
        "anatomical": bool(getattr(C, "LATERAL_SIGN_VERIFIED", False)),
        "_s_null": s_null, "_t_null": t_null, "_z": z,
    }


# ===========================================================================
# Figures
# ===========================================================================
def _figure(label: str, tr: dict, res: dict, out: Path):
    acc, rt = res["accuracy"], res["reaction_time"]
    names = res["foot_names"]
    cols = (S.BLUE, S.ORANGE)
    fig, ax = plt.subplots(1, 4, figsize=(16.0, 4.3),
                           gridspec_kw={"width_ratios": [1.0, 1.0, 1.25, 1.5]})
    fig.suptitle(f"Stance-foot contrast — {S.nice_label(label)}  "
                 f"({res['n_trials']} walking trials)",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")

    # (a) accuracy per foot, Wilson CI
    ks = [int(round(acc["mean_foot0"] * acc["n_foot0"])), int(round(acc["mean_foot1"] * acc["n_foot1"]))]
    ns = [acc["n_foot0"], acc["n_foot1"]]
    ms = [acc["mean_foot0"], acc["mean_foot1"]]
    err = np.array([[m - _wilson(k, n)[0] for m, k, n in zip(ms, ks, ns)],
                    [_wilson(k, n)[1] - m for m, k, n in zip(ms, ks, ns)]])
    ax[0].bar([0, 1], ms, color=cols, width=0.62)
    ax[0].errorbar([0, 1], ms, yerr=err, fmt="none", ecolor=S.INK, capsize=4, lw=1.2)
    for i, n in enumerate(ns):
        ax[0].annotate(f"n={n}", (i, 0.02), ha="center", va="bottom", fontsize=8, color="white")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 1.14)
    S.reference_line(ax[0], y=0.5, label="chance")
    ax[0].legend(loc="upper center", fontsize=8)
    S.panel(ax[0], "a", f"Accuracy  (Δ={100 * acc['delta']:+.1f} pp)", fontsize=10)

    # (b) RT per foot as points with CI; bars would need a truncated axis for a difference
    # of tens of ms on ~0.7 s and would exaggerate it.
    if rt:
        gm, err = [], []
        for k in (0, 1):
            v = np.asarray(tr["logrt"], float)[np.asarray(tr["foot"]) == k]
            v = v[np.isfinite(v)]
            gm.append(10 ** np.mean(v))
            se = np.std(v, ddof=1) / np.sqrt(max(v.size, 1)) if v.size > 1 else 0.0
            err.append([gm[-1] - 10 ** (np.mean(v) - 1.96 * se),
                        10 ** (np.mean(v) + 1.96 * se) - gm[-1]])
        err = np.asarray(err).T
        for k in (0, 1):
            ax[1].errorbar([k], [gm[k]], yerr=err[:, k:k + 1], fmt="o", ms=7,
                           color=cols[k], ecolor=cols[k], capsize=5, lw=1.6)
        ax[1].set_ylabel("geometric-mean RT (s)")
        ax[1].set_xlim(-0.6, 1.6)
        S.panel(ax[1], "b", f"Reaction time  (p={rt['p']:.3f})", fontsize=10)
    else:
        ax[1].text(0.5, 0.5, "too few clean RTs", ha="center", va="center",
                   transform=ax[1].transAxes, color=S.GREY, fontsize=9)
        S.panel(ax[1], "b", "Reaction time", fontsize=10)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(names)

    # (c) stride-flip null and observed value
    ax[2].hist(100 * acc["_null"], bins=40, color=S.MIST, edgecolor="white",
               label="stride-flip null")
    ax[2].axvline(100 * acc["delta"], color=S.RED, lw=2.2,
                  label=f"observed  p={acc['p']:.3f}")
    ax[2].set_xlabel("Δ accuracy (percentage points)"); ax[2].set_ylabel("permutations")
    ax[2].legend(loc="upper right", fontsize=8)
    S.panel(ax[2], "c", "Permutation test", fontsize=10)

    # (d) accuracy across the step, per foot
    for k in (0, 1):
        m = (np.asarray(tr["foot"]) == k) & np.isfinite(np.asarray(tr["acc"], float))
        cen, binned, sem = pa._bin(np.asarray(tr["pct_step"])[m], np.asarray(tr["acc"])[m],
                                   N_STEP_BINS, "mean")
        ax[3].plot(cen, binned, color=cols[k], lw=1.9, label=names[k])
        ax[3].fill_between(cen, binned - sem, binned + sem, color=cols[k], alpha=0.16, lw=0)
    ax[3].set_xlabel("% of step (0 = double support)"); ax[3].set_ylabel("accuracy")
    ax[3].legend(loc="lower right", fontsize=8, ncol=2)
    S.panel(ax[3], "d", "Accuracy through the step, by stance foot", fontsize=10)

    note = ("Null flips the foot label of whole strides, so each trial keeps its "
            "within-step phase and its time on task — a 2 cps oscillation or a drift "
            "across the block cannot produce this effect. ")
    note += ("Left/right is anatomical (sway sign verified for this rig)."
             if res["anatomical"] else
             "Feet are named A/B: they are separated reliably, but the sway-sign "
             "convention is not yet verified, so the SIGN of Δ has no anatomical meaning.")
    S.caption(fig, note, wrap=155)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    S.save(fig, out)


def _population_figure(rows: list[dict], pop: dict, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6),
                           gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    fig.suptitle(f"Stance-foot contrast — population (n = {pop['n_participants']} participants)",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")

    # (a) per-participant accuracy contrasts
    order = np.argsort([r["accuracy"]["delta"] for r in rows])
    ys = np.arange(len(order))
    for y, j in zip(ys, order):
        a = rows[j]["accuracy"]
        d, sd = 100 * a["delta"], 100 * a["null_sd"]
        col = S.RED if a["p"] < 0.05 else S.GREY
        ax[0].errorbar(d, y, xerr=1.959964 * sd, fmt="o", color=col, ecolor=col,
                       capsize=3, ms=5, lw=1.2)
    ax[0].axvline(0, color=S.INK, lw=1.0)
    ax[0].set_yticks(ys)
    ax[0].set_yticklabels([rows[j]["sub"] for j in order], fontsize=8)
    ax[0].set_xlabel("Δ accuracy (pp), foot A − foot B" if not pop["anatomical"]
                     else "Δ accuracy (pp), right − left")
    S.panel(ax[0], "a", "Per participant  (±1.96 null SD; red = p < .05)", fontsize=10)

    # (b) direction-agnostic statistic
    ax[1].hist(pop["_s_null"], bins=40, color=S.MIST, edgecolor="white", label="null")
    ax[1].axvline(pop["agnostic_S"], color=S.RED, lw=2.2,
                  label=f"observed  p={pop['agnostic_p']:.3f}")
    ax[1].set_xlabel("S = Σ zᵢ²"); ax[1].set_ylabel("draws")
    ax[1].legend(loc="upper right", fontsize=8)
    S.panel(ax[1], "b", "PRIMARY: direction-agnostic", fontsize=10)

    # (c) direction-specific statistic
    ax[2].hist(pop["_t_null"], bins=40, color=S.MIST, edgecolor="white", label="null")
    ax[2].axvline(pop["common_T"], color=S.RED, lw=2.2,
                  label=f"observed  p={pop['common_p']:.3f}")
    ax[2].set_xlabel("T = mean zᵢ"); ax[2].set_ylabel("draws")
    ax[2].legend(loc="upper right", fontsize=8)
    S.panel(ax[2], "c", "Secondary: same direction in everyone"
                        + ("" if pop["anatomical"] else "  [PROVISIONAL]"), fontsize=10)

    note = ("S is rotation-free, so it is valid whether or not the sway-sign convention "
            "has been verified. T is an anatomical claim: until "
            "config.LATERAL_SIGN_VERIFIED is set, only its magnitude is interpretable, "
            "not its sign.")
    S.caption(fig, note, wrap=150)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    S.save(fig, out)


RESPONSE_CONTRASTS = {
    "criterion": ("criterion c", 1.0, "Δc, foot A − foot B"),
    "towards_accuracy": ("accuracy, tilt towards vs away", 100.0, "Δ accuracy (pp), towards − away"),
    "towards_logrt": ("log RT, tilt towards vs away", 1000.0, "Δ log10 RT (x 1000), towards − away"),
}


def response_population(rows: list[dict]) -> dict:
    """Population tests of the three response-direction contrasts."""
    out = {}
    for key in RESPONSE_CONTRASTS:
        have = [r for r in rows if (r.get("response") or {}).get(key)]
        if len(have) < 2:
            continue
        out[key] = population_test([r["response"][key]["_null"] for r in have],
                                   [r["response"][key]["delta"] for r in have])
        out[key]["n_significant_participants"] = int(sum(r["response"][key]["p"] < 0.05
                                                         for r in have))
    return out


def _response_figure(rows: list[dict], pop: dict, out: Path):
    keys = [k for k in RESPONSE_CONTRASTS if k in pop]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.0 * len(keys), 4.6), squeeze=False)
    for ax, key, tag in zip(axes[0], keys, "abc"):
        title, scale, xlabel = RESPONSE_CONTRASTS[key]
        have = [r for r in rows if (r.get("response") or {}).get(key)]
        d = np.array([r["response"][key]["delta"] for r in have]) * scale
        sd = np.array([r["response"][key]["null_sd"] for r in have]) * scale
        pv = [r["response"][key]["p"] for r in have]
        order = np.argsort(d)
        for y, j in enumerate(order):
            col = S.RED if pv[j] < 0.05 else S.GREY
            ax.errorbar(d[j], y, xerr=1.959964 * sd[j], fmt="o", color=col, ecolor=col,
                        capsize=3, ms=5, lw=1.2)
        ax.axvline(0, color=S.INK, lw=1.0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([have[j]["sub"] for j in order], fontsize=8)
        ax.set_xlabel(xlabel)
        p = pop[key]
        S.panel(ax, tag, f"{title}\nS={p['agnostic_S']:.1f} (p={p['agnostic_p']:.3f}), "
                         f"T={p['common_T']:+.2f} (p={p['common_p']:.3f})", fontsize=9.5)
    fig.suptitle(f"Response direction by stance foot (n = {len(rows)} participants)",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "EXPLORATORY. Per participant ±1.96 SD of the stride-flip null; red, p < .05. "
                   "S: direction-agnostic population statistic; T: same direction in everyone. "
                   "'Towards' = the tilt points to the side of the stance foot under the "
                   "current sway-sign convention; until it is verified, only S and |T| are "
                   "interpretable.", wrap=150)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    S.save(fig, out)


# ===========================================================================
# Driver
# ===========================================================================
def _json_safe(o):
    """Drop keys with a leading underscore (permutation arrays) and convert numpy types."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def stance_main():
    """Test the stance-foot contrast per participant and across participants."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    if not recs:
        print("No walking recordings found. Run the gait stage first.")
        return
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)

    out_dir = bids_io.deriv_dir("stance_foot")
    rows = []
    print("=== Stance-foot contrast (EXPLORATORY) ===")
    if not getattr(C, "LATERAL_SIGN_VERIFIED", False):
        print("  NOTE: the rig's sway-sign convention is unverified, so the two feet are "
              "reported as A/B.\n        The direction-agnostic population test is "
              "unaffected; the direction-specific one is PROVISIONAL.")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        tr = _pool_subject(subrecs)
        if tr is None:
            continue
        for lbl, why in tr.get("excluded_runs", []):
            print(f"  {lbl}: SKIPPED - foot labels {why}")
        # crc32, not hash(): str hashes are salted per process, so hash() gave a different
        # permutation null (and p-value) on every run.
        res = analyse_subject(tr, rng=np.random.default_rng(zlib.crc32(f"{sub}_{ses}".encode())))
        if res is None:
            print(f"  {sub} {ses}: too few trials on one foot - skipped")
            continue
        res.update({"sub": sub, "ses": ses})
        rows.append(res)
        label = f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking"
        (out_dir / f"{label}_stance_foot.json").write_text(
            json.dumps(_json_safe(res), indent=2), encoding="utf-8")
        _figure(label, tr, res, bids_io.fig_dir("stance_foot") / f"{label}_stance_foot.png")
        a = res["accuracy"]
        sens = res["accuracy_confident_steps"]
        print(f"  {sub} {ses}: acc {a['mean_foot0']:.3f} vs {a['mean_foot1']:.3f} "
              f"(delta={100 * a['delta']:+.2f} pp, p={a['p']:.3f})"
              + (f" | confident-steps delta={100 * sens['delta']:+.2f} pp, p={sens['p']:.3f}"
                 if sens else "")
              + (f" | RT p={res['reaction_time']['p']:.3f}" if res["reaction_time"] else ""))

    bids_io.gate_report("stance_foot", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("stance_foot",))
    if len(rows) < 2:
        bids_io.clear_stale("stance_foot", "stance_foot_population.json",
                            "stance_foot_population.png",
                            "stance_foot_response_population.json",
                            "stance_foot_response_population.png")
    if len(rows) >= 2:
        pop = population_test([r["accuracy"]["_null"] for r in rows],
                              [r["accuracy"]["delta"] for r in rows])
        pop["note"] = ("EXPLORATORY. Stride-flip permutation null. The confirmatory tests "
                       "remain phase_analysis (2/4 cps) and the phase-agnostic population test.")
        (out_dir / "stance_foot_population.json").write_text(
            json.dumps(_json_safe(pop), indent=2), encoding="utf-8")
        _population_figure(rows, pop, bids_io.fig_dir("stance_foot") / "stance_foot_population.png")
        print(f"\n  POPULATION (n={pop['n_participants']}):")
        print(f"    PRIMARY  direction-agnostic : S={pop['agnostic_S']:.2f}, "
              f"p={pop['agnostic_p']:.4f} -> "
              f"{'ASYMMETRY PRESENT' if pop['agnostic_significant'] else 'n.s.'}")
        print(f"    secondary same-direction    : T={pop['common_T']:+.3f}, "
              f"p={pop['common_p']:.4f} -> "
              f"{'CONSISTENT DIRECTION' if pop['common_significant'] else 'n.s.'}"
              + ("" if pop["anatomical"] else "  [PROVISIONAL: sway sign unverified]"))
        rpop = response_population(rows)
        if rpop:
            (out_dir / "stance_foot_response_population.json").write_text(
                json.dumps(_json_safe({"note": "EXPLORATORY. Response direction by stance "
                                               "foot; stride-flip null.", **rpop}), indent=2),
                encoding="utf-8")
            _response_figure(rows, rpop, bids_io.fig_dir("stance_foot")
                             / "stance_foot_response_population.png")
            print("\n  RESPONSE DIRECTION:")
            for key, p in rpop.items():
                print(f"    {RESPONSE_CONTRASTS[key][0]:<32}: S={p['agnostic_S']:.2f} "
                      f"(p={p['agnostic_p']:.4f}), T={p['common_T']:+.3f} (p={p['common_p']:.4f})")
    print(f"\nOutputs -> {out_dir}")


# ###########################################################################
# Sensitivity vs criterion (SDT)
# ###########################################################################
# 20 bins: each bin holds two rates (hit, false alarm), so needs about twice the trials.
N_BINS = 20
SDT_N_PERM = 1000
FREQS = (2.0, 4.0)
STRIDE_FREQ = 1.0         # separate stride-rate test, outside the Holm family above
IDX_1CPS = int(np.argmin(np.abs(pa.FREQS - STRIDE_FREQ)))
ALPHA = 0.05
# Bins with fewer answered trials are dropped (only relevant for short recordings).
MIN_BIN_N = 8
MIN_TRIALS = 200          # minimum answered trials per participant
SDT_RNG_SEED = 23
MEASURES = ("dprime", "criterion")
LABEL = {"dprime": "sensitivity d'", "criterion": "criterion c"}
COLOR = {"dprime": S.BLUE, "criterion": S.PURPLE}

_Z_CLIP = 5.0             # clip for degenerate cells


def _z(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF without scipy.

    Acklam's rational approximation with one Halley step using ``erfc``. Maximum absolute
    error vs ``scipy.stats.norm.ppf`` over 20 000 points in [1e-6, 1 - 1e-6] is 1e-11.
    """
    import math
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        x[lo] = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        x[hi] = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                  / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        x[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                  / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))
    # one Halley refinement
    e = 0.5 * np.array([math.erfc(-v / math.sqrt(2)) for v in x.ravel()]).reshape(x.shape) - p
    u = e * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
    x = x - u / (1 + x * u / 2)
    return np.clip(x, -_Z_CLIP, _Z_CLIP)


# ===========================================================================
# Binned SDT
# ===========================================================================
def sdt_bins(phase, is_signal, said_signal, nbins=N_BINS, min_n=MIN_BIN_N):
    """Per-phase-bin ``(centers, d', c, hit rate, FA rate, n_bin, valid mask)``.

    Rates use the log-linear correction, so d' and c are always finite. ``valid`` marks
    bins with at least ``min_n`` answered trials and at least one trial per stimulus class.
    """
    phase = np.asarray(phase, float)
    sig = np.asarray(is_signal, bool)
    said = np.asarray(said_signal, bool)
    edges = np.linspace(0, 100, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)

    n_sig = np.bincount(idx[sig], minlength=nbins).astype(float)
    n_noi = np.bincount(idx[~sig], minlength=nbins).astype(float)
    n_hit = np.bincount(idx[sig & said], minlength=nbins).astype(float)
    n_fa = np.bincount(idx[~sig & said], minlength=nbins).astype(float)

    # Log-linear correction (Hautus 1995): +0.5 per count, +1 per denominator.
    hr = (n_hit + 0.5) / (n_sig + 1.0)
    fa = (n_fa + 0.5) / (n_noi + 1.0)
    zh, zf = _z(hr), _z(fa)
    dprime = zh - zf
    crit = -0.5 * (zh + zf)
    n_bin = n_sig + n_noi
    valid = (n_bin >= min_n) & (n_sig >= 1) & (n_noi >= 1)
    return centers, dprime, crit, hr, fa, n_bin, valid


def analyse_measure(phase, is_signal, said_signal, which: str, *,
                    nbins=N_BINS, nperm=SDT_N_PERM, rng=None) -> dict | None:
    """Confirmatory-style 2/4 cps test on the binned d' or criterion curve.

    As ``phase_analysis.analyse_measure``: weighted cosine fit, Holm over two frequencies,
    +1-corrected permutation p, and exported ``(cos, sin)`` coefficients for the population
    test.
    """
    rng = rng or np.random.default_rng(SDT_RNG_SEED)
    phase = np.asarray(phase, float)
    sig = np.asarray(is_signal, bool)
    said = np.asarray(said_signal, bool)

    centers, dp, cr, hr, fa, n_bin, valid = sdt_bins(phase, sig, said, nbins)
    if valid.sum() < 6:
        return None
    x = centers[valid]
    y = (dp if which == "dprime" else cr)[valid]
    # Bin counts are fixed across permutations, so weights are constant. Total bin n is
    # used instead of the SDT variance: with balanced stimulus classes it tracks precision,
    # and a data-dependent weight would change with each shuffle.
    w = n_bin[valid] if getattr(C, "WEIGHT_BINS_BY_N", True) else None

    obs, obs_beta = pa._spectrum_with_coef(x, y, w)
    i2 = pa.IDX_2CPS
    i4 = pa.IDX_4CPS
    i1 = IDX_1CPS
    obs_2, obs_4, obs_1 = float(obs[i2]), float(obs[i4]), float(obs[i1])

    null_2 = np.empty(nperm)
    null_4 = np.empty(nperm)
    null_1 = np.zeros(nperm)
    null_ab_2 = np.zeros((nperm, 2))
    null_ab_4 = np.zeros((nperm, 2))
    null_ab_1 = np.zeros((nperm, 2))
    n = phase.size
    for k in range(nperm):
        # shuffle (stimulus, response) pairs against fixed phases
        perm = rng.permutation(n)
        _, dpp, crp, _, _, _, vp = sdt_bins(phase, sig[perm], said[perm], nbins)
        use = valid & vp
        if use.sum() < 6:
            null_2[k] = null_4[k] = 0.0
            continue
        yp = (dpp if which == "dprime" else crp)[use]
        wp = n_bin[use] if w is not None else None
        spec, beta = pa._spectrum_with_coef(centers[use], yp, wp)
        null_2[k], null_4[k], null_1[k] = float(spec[i2]), float(spec[i4]), float(spec[i1])
        null_ab_2[k] = beta[i2, 1:]
        null_ab_4[k] = beta[i4, 1:]
        null_ab_1[k] = beta[i1, 1:]

    p_2 = float((1 + np.sum(null_2 >= obs_2)) / (1 + nperm))
    p_4 = float((1 + np.sum(null_4 >= obs_4)) / (1 + nperm))
    p_1 = float((1 + np.sum(null_1 >= obs_1)) / (1 + nperm))
    p_2h, p_4h = (float(v) for v in pa.holm_bonferroni([p_2, p_4]))
    r2_2, amp_2, phi_2, _ = pa.fit_one(x, y, 2.0, w)
    r2_4, amp_4, phi_4, _ = pa.fit_one(x, y, 4.0, w)
    r2_1, amp_1, phi_1, _ = pa.fit_one(x, y, STRIDE_FREQ, w)
    return {
        "measure": which,
        "centers": centers, "binned": np.where(valid, dp if which == "dprime" else cr, np.nan),
        "hit_rate": hr, "fa_rate": fa, "bin_n": n_bin, "valid": valid,
        "mean": float(np.average(y, weights=w if w is not None else None)),
        "p_2cps": p_2, "p_4cps": p_4, "p_2cps_holm": p_2h, "p_4cps_holm": p_4h,
        "significant_2cps": bool(p_2h < ALPHA), "significant_4cps": bool(p_4h < ALPHA),
        "significant": bool(p_2h < ALPHA or p_4h < ALPHA),
        "r2_at_2cps": float(r2_2), "amp_at_2cps": float(amp_2),
        "preferred_phase_2cps_rad": float(phi_2),
        "r2_at_4cps": float(r2_4), "amp_at_4cps": float(amp_4),
        "preferred_phase_4cps_rad": float(phi_4),
        "conf_best_freq_cps": 2.0 if (p_2h, -obs_2) <= (p_4h, -obs_4) else 4.0,
        "coef_2cps": obs_beta[i2, 1:], "coef_4cps": obs_beta[i4, 1:],
        "null_coef_2cps": null_ab_2, "null_coef_4cps": null_ab_4,
        "_null_2": null_2, "_null_4": null_4,
        # stride rate: a separate test, not in the Holm family of 2 and 4 cps
        "p_1cps": p_1, "r2_at_1cps": float(r2_1), "amp_at_1cps": float(amp_1),
        "preferred_phase_1cps_rad": float(phi_1),
        "coef_1cps": obs_beta[i1, 1:], "null_coef_1cps": null_ab_1,
    }


# ===========================================================================
# Trial extraction
# ===========================================================================
def _sdt_recording_trials(rec):
    """``(phase, is_signal, said_signal)`` for one walking recording, or ``None``.

    Uses the phase assignment of ``phase_analysis``, so SDT and accuracy curves share the
    same phases.
    """
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    cont = pa.continuous_phase(g)
    tp = pa._trial_phases(rec, g["troughs"], step_foot, tg, cont)
    if tp is None:
        return None
    ev, _ = tp
    if "stimulus_side" not in ev.columns or "response" not in ev.columns:
        return None
    side = ev["stimulus_side"].astype(object)
    resp = ev["response"].astype(object)
    # timeouts are neither hits nor false alarms
    keep = resp.notna() & side.notna() & ev["phase_target"].notna()
    if keep.sum() < 20:
        return None
    return {
        "phase": ev.loc[keep, "phase_target"].to_numpy(float),
        "is_signal": (side[keep].str.lower() == "right").to_numpy(bool),
        "said_signal": (resp[keep].str.lower() == "right").to_numpy(bool),
        "n_timeout": int((~resp.notna()).sum()),
    }


def _sdt_pool_subject(recs):
    """Concatenate all walking runs of one participant."""
    parts = [t for t in (_sdt_recording_trials(r) for r in recs) if t]
    if not parts:
        return None
    return {
        "phase": np.concatenate([p["phase"] for p in parts]),
        "is_signal": np.concatenate([p["is_signal"] for p in parts]),
        "said_signal": np.concatenate([p["said_signal"] for p in parts]),
        "n_timeout": int(sum(p["n_timeout"] for p in parts)),
        "n_runs": len(parts),
    }


def sdt_analyse_subject(tr, *, rng=None) -> dict | None:
    """Both SDT measures for one participant's pooled walking trials."""
    if tr["phase"].size < MIN_TRIALS:
        return None
    rng = rng or np.random.default_rng(SDT_RNG_SEED)
    out = {"n_trials": int(tr["phase"].size), "n_runs": int(tr["n_runs"]),
           "n_timeout_excluded": int(tr["n_timeout"]),
           "n_signal": int(tr["is_signal"].sum()),
           "n_said_signal": int(tr["said_signal"].sum())}
    # Overall (unbinned) d' and c; a large |c| is a response bias.
    _, dp0, cr0, hr0, fa0, _, _ = sdt_bins(tr["phase"], tr["is_signal"],
                                           tr["said_signal"], nbins=1, min_n=0)
    out["overall_dprime"] = float(dp0[0])
    out["overall_criterion"] = float(cr0[0])
    out["overall_hit_rate"] = float(hr0[0])
    out["overall_fa_rate"] = float(fa0[0])
    for m in MEASURES:
        out[m] = analyse_measure(tr["phase"], tr["is_signal"], tr["said_signal"], m,
                                 rng=rng)
    return out if any(out.get(m) for m in MEASURES) else None


# ===========================================================================
# Figures
# ===========================================================================
def _curve(ax, res, which, color):
    m = res[which]
    if not m:
        ax.text(0.5, 0.5, "not estimable", ha="center", va="center",
                transform=ax.transAxes, color=S.GREY)
        return
    ok = np.isfinite(m["binned"])
    ax.scatter(m["centers"][ok], m["binned"][ok], s=14, color=S.MIST, zorder=1)
    ax.plot(m["centers"][ok], m["binned"][ok], "-", color=S.INK, lw=1, zorder=2)
    f = m["conf_best_freq_cps"]
    w = 2 * np.pi * f / 100.0
    Xo = np.column_stack([np.ones(ok.sum()), np.cos(w * m["centers"][ok]),
                          np.sin(w * m["centers"][ok])])
    rt = np.sqrt(m["bin_n"][ok])
    beta, *_ = np.linalg.lstsq(Xo * rt[:, None], m["binned"][ok] * rt, rcond=None)
    xf = np.linspace(0, 100, 200)
    Xf = np.column_stack([np.ones_like(xf), np.cos(w * xf), np.sin(w * xf)])
    ph = m[f"p_{int(f)}cps_holm"]
    ax.plot(xf, Xf @ beta, color=color, lw=2.6,
            label=f"{f:.0f} cps, R²={m[f'r2_at_{int(f)}cps']:.2f}, "
                  f"Holm p={ph:.3f}" + ("*" if ph < ALPHA else ""))
    if which == "criterion":
        S.reference_line(ax, y=0.0, label="no bias")
    ax.set_xlabel("% stride cycle")
    ax.set_ylabel(LABEL[which])
    # Room for the legend above the noisy per-bin trace.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.28 * (hi - lo))
    ax.legend(loc="upper center", fontsize=8, ncol=1, framealpha=0.9)


def _sdt_figure(label, res, out: Path):
    fig, axg = plt.subplots(2, 2, figsize=(11.5, 7.6))
    ax = axg.ravel()
    fig.suptitle(f"Sensitivity vs criterion over the stride — {S.nice_label(label)}  "
                 f"(n = {res['n_trials']} responded trials)",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")

    _curve(ax[0], res, "dprime", COLOR["dprime"])
    S.panel(ax[0], "a", "Sensitivity", fontsize=10)
    _curve(ax[1], res, "criterion", COLOR["criterion"])
    S.panel(ax[1], "b", "Criterion (decision bias)", fontsize=10)

    # c) hit and false-alarm rates
    m = res["dprime"] or res["criterion"]
    if m:
        ok = np.isfinite(m["binned"])
        ax[2].plot(m["centers"][ok], m["hit_rate"][ok], "-o", ms=3, color=S.GREEN,
                   lw=1.2, label="hit  P(say right | right)")
        ax[2].plot(m["centers"][ok], m["fa_rate"][ok], "-o", ms=3, color=S.RED,
                   lw=1.2, label="false alarm  P(say right | left)")
        ax[2].set_ylim(0, 1.22)              # room for the legend
        S.reference_line(ax[2], y=0.5)
        ax[2].set_xlabel("% stride cycle"); ax[2].set_ylabel("rate")
        ax[2].legend(loc="upper center", fontsize=8, ncol=2, framealpha=0.9)
    S.panel(ax[2], "c", "Underlying rates (log-linear corrected)", fontsize=10)

    # d) permutation nulls at the reported frequency
    drawn = False
    for which in MEASURES:
        mm = res[which]
        if not mm:
            continue
        f = int(mm["conf_best_freq_cps"])
        ns = mm[f"_null_{f}"]
        ax[3].hist(ns, bins=40, alpha=0.55, color=COLOR[which],
                   label=f"{LABEL[which]} null @ {f} cps")
        ax[3].axvline(mm[f"r2_at_{f}cps"], color=COLOR[which], lw=2.2, ls="-")
        drawn = True
    if drawn:
        ax[3].set_xlabel("R² under the stride-phase permutation null")
        ax[3].set_ylabel("permutations")
        ax[3].legend(fontsize=8)
    S.panel(ax[3], "d", "Permutation nulls (vertical line = observed)", fontsize=10)

    S.caption(fig, "EXPLORATORY. Signal = tilt right. Timeouts excluded (no response = "
                   "neither hit nor false alarm). Null shuffles the (stimulus, response) "
                   "pair against fixed stride phase, so both marginals and every trial "
                   "outcome are held; only when each outcome occurred is randomised. "
                   f"Overall d'={res['overall_dprime']:.2f}, c={res['overall_criterion']:+.3f}.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    S.save(fig, out)


def _sdt_population_figure(rows, pop, out: Path):
    fig, axg = plt.subplots(1, 4, figsize=(19.0, 4.6))
    subs = [r["sub"] for r in rows]
    ypos = np.arange(len(rows))

    # a) overall criterion per participant (response bias)
    crit = [r["overall_criterion"] for r in rows]
    axg[0].barh(ypos, crit, color=S.PURPLE, alpha=0.85)
    S.reference_line(axg[0], x=0.0, label="no bias")
    axg[0].set_yticks(ypos); axg[0].set_yticklabels(subs, fontsize=8)
    axg[0].set_xlabel("criterion c  (> 0: answers 'left' more often)")
    axg[0].legend(fontsize=8)
    S.panel(axg[0], "a", "Overall response bias (walking)", fontsize=10)

    # b/c) population nulls per measure
    for i, which in enumerate(MEASURES):
        ax = axg[i + 1]
        d = pop.get(which)
        if not d:
            ax.text(0.5, 0.5, "not estimable", ha="center", va="center",
                    transform=ax.transAxes, color=S.GREY)
            S.panel(ax, "bc"[i], LABEL[which], fontsize=10)
            continue
        best = d["best_freq"]
        r = d["by_freq"][best]
        ax.hist(r["_null_stat"], bins=40, color=S.MIST, edgecolor="white")
        ax.axvline(r["statistic"], color=COLOR[which], lw=2.2,
                   label=f"observed (Holm p={r['p_holm']:.3f}"
                         f"{'*' if r['significant'] else ''})")
        S.reference_line(ax, x=r["null_p95"], label="null 95th pct")
        ax.set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$" + f"  @ {best:.0f} cps")
        ax.set_ylabel("permutations")
        # Legend on the side away from the observed line, so it never sits on it.
        right = r["statistic"] < r["null_p95"]
        ax.legend(loc="upper right" if right else "upper left", fontsize=8)
        S.panel(ax, "bc"[i], f"Stride-phase modulation of {LABEL[which]}", fontsize=10)

    # d) the separate stride-rate test of the criterion
    ax = axg[3]
    sr = (pop.get("criterion") or {}).get("stride_rate")
    if sr:
        ax.hist(sr["_null_stat"], bins=40, color=S.MIST, edgecolor="white")
        ax.axvline(sr["statistic"], color=COLOR["criterion"], lw=2.2,
                   label=f"observed (p={sr['p_value']:.3f})")
        S.reference_line(ax, x=sr["null_p95"], label="null 95th pct")
        ax.set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$  @ 1 cps")
        ax.set_ylabel("permutations")
        ax.legend(loc="upper right" if sr["statistic"] < sr["null_p95"] else "upper left",
                  fontsize=8)
    S.panel(ax, "d", "Criterion at the stride rate (1 cps)", fontsize=10)

    fig.suptitle("Sensitivity vs response bias over the stride (exploratory), "
                 f"n = {len(rows)} walking participants",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "Tilt right counts as the signal; timeouts excluded. a: each participant's "
                   "criterion over all walking trials. b-c: the phase-agnostic population "
                   "statistic S (the same test as the confirmatory one) for d' and for c, at "
                   "whichever of 2 and 4 cps has the smaller Holm p; grey = S over shuffles of "
                   "the (stimulus, response) pairs across phases. d: the same statistic for c "
                   "at 1 cps, where a bias that flips with the stance foot would appear; a "
                   "separate test, outside the Holm family.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    S.save(fig, out)


# ===========================================================================
# Population
# ===========================================================================
def sdt_population_test(rows) -> dict:
    """Studentised phase-agnostic test per SDT measure, at 2 and 4 cps (Holm)."""
    out = {}
    for which in MEASURES:
        have = [r for r in rows if r.get(which)]
        if len(have) < 2:
            continue
        by_freq = {}
        for f in FREQS:
            key = f"{f:.0f}cps"
            obs = np.array([r[which][f"coef_{key}"] for r in have], float)
            nper = min(r[which][f"null_coef_{key}"].shape[0] for r in have)
            null = np.stack([r[which][f"null_coef_{key}"][:nper] for r in have])
            by_freq[f] = population.agnostic_test(obs, null)
        padj = pa.holm_bonferroni([by_freq[f]["p_value"] for f in FREQS])
        for f, adj in zip(FREQS, padj):
            by_freq[f]["p_holm"] = float(adj)
            by_freq[f]["significant"] = bool(adj < ALPHA)
        best = min(FREQS, key=lambda f: (by_freq[f]["p_holm"], -by_freq[f]["statistic"]))
        out[which] = {
            "by_freq": by_freq, "best_freq": best,
            "n_participants": int(len(have)),
            "significant_any": bool(any(by_freq[f]["significant"] for f in FREQS)),
            "stride_rate": stride_rate_test(have, which),
        }
    return out


def stride_rate_test(have: list[dict], which: str) -> dict:
    """The separate 1-cps test: phase-agnostic S and the common-phase Hotelling T2."""
    obs = np.array([r[which]["coef_1cps"] for r in have], float)
    nper = min(r[which]["null_coef_1cps"].shape[0] for r in have)
    null = np.stack([r[which]["null_coef_1cps"][:nper] for r in have])
    res = population.agnostic_test(obs, null)
    res["hotelling"] = population.hotelling_t2(obs)
    res["n_significant_participants"] = int(sum(r[which]["p_1cps"] < ALPHA for r in have))
    return res


def _sdt_json_safe(o):
    if isinstance(o, dict):
        # Keys may be float frequencies; convert to strings before filtering "_" keys.
        out = {}
        for k, v in o.items():
            ks = f"{k:.0f}cps" if isinstance(k, float) else str(k)
            if ks.startswith("_"):
                continue
            out[ks] = _sdt_json_safe(v)
        return out
    if isinstance(o, (list, tuple)):
        return [_sdt_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_sdt_json_safe(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


# ===========================================================================
def sdt_main():
    """Decompose the stride-phase effect into sensitivity and criterion."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    if not recs:
        print("No walking recordings found. Run the gait stage first.")
        return
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)

    out_dir = bids_io.deriv_dir("sdt")
    rows = []
    print("=== Sensitivity vs criterion over the stride (EXPLORATORY) ===")
    print("  Signal = tilt right. Timeouts excluded. 20 phase bins, log-linear rates.")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        tr = _sdt_pool_subject(subrecs)
        if tr is None:
            continue
        # crc32, not hash(): str hashes are salted per process, so hash() gave a different
        # permutation null (and p-value) on every run.
        res = sdt_analyse_subject(tr, rng=np.random.default_rng(
            zlib.crc32(f"{sub}_{ses}_sdt".encode())))
        if res is None:
            print(f"  {sub} {ses}: below {MIN_TRIALS} responded trials - skipped")
            continue
        res.update({"sub": sub, "ses": ses})
        rows.append(res)
        label = f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking"
        (out_dir / f"{label}_sdt.json").write_text(
            json.dumps(_sdt_json_safe(res), indent=2), encoding="utf-8")
        _sdt_figure(label, res, bids_io.fig_dir("sdt") / f"{label}_sdt_phase.png")
        bits = []
        for m in MEASURES:
            mm = res[m]
            if mm:
                f = int(mm["conf_best_freq_cps"])
                bits.append(f"{m} {f}cps R2={mm[f'r2_at_{f}cps']:.3f} "
                            f"p={mm[f'p_{f}cps_holm']:.3f}"
                            + ("*" if mm["significant"] else ""))
        print(f"  {sub} {ses}: n={res['n_trials']} (+{res['n_timeout_excluded']} timeouts), "
              f"d'={res['overall_dprime']:.2f}, c={res['overall_criterion']:+.3f}"
              + ("  |  " + "; ".join(bits) if bits else ""))

    bids_io.gate_report("sdt", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("sdt",))
    if len(rows) < 2:
        bids_io.clear_stale("sdt", "sdt_population.json", "sdt_population.png")
    if len(rows) >= 2:
        pop = sdt_population_test(rows)
        payload = {"n_participants": len(rows),
                   "note": "EXPLORATORY signal-detection decomposition. Confirmatory "
                           "tests remain phase_analysis and the phase-agnostic population test on accuracy.",
                   "measures": _sdt_json_safe(pop)}
        (out_dir / "sdt_population.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        _sdt_population_figure(rows, pop, bids_io.fig_dir("sdt") / "sdt_population.png")
        print(f"\n  POPULATION (n={len(rows)}):")
        for which in MEASURES:
            d = pop.get(which)
            if not d:
                continue
            f = d["best_freq"]
            r = d["by_freq"][f]
            print(f"    {LABEL[which]:<16}: S={r['statistic']:.2f} vs null mean "
                  f"{r['null_mean']:.2f} @ {f:.0f} cps, Holm p={r['p_holm']:.4f} -> "
                  f"{'MODULATED' if r['significant'] else 'n.s.'}")
            sr = d["stride_rate"]
            print(f"    {'':<16}  stride rate 1 cps: S={sr['statistic']:.2f}, p={sr['p_value']:.4f}; "
                  f"common phase F={sr['hotelling']['f']:.2f}, p={sr['hotelling']['p_value']:.4f}")
        d_sig = pop.get("dprime", {}).get("significant_any")
        c_sig = pop.get("criterion", {}).get("significant_any")
        if d_sig and not c_sig:
            print("    => reads as SENSORY: sensitivity oscillates, criterion does not.")
        elif c_sig and not d_sig:
            print("    => reads as DECISIONAL: the criterion oscillates, sensitivity does not.")
        elif d_sig and c_sig:
            print("    => BOTH move; the accuracy effect is not attributable to one locus.")
        else:
            print("    => neither reaches significance; no decomposition claim.")
    print(f"\nOutputs -> {out_dir}")


# ###########################################################################
# Head roll and the left/right response
# ###########################################################################
ROLL_N_PERM = 1000
WINDOW_POINTS = 5             # roll sampled at this many points across the on-screen time
DEFAULT_DURATION_S = 0.10     # the nominal presentation, if the measured one is missing
N_ROLL_BINS = 5               # participant-specific roll quintiles in the figure
PHASE_FREQS = (1.0, 2.0)      # stride-phase covariates of the second model


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
def logit_fit(X: np.ndarray, y: np.ndarray, beta0: np.ndarray | None = None,
              iters: int = 30, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood logistic regression by Newton-Raphson; ``(beta, covariance)``.

    A ridge of 1e-8 keeps the Hessian invertible when a block happens to be answered one
    way throughout; it is far below the information any real column carries.
    """
    beta = np.zeros(X.shape[1]) if beta0 is None else np.asarray(beta0, float).copy()
    ridge = 1e-8 * np.eye(X.shape[1])
    for _ in range(iters):
        p = expit(X @ beta)
        w = np.clip(p * (1 - p), 1e-9, None)
        h = X.T @ (X * w[:, None]) + ridge
        step = np.linalg.solve(h, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    p = expit(X @ beta)
    h = X.T @ (X * np.clip(p * (1 - p), 1e-9, None)[:, None]) + ridge
    return beta, np.linalg.inv(h)


def _phase_terms(phase_pct: np.ndarray) -> np.ndarray:
    """Cos and sin of the stride phase at each frequency in ``PHASE_FREQS``."""
    ang = 2 * np.pi * np.asarray(phase_pct, float) / 100.0
    return np.column_stack([f(k * ang) for k in PHASE_FREQS for f in (np.cos, np.sin)])


def _time_terms(time: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """Linear and quadratic trend of time within each block (block x [t, t^2])."""
    cols = []
    for b in range(blk.max() + 1):
        m = blk == b
        t = np.zeros(time.size)
        if m.sum() > 2 and np.std(time[m]) > 0:
            t[m] = (time[m] - time[m].mean()) / time[m].std()
        cols += [t * m, (t ** 2 - (t[m] ** 2).mean() if m.any() else 0.0) * m]
    return np.column_stack(cols)


def roll_effect(said_right, tilt, roll, block, phase=None, time=None, *,
                n_perm: int = ROLL_N_PERM, rng: np.random.Generator | None = None) -> dict | None:
    """Fit the response model and its residual-permutation null for one participant.

    ``phase`` (% stride) adds the stride-phase covariates and ``time`` (s) a quadratic
    trend within each block. Returns ``beta_roll``, ``beta_tilt``, the equivalent tilt
    ``g`` with a delta-method SE, the permutation p and ``z``, and the null (``_null``).
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(said_right, float)
    tilt = np.asarray(tilt, float)
    roll = np.asarray(roll, float)
    _, blk = np.unique(np.asarray(block), return_inverse=True)
    ok = np.isfinite(y) & np.isfinite(tilt) & np.isfinite(roll)
    for extra in (phase, time):
        if extra is not None:
            ok &= np.isfinite(np.asarray(extra, float))
    if ok.sum() < MIN_TRIALS:
        return None
    y, tilt, roll, blk = y[ok], tilt[ok], roll[ok], blk[ok]
    _, blk = np.unique(blk, return_inverse=True)
    nuis = np.eye(blk.max() + 1)[blk]
    if phase is not None:
        nuis = np.column_stack([nuis, _phase_terms(np.asarray(phase, float)[ok])])
    if time is not None:
        nuis = np.column_stack([nuis, _time_terms(np.asarray(time, float)[ok], blk)])
    X = np.column_stack([nuis, tilt, roll])
    i_t, i_r = X.shape[1] - 2, X.shape[1] - 1
    beta, cov = logit_fit(X, y)
    # Residual permutation: the part of roll the nuisance terms explain stays with its
    # trial, the rest is shuffled within blocks.
    coef, *_ = np.linalg.lstsq(nuis, roll, rcond=None)
    fitted = nuis @ coef
    resid = roll - fitted
    order = [np.flatnonzero(blk == b) for b in range(blk.max() + 1)]
    null = np.empty(n_perm)
    Xp = X.copy()
    for k in range(n_perm):
        r = resid.copy()
        for idx in order:
            r[idx] = r[rng.permutation(idx)]
        Xp[:, i_r] = fitted + r
        null[k] = logit_fit(Xp, y, beta0=beta, iters=8)[0][i_r]
    br, bt = float(beta[i_r]), float(beta[i_t])
    g = br / bt if bt != 0 else float("nan")
    grad = np.array([-br / bt ** 2, 1.0 / bt])            # d g / d (bt, br)
    sub = cov[np.ix_([i_t, i_r], [i_t, i_r])]
    g_se = float(np.sqrt(max(grad @ sub @ grad, 0.0)))
    sd = float(np.std(null))
    return {
        "n_trials": int(y.size), "n_blocks": int(blk.max() + 1),
        "beta_roll": br, "beta_roll_se": float(np.sqrt(cov[i_r, i_r])),
        "beta_tilt": bt, "equivalent_tilt": g, "equivalent_tilt_se": g_se,
        "p": float((1 + np.sum(np.abs(null - null.mean()) >= abs(br - null.mean())))
                   / (1 + n_perm)),
        "null_sd": sd, "z": br / sd if sd > 0 else float("nan"),
        "roll_sd_deg": float(np.std(roll)),
        "_null": null,
    }


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------
def roll_at(times: np.ndarray, tg: np.ndarray, roll: np.ndarray,
            durations: np.ndarray) -> np.ndarray:
    """Mean roll over ``[t, t + duration]`` per trial; NaN if any of it is unmeasured."""
    ok = np.isfinite(roll)
    if ok.sum() < 2:
        return np.full(np.size(times), np.nan)
    filled = np.interp(tg, tg[ok], roll[ok])
    frac = np.linspace(0.0, 1.0, WINDOW_POINTS)
    pts = np.asarray(times, float)[:, None] + np.asarray(durations, float)[:, None] * frac
    val = np.interp(pts, tg, filled).mean(1)
    good = np.interp(pts, tg, ok.astype(float)).min(1) > 0.999
    inside = (pts.min(1) >= tg[0]) & (pts.max(1) <= tg[-1])
    return np.where(good & inside, val, np.nan)


def _roll_recording_trials(rec) -> dict | None:
    """Answered, phased walking trials of one block with the roll during presentation."""
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if np.size(g["step_foot"]) else None
    tp = pa._trial_phases(rec, g["troughs"], step_foot, g["t"], pa.continuous_phase(g))
    if tp is None:
        return None
    ev, _ = tp
    ev = ev[ev["response"].notna() & ev["stimulus_side"].notna()]
    if ev.empty:
        return None
    rs = hk.roll_series(rec)
    if rs is None:
        return None
    tg, roll = rs
    fast = hk._band(roll, 1.0 / float(np.median(np.diff(tg))))
    dur = ev["duration"].to_numpy(float) if "duration" in ev else np.full(len(ev), np.nan)
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, DEFAULT_DURATION_S)
    side = ev["stimulus_side"].astype(str).str.lower().to_numpy()
    onset = ev["onset"].to_numpy(float)
    return {
        "said_right": (ev["response"].astype(str).str.lower() == "right").to_numpy(float),
        "tilt": np.where(side == "right", 1.0, -1.0) * ev["tilt_angle_deg"].to_numpy(float),
        "roll": roll_at(onset, tg, roll, dur),
        "roll_fast": roll_at(onset, tg, fast, dur),
        "phase": ev["phase_target"].to_numpy(float),
        "time": onset,
        "block": np.full(len(ev), rec.run or rec.label()),
    }


def _roll_pool_subject(recs) -> dict | None:
    """A participant's walking trials pooled over blocks, or None."""
    parts = [t for t in (_roll_recording_trials(r) for r in recs) if t]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def roll_analyse_subject(tr: dict, *, rng=None) -> dict | None:
    """The three models for one participant, plus the binned bias curve for the figure."""
    rng = rng or np.random.default_rng(0)
    fit_main = roll_effect(tr["said_right"], tr["tilt"], tr["roll"], tr["block"], rng=rng)
    if fit_main is None:
        return None
    adj = roll_effect(tr["said_right"], tr["tilt"], tr["roll"], tr["block"], tr["phase"],
                      tr["time"], rng=rng)
    fast = roll_effect(tr["said_right"], tr["tilt"], tr["roll_fast"], tr["block"], rng=rng)
    # Bias curve: response minus the prediction without roll, by roll quintile.
    ok = np.isfinite(tr["roll"]) & np.isfinite(tr["tilt"])
    y, tilt, roll = tr["said_right"][ok], tr["tilt"][ok], tr["roll"][ok]
    _, blk = np.unique(tr["block"][ok], return_inverse=True)
    X0 = np.column_stack([np.eye(blk.max() + 1)[blk], tilt])
    b0, _ = logit_fit(X0, y)
    p0 = expit(X0 @ b0)
    edges = np.quantile(roll, np.linspace(0, 1, N_ROLL_BINS + 1))
    idx = np.clip(np.digitize(roll, edges[1:-1]), 0, N_ROLL_BINS - 1)
    curve = {"roll": [float(roll[idx == k].mean()) for k in range(N_ROLL_BINS)],
             "residual": [float((y - p0)[idx == k].mean()) for k in range(N_ROLL_BINS)],
             # probability change per degree of roll if the roll reached the retina whole
             "slope_if_uncompensated": float(np.mean(p0 * (1 - p0)) * b0[-1])}
    return {"main": fit_main, "adjusted": adj, "fast": fast, "curve": curve}


MODELS = {"main": "roll", "adjusted": "roll, adjusted for stride phase and time in block",
          "fast": "gait-band roll (0.3-5 Hz)"}


def leave_one_out(nulls: list[np.ndarray], deltas: list[float]) -> list[float]:
    """Direction-agnostic population p with each participant left out in turn."""
    out = []
    for i in range(len(deltas)):
        keep = [j for j in range(len(deltas)) if j != i]
        out.append(population_test([nulls[j] for j in keep],
                                         [deltas[j] for j in keep])["agnostic_p"])
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _roll_figure(rows: list[dict], pops: dict, out: Path) -> None:
    """Figure: head roll against the threshold, the response bias by roll, the
    equivalent tilt per participant, and the population statistic."""
    pop = pops["main"]
    fig, ax = plt.subplots(1, 4, figsize=(S.COL2_IN * 2.15, 4.4),
                           gridspec_kw={"width_ratios": [1.1, 1.15, 1.15, 1.0]})
    subs = [r["sub"] for r in rows]
    ypos = np.arange(len(rows))

    # (a) how large the roll is against the threshold
    sd = [r["main"]["roll_sd_deg"] for r in rows]
    thr = [r.get("threshold_walking_deg") or np.nan for r in rows]
    ax[0].barh(ypos, sd, color=S.MIST, label="roll SD")
    ax[0].plot(thr, ypos, "o", color=S.TEAL, label="threshold")
    ax[0].set_yticks(ypos); ax[0].set_yticklabels(subs, fontsize=8)
    ax[0].set_xlabel("degrees")
    ax[0].set_xlim(0, 1.45 * np.nanmax(np.r_[sd, thr]))
    ax[0].legend(fontsize=7.5, loc="lower right")
    S.panel(ax[0], "a", "Head roll vs threshold", fontsize=10)

    # (b) residual response by roll quintile, against the uncompensated prediction
    xs = np.array([r["curve"]["roll"] for r in rows])
    ys = np.array([r["curve"]["residual"] for r in rows])
    mx, my = xs.mean(0), ys.mean(0)
    se = ys.std(0, ddof=1) / np.sqrt(len(rows))
    ax[1].errorbar(mx, my, yerr=se, fmt="o-", color=S.PURPLE, capsize=3, lw=1.6,
                   label="observed")
    slope = float(np.mean([r["curve"]["slope_if_uncompensated"] for r in rows]))
    xx = np.linspace(mx.min(), mx.max(), 10)
    ax[1].plot(xx, slope * xx, color=S.GREY, ls="--", lw=1.2, label="uncompensated, g = +1")
    ax[1].plot(xx, -slope * xx, color=S.GREY, ls=":", lw=1.2, label="g = -1")
    ax[1].axhline(0, color=S.INK, lw=0.8)
    ax[1].set_xlabel("head roll during presentation (deg)")
    ax[1].set_ylabel("P(say right) - model without roll")
    ax[1].legend(fontsize=7.5, loc="upper center")
    S.panel(ax[1], "b", "Response bias by head roll", fontsize=10)

    # (c) equivalent tilt per participant
    g = np.array([r["main"]["equivalent_tilt"] for r in rows])
    gse = np.array([r["main"]["equivalent_tilt_se"] for r in rows])
    cols = [S.RED if r["main"]["p"] < 0.05 else S.GREY for r in rows]
    for y0, gi, si, c in zip(ypos, g, gse, cols):
        ax[2].errorbar(gi, y0, xerr=1.96 * si, fmt="o", color=c, ecolor=c, capsize=3, ms=5)
    ax[2].axvline(0, color=S.INK, lw=0.8)
    for v in (-1, 1):
        S.reference_line(ax[2], x=v)
    ax[2].set_yticks(ypos); ax[2].set_yticklabels(subs, fontsize=8)
    ax[2].set_xlabel("equivalent tilt g (deg tilt per deg roll)")
    S.panel(ax[2], "c", "Per participant (±95 % CI)", fontsize=10)

    # (d) population null
    ax[3].hist(pop["_s_null"], bins=40, color=S.MIST, edgecolor="white")
    ax[3].axvline(pop["agnostic_S"], color=S.PURPLE, lw=2.2,
                  label=f"S = {pop['agnostic_S']:.1f}, p = {pop['agnostic_p']:.3f}")
    S.reference_line(ax[3], x=float(np.percentile(pop["_s_null"], 95)), label="null 95th pct")
    ax[3].set_xlabel("S = Σ zᵢ²"); ax[3].set_ylabel("permutations")
    ax[3].legend(fontsize=7.5, loc="upper right")
    S.panel(ax[3], "d", "Population, direction-agnostic", fontsize=10)

    fig.suptitle(f"Head roll and the left/right response, n = {len(rows)} walking participants",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "EXPLORATORY. The Gabor plane follows head position but not rotation, so "
                   "head roll rotates it on the retina. a: SD of head roll during "
                   "presentation against the walking 75 % tilt threshold. Logistic model per "
                   "participant: "
                   "P(say right) ~ block + signed tilt + roll, roll averaged over the "
                   "on-screen time. g = beta_roll / beta_tilt; |g| = 1 is a roll that "
                   "reaches the retina whole. Null: roll residuals permuted within blocks. "
                   f"Same-direction test T = {pop['common_T']:+.2f}, p = {pop['common_p']:.3f}. "
                   f"Adjusted for stride phase and time in block: S = "
                   f"{pops['adjusted']['agnostic_S']:.1f}, p = {pops['adjusted']['agnostic_p']:.3f}; "
                   f"gait-band roll only: S = {pops['fast']['agnostic_S']:.1f}, p = "
                   f"{pops['fast']['agnostic_p']:.3f}. Positive roll = right ear up (rig check).")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    S.save(fig, out)


def _roll_json_safe(o):
    """JSON-serialisable copy of a result, without private keys."""
    if isinstance(o, dict):
        return {k: _roll_json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_roll_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def _thresholds() -> dict:
    """Walking 75 % threshold per participant from the H5 threshold fit (``conditions``)."""
    f = bids_io.deriv_dir("conditions") / "threshold_by_condition.json"
    if not f.exists():
        return {}
    pp = json.loads(f.read_text(encoding="utf-8")).get("per_participant", {})
    return {s: (d.get("walking") or {}).get("thresh75_deg") for s, d in pp.items()}


def roll_main() -> dict | None:
    """Head roll at presentation against the left/right response, per participant."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)
    thr = _thresholds()
    out_dir = bids_io.deriv_dir("roll_bias")
    rows = []
    print("=== Head roll and the left/right response (EXPLORATORY) ===")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        tr = _roll_pool_subject(subrecs)
        if tr is None:
            continue
        res = roll_analyse_subject(tr, rng=np.random.default_rng(zlib.crc32(f"{sub}_roll".encode())))
        if res is None:
            print(f"  {sub}: fewer than {MIN_TRIALS} trials with a measured roll - skipped")
            continue
        res.update({"sub": sub, "ses": ses, "threshold_walking_deg": thr.get(sub)})
        rows.append(res)
        m, a, f = res["main"], res["adjusted"], res["fast"]
        print(f"  {sub}: n={m['n_trials']}, roll SD {m['roll_sd_deg']:.2f} deg | "
              f"g={m['equivalent_tilt']:+.3f} +- {m['equivalent_tilt_se']:.3f}, p={m['p']:.3f}"
              + (f" | adjusted g={a['equivalent_tilt']:+.3f}, p={a['p']:.3f}" if a else "")
              + (f" | fast g={f['equivalent_tilt']:+.3f}, p={f['p']:.3f}" if f else ""))
    bids_io.gate_report("roll_bias", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("roll_bias",))
    if len(rows) < 2:
        bids_io.clear_stale("roll_bias", "roll_bias_population.json", "roll_bias_population.png")
        return None
    pops = {}
    for key in MODELS:
        have = [r for r in rows if r.get(key)]
        nulls = [r[key]["_null"] for r in have]
        deltas = [r[key]["beta_roll"] for r in have]
        pops[key] = population_test(nulls, deltas)
        g = np.array([r[key]["equivalent_tilt"] for r in have])
        pops[key]["equivalent_tilt_mean"] = float(g.mean())
        pops[key]["equivalent_tilt_median"] = float(np.median(g))
        pops[key]["equivalent_tilt_ci95"] = _tci(g)
        pops[key]["n_positive"] = int((g > 0).sum())
        pops[key]["n_significant"] = int(sum(r[key]["p"] < 0.05 for r in have))
        pops[key]["equivalent_tilt_max_abs_upper95"] = float(np.max(
            np.abs(g) + 1.96 * np.array([r[key]["equivalent_tilt_se"] for r in have])))
        pops[key]["leave_one_out_agnostic_p_max"] = float(max(leave_one_out(nulls, deltas)))
    payload = {
        "note": "EXPLORATORY. Logistic model of the left/right response on signed tilt and "
                "head roll during presentation. Residual-permutation null within blocks.",
        "roll_sign_verified": bool(getattr(C, "HEAD_ROLL_SIGN_VERIFIED", False)),
        "n_participants": len(rows),
        "median_roll_sd_deg": float(np.median([r["main"]["roll_sd_deg"] for r in rows])),
        "population": _roll_json_safe(pops),
        "participants": _roll_json_safe(rows),
    }
    (out_dir / "roll_bias_population.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
    _roll_figure(rows, pops, bids_io.fig_dir("roll_bias") / "roll_bias_population.png")
    for key, lab in MODELS.items():
        p = pops[key]
        print(f"\n  POPULATION, {lab}: S={p['agnostic_S']:.2f} (p={p['agnostic_p']:.4f}, "
              f"leave-one-out max p={p['leave_one_out_agnostic_p_max']:.4f}), "
              f"T={p['common_T']:+.3f} (p={p['common_p']:.4f}); mean g="
              f"{p['equivalent_tilt_mean']:+.3f} [{p['equivalent_tilt_ci95'][0]:+.3f}, "
              f"{p['equivalent_tilt_ci95'][1]:+.3f}], {p['n_positive']}/{len(rows)} positive")
    print(f"\nOutputs -> {out_dir}")
    return payload


def _tci(g: np.ndarray) -> list[float]:
    """95 % t interval of the mean of ``g``."""
    from scipy import stats
    h = stats.t.ppf(0.975, g.size - 1) * g.std(ddof=1) / np.sqrt(g.size)
    return [float(g.mean() - h), float(g.mean() + h)]


# ###########################################################################
# Pipeline stage
# ###########################################################################
def main() -> None:
    """Pipeline stage ``response_bias``: stance foot, SDT, head roll."""
    stance_main()
    sdt_main()
    roll_main()


if __name__ == "__main__":
    main()
