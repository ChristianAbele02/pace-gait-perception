"""Exploratory: does performance depend on which foot is in stance?

The confirmatory analysis (``phase_analysis``, ``group_agnostic``) tests 2 and 4 cycles per
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

Population level, as in ``group_agnostic`` (Davidson's phase coherence is only R = 0.33,
so same-direction tests are weak):

  * direction-agnostic (primary): ``S = Σ_i z_i²``, ``z_i = Δ_i / sd(null_i)``. Tests for
    an asymmetry in any direction per participant. Valid even with the sway-sign convention
    unverified, since it only needs consistent foot labels within a participant.
  * direction-specific (secondary): ``T = mean_i z_i``. Tests for the same direction in
    everyone. This is an anatomical claim, reported as provisional until
    ``config.LATERAL_SIGN_VERIFIED`` is set; until then only its magnitude is meaningful.

Both nulls are built from the participants' own permutations.

Confirmatory tests remain ``phase_analysis`` (per participant, 2/4 cps) and
``group_agnostic`` (population).

Usage::

    python -m Analysis.stance_foot
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import bids_io
    from . import config as C
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import gait as gait_mod
    import phase_analysis as pa
    import plotstyle as S

S.apply()

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

    acc = np.where(ev["response_correct"].to_numpy() == 1, 1.0, 0.0)
    rt = ev["response_time"].to_numpy(float)
    responded = ev["response"].notna().to_numpy()
    logrt = np.full(acc.size, np.nan)
    good = responded & (acc == 1.0) & (rt > pa.RT_MIN_S) & np.isfinite(rt)
    if good.any():
        lr = np.log10(rt[good])
        med = np.median(lr)
        mad = 1.4826 * np.median(np.abs(lr - med)) or 1e-9
        keep = np.abs(lr - med) <= pa.RT_MAD_K * mad
        idx = np.where(good)[0][keep]
        logrt[idx] = np.log10(rt[idx])

    keep = inside & (foot >= 0)
    return {k: v[keep] for k, v in
            {"foot": foot, "stride": stride, "pct_step": pct_step, "acc": acc,
             "logrt": logrt, "confidence": conf}.items()}


def _pool_subject(recs) -> dict | None:
    """Concatenate a subject's walking runs.

    Stride ids are offset per run so strides from different runs stay distinct.
    """
    keys = ("foot", "stride", "pct_step", "acc", "logrt", "confidence")
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

    return {
        "accuracy": acc, "reaction_time": rt, "accuracy_confident_steps": acc_conf,
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
    ``group_agnostic``.
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
        m = np.asarray(tr["foot"]) == k
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


def main():
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
        res = analyse_subject(tr, rng=np.random.default_rng(abs(hash((sub, ses))) % 2**32))
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
                            "stance_foot_population.png")
    if len(rows) >= 2:
        pop = population_test([r["accuracy"]["_null"] for r in rows],
                              [r["accuracy"]["delta"] for r in rows])
        pop["note"] = ("EXPLORATORY. Stride-flip permutation null. The confirmatory tests "
                       "remain phase_analysis (2/4 cps) and group_agnostic.")
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
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
