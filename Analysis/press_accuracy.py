"""Exploratory: does accuracy depend on the stride phase at the button press?

The primary analysis reads the stride phase at target onset, where the stimulus is
sampled. A trial also has a second time point, the press, and the decision and motor output
end there. If walking modulated how well a response came out at the moment it was made
rather than how well the target was seen, accuracy would follow the phase at the press. This
module tests that, next to the same estimator locked to the onset.

Why not a Rayleigh test on the press phases of the correct trials. The presses
themselves cluster in the step (``press_phase.py``), so the correct presses cluster too, and
a Rayleigh test would be significant whatever accuracy does. What carries the question is
the difference between where the correct and the wrong presses fall. That difference is the
Fourier coefficient of the centred accuracy::

    z_k = mean_j (y_j - ybar) exp(i 2 pi k phase_j / 100)

(y = difficulty-adjusted accuracy, ``phase_analysis.difficulty_adjust``). For y = c + A cos(wx
- psi), ``z_k = (A/2) exp(i psi)``, so ``2|z_k|`` is the modulation in hit-rate units and the
angle of ``z_k`` its peak. It is the score-test direction of a logistic regression of
accuracy on cos and sin of the phase.

The null. Press phase is onset + RT, and RT and accuracy are linked (errors are
slower). The null shuffles each answered trial's (RT, accuracy) pair as a unit across the
trials of its block and adds the RT back to the fixed onsets, recomputing the press phases
with the block's own gait. That keeps the onset timing, the RT distribution and the
RT-accuracy link, and breaks only their pairing with the gait. The coefficient is centred on
its null mean, as in ``press_phase``. The onset-locked companion shuffles accuracy against
the fixed onset phases, which is the null of the confirmatory test.

What each locking can see. An oscillation locked to one time point appears in the other
locking attenuated by the spread of the RTs: by ``|mean exp(i 2 pi k RT / stride)|``, reported
per participant as ``rt_attenuation``. With RT SDs of 0.1-0.3 s and ~1.2 s strides that keeps
roughly half at 2 cps and a tenth at 4 cps, so the two lockings are complementary tests of
two loci rather than two tests of one effect.

Per participant: 2 and 4 cps, Holm over the pair, per locking. Population: the phase-agnostic
``S`` of ``group_agnostic.agnostic_test`` and the common-phase Hotelling T2 of
``common_phase.hotelling_t2``, each Holm over 2 and 4 cps, per locking. Only answered trials
with RT above ``phase_analysis.RT_MIN_S`` enter, in both lockings, so they compare the same
trials. The trial-level Bayesian model with the phase at the press is
``bayes_glmm.py --lock press``.

Usage::

    python -m Analysis.press_accuracy
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

try:
    from . import bids_io
    from . import common_phase as cph
    from . import config as C
    from . import group_agnostic as ga
    from . import phase_analysis as pa
    from . import plotstyle as S
    from . import press_phase as pp
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import common_phase as cph
    import config as C
    import group_agnostic as ga
    import phase_analysis as pa
    import plotstyle as S
    import press_phase as pp

S.apply()

N_PERM = 1000
FREQS = (2.0, 4.0)
LOCKS = ("press", "onset")
MIN_TRIALS = 200
N_BINS = 20
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Per block
# ---------------------------------------------------------------------------
def _block(rec, gait_out: dict[str, str]) -> dict | None:
    """Onsets, RTs, adjusted accuracy and the gait arrays to place presses, for one block."""
    if rec.condition != "walking" or rec.label() in gait_out:
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if np.size(g["step_foot"]) else None
    tg = g["t"] if np.size(g["t"]) else None
    tp = pa._trial_phases(rec, g["troughs"], step_foot, tg, pa.continuous_phase(g))
    if tp is None:
        return None
    ev, _ = tp
    rt = ev["rt"].to_numpy(float)
    y = ev["acc_adj"].to_numpy(float)
    keep = (ev["response"].notna().to_numpy() & np.isfinite(rt) & (rt > pa.RT_MIN_S)
            & np.isfinite(y))
    if keep.sum() < 10:
        return None
    troughs = np.asarray(g["troughs"], float)
    return {"onset": ev["onset"].to_numpy(float)[keep], "rt": rt[keep], "y": y[keep],
            "onset_phase": ev["phase_target"].to_numpy(float)[keep],
            "gait": (troughs, step_foot, tg, pa.continuous_phase(g)),
            # consecutive troughs bound a step; a stride is two
            "stride_s": 2.0 * float(np.median(np.diff(troughs)))}


def coefficients(phase: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``(..., n_freq, 2)`` (cos, sin) of mean (y - ybar) exp(i k phase) over finite phases.

    ``phase`` and ``y`` share their shape; the last axis is trials. ``ybar`` is taken over
    the same finite entries, so an unphased trial drops out of both.
    """
    phase = np.asarray(phase, float)
    y = np.broadcast_to(np.asarray(y, float), phase.shape)
    ok = np.isfinite(phase)
    n = np.maximum(ok.sum(-1), 1)
    ybar = np.where(ok, y, 0.0).sum(-1) / n
    yc = np.where(ok, y - ybar[..., None], 0.0)
    ang = 2 * np.pi * np.where(ok, phase, 0.0) / 100.0
    out = [np.stack([(yc * np.cos(k * ang)).sum(-1) / n, (yc * np.sin(k * ang)).sum(-1) / n], -1)
           for k in FREQS]
    return np.stack(out, -2)


# ---------------------------------------------------------------------------
# Per participant
# ---------------------------------------------------------------------------
def analyse_blocks(blocks: list[dict], lock: str, *, n_perm: int = N_PERM,
                   rng: np.random.Generator | None = None) -> dict | None:
    """Observed and shuffled accuracy coefficients at the press or onset phase, pooled over a
    participant's blocks (each trial weighted equally)."""
    rng = rng or np.random.default_rng(0)
    nf = len(FREQS)
    obs_sum, obs_n = np.zeros((nf, 2)), 0
    null_sum, null_n = np.zeros((n_perm, nf, 2)), np.zeros(n_perm)
    for b in blocks:
        perm = np.argsort(rng.random((n_perm, b["rt"].size)), axis=1)
        if lock == "press":
            ph = pp.press_phases(b["onset"], b["rt"], b["gait"])
            nph = pp.press_phases(np.broadcast_to(b["onset"], perm.shape), b["rt"][perm],
                                  b["gait"])
        else:
            ph = b["onset_phase"]
            nph = np.broadcast_to(ph, perm.shape)
        n_ok = int(np.isfinite(ph).sum())
        obs_sum += coefficients(ph, b["y"]) * n_ok
        obs_n += n_ok
        nok = np.isfinite(nph).sum(1)
        null_sum += coefficients(nph, b["y"][perm]) * nok[:, None, None]
        null_n += nok
    if obs_n < MIN_TRIALS:
        return None
    obs = obs_sum / obs_n
    null = null_sum / np.maximum(null_n, 1)[:, None, None]
    centre = null.mean(0)
    obs_c, null_c = obs - centre, null - centre
    res, p_raw = {"n_trials": obs_n}, []
    for i, k in enumerate(FREQS):
        stat = float((obs_c[i] ** 2).sum())
        p = float((1 + np.sum((null_c[:, i] ** 2).sum(1) >= stat)) / (1 + n_perm))
        z = complex(*obs_c[i])
        p_raw.append(p)
        res[f"{k:.0f}cps"] = {
            "p": p, "amplitude": float(2 * abs(z)),
            "null_amplitude_p95": float(2 * np.sqrt(np.percentile((null_c[:, i] ** 2).sum(1), 95))),
            "peak_pct_stride": float((np.angle(z) / (2 * np.pi * k) * 100.0) % (100.0 / k)),
            "coef": obs_c[i], "null_coef": null_c[:, i]}
    for k, ph in zip(FREQS, pa.holm_bonferroni(p_raw)):
        res[f"{k:.0f}cps"]["p_holm"] = float(ph)
    return res


def rt_attenuation(blocks: list[dict]) -> dict:
    """How much of an oscillation locked to one time point survives in the other locking."""
    rt = np.concatenate([b["rt"] for b in blocks])
    stride = float(np.median([b["stride_s"] for b in blocks]))
    return {f"{k:.0f}cps": float(abs(np.mean(np.exp(2j * np.pi * k * rt / stride))))
            for k in FREQS}


def binned_curve(blocks: list[dict], lock: str) -> np.ndarray:
    """Adjusted accuracy minus its mean in ``N_BINS`` stride bins (NaN for an empty bin)."""
    ph = np.concatenate([pp.press_phases(b["onset"], b["rt"], b["gait"]) if lock == "press"
                         else b["onset_phase"] for b in blocks])
    y = np.concatenate([b["y"] for b in blocks])
    ok = np.isfinite(ph)
    idx = np.minimum((ph[ok] / 100.0 * N_BINS).astype(int), N_BINS - 1)
    s = np.bincount(idx, y[ok] - y[ok].mean(), N_BINS)
    n = np.bincount(idx, minlength=N_BINS)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def population_test(rows: list[dict], lock: str) -> dict:
    """Phase-agnostic S and common-phase Hotelling T2 per frequency (Holm over 2 and 4)."""
    out = {}
    for k in FREQS:
        key = f"{k:.0f}cps"
        obs = np.array([r[lock][key]["coef"] for r in rows], float)
        nper = min(r[lock][key]["null_coef"].shape[0] for r in rows)
        null = np.stack([r[lock][key]["null_coef"][:nper] for r in rows])
        out[key] = {"agnostic": ga.agnostic_test(obs, null), "hotelling": cph.hotelling_t2(obs),
                    "n_significant_participants": int(sum(
                        r[lock][key]["p_holm"] < ALPHA for r in rows))}
    for test in ("agnostic", "hotelling"):
        holm = pa.holm_bonferroni([out[f"{k:.0f}cps"][test]["p_value"] for k in FREQS])
        for k, ph in zip(FREQS, holm):
            out[f"{k:.0f}cps"][test]["p_holm"] = float(ph)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _figure(rows: list[dict], pop: dict, out: Path) -> None:
    """Figure: binned accuracy at the onset and at the press, the press-locked
    coefficients per participant, and the population statistic per locking."""
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6),
                           gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    x = (np.arange(N_BINS) + 0.5) * 100.0 / N_BINS
    for lock, col, lab in (("onset", S.GREY, "at target onset"), ("press", S.TEAL, "at the press")):
        cur = 100 * np.vstack([r["curve"][lock] for r in rows])
        m = np.nanmean(cur, 0)
        se = np.nanstd(cur, 0, ddof=1) / np.sqrt(np.isfinite(cur).sum(0))
        ax[0].fill_between(x, m - se, m + se, color=col, alpha=0.22, lw=0)
        ax[0].plot(x, m, color=col, lw=2.0, label=lab)
    ax[0].axhline(0.0, color=S.INK, lw=0.8)
    for xs in (0, 50, 100):
        ax[0].axvline(xs, color=S.GREY, ls=":", lw=0.8)
    ax[0].set_xlabel("% of stride (0, 50 = double support)")
    ax[0].set_ylabel("accuracy − participant mean (pp)")
    ax[0].legend(fontsize=8, loc="upper right")
    S.panel(ax[0], "a", "Accuracy over the stride, by phase reference", fontsize=10)

    k = "2cps"
    Z = 100 * 2 * np.array([r["press"][k]["coef"] for r in rows])
    cols = [S.RED if r["press"][k]["p_holm"] < ALPHA else S.GREY for r in rows]
    ax[1].scatter(Z[:, 0], Z[:, 1], c=cols, s=28, zorder=3)
    mu = Z.mean(0)
    ax[1].plot(*mu, "x", color=S.INK, ms=9, mew=2)
    ell = cph._ellipse(np.cov(Z, rowvar=False), mu, Z.shape[0])
    if ell is not None:
        ax[1].plot(ell[:, 0], ell[:, 1], color=S.INK, ls="--", lw=1.0)
    ax[1].axhline(0, color=S.GREY, lw=0.7); ax[1].axvline(0, color=S.GREY, lw=0.7)
    lim = 1.15 * np.nanmax(np.abs(np.r_[Z.ravel(), ell.ravel() if ell is not None else []]))
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim); ax[1].set_aspect("equal")
    ax[1].set_xlabel("cos (pp)"); ax[1].set_ylabel("sin (pp)")
    h = pop["press"][k]["hotelling"]
    S.panel(ax[1], "b", f"At the press, 2 cps per participant\ncommon phase, Holm p="
            f"{h['p_holm']:.3f}", fontsize=10)

    labels, ypos = [], 0
    for lock in LOCKS:
        for f in FREQS:
            a = pop[lock][f"{f:.0f}cps"]["agnostic"]
            col = S.TEAL if lock == "press" else S.GREY
            ax[2].barh(ypos, a["statistic"], color=col, height=0.6)
            ax[2].plot([a["null_p95"]] * 2, [ypos - 0.4, ypos + 0.4], color=S.INK, lw=1.4)
            ax[2].text(max(a["statistic"], a["null_p95"]) + 0.4, ypos,
                       f"Holm p={a['p_holm']:.2f}", va="center", fontsize=8)
            labels.append((ypos, f"{lock}, {f:.0f} cps"))
            ypos -= 1
    ax[2].axvline(len(rows), color=S.GREY, ls="--", lw=0.9)
    ax[2].set_yticks([p for p, _ in labels]); ax[2].set_yticklabels([t for _, t in labels])
    ax[2].set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$")
    ax[2].set_xlim(0, 1.45 * max(ax[2].get_xlim()[1], 1.0))
    ax[2].grid(axis="y", visible=False)
    S.panel(ax[2], "c", "Population, phase-agnostic", fontsize=10)

    fig.suptitle(f"Accuracy and the stride phase at the press, n = {len(rows)} walking "
                 "participants", fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "EXPLORATORY. Answered trials with RT > 150 ms, difficulty-adjusted accuracy. "
                   "a: binned accuracy deviation, mean ± SEM over participants. b: centred "
                   "press-locked coefficients (x 2 = modulation), red = Holm p < .05; dashed, "
                   "95 % region of the mean. c: observed S per locking and frequency; black "
                   "tick, null 95th percentile; dashed, null mean (= n). Press-locked null: "
                   "(RT, accuracy) pairs shuffled within blocks and added to the fixed onsets.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    S.save(fig, out)


def main() -> dict | None:
    """Accuracy against the press phase and the onset phase, per participant and population."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    gait_out = bids_io.excluded_recordings(scope="gait")
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)
    out_dir = bids_io.deriv_dir("press_accuracy")
    rows = []
    print("=== Accuracy and the stride phase at the press (EXPLORATORY) ===")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        blocks = [b for b in (_block(r, gait_out) for r in subrecs) if b]
        if not blocks:
            continue
        res = {"sub": sub, "ses": ses}
        for lock in LOCKS:
            seed = zlib.crc32(f"{sub}_press_accuracy_{lock}".encode())
            res[lock] = analyse_blocks(blocks, lock, rng=np.random.default_rng(seed))
        if res["press"] is None or res["onset"] is None:
            print(f"  {sub}: fewer than {MIN_TRIALS} phased answered trials - skipped")
            continue
        res["rt_attenuation"] = rt_attenuation(blocks)
        res["curve"] = {lock: binned_curve(blocks, lock) for lock in LOCKS}
        rows.append(res)
        (out_dir / f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking_press_accuracy.json").write_text(
            json.dumps(pp._json_safe(res), indent=2), encoding="utf-8")
        print(f"  {sub}: {res['press']['n_trials']} trials | "
              + " | ".join(f"{lock} {k:.0f} cps {100 * res[lock][f'{k:.0f}cps']['amplitude']:.1f} pp "
                           f"(Holm p={res[lock][f'{k:.0f}cps']['p_holm']:.3f})"
                           for lock in LOCKS for k in FREQS)
              + f" | RT attenuation 2/4 cps {res['rt_attenuation']['2cps']:.2f}/"
                f"{res['rt_attenuation']['4cps']:.2f}")
    bids_io.gate_report("press_accuracy", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("press_accuracy",))
    if len(rows) < 2:
        bids_io.clear_stale("press_accuracy", "press_accuracy_population.json",
                            "press_accuracy_population.png")
        return None
    pop = {lock: population_test(rows, lock) for lock in LOCKS}
    payload = {"note": "EXPLORATORY. Difficulty-adjusted accuracy against the stride phase at "
                       "the press and at onset. Press-locked null: (RT, accuracy) pairs "
                       "shuffled within blocks and added to the fixed onsets; coefficients "
                       "centred on the null mean.",
               "n_participants": len(rows),
               "rt_attenuation_median": {k: float(np.median([r["rt_attenuation"][k]
                                                             for r in rows]))
                                         for k in ("2cps", "4cps")},
               "population": pp._json_safe(pop)}
    (out_dir / "press_accuracy_population.json").write_text(json.dumps(payload, indent=2),
                                                           encoding="utf-8")
    _figure(rows, pop, bids_io.fig_dir("press_accuracy") / "press_accuracy_population.png")
    print(f"\n  POPULATION (n={len(rows)}):")
    for lock in LOCKS:
        for k in FREQS:
            d = pop[lock][f"{k:.0f}cps"]
            a, h = d["agnostic"], d["hotelling"]
            print(f"    {lock:5s} {k:.0f} cps: S={a['statistic']:.2f} vs null mean "
                  f"{a['null_mean']:.2f}, Holm p={a['p_holm']:.3f} | common phase "
                  f"F={h['f']:.2f}, Holm p={h['p_holm']:.3f}")
    print(f"\nOutputs -> {out_dir}")
    return payload


if __name__ == "__main__":
    main()
