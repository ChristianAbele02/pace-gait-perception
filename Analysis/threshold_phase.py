"""Stride-phase modulation of the THRESHOLD, with every trial weighted by what it can show.

The confirmatory test asks whether accuracy oscillates over the stride. How far a given
oscillation of the threshold moves accuracy depends on where a trial sits on the
psychometric function. A threshold modulation by a factor ``exp(theta)`` moves accuracy
by ``dP/dln(alpha) * theta``. That slope is largest near 84 % correct and falls to zero at
chance and at ceiling. Each block presents its tilts at the calibrated threshold, and the
calibration does not always carry over (``conditions.py``). So the blocks sit
at different points of their participant's function, and their trials carry different
amounts of information about the same threshold modulation.

This stage measures that and runs the stride-phase test on the threshold scale:

1. Expected accuracy per trial. Each block's Weibull threshold is fitted to its
   answered trials with the QUEST+ slope, lapse and guess held fixed
   (``calibration_transfer`` grid). Each trial's expected accuracy ``P`` follows at its
   own tilt.
2. Information per trial. The Fisher information a trial carries about a threshold
   shift, ``w(P) = (dP/dln alpha)^2 / (P (1 - P))``, relative to a trial at 75 % (the
   design point): ``eta = w(P) / w(0.75)``. The sum of ``eta`` over a participant's trials
   is their effective number of trials at the design point.
3. Information-weighted test. Each trial enters as its efficient-score contribution
   ``v = h (y - m)``, with ``h = (dP/dln alpha) / (P (1 - P))``, and ``m`` the mean of
   its block x tilt-level cell, the same cells ``phase_analysis.difficulty_adjust``
   centres. The values then run through the confirmatory machinery unchanged
   (``phase_analysis.analyse_measure``): 40 stride bins, the occupancy-weighted Fourier
   fit at 2 and 4 cps, and 1000 trial-level permutations, Holm over the pair. A
   permutation moves values, never phases, so the test is exact whatever the true slope
   and lapse. A wrong model only makes the weights less than optimal.
4. Threshold units. ``E[v | phase] = -w * theta(phase)``, so ``theta = -z / mean(w)``
   turns a fitted coefficient ``z`` into a modulation of ln threshold. ``100 |theta|`` is
   the peak modulation in percent of the threshold.
5. Population. The studentised phase-agnostic sum of ``population.agnostic_test``
   on the weighted coefficients and their permutations, Holm over 2 and 4 cps.

Role: secondary. The confirmatory outcome stays accuracy (``config.PRIMARY_MEASURE``).
This stage asks the same question on the threshold scale, which (a) does not depend on
the operating point each block happened to run at, (b) states the effect as a threshold
modulation that compares across participants and studies, and (c) shows how much
information the blocks carried. Weighting cannot restore information a block at chance
or at ceiling never collected.

``P`` is clipped to ``[P_MIN, P_MAX]`` for the weights. Under the fixed lapse a block at
ceiling would otherwise get weight zero, although a participant with a smaller lapse
still carries information there.

Outputs (``derivatives/threshold_phase/``):
  * ``threshold_phase_blocks.tsv``        per walking block: THR accuracy, fitted
                                          threshold, information retained
  * ``threshold_phase_participants.tsv``  per participant: effective trials, threshold
                                          modulation at 2 and 4 cps, p-values of both tests
  * ``threshold_phase.json``              population test and settings
  * ``figures/threshold_phase.png`` / ``.pdf``

Usage::

    python -m Analysis.threshold_phase
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import bids_io
    from . import calibration as cref
    from . import conditions
    from . import config as C
    from . import gait as gait_mod
    from . import population as ga
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import calibration as cref
    import conditions
    import config as C
    import gait as gait_mod
    import population as ga
    import phase_analysis as pa
    import plotstyle as S

S.apply()

GUESS, LAPSE, BETA = conditions.GUESS, conditions.LAPSE_FIX, conditions.BETA_FIX
DESIGN_PC = conditions.THRESHOLD_TARGET                 # the THR level's planned accuracy
P_MIN, P_MAX = 0.55, 0.92              # range of P used for the weights
FREQS = ga.FREQS
ALPHA = 0.05
SEED = 20261027


# ===========================================================================
# Information a trial carries about a threshold shift
# ===========================================================================
def _u(pc):
    """Weibull ``(x / alpha)^beta`` at which the rig's function reaches ``pc``."""
    q = (np.asarray(pc, float) - GUESS) / (1 - GUESS - LAPSE)
    return -np.log1p(-np.clip(q, 1e-9, 1 - 1e-9))


def slope(pc):
    """``|dP / dln(alpha)|`` at accuracy ``pc``: how far accuracy moves per unit of ln threshold."""
    u = _u(pc)
    return (1 - GUESS - LAPSE) * BETA * u * np.exp(-u)


def information(pc):
    """Fisher information of one trial at accuracy ``pc`` about a shift of ln threshold."""
    pc = np.asarray(pc, float)
    return slope(pc) ** 2 / (pc * (1 - pc))


def efficiency(pc):
    """Information of a trial at ``pc`` relative to one at the 75 % design point."""
    return information(pc) / information(DESIGN_PC)


def score_weight(pc):
    """``h = (dP/dln alpha) / (P (1 - P))``: the efficient-score weight of a trial's residual."""
    pc = np.asarray(pc, float)
    return slope(pc) / (pc * (1 - pc))


def expected_accuracy(tilt, alpha):
    """Accuracy the rig's psychometric model predicts at ``tilt`` for threshold
    ``alpha``."""
    return cref.weibull(np.asarray(tilt, float), alpha, BETA, LAPSE, GUESS)


def block_alpha(tilt, correct) -> float:
    """Weibull threshold of one block's answered trials, slope/lapse/guess fixed (grid ML)."""
    return float(np.exp(conditions.GRID[int(np.argmin(conditions._nll_grid(tilt, correct)))]))


def weighted_values(y, cell, p_hat) -> np.ndarray:
    """``h(P) (y - m_cell)``: each trial's residual from its cell mean, information-weighted."""
    y = np.asarray(y, float)
    cell = np.asarray(cell)
    resid = np.empty_like(y)
    for c in np.unique(cell):
        k = cell == c
        resid[k] = y[k] - y[k].mean()
    return score_weight(np.clip(p_hat, P_MIN, P_MAX)) * resid


def to_threshold_units(coef, p_hat) -> np.ndarray:
    """Fitted coefficient(s) of the weighted values -> modulation of ln threshold."""
    return -np.asarray(coef, float) / float(np.mean(information(np.clip(p_hat, P_MIN, P_MAX))))


# ===========================================================================
# Data: the confirmatory test's own trials
# ===========================================================================
def _block_trials(rec) -> tuple[pd.DataFrame | None, dict | None]:
    """The phased, answered main trials of one walking block, with weights and values.

    The trials, phases and accuracies are ``phase_analysis._trial_phases``'s, so the
    test runs on exactly the trials of the confirmatory one. Cell means and the
    threshold fit use every answered main trial of the block, with or without a phase,
    as ``difficulty_adjust`` does.
    """
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        gait_mod.gait_recording(rec)
    if not npz.exists():
        return None, None
    g = np.load(npz)
    step_foot = g["step_foot"] if "step_foot" in g and np.size(g["step_foot"]) else None
    tg = g["t"] if "t" in g and np.size(g["t"]) else None
    tp = pa._trial_phases(rec, g["troughs"], step_foot, tg, pa.continuous_phase(g))
    if tp is None:
        return None, None

    ev = bids_io.load_events(rec)
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna() & ev["response"].notna()]
    if len(ev) < 20:
        return None, None
    tilt = pd.to_numeric(ev["tilt_angle_deg"], errors="coerce").abs().to_numpy(float)
    y = (ev["response_correct"] == 1).to_numpy(float)
    level = ev["difficulty"].astype(str).to_numpy()
    alpha = block_alpha(tilt, y)
    p_hat = expected_accuracy(tilt, alpha)
    v = weighted_values(y, level, p_hat)
    by_onset = pd.Series(np.arange(len(ev)), index=ev["onset"].to_numpy(float))

    phased = tp[0]
    phased = phased[phased["acc"].notna()]
    idx = by_onset.reindex(phased["onset"].to_numpy(float)).to_numpy()
    keep = np.isfinite(idx)
    idx = idx[keep].astype(int)
    out = pd.DataFrame({
        "recording": rec.label(), "difficulty": level[idx],
        "phase": phased["phase_target"].to_numpy(float)[keep],
        "acc": y[idx], "acc_adj": phased["acc_adj"].to_numpy(float)[keep],
        "value": v[idx], "p_hat": p_hat[idx],
        "eta": efficiency(np.clip(p_hat[idx], P_MIN, P_MAX))})
    thr = level == "THR"
    visit = (bids_io.scan_meta(rec) or {}).get("visit")
    info = {"recording": rec.label(), "participant": rec.sub,
            "visit": int(visit) if visit is not None else np.nan, "run": rec.run,
            "n_trials": int(len(out)),
            "thr_accuracy": round(float(y[thr].mean()), 4) if thr.any() else np.nan,
            "thr_tilt_deg": round(float(np.median(tilt[thr])), 4) if thr.any() else np.nan,
            "block_t75_deg": round(float(conditions._to_t75(alpha)), 4),
            "information_retained": round(float(out["eta"].mean()), 4) if len(out) else np.nan}
    return out, info


def _participant(sub: str, trials: pd.DataFrame) -> dict:
    """Both versions of the per-participant test on one participant's pooled walking trials.

    The module RNG of ``phase_analysis`` draws the permutations. It is swapped for a
    generator seeded on the participant label and restored afterwards, so this stage
    neither depends on nor changes the confirmatory stage's random stream.
    """
    saved = pa.RNG
    pa.RNG = np.random.default_rng([SEED, zlib.crc32(sub.encode())])
    try:
        m = pa.analyse_measure(trials["phase"].to_numpy(), trials["value"].to_numpy(), "mean")
    finally:
        pa.RNG = saved
    if m is None:
        return {}
    p_hat = trials["p_hat"].to_numpy()
    out = {"participant": sub, "n_trials": int(len(trials)),
           "effective_trials": round(float(trials["eta"].sum()), 1),
           "information_retained": round(float(trials["eta"].mean()), 4),
           "_m": m}
    for f in FREQS:
        tag = f"{f:.0f}cps"
        theta = to_threshold_units(m[f"coef_{tag}"], p_hat)
        null = to_threshold_units(m[f"null_coef_{tag}"], p_hat)
        out[f"theta_{tag}_pct"] = round(100 * float(np.hypot(*theta)), 3)
        out[f"theta_{tag}_null95_pct"] = round(100 * float(np.percentile(np.hypot(*null.T), 95)), 3)
        out[f"p_{tag}"] = m[f"p_{tag}"]
        out[f"p_{tag}_holm"] = m[f"p_{tag}_holm"]
        out[f"_theta_{tag}"] = theta
        out[f"_null_theta_{tag}"] = null
    out["significant"] = bool(m["significant"])
    return out


def _confirmatory_terms(sub: str, ses: str) -> dict:
    """The confirmatory accuracy test's studentised term ``|z|^2 / s^2`` per frequency,
    read from the pooled phase result it wrote."""
    f = bids_io.deriv_dir("phase", sub, ses) / f"{pa.subject_walk_label(sub, ses)}_phase.npz"
    if not f.exists():
        return {}
    out = {}
    with np.load(f) as z:
        for fr in FREQS:
            tag = f"{fr:.0f}cps"
            if f"accuracy_coef_{tag}" in z:
                obs, null = z[f"accuracy_coef_{tag}"], z[f"accuracy_null_coef_{tag}"]
                out[tag] = float((obs ** 2).sum() / (null ** 2).sum(1).mean())
                out[f"_obs_{tag}"] = np.asarray(obs, float)
    return out


def _studentised(m: dict, tag: str) -> float:
    """A participant's studentised statistic ``|z|^2 / s^2`` at frequency ``tag``."""
    obs, null = m[f"coef_{tag}"], m[f"null_coef_{tag}"]
    return float((obs ** 2).sum() / (null ** 2).sum(1).mean())


# ===========================================================================
# Stage
# ===========================================================================
def analyse() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Per-block information, per-participant tests and the population test."""
    by_sub: dict[tuple, list] = {}
    for rec in bids_io.find_recordings():
        if rec.condition == "walking":
            by_sub.setdefault((rec.sub, rec.ses), []).append(rec)
    blocks, parts, raw = [], [], {}
    for (sub, ses), recs in sorted(by_sub.items()):
        frames = []
        for rec in recs:
            tr, info = _block_trials(rec)
            if tr is not None and len(tr):
                frames.append(tr)
                blocks.append(info)
        if not frames:
            continue
        r = _participant(sub, pd.concat(frames, ignore_index=True))
        if not r:
            continue
        conf = _confirmatory_terms(sub, ses)
        for fr in FREQS:
            tag = f"{fr:.0f}cps"
            r[f"term_weighted_{tag}"] = round(_studentised(r["_m"], tag), 4)
            r[f"term_accuracy_{tag}"] = round(conf[tag], 4) if tag in conf else np.nan
        r["_conf"] = conf
        raw[sub] = r
        parts.append({k: v for k, v in r.items() if not k.startswith("_")})

    pop = {}
    if len(raw) >= 2:
        for fr in FREQS:
            tag = f"{fr:.0f}cps"
            obs = np.array([r["_m"][f"coef_{tag}"] for r in raw.values()])
            k = min(r["_m"][f"null_coef_{tag}"].shape[0] for r in raw.values())
            null = np.stack([r["_m"][f"null_coef_{tag}"][:k] for r in raw.values()])
            pop[fr] = ga.agnostic_test(obs, null)
        for fr, ph in zip(FREQS, pa.holm_bonferroni([pop[f]["p_value"] for f in FREQS])):
            pop[fr]["p_holm"] = float(ph)
            pop[fr]["significant"] = bool(ph < ALPHA)
            tag = f"{fr:.0f}cps"
            theta2 = np.array([(r[f"_theta_{tag}"] ** 2).sum() for r in raw.values()])
            null2 = np.array([(r[f"_null_theta_{tag}"] ** 2).sum(1).mean() for r in raw.values()])
            pop[fr]["rms_theta_pct"] = round(100 * float(np.sqrt(theta2.mean())), 3)
            pop[fr]["rms_theta_debiased_pct"] = round(
                100 * float(np.sqrt(max((theta2 - null2).mean(), 0.0))), 3)
    return pd.DataFrame(blocks), pd.DataFrame(parts), pop, raw


def _confirmatory_population() -> dict:
    """Holm p and statistic of the confirmatory accuracy test, for comparison."""
    f = C.DERIV_DIR / "population" / "population_agnostic.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text(encoding="utf-8")).get(getattr(C, "PRIMARY_MEASURE", "accuracy"), {})
    return {k: {"p_holm": v.get("p_holm"), "statistic": v.get("statistic")}
            for k, v in d.get("by_freq", {}).items()}


# ===========================================================================
# Figure
# ===========================================================================
def _figure(blocks: pd.DataFrame, parts: pd.DataFrame, pop: dict, stem: Path) -> None:
    """Figure: information per trial, information per participant, the evidence of
    both tests, and the threshold modulation per participant."""
    fig, axes = plt.subplots(2, 2, figsize=(S.COL2_IN, 6.2))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.94, wspace=0.34, hspace=0.48)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    subs = list(parts["participant"])
    xs = np.arange(len(subs))
    short = [s.replace("sub-", "") for s in subs]

    # a: information per trial over the psychometric function, and where the blocks sat
    pc = np.linspace(0.505, 0.985, 400)
    eta = efficiency(np.clip(pc, 0.5 + 1e-6, 1 - LAPSE - 1e-6))
    eta[pc >= 1 - LAPSE] = 0.0
    ax_a.plot(pc, eta, color=S.INK, lw=1.6, zorder=2, label="model")
    for lo, hi, edge in ((0.5, P_MIN, P_MIN), (P_MAX, 1.0, P_MAX)):
        ax_a.plot([lo, hi], [efficiency(edge)] * 2, color=S.INK, lw=1.0, ls=":", zorder=2)
    ax_a.plot([], [], color=S.INK, lw=1.0, ls=":", label="clipped weight")
    S.reference_line(ax_a, x=DESIGN_PC)
    S.reference_line(ax_a, y=1.0, ls=":")
    best = pc[np.argmax(eta)]
    ax_a.annotate(f"peak {100 * best:.0f} %", (best, eta.max()), xytext=(0, 5),
                  textcoords="offset points", ha="center", fontsize=8, color=S.GREY)
    ok = blocks["thr_accuracy"].notna()
    acc = blocks.loc[ok, "thr_accuracy"].to_numpy(float)
    ax_a.plot(acc, efficiency(np.clip(acc, P_MIN, P_MAX)), "o", ms=4.2, mfc=S.COND["walking"],
              mec="white", mew=0.6, zorder=3, label="walking blocks")
    ax_a.set_xlim(0.5, 1.0)
    ax_a.set_ylim(0, 1.5)
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f} %"))
    ax_a.set_xlabel("Accuracy")
    ax_a.set_ylabel("Information per trial\n(relative to 75 %)")
    ax_a.legend(loc="upper left", fontsize=8, handlelength=1.6)
    S.panel(ax_a, "a", "Information per trial")

    # b: information retained per participant
    ret = parts["information_retained"].to_numpy(float)
    ax_b.bar(xs, ret, color=S.COND["walking"], width=0.65, zorder=2)
    S.reference_line(ax_b, y=1.0, ls=":")
    for x, r, n in zip(xs, ret, parts["n_trials"]):
        ax_b.text(x, r + 0.02, f"{n}", ha="center", va="bottom", fontsize=8, color=S.GREY)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(short)
    ax_b.set_xlabel("Participant (walking trials above the bars)")
    ax_b.set_ylabel("Information per trial\n(mean, relative to 75 %)")
    ax_b.set_ylim(0, max(1.4, float(np.nanmax(ret)) + 0.15))
    ax_b.grid(axis="x", visible=False)
    S.panel(ax_b, "b", "Information per participant")

    # c: per-participant evidence, accuracy test vs information-weighted test
    hi = 1.0
    for fr in FREQS:
        tag = f"{fr:.0f}cps"
        x = parts[f"term_accuracy_{tag}"].to_numpy(float)
        y = parts[f"term_weighted_{tag}"].to_numpy(float)
        hi = max(hi, np.nanmax(x), np.nanmax(y))
        ax_c.plot(x, y, "o", ms=4.5, color=S.FREQGRP[tag], mec="white", mew=0.6,
                  label=f"{fr:.0f} cps", zorder=3)
    lim = (0, hi * 1.08)
    ax_c.plot(lim, lim, color=S.GREY, lw=0.9, zorder=1, label="equal evidence")
    ax_c.set_xlim(*lim)
    ax_c.set_ylim(*lim)
    ax_c.set_xlabel(r"Accuracy test  $|z|^2/s^2$")
    ax_c.set_ylabel(r"Information-weighted  $|z|^2/s^2$")
    ax_c.legend(loc="upper left", fontsize=8)
    S.panel(ax_c, "c", "Evidence, both tests")

    # d: threshold modulation per participant, against its own permutation 95th percentile
    for j, fr in enumerate(FREQS):
        tag = f"{fr:.0f}cps"
        off = (j - 0.5) * 0.3
        th = parts[f"theta_{tag}_pct"].to_numpy(float)
        nl = parts[f"theta_{tag}_null95_pct"].to_numpy(float)
        ax_d.plot(xs + off, nl, "_", ms=9, mew=1.4, color=S.MIST, zorder=2)
        sig = parts[f"p_{tag}_holm"].to_numpy(float) < ALPHA
        col = S.FREQGRP[tag]
        ax_d.plot(xs[~sig] + off, th[~sig], "o", ms=4.5, mfc="white", mec=col, mew=1.1,
                  zorder=3, label=(f"{fr:.0f} cps" + (f", population Holm p = {pop[fr]['p_holm']:.2f}"
                                                       if fr in pop else "")))
        ax_d.plot(xs[sig] + off, th[sig], "o", ms=4.5, mfc=col, mec=col, mew=1.1, zorder=3)
    ax_d.plot([], [], "o", ms=4.5, mfc=S.INK, mec=S.INK, label="filled: Holm p < 0.05")
    ax_d.plot([], [], "_", ms=9, mew=1.4, color=S.MIST, label="null 95th pct")
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(short)
    ax_d.set_xlabel("Participant")
    ax_d.set_ylabel("Threshold modulation (%)")
    top = np.nanmax(parts[[f"theta_{f:.0f}cps_null95_pct" for f in FREQS]
                          + [f"theta_{f:.0f}cps_pct" for f in FREQS]].to_numpy(float))
    ax_d.set_ylim(0, 1.9 * top)
    ax_d.grid(axis="x", visible=False)
    ax_d.legend(loc="upper left", fontsize=8, ncol=1, handletextpad=0.3, labelspacing=0.25)
    S.panel(ax_d, "d", "Threshold modulation")
    S.save_pub(fig, stem)


# ===========================================================================
# Main
# ===========================================================================
OUTPUTS = ("threshold_phase_blocks.tsv", "threshold_phase_participants.tsv",
           "threshold_phase.json", "threshold_phase.png", "threshold_phase.pdf")


def main() -> dict:
    """Information per walking block and the information-weighted stride-phase test."""
    blocks, parts, pop, _ = analyse()
    bids_io.gate_report("threshold_phase", condition="walking",
                        analysed_subjects=sorted(parts["participant"]) if not parts.empty else [])
    if parts.empty:
        bids_io.clear_stale("threshold_phase", *OUTPUTS)
        print("No walking participant with phased trials. Run the gait and phase stages first.")
        return {}
    out = bids_io.deriv_dir("threshold_phase")
    blocks.to_csv(out / OUTPUTS[0], sep="\t", index=False, na_rep="n/a")
    parts.to_csv(out / OUTPUTS[1], sep="\t", index=False, na_rep="n/a")
    conf = _confirmatory_population()
    summary = {
        "Description": "Stride-phase modulation of the threshold: every walking trial "
                       "weighted by the information it carries about a threshold shift, "
                       "tested with the confirmatory machinery (threshold_phase.py). "
                       "SECONDARY; the confirmatory outcome is accuracy.",
        "Model": {"beta": BETA, "lapse": LAPSE, "guess": GUESS, "design_accuracy": DESIGN_PC,
                  "weight_accuracy_range": [P_MIN, P_MAX], "seed": SEED},
        "n_participants": int(len(parts)),
        "n_trials": int(parts["n_trials"].sum()),
        "effective_trials": round(float(parts["effective_trials"].sum()), 1),
        "information_retained": round(float(parts["effective_trials"].sum()
                                            / parts["n_trials"].sum()), 4),
        "population": {f"{f:.0f}cps": {k: v for k, v in r.items() if not k.startswith("_")}
                       for f, r in pop.items()},
        "confirmatory_accuracy_test": conf,
    }
    (out / OUTPUTS[2]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _figure(blocks, parts, pop, bids_io.fig_dir("threshold_phase") / "threshold_phase")

    print("=== Threshold-scale stride-phase test (information-weighted, SECONDARY) ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(parts[["participant", "n_trials", "effective_trials", "information_retained",
                     "theta_2cps_pct", "p_2cps_holm", "theta_4cps_pct", "p_4cps_holm",
                     "term_accuracy_2cps", "term_weighted_2cps",
                     "term_accuracy_4cps", "term_weighted_4cps"]].to_string(index=False))
    print(f"  information retained: {100 * summary['information_retained']:.1f} % "
          f"({summary['effective_trials']:.0f} effective of {summary['n_trials']} trials)")
    for f, r in pop.items():
        tag = f"{f:.0f}cps"
        c = conf.get(tag, {})
        print(f"  {tag}: S={r['statistic']:.2f} (null mean {r['null_mean']:.2f}), Holm p="
              f"{r['p_holm']:.4f}, RMS threshold modulation {r['rms_theta_pct']:.2f} % "
              f"(debiased {r['rms_theta_debiased_pct']:.2f} %) | accuracy test Holm p="
              f"{c.get('p_holm', float('nan')):.4f}")
    print(f"Outputs -> {out}")
    return summary


if __name__ == "__main__":
    main()
