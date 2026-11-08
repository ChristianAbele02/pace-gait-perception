"""Exploratory: are button presses timed by the stride?

During walking, motor output is entrained to the gait: in Davidson et al. (2024) presses
to visual targets clustered at a particular point of the step. This module asks the
motor question directly: do the presses themselves cluster at a
stride phase, beyond what the timing of the targets and the spread of reaction times
produce?

A press histogram on its own cannot answer that. Press time is onset time plus reaction
time, so the press phases are the onset phases convolved with the RT distribution, and
any unevenness in when targets fell shows up in them. The null keeps both ingredients and
breaks only their pairing: within each block the reaction times are shuffled across the
answered trials and added to the fixed onsets, and the press phases are recomputed with
the block's own gait (``phase_analysis._phase_of``). Under H0, reaction time does not
depend on where in the stride the target fell, and presses are then placed exactly as the
shuffles place them.

Statistic: the Fourier coefficient of the press phases at k cycles per stride,
``z_k = mean exp(i 2 pi k phase / 100)``, centred on the mean of its null so that the
clustering onsets and RTs produce by themselves is removed. ``2 |z_k|`` is the depth of
the press-density modulation as a fraction of its mean. Per participant: the 2 and 4 cps
tests, Holm over the pair, and a separate 1 cps test. Population: the phase-agnostic
``S`` of ``population.agnostic_test`` and the common-phase Hotelling T2 of
``population.hotelling_t2``, each Holm over 2 and 4 cps. At 2 and 4 cps the stride
origin does not depend on the foot labels, so a common phase is meaningful across
participants.

Only answered trials with a reaction time above ``phase_analysis.RT_MIN_S`` enter; an
anticipation is not a response to the target.

Usage::

    python -m Analysis.press_phase
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
    from . import population
    from . import config as C
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import population
    import config as C
    import phase_analysis as pa
    import plotstyle as S

S.apply()

N_PERM = 1000
FREQS = (2.0, 4.0)
STRIDE_FREQ = 1.0
ALL_FREQS = (STRIDE_FREQ,) + FREQS
MIN_PRESSES = 200
N_BINS = 40
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Per block
# ---------------------------------------------------------------------------
def _block(rec) -> dict | None:
    """Onsets, reaction times and the gait arrays needed to place presses, for one block."""
    if rec.condition != "walking":
        return None
    npz = bids_io.deriv_dir("gait", rec.sub, rec.ses) / f"{rec.label()}_gait.npz"
    if not npz.exists():
        return None
    g = np.load(npz)
    step_foot = g["step_foot"] if np.size(g["step_foot"]) else None
    tg = g["t"] if np.size(g["t"]) else None
    cont = pa.continuous_phase(g)
    tp = pa._trial_phases(rec, g["troughs"], step_foot, tg, cont)
    if tp is None:
        return None
    ev, _ = tp
    rt = ev["rt"].to_numpy(float)
    keep = ev["response"].notna().to_numpy() & np.isfinite(rt) & (rt > pa.RT_MIN_S)
    if keep.sum() < 10:
        return None
    return {"onset": ev["onset"].to_numpy(float)[keep], "rt": rt[keep],
            "gait": (np.asarray(g["troughs"]), step_foot, tg, cont)}


def press_phases(onset: np.ndarray, rt: np.ndarray, gait) -> np.ndarray:
    """Stride phase (%) of ``onset + rt``; any shape, NaN where the press is unphased."""
    t = (np.asarray(onset, float) + np.asarray(rt, float))
    troughs, step_foot, tg, cont = gait
    return pa._phase_of(t.ravel(), troughs, step_foot, tg, cont).reshape(t.shape)


def coefficients(phase: np.ndarray) -> np.ndarray:
    """``(..., n_freq, 2)`` (cos, sin) means of the press phases at ``ALL_FREQS``."""
    ang = 2 * np.pi * np.asarray(phase, float) / 100.0
    ok = np.isfinite(ang)
    n = np.maximum(ok.sum(-1), 1)[..., None]
    out = []
    for k in ALL_FREQS:
        c = np.where(ok, np.cos(k * ang), 0.0).sum(-1)[..., None] / n
        s = np.where(ok, np.sin(k * ang), 0.0).sum(-1)[..., None] / n
        out.append(np.concatenate([c, s], -1))
    return np.stack(out, -2)


# ---------------------------------------------------------------------------
# Per participant
# ---------------------------------------------------------------------------
def analyse_blocks(blocks: list[dict], *, n_perm: int = N_PERM,
                   rng: np.random.Generator | None = None) -> dict | None:
    """Observed and RT-shuffled press-phase coefficients pooled over a participant's blocks.

    Pooling weights each press equally: the per-block sums of (cos, sin) and the per-block
    counts are added before dividing, for the observed presses and for every shuffle.
    """
    rng = rng or np.random.default_rng(0)
    nf = len(ALL_FREQS)
    obs_sum, obs_n = np.zeros((nf, 2)), 0
    null_sum, null_n = np.zeros((n_perm, nf, 2)), np.zeros(n_perm)
    phases, null_hist = [], np.zeros(N_BINS)
    for b in blocks:
        ph = press_phases(b["onset"], b["rt"], b["gait"])
        ok = np.isfinite(ph)
        obs_sum += coefficients(ph[ok]) * ok.sum()
        obs_n += int(ok.sum())
        phases.append(ph[ok])
        perm = np.argsort(rng.random((n_perm, b["rt"].size)), axis=1)
        nph = press_phases(np.broadcast_to(b["onset"], perm.shape), b["rt"][perm], b["gait"])
        nok = np.isfinite(nph).sum(1)
        null_sum += coefficients(nph) * nok[:, None, None]
        null_n += nok
        null_hist += np.histogram(nph[np.isfinite(nph)], bins=N_BINS, range=(0, 100))[0] / n_perm
    if obs_n < MIN_PRESSES:
        return None
    obs = obs_sum / obs_n
    null = null_sum / np.maximum(null_n, 1)[:, None, None]
    centre = null.mean(0)
    obs_c, null_c = obs - centre, null - centre
    res = {"n_presses": obs_n, "n_blocks": len(blocks)}
    p_raw = {}
    for i, k in enumerate(ALL_FREQS):
        stat = float((obs_c[i] ** 2).sum())
        nstat = (null_c[:, i] ** 2).sum(1)
        p = float((1 + np.sum(nstat >= stat)) / (1 + n_perm))
        z = complex(*obs_c[i])
        p_raw[k] = p
        res[f"{k:.0f}cps"] = {
            "p": p, "modulation_depth": float(2 * abs(z)),
            "null_depth_p95": float(2 * np.sqrt(np.percentile(nstat, 95))),
            "peak_pct_stride": float((np.angle(z) / (2 * np.pi * k) * 100.0) % (100.0 / k)),
            "coef": obs_c[i], "null_coef": null_c[:, i]}
    holm = pa.holm_bonferroni([p_raw[k] for k in FREQS])
    for k, ph in zip(FREQS, holm):
        res[f"{k:.0f}cps"]["p_holm"] = float(ph)
    phases = np.concatenate(phases)
    hist = np.histogram(phases, bins=N_BINS, range=(0, 100))[0]
    res["relative_density"] = hist / np.maximum(null_hist, 1e-9)
    return res


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
def population_test(rows: list[dict]) -> dict:
    """Phase-agnostic S and common-phase Hotelling T2 per frequency (Holm over 2 and 4)."""
    out = {}
    for k in ALL_FREQS:
        key = f"{k:.0f}cps"
        obs = np.array([r[key]["coef"] for r in rows], float)
        nper = min(r[key]["null_coef"].shape[0] for r in rows)
        null = np.stack([r[key]["null_coef"][:nper] for r in rows])
        out[key] = {"agnostic": population.agnostic_test(obs, null), "hotelling": population.hotelling_t2(obs),
                    "n_significant_participants": int(sum(
                        r[key].get("p_holm", r[key]["p"]) < ALPHA for r in rows))}
    for test in ("agnostic", "hotelling"):
        holm = pa.holm_bonferroni([out[f"{k:.0f}cps"][test]["p_value"] for k in FREQS])
        for k, ph in zip(FREQS, holm):
            out[f"{k:.0f}cps"][test]["p_holm"] = float(ph)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _figure(rows: list[dict], pop: dict, out: Path) -> None:
    """Figure: relative press density over the stride, the 2-cps coefficients per
    participant, and the population statistic against its null."""
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6),
                           gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    x = (np.arange(N_BINS) + 0.5) * 100.0 / N_BINS
    dens = np.vstack([r["relative_density"] for r in rows])
    m = dens.mean(0)
    se = dens.std(0, ddof=1) / np.sqrt(dens.shape[0])
    ax[0].fill_between(x, m - se, m + se, color=S.TEAL, alpha=0.25, lw=0)
    ax[0].plot(x, m, color=S.TEAL, lw=2.0, label="observed / RT-shuffled")
    ax[0].axhline(1.0, color=S.INK, lw=0.8)
    for xs in (0, 50, 100):
        ax[0].axvline(xs, color=S.GREY, ls=":", lw=0.8)
    ax[0].set_xlabel("% of stride at the press (0, 50 = double support)")
    ax[0].set_ylabel("relative press density")
    ax[0].legend(fontsize=8, loc="upper right")
    S.panel(ax[0], "a", "When presses happen in the stride", fontsize=10)

    k = "2cps"
    Z = np.array([r[k]["coef"] for r in rows]) * 2
    cols = [S.RED if r[k]["p_holm"] < ALPHA else S.GREY for r in rows]
    ax[1].scatter(Z[:, 0], Z[:, 1], c=cols, s=28, zorder=3)
    mu = Z.mean(0)
    ax[1].plot(*mu, "x", color=S.INK, ms=9, mew=2)
    ell = population._ellipse(np.cov(Z, rowvar=False), mu, Z.shape[0])
    if ell is not None:
        ax[1].plot(ell[:, 0], ell[:, 1], color=S.INK, ls="--", lw=1.0)
    ax[1].axhline(0, color=S.GREY, lw=0.7); ax[1].axvline(0, color=S.GREY, lw=0.7)
    lim = 1.15 * np.nanmax(np.abs(np.r_[Z.ravel(), ell.ravel() if ell is not None else []]))
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim); ax[1].set_aspect("equal")
    ax[1].set_xlabel("cos coefficient (x 2)"); ax[1].set_ylabel("sin coefficient (x 2)")
    h = pop[k]["hotelling"]
    S.panel(ax[1], "b", f"2 cps per participant\ncommon phase, Holm p={h['p_holm']:.3f}",
            fontsize=10)

    a = pop[k]["agnostic"]
    ax[2].hist(a["_null_stat"], bins=40, color=S.MIST, edgecolor="white")
    ax[2].axvline(a["statistic"], color=S.TEAL, lw=2.2,
                  label=f"observed (Holm p={a['p_holm']:.3f})")
    S.reference_line(ax[2], x=a["null_p95"], label="null 95th pct")
    ax[2].set_xlabel(r"$S=\sum_i |z_i|^2/s_i^2$  @ 2 cps"); ax[2].set_ylabel("permutations")
    ax[2].legend(fontsize=8, loc="upper right" if a["statistic"] < a["null_p95"] else "upper left")
    S.panel(ax[2], "c", "Population, phase-agnostic", fontsize=10)

    p4, p1 = pop["4cps"], pop["1cps"]
    fig.suptitle(f"Button presses over the stride, n = {len(rows)} walking participants",
                 fontsize=12.5, fontweight="semibold", x=0.012, ha="left")
    S.caption(fig, "EXPLORATORY. Null: reaction times shuffled within blocks and added to the "
                   "fixed onsets, so presses keep the onset timing and the RT distribution but "
                   "lose their pairing. a: press histogram divided by the mean RT-shuffled "
                   "histogram, mean ± SEM over participants. b: centred coefficients, red = "
                   "Holm p < .05; dashed, 95 % region of the mean. "
                   f"4 cps: S p_Holm = {p4['agnostic']['p_holm']:.3f}, common phase p_Holm = "
                   f"{p4['hotelling']['p_holm']:.3f}; 1 cps: S p = {p1['agnostic']['p_value']:.3f}.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    S.save(fig, out)


def _json_safe(o):
    """JSON-serialisable copy of a result, without private keys and null draws."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()
                if not str(k).startswith("_") and k != "null_coef"}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main() -> dict | None:
    """Press-phase clustering per participant and in the population."""
    recs = [r for r in bids_io.find_recordings() if r.condition == "walking"]
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[(r.sub, r.ses)].append(r)
    out_dir = bids_io.deriv_dir("press_phase")
    rows = []
    print("=== Button presses over the stride (EXPLORATORY) ===")
    for (sub, ses), subrecs in sorted(by_sub.items()):
        blocks = [b for b in (_block(r) for r in subrecs) if b]
        if not blocks:
            continue
        res = analyse_blocks(blocks, rng=np.random.default_rng(zlib.crc32(f"{sub}_press".encode())))
        if res is None:
            print(f"  {sub}: fewer than {MIN_PRESSES} phased presses - skipped")
            continue
        res.update({"sub": sub, "ses": ses})
        rows.append(res)
        (out_dir / f"{sub}_{ses}_task-{C.TASK_LABEL}_acq-walking_press_phase.json").write_text(
            json.dumps(_json_safe(res), indent=2), encoding="utf-8")
        print(f"  {sub}: {res['n_presses']} presses | 2 cps depth {res['2cps']['modulation_depth']:.3f} "
              f"(Holm p={res['2cps']['p_holm']:.3f}, peak {res['2cps']['peak_pct_stride']:.0f} %) | "
              f"4 cps depth {res['4cps']['modulation_depth']:.3f} (Holm p={res['4cps']['p_holm']:.3f}) | "
              f"1 cps p={res['1cps']['p']:.3f}")
    bids_io.gate_report("press_phase", condition="walking",
                        analysed_subjects={r["sub"] for r in rows}, prune=("press_phase",))
    if len(rows) < 2:
        bids_io.clear_stale("press_phase", "press_phase_population.json", "press_phase_population.png")
        return None
    pop = population_test(rows)
    payload = {"note": "EXPLORATORY. Press-phase clustering against reaction times shuffled "
                       "within blocks; coefficients centred on the null mean.",
               "n_participants": len(rows), "population": _json_safe(pop)}
    (out_dir / "press_phase_population.json").write_text(json.dumps(payload, indent=2),
                                                        encoding="utf-8")
    _figure(rows, pop, bids_io.fig_dir("press_phase") / "press_phase_population.png")
    print(f"\n  POPULATION (n={len(rows)}):")
    for k in ALL_FREQS:
        d = pop[f"{k:.0f}cps"]
        a, h = d["agnostic"], d["hotelling"]
        print(f"    {k:.0f} cps: S={a['statistic']:.2f} vs null mean {a['null_mean']:.2f}, "
              f"p={a['p_value']:.4f}" + (f" (Holm {a['p_holm']:.4f})" if "p_holm" in a else "")
              + f" | common phase F={h['f']:.2f}, p={h['p_value']:.4f}"
              + (f" (Holm {h['p_holm']:.4f})" if "p_holm" in h else ""))
    print(f"\nOutputs -> {out_dir}")
    return payload


if __name__ == "__main__":
    main()
