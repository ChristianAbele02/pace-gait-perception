"""Power of this pipeline's own tests at the effect size measured in Davidson et al. (2024).

One simulator for all analyses in this folder. It draws each participant's walking trials
and applies the pipeline's estimators:

1. difficulty residualisation ............................ ``phase_analysis.difficulty_adjust``
2. 40 stride-phase bins, Fourier fit weighted by occupancy  ``phase_analysis._spectrum_with_coef``
3. per participant: weighted R2 at 2 and 4 cps against 1000 trial-level permutations,
   Holm over the pair .................................... ``phase_analysis.analyse_measure``
4. population, confirmatory: ``S = sum_i |z_i|^2 / s_i^2`` per frequency, Holm
   ....................................................... ``group_agnostic.agnostic_test``
5. population, secondary: Hotelling T2 on the (cos, sin) pairs, Holm
   ....................................................... ``common_phase.hotelling_t2``

``validate`` checks each step against those functions. Three exact shortcuts keep it fast:

* The estimators use only the (stride bin x difficulty level) table, and trials in a cell
  are independent Bernoulli draws. So the 40 x 3 table is drawn directly (multinomial
  counts, binomial hits).
* ``s_i^2``, estimated in the pipeline as mean ``|z|^2`` over 1000 shuffles, has the closed
  form ``n/(n-1) Var(v) [(G^-1)_11 + (G^-1)_22]`` with ``G`` the weighted design Gram matrix.
* The population null is the statistic's distribution over studies simulated without an
  effect; with exchangeable trials under H0 this is what the permutation null estimates
  (``validate`` compares both).

Effect model: participant ``i``'s coefficient at frequency ``f`` is ``mu_f + u_if`` with
``u_if ~ CN(0, RMS_f^2 - mu_f^2)``. RMS and coherent ``mu`` come from
``davidson_effect.json``; both frequencies are present in the same data. ``robustness``
replaces the Gaussian scatter with a spike-and-slab population.

Usage::

    python -m Analysis.davidson_power.power {validate,checks,sweep,population,robustness,mda,tradeoff,all}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # pipeline modules, for validate / tradeoff

NBINS = 40
EDGES = np.linspace(0, 100, NBINS + 1)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
FREQS = np.array([2.0, 4.0])
OMEGA = 2 * np.pi * FREQS / 100
DESIGN = np.stack([np.column_stack([np.ones(NBINS), np.cos(w * CENTRES), np.sin(w * CENTRES)])
                   for w in OMEGA])           # (freq, bin, [1, cos, sin])
BIN_GAIN = np.sinc(FREQS * (EDGES[1] - EDGES[0]) / 100)   # cos averaged over a bin
ALPHA = 0.05
ALPHA_ANY = ALPHA / 2                          # Holm over two tests rejects iff min p < alpha/2
N_PERM = 1000                                  # phase_analysis.N_PERM
TRIALS_PER_BLOCK = 180                         # at ISI 4 s
DESIGNS = {"former": {"walking_blocks": 6, "standing_blocks": 4},
           "adopted": {"walking_blocks": 8, "standing_blocks": 2}}     # over both visits
ADOPTED_TRIALS = DESIGNS["adopted"]["walking_blocks"] * TRIALS_PER_BLOCK

_EFFECT = json.loads((HERE / "davidson_effect.json").read_text(encoding="utf-8"))["summary"]
RMS = np.array([_EFFECT["rms_amp_2cps"], _EFFECT["rms_amp_4cps"]])
COHERENT = np.array([_EFFECT["group_level"][f]["coherent_amp"] for f in ("2.01", "4.01")])


# --- difficulty mix -------------------------------------------------------------------
# QUEST+ places the levels at the 75 / 85 / 65 % points of the rig's Weibull
# P = g + (1 - g - l)(1 - exp(-(x/alpha)^beta)) (StaircaseManager.targetThresholdPC ...),
# mixed 0.85 / 0.10 / 0.05. An oscillation of alpha changes each level's accuracy by
# dP/dln(alpha) = -(1 - g - l) beta u e^-u, u = (x/alpha)^beta, which with
# q = (P - g)/(1 - g - l) = 1 - e^-u is proportional to (1 - q)(-ln(1 - q)).
GUESS, LAPSE, BETA = 0.5, 0.05, 2.5
LEVELS = {"threshold": (0.85, 0.75), "easy": (0.10, 0.85), "hard": (0.05, 0.65)}   # weight, P(correct)


def accuracy_gain(pc):
    q = (np.asarray(pc, float) - GUESS) / (1 - GUESS - LAPSE)
    return -(1 - q) * np.log1p(-q)


@dataclass(frozen=True)
class Mix:
    weight: np.ndarray
    base: np.ndarray           # accuracy at each level
    gain: np.ndarray           # oscillation per level relative to a threshold trial

    @classmethod
    def rig(cls, all_levels_carry=True) -> Mix:
        weight, pc = map(np.array, zip(*LEVELS.values()))
        gain = accuracy_gain(pc) / accuracy_gain(0.75) if all_levels_carry else (pc == 0.75) * 1.0
        return cls(weight, pc, gain)


MIX = Mix.rig()                                # threshold oscillation, carried by all levels
THRESHOLD_ONLY = Mix.rig(False)                # pessimistic case: only threshold trials carry it
ALL_AT_THRESHOLD = Mix(np.ones(1), np.array([0.75]), np.ones(1))    # Davidson's design, all trials at threshold


# --- simulation -------------------------------------------------------------------------
@dataclass
class Tables:
    """Per participant (row): trials and hits in every (stride bin, difficulty level) cell."""
    n: np.ndarray              # (M, NBINS, K)
    hits: np.ndarray


def simulate_tables(rng, coef, ntrials, mix=MIX) -> Tables:
    """``coef`` (M, 2): complex ``a + ib`` of the true ``a cos + b sin`` at 2 and 4 cps.

    Phases are uniform over the stride (verified in ``checks``), so cell probabilities use
    the bin-averaged oscillation (``cos`` at the bin centre times ``BIN_GAIN``). The
    0.01-0.99 clip is not reached at these amplitudes.
    """
    m, k = coef.shape[0], mix.weight.size
    cells = np.outer(np.full(NBINS, 1 / NBINS), mix.weight).ravel()
    n = rng.multinomial(ntrials, cells, size=m).reshape(m, NBINS, k)
    osc = (np.conj(coef)[:, :, None] * np.exp(1j * OMEGA[:, None] * CENTRES)).real
    osc = np.einsum("mfb,f->mb", osc, BIN_GAIN)
    p = np.clip(mix.base + mix.gain * osc[..., None], 0.01, 0.99)
    return Tables(n, rng.binomial(n, p))


def tables_from_trials(phase, level, outcome, nlevels) -> Tables:
    """Same tables from trial-level data, binned as in ``phase_analysis._bin``."""
    b = np.clip(np.digitize(phase, EDGES) - 1, 0, NBINS - 1)
    cell = b * nlevels + level
    count = lambda w=None: np.bincount(cell, w, NBINS * nlevels).reshape(1, NBINS, nlevels).astype(int)
    return Tables(count(), count(outcome))


@dataclass
class Estimate:
    z: np.ndarray              # (M, 2) fitted complex coefficient a + ib at 2 and 4 cps
    s2: np.ndarray             # (M, 2) permutation-null E|z|^2
    shift: np.ndarray          # (M, K) residualisation: v = outcome + shift[level]
    gram_inv: np.ndarray       # (M, 2, 3, 3)


def estimate(t: Tables) -> Estimate:
    """Residualise by level, bin, and fit ``c + a cos + b sin`` weighted by bin counts.

    The weighted fit equals trial-level OLS on the bin design rows, so per-bin counts and
    sums suffice: ``beta = G^-1 X' sum_b``, ``G = X' diag(n_b) X``.
    """
    n_level, h_level = t.n.sum(1), t.hits.sum(1)                          # (M, K)
    total = n_level.sum(1)
    grand = h_level.sum(1) / total
    shift = grand[:, None] - h_level / np.maximum(n_level, 1)            # shift level mean to grand mean
    count = t.n.sum(2)
    sums = t.hits.sum(2) + (t.n * shift[:, None]).sum(2)
    sq = (t.hits * (1 + shift[:, None]) ** 2 + (t.n - t.hits) * shift[:, None] ** 2).sum((1, 2))
    var = sq / total - grand ** 2                                         # residualised mean equals grand mean
    gram_inv = np.linalg.inv(np.einsum("mb,fbi,fbj->mfij", count, DESIGN, DESIGN))
    beta = np.einsum("mfij,fbj,mb->mfi", gram_inv, DESIGN, sums)
    s2 = (total / (total - 1) * var)[:, None] * (gram_inv[..., 1, 1] + gram_inv[..., 2, 2])
    return Estimate(beta[..., 1] + 1j * beta[..., 2], s2, shift, gram_inv)


def trials(t: Tables, est: Estimate, row: int):
    """One participant's trials as ``(bin, residualised value)``, grouped by bin."""
    n, h = t.n[row], t.hits[row]
    value = np.stack([1 + est.shift[row], est.shift[row]])               # hit, miss
    reps = np.stack([h, n - h], -1)                                       # (bin, level, outcome)
    bins = np.repeat(np.arange(NBINS), n.sum(1))
    return bins, np.repeat(np.broadcast_to(value.T, reps.shape).ravel(), reps.ravel())


def bin_sums(bins, values) -> np.ndarray:
    """Per-bin sums ``(..., NBINS)`` of every row of ``values`` ``(..., n)``; empty bins are 0."""
    count = np.bincount(bins, minlength=NBINS)
    ok = count > 0
    out = np.zeros(values.shape[:-1] + (NBINS,))
    out[..., ok] = np.add.reduceat(values[..., np.argsort(bins, kind="stable")],
                                   np.r_[0, np.cumsum(count[ok])[:-1]], axis=-1)
    return out


def participant_test(bins, values, shuffles) -> np.ndarray:
    """``(p_2cps, p_4cps)`` of the pipeline's per-participant test, before Holm.

    Weighted R2 = ``|Q' T|^2 / (sum T^2 - (sum S)^2 / N)``, with ``T = S / sqrt(n)`` the
    scaled bin sums and ``Q`` an orthonormal basis of the count-centred cos/sin columns.
    Bin counts do not change under permutation, so ``Q`` is computed once.
    """
    count = np.bincount(bins, minlength=NBINS)
    ok = count > 0
    sums = bin_sums(bins, np.vstack([values, shuffles]))[:, ok]
    root = np.sqrt(count[ok])
    scaled = sums / root
    den = (scaled ** 2).sum(1) - sums.sum(1) ** 2 / count.sum()
    p = np.empty(2)
    for f in range(2):
        cols = DESIGN[f][ok, 1:] * root[:, None]
        cols -= np.outer(root, root @ cols) / count.sum()
        q = np.linalg.qr(cols)[0]
        r2 = np.divide(((scaled @ q) ** 2).sum(1), den, out=np.zeros_like(den), where=den > 0)
        p[f] = (1 + np.sum(r2[1:] >= r2[0])) / (1 + shuffles.shape[0])
    return p


def participant_power(rng, ntrials, amp, nsim, freq_index, mix=MIX, nperm=N_PERM) -> float:
    """Fraction of ``nsim`` participants oscillating at ``amp`` (random phase, one
    frequency) flagged by the per-participant test."""
    coef = np.zeros((nsim, 2), complex)
    coef[:, freq_index] = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, nsim))
    t = simulate_tables(rng, coef, ntrials, mix)
    est = estimate(t)
    hits = 0
    for row in range(nsim):
        bins, values = trials(t, est, row)
        shuffles = rng.permuted(np.broadcast_to(values, (nperm, values.size)), axis=1)
        hits += participant_test(bins, values, shuffles).min() < ALPHA_ANY
    return hits / nsim


def draw_coefficients(rng, shape, scale=1.0, prevalence=None) -> np.ndarray:
    """True per-participant coefficients, complex ``shape + (2,)``: Gaussian scatter around
    the coherent mean, or spike-and-slab (a fraction ``prevalence`` oscillate at their own
    phase, the rest are flat), with ``E|c|^2 = RMS^2`` in both."""
    if prevalence is None:
        sd = np.sqrt(np.maximum(RMS ** 2 - COHERENT ** 2, 0) / 2)
        c = COHERENT + sd * (rng.standard_normal(shape + (2,)) + 1j * rng.standard_normal(shape + (2,)))
    else:
        on = rng.random(shape + (1,)) < prevalence
        c = on * RMS / np.sqrt(prevalence) * np.exp(1j * rng.uniform(0, 2 * np.pi, shape + (2,)))
    return scale * c


def hotelling_p(z) -> np.ndarray:
    """Batched one-sample Hotelling T2 p-value of rows ``(..., N, 2)`` against the origin,
    real or complex (``a + ib`` as ``(a, b)``); as ``common_phase.hotelling_t2``."""
    z = np.asarray(z)
    if np.iscomplexobj(z):
        z = np.stack([z.real, z.imag], -1)
    n = z.shape[-2]
    if n < 4:                                   # common_phase.MIN_N
        return np.ones(z.shape[:-2])
    mu = z.mean(-2)
    dev = z - mu[..., None, :]
    cov = np.einsum("...ni,...nj->...ij", dev, dev) / (n - 1)
    t2 = n * np.einsum("...i,...ij,...j->...", mu, np.linalg.inv(cov), mu)
    return stats.f.sf((n - 2) / (2 * (n - 1)) * t2, 2, n - 2)


def population_stats(rng, npp, ntrials, nsim, *, scale=1.0, prevalence=None, mix=MIX, rows=40_000):
    """``(S, p_hotelling)``, each ``(nsim, 2)``, over ``nsim`` simulated studies."""
    chunk = max(1, rows // npp)
    out_s, out_p = [], []
    for start in range(0, nsim, chunk):
        k = min(chunk, nsim - start)
        coef = draw_coefficients(rng, (k, npp), scale, prevalence).reshape(k * npp, 2)
        est = estimate(simulate_tables(rng, coef, ntrials, mix))
        z = est.z.reshape(k, npp, 2)
        out_s.append((np.abs(z) ** 2 / est.s2.reshape(k, npp, 2)).sum(1))
        out_p.append(hotelling_p(np.moveaxis(z, 2, 1)))
    return np.concatenate(out_s), np.concatenate(out_p)


@dataclass
class Null:
    """Null distribution of the studentised statistic for one (participants, trials) cell."""
    stat: np.ndarray           # (nnull, 2), each column sorted

    @classmethod
    def simulate(cls, rng, npp, ntrials, nnull, mix=MIX) -> Null:
        return cls(np.sort(population_stats(rng, npp, ntrials, nnull, scale=0.0, mix=mix)[0], 0))

    def p(self, s) -> np.ndarray:
        """``(1 + #null >= s) / (1 + nnull)`` per frequency."""
        nnull = self.stat.shape[0]
        exceed = np.stack([nnull - np.searchsorted(self.stat[:, f], s[:, f]) for f in range(2)], 1)
        return (1 + exceed) / (1 + nnull)


def population_power(rng, npp, ntrials, nsim, null: Null, **effect) -> dict:
    s, p_hot = population_stats(rng, npp, ntrials, nsim, **effect)
    reject = {"agnostic": null.p(s).min(1) < ALPHA_ANY, "coherent": p_hot.min(1) < ALPHA_ANY}
    return {k: float(v.mean()) for k, v in reject.items()}


def mc_se(p, n) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-4) / n))


def save(name, payload) -> None:
    (HERE / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"saved {name}")


# --- threshold comparison (H2) -----------------------------------------------------------
Q75 = (0.75 - GUESS) / (1 - GUESS - LAPSE)
LOG_ALPHA_GRID = np.linspace(np.log(0.1), np.log(45.0), 6001)    # fit_alpha's search bounds


def tilt_for(pc, alpha):
    """Tilt at which the rig's Weibull reaches ``pc`` (QUEST+ level placement)."""
    return alpha * (-np.log1p(-(np.asarray(pc) - GUESS) / (1 - GUESS - LAPSE))) ** (1 / BETA)


def fit_thresholds(tilts, n, hits) -> np.ndarray:
    """Fixed-slope Weibull ML 75 % thresholds (estimator of
    ``threshold_by_condition.fit_alpha``) for many datasets: grid search over log alpha,
    refined by a parabola."""
    u = (tilts[None] / np.exp(LOG_ALPHA_GRID)[:, None]) ** BETA
    p = np.clip(GUESS + (1 - GUESS - LAPSE) * -np.expm1(-u), 1e-6, 1 - 1e-6)
    ll = hits @ np.log(p).T + (n - hits) @ np.log1p(-p).T                   # (M, grid)
    i = np.clip(ll.argmax(1), 1, LOG_ALPHA_GRID.size - 2)
    y0, y1, y2 = (np.take_along_axis(ll, (i + d)[:, None], 1)[:, 0] for d in (-1, 0, 1))
    step = LOG_ALPHA_GRID[1] - LOG_ALPHA_GRID[0]
    curv = y0 - 2 * y1 + y2
    offset = np.divide(0.5 * (y0 - y2), curv, out=np.zeros_like(curv), where=curv < 0)
    alpha = np.exp(LOG_ALPHA_GRID[i] + step * offset)
    return alpha * (-np.log1p(-Q75)) ** (1 / BETA)


def simulate_thresholds(rng, ntrials, threshold, nsim) -> np.ndarray:
    """ML 75 % thresholds from ``nsim`` blocks of ``ntrials`` at the rig's three levels."""
    alpha = threshold / (-np.log1p(-Q75)) ** (1 / BETA)
    tilts = tilt_for(MIX.base, alpha)
    n = rng.multinomial(ntrials, MIX.weight, size=nsim)
    return fit_thresholds(tilts, n, rng.binomial(n, MIX.base))


# --- analyses ---------------------------------------------------------------------------
SWEEP_TRIALS = [180, 360, 540, 720, 836, 900, 1080, 1440, 1800, 2160]
POP_NPP = [6, 8, 10, 12, 14, 16, 20, 24, 30, 36]
POP_TRIALS = [540, 720, 1080, 1440, 2160]


def sweep(nsim=500) -> None:
    """Per-participant power over trials x amplitude, averaged over 2 and 4 cps oscillators
    (Davidson's responders peak at either)."""
    rng = np.random.default_rng(2024)
    amps = {"typical (population RMS)": _EFFECT["rms_amp_2cps"],
            "upper quartile": _EFFECT["amp_max24_p75"],
            "Davidson's flagged (selected)": _EFFECT["amp_max24_davidson_responders"]}
    fpr = participant_power(rng, ADOPTED_TRIALS, 0.0, 2 * nsim, 0)
    print(f"per-participant FPR at {ADOPTED_TRIALS} trials: {100 * fpr:.1f} %  "
          f"(nominal 5 %, MC SE {100 * mc_se(0.05, 2 * nsim):.1f})")
    res = {}
    for ntr in SWEEP_TRIALS:
        res[str(ntr)] = {}
        for label, amp in amps.items():
            s2, s4 = (participant_power(rng, ntr, amp, nsim, f) for f in (0, 1))
            s = (s2 + s4) / 2
            res[str(ntr)][label] = {"s": s, "s_2cps": s2, "s_4cps": s4, "mc_se": mc_se(s, 2 * nsim)}
        print(f"{ntr:5d} trials | " + " | ".join(f"{a:.3f}: {100 * v['s']:5.1f} %"
                                                  for a, v in zip(amps.values(), res[str(ntr)].values())))
    save("sweep_results.json", {"ns": SWEEP_TRIALS, "amps": amps, "nsim": 2 * nsim, "nperm": N_PERM,
                                "fpr": fpr, "res": res})


def population(nsim=4000, nnull=10000) -> None:
    """Population power of both tests over participants x trials."""
    rng = np.random.default_rng(4)
    null = Null.simulate(rng, 20, ADOPTED_TRIALS, nnull)
    fpr = population_power(rng, 20, ADOPTED_TRIALS, nnull, null, scale=0.0)
    print(f"FPR, no effect, 20 x {ADOPTED_TRIALS}: phase-agnostic {100 * fpr['agnostic']:.1f} %, "
          f"common-phase {100 * fpr['coherent']:.1f} %  (nominal 5 %)")
    power = {}
    for npp in POP_NPP:
        power[str(npp)] = {}
        for ntr in POP_TRIALS:
            cell = population_power(rng, npp, ntr, nsim, Null.simulate(rng, npp, ntr, nnull))
            power[str(npp)][str(ntr)] = cell
        print(f"N = {npp:2d} | " + " | ".join(f"{t}: {100 * c['agnostic']:5.1f} / {100 * c['coherent']:4.1f}"
                                            for t, c in power[str(npp)].items()))
    save("group_power_results.json", {
        "rms": RMS.tolist(), "coherent": COHERENT.tolist(), "alpha_any": ALPHA_ANY,
        "fpr_agnostic": fpr["agnostic"], "fpr_coherent": fpr["coherent"],
        "npp": POP_NPP, "ntrials": POP_TRIALS, "nsim": nsim, "nnull": nnull, "power": power})


def robustness(nsim=4000, nnull=10000) -> None:
    """Confirmatory power under different distributions of the effect across participants."""
    rng = np.random.default_rng(0)
    models = {"gaussian": None, "spike-slab 0.50": 0.5, "0.25": 0.25, "0.12": 0.12}
    print("phase-agnostic power by population structure (matched E|c|^2 = RMS^2)")
    for npp, ntr in [(12, 1080), (12, 1440), (20, 810), (20, 1440)]:
        null = Null.simulate(rng, npp, ntr, nnull)
        cells = [population_power(rng, npp, ntr, nsim, null, prevalence=g)["agnostic"] for g in models.values()]
        print(f"  {npp} x {ntr:4d} | " + " | ".join(f"{m}: {100 * c:5.1f} %" for m, c in zip(models, cells)))


def mda(nsim=2000, nnull=10000) -> None:
    """Smallest amplitude detected with 80 % power per design; this comparison does not
    depend on Davidson's exact amplitude."""
    rng = np.random.default_rng(11)
    designs = {"one_visit": 810, "two_visits": ADOPTED_TRIALS}
    scales = [0.30, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.45, 1.75, 2.10]
    out = {"npp": POP_NPP, "scales": scales, "designs": designs, "rms_2cps": float(RMS[0]),
           "target": 0.80, "ref_scale": 1.0, "nsim": nsim, "nsim_ref": 2 * nsim, "nnull": nnull,
           "power": {}, "mda": {}}
    for name, ntr in designs.items():
        out["power"][name], out["mda"][name] = {}, {}
        for npp in POP_NPP:
            null = Null.simulate(rng, npp, ntr, nnull)
            row = [population_power(rng, npp, ntr, 2 * nsim if s == 1.0 else nsim, null, scale=s)["agnostic"]
                   for s in scales]
            # Power increases with effect size; dips are MC noise and are removed before
            # interpolating.
            mono = np.maximum.accumulate(row)
            out["power"][name][str(npp)] = row
            out["mda"][name][str(npp)] = (float(np.interp(0.8, mono, np.array(scales) * RMS[0]))
                                          if mono[-1] >= 0.8 else None)
            print(f"{name} N = {npp:2d}: power at 1x {100 * row[scales.index(1.0)]:5.1f} %, "
                  f"MDA {out['mda'][name][str(npp)]}")
    save("design_compare_results.json", out)


def tradeoff(nsim=4000, nnull=10000, nsim_threshold=4000, nsim_h2=2000) -> None:
    """Walking/standing split: confirmatory power vs walking trials, and the cost
    of fewer standing trials for the threshold comparison H2."""
    rng = np.random.default_rng(902)
    walking = [720, 900, 1080, 1260, 1440]
    gain = {}
    for npp in (12, 15, 20):
        gain[str(npp)] = [population_power(rng, npp, t, nsim, Null.simulate(rng, npp, t, nnull))["agnostic"]
                          for t in walking]
        print(f"gain  N = {npp}: " + ", ".join(f"{t}: {p:.3f}" for t, p in zip(walking, gain[str(npp)])))

    true = 6.0
    standing = [180, 360, 540, 720]
    est = [simulate_thresholds(rng, t, true, nsim_threshold) for t in standing]
    sd = [float(e.std(ddof=1)) for e in est]
    print("cost  threshold SD: " + ", ".join(f"{t}: {s:.3f} deg" for t, s in zip(standing, sd)))

    # H2: crossover condition contrast 0.5 [(wf + ws) - (sf + ss)] on log10 threshold,
    # one-sample t. The estimator is scale-invariant (levels at fixed points of each
    # condition's function), so between-participant spread cancels; only estimation error
    # remains.
    per_visit = {"standing": DESIGNS["adopted"]["standing_blocks"] // 2 * TRIALS_PER_BLOCK,
                 "walking": DESIGNS["adopted"]["walking_blocks"] // 2 * TRIALS_PER_BLOCK}
    h2 = {}
    for ratio in (1.10, 1.15, 1.30):
        err = {c: np.log10(simulate_thresholds(rng, n, true, 2 * 12 * nsim_h2) / true).reshape(nsim_h2, 12, 2)
               for c, n in per_visit.items()}
        contrast = np.log10(ratio) + err["walking"].mean(2) - err["standing"].mean(2)
        h2[f"power_ratio_{ratio:.2f}".replace(".", "_")] = float(
            (stats.ttest_1samp(contrast, 0, axis=1).pvalue < ALPHA).mean())
    print("cost  H2 power at N = 12: " + ", ".join(f"{k[-4:]}: {v:.3f}" for k, v in h2.items()))

    blocks = lambda trials: [t // TRIALS_PER_BLOCK for t in trials]
    save("design_tradeoff.json", {
        "_note": "Inputs to the 8/2 walking/standing block split, generated by "
                 "`python -m Analysis.davidson_power.power tradeoff`.",
        "gain": {"_what": "Power of the confirmatory phase-agnostic population test vs walking trials "
                          "per participant (power.population_power, Holm over 2 and 4 cps).",
                 "n_sim": nsim, "walking_trials": walking, "walking_blocks": blocks(walking), "power": gain},
        "cost_threshold": {"_what": "SD of the fixed-slope Weibull ML 75 % threshold (the estimator "
                                    "threshold_by_condition uses) vs standing trials, levels at the rig's "
                                    "65/75/85 % points mixed 0.05/0.85/0.10.",
                           "n_sim": nsim_threshold, "true_threshold_deg": true, "standing_trials": standing,
                           "standing_blocks": blocks(standing), "sd_deg": [round(s, 3) for s in sd],
                           "sd_pct_of_threshold": [round(100 * s / true, 1) for s in sd],
                           "ci95_width_deg": [round(float(np.subtract(*np.percentile(e, [97.5, 2.5]))), 2)
                                              for e in est]},
        "cost_h2": {"_what": "Power of H2, the crossover model's walking-vs-standing contrast on log10 "
                             "threshold (one-sample t, N = 12), at the adopted 1 standing + 4 walking "
                             "blocks per visit.", "n_sim": nsim_h2, "n_participants": 12,
                    "trials_per_visit": per_visit, **h2},
        "adopted": {k: {**d, "walking_trials": d["walking_blocks"] * TRIALS_PER_BLOCK,
                        "standing_trials": d["standing_blocks"] * TRIALS_PER_BLOCK}
                    for k, d in (("old", DESIGNS["former"]), ("new", DESIGNS["adopted"]))}})


def checks() -> None:
    """Numerical checks of the simulation's assumptions."""
    rng = np.random.default_rng(11)
    print("1. Null scale: simulated E|z|^2 vs the closed-form s^2 vs 4 Var/N")
    var = float((MIX.weight * MIX.base * (1 - MIX.base)).sum())
    for ntr in (360, 720, 1440, 2160):
        est = estimate(simulate_tables(rng, np.zeros((20000, 2), complex), ntr))
        print(f"   {ntr:4d} trials: E|z|^2 {np.mean(np.abs(est.z[:, 0]) ** 2):.3e}, "
              f"s^2 {est.s2[:, 0].mean():.3e}, 4 Var/N {4 * var / ntr:.3e}")

    print("\n2. Amplitude recovery at 1440 trials, sqrt(E|z|^2 - E s^2), all trials at threshold")
    for amp in (0.0, RMS[0], 0.05, 0.10):
        coef = np.zeros((20000, 2), complex)
        coef[:, 0] = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, 20000))
        est = estimate(simulate_tables(rng, coef, ADOPTED_TRIALS, ALL_AT_THRESHOLD))
        got = np.sqrt(max(np.mean(np.abs(est.z[:, 0]) ** 2 - est.s2[:, 0]), 0))
        print(f"   true {amp:.3f} -> recovered {got:.4f} (expected {amp * BIN_GAIN[0]:.4f} after bin averaging)")

    print("\n3. Per-participant test calibration (no effect)")
    for ntr in (360, 1440, 2160):
        fpr = participant_power(rng, ntr, 0.0, 1000, 0)
        print(f"   {ntr:4d} trials: FPR {100 * fpr:.1f} %  (nominal 5 %, MC SE {100 * mc_se(0.05, 1000):.1f})")

    print("\n4. Target-onset phase coverage under ISI 4 s +- 0.75 s (uniformity is assumed above)")
    for cv in (0.0, 0.02, 0.04):
        for stride in (1.05, 1.10):
            onset = np.cumsum(4.0 + rng.uniform(-0.75, 0.75, ADOPTED_TRIALS))
            n = int(onset[-1] / (stride * (1 - 4 * cv)) + 100)
            boundaries = np.cumsum(stride * (1 + rng.normal(0, cv, n)))
            phase = np.interp(onset, boundaries, np.arange(n)) % 1.0
            r = [abs(np.exp(2j * np.pi * k * phase).mean()) for k in (1, 2, 4)]
            print(f"   stride {stride:.2f} s, CV {cv:.0%}: resultant length @1/2/4 cps "
                  + " / ".join(f"{v:.3f}" for v in r))
    print(f"   (chance level for {ADOPTED_TRIALS} onsets ~{np.sqrt(np.pi / 4 / ADOPTED_TRIALS):.3f})")

    print("\n5. What the difficulty mix costs (per-participant power, amp 0.05, 1440 trials)")
    print("   level gains " + ", ".join(f"{k} {g:.3f}" for k, g in zip(LEVELS, MIX.gain))
          + f"; weighted {MIX.weight @ MIX.gain:.3f}")
    for label, mix in (("all trials at threshold", ALL_AT_THRESHOLD), ("rig mix, all levels carry", MIX),
                       ("rig mix, threshold trials only", THRESHOLD_ONLY)):
        print(f"   {label:32s} {100 * participant_power(rng, ADOPTED_TRIALS, 0.05, 600, 0, mix):5.1f} %")


def validate() -> None:
    """Check each estimator step against the pipeline function."""
    import common_phase as cph
    import group_agnostic as ga
    import phase_analysis as pa
    import threshold_by_condition as tbc

    rng = np.random.default_rng(7)
    names = np.array(list(LEVELS))

    def dataset(ntr, amp):
        phase = rng.uniform(0, 100, ntr)
        level = rng.choice(3, ntr, p=MIX.weight)
        p = MIX.base[level] + MIX.gain[level] * amp * np.cos(OMEGA[0] * phase + rng.uniform(0, 2 * np.pi))
        return phase, level, (rng.random(ntr) < p).astype(float)

    def permuted_z(t, est, row, nperm):
        """Coefficients of ``nperm`` trial-level shuffles of one participant, ``(nperm, 2)``."""
        bins, values = trials(t, est, row)
        sums = bin_sums(bins, rng.permuted(np.broadcast_to(values, (nperm, values.size)), axis=1))
        b = np.einsum("fij,fbj,kb->kfi", est.gram_inv[row], DESIGN, sums)
        return b[..., 1] + 1j * b[..., 2]

    worst = np.zeros(2)
    for _ in range(25):
        phase, level, y = dataset(int(rng.integers(200, 1500)), 0.08)
        est = estimate(tables_from_trials(phase, level, y, 3))
        v = pa.difficulty_adjust(y, names[level].astype(object))
        _, mean, _, bin_n = pa._bin(phase, v, NBINS, "mean", with_counts=True)
        ok = np.isfinite(mean)
        beta = pa._spectrum_with_coef(CENTRES[ok], mean[ok], bin_n[ok])[1][[pa.IDX_2CPS, pa.IDX_4CPS]]
        worst = np.maximum(worst, [np.abs(v - (y + est.shift[0][level])).max(),
                                   np.abs(beta[:, 1] + 1j * beta[:, 2] - est.z[0]).max()])
    print(f"residualisation vs difficulty_adjust: max |diff| {worst[0]:.1e}")
    print(f"weighted (cos, sin) coefficients vs _spectrum_with_coef: max |diff| {worst[1]:.1e}")

    agree, n = 0, 25
    for i in range(n):
        phase, level, y = dataset(int(rng.integers(300, 1200)), rng.choice([0.0, 0.06, 0.1]))
        v = pa.difficulty_adjust(y, names[level].astype(object))
        pa.RNG = np.random.default_rng(100 + i)           # the pipeline uses its module RNG
        theirs = pa.analyse_measure(phase, v, "mean", nperm=400)
        shared = np.random.default_rng(100 + i)
        shuffles = np.stack([shared.permutation(v) for _ in range(400)])
        mine = participant_test(np.clip(np.digitize(phase, EDGES) - 1, 0, NBINS - 1), v, shuffles)
        agree += np.allclose(mine, [theirs["p_2cps"], theirs["p_4cps"]])
    print(f"per-participant p-values vs analyse_measure on shared permutations: {agree}/{n} identical")

    z = rng.normal(0.2, 1.0, (60, 15, 2))
    print("Hotelling vs common_phase.hotelling_t2: max |diff| "
          f"{max(abs(hotelling_p(x) - cph.hotelling_t2(x)['p_value']) for x in z):.1e}")

    t = simulate_tables(rng, draw_coefficients(rng, (1,)), ADOPTED_TRIALS)
    est = estimate(t)
    ratio = est.s2[0] / (np.abs(permuted_z(t, est, 0, 20000)) ** 2).mean(0)
    print(f"closed-form s^2 / permutation mean |z|^2 (20000 shuffles): {ratio.round(4)}")

    npp, nsim = 12, 300
    null = Null.simulate(rng, npp, ADOPTED_TRIALS, 4000)
    mc = population_power(rng, npp, ADOPTED_TRIALS, nsim, null)["agnostic"]
    hits = 0
    for _ in range(nsim):
        t = simulate_tables(rng, draw_coefficients(rng, (npp,)), ADOPTED_TRIALS)
        est = estimate(t)
        null_z = np.stack([permuted_z(t, est, row, N_PERM) for row in range(npp)])
        p = [ga.agnostic_test(np.stack([est.z[:, f].real, est.z[:, f].imag], 1),
                              np.stack([null_z[..., f].real, null_z[..., f].imag], 2))["p_value"] for f in range(2)]
        hits += min(p) < ALPHA_ANY
    print(f"population power, {npp} x {ADOPTED_TRIALS}: simulated null {100 * mc:.1f} % vs "
          f"group_agnostic.agnostic_test on real permutations {100 * hits / nsim:.1f} % "
          f"(MC SE ~{100 * mc_se(mc, nsim):.0f})")

    gap = 0.0
    for _ in range(20):
        tilt = rng.choice(tilt_for(MIX.base, 6.0), 400, p=MIX.weight)
        hit = (rng.random(400) < GUESS + (1 - GUESS - LAPSE) * -np.expm1(-(tilt / 6.0) ** BETA)).astype(float)
        levels = np.unique(tilt)
        n = np.array([[np.sum(tilt == x) for x in levels]])
        h = np.array([[hit[tilt == x].sum() for x in levels]])
        gap = max(gap, abs(fit_thresholds(levels, n, h)[0] / tbc.fit_alpha(tilt, hit)["thresh75_deg"] - 1))
    print(f"threshold ML vs threshold_by_condition.fit_alpha: max relative diff {gap:.1e}")


TASKS = {"validate": validate, "checks": checks, "sweep": sweep, "population": population,
         "robustness": robustness, "mda": mda, "tradeoff": tradeoff}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", choices=[*TASKS, "all"])
    task = ap.parse_args().task
    for name, fn in TASKS.items():
        if task in (name, "all") and not (task == "all" and name in ("validate", "checks")):
            t0 = time.time()
            print(f"\n=== {name} ===")
            fn()
            print(f"({time.time() - t0:.0f} s)")
