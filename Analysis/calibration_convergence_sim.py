"""Simulation of calibration convergence and reliability over a broad participant range.

The runtime rule is C# (StaircaseManager), so this reimplements the same 1-parameter
QUEST+ with two trackers in Python and tests the precision-based stop under realistic
between- and within-participant variation.

  Stop rule: the two trackers run interleaved; stop once the pooled posterior SD (the final
  estimate) is <= convergenceSDLog (0.10 log10) after >= minScoredTrials (30). Hard cap 100;
  abort at 80 if the pooled SD is still > 0.18. "Converged" means this precision was reached.
  Reliable = converged, catch trials OK and estimate not at the grid ceiling (A/B
  agreement |aA-aB| <= 0.15 is logged only).

What it checks:
  1. The stop is efficient and accurate for attentive observers (ideal cohort).
  2. Thresholds across 0.3-6 deg, not only the central 0.5-2 deg.
  3. Failure modes that threshold trials alone cannot detect:
       * high-lapse (inattentive) observers often converge to a precise but biased value.
         Easy catch trials flag them (catch error rate ~ true lapse rate); this is why
         catch trials were added.
       * drifting (non-stationary) observers: larger error and more aborts.

  python -m Analysis.calibration_convergence_sim
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- model and grid (StaircaseManager defaults) ---
MIN_DEG, MAX_DEG, N_STIM, N_PARAM = 0.3, 25.0, 41, 61
GAMMA, LAMBDA_FIT, BETA_FIT = 0.5, 0.05, 2.5      # fixed in QUEST+ (assumed)
START_A, START_B, PRIOR_SD = 4.0, 2.5, 0.5        # high starts, broad prior
CAP = 100

# current stop rule (StaircaseManager defaults)
CONV_SD, MIN_TRIALS, ABORT_AT, ABORT_SD = 0.10, 30, 80, 0.18
# previous stop rule, for the efficiency comparison
BASE_PER_TRACKER, AGREE_TOL = 25, 0.15
QC_SD = 0.12                                       # analysis "wide posterior" threshold
# catch trials (easy anchor)
CATCH_INTERVAL, MAX_CATCH_RATE, MIN_CATCH = 6, 0.10, 4
RAIL_DEG = 0.5 * MAX_DEG          # estimates above this ran away to the ceiling


def _weibull(x_log, a_log, beta=BETA_FIT, lam=LAMBDA_FIT):
    t = np.power(10.0, beta * (x_log - a_log))
    return GAMMA + (1.0 - GAMMA - lam) * (1.0 - np.exp(-t))


class Quest:
    """1-parameter (threshold) QUEST+ with minimum-expected-entropy placement."""

    def __init__(self, center_deg, prior_sd):
        lo, hi = np.log10(MIN_DEG), np.log10(MAX_DEG)
        self.stim = np.linspace(lo, hi, N_STIM)
        self.par = np.linspace(lo, hi, N_PARAM)
        self.pc = np.clip(_weibull(self.stim[:, None], self.par[None, :]), 1e-6, 1 - 1e-6)
        if prior_sd > 0:
            z = (self.par - np.log10(center_deg)) / prior_sd
            p = np.exp(-0.5 * z * z)
        else:
            p = np.ones_like(self.par)
        self.post = p / p.sum()
        self.last = 0

    def next_stim(self):
        """Choose the level with minimum expected posterior entropy over both responses."""
        pcorr = np.clip(self.pc @ self.post, 1e-6, 1 - 1e-6)
        pc_post = self.post[None, :] * self.pc / pcorr[:, None]
        pi_post = self.post[None, :] * (1 - self.pc) / (1 - pcorr)[:, None]
        Hc = -np.sum(pc_post * np.log(np.clip(pc_post, 1e-12, None)), axis=1)
        Hi = -np.sum(pi_post * np.log(np.clip(pi_post, 1e-12, None)), axis=1)
        eh = pcorr * Hc + (1 - pcorr) * Hi
        self.last = int(np.argmin(eh))
        return self.stim[self.last]

    def update(self, correct):
        """Update the posterior with the response to the last stimulus."""
        like = self.pc[self.last] if correct else (1 - self.pc[self.last])
        self.post *= like
        self.post /= self.post.sum()

    def update_at(self, x_log, correct):
        """Update from a response at any level, not one chosen by this tracker.

        Used for the pooled posterior, which receives both trackers' trials.
        """
        row = np.clip(_weibull(x_log, self.par), 1e-6, 1 - 1e-6)
        self.post *= row if correct else (1 - row)
        self.post /= self.post.sum()

    def thr_log(self):
        """Posterior mean threshold, in log10 degrees."""
        return float(self.post @ self.par)

    def sd_log(self):
        """Posterior SD of the threshold (log10 deg), the convergence criterion."""
        m = self.thr_log()
        return float(np.sqrt(max(0.0, self.post @ ((self.par - m) ** 2))))


def _run_one(thr_log_fn, true_lambda, true_beta, rng):
    """Run one observer for CAP scored trials with both trackers interleaved.

    thr_log_fn(t) is the true threshold (log10 deg) at trial t: constant, or a ramp for
    drifting observers. Returns per-trial pooled SD, A/B gap, pooled threshold and true
    threshold.
    """
    A, B = Quest(START_A, PRIOR_SD), Quest(START_B, PRIOR_SD)
    pool = Quest(START_A, 0.0)                              # uniform prior
    pooledSD, gap, pthr, a_trues = [], [], [], []
    for t in range(CAP):
        trk = A if (t % 2 == 0) else B
        x = trk.next_stim()
        a_true = thr_log_fn(t)
        a_trues.append(a_true)
        p = float(np.clip(_weibull(x, a_true, true_beta, true_lambda), 0, 1))
        correct = rng.random() < p
        trk.update(correct)
        pool.update_at(x, correct)
        pooledSD.append(pool.sd_log())
        gap.append(abs(A.thr_log() - B.thr_log()))
        pthr.append(pool.thr_log())
    return np.array(pooledSD), np.array(gap), np.array(pthr), np.array(a_trues)


def _stop_old(gap):
    base = 2 * BASE_PER_TRACKER
    for t in range(base - 1, CAP):
        if gap[t] <= AGREE_TOL:
            return t + 1
    return CAP


def _stop_new(pooledSD):
    for t in range(MIN_TRIALS - 1, CAP):
        if pooledSD[t] <= CONV_SD:
            return t + 1, True                              # converged
        if ABORT_AT > 0 and (t + 1) >= ABORT_AT and pooledSD[t] > ABORT_SD:
            return t + 1, False                             # abort
    return CAP, pooledSD[CAP - 1] <= CONV_SD


def simulate_cohort(n, true_lambda, drift_log, rng, thr_lo=0.3, thr_hi=6.0):
    """Return per-observer records for one cohort across thr_lo..thr_hi (log-uniform)."""
    thr0 = 10 ** rng.uniform(np.log10(thr_lo), np.log10(thr_hi), n)
    recs = []
    for thr in thr0:
        base = np.log10(thr)
        thr_fn = (lambda t, b=base: b) if drift_log == 0 else \
                 (lambda t, b=base: b + drift_log * (t / (CAP - 1)))
        pooledSD, gap, pthr, a_trues = _run_one(thr_fn, true_lambda, BETA_FIT, rng)
        tn, conv = _stop_new(pooledSD)
        agree = gap[tn - 1] <= AGREE_TOL
        # Catch trials up to the stop; with the model's upper asymptote, P(wrong) = true_lambda.
        n_catch = tn // CATCH_INTERVAL
        c_err = int(rng.binomial(n_catch, true_lambda)) if n_catch > 0 else 0
        c_rate = (c_err / n_catch) if n_catch else 0.0
        # Flag only with >= 2 errors and a rate above threshold; one lapse is not enough.
        catch_ok = not (n_catch >= MIN_CATCH and c_err >= 2 and c_rate > MAX_CATCH_RATE)
        # An estimate at the grid ceiling means no threshold was found (e.g. random
        # responding); never reliable, however narrow the posterior.
        railed = (10 ** pthr[tn - 1]) >= RAIL_DEG
        # The banner requires precision, passed catch trials and no ceiling estimate. A/B
        # agreement is too noisy at ~20 trials per tracker to gate it; it is logged for QC.
        reliable = conv and catch_ok and not railed
        a_ref = float(a_trues[:tn].mean())                 # mean true threshold up to the stop
        err = abs(pthr[tn - 1] - a_ref)
        recs.append((tn, conv, agree, catch_ok, reliable,
                     (not conv and tn < CAP), (tn >= CAP and not conv), err))
    return np.array(recs, float)


def _row(name, r):
    trials, conv, agree, cok, rel, abrt, cap, err = (r[:, i] for i in range(8))
    return (f"  {name:11s} n={len(r):3d} | trials {np.median(trials):3.0f} "
            f"| conv {100*conv.mean():3.0f}% | RELIABLE {100*rel.mean():3.0f}% "
            f"| catch-flag {100*(1-cok).mean():3.0f}% | A/B-disagree {100*(1-agree).mean():3.0f}% "
            f"| abort {100*abrt.mean():3.0f}% | cap {100*cap.mean():3.0f}% "
            f"| |err| {np.median(err):.3f} ({10**np.median(err):.2f}x)")


def main():
    """Simulate QUEST+ convergence to justify the calibration trial budget and stop rule."""
    rng = np.random.default_rng(0)
    N = 200
    print("=== Calibration across a broad participant range (NEW precision rule) ===")
    print(f"  {N} observers/cohort, thresholds 0.3-6.0 deg (log-uniform), cap {CAP}")
    print("  RELIABLE = converged AND catch-OK (A/B agreement logged but NOT gating)")
    ideal = simulate_cohort(N, 0.03, 0.0, rng)
    hilapse = simulate_cohort(N, 0.15, 0.0, rng)
    diseng = simulate_cohort(N, 0.45, 0.0, rng)            # ~random responder
    drift = simulate_cohort(N, 0.03, 0.30, rng)            # threshold ~2x higher by the end
    print(_row("ideal", ideal))
    print(_row("high-lapse", hilapse))
    print(_row("disengaged", diseng))
    print(_row("drifting", drift))

    print("\n  Reading the table:")
    print(f"  * ideal: {100*ideal[:,4].mean():.0f}% RELIABLE, ~{np.median(ideal[:,0]):.0f} trials, "
          f"err {10**np.median(ideal[:,7]):.2f}x. Note A/B-disagree {100*(1-ideal[:,2]).mean():.0f}% "
          f"even here\n           - that is why agreement is logged but does NOT gate the banner.")
    print(f"  * high-lapse (0.15): {100*hilapse[:,1].mean():.0f}% still CONVERGE (precise) but "
          f"only {100*hilapse[:,4].mean():.0f}% RELIABLE - catch + abort demote the rest; "
          f"threshold is biased ({10**np.median(hilapse[:,7]):.2f}x).\n           Borderline lapse "
          f"is only PARTLY caught with the ~6-8 catches a short calibration affords.")
    print(f"  * disengaged (0.45): caught hard - {100*(1-diseng[:,3]).mean():.0f}% catch-flagged "
          f"plus the ceiling-rail; {100*diseng[:,4].mean():.0f}% slip through as RELIABLE "
          f"(err {10**np.median(diseng[:,7]):.1f}x).")
    print(f"  * drifting: mild drift (2x over the run) is ABSORBED into the time-mean estimate "
          f"(err {10**np.median(drift[:,7]):.2f}x); only severe/abrupt drift widens the "
          f"posterior enough to abort.")

    # Previous vs current stop rule on attentive observers (same response streams).
    rng2 = np.random.default_rng(1)
    olds, news = [], []
    for thr in 10 ** rng2.uniform(np.log10(0.5), np.log10(2.0), N):
        b = np.log10(thr)
        pooledSD, gap_arr, pthr, _ = _run_one(lambda t, b=b: b, 0.03, BETA_FIT, rng2)
        olds.append(_stop_old(gap_arr))
        news.append(_stop_new(pooledSD)[0])
    print(f"\n  Efficiency (attentive, 0.5-2 deg): OLD median {np.median(olds):.0f} trials "
          f"vs NEW {np.median(news):.0f} -> ~{np.median(olds)-np.median(news):.0f} fewer.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
