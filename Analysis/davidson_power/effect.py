"""Measure the stride-phase effect size in Davidson et al. (2024)'s raw data.

Source: https://osf.io/8djtq/ (Nat Commun 15:2027)::

    python -m Analysis.davidson_power.effect --download   # the two .mat files, ~49 MB
    python -m Analysis.davidson_power.effect              # -> davidson_effect.json
    python -m Analysis.davidson_power.effect --verify     # evidence for the choices below

Each choice follows Davidson's MATLAB code on the same OSF node
(``j4_crunchPFX_concatGFX.m``, ``j5_createNullGFX.m``, ``j8_testfourier_Obs_Null_GFX.m``):

* Curve. The second index of ``GFX_TargPosData`` is the foot split
  ``{LRL, RLR, combined}``, not the gait type. ``j8`` hard-codes ``iLR = 3``, so the
  published fits use index 2; only that slice reproduces the stored spectra.
* Fit. ``fourier1`` at fixed ``w = 2*pi*Hz/xvec(end)`` on bin centres
  ``xvec = pidx2(1:40) + 1``, over ``Hzspace = 0.01:0.2:10``, hence peaks at 2.01 / 4.01.
* Binning. Unweighted mean of per-percentile accuracies over bin ranges that overlap at
  their endpoints.
* Duplication. ``j2C_append_doubGCData.m`` writes each target twice, once per foot
  parameterisation, about half a stride apart, and ``j4`` concatenates both for
  ``iLR = 3``. The combined curve has ~1670 rows from ~836 independent trials, so the
  noise floor is bootstrapped with this duplication instead of using ``4p(1-p)/N``.

Amplitudes are taken at the two pre-specified frequencies, not at each participant's
peak, and are not conditioned on Davidson's per-participant flag (its actual FPR under
the duplication is ~15 %).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
OSF = {HERE / "GFX_Data_inGaits.mat": "https://osf.io/download/653cd22c8a28b115ecffcb39/",
       HERE / "GFX_Data_inGaits_FourierFits.mat": "https://osf.io/download/ckyq9/"}
MAT_MAIN, MAT_FITS = OSF
OUT = HERE / "davidson_effect.json"

COMBINED = 2            # foot slice Davidson fit (iLR = 3 in his MATLAB)
JITTER = 2              # LRL/RLR offset jitter in stride percentiles; --verify gives +-0..2,
                        # and the RMS amplitude changes < 0.0003 over +-1..3
N_PP = 36
N_BOOT = 4000
PERCENTILES = np.arange(1, 101)


def download() -> None:
    """Fetch the two OSF archives, skipping any already present."""
    for path, url in OSF.items():
        if path.exists():
            print(f"  have {path.name}")
            continue
        print(f"  fetching {path.name} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "python"})
        with urllib.request.urlopen(request, timeout=900) as r:
            path.write_bytes(r.read())
        print(f"  saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def load():
    """``(behavioural struct, published-fit struct)``; the fits are used for validation."""
    if not all(p.exists() for p in OSF):
        raise SystemExit("missing .mat files: run with --download first")
    return tuple(sio.loadmat(p, squeeze_me=True, struct_as_record=False) for p in OSF)


class Geometry:
    """Davidson's binning and Fourier basis, read from his archive so the bins match exactly."""

    def __init__(self, m, ff):
        self.pidx = np.asarray(m["pidx2"], int)                     # ceil(linspace(1,100,41))
        self.xvec = (self.pidx[:-1] + round(np.diff(self.pidx).mean() / 2)).astype(float)
        self.hz = np.asarray(ff["Hzspace"], float)
        w = 2 * np.pi * self.hz / self.xvec[-1]
        self.cos, self.sin = np.cos(np.outer(w, self.xvec)), np.sin(np.outer(w, self.xvec))
        self.i2, self.i4 = (int(np.abs(self.hz - f).argmin()) for f in (2.0, 4.0))
        nb = self.xvec.size
        self.members = np.zeros((nb, 100))                          # overlapping bin ranges
        for i in range(nb):
            self.members[i, self.pidx[i] - 1:self.pidx[i + 1]] = 1.0

    def fit(self, Y):
        """Fit binned curves ``(n, 40)`` (NaN allowed).

        Returns ``R2 (n, nfreq)`` and the complex coefficient ``a + i*b`` of
        ``c + a cos + b sin`` per frequency. OLS with intercept via the centred 2 x 2 normal
        equations, solved once per pattern of missing bins."""
        Y = np.atleast_2d(Y)
        R2 = np.zeros((Y.shape[0], self.hz.size))
        Z = np.zeros_like(R2, complex)
        ok = np.isfinite(Y)
        patterns, inverse = np.unique(ok, axis=0, return_inverse=True)
        for k, mask in enumerate(patterns):
            rows = np.flatnonzero(inverse.ravel() == k)
            if mask.sum() < 6:
                continue
            yc = Y[np.ix_(rows, mask)]
            yc = yc - yc.mean(1, keepdims=True)
            C, S = (B[:, mask] - B[:, mask].mean(1, keepdims=True) for B in (self.cos, self.sin))
            a, b, d = (C * C).sum(1), (C * S).sum(1), (S * S).sum(1)
            gc, gs = yc @ C.T, yc @ S.T
            det = a * d - b * b
            bc, bs = (d * gc - b * gs) / det, (a * gs - b * gc) / det
            sst = (yc * yc).sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                R2[rows] = np.clip(np.where(sst > 0, (bc * gc + bs * gs) / sst, 0.0), 0, 1)
            Z[rows] = bc + 1j * bs
        return R2, Z

    def bin_rows(self, pos, val):
        """Per-percentile accuracy from stride percentiles (1..100) and 0/1 outcomes, then
        Davidson's unweighted mean over each overlapping bin range."""
        n = pos.shape[0]
        flat = ((pos - 1) + 100 * np.arange(n)[:, None]).ravel()
        hits = np.bincount(flat, weights=val.ravel(), minlength=100 * n).reshape(n, 100)
        count = np.bincount(flat, minlength=100 * n).reshape(n, 100)
        seen = count > 0
        acc = np.divide(hits, count, out=np.zeros_like(hits), where=seen)
        num, den = acc @ self.members.T, seen @ self.members.T
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


class Participant(NamedTuple):
    curve: np.ndarray          # 40 binned accuracies (combined, duplicated slice)
    baseline: float
    phase_prob: np.ndarray     # distribution of target stride percentiles (LRL slice)
    n_independent: int         # trials, not rows (each target appears twice)


def participant(m, p) -> Participant:
    cell = m["GFX_TargPosData"]
    curve = np.asarray(cell[p, COMBINED].doubgc_binned_Acc, float)
    lrl = np.nan_to_num(np.asarray(cell[p, 0].doubgc_counts, float))
    rlr = np.nansum(np.asarray(cell[p, 1].doubgc_counts, float))
    return Participant(curve, float(np.nanmean(curve)), lrl / lrl.sum(),
                       round((lrl.sum() + rlr) / 2))


def paired_rows(rng, prob, nuniq, nsim, jitter=JITTER):
    """Duplicated rows: each target's LRL percentile and its RLR copy ~50 % later."""
    lrl = rng.choice(PERCENTILES, size=(nsim, nuniq), p=prob)
    j = rng.integers(-jitter, jitter + 1, size=(nsim, nuniq)) if jitter else 0
    return lrl, (lrl + 50 + j - 1) % 100 + 1


def simulate(g: Geometry, part: Participant, nsim, rng, amp=0.0, freq=2.01, jitter=JITTER):
    """Simulate Davidson's design with a known effect: independent trials, each entered twice."""
    lrl, rlr = paired_rows(rng, part.phase_prob, part.n_independent, nsim, jitter)
    phi = rng.uniform(0, 2 * np.pi, (nsim, 1))
    p = np.clip(part.baseline + amp * np.cos(2 * np.pi * freq * lrl / 100.0 + phi), 0.01, 0.99)
    hit = (rng.random(lrl.shape) < p).astype(float)
    return g.fit(g.bin_rows(np.concatenate([lrl, rlr], 1), np.concatenate([hit, hit], 1)))


def debiased_rms(amp, floor, axis=-1):
    """RMS true amplitude: ``E|z_obs|^2 = |z_true|^2 + floor^2``, averaged over participants."""
    return np.sqrt(np.maximum((amp ** 2).mean(axis) - (floor ** 2).mean(axis), 0))


def coherent_amplitude(z, axis=-1):
    """Unbiased |population mean coefficient|: ``|z_bar|^2 - sum|z_i - z_bar|^2 / (N (N-1))``.

    The subtracted term is the variance of the mean, removing both measurement noise and
    between-participant phase scatter. Subtracting the noise alone would leave
    ``var_between / N`` and overstate the coherent amplitude.
    """
    n = z.shape[axis]
    zbar = z.mean(axis, keepdims=True)
    var_mean = (np.abs(z - zbar) ** 2).sum(axis) / (n * (n - 1))
    return np.sqrt(np.maximum(np.abs(zbar.squeeze(axis)) ** 2 - var_mean, 0))


def main(nsim=1500) -> None:
    m, ff = load()
    g = Geometry(m, ff)
    fits = ff["PFX_FourierNull"]
    rng = np.random.default_rng(20240624)
    parts = [participant(m, p) for p in range(N_PP)]

    Y = np.array([p.curve for p in parts])
    R2, Z = g.fit(Y)
    his_threshold = np.array([float(f.TargetOns_doubgc_binned_Acc_fitsRsq_ShuffCV_max[-1]) for f in fits])
    his_spectra = np.array([np.asarray(f.TargetOns_doubgc_binned_Acc_fitsRsq_Obs, float) for f in fits])
    flagged = R2.max(1) > his_threshold
    print(f"fidelity: max |our R2 - his stored per-participant R2| = {np.nanmax(np.abs(R2 - his_spectra)):.1e}")
    print(f"his flag reproduced: {flagged.sum()}/{N_PP} (the paper reports 8/36)\n")

    print(f"bootstrapping the duplicated noise floor (nsim = {nsim}) ...")
    floor, fpr = np.empty((N_PP, 2)), np.empty(N_PP)
    for p, part in enumerate(parts):
        R2n, Zn = simulate(g, part, nsim, rng)
        floor[p] = np.sqrt((np.abs(Zn[:, [g.i2, g.i4]]) ** 2).mean(0))
        fpr[p] = (R2n.max(1) > his_threshold[p]).mean()

    amp = np.abs(Z[:, [g.i2, g.i4]])                                  # (N_PP, 2)
    debiased = np.sqrt(np.maximum(amp ** 2 - floor ** 2, 0))
    best = debiased.max(1)
    n_trials = np.array([p.n_independent for p in parts], float)
    baseline = np.array([p.baseline for p in parts])
    boot = rng.integers(0, N_PP, (N_BOOT, N_PP))
    ci = lambda draws: np.percentile(draws, [2.5, 97.5]).tolist()

    group = {}
    for i, f in ((g.i2, 0), (g.i4, 1)):
        z = Z[:, i]
        group[f"{g.hz[i]:.2f}"] = dict(
            coherent_amp=float(coherent_amplitude(z)),
            ci=ci(coherent_amplitude(z[boot])),
            phase_coherence=float(np.abs(z.sum()) / np.abs(z).sum()))

    # Davidson's group-level test on the mean curve. The duplication affects observed and
    # shuffled curves equally there, so the group result holds.
    gf = ff["GFX_FourierNull"]
    r2g = g.fit(np.nanmean(Y, 0))[0][0]
    peak = int(r2g.argmax())
    group_test = dict(
        reproduces_his_stored_spectrum=float(np.nanmax(np.abs(
            r2g - np.asarray(gf.TargetOns_doubgc_binned_Acc_fitsRsq_Obs, float)))),
        max_r2=float(r2g[peak]), peak_freq_cps=float(g.hz[peak]),
        his_max_stat_threshold=float(np.asarray(gf.TargetOns_doubgc_binned_Acc_fitsRsq_ShuffCV_max, float)[-1]))
    group_test["significant"] = group_test["max_r2"] > group_test["his_max_stat_threshold"]

    summary = dict(
        n_participants=N_PP,
        n_trials_median=float(np.median(n_trials)), n_trials_range=[n_trials.min(), n_trials.max()],
        baseline_median=float(np.median(baseline)),
        floor_2cps_median=float(np.median(floor[:, 0])), floor_4cps_median=float(np.median(floor[:, 1])),
        floor_closed_form_median=float(np.median(np.sqrt(4 * baseline * (1 - baseline) / n_trials))),
        rms_amp_2cps=float(debiased_rms(amp[:, 0], floor[:, 0])),
        rms_amp_2cps_ci=ci(debiased_rms(amp[boot, 0], floor[boot, 0])),
        rms_amp_4cps=float(debiased_rms(amp[:, 1], floor[:, 1])),
        amp_max24_median=float(np.median(best)),
        amp_max24_p75=float(np.percentile(best, 75)),
        amp_max24_p90=float(np.percentile(best, 90)),
        amp_max24_davidson_responders=float(np.median(best[flagged])),
        n_significant=int(flagged.sum()),
        davidson_test_fpr_median=float(np.median(fpr)),
        group_level=group, group_level_test=group_test)
    rows = [dict(p=p + 1, n_trials=parts[p].n_independent, baseline=parts[p].baseline,
                 amp_2cps=amp[p, 0], amp_4cps=amp[p, 1], floor_2cps=floor[p, 0], floor_4cps=floor[p, 1],
                 floor_closed_form=float(np.sqrt(4 * baseline[p] * (1 - baseline[p]) / n_trials[p])),
                 amp_2cps_debiased=debiased[p, 0], amp_4cps_debiased=debiased[p, 1],
                 coef_2cps=[Z[p, g.i2].real, Z[p, g.i2].imag], coef_4cps=[Z[p, g.i4].real, Z[p, g.i4].imag],
                 peak_freq_cps=float(g.hz[R2[p].argmax()]), r2_max=float(R2[p].max()),
                 davidson_threshold=float(his_threshold[p]), davidson_significant=bool(flagged[p]),
                 davidson_test_fpr=float(fpr[p]))
            for p in range(N_PP)]

    s = summary
    print(f"\nindependent targets / participant: median {s['n_trials_median']:.0f} "
          f"[{n_trials.min():.0f}-{n_trials.max():.0f}], baseline accuracy {s['baseline_median']:.3f}")
    print(f"noise floor (bootstrap) {s['floor_2cps_median']:.4f} @2 cps, closed form {s['floor_closed_form_median']:.4f}")
    print(f"RMS amplitude @2.01 cps {s['rms_amp_2cps']:.4f} (95 % CI {s['rms_amp_2cps_ci'][0]:.4f}-"
          f"{s['rms_amp_2cps_ci'][1]:.4f}), @4.01 cps {s['rms_amp_4cps']:.4f}")
    print(f"debiased max(A2, A4): median {s['amp_max24_median']:.4f}, p75 {s['amp_max24_p75']:.4f}, "
          f"his 8 flagged {s['amp_max24_davidson_responders']:.4f} (selected)")
    print(f"his per-participant flag under the duplication: FPR {100 * s['davidson_test_fpr_median']:.0f} % (nominal 5 %)")
    for f, v in group.items():
        print(f"coherent amplitude @{f} cps {v['coherent_amp']:.4f} (95 % CI {v['ci'][0]:.4f}-{v['ci'][1]:.4f}), "
              f"phase coherence R = {v['phase_coherence']:.2f}")

    OUT.write_text(json.dumps({"summary": summary, "participants": rows}, indent=1,
                              default=lambda o: o.item() if hasattr(o, "item") else o), encoding="utf-8")
    print(f"\nsaved {OUT.name}")


def verify(nsim=400) -> None:
    """Check the three extraction choices that changed the result."""
    m, ff = load()
    g = Geometry(m, ff)
    fits = ff["PFX_FourierNull"]
    cell = m["GFX_TargPosData"]
    rng = np.random.default_rng(3)
    his = [np.asarray(f.TargetOns_doubgc_binned_Acc_fitsRsq_Obs, float) for f in fits]

    print("1. Which slice did Davidson fit?  max |our R2 spectrum - his stored one|")
    for s, name in enumerate(("LRL", "RLR (the slice used before)", "combined")):
        errs = [np.nanmax(np.abs(g.fit(np.asarray(cell[p, s].doubgc_binned_Acc, float))[0][0] - his[p]))
                for p in range(N_PP)]
        print(f"   slice {s} {name:28s} median {np.median(errs):.1e}, max {max(errs):.1e}")

    print("\n2. Is every target entered twice, half a stride apart?")
    rows = np.array([[np.nansum(np.asarray(cell[p, s].doubgc_counts, float)) for s in range(3)]
                     for p in range(N_PP)])
    print("   median rows: LRL {:.0f}, RLR {:.0f}, combined {:.0f} (= LRL + RLR)".format(*np.median(rows, 0)))

    def contrast(Y):        # half-stride vs quarter-stride bin differences
        Y = np.atleast_2d(Y)
        return np.nanmean(np.abs(Y - np.roll(Y, 20, -1)), -1) / np.nanmean(np.abs(Y - np.roll(Y, 10, -1)), -1)

    obs = np.median(contrast(np.array([participant(m, p).curve for p in range(N_PP)])))
    print(f"   |y_b - y_b+20| / |y_b - y_b+10|: observed {obs:.3f}")
    sample = [participant(m, p) for p in range(0, N_PP, 6)]
    for jitter in (0, 2, 3, 5, 10):
        sims = []
        for part in sample:
            lrl, rlr = paired_rows(rng, part.phase_prob, part.n_independent, 60, jitter)
            hit = (rng.random(lrl.shape) < part.baseline).astype(float)
            sims.append(np.nanmedian(contrast(g.bin_rows(np.concatenate([lrl, rlr], 1),
                                                         np.concatenate([hit, hit], 1)))))
        print(f"   simulated, paired, jitter +-{jitter:2d}: {np.mean(sims):.3f}")
    sims = []
    for part in sample:
        pos = rng.choice(PERCENTILES, size=(60, 2 * part.n_independent), p=part.phase_prob)
        sims.append(np.nanmedian(contrast(g.bin_rows(pos, (rng.random(pos.shape) < part.baseline).astype(float)))))
    print(f"   simulated, independent rows (his null's assumption): {np.mean(sims):.3f}")

    print("\n3. What the duplication does to his per-participant flag")
    fp, thr = [], []
    for p in range(0, N_PP, 3):
        R2n, _ = simulate(g, participant(m, p), nsim, rng)
        t = float(fits[p].TargetOns_doubgc_binned_Acc_fitsRsq_ShuffCV_max[-1])
        fp.append((R2n.max(1) > t).mean())
        thr.append((t, np.percentile(R2n.max(1), 95)))
    thr = np.array(thr)
    print(f"   his stored threshold median {np.median(thr[:, 0]):.3f}; paired-null 95th pct {np.median(thr[:, 1]):.3f}")
    print(f"   FPR of his flag {100 * np.median(fp):.0f} % (nominal 5 %), ~{N_PP * np.mean(fp):.0f} of 36 by chance "
          "vs 8 observed: his shuffle resamples stride positions independently and so discards the duplication.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true", help="fetch the OSF .mat files")
    ap.add_argument("--verify", action="store_true", help="evidence for the extraction choices")
    ap.add_argument("--nsim", type=int, help="bootstrap draws (default 1500, or 400 with --verify)")
    args = ap.parse_args()
    if args.download:
        download()
    elif args.verify:
        verify(args.nsim or 400)
    else:
        main(args.nsim or 1500)
