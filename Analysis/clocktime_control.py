"""Clock-time control, after Davidson et al. (2024) Supplementary Figures 5 and 6.

Are performance oscillations locked to the stride cycle, or would any temporal reference
fit as well? Davidson aligned performance to clock time (fixed period in Hz) instead of
the stride (cycles per stride) and found a much weaker fit. The same control is run for
walking (Supp Fig 5) and standing (Supp Fig 6; no stride, so only clock time applies and
no oscillation is expected).

Davidson used discrete ~9 s walks and folded targets onto the 0-9 s trial clock. Here
walking is continuous on a treadmill, so a period-folding test is used: for each frequency
``f`` (Hz) the absolute onset times are folded modulo ``1/f``, binned (40) and fitted with
one cosine, as in ``phase_analysis`` but with a fixed clock period. The max R² over the Hz
grid is directly comparable to the stride (cps) max R². Expected: stride max R² well above
clock max R².

Significance uses the same trial-level permutation max-R² null as the stride test.

Usage::

    python -m Analysis.clocktime_control
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import bids_io
    from . import phase_analysis as pa
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import phase_analysis as pa
    import plotstyle as S

S.apply()

# Clock-time frequency grid (Hz), same count and step as the stride grid (0.1-10).
FREQS_HZ = np.round(np.arange(0.1, 10.0001, 0.1), 1)
N_BINS = pa.N_BINS
N_PERM = pa.N_PERM
RNG = np.random.default_rng(43)
# response_counts removed (see config.SECONDARY_MEASURES).
MEASURES = ["accuracy", "reaction_time"]


def _fold_pct(onset: np.ndarray, f_hz: float) -> np.ndarray:
    """Phase (% of period) of each absolute onset time, folded modulo ``1/f``."""
    return np.mod(onset * f_hz, 1.0) * 100.0


def _r2_at(onset, value, f_hz, how) -> float:
    """R² of a single cosine fitted to values folded at ``f_hz`` and binned (40 bins)."""
    phase = _fold_pct(onset, f_hz)
    centers, binned, _ = pa._bin(phase, value, N_BINS, how)
    ok = np.isfinite(binned)
    if ok.sum() < 6:
        return 0.0
    return pa.fit_one(centers[ok], binned[ok], 1.0)[0]   # 1 cycle per fold


_FOLD_CACHE: dict[tuple, list] = {}


def _design_for(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(c.size), np.cos(2 * np.pi * c / 100.0),
                         np.sin(2 * np.pi * c / 100.0)])
    # same singular-value cutoff as np.linalg.lstsq(..., rcond=None)
    return X, np.linalg.pinv(X, rcond=max(X.shape) * np.finfo(float).eps)


_CENTERS = 0.5 * (np.linspace(0, 100, N_BINS + 1)[:-1] + np.linspace(0, 100, N_BINS + 1)[1:])
_FULL_DESIGN = _design_for(_CENTERS)          # COUNT mode: all 40 bins, every frequency


def _fold_design(onset: np.ndarray, how_count_ok: bool) -> list:
    """Per-frequency fold, bin assignment and fitted design for these onsets.

    The permutation null shuffles values against fixed onsets, so bin assignment and
    occupancy per frequency do not change across permutations. Precomputing them and the
    design pseudo-inverse replaces per-frequency ``lstsq`` calls. Cached on the onsets.
    """
    key = (onset.tobytes(), how_count_ok)
    hit = _FOLD_CACHE.get(key)
    if hit is not None:
        return hit
    edges = np.linspace(0, 100, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    design = []
    for f in FREQS_HZ:
        idx = np.clip(np.digitize(_fold_pct(onset, float(f)), edges) - 1, 0, N_BINS - 1)
        cnt = np.bincount(idx, minlength=N_BINS)
        # COUNT mode sets empty bins to 0 and fits all 40; MEAN mode leaves them NaN and
        # fits occupied bins only. Merging the two would change the count spectrum.
        ok = np.ones(N_BINS, bool) if how_count_ok else (cnt > 0)
        if ok.sum() < 6:
            design.append(None)
            continue
        X, pinv = _design_for(centers[ok])
        design.append((idx, ok, np.maximum(cnt[ok], 1).astype(float), X, pinv))
    if len(_FOLD_CACHE) > 16:
        _FOLD_CACHE.clear()
    _FOLD_CACHE[key] = design
    return design


def _spectrum(onset, value, how) -> np.ndarray:
    """Clock-time R² spectrum, equal to ``[_r2_at(onset, value, f, how) ...]`` (checked in
    tests/test_pipeline.py) without per-frequency ``lstsq``.

    MEAN permutes values against fixed onsets, so the per-frequency design is cached on
    the onsets. COUNT resamples onsets each permutation but uses all 40 bins, so its
    design is the same full-grid matrix at every frequency, computed once at import.
    """
    onset = np.asarray(onset, float)
    out = np.zeros(FREQS_HZ.size)
    if how == "count":
        X, pinv = _FULL_DESIGN
        edges = np.linspace(0, 100, N_BINS + 1)
        for j, f in enumerate(FREQS_HZ):
            idx = np.clip(np.digitize(_fold_pct(onset, float(f)), edges) - 1, 0, N_BINS - 1)
            y = np.bincount(idx, minlength=N_BINS).astype(float)
            yc = y - y.mean()
            sst = float(yc @ yc)
            if sst <= 0:
                continue
            r = y - X @ (pinv @ y)
            out[j] = max(0.0, 1.0 - float(r @ r) / sst)
        return out
    w = np.asarray(value, float)
    for j, d in enumerate(_fold_design(onset, False)):
        if d is None:
            continue
        idx, ok, cnt, X, pinv = d
        y = np.bincount(idx, weights=w, minlength=N_BINS)[ok] / cnt
        yc = y - y.mean()
        sst = float(yc @ yc)
        if sst <= 0:
            continue
        r = y - X @ (pinv @ y)
        out[j] = max(0.0, 1.0 - float(r @ r) / sst)
    return out


def _r2_columns(Y: np.ndarray, X: np.ndarray, pinv: np.ndarray) -> np.ndarray:
    """Cosine-fit R² (as in ``_spectrum``) for every column of binned values ``Y``."""
    yc = Y - Y.mean(axis=0)
    sst = np.einsum("ij,ij->j", yc, yc)
    r = Y - X @ (pinv @ Y)
    sse = np.einsum("ij,ij->j", r, r)
    out = np.zeros(Y.shape[1])
    pos = sst > 0
    out[pos] = np.maximum(0.0, 1.0 - sse[pos] / sst[pos])
    return out


def _spectra_mean_batch(onset: np.ndarray, values: np.ndarray) -> np.ndarray:
    """MEAN-mode spectra for each column of ``values`` (n_trials x n_perm) with the same
    onsets; returns (n_perm x n_freq). One matrix product per frequency."""
    out = np.zeros((values.shape[1], FREQS_HZ.size))
    for j, d in enumerate(_fold_design(onset, False)):
        if d is None:
            continue
        idx, ok, cnt, X, pinv = d
        row = np.full(N_BINS, -1)
        row[np.flatnonzero(ok)] = np.arange(int(ok.sum()))
        assign = np.zeros((int(ok.sum()), onset.size))
        assign[row[idx], np.arange(onset.size)] = 1.0
        out[:, j] = _r2_columns((assign @ values) / cnt[:, None], X, pinv)
    return out


def _spectra_count_batch(onsets: np.ndarray) -> np.ndarray:
    """COUNT-mode spectra for each row of ``onsets`` (n_perm x n_trials); returns
    (n_perm x n_freq), each row equal to ``_spectrum(row, None, "count")``."""
    X, pinv = _FULL_DESIGN
    edges = np.linspace(0, 100, N_BINS + 1)
    n_perm = onsets.shape[0]
    offset = (np.arange(n_perm) * N_BINS)[:, None]
    out = np.zeros((n_perm, FREQS_HZ.size))
    for j, f in enumerate(FREQS_HZ):
        idx = np.clip(np.digitize(_fold_pct(onsets, float(f)), edges) - 1, 0, N_BINS - 1)
        counts = np.bincount((idx + offset).ravel(), minlength=n_perm * N_BINS)
        out[:, j] = _r2_columns(counts.reshape(n_perm, N_BINS).T.astype(float), X, pinv)
    return out


def analyse_clock(onset, value, how, nperm=N_PERM) -> dict | None:
    """Clock-time (Hz) folding spectrum with a trial-permutation max-R² null."""
    onset = np.asarray(onset, float)
    value_arr = None if value is None else np.asarray(value, float)
    if value_arr is None:
        onset = onset[np.isfinite(onset)]
    else:
        m = np.isfinite(onset) & np.isfinite(value_arr)
        onset, value_arr = onset[m], value_arr[m]
    if onset.size < 6:
        return None
    obs = _spectrum(onset, value_arr, how)
    dur = float(onset.max() - onset.min()) or 1.0
    # Permutations are drawn one by one (same random sequence), then computed in batch.
    if how == "count":
        # count null: uniform random onsets
        draws = np.stack([RNG.uniform(onset.min(), onset.max(), onset.size)
                          for _ in range(nperm)])
        null_max = _spectra_count_batch(draws).max(axis=1)
    else:
        perms = np.stack([RNG.permutation(value_arr) for _ in range(nperm)], axis=1)
        null_max = _spectra_mean_batch(onset, perms).max(axis=1)
    ci95 = float(np.percentile(null_max, 95))   # plotted CI line, as in Davidson
    bi = int(np.argmax(obs))
    # Permutation p-value with +1 correction: p = (1 + #{null_max >= obs_max}) / (1 + nperm),
    # valid at finite nperm (unlike "obs > 95th percentile").
    p_perm = (1 + int(np.sum(null_max >= obs[bi]))) / (1 + nperm)
    return {
        "freqs_hz": FREQS_HZ, "r2": obs, "ci95": ci95,
        "best_freq_hz": float(FREQS_HZ[bi]), "best_r2": float(obs[bi]),
        "p_perm": float(p_perm), "significant": bool(p_perm <= 0.05),
        "duration_s": dur,
    }


# ===========================================================================
# Per-recording analysis
# ===========================================================================
def _trial_level(rec):
    """(onset, value) per measure for one recording, with the trial definitions and RT
    filter of ``phase_analysis`` but indexed by absolute onset time."""
    ev = bids_io.load_events(rec)
    ev = ev[(ev["is_training"] == 0) & ev["onset"].notna()].copy()
    if ev.empty:
        return None
    ev["onset"] = ev["onset"].astype(float)
    # accuracy over answered trials, as in phase_analysis (a timeout is NaN)
    ev["acc"] = np.where(ev["response"].notna(),
                         np.where(ev["response_correct"] == 1, 1.0, 0.0), np.nan)
    ev["rt"] = ev["response_time"].astype(float)
    # phase_analysis' RT filter uses ev["phase_target"]; here it holds the onset time.
    ev["phase_target"] = ev["onset"]
    ans = ev[ev["acc"].notna()]
    out = {"accuracy": (ans["onset"].to_numpy(float), ans["acc"].to_numpy(float))}
    # RT: correct trials, anticipations/outliers removed, log10; _clean_rt returns
    # (onset[keep], log10_rt[keep], info).
    rt = pa._clean_rt(ev)
    if rt:
        out["reaction_time"] = (rt[0], rt[1])
    return out


def _stride_reference(rec) -> dict:
    """Stride (cps) max R² and CI per measure from the phase_analysis output ({} if none)."""
    js = bids_io.deriv_dir("phase", rec.sub, rec.ses) / f"{rec.label()}_phase.json"
    if not js.exists():
        return {}
    rep = json.loads(js.read_text())
    return {k: {"best_r2": v["best_r2"], "ci95": v["ci95"],
                "best_freq_cps": v["best_freq_cps"], "significant": v["significant"],
                "significant_grid": v.get("significant_grid", False)}
            for k, v in rep.get("measures", {}).items()}


def analyse(rec) -> dict | None:
    """Clock-time control for one recording: gait-locked or merely periodic?

    Folds onsets modulo each period on a Hz grid and compares the best clock-time fit
    with the stride fit (max-statistic permutation test). Also run for standing, which
    has no stride phase; clock time is the only axis both conditions share.
    """
    if rec.condition not in ("walking", "standing"):
        return None
    tr = _trial_level(rec)
    if not tr:
        return None
    clock = {}
    for k in MEASURES:
        if k not in tr:
            continue
        onset, value = tr[k]
        m = analyse_clock(onset, value, "mean")
        if m:
            clock[k] = m
    if not clock:
        return None
    stride = _stride_reference(rec) if rec.condition == "walking" else {}

    out_dir = bids_io.deriv_dir("clocktime", rec.sub, rec.ses)
    report = {"recording": rec.label(), "condition": rec.condition,
              "clock_hz": {k: {kk: clock[k][kk] for kk in
                               ("best_freq_hz", "best_r2", "ci95", "p_perm", "significant")}
                           for k in clock},
              "stride_cps": stride}
    (out_dir / f"{rec.label()}_clocktime.json").write_text(json.dumps(report, indent=2),
                                                           encoding="utf-8")
    _figure(rec, clock, stride,
            bids_io.fig_dir("clocktime", rec.sub, rec.ses) / f"{rec.label()}_clocktime.png")
    return report


# ===========================================================================
# Figure (Supp Fig 5 / 6)
# ===========================================================================
def _figure(rec, clock, stride, out: Path):
    fig, ax = plt.subplots(1, len(MEASURES), figsize=(5 * len(MEASURES), 4.4))
    walking = rec.condition == "walking"
    fig.suptitle(f"Clock-time control — {S.nice_label(rec.label())}  "
                 f"({'walking: stride vs clock' if walking else 'standing: clock only'})",
                 fontsize=13, fontweight="semibold", x=0.012, ha="left")
    for a, tag, meas in zip(ax, "abc", MEASURES):
        m = clock.get(meas)
        if not m:
            S.panel(a, tag, S.MEASURE_LABEL[meas].title()); continue
        col = S.MEASURE[meas]
        a.plot(m["freqs_hz"], m["r2"], color=col, lw=2,
               label=f"clock (Hz): max R²={m['best_r2']:.2f}"
                     + (" *" if m["significant"] else ""))
        a.axhline(m["ci95"], color=col, ls=":", lw=1, label="clock 95% CI")
        sref = stride.get(meas)
        if sref:
            a.axhline(sref["best_r2"], color=S.INK, ls="--", lw=1.4,
                      label=f"stride (cps): max R²={sref['best_r2']:.2f}"
                            + (" *" if sref.get("significant_grid") else ""))
        a.set_xlabel("frequency (Hz, clock period)"); a.set_ylabel("R²")
        a.set_xlim(0, 10)
        a.set_ylim(0, a.get_ylim()[1] * 1.3)      # room for the legend above the CI
        a.legend(loc="upper right", ncol=2, fontsize=7.5)
        S.panel(a, tag, S.MEASURE_LABEL[meas].title())
    note = ("Performance folded onto a fixed clock period (Hz); dashed = the best fit "
            "folded onto the stride (cps). A stride-locked oscillation fits the stride "
            "better than any clock period."
            if walking else
            "Standing has no stride, so performance is folded onto clock periods only; "
            "dotted = 95% permutation CI.")
    S.caption(fig, note, wrap=110)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    S.save(fig, out)


def _print(r: dict):
    print(f"\n=== Clock-time control: {r['recording']} ({r['condition']}) ===")
    for k, m in r["clock_hz"].items():
        sig = "SIGNIFICANT" if m["significant"] else "n.s."
        sref = r["stride_cps"].get(k)
        cmp = (f" | stride cps R²={sref['best_r2']:.2f}@{sref['best_freq_cps']:.1f}cps"
               if sref else "")
        print(f"  {k:16s}: clock best {m['best_freq_hz']:.1f} Hz R²={m['best_r2']:.2f} "
              f"(CI {m['ci95']:.2f}) -> {sig}{cmp}")


# ===========================================================================
# Standing vs walking: interaction test
# ===========================================================================
N_SUBSAMPLE = 20        # trial-count-matched draws per participant


def condition_contrast(recs, measure="accuracy", nperm=200, rng=None) -> dict:
    """Does the clock-locked oscillation differ between walking and standing?

    "Significant while walking, not while standing" is not a valid comparison: a
    difference in significance is not a significant difference, and standing has fewer
    trials (~360 vs ~1440; 1 vs 4 blocks per visit), so it has less power.

    This tests the within-participant difference in clock-time max R² on trial-matched
    data: both conditions are subsampled to the smaller N, averaged over ``N_SUBSAMPLE``
    draws. The paired difference is tested against a null that permutes condition labels
    within participants. Clock time is used because standing has no stride phase.
    """
    rng = rng or np.random.default_rng(0)
    by_sub: dict[str, dict[str, list]] = {}
    for rec in recs:
        if rec.condition not in ("walking", "standing"):
            continue
        tl = _trial_level(rec)
        if not tl or measure not in tl:
            continue
        by_sub.setdefault(rec.sub, {}).setdefault(rec.condition, []).append(tl[measure])

    per_sub = {}
    for sub, conds in by_sub.items():
        if not ("walking" in conds and "standing" in conds):
            continue
        pooled = {}
        for c, chunks in conds.items():
            on = np.concatenate([k[0] for k in chunks])
            va = (None if chunks[0][1] is None
                  else np.concatenate([k[1] for k in chunks]))
            pooled[c] = (on, va)
        n_match = min(pooled["walking"][0].size, pooled["standing"][0].size)
        if n_match < 40:
            continue
        best = {}
        for c, (on, va) in pooled.items():
            draws = []
            for _ in range(N_SUBSAMPLE):
                idx = rng.choice(on.size, n_match, replace=False)
                sub_v = None if va is None else va[idx]
                draws.append(_spectrum(on[idx], sub_v,
                                       "count" if va is None else "mean").max())
            best[c] = float(np.mean(draws))
        per_sub[sub] = {"walking_max_r2": best["walking"],
                        "standing_max_r2": best["standing"],
                        "difference": best["walking"] - best["standing"],
                        "n_matched_trials": int(n_match)}

    out = {"measure": measure, "n_participants": len(per_sub),
           "n_subsample_draws": N_SUBSAMPLE, "per_participant": per_sub}
    if len(per_sub) < 2:
        out["status"] = ("not estimable — needs >=2 participants with BOTH conditions; "
                         "do NOT substitute per-condition significance calls")
        return out
    d = np.array([v["difference"] for v in per_sub.values()])
    # permutation null: random sign flip per participant
    obs = float(np.mean(d))
    null = np.array([np.mean(d * rng.choice([-1.0, 1.0], d.size)) for _ in range(nperm)])
    p = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (1 + nperm)
    out.update({"status": "ok", "mean_difference": round(obs, 4),
                "p_perm": round(float(p), 4), "significant": bool(p < 0.05),
                "direction": ("walking > standing" if obs > 0 else "standing >= walking")})
    return out


def main():
    """Run the clock-time control per recording and the standing-vs-walking contrast."""
    recs = bids_io.find_recordings()
    if not recs:
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return
    any_done = False
    for rec in recs:
        r = analyse(rec)
        if r:
            any_done = True
            _print(r)
    if any_done:
        out = bids_io.deriv_dir("clocktime")
        print(f"\nClock-time control outputs -> {out}")
        print("NOTE: the control's claim (stride cps R² >> clock Hz R²) only becomes "
              "interpretable with Davidson-scale data; the machinery is faithful.")

    bids_io.gate_report("clocktime", n_analysed=len({r.sub for r in recs}),
                        prune=("clocktime",))
    xc = condition_contrast(recs)
    print("\n=== Standing vs walking: interaction test (trial-count matched) ===")
    if xc.get("status") != "ok":
        print(f"  {xc.get('status')}  [{xc['n_participants']} participant(s) with both]")
    else:
        print(f"  {xc['n_participants']} participants, {xc['n_subsample_draws']} matched "
              f"subsamples each")
        print(f"  mean paired difference in clock max-R² (walking - standing): "
              f"{xc['mean_difference']:+.4f}, permutation p={xc['p_perm']:.4f} -> "
              f"{'SIGNIFICANT' if xc['significant'] else 'n.s.'} ({xc['direction']})")
    print("  Report THIS, never 'significant in walking but not standing' — the two "
          "conditions do not have equal trial counts (3 vs 2 blocks per visit), so a "
          "standing null is partly a power artefact.")
    (bids_io.deriv_dir("clocktime") / "condition_contrast.json").write_text(
        json.dumps(xc, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
