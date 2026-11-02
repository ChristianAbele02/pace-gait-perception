"""Exploratory: does head roll at stimulus onset bias the tilt judgement?

The Gabor plane follows the tracked head POSITION but not its rotation (``CameraHolder``
tracks position only), so the grating stays upright in the room. A head roll of r degrees
therefore rotates the grating on the retina by r, less whatever ocular counter-roll
compensates; counter-roll covers only a fraction of a dynamic head roll (Collewijn et al.,
1985). During walking the head rolls by about a degree once per stride
(``head_kinematics.py``), which is the size of the tilt thresholds here. Retinal-image
accounts of stride-phase effects in orientation tasks (Matthis et al., 2022) can usually
be set aside when the tilts judged are large; at a threshold near 1 degree they cannot.

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
* fast: replaces roll by its gait-band part (0.3-5 Hz, ``head_kinematics.BAND_HZ``),
  so only the roll that changes from step to step is used; no slow drift reaches it.

Null: the roll values are permuted within blocks after removing what the nuisance terms
(block, and the phase and time terms of the adjusted model) explain, then added back to
those fitted values (permutation of residuals; Kennedy, 1995). Each permutation refits
the model. The per-participant ``z = beta_roll / SD(null)`` enters the population tests of
``stance_foot.population_test``: the direction-agnostic ``S = sum z^2`` and the
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

Usage::

    python -m Analysis.roll_bias
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
    from . import head_kinematics as hk
    from . import phase_analysis as pa
    from . import plotstyle as S
    from . import stance_foot as sfoot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import head_kinematics as hk
    import phase_analysis as pa
    import plotstyle as S
    import stance_foot as sfoot

S.apply()

N_PERM = 1000
MIN_TRIALS = 200
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
                n_perm: int = N_PERM, rng: np.random.Generator | None = None) -> dict | None:
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


def _recording_trials(rec) -> dict | None:
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


def _pool_subject(recs) -> dict | None:
    """A participant's walking trials pooled over blocks, or None."""
    parts = [t for t in (_recording_trials(r) for r in recs) if t]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def analyse_subject(tr: dict, *, rng=None) -> dict | None:
    """The three models for one participant, plus the binned bias curve for the figure."""
    rng = rng or np.random.default_rng(0)
    main = roll_effect(tr["said_right"], tr["tilt"], tr["roll"], tr["block"], rng=rng)
    if main is None:
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
    return {"main": main, "adjusted": adj, "fast": fast, "curve": curve}


MODELS = {"main": "roll", "adjusted": "roll, adjusted for stride phase and time in block",
          "fast": "gait-band roll (0.3-5 Hz)"}


def leave_one_out(nulls: list[np.ndarray], deltas: list[float]) -> list[float]:
    """Direction-agnostic population p with each participant left out in turn."""
    out = []
    for i in range(len(deltas)):
        keep = [j for j in range(len(deltas)) if j != i]
        out.append(sfoot.population_test([nulls[j] for j in keep],
                                         [deltas[j] for j in keep])["agnostic_p"])
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _figure(rows: list[dict], pops: dict, out: Path) -> None:
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


def _json_safe(o):
    """JSON-serialisable copy of a result, without private keys."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def _thresholds() -> dict:
    """Walking 75 % threshold per participant from ``threshold_by_condition``."""
    f = bids_io.deriv_dir("conditions") / "threshold_by_condition.json"
    if not f.exists():
        return {}
    pp = json.loads(f.read_text(encoding="utf-8")).get("per_participant", {})
    return {s: (d.get("walking") or {}).get("thresh75_deg") for s, d in pp.items()}


def main() -> dict | None:
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
        tr = _pool_subject(subrecs)
        if tr is None:
            continue
        res = analyse_subject(tr, rng=np.random.default_rng(zlib.crc32(f"{sub}_roll".encode())))
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
        pops[key] = sfoot.population_test(nulls, deltas)
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
        "population": _json_safe(pops),
        "participants": _json_safe(rows),
    }
    (out_dir / "roll_bias_population.json").write_text(json.dumps(payload, indent=2),
                                                      encoding="utf-8")
    _figure(rows, pops, bids_io.fig_dir("roll_bias") / "roll_bias_population.png")
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


if __name__ == "__main__":
    main()
