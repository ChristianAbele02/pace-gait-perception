"""QUEST+ calibration: reliability check, convergence simulation and offline refit.

* Calibration check (pipeline stage ``calibration``): convergence figure and reliability
  verdict for every calibration.
* QUEST+ convergence simulation (on demand): a re-implementation of the dual-tracker
  QUEST+ used to set and validate the runtime stop and reliability rules.
* Offline slope and lapse refit (stage ``calib_refit``): runs after the gate, which decides
  which calibrations are reliable enough to enter it.

Usage::

    python -m Analysis.calibration                  # calibration check + figures
    python -m Analysis.calibration refit [--bids]   # slope/lapse refit
    python -m Analysis.calibration convergence      # QUEST+ convergence simulation


Calibration check and figures
=============================

QUEST+ calibration convergence figures and reliability reports in derivatives/.

Reads the gzipped raw calibration log from ``sourcedata``, which keeps the two ``#``
metadata lines (final threshold/easy/hard tilts, QUEST+ parameters) that the BIDS
``_beh.tsv`` drops, and plots with the Agg backend.

Also writes ``*_calibration_qc.json``. Blocking ``flags`` (verdict UNRELIABLE): inattentive
(catch lapses), wide posterior, high-tilt lapses (only if significantly above the model
lapse rate), ceiling rail (threshold at grid max, no real threshold) and floor rail
(threshold near grid min or difficulty levels clamped: task too easy or grid mis-scaled).
Non-blocking ``warnings``: A/B tracker divergence, only when the posteriors are
statistically separated given their widths, since two short posteriors disagree ~30 % of
the time by chance.


QUEST+ convergence simulation
=============================

Simulation of calibration convergence and reliability over a broad participant range.

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

  python -m Analysis.calibration convergence


Offline slope and lapse refit
=============================

Offline refit of the psychometric slope and lapse from calibration data (b8).

QUEST+ (``StaircaseManager``) estimates only the threshold, with Weibull slope and lapse
fixed at assumed values. With several participants' calibrations these can be checked by
fitting threshold, slope and lapse by maximum likelihood. The refined values can be fed
back into QUEST+ (with priors re-centred on the observed thresholds), making calibrations
faster and less prone to lapse-driven runaway.

Model (2-AFC Weibull on tilt magnitude, guess rate fixed at 0.5)::

    psi(x) = 0.5 + (0.5 - lambda) * (1 - exp(-(x/alpha)^beta))
    alpha = threshold (deg), beta = slope, lambda = lapse

``refit_from_bids()`` (``--bids``) fits the real calibrations: one shared slope and lapse per
condition, a threshold per calibration (``conditions`` uses it). By default
``main`` simulates that refit at the rig's calibration design: calibrations drawn like the
real ones (thresholds, QUEST+ trial counts, catch trials) run through the same QUEST+
placement, repeated for 12 and 24 calibrations per condition, with the lapse free, fixed at
the QUEST+ value, or free with the catch trials added. The repetitions are cached
(``stage_cache``) and rerun only when this code or the real calibrations change.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

try:
    from . import bids_io
    from . import config as C
    from . import plotstyle as S
    from . import stage_cache
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import plotstyle as S
    import stage_cache

S.apply()


# ###########################################################################
# Calibration check and figures
# ###########################################################################
# Reliability thresholds.
MAX_FINAL_SD_LOG = 0.12      # widest acceptable posterior SD (log10)
LAPSE_TILT_FACTOR = 2.0      # error above factor*threshold = high-tilt lapse
# High-tilt lapses flag only if there are more than the calibration's own model predicts
# (guess/slope/lapse from the CSV header; one-sided Poisson-binomial test at this alpha).
# A fixed count ignores the number of high-tilt trials: 2 errors in 25 occur 36 % of the
# time at a 5 % lapse rate. A real runaway is caught by the ceiling rail.
HIGH_TILT_LAPSE_ALPHA = 0.05
MAX_CATCH_LAPSE_RATE = 0.10  # max error rate on catch trials
MIN_CATCH_TO_JUDGE = 4       # min answered catch trials to judge
RAIL_THRESHOLD_DEG = 12.5    # threshold at/above this = grid ceiling

# Grid bounds (StaircaseManager minAngle/maxAngle) for the floor rail and clamp check.
# Taken from the CSV metadata (min_angle_deg/max_angle_deg) if present.
GRID_MIN_DEG = 0.3
GRID_MAX_DEG = 25.0
# A reliable threshold must be at least this factor above the grid floor. Below it the
# posterior hits the grid edge (biased estimate) and the HARD level (~0.76x threshold) is
# clamped at minAngle, so levels miss their target accuracies. 2.0x keeps HARD ~1.5x above.
FLOOR_RAIL_FACTOR = 2.0
CLAMP_TOL = 1.02             # within 2 % of a grid bound = clamped
# A/B divergence is a non-blocking warning: short per-tracker posteriors disagree ~30 % of
# the time even for good observers (README, the convergence simulation). Warn only if
# |logA-logB| exceeds this many combined SDs, suggesting an inconsistent or side-biased
# responder.
DIVERGENCE_SEP_Z = 2.0




def _calibration_archives(bids_dir: Path) -> list[Path]:
    sd = bids_dir / "sourcedata"
    return sorted(list(sd.glob("sub-*/ses-*/beh/*_beh.csv.gz"))
                  + list(sd.glob("sub-*/ses-*/beh/*_beh.csv.zip")))


def _read_archive_text(archive: Path) -> str:
    if archive.name.endswith(".csv.gz"):
        with gzip.open(archive, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with zipfile.ZipFile(archive) as z:
        inner = next(n for n in z.namelist() if n.endswith(".csv"))
        return z.read(inner).decode("utf-8", "ignore")


def _parse_meta(text: str) -> dict:
    """Parse the '#' key=value metadata lines into a dict of floats."""
    meta = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        for part in line.lstrip("# ").split(";"):
            k, _, v = part.partition("=")
            try:
                meta[k.strip()] = float(v)
            except ValueError:
                pass
    return meta


def _load_calibration(archive: Path) -> tuple[dict, pd.DataFrame]:
    text = _read_archive_text(archive)
    meta = _parse_meta(text)
    from io import StringIO
    df = pd.read_csv(StringIO(text), sep=";", comment="#")
    return meta, df


def _high_tilt_lapse_test(tilts: np.ndarray, n_wrong: int, thr: float, meta: dict) -> dict:
    """Test whether ``n_wrong`` high-tilt errors exceed what an attentive observer makes.

    Error probability per trial is 1 - psi(tilt) under the header's QUEST+ model (2-AFC
    Weibull; design defaults as fallback). The error count is Poisson-binomial; returns
    its expectation and upper-tail p-value.
    """
    g = float(meta.get("guess", 0.5))
    beta = float(meta.get("slope_beta", 2.5))
    lam = float(meta.get("lapse", 0.05))
    x = np.asarray(tilts, float)
    p_err = 1.0 - (g + (1.0 - g - lam) * (1.0 - np.exp(-(x / thr) ** beta)))
    dist = np.array([1.0])                        # P(k errors), built up trial by trial
    for p in p_err:
        dist = np.convolve(dist, [1.0 - p, p])
    p_value = float(dist[n_wrong:].sum()) if n_wrong <= len(p_err) else 0.0
    return {"n_high_tilt_trials": int(len(x)), "expected": round(float(p_err.sum()), 2),
            "p_value": round(min(1.0, p_value), 4)}


def reliability_report(df: pd.DataFrame, meta: dict) -> dict:
    """Assess whether the calibration converged reliably.

    The two trackers start from different priors and should converge to the same
    threshold. Also checks posterior width, high-tilt lapses, catch trials and grid rails.
    """
    # Catch trials (attentiveness) and warm-up trials (initial practice) are not QUEST+
    # trials and are kept out of the tracker and threshold statistics.
    tracker_str = df["tracker"].astype(str)
    is_catch = tracker_str == "catch"
    is_aux = is_catch | (tracker_str == "warmup")
    catch = df[is_catch]
    qp = df[~is_aux]
    scored = qp[qp["correct"] != -1]
    out = {"n_trials": int(len(scored)), "n_timeouts": int((qp["correct"] == -1).sum()),
           "trackers": {}}
    for trk in sorted(scored["tracker"].dropna().unique()):
        t = scored[scored["tracker"] == trk]
        out["trackers"][str(trk)] = {
            "final_threshold_deg": round(float(t["post_mean_threshold_deg"].iloc[-1]), 3),
            "final_sd_log": round(float(t["post_sd_log"].iloc[-1]), 3),
            "n_trials": int(len(t)),
            "accuracy": round(float(t["correct"].mean()), 3),
        }
    thr = [v["final_threshold_deg"] for v in out["trackers"].values()]
    if len(thr) >= 2 and min(thr) > 0:
        out["divergence_ratio"] = round(max(thr) / min(thr), 2)
        out["divergence_log10"] = round(abs(np.log10(max(thr)) - np.log10(min(thr))), 3)
    out["max_final_sd_log"] = round(max((v["final_sd_log"] for v in out["trackers"].values()),
                                        default=float("nan")), 3)
    thr_ref = float(meta.get("threshold_deg", np.median(thr) if thr else float("nan")))
    high = scored[scored["presented_tilt_deg"] > LAPSE_TILT_FACTOR * thr_ref]
    out["high_tilt_lapses"] = int((high["correct"] == 0).sum())
    if np.isfinite(thr_ref) and thr_ref > 0:
        out["high_tilt_lapse_test"] = _high_tilt_lapse_test(
            high["presented_tilt_deg"].to_numpy(float), out["high_tilt_lapses"], thr_ref, meta)

    # Runtime convergence metadata from the CSV header (missing in older files).
    if "pooled_deg" in meta:
        out["pooled_threshold_deg"] = round(float(meta["pooled_deg"]), 3)
    if "pooled_sd_log" in meta:
        out["pooled_sd_log"] = round(float(meta["pooled_sd_log"]), 3)
    if "n_scored_trials" in meta:
        out["n_scored_trials"] = int(meta["n_scored_trials"])
    if "converged" in meta:
        out["runtime_converged"] = bool(round(meta["converged"]))
    if "agree" in meta:
        out["runtime_trackers_agree"] = bool(round(meta["agree"]))

    # Catch-trial attentiveness from the runtime metadata, else counted from the rows.
    if "n_catch" in meta:
        n_catch = int(meta["n_catch"])
        catch_err = int(meta.get("catch_errors", 0))
        catch_rate = float(meta.get("catch_lapse_rate", (catch_err / n_catch) if n_catch else 0.0))
        catch_timeouts = int(meta.get("catch_timeouts", 0))
    else:
        cs = catch[catch["correct"] != -1]
        n_catch = int(len(cs))
        catch_err = int((cs["correct"] == 0).sum())
        catch_rate = (catch_err / n_catch) if n_catch else 0.0
        catch_timeouts = int((catch["correct"] == -1).sum())
    if n_catch > 0 or catch_timeouts > 0:
        out["catch"] = {"n_answered": n_catch, "errors": catch_err,
                        "lapse_rate": round(catch_rate, 3), "timeouts": catch_timeouts}

    # Warm-up summary, for information only.
    if "n_warmup" in meta and int(meta["n_warmup"]) > 0:
        out["warmup"] = {"n_answered": int(meta["n_warmup"]),
                         "correct": int(meta.get("warmup_correct", 0)),
                         "timeouts": int(meta.get("warmup_timeouts", 0))}

    # Grid bounds from the CSV metadata, else StaircaseManager defaults.
    grid_min = float(meta.get("min_angle_deg", GRID_MIN_DEG))
    grid_max = float(meta.get("max_angle_deg", GRID_MAX_DEG))

    flags = []       # make the verdict UNRELIABLE
    warnings = []    # reported, do not change the verdict
    if n_catch >= MIN_CATCH_TO_JUDGE and catch_err >= 2 and catch_rate > MAX_CATCH_LAPSE_RATE:
        flags.append(f"inattentive: {catch_err} lapses ({catch_rate:.0%}) over {n_catch} easy "
                     f"anchors (> {MAX_CATCH_LAPSE_RATE:.0%})")
    # Ceiling rail: threshold at the grid max, no real threshold found.
    est_for_rail = out.get("pooled_threshold_deg",
                           float(np.median(thr)) if thr else float("nan"))
    if est_for_rail >= RAIL_THRESHOLD_DEG:
        flags.append(f"threshold {est_for_rail:.1f} deg at grid ceiling "
                     f"(>= {RAIL_THRESHOLD_DEG}) - no real threshold (runaway/random)")
    # Floor rail: threshold within FLOOR_RAIL_FACTOR of the grid minimum is edge-biased and
    # levels clamp (task too easy or grid mis-scaled).
    if np.isfinite(est_for_rail) and est_for_rail <= FLOOR_RAIL_FACTOR * grid_min:
        flags.append(f"threshold {est_for_rail:.2f} deg near grid floor "
                     f"(<= {FLOOR_RAIL_FACTOR:.0f}x minAngle {grid_min:.2f}) - task too easy / "
                     f"grid mis-scaled; difficulty levels unreliable")
    # Clamp check: a level at a grid bound is not at its target accuracy.
    clamped = [name for name, key in (("easy", "easy_deg"), ("threshold", "threshold_deg"),
                                      ("hard", "hard_deg"))
               if key in meta and (meta[key] <= grid_min * CLAMP_TOL
                                   or meta[key] >= grid_max / CLAMP_TOL)]
    if clamped:
        flags.append(f"difficulty level(s) clamped to a grid bound: {', '.join(clamped)} "
                     f"(grid [{grid_min:.2f}, {grid_max:.1f}] deg) - level not at target accuracy")
    # Precision: use the pooled posterior SD, as in the runtime stop rule (whose target is
    # below this threshold, so converged runs are never flagged). Older files without it use
    # the larger per-tracker SD.
    sd_for_flag = out.get("pooled_sd_log", out["max_final_sd_log"])
    sd_src = "pooled" if "pooled_sd_log" in out else "max per-tracker"
    if sd_for_flag > MAX_FINAL_SD_LOG:
        flags.append(f"wide posterior ({sd_src} SD_log {sd_for_flag} > {MAX_FINAL_SD_LOG})")
    ht = out.get("high_tilt_lapse_test")
    if ht and out["high_tilt_lapses"] and ht["p_value"] < HIGH_TILT_LAPSE_ALPHA:
        flags.append(f"{out['high_tilt_lapses']} high-tilt lapses over {ht['n_high_tilt_trials']} "
                     f"trials above {LAPSE_TILT_FACTOR:.0f}x threshold (expected {ht['expected']}, "
                     f"p={ht['p_value']}) - more than the model lapse rate (runaway risk)")

    # A/B divergence warning only if |logA-logB| > DIVERGENCE_SEP_Z combined SDs; a large
    # ratio with overlapping posteriors is sampling noise.
    tvals = out["trackers"]
    if len(tvals) >= 2:
        (a, b) = list(tvals.values())[:2]
        la, lb = np.log10(max(a["final_threshold_deg"], 1e-6)), np.log10(max(b["final_threshold_deg"], 1e-6))
        sd_comb = float(np.hypot(a["final_sd_log"], b["final_sd_log"])) or 1e-9
        z = abs(la - lb) / sd_comb
        out["divergence_sep_z"] = round(z, 2)
        if z > DIVERGENCE_SEP_Z:
            warnings.append(f"trackers statistically separated (z={z:.1f} > {DIVERGENCE_SEP_Z}); "
                            f"A={a['final_threshold_deg']} vs B={b['final_threshold_deg']} deg - "
                            f"possible inconsistent / side-biased responder")

    out["flags"] = flags
    out["warnings"] = warnings
    out["verdict"] = "RELIABLE" if not flags else "UNRELIABLE - recommend recalibration"
    return out


def make_calibration_figure(archive: Path) -> Path | None:
    """Plot one calibration run and write its reliability QC; returns the figure path.

    Shows both QUEST+ trackers, the presented tilts and the posterior threshold. The
    ``*_calibration_qc.json`` holds the verdict used by the exclusion gate.
    """
    stem = (archive.name[:-len(".csv.gz")] if archive.name.endswith(".csv.gz")
            else archive.name[:-len(".csv.zip")])          # sub-01_ses-01_task-ILL_beh
    sub, ses = stem.split("_")[0], stem.split("_")[1]
    # Condition from acq-<standing|walking>, to tell the two calibrations apart.
    _m = re.search(r"_acq-(standing|walking)", archive.name)
    cond = _m.group(1) if _m else None

    meta, df = _load_calibration(archive)
    if df.empty:
        print(f"  {archive.name}: no trials -> skipped")
        return None

    rel = reliability_report(df, meta)
    out_dir = bids_io.deriv_dir("calibration", sub, ses)
    (out_dir / f"{stem}_calibration_qc.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 6))
    for trk in sorted(df["tracker"].dropna().unique()):
        if str(trk) in ("catch", "warmup"):
            continue                          # catch and warm-up trials are drawn below
        tr = df[df["tracker"] == trk]
        x = tr["global_trial"].to_numpy()
        tilt = tr["presented_tilt_deg"].to_numpy()
        corr = tr["correct"].to_numpy()
        est = tr["post_mean_threshold_deg"].to_numpy()
        sd = tr["post_sd_log"].to_numpy()
        c = S.TRACKER.get(str(trk), S.GREY)

        ax.plot(x, est, "-", color=c, lw=2, label=f"Tracker {trk} — estimate")
        ax.fill_between(x, est * (10.0 ** -sd), est * (10.0 ** sd), color=c, alpha=0.12)
        ax.scatter(x[corr == 1], tilt[corr == 1], marker="o", facecolors=c, edgecolors=c,
                   s=40, zorder=3, label=f"Tracker {trk} — correct")
        ax.scatter(x[corr == 0], tilt[corr == 0], marker="o", facecolors="white",
                   edgecolors=c, s=40, linewidths=1.4, zorder=3, label=f"Tracker {trk} — wrong")
        if (corr == -1).any():  # no response (not given to QUEST+)
            ax.scatter(x[corr == -1], tilt[corr == -1], marker="x", color=S.GREY, s=44,
                       linewidths=1.6, zorder=3, label=f"Tracker {trk} — no response")

    # Catch trials: grey diamond = passed, X = lapse.
    catch = df[df["tracker"].astype(str) == "catch"]
    cscored = catch[catch["correct"] != -1] if not catch.empty else catch
    if not cscored.empty:
        xc = cscored["global_trial"].to_numpy()
        tc = cscored["presented_tilt_deg"].to_numpy()
        cc = cscored["correct"].to_numpy()
        ax.scatter(xc[cc == 1], tc[cc == 1], marker="D", facecolors=S.MIST,
                   edgecolors=S.GREY, s=32, zorder=4, label="catch — passed")
        if (cc == 0).any():
            ax.scatter(xc[cc == 0], tc[cc == 0], marker="X", color=S.RED, s=78,
                       zorder=5, label="catch — LAPSE")

    # Warm-up trials: grey squares (correct) / open (wrong); not part of the estimate.
    warm = df[df["tracker"].astype(str) == "warmup"]
    wscored = warm[warm["correct"] != -1] if not warm.empty else warm
    if not wscored.empty:
        xw = wscored["global_trial"].to_numpy()
        tw = wscored["presented_tilt_deg"].to_numpy()
        cw = wscored["correct"].to_numpy()
        ax.scatter(xw[cw == 1], tw[cw == 1], marker="s", facecolors=S.MIST,
                   edgecolors=S.GREY, s=28, zorder=4, label="warm-up — practice")
        if (cw == 0).any():
            ax.scatter(xw[cw == 0], tw[cw == 0], marker="s", facecolors="white",
                       edgecolors=S.GREY, s=28, linewidths=1.2, zorder=4,
                       label="warm-up — wrong")

    for key, lbl, ls in [("threshold_deg", "threshold", "-"),
                         ("easy_deg", "easy", "--"), ("hard_deg", "hard", ":")]:
        if key in meta:
            ax.axhline(meta[key], color=S.GREY, ls=ls, lw=1.2, label=f"{lbl} = {meta[key]:.2f}°")

    ax.set_yscale("log")
    ax.set_xlabel("Trial (interleaved)")
    ax.set_ylabel("Tilt magnitude (°, log scale)")
    ax.grid(True, which="both")
    verdict = "RELIABLE" if not rel["flags"] else "UNRELIABLE — recommend recalibration"
    if "divergence_ratio" in rel:
        verdict += f"   (A/B {rel['divergence_ratio']}×)"
    ax.set_title(f"QUEST+ calibration — {sub} · {ses}"
                 + (f" · {cond}" if cond else ""), color=S.INK)
    # Verdict in the upper-right corner, clear of the title.
    ax.text(0.985, 0.97, verdict, transform=ax.transAxes, fontsize=9.5, va="top",
            ha="right", fontweight="semibold", color=(S.RED if rel["flags"] else S.GREEN),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
    S.legend_outside(ax, ncol=1)
    # QUEST+ parameters as a footnote.
    if all(k in meta for k in ("guess", "slope_beta", "lapse")):
        S.caption(fig, f"QUEST+ Weibull: γ = {meta['guess']:.2f},  "
                       f"β = {meta['slope_beta']:.2f},  λ = {meta['lapse']:.2f}")
    fig.tight_layout()

    dest = bids_io.fig_dir("calibration", sub, ses) / f"{stem}_calibration.png"
    S.save(fig, dest)
    print(f"  {sub}/{ses}{('/' + cond) if cond else ''}: {rel['verdict']}")
    trk = " ; ".join(f"{k}={v['final_threshold_deg']}deg(+/-{v['final_sd_log']})"
                     for k, v in rel["trackers"].items())
    print(f"           trackers: {trk}")
    for f in rel["flags"]:
        print(f"           [FLAG] {f}")
    for w in rel.get("warnings", []):
        print(f"           [warn] {w}")
    print(f"           figure -> {dest}")
    return dest


def check_main() -> None:
    """Plot every calibration and write its reliability QC."""
    archives = _calibration_archives(C.BIDS_DIR)
    if not archives:
        print("No calibration data found under sourcedata. Run Analysis/bidsify.py first.")
        return
    for a in archives:
        make_calibration_figure(a)
    print(f"\nCalibration figures -> {C.DERIV_DIR / 'calibration'}")


# ###########################################################################
# QUEST+ convergence simulation
# ###########################################################################
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


def sim_main():
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


# ###########################################################################
# Offline slope and lapse refit
# ###########################################################################
GUESS = 0.5                 # 2-AFC chance level
LAPSE_MAX = 0.20

# Limits near the optimiser bounds that mark a fit as "railed": the data do not identify
# the parameter (e.g. near-ceiling accuracy at all tilts). Same alpha check as
# `conditions.fit_alpha`, extended to beta and lapse.
ALPHA_RAIL = (0.12, 40.0)   # search bounds (0.1, 45)
BETA_RAIL = (0.6, 11.0)     # search bounds (0.5, 12)
LAPSE_RAIL_HI = 0.19        # search bound (0, LAPSE_MAX=0.20)


def _railed(alpha: float | None = None, beta: float | None = None,
           lapse: float | None = None) -> bool:
    r = False
    if alpha is not None:
        r = r or alpha <= ALPHA_RAIL[0] or alpha >= ALPHA_RAIL[1]
    if beta is not None:
        r = r or beta <= BETA_RAIL[0] or beta >= BETA_RAIL[1]
    if lapse is not None:
        r = r or lapse >= LAPSE_RAIL_HI
    return bool(r)


# ===========================================================================
# Psychometric function and MLE
# ===========================================================================
def weibull(x, alpha, beta, lapse, guess=GUESS):
    """Weibull psychometric function: P(correct) at stimulus level ``x``.

    ``alpha`` is the threshold (scale), ``beta`` the slope, ``lapse`` the gap below 1 at the
    upper asymptote and ``guess`` the floor (0.5 for 2-AFC). Same parameterisation as the
    rig's QUEST+, so offline and online fits are comparable.
    """
    x = np.asarray(x, float)
    f = 1.0 - np.exp(-np.power(np.clip(x, 1e-9, None) / alpha, beta))
    return guess + (1.0 - guess - lapse) * f


def _neg_log_lik(theta, x, correct):
    log_alpha, log_beta, lapse = theta
    p = weibull(x, np.exp(log_alpha), np.exp(log_beta), lapse)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -float(np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def fit_psychometric(x, correct, alpha0=4.0, beta0=3.0) -> dict:
    """MLE of (threshold alpha, slope beta, lapse) for 2-AFC Weibull data."""
    x = np.asarray(x, float); correct = np.asarray(correct, float)
    if x.size < 8 or np.unique(correct).size < 2:
        return {"ok": False, "reason": "insufficient/degenerate data"}
    res = minimize(_neg_log_lik, x0=[np.log(alpha0), np.log(beta0), 0.03],
                   args=(x, correct), method="L-BFGS-B",
                   bounds=[(np.log(0.1), np.log(45)), (np.log(0.5), np.log(12)),
                           (0.0, LAPSE_MAX)])
    a, b, lam = np.exp(res.x[0]), np.exp(res.x[1]), res.x[2]
    # threshold at 75 % correct (QUEST+ target)
    thr75 = _inverse(0.75, a, b, lam)
    railed = _railed(a, b, lam)
    return {"ok": bool(res.success) and not railed, "railed": railed,
            "threshold_alpha_deg": round(float(a), 4),
            "slope_beta": round(float(b), 4), "lapse": round(float(lam), 4),
            "thresh75_deg": round(float(thr75), 4), "neg_log_lik": round(float(res.fun), 3),
            "n_trials": int(x.size)}


def _inverse(p_target, alpha, beta, lapse, guess=GUESS):
    """Tilt giving p_target correct."""
    num = (p_target - guess) / (1.0 - guess - lapse)
    num = np.clip(num, 1e-9, 1 - 1e-9)
    return alpha * (-np.log(1.0 - num)) ** (1.0 / beta)


def fit_joint(observers, lapse: float | None = None) -> dict:
    """MLE of one shared slope and lapse with per-participant thresholds.

    Pooling trials across observers with different thresholds flattens the slope; a
    joint fit with individual thresholds avoids this. ``observers`` is a list of
    ``(tilt, correct)`` arrays; ``lapse`` fixes the lapse instead of fitting it.
    """
    obs = [(np.asarray(x, float), np.asarray(c, float)) for x, c in observers
           if np.asarray(x).size >= 8 and np.unique(c).size >= 2]
    K = len(obs)
    if K == 0:
        return {"ok": False, "reason": "no usable observers"}
    x_all = np.concatenate([x for x, _ in obs])
    c_all = np.concatenate([c for _, c in obs])
    idx = np.concatenate([np.full(x.size, j) for j, (x, _) in enumerate(obs)])
    fixed = lapse is not None
    n_shared = 1 if fixed else 2

    def nll(theta):
        lam = lapse if fixed else theta[1]
        p = weibull(x_all, np.exp(theta[n_shared:])[idx], np.exp(theta[0]), lam)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -float(np.sum(c_all * np.log(p) + (1 - c_all) * np.log(1 - p)))

    x0 = ([np.log(3.0)] + ([] if fixed else [0.03])
          + [np.log(max(1.0, np.median(x))) for x, _ in obs])
    bounds = ([(np.log(0.5), np.log(12))] + ([] if fixed else [(0.0, LAPSE_MAX)])
              + [(np.log(0.1), np.log(45))] * K)
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)
    beta = float(np.exp(res.x[0]))
    lapse = float(lapse) if fixed else float(res.x[1])
    alphas = [float(np.exp(v)) for v in res.x[n_shared:]]
    thr75 = [float(_inverse(0.75, a, beta, lapse)) for a in alphas]
    # "ok" (used by conditions.py) depends on the shared parameters only, so one
    # observer with a railed threshold does not invalidate the fit. Railed thresholds are
    # left out of thresh75_median_deg and counted in n_railed_observers.
    alpha_railed = [_railed(alpha=a) for a in alphas]
    shared_railed = _railed(beta=beta, lapse=lapse)
    thr75_clean = [t for t, r in zip(thr75, alpha_railed) if not r]
    return {"ok": bool(res.success) and not shared_railed, "railed": shared_railed,
            "n_railed_observers": int(sum(alpha_railed)),
            "slope_beta": round(beta, 4), "lapse": round(lapse, 4),
            "thresholds_alpha_deg": [round(a, 3) for a in alphas],
            "alpha_railed": alpha_railed,
            "thresh75_median_deg": round(float(np.median(thr75_clean or thr75)), 4),
            "n_participants": K, "n_trials": int(sum(x.size for x, _ in obs))}


# ===========================================================================
# Slope-invariance test across conditions (standing vs walking)
# ===========================================================================
def _fit_conditioned(obs, conds, share_slope, share_lapse) -> dict:
    """Joint MLE over conditions with optionally shared slope/lapse.

    ``obs`` maps condition -> list of ``(tilt, correct)``. Each observer in each condition
    has its own threshold; slope and lapse are shared (``share_*=True``) or per condition.
    Returns the negative log-likelihood, slope/lapse per condition and the number of free
    parameters, for a likelihood-ratio test.
    """
    thr_keys = [(c, j) for c in conds for j in range(len(obs[c]))]
    nthr = len(thr_keys)
    nbeta = 1 if share_slope else len(conds)
    nlapse = 1 if share_lapse else len(conds)
    cidx = {c: i for i, c in enumerate(conds)}

    def unpack(theta):
        bvec, lvec, avec = theta[:nbeta], theta[nbeta:nbeta + nlapse], theta[nbeta + nlapse:]
        betas = {c: float(np.exp(bvec[0 if share_slope else cidx[c]])) for c in conds}
        lapses = {c: float(lvec[0 if share_lapse else cidx[c]]) for c in conds}
        return betas, lapses, avec

    def nll(theta):
        betas, lapses, avec = unpack(theta)
        total = 0.0
        for k, (c, j) in enumerate(thr_keys):
            x, cc = obs[c][j]
            p = np.clip(weibull(x, np.exp(avec[k]), betas[c], lapses[c]), 1e-6, 1 - 1e-6)
            total -= np.sum(cc * np.log(p) + (1 - cc) * np.log(1 - p))
        return float(total)

    x0 = ([np.log(3.0)] * nbeta + [0.03] * nlapse
          + [np.log(max(1.0, np.median(obs[c][j][0]))) for c, j in thr_keys])
    bounds = ([(np.log(0.5), np.log(12))] * nbeta + [(0.0, LAPSE_MAX)] * nlapse
              + [(np.log(0.1), np.log(45))] * nthr)
    res = minimize(nll, x0=x0, method="L-BFGS-B", bounds=bounds)
    betas, lapses, _ = unpack(res.x)
    return {"nll": float(res.fun), "betas": betas, "lapses": lapses,
            "n_params": nbeta + nlapse + nthr, "success": bool(res.success)}


def fit_slope_invariance(observers_by_cond) -> dict:
    """Test whether the slope differs between conditions (slope invariance, Davidson 2024).

    Two nested joint models, with per-participant thresholds:
      * ``free``: slope and lapse per condition;
      * ``shared_slope``: one slope, lapse per condition.
    The likelihood-ratio test (df = number of conditions - 1) tests equal slopes. A small p
    would undermine the fixed-slope comparison in ``conditions``.
    ``observers_by_cond`` maps condition -> list of ``(tilt, correct)``.
    """
    obs = {c: [(np.asarray(x, float), np.asarray(cc, float)) for x, cc in v
               if np.asarray(x).size >= 8 and np.unique(cc).size >= 2]
           for c, v in observers_by_cond.items()}
    conds = [c for c in obs if obs[c]]
    if len(conds) < 2:
        return {"ok": False, "reason": "need >=2 conditions with usable data"}

    free = _fit_conditioned(obs, conds, share_slope=False, share_lapse=False)
    shared = _fit_conditioned(obs, conds, share_slope=True, share_lapse=False)
    df = len(conds) - 1
    lr = max(0.0, 2.0 * (shared["nll"] - free["nll"]))   # shared is the nested model
    p = float(chi2.sf(lr, df))
    return {"ok": bool(free["success"] and shared["success"]),
            "conditions": conds,
            "slope_per_condition": {c: round(free["betas"][c], 4) for c in conds},
            "lapse_per_condition": {c: round(free["lapses"][c], 4) for c in conds},
            "shared_slope": round(shared["betas"][conds[0]], 4),
            "lr_stat": round(lr, 4), "df": df, "p_value": round(p, 4),
            "slope_differs": bool(p < 0.05),
            "n_participants_per_condition": {c: len(obs[c]) for c in conds}}


# ===========================================================================
# Simulation study (parameter recovery at the rig's calibration design)
# ===========================================================================
# The rig's calibration, as the simulation reproduces it. Thresholds and trial counts
# come from the 23 calibrations the exclusion gate kept (2026-10-13): QUEST+ alpha has
# log10 mean 0.08 and SD 0.19 (about 1.2 deg; standing 0.03 / 0.16, walking 0.13 / 0.21);
# 34-66 QUEST+ trials per calibration (median 49); a catch trial at 12 deg every ~6
# trials, of which 2 of 166 were wrong. QUEST+ itself (StaircaseManager, mirrored in
# the convergence simulation) assumes slope 2.5 and lapse 0.05 and places its trials on
# the 0.3-25 deg grid by minimum expected entropy; catch trials do not update it.
RIG_THR_LOG10 = (0.08, 0.19)
RIG_TRIALS = (34, 66)
CATCH_TILT_DEG, CATCH_EVERY = 12.0, 6
RIG_SLOPE, RIG_LAPSE = 2.5, 0.05
SIM_TRUE_SLOPE = 3.0        # between the QUEST+ design value and the standing refit
SIM_TRUE_LAPSE = 0.01       # the catch-trial error rate (2/166)
SIM_N_CALIBRATIONS = (12, 24)   # per condition: now, and 12 participants x 2 visits
SIM_REPS = 200
SIM_SEED = 2026
STRATEGIES = {
    "free": "lapse free, QUEST+ trials only",
    "rig": f"lapse fixed at {RIG_LAPSE} (QUEST+ value)",
    "catch": "lapse free, with catch trials",
}


def simulate_calibration(rng, alpha: float, beta: float, lapse: float, n_trials: int):
    """One calibration as the rig runs it: two interleaved QUEST+ trackers placing
    ``n_trials`` trials, plus a catch trial at ``CATCH_TILT_DEG`` every ``CATCH_EVERY``.

    Returns the QUEST+ trials ``(tilt, correct)``, the catch trials ``(tilt, correct)``
    and the pooled posterior threshold (deg), the calibration's estimate.
    """
    try:
        pass
    except ImportError:
        pass
    a_trk, b_trk = Quest(START_A, PRIOR_SD), Quest(START_B, PRIOR_SD)
    pool = Quest(START_A, 0.0)
    a_log = np.log10(alpha)
    x = np.empty(n_trials)
    c = np.empty(n_trials)
    for t in range(n_trials):
        trk = a_trk if t % 2 == 0 else b_trk
        x_log = trk.next_stim()
        ok = rng.random() < _weibull(x_log, a_log, beta, lapse)
        trk.update(ok)
        pool.update_at(x_log, ok)
        x[t], c[t] = 10 ** x_log, float(ok)
    n_catch = n_trials // CATCH_EVERY
    xc = np.full(n_catch, CATCH_TILT_DEG)
    cc = (rng.random(n_catch) < weibull(xc, alpha, beta, lapse)).astype(float)
    return (x, c), (xc, cc), float(10 ** pool.thr_log())


def simulation_study(n_reps: int = SIM_REPS, n_calibrations=SIM_N_CALIBRATIONS,
                     true_slope: float = SIM_TRUE_SLOPE, true_lapse: float = SIM_TRUE_LAPSE,
                     seed: int = SIM_SEED) -> dict:
    """Repeat the rig's calibration ``n_reps`` times per cohort size and refit each.

    Every repetition draws thresholds and trial counts like the rig's, runs QUEST+,
    and fits the shared slope and lapse three ways (``STRATEGIES``). Returns the
    per-repetition estimates and their summary (median, quartiles, bias, RMSE).
    """
    rng = np.random.default_rng(seed)
    rows, placement = [], []
    for k in n_calibrations:
        for rep in range(n_reps):
            stair, catch = [], []
            for _ in range(k):
                alpha = float(10 ** rng.normal(*RIG_THR_LOG10))
                n = int(rng.integers(RIG_TRIALS[0], RIG_TRIALS[1] + 1))
                s, ct, est = simulate_calibration(rng, alpha, true_slope, true_lapse, n)
                stair.append(s)
                catch.append(ct)
                if rep == 0:
                    placement.append((s[0] / est, n))
            fits = {"free": fit_joint(stair),
                    "rig": fit_joint(stair, lapse=RIG_LAPSE),
                    "catch": fit_joint([(np.r_[x, xc], np.r_[c, cc])
                                        for (x, c), (xc, cc) in zip(stair, catch)])}
            for name, f in fits.items():
                rows.append({"n_calibrations": k, "rep": rep, "strategy": name,
                             "slope": f.get("slope_beta"), "lapse": f.get("lapse"),
                             "railed": bool(f.get("railed", True))})
    df = pd.DataFrame(rows)
    summary = []
    for (k, name), g in df.groupby(["n_calibrations", "strategy"], sort=False):
        row = {"n_calibrations": int(k), "strategy": name, "label": STRATEGIES[name],
               "n_reps": int(len(g)), "railed_pct": round(100 * float(g["railed"].mean()), 1)}
        for par, truth in (("slope", true_slope), ("lapse", true_lapse)):
            v = g[par].astype(float)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            row.update({f"{par}_median": round(float(med), 4),
                        f"{par}_iqr": [round(float(q1), 4), round(float(q3), 4)],
                        f"{par}_bias": round(float(med - truth), 4),
                        f"{par}_rmse": round(float(np.sqrt(np.mean((v - truth) ** 2))), 4)})
        summary.append(row)
    return {"design": {"thr_log10_mean_sd": list(RIG_THR_LOG10), "quest_trials": list(RIG_TRIALS),
                       "catch_tilt_deg": CATCH_TILT_DEG, "catch_every": CATCH_EVERY,
                       "quest_assumed": {"slope": RIG_SLOPE, "lapse": RIG_LAPSE},
                       "true_slope": true_slope, "true_lapse": true_lapse,
                       "n_calibrations": list(n_calibrations), "n_reps": n_reps, "seed": seed},
            "summary": summary, "_estimates": df, "_placement": placement}


def _real_placement() -> tuple[np.ndarray, list[int]] | None:
    """Presented tilt / final QUEST+ threshold, and QUEST+ trial counts, of the gated
    real calibrations (``None`` when there are none)."""
    gated = bids_io.excluded_calibrations()
    ratios, counts = [], []
    for f in bids_io.find_calibration_tables():
        if bids_io.calibration_label(f) in gated:
            continue
        df = pd.read_csv(f, sep="\t", na_values="n/a")
        if not {"presented_tilt_deg", "post_mean_threshold_deg", "tracker"} <= set(df):
            continue
        d = df[~df["tracker"].astype(str).isin(["catch", "warmup"])]
        thr = pd.to_numeric(d["post_mean_threshold_deg"], errors="coerce").dropna()
        if thr.empty:
            continue
        ratios.append(pd.to_numeric(d["presented_tilt_deg"], errors="coerce").dropna()
                      / float(thr.iloc[-1]))
        counts.append(len(d))
    return (np.concatenate(ratios), counts) if ratios else None


def _fig_recovery(rec: dict, out: Path, real=None):
    """Three panels: (a) the simulated calibrations against the real ones, (b) slope and
    (c) lapse recovered by each lapse strategy, for each cohort size."""
    des, est = rec["design"], rec["_estimates"]
    colors = {"free": S.BLUE, "rig": S.ORANGE, "catch": S.GREEN}
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2_IN, 3.1),
                           gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})

    # a) where QUEST+ puts its trials, relative to the calibrated threshold
    bins = np.geomspace(0.1, 30, 36)
    sim_n = [n for _, n in rec["_placement"]]
    ax[0].hist(np.concatenate([r for r, _ in rec["_placement"]]), bins=bins, density=True,
               color=S.MIST, label=f"simulated ({min(sim_n)}-{max(sim_n)} trials)")
    if real is not None:
        ax[0].hist(real[0], bins=bins, density=True, histtype="step", color=S.INK, lw=1.2,
                   label=f"rig ({min(real[1])}-{max(real[1])} trials)")
    catch_x = CATCH_TILT_DEG / 10 ** des["thr_log10_mean_sd"][0]
    ax[0].axvline(catch_x, color=S.GREY, ls=":", lw=1)
    ax[0].text(catch_x * 0.85, 0.5, f"catch trials ({CATCH_TILT_DEG:g}°)", rotation=90,
               ha="right", va="center", transform=ax[0].get_xaxis_transform(),
               fontsize=S.MIN_PT, color=S.GREY)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("tilt / calibrated threshold")
    ax[0].set_ylabel("density")
    ax[0].legend(loc="upper left", fontsize=S.MIN_PT, handlelength=1.2)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.3)
    S.panel(ax[0], "a", "Trial placement")

    # b, c) recovered slope and lapse per cohort size, one box per strategy
    ks = des["n_calibrations"]
    names = list(STRATEGIES)
    width = 0.26
    for a, par, truth in ((ax[1], "slope", des["true_slope"]), (ax[2], "lapse", des["true_lapse"])):
        for i, name in enumerate(names):
            pos = [g + (i - 1) * width for g in range(len(ks))]
            if par == "lapse" and name == "rig":          # fixed, not estimated
                for xpos in pos:
                    a.plot([xpos - width * 0.4, xpos + width * 0.4], [RIG_LAPSE] * 2,
                           color=colors[name], lw=2.5, solid_capstyle="butt")
                continue
            data = [est[(est["n_calibrations"] == k) & (est["strategy"] == name)][par].astype(float)
                    for k in ks]
            bp = a.boxplot(data, positions=pos, widths=width * 0.8, patch_artist=True,
                           showfliers=False, medianprops={"color": S.INK, "lw": 1.2},
                           whiskerprops={"color": colors[name]}, capprops={"color": colors[name]})
            for box in bp["boxes"]:
                box.set(facecolor=colors[name], edgecolor=colors[name], alpha=0.85)
        a.axhline(truth, color=S.INK, ls="--", lw=1, zorder=0)
        a.set_xticks(range(len(ks)), [str(k) for k in ks])
        a.set_xlim(-0.55, len(ks) - 0.45)
        a.set_xlabel("calibrations per condition")
    ax[1].axhline(RIG_SLOPE, color=S.GREY, ls=":", lw=1.2, zorder=0)
    ax[2].axhline(RIG_LAPSE, color=S.GREY, ls=":", lw=1.2, zorder=0)
    ax[1].set_ylabel("recovered slope β")
    ax[2].set_ylabel("recovered lapse λ")
    S.panel(ax[1], "b", "Slope")
    S.panel(ax[2], "c", "Lapse")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[n]) for n in names]
    labels = [STRATEGIES[n] for n in names]
    handles += [plt.Line2D([], [], color=S.INK, ls="--", lw=1),
                plt.Line2D([], [], color=S.GREY, ls=":", lw=1.2)]
    labels += [f"true value (β {des['true_slope']:g}, λ {des['true_lapse']:g})",
               f"QUEST+ assumption (β {RIG_SLOPE:g}, λ {RIG_LAPSE:g})"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=S.MIN_PT, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    S.save(fig, out)


# ===========================================================================
# Real data (BIDS calibration tables)
# ===========================================================================
def refit_from_bids() -> dict | None:
    """Refit threshold/slope/lapse from the ses-01 calibration _beh.tsv files.

    Each participant has a standing and a walking calibration (acq- entity). A joint fit is
    run per condition; legacy files without acq go under 'calibration'. A slope-invariance
    LRT compares the conditions. Returns ``{"by_condition": {condition: {joint,
    pooled_naive, n_calibrations}}, "slope_invariance": {...}}``.

    Calibrations rejected by the exclusion gate (UNRELIABLE, rule active) are left out: a
    participant at chance has no slope, and a flat calibration pushes the shared slope to
    its bound and creates a spurious standing-vs-walking difference.

    Each calibration contributes its QUEST+ and catch trials; warm-up trials are left out.
    """
    beh = bids_io.find_calibration_tables()
    gated = bids_io.excluded_calibrations()
    by_cond: dict[str, list] = {}
    for f in beh:
        if bids_io.calibration_label(f) in gated:
            continue
        df = pd.read_csv(f, sep="\t")
        if "presented_tilt_deg" not in df or "correct" not in df:
            continue
        # Catch trials (12 deg, ~10x threshold) stay in: with a lapse term they inform only
        # the lapse, which the QUEST+ trials barely constrain (see simulation_study).
        # Warm-up practice trials are dropped: errors there are learning, not lapses.
        if "tracker" in df:
            df = df[df["tracker"].astype(str) != "warmup"]
        d = df[pd.to_numeric(df["correct"], errors="coerce").isin([0, 1])]
        x = pd.to_numeric(d["presented_tilt_deg"], errors="coerce").to_numpy()
        c = pd.to_numeric(d["correct"], errors="coerce").to_numpy()
        m = re.search(r"_acq-(standing|walking)", f.name)
        cond = m.group(1) if m else "calibration"
        by_cond.setdefault(cond, []).append((x, c))
    if not by_cond:
        return None
    out: dict[str, dict] = {}
    for cond, observers in by_cond.items():
        joint = fit_joint(observers)                                          # shared slope/lapse
        pooled = fit_psychometric(np.concatenate([x for x, _ in observers]),
                                  np.concatenate([c for _, c in observers]))  # naive pooling, for comparison
        out[cond] = {"joint": joint, "pooled_naive": pooled,
                     "n_calibrations": len(observers)}
    invariance = fit_slope_invariance(by_cond)   # tests Davidson's slope invariance
    return {"by_condition": out, "slope_invariance": invariance,
            "excluded_by_gate": sorted(gated)}


def _print_simulation(rec: dict) -> None:
    des = rec["design"]
    print("=== Calibration slope/lapse refit — simulation study at the rig's design ===")
    print(f"  {des['n_reps']} repetitions per cohort; true slope {des['true_slope']}, "
          f"true lapse {des['true_lapse']}; QUEST+ assumes slope {RIG_SLOPE}, "
          f"lapse {RIG_LAPSE}")
    for r in rec["summary"]:
        print(f"  {r['n_calibrations']:>2} calibrations | {r['label']:<34}: slope "
              f"{r['slope_median']:.2f} [{r['slope_iqr'][0]:.2f}, {r['slope_iqr'][1]:.2f}] "
              f"(bias {r['slope_bias']:+.2f}), lapse {r['lapse_median']:.3f} "
              f"(bias {r['lapse_bias']:+.3f}), railed {r['railed_pct']:.0f} %")


def stage_simulation(out_dir: Path) -> dict:
    """The pipeline step: run (or reuse) the simulation study and draw its figure."""
    out_json = out_dir / "calibration_refit_simulation.json"
    out_tsv = out_dir / "calibration_refit_simulation_estimates.tsv"
    out_png = bids_io.fig_dir("calibration") / "calibration_refit_simulation.png"
    real = _real_placement()
    fp = stage_cache.fingerprint(
        stage_cache.code_digest("calibration.py",
                                "plotstyle.py"),
        np.__version__, [] if real is None else [real[0], real[1]])
    outputs = [out_json, out_tsv, out_png]
    if stage_cache.is_fresh("calib_refit", "simulation", fp, outputs):
        rec = json.loads(out_json.read_text(encoding="utf-8"))
        print("  code and real calibrations unchanged since the last run: reusing it")
        _print_simulation(rec)
        return rec
    rec = simulation_study()
    rec["_estimates"].to_csv(out_tsv, sep="\t", index=False)
    _fig_recovery(rec, out_png, real)
    pub = {k: v for k, v in rec.items() if not k.startswith("_")}
    out_json.write_text(json.dumps(pub, indent=2), encoding="utf-8")
    stage_cache.record("calib_refit", "simulation", fp, outputs)
    _print_simulation(rec)
    return rec


# ===========================================================================
def refit_main():
    """Run the simulation study, or with ``--bids`` refit the real calibrations."""
    out_dir = bids_io.deriv_dir("calibration")
    if "--bids" in sys.argv:
        fit = refit_from_bids()
        if fit is None:
            print("No BIDS calibration tables found.")
            return
        bids_io.gate_report("calib_refit", unit="calibrations")
        (out_dir / "calibration_refit_real.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")
        print("=== Real-calibration refit (joint shared slope/lapse, per condition) ===")
        for cond, res in fit["by_condition"].items():
            j, pn = res.get("joint", {}), res.get("pooled_naive", {})
            print(f"  [{cond}] {res.get('n_calibrations')} calibration(s), {j.get('n_trials')} trials")
            print(f"     JOINT : slope beta={j.get('slope_beta')}, lapse={j.get('lapse')}, "
                  f"median threshold(75%)={j.get('thresh75_median_deg')} deg"
                  + (" [RAILED - not trustworthy]" if j.get("railed") else ""))
            if j.get("n_railed_observers"):
                print(f"     ({j['n_railed_observers']}/{j.get('n_participants')} observer(s) had a "
                      "railed own threshold — excluded from the median above)")
            print(f"     (naive pooled, flattens slope: beta={pn.get('slope_beta')}, lapse={pn.get('lapse')})")
        inv = fit["slope_invariance"]
        if inv.get("ok"):
            print("\n  --- Slope-invariance test (Davidson claim: walking shifts threshold, NOT slope) ---")
            print(f"     per-condition slope: {inv['slope_per_condition']}")
            print(f"     shared-slope LRT: chi2({inv['df']})={inv['lr_stat']}, p={inv['p_value']} "
                  f"-> slope {'DIFFERS between conditions' if inv['slope_differs'] else 'invariant (n.s., supports Davidson)'}")
        elif inv.get("reason"):
            print(f"\n  Slope-invariance test skipped: {inv['reason']}")
        print(f"\nOutput -> {out_dir/'calibration_refit_real.json'}")
        return

    stage_simulation(out_dir)


# ###########################################################################
# Entry point
# ###########################################################################
PARTS = {"check": check_main, "refit": refit_main, "convergence": sim_main}


def main() -> None:
    """Pipeline stage ``calibration``: the calibration check (the refit runs after the gate)."""
    check_main()


if __name__ == "__main__":
    PARTS[next((a for a in sys.argv[1:] if not a.startswith("-")), "check")]()
