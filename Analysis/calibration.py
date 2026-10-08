"""QUEST+ calibration convergence figures and reliability reports in derivatives/.

Reads the gzipped raw calibration log from ``sourcedata``, which keeps the two ``#``
metadata lines (final threshold/easy/hard tilts, QUEST+ parameters) that the BIDS
``_beh.tsv`` drops, and plots with the Agg backend. ``Analysis/plot_calibration.py``
remains available as a standalone script.

Also writes ``*_calibration_qc.json``. Blocking ``flags`` (verdict UNRELIABLE): inattentive
(catch lapses), wide posterior, high-tilt lapses (only if significantly above the model
lapse rate), ceiling rail (threshold at grid max, no real threshold) and floor rail
(threshold near grid min or difficulty levels clamped: task too easy or grid mis-scaled).
Non-blocking ``warnings``: A/B tracker divergence, only when the posteriors are
statistically separated given their widths, since two short posteriors disagree ~30 % of
the time by chance.

Usage::

    python -m Analysis.calibration
    python Analysis/calibration.py
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
# the time even for good observers (README, calibration_convergence_sim). Warn only if
# |logA-logB| exceeds this many combined SDs, suggesting an inconsistent or side-biased
# responder.
DIVERGENCE_SEP_Z = 2.0

try:
    from . import config as C
    from . import bids_io
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C
    import bids_io
    import plotstyle as S

S.apply()


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


def main() -> None:
    """Plot every calibration and write its reliability QC."""
    archives = _calibration_archives(C.BIDS_DIR)
    if not archives:
        print("No calibration data found under sourcedata. Run Analysis/bidsify.py first.")
        return
    for a in archives:
        make_calibration_figure(a)
    print(f"\nCalibration figures -> {C.DERIV_DIR / 'calibration'}")


if __name__ == "__main__":
    main()
