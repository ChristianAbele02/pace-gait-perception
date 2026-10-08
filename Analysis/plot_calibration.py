#!/usr/bin/env python3
"""
Convergence plot for one QUEST+ tilt calibration.

Reads a `*_calibration.csv` written by StaircaseManager (one row per trial) and plots
both interleaved trackers (A/B): presented tilt per trial with correct/incorrect
markers, the running threshold estimate with a ±SD band, and the final hard /
threshold / easy tilts.

Usage:
    python plot_calibration.py                      # newest *_calibration*.csv in CWD
    python plot_calibration.py path/to/p05-b1-t03_calibration.csv
    python plot_calibration.py --dir /path/to/data  # newest in that folder
    python plot_calibration.py file.csv --no-show   # just save the PNG

Output: a PNG next to the CSV (same name, .png). Requires only matplotlib.

CSV format (semicolon-separated; lines starting with '#' are metadata):
    # threshold_deg=...;easy_deg=...;hard_deg=...
    # guess=...;slope_beta=...;lapse=...;targetPC_hard=...;targetPC_thr=...;targetPC_easy=...
    tracker;global_trial;tracker_trial;presented_tilt_deg;tilt_direction;response;correct;post_mean_threshold_deg;post_sd_log
"""

import argparse
import csv
import glob
import os
import sys

import os.path as _osp
sys.path.insert(0, _osp.dirname(_osp.abspath(__file__)))

import matplotlib
matplotlib.use("Agg") if "--no-show" in sys.argv else None
import matplotlib.pyplot as plt

import plotstyle as S

S.apply()


def find_latest(folder):
    """Newest ``*_calibration*.csv`` in ``folder``."""
    cands = glob.glob(os.path.join(folder, "*_calibration*.csv"))
    if not cands:
        sys.exit("No *_calibration*.csv found in: " + os.path.abspath(folder))
    return max(cands, key=os.path.getmtime)


def parse_meta(line):
    """Parse a ``# key=value; ...`` header line into a dict, converting numbers.

    These lines hold the QUEST+ settings and the final ``threshold_deg``, which the
    operator uses to resume a participant without recalibrating.
    """
    out = {}
    for part in line.lstrip("#").strip().split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def load(path):
    """Read a raw calibration CSV into ``(metadata dict, trial rows)``.

    Works on the rig's raw format, so a calibration can be checked before BIDS conversion.
    """
    meta = {}
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader((l for l in f), delimiter=";")
        header = None
        for raw in reader:
            if not raw:
                continue
            first = raw[0]
            if first.startswith("#"):
                meta.update(parse_meta(";".join(raw)))
                continue
            if header is None:
                header = raw
                continue
            rows.append(dict(zip(header, raw)))
    return meta, rows


def main():
    """Plot one raw calibration CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="calibration CSV (default: newest in --dir)")
    ap.add_argument("--dir", default=".", help="folder to search if no file given")
    ap.add_argument("--no-show", action="store_true", help="save PNG without opening a window")
    args = ap.parse_args()

    path = args.csv or find_latest(args.dir)
    meta, rows = load(path)
    if not rows:
        sys.exit("No data rows in " + path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for trk in sorted(set(r["tracker"] for r in rows)):
        tr = [r for r in rows if r["tracker"] == trk]
        x = [int(r["global_trial"]) for r in tr]
        tilt = [float(r["presented_tilt_deg"]) for r in tr]
        corr = [int(r["correct"]) for r in tr]
        est = [float(r["post_mean_threshold_deg"]) for r in tr]
        sd = [float(r["post_sd_log"]) for r in tr]
        c = S.TRACKER.get(trk, S.GREY)

        # Running threshold estimate with ±SD band (SD in log10 units).
        ax.plot(x, est, "-", color=c, lw=2, label="Tracker %s — estimate" % trk)
        lo = [e * (10 ** (-s)) for e, s in zip(est, sd)]
        hi = [e * (10 ** (s)) for e, s in zip(est, sd)]
        ax.fill_between(x, lo, hi, color=c, alpha=0.12)

        # Filled = correct, open = incorrect, grey x = timeout (correct == -1, not given
        # to QUEST+).
        xc = [xi for xi, k in zip(x, corr) if k == 1]
        yc = [yi for yi, k in zip(tilt, corr) if k == 1]
        xi_ = [xi for xi, k in zip(x, corr) if k == 0]
        yi_ = [yi for yi, k in zip(tilt, corr) if k == 0]
        xt = [xi for xi, k in zip(x, corr) if k == -1]
        yt = [yi for yi, k in zip(tilt, corr) if k == -1]
        ax.scatter(xc, yc, marker="o", facecolors=c, edgecolors=c, s=40,
                   label="Tracker %s — correct" % trk, zorder=3)
        ax.scatter(xi_, yi_, marker="o", facecolors="white", edgecolors=c, s=40,
                   linewidths=1.4, label="Tracker %s — wrong" % trk, zorder=3)
        if xt:
            ax.scatter(xt, yt, marker="x", color=S.GREY, s=44, linewidths=1.6,
                       label="Tracker %s — no response" % trk, zorder=3)

    # Final difficulty tilts.
    for key, lbl, ls in [("threshold_deg", "threshold", "-"),
                          ("easy_deg", "easy", "--"),
                          ("hard_deg", "hard", ":")]:
        if key in meta:
            ax.axhline(meta[key], color=S.GREY, ls=ls, lw=1.2,
                       label="%s = %.2f°" % (lbl, meta[key]))

    ax.set_yscale("log")
    ax.set_xlabel("Trial (interleaved)")
    ax.set_ylabel("Tilt magnitude (°, log scale)")
    ax.set_title("QUEST+ calibration — " + os.path.basename(path), color=S.INK)
    ax.grid(True, which="both")
    S.legend_outside(ax, ncol=1)
    if all(k in meta for k in ("slope_beta", "lapse", "guess")):
        S.caption(fig, "QUEST+ Weibull: γ = %.2f,  β = %.2f,  λ = %.2f"
                  % (meta["guess"], meta["slope_beta"], meta["lapse"]))
    fig.tight_layout()

    out_png = os.path.splitext(path)[0] + ".png"
    fig.savefig(out_png)
    print("Saved figure to", out_png)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
