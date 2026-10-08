"""Standing vs walking comparison, after Davidson et al. (2024) Fig 2.

Aggregates behaviour per (participant, condition) and plots:
  * a) overall hit rate, standing vs walking (paired);
  * b) hit rate by difficulty (HARD / THRESHOLD / EASY, targets 65 / 75 / 85 %), the
       counterpart of Davidson's psychometric function over staircase levels;
  * c) overall reaction time, standing vs walking (paired);
  * d) reaction time by difficulty and condition.

If only one condition is present, the other side is left empty.

Usage::

    python -m Analysis.conditions
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import bids_io
    from . import plotstyle as S
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import plotstyle as S

S.apply()

CONDITIONS = ["standing", "walking"]
COND_COL = S.COND
DIFF_ORDER = ["HARD", "THR", "EASY"]      # increasing target accuracy (65/75/85 %)


def _aggregate():
    """One row per (participant, condition) with overall and per-difficulty hit rate and RT."""
    rows = []
    for rec in bids_io.find_recordings():
        if rec.condition not in CONDITIONS:
            continue
        ev = bids_io.load_events(rec)
        if ev.empty:
            continue
        ev = ev[ev["is_training"] == 0].copy()
        ev["acc"] = (ev["response_correct"] == 1).astype(float)
        ev["rt"] = pd.to_numeric(ev["response_time"], errors="coerce")
        row = {"sub": rec.sub, "condition": rec.condition,
               "hit_rate": float(ev["acc"].mean()),
               "rt": float(ev["rt"].mean())}
        for d in DIFF_ORDER:
            g = ev[ev["difficulty"] == d]
            row[f"hr_{d}"] = float(g["acc"].mean()) if len(g) else np.nan
            row[f"rt_{d}"] = float(g["rt"].mean()) if len(g) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Average over a participant's blocks so each participant counts once per condition.
    num = [c for c in df.columns if c not in ("sub", "condition")]
    return df.groupby(["sub", "condition"], as_index=False)[num].mean()


def _paired(ax, df, col, ylabel):
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        x = CONDITIONS.index(cond)
        ax.scatter(np.full(len(sub), x) + np.random.uniform(-0.05, 0.05, len(sub)),
                   sub[col], color=COND_COL[cond], alpha=0.7, zorder=2)
        if len(sub):
            ax.errorbar(x, sub[col].mean(), yerr=sub[col].sem() if len(sub) > 1 else 0,
                        fmt="o", ms=13, mfc="none", mew=1.8, color=COND_COL[cond], zorder=3)
    # connect paired participants
    for s in df["sub"].unique():
        d = df[df["sub"] == s]
        if set(d["condition"]) == set(CONDITIONS):
            ax.plot([0, 1], [d[d.condition == "standing"][col].iloc[0],
                             d[d.condition == "walking"][col].iloc[0]], color=S.MIST, zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel(ylabel); ax.set_xlim(-0.5, 1.5)


def _by_difficulty(ax, df, prefix, ylabel):
    x = np.arange(len(DIFF_ORDER))
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        means = [sub[f"{prefix}_{d}"].mean() for d in DIFF_ORDER]
        sems = [sub[f"{prefix}_{d}"].sem() if len(sub) > 1 else 0 for d in DIFF_ORDER]
        ax.errorbar(x, means, yerr=sems, fmt="-o", color=COND_COL[cond], label=cond, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{d}\n({p})" for d, p in
                                          zip(DIFF_ORDER, ["65%", "75%", "85%"])])
    ax.set_xlabel("difficulty (target % correct)"); ax.set_ylabel(ylabel); ax.legend()


def main():
    """Compare standing against walking on hit rate and reaction time (Fig 2)."""
    df = _aggregate()
    bids_io.gate_report("conditions", condition=CONDITIONS,
                        n_analysed=int(df["sub"].nunique()) if not df.empty else 0)
    if df.empty:
        print("No standing/walking recordings found. Run Analysis/bidsify.py first.")
        return
    present = sorted(df["condition"].unique())
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    _paired(ax[0, 0], df, "hit_rate", "hit rate"); S.panel(ax[0, 0], "a", "Hit rate")
    _by_difficulty(ax[0, 1], df, "hr", "hit rate"); S.panel(ax[0, 1], "b", "Hit rate × difficulty")
    _paired(ax[1, 0], df, "rt", "reaction time (s)"); S.panel(ax[1, 0], "c", "Reaction time")
    _by_difficulty(ax[1, 1], df, "rt", "reaction time (s)"); S.panel(ax[1, 1], "d", "RT × difficulty")

    fig.suptitle("Standing vs walking", fontsize=13, fontweight="semibold", x=0.012, ha="left")
    if set(present) != set(CONDITIONS):
        S.caption(fig, f"Only {', '.join(present)} present — standing block pending.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = bids_io.fig_dir("conditions") / "conditions_standing_vs_walking.png"
    S.save(fig, out)

    print(f"Conditions present: {present}")
    for _, r in df.iterrows():
        print(f"  {r['sub']} {r['condition']:8s}: hit rate {r['hit_rate']:.3f}, RT {r['rt']:.3f}s")
    if set(present) != set(CONDITIONS):
        print("  NOTE: standing block not yet recorded — Fig 2 will populate both sides "
              "once a standing session (condition=standing) exists.")
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
