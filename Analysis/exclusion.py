"""Exclusion gate: the presentation rule, the block rule, and the effect of the rule.

The pipeline stage ``gate`` runs after the calibration check and before any analysis.

* Presentation rule: flags every trial whose presentation window overlaps a
  head-tracking artefact (``qc/presentation_flags.tsv``). ``bids_io.load_events`` drops
  those trials from every trial-based analysis.
* Block rule: writes the exclusion gate ``qc/gate.tsv`` from frame rate, reversal,
  timeouts and chance-level accuracy (on the trials the presentation rule keeps), the
  calibration verdicts and the manual list.
* Comparison with and without the presentation rule (on demand, not a pipeline stage):
  reruns the pipeline with the rule off on a temporary copy and compares the results.

Usage::

    python -m Analysis.exclusion                                # presentation flags + gate
    python -m Analysis.exclusion compare [--keep | --reuse DIR]  # rule off vs on


Presentation rule
=================

Presentation validity: trials shown while the head tracking failed.

The CAVE renders from the tracked head. The logged ``head_local_pos_*`` is that head
transform (``ILLLoggerController.cs``), so an artefact in it is an artefact in the render
eye. A frozen pose freezes the view; a jump or a dropout shifts it. If this happens around
the 100 ms Gabor, the stimulus can be displaced or not visible, and the trial does not
measure the task. Such trials are dropped from every trial-based analysis: accuracy, RT,
the timeout rate, the stride-phase tests, the thresholds, SDT, DDM, the gait PRC and the
block rule itself.

A trial is flagged when one of these overlaps its presentation window,
``[onset - PRESENTATION_PRE_S, onset + duration + PRESENTATION_POST_S]``, in the raw head
position before any repair (``duration`` is the measured on-screen time):

* ``spike``: a sample ``clean.detect_outliers`` marks on x, y or z (non-finite, a Hampel
  outlier beyond the physiological floor, or an isolated velocity spike).
* ``jump``: a sample beyond the per-axis plausibility radius from the block median
  (``clean.HEAD_CLEAN_AXES``), i.e. a long excursion the Hampel filter misses.
* ``freeze``: stale frames (``clean.stale_frames``) lasting at least
  ``PRESENTATION_STALE_MIN_S``. A single repeated frame does not move the view.
* ``gap``: no head sample for longer than ``PRESENTATION_GAP_S``.

The 0.3 s lead covers the render-eye smoothing (1.8 Hz), which needs about that long to
settle after a jump. Short spikes count here although ``clean.py`` interpolates them for
the gait signal: interpolation repairs the analysis signal, not what was on the wall.

The flags go to ``derivatives/qc/presentation_flags.tsv`` (one row per trial, training
included). ``bids_io.load_events`` reads them and drops flagged trials while
``config.APPLY_PRESENTATION_RULE`` is on. QC, the gait sheets and the trial table read
every trial (``presentation_rule=False``); the trial table keeps the flag as a column.
The file is rebuilt when stale (settings changed, events or motion newer than it), for
the configured dataset only; other datasets are flagged in memory.

Calibration runs log no head tracking, so the rule cannot apply to calibration trials.


Block rule (the exclusion gate)
===============================

Exclusion gate: which blocks and calibrations each analysis may use.

Builds ``derivatives/qc/gate.tsv``, which every analysis stage reads (via
``bids_io.find_recordings`` and ``bids_io.excluded_calibrations``). It runs as the third
stage, after the calibration verdicts; ``bids_io.load_gate`` rebuilds it when stale, so a
stage run alone never uses an old gate.

Automatic rule. Criteria 1-3 judge each block on its own; criterion 4 judges a
participant in one condition, over all of their blocks:

1. Frame rate. A block is excluded if its effective sampling rate (``scans.tsv``) is below
   ``BLOCK_RULE_MIN_SAMPLING_HZ`` (30 Hz), or the median measured on-screen time of its
   stimuli is above ``BLOCK_RULE_MAX_STIM_FACTOR`` (2) times ``STIM_DURATION_S``. The head
   node runs the task and the logger once per rendered frame, normally at ~60 Hz. A head
   node running far slower keeps the Gabor up for several frames, which is a different,
   easier task, and samples the head too coarsely to time the steps: at 6 Hz a trough is
   only located to within ~80 ms, about 100 deg of phase at 4 cps. Neither the responses
   nor the gait of such a block can be analysed, so it leaves both scopes.
2. Reversal. A block significantly below chance is excluded as invalid. This indicates
   reversed answers or a broken stimulus (e.g. a flipped CAVE view), not
   guessing; check the session log.
3. Timeouts. A block with ``BLOCK_RULE_MAX_TIMEOUT_FRACTION`` (25 %) or more main trials
   unanswered is excluded: the task was not done as instructed, and accuracy would rest on
   a self-selected subset.
4. Chance. A participant's blocks in one condition are excluded if their pooled accuracy
   over the answered main trials is not above chance (one-sided binomial test against 0.5
   at ``BLOCK_RULE_ALPHA``), over the blocks that pass 1-3 and are not on the manual list.
   Timeouts are left out rather than scored wrong: a missed response says nothing about
   whether the tilt was seen, and scoring it wrong would push a guesser below 0.5. The test
   pools the blocks because one block is too small to judge: with ~150 answered trials it
   needs ~57 %, and a block whose true accuracy is 60 % fails about one time in four by
   chance alone. Pooled over a condition, a participant who does the task passes and one
   who guesses fails. A weak block of a participant who does the task is kept: dropping
   blocks for their accuracy would select on the outcome, and the harder condition would
   lose the most blocks, biasing the standing-vs-walking comparison. For the same reason
   it is not a fixed cut-off like 60-70 %.
5. Calibration. An UNRELIABLE calibration is left out of the pooled calibration refit
   (slope/lapse fit and standing-vs-walking slope test). Its blocks are not excluded for
   this; they are judged on criteria 1-4, and the verdict is stored with them. An off-target
   threshold makes levels harder or easier than planned, which the difficulty
   residualisation absorbs. If a condition was recalibrated within a visit, the latest
   calibration applies.

Criteria 2-4 use only the trials the presentation rule keeps (``exclusion.py``).
A trial shown while the head tracking froze, jumped or dropped out says nothing about the
participant, so it counts neither for nor against the block. ``n_trials`` counts the valid
main trials, ``n_presentation_flagged`` the main trials the rule removed.

Easy-trial accuracy is not a criterion: with ~17 easy trials per block, a binomial test
fails an engaged participant 10-40 % of the time. It remains a QC warning (``qc.py``).

Scopes. Criteria 2-4 concern responses, so they gate only the ``behaviour`` scope. The
``gait`` scope (gait extraction, phase methods, head stability, gait dynamics, gait PRC)
keeps those blocks, since someone answering at chance can still walk normally. The
frame-rate check (1) and the manual list ``config.EXCLUDED_RECORDINGS`` exclude from both
scopes, and so does a participant rated "not useful" (``config.PARTICIPANT_RATINGS``,
while ``config.EXCLUDE_NOT_USEFUL_PARTICIPANTS`` is on): all their blocks and
calibrations become manual exclusions, with their automatic verdicts still recorded in
``rule``. With ``config.APPLY_BLOCK_RULE`` False, verdicts are still written but only
the manual list excludes.

Each stage logs its N (``bids_io.gate_report``) to ``derivatives/qc/gate_ledger.tsv``.


With and without the presentation rule (on demand)
==================================================

Sensitivity analysis: the pipeline with and without the presentation rule.

Results are computed with the presentation rule on (``exclusion.py``). This tool
reruns the whole pipeline with the rule off (``ILL_PRESENTATION_RULE=0``) on a temporary
copy of the project and compares the two runs, so the effect of the rule can be reported.

Run it after a full pipeline run, so ``derivatives/`` is current with the rule on::

    python -m Analysis.exclusion compare               # rule-off run + comparison
    python -m Analysis.exclusion compare --keep        # keep the rule-off copy
    python -m Analysis.exclusion compare --reuse DIR   # compare against a kept copy

The rule-off run takes as long as a full pipeline run. Output, in
``derivatives/sensitivity/presentation_rule/``:

* ``flagged_trials.tsv``: per block, the main trials, the flagged trials by artefact kind,
  the accuracy of kept and of flagged answered trials, and the gate decision with the
  rule on and off.
* ``flagged_accuracy.tsv``: per condition, whether flagged trials are answered worse than
  the kept trials of the same blocks (observed vs expected correct, normal
  approximation), over the blocks the gate keeps and over all blocks.
* ``gate_changes.tsv``: blocks whose behaviour-scope decision differs.
* ``key_results.tsv``: the headline results, rule on vs rule off.
* ``pvalues.tsv``: every p-value in the result JSONs below 0.10 in either run;
  ``significance_flips.tsv``: every significance flag that differs.
* ``comparison.json``: settings, the dataset compared, and the totals.

Permutation p-values move between the runs even for blocks that lost no trial:
``phase_analysis`` draws every block's permutations from one seeded random stream, so
removing trials earlier redraws the null of every later block.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import norm

try:
    from . import bids_io
    from . import clean
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import clean
    import config as C



# ###########################################################################
# Presentation rule
# ###########################################################################
FLAGS_FILE = "presentation_flags.tsv"
REASONS = ("spike", "jump", "freeze", "gap")
_CACHE: dict[str, tuple[float, dict, dict]] = {}   # path -> (mtime, settings, {label: frame})


def settings() -> dict:
    """What the flags depend on; a change makes the flags file stale."""
    # Bump RuleVersion whenever what counts as an artefact changes.
    return {"RuleVersion": 1,
            "PreS": float(C.PRESENTATION_PRE_S), "PostS": float(C.PRESENTATION_POST_S),
            "StaleMinS": float(C.PRESENTATION_STALE_MIN_S),
            "GapS": float(C.PRESENTATION_GAP_S),
            "RadiusM": dict(clean.HEAD_CLEAN_AXES)}


# ===========================================================================
# Detection
# ===========================================================================
def _spans(t: np.ndarray, mask: np.ndarray, min_s: float = 0.0) -> np.ndarray:
    """``(n, 2)`` start/end times of the runs of ``mask``, those lasting >= ``min_s``."""
    e = np.diff(np.r_[0, mask.astype(int), 0])
    out = [(t[a], t[b - 1]) for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))
           if t[b - 1] - t[a] >= min_s]
    return np.array(out, float).reshape(-1, 2)


def artefact_spans(motion: pd.DataFrame) -> dict[str, np.ndarray]:
    """Time spans of each artefact kind in one block's raw head position."""
    t = motion["t"].to_numpy(float)
    spike = np.zeros(t.size, bool)
    jump = np.zeros(t.size, bool)
    for col, radius in clean.HEAD_CLEAN_AXES.items():
        if col not in motion:
            continue
        v = motion[col].to_numpy(float)
        spike |= clean.detect_outliers(t, v)
        with np.errstate(invalid="ignore"):
            jump |= np.abs(v - np.nanmedian(v)) > radius
    stale, _ = clean.stale_frames(motion)
    stale = np.zeros(t.size, bool) if stale is None else np.asarray(stale, bool)
    i = np.flatnonzero(np.diff(t) > C.PRESENTATION_GAP_S)
    return {"spike": _spans(t, spike), "jump": _spans(t, jump & ~spike),
            "freeze": _spans(t, stale, C.PRESENTATION_STALE_MIN_S),
            "gap": np.c_[t[i], t[i + 1]] if i.size else np.empty((0, 2))}


def flag_trials(onset: np.ndarray, duration: np.ndarray,
                spans: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Which trials' presentation windows overlap an artefact, and which kinds.

    Returns ``(flagged, reasons)``: a boolean per trial and a ``;``-joined list of the
    artefact kinds in its window ("-" when there are none).
    """
    onset = np.asarray(onset, float)
    dur = np.where(np.isfinite(duration), duration, C.STIM_DURATION_S)
    w0 = onset - C.PRESENTATION_PRE_S
    w1 = onset + dur + C.PRESENTATION_POST_S
    hits = {}
    for kind in REASONS:
        s = spans.get(kind, np.empty((0, 2)))
        hits[kind] = (((s[None, :, 0] <= w1[:, None]) & (s[None, :, 1] >= w0[:, None])).any(1)
                      if len(s) else np.zeros(onset.size, bool))
    flagged = np.any([hits[k] for k in REASONS], axis=0) if onset.size else np.zeros(0, bool)
    reasons = [";".join(k for k in REASONS if hits[k][j]) or "-" for j in range(onset.size)]
    return flagged, reasons


def flag_recording(rec) -> pd.DataFrame:
    """One row per event of ``rec`` (training included) with its presentation verdict."""
    ev = bids_io.load_events(rec, presentation_rule=False)
    if ev.empty:
        return pd.DataFrame(columns=["recording", "event", "onset", "is_training",
                                     "presentation_valid", "presentation_flag"])
    motion, _, _ = bids_io.load_motion(rec)
    flagged, reasons = flag_trials(
        ev["onset"].to_numpy(float),
        pd.to_numeric(ev.get("duration"), errors="coerce").to_numpy(float)
        if "duration" in ev else np.full(len(ev), np.nan),
        artefact_spans(motion))
    return pd.DataFrame({
        "recording": rec.label(), "event": np.arange(len(ev)),
        "onset": ev["onset"].to_numpy(float),
        "is_training": pd.to_numeric(ev["is_training"], errors="coerce").fillna(0).astype(int),
        "presentation_valid": (~flagged).astype(int), "presentation_flag": reasons})


# ===========================================================================
# The flags file
# ===========================================================================
def _file(bids_dir: Path | None = None) -> Path:
    return Path(bids_dir or C.BIDS_DIR) / "derivatives" / "qc" / FLAGS_FILE


def build(bids_dir: Path | None = None, quiet: bool = False) -> pd.DataFrame:
    """Flag every trial of every block and write ``qc/presentation_flags.tsv`` (+ .json)."""
    bids_dir = Path(bids_dir or C.BIDS_DIR)
    recs = bids_io.find_recordings(bids_dir, include_excluded=True)
    frames = [flag_recording(r) for r in recs]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = _file(bids_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, na_rep="n/a")
    summary = summarise(df, {r.label(): r.condition for r in recs})
    out.with_suffix(".json").write_text(json.dumps({
        "Description": "Presentation-validity flags (Analysis/exclusion.py): one row per "
                       "trial, training included. A trial is flagged when a head-tracking "
                       "artefact overlaps [onset - PreS, onset + duration + PostS]; flagged "
                       "trials are left out of every trial-based analysis while "
                       "config.APPLY_PRESENTATION_RULE is on.",
        "Settings": settings(),
        "Summary": summary,
        "event": {"Description": "Row of the trial in the block's _events.tsv (0 = first)."},
        "presentation_valid": {"Description": "1 = no artefact in the window, 0 = flagged."},
        "presentation_flag": {"Description": "Artefact kinds in the window (spike, jump, "
                                             "freeze, gap), ';'-joined; '-' if none."},
    }, indent=2), encoding="utf-8")
    _CACHE.pop(str(out), None)
    if not quiet:
        print_summary(summary)
        print(f"Presentation flags -> {out}")
    return df


def summarise(df: pd.DataFrame, condition: dict[str, str]) -> dict:
    """Flagged main trials per condition and per artefact kind."""
    out = {}
    if df.empty:
        return out
    main_trials = df[df["is_training"] == 0].assign(condition=lambda d: d["recording"].map(condition))
    for cond, g in [("all", main_trials)] + list(main_trials.groupby("condition")):
        bad = g[g["presentation_valid"] == 0]
        out[cond] = {"n_main": int(len(g)), "n_flagged": int(len(bad)),
                     "pct_flagged": round(100 * len(bad) / len(g), 2) if len(g) else 0.0,
                     **{k: int(bad["presentation_flag"].str.contains(k).sum())
                        for k in REASONS}}
    return out


def print_summary(summary: dict) -> None:
    state = "ACTIVE" if C.APPLY_PRESENTATION_RULE else "NOT ACTIVE: flags recorded, no trial removed"
    print(f"Presentation rule ({state}):")
    for cond, s in summary.items():
        kinds = ", ".join(f"{k} {s[k]}" for k in REASONS)
        print(f"  {cond:<9}: {s['n_flagged']}/{s['n_main']} main trials flagged "
              f"({s['pct_flagged']}%; {kinds})")


def _read(f: Path) -> tuple[dict, dict]:
    """``(settings, {recording: frame})`` of a flags file, cached by modification time."""
    m = f.stat().st_mtime
    hit = _CACHE.get(str(f))
    if hit and hit[0] == m:
        return hit[1], hit[2]
    try:
        meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        df = pd.read_csv(f, sep="\t", dtype={"presentation_flag": str}, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}, {}
    by_rec = {k: g.reset_index(drop=True) for k, g in df.groupby("recording")} if len(df) else {}
    _CACHE[str(f)] = (m, meta.get("Settings", {}), by_rec)
    return meta.get("Settings", {}), by_rec


def _current(rec, f: Path) -> pd.DataFrame | None:
    """This recording's rows of the flags file, or None if the file is out of date for it."""
    if not f.exists() or not f.with_suffix(".json").exists():
        return None
    st, by_rec = _read(f)
    if st != json.loads(json.dumps(settings())):
        return None
    m = f.stat().st_mtime
    if any(p.exists() and p.stat().st_mtime > m for p in (rec.events_tsv, rec.motion_tsv)):
        return None
    return by_rec.get(rec.label())


def trial_flags(rec, ev: pd.DataFrame) -> pd.DataFrame:
    """The presentation verdict of every row of ``ev`` (the block's full events table).

    Read from the flags file when it is current for this block; the configured dataset's
    file is rebuilt when it is not, and any other dataset is flagged in memory.
    """
    root = rec.motion_tsv.parents[3]
    f = _file(root)
    rows = _current(rec, f)
    if rows is None and root.resolve() == Path(C.BIDS_DIR).resolve():
        print(f"  [presentation] rebuilding {FLAGS_FILE}: out of date for {rec.label()}")
        build(root, quiet=True)
        rows = _current(rec, f)
    ok = (rows is not None and len(rows) == len(ev)
          and np.allclose(rows["onset"].to_numpy(float), ev["onset"].to_numpy(float),
                          equal_nan=True))
    return rows if ok else flag_recording(rec)


def valid_mask(rec, ev: pd.DataFrame) -> np.ndarray:
    """True for the rows of ``ev`` whose presentation was valid."""
    return trial_flags(rec, ev)["presentation_valid"].to_numpy(int) == 1


def presentation_main(quiet: bool = False) -> pd.DataFrame:
    """Build the flags file (pipeline stage ``gate``, before the block rule)."""
    if not bids_io.find_recordings(include_excluded=True):
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return pd.DataFrame()
    return build(quiet=quiet)


# ###########################################################################
# Block rule (the exclusion gate)
# ###########################################################################
def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def calibration_verdict(sub: str, condition: str, visit: int | str | None = None) -> str:
    """Reliability verdict of the calibration a block's threshold came from.

    Each condition is calibrated once per visit (INSTRUCTIONS.md §5), filed as
    ``run-<visit>``. A recalibration in the same visit reuses the Trial number, so bidsify
    files it as ``run-<100k + visit>``; the latest one applies to the following blocks.
    Without a visit, return the worst verdict across the condition's calibrations so an
    unreliable one is not hidden.
    """
    d = C.DERIV_DIR / "calibration" / sub / C.CALIBRATION_SESSION
    js = sorted(d.glob("*_calibration_qc.json")) if d.exists() else []
    if not js:
        return "n/a"
    if condition:
        matched = [j for j in js
                   if f"acq-{condition}" in j.name.lower() or f"_{condition}" in j.name.lower()]
    else:
        matched = list(js)
    if not matched:
        return _load(js[0]).get("verdict", "n/a") if len(js) == 1 else "n/a"
    if visit:
        def run_of(j: Path) -> int | None:
            m = re.search(r"_run-(\d+)", j.name)
            return int(m.group(1)) if m else None
        same = [j for j in matched if run_of(j) is not None and run_of(j) % 100 == int(visit)]
        if same:
            return _load(max(same, key=run_of)).get("verdict", "n/a")
        # No calibration for this visit: report that instead of using another visit's.
        if len(matched) > 1:
            return f"n/a (no calibration for visit {visit})"
    verdicts = [str(_load(j).get("verdict", "n/a")) for j in matched]
    bad = [v for v in verdicts if v.startswith("UNRELIABLE")]
    if bad:
        return bad[0] if len(matched) == 1 else f"{bad[0]} [worst of {len(matched)} visits]"
    return verdicts[0]


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def evaluate(rec, alpha: float | None = None) -> dict:
    """Block-level criteria (frame rate, reversal, timeouts) for one block, with the
    underlying numbers. Chance is judged across blocks by ``pool_chance``."""
    alpha = C.BLOCK_RULE_ALPHA if alpha is None else alpha
    raw = bids_io.load_events(rec, presentation_rule=False)
    ev = bids_io.load_events(rec)
    main_trials = ev[ev["is_training"] == 0] if not ev.empty else ev
    n_main = int(len(main_trials))
    raw_main = raw[raw["is_training"] == 0] if not raw.empty else raw
    n_flagged = int(len(raw_main)) - n_main
    rc = pd.to_numeric(main_trials["response_correct"], errors="coerce") if n_main else pd.Series()
    n = int(rc.isin([0, 1]).sum())                      # answered; timeouts left out
    k = int((rc == 1).sum())
    p_above = binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
    p_below = binomtest(k, n, 0.5, alternative="less").pvalue if n else float("nan")
    meta = bids_io.scan_meta(rec) or {}
    visit = meta.get("visit")
    cal = calibration_verdict(rec.sub, rec.condition, visit)
    hz = _float(meta.get("effective_sampling_hz"))
    dur = (float(pd.to_numeric(raw_main["duration"], errors="coerce").median())
           if len(raw_main) and "duration" in raw_main else float("nan"))

    reasons = []
    slow = []
    min_hz = float(getattr(C, "BLOCK_RULE_MIN_SAMPLING_HZ", 30.0))
    max_dur = float(getattr(C, "BLOCK_RULE_MAX_STIM_FACTOR", 2.0)) * C.STIM_DURATION_S
    if hz < min_hz:
        slow.append(f"{hz:.1f} Hz effective sampling (< {min_hz:g} Hz)")
    if dur > max_dur:
        slow.append(f"median stimulus on screen {dur:.3f} s (> {max_dur:g} s)")
    if slow:
        reasons.append("frame rate: " + "; ".join(slow))
    max_to = float(getattr(C, "BLOCK_RULE_MAX_TIMEOUT_FRACTION", 0.25))
    n_timeout = n_main - n
    if n_main and n_timeout / n_main >= max_to:
        reasons.append(f"timeouts: {n_timeout}/{n_main} unanswered "
                       f"({n_timeout / n_main:.0%} >= {max_to:.0%})")
    if n and p_below < alpha:
        reasons.append(f"reversal: accuracy {k}/{n} answered significantly BELOW chance "
                       f"(p={p_below:.2g}) - reversed answers or invalid stimulus")
    return {
        "recording": rec.label(), "participant": rec.sub, "condition": rec.condition,
        "visit": visit if visit is not None else "n/a",
        "effective_sampling_hz": round(hz, 2) if hz == hz else "n/a",
        "median_stim_duration_s": round(dur, 4) if dur == dur else "n/a",
        "n_trials": n_main, "n_presentation_flagged": n_flagged,
        "n_answered": n, "n_correct": k,
        "accuracy": round(k / n, 3) if n else "n/a",
        "p_above_chance": round(float(p_above), 6) if n else "n/a",
        "p_below_chance": round(float(p_below), 6) if n else "n/a",
        "calibration_verdict": cal,
        "exclude": bool(reasons),
        "technical_exclude": bool(slow),
        "reason": "; ".join(reasons) or "pass",
        "calibration_flagged": str(cal).startswith("UNRELIABLE"),
    }


def pool_chance(rows: list[dict], manual: dict | None = None,
                alpha: float | None = None) -> None:
    """Criterion 4, in place on the block rows from ``evaluate``.

    Per participant and condition, pools the answered main trials of the blocks that pass
    the block-level criteria and are not on the manual list, and tests that accuracy
    against chance (one-sided binomial). If it is not above chance, every one of those
    blocks fails with a ``chance`` reason. Each row gets the pooled numbers
    (``chance_n_answered``, ``chance_n_correct``, ``chance_p_above``; n/a for blocks
    outside the pool).
    """
    alpha = C.BLOCK_RULE_ALPHA if alpha is None else alpha
    manual = manual or {}
    pools: dict[tuple, list[dict]] = {}
    for v in rows:
        v["chance_n_answered"] = v["chance_n_correct"] = v["chance_p_above"] = "n/a"
        if v["recording"] in manual or v["rule_exclude"]:
            continue
        pools.setdefault((v["participant"], v["condition"]), []).append(v)
    for (sub, cond), blocks in pools.items():
        n = sum(int(b["n_answered"]) for b in blocks)
        k = sum(int(b["n_correct"]) for b in blocks)
        p = binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
        for b in blocks:
            b["chance_n_answered"], b["chance_n_correct"] = n, k
            b["chance_p_above"] = round(float(p), 6) if n else "n/a"
            if not n or p >= alpha:
                b["rule"] = (f"chance: {sub} {cond} accuracy {k}/{n} answered over "
                             f"{len(blocks)} block(s) not above chance (p={p:.2g})")
                b["rule_exclude"] = True


def evaluate_calibration(path: Path) -> dict:
    """Rule verdict for one calibration table (its QUEST+ reliability)."""
    df = pd.read_csv(path, sep="\t", na_values="n/a")
    if "tracker" in df:
        df = df[~df["tracker"].astype(str).isin(["catch", "warmup"])]
    c = pd.to_numeric(df["correct"], errors="coerce") if "correct" in df else pd.Series(dtype=float)
    n, k = int(c.notna().sum()), int((c == 1).sum())
    sub = path.parts[-4]
    acq = re.search(r"_acq-([a-z]+)", path.name)
    run = re.search(r"_run-(\d+)", path.name)
    qc = (C.DERIV_DIR / "calibration" / sub / path.parts[-3]
          / f"{bids_io.calibration_label(path)}_calibration_qc.json")
    verdict = str(_load(qc).get("verdict", "n/a"))
    bad = verdict.startswith("UNRELIABLE")
    return {
        "unit": "calibration", "recording": bids_io.calibration_label(path),
        "participant": sub, "condition": acq.group(1) if acq else "calibration",
        "visit": int(run.group(1)) % 100 if run else "n/a",
        "n_trials": n, "n_correct": k, "accuracy": round(k / n, 3) if n else "n/a",
        "p_above_chance": "n/a", "p_below_chance": "n/a",
        "calibration_verdict": verdict,
        "rule": f"calibration: {verdict}" if bad else "pass",
        "rule_exclude": bad,
    }


def rules_main(quiet: bool = False) -> list[dict]:
    """Evaluate all blocks and calibrations and write ``qc/gate.tsv``.

    A sidecar ``.json`` stores the rule, its settings and a fingerprint."""
    recs = bids_io.find_recordings(include_excluded=True, apply_block_rule=False)
    cals = bids_io.find_calibration_tables()
    if not recs and not cals:
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return []
    settings = bids_io.gate_settings()
    active, manual = settings["RuleActive"], settings["Manual"]
    parts = settings.get("ManualParticipants", {})
    rows = []
    for r in recs:
        v = evaluate(r)
        rows.append({"unit": "block", **v, "rule": v.pop("reason"),
                     "rule_exclude": v.pop("exclude")})
    pool_chance(rows, manual)
    kept = [v.pop("calibration_flagged") and not v["rule_exclude"] for v in rows]
    rows += [evaluate_calibration(c) for c in cals]
    for v in rows:
        man = manual.get(v["recording"]) or parts.get(v["participant"])
        rule_out = active and v["rule_exclude"]
        tech_out = active and bool(v.get("technical_exclude"))
        v["manual"] = man or "-"
        v["behaviour"] = "exclude" if (man or rule_out) else "include"
        v["gait"] = ("n/a" if v["unit"] == "calibration"
                     else "exclude" if (man or tech_out) else "include")
        why = ([f"manual: {man.rstrip('.')}"] if man else []) + ([v["rule"]] if rule_out else [])
        v["reason"] = "; ".join(why) or "include"

    out = bids_io.deriv_dir("qc")
    for old in ("block_exclusions.tsv", "block_exclusions.json"):   # legacy files
        (out / old).unlink(missing_ok=True)
    df = pd.DataFrame(rows)
    gid = hashlib.sha1(df[["recording", "behaviour", "gait"]].to_csv(index=False)
                       .encode()).hexdigest()[:10]
    df.to_csv(out / bids_io.GATE_FILE, sep="\t", index=False, na_rep="n/a")
    (out / bids_io.GATE_FILE.replace(".tsv", ".json")).write_text(json.dumps({
        "Description": "Exclusion gate (Analysis/exclusion.py): one row per block and "
                       "per calibration, with the decision for each analysis scope. Every "
                       "analysis stage reads it; derivatives/qc/gate_ledger.tsv logs the "
                       "N each stage was left with.",
        "Settings": settings,
        "GateId": gid,
        "Scopes": {
            "behaviour": "analyses of the responses: excluded = manual list, plus the "
                         "automatic rule while RuleActive",
            "gait": "analyses of the walking alone: excluded = manual list, plus the "
                    "frame-rate criterion while RuleActive",
        },
        "Criteria": {
            "frame rate": "per block: effective sampling below BLOCK_RULE_MIN_SAMPLING_HZ, "
                          "or median stimulus on-screen time above "
                          "BLOCK_RULE_MAX_STIM_FACTOR x STIM_DURATION_S; both scopes",
            "reversal": "per block: accuracy over the answered main trials significantly "
                        "below 0.5, one-sided binomial",
            "timeouts": "per block: at least BLOCK_RULE_MAX_TIMEOUT_FRACTION of the main "
                        "trials unanswered",
            "chance": "per participant x condition: accuracy over the answered main trials "
                      "(timeouts left out) of the blocks passing the per-block criteria "
                      "and not on the manual list, pooled, not above 0.5, one-sided "
                      "binomial; all those blocks fail",
            "calibration": "an UNRELIABLE calibration is left out of the pooled "
                           "calibration refit; its blocks are judged by the other "
                           "criteria alone (calibration_verdict records it)",
            "presentation": "reversal, timeouts and chance count only the main trials the "
                            "presentation rule keeps (exclusion.py); trials shown "
                            "during a head-tracking artefact are not counted",
        },
        "unit": {"Description": "block (a motion recording) or calibration (a _beh.tsv)."},
        "effective_sampling_hz": {"Description": "Blocks: logged frames per second "
                                                 "(scans.tsv)."},
        "median_stim_duration_s": {"Description": "Blocks: median measured on-screen time "
                                                  "of the main trials' stimuli."},
        "technical_exclude": {"Description": "True if the block fails the frame-rate "
                                             "criterion, which applies to both scopes."},
        "chance_n_answered": {"Description": "Answered main trials pooled for the chance "
                                             "criterion over the participant's blocks in "
                                             "this condition; n/a outside the pool."},
        "chance_n_correct": {"Description": "Correct trials of that pool."},
        "chance_p_above": {"Description": "One-sided binomial p of that pool against 0.5."},
        "p_above_chance": {"Description": "The block's own one-sided binomial p against 0.5 "
                                          "(descriptive; chance is judged on the pool)."},
        "n_trials": {"Description": "Blocks: main trials with a valid presentation. "
                                    "Calibrations: trials scored (warm-up and catch out)."},
        "n_presentation_flagged": {"Description": "Main trials the presentation rule "
                                                  "removed (0 while it is off)."},
        "rule": {"Description": "Which criteria failed, with the numbers; 'pass' if none. "
                                "Recorded whether or not the rule is active."},
        "rule_exclude": {"Description": "True if the automatic rule fails this unit."},
        "manual": {"Description": "Reason from config.EXCLUDED_RECORDINGS, or the "
                              "participant's 'not useful' rating "
                              "(config.PARTICIPANT_RATINGS), or '-'."},
        "behaviour": {"Description": "include / exclude for the behaviour scope."},
        "gait": {"Description": "include / exclude for the gait scope (n/a: calibration)."},
        "reason": {"Description": "Why the unit is excluded from the behaviour scope (a "
                                  "frame-rate failure or a manual exclusion also removes it "
                                  "from the gait scope)."},
    }, indent=2), encoding="utf-8")

    blocks = [v for v in rows if v["unit"] == "block"]
    calib = [v for v in rows if v["unit"] == "calibration"]
    bx = [v for v in blocks if v["behaviour"] == "exclude"]
    gx = [v for v in blocks if v["gait"] == "exclude"]
    cx = [v for v in calib if v["behaviour"] == "exclude"]
    subs = sorted({v["participant"] for v in blocks})
    subs_b = [s for s in subs if all(v["behaviour"] == "exclude"
                                     for v in blocks if v["participant"] == s)]
    state = ("rule ACTIVE" if active else
             "rule NOT ACTIVE: verdicts recorded, only the manual list excludes")
    print(f"Gate {gid} ({state}):")
    print(f"  behaviour   : {len(blocks) - len(bx)}/{len(blocks)} blocks, "
          f"{len(subs) - len(subs_b)}/{len(subs)} participants"
          + (f" (out entirely: {', '.join(subs_b)})" if subs_b else ""))
    print(f"  gait        : {len(blocks) - len(gx)}/{len(blocks)} blocks")
    print(f"  calibrations: {len(calib) - len(cx)}/{len(calib)} enter the pooled refit")
    if not quiet:
        for v, k in zip(rows, kept + [False] * len(calib)):
            if v["rule_exclude"] or v["manual"] != "-":
                tag = ("EXCLUDED" if v["behaviour"] == "exclude"
                       else "flagged, rule not active")
                why = v["reason"] if v["reason"] != "include" else v["rule"]
                print(f"  ! {v['recording']} [{tag}]: {why}")
            elif k:
                print(f"  ~ {v['recording']} [kept, block at {v['accuracy']:.0%}]: "
                      f"calibration {v['calibration_verdict']}")
        print(f"Gate -> {out / bids_io.GATE_FILE}")
    return rows


# ###########################################################################
# With and without the presentation rule (on demand)
# ###########################################################################
OUT = ("sensitivity", "presentation_rule")
SKIP = ("qc/", "gait/", "gait_", "head_stability", "step_table", "trial_table",
        "sensitivity/")
_PKEY = re.compile(r"(^p$|^p_|_p$|p_value|holm|pval)", re.I)

# (label, file, dotted key) of the headline results.
KEY_RESULTS = [
    *[(f"Confirmatory population test, {m} {f}: p", "population/population_agnostic.json",
       f"{k}.by_freq.{f}.p_value")
      for m, k in (("accuracy", "accuracy"), ("RT", "reaction_time"),
                   ("timeout rate", "timeouts")) for f in ("2cps", "4cps")],
    *[(f"Common-phase test, accuracy {f}: p", "population/population_common_phase.json",
       f"measures.accuracy.by_freq.{f}.p_value") for f in ("2cps", "4cps")],
    *[(f"Common-phase walking - standing, {f}: Holm p",
       "population/population_common_phase.json",
       f"locomotion_interaction.{f}.difference.p_holm") for f in ("2cps", "4cps")],
    ("H5 threshold walking/standing ratio", "conditions/threshold_by_condition.json",
     "crossover.effects.condition_walking_minus_standing.ratio"),
    ("H5 threshold walking vs standing: p", "conditions/threshold_by_condition.json",
     "crossover.effects.condition_walking_minus_standing.p"),
    ("Calibrations outside their sampling range", "calibration_transfer/calibration_transfer.json",
     "outside_sampling_range"),
    ("Calibration error factor", "calibration_transfer/calibration_transfer.json",
     "calibration_error_factor"),
    *[(f"Bayesian GLMM, walking {f}: BF10 shared phase", "bayes_glmm/bayes_glmm.json",
       f"accuracy.by_frequency.{f}.walk.bf10_common") for f in ("2cps", "4cps")],
    *[(f"Bayesian GLMM, walking {f}: BF10 per-person", "bayes_glmm/bayes_glmm.json",
       f"accuracy.by_frequency.{f}.walk.bf10_individual") for f in ("2cps", "4cps")],
    ("Bayesian GLMM, walking main effect (pp)", "bayes_glmm/bayes_glmm.json",
     "accuracy.walking_main_effect.pp"),
]


def _flat(o, p: str = ""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _flat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _flat(v, f"{p}[{i}]")
    else:
        yield p, o


def _json(path: Path) -> dict:
    try:
        return dict(_flat(json.loads(path.read_text(encoding="utf-8"))))
    except (OSError, ValueError):
        return {}


def _num(v) -> float:
    ok = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
    return float(v) if ok else np.nan


# ===========================================================================
# The rule-off run
# ===========================================================================
def run_rule_off(dst: Path) -> Path:
    """Copy the code and the dataset (no derivatives) to ``dst`` and run the pipeline
    there with the presentation rule off. Returns the copy's derivatives folder."""
    root = Path(C.PROJECT_ROOT)
    shutil.copytree(root / "Analysis", dst / "Analysis",
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    bids = Path(C.BIDS_DIR)
    shutil.copytree(bids, dst / bids.relative_to(root),
                    ignore=lambda d, names: ["derivatives"] if Path(d) == bids else [])
    log = dst / "pipeline_rule_off.log"
    env = {**os.environ, "ILL_PRESENTATION_RULE": "0", "PYTHONIOENCODING": "utf-8"}
    print(f"Running the pipeline with the presentation rule OFF in {dst} (log: {log}) ...")
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", "-m", "Analysis.run_pipeline", "--skip-bids"],
                           cwd=dst, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"rule-off pipeline failed (exit {r.returncode}); see {log}")
    return dst / bids.relative_to(root) / "derivatives"


# ===========================================================================
# Comparison
# ===========================================================================
def flagged_trials(on: Path, off: Path) -> pd.DataFrame:
    """Per block: flagged main trials by kind and the accuracy of kept vs flagged trials."""
    flags = pd.read_csv(on / "qc" / FLAGS_FILE, sep="\t",
                        dtype={"presentation_flag": str}, keep_default_na=False)
    g_on = pd.read_csv(on / "qc" / bids_io.GATE_FILE, sep="\t", dtype=str,
                       keep_default_na=False).set_index("recording")
    g_off = pd.read_csv(off / "qc" / bids_io.GATE_FILE, sep="\t", dtype=str,
                        keep_default_na=False).set_index("recording")
    rows = []
    for rec in bids_io.find_recordings(include_excluded=True):
        ev = bids_io.load_events(rec, presentation_rule=False)
        f = flags[flags["recording"] == rec.label()]
        if ev.empty or len(f) != len(ev):
            continue
        is_main = (ev["is_training"] == 0).to_numpy()
        valid = f["presentation_valid"].to_numpy(int) == 1
        corr = pd.to_numeric(ev["response_correct"], errors="coerce").to_numpy(float)
        ans = np.isfinite(corr)
        kept, bad = is_main & valid & ans, is_main & ~valid & ans
        kinds = f["presentation_flag"].to_numpy(str)
        rows.append({
            "recording": rec.label(), "participant": rec.sub, "condition": rec.condition,
            "n_main": int(is_main.sum()), "n_flagged": int((is_main & ~valid).sum()),
            **{k: int(sum(k in s for s in kinds[is_main & ~valid])) for k in REASONS},
            "n_kept_answered": int(kept.sum()), "n_flagged_answered": int(bad.sum()),
            "n_flagged_correct": int(np.nansum(corr[bad])),
            "accuracy_kept": round(float(np.nanmean(corr[kept])), 4) if kept.any() else np.nan,
            "accuracy_flagged": round(float(np.nanmean(corr[bad])), 4) if bad.any() else np.nan,
            "behaviour_rule_on": g_on.at[rec.label(), "behaviour"] if rec.label() in g_on.index else "n/a",
            "behaviour_rule_off": g_off.at[rec.label(), "behaviour"] if rec.label() in g_off.index else "n/a",
        })
    return pd.DataFrame(rows)


def flagged_accuracy(ft: pd.DataFrame) -> pd.DataFrame:
    """Flagged trials against the kept trials of the same blocks, per condition."""
    rows = []
    for scope, d in (("blocks kept by the gate", ft[ft["behaviour_rule_on"] == "include"]),
                     ("all blocks", ft)):
        d = d[d["n_flagged_answered"] > 0]
        for cond, g in [("all", d)] + list(d.groupby("condition")):
            n, k = g["n_flagged_answered"].sum(), g["n_flagged_correct"].sum()
            p = g["accuracy_kept"].fillna(0.5)
            exp = float((g["n_flagged_answered"] * p).sum())
            var = float((g["n_flagged_answered"] * p * (1 - p)).sum())
            z = (k - exp) / np.sqrt(var) if var > 0 else np.nan
            rows.append({"blocks": scope, "condition": cond, "n_blocks": len(g),
                         "n_flagged_answered": int(n), "n_flagged_correct": int(k),
                         "accuracy_flagged": round(k / n, 4) if n else np.nan,
                         "accuracy_expected": round(exp / n, 4) if n else np.nan,
                         "z": round(z, 3) if np.isfinite(z) else np.nan,
                         "p_two_sided": round(float(2 * norm.sf(abs(z))), 4)
                         if np.isfinite(z) else np.nan})
    return pd.DataFrame(rows)


def gate_changes(on: Path, off: Path) -> pd.DataFrame:
    """Blocks whose behaviour-scope decision differs between the runs."""
    cols = ["recording", "behaviour", "n_trials", "n_answered", "accuracy",
            "p_above_chance", "rule"]
    a = pd.read_csv(on / "qc" / bids_io.GATE_FILE, sep="\t", dtype=str, keep_default_na=False)
    b = pd.read_csv(off / "qc" / bids_io.GATE_FILE, sep="\t", dtype=str, keep_default_na=False)
    m = (a[a["unit"] == "block"][cols]
         .merge(b[b["unit"] == "block"][cols], on="recording", suffixes=("_rule_on", "_rule_off")))
    return m[m["behaviour_rule_on"] != m["behaviour_rule_off"]].reset_index(drop=True)


def key_results(on: Path, off: Path) -> pd.DataFrame:
    rows = []
    for label, f, key in KEY_RESULTS:
        a, b = _json(on / f).get(key), _json(off / f).get(key)
        rows.append({"result": label, "rule_on": _num(a), "rule_off": _num(b),
                     "source": f"{f}: {key}"})
    for label, col in (("Blocks kept (behaviour)", "blocks_kept"),
                       ("Trials kept (answered, valid)", "trials_kept"),
                       ("Accuracy on kept trials", "accuracy_kept")):
        vals = []
        for d in (on, off):
            s = pd.read_csv(d / "summary" / "data_quality_summary.tsv", sep="\t")
            vals.append(_num(pd.to_numeric(s.loc[s["level"] == "total", col]).iloc[0]))
        rows.append({"result": label, "rule_on": vals[0], "rule_off": vals[1],
                     "source": f"summary/data_quality_summary.tsv: {col}"})
    return pd.DataFrame(rows)


def pvalue_changes(on: Path, off: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """p-values below 0.10 in either run, and significance flags that differ."""
    ps, flips = [], []
    for f in sorted(on.rglob("*.json")):
        rel = f.relative_to(on).as_posix()
        if rel.startswith(SKIP) or f.name == "dataset_description.json":
            continue
        a, b = _json(f), _json(off / rel)
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k), b.get(k)
            last = k.rsplit(".", 1)[-1]
            if "signific" in last.lower() and (isinstance(x, bool) or isinstance(y, bool)):
                if x != y:
                    flips.append({"file": rel, "key": k, "rule_on": x, "rule_off": y})
            elif _PKEY.search(last):
                xa, yb = _num(x), _num(y)
                if np.isfinite(xa) and np.isfinite(yb) and min(xa, yb) < 0.10:
                    ps.append({"file": rel, "key": k, "rule_on": xa, "rule_off": yb,
                               "crosses_0.05": (xa < 0.05) != (yb < 0.05)})
    return pd.DataFrame(ps), pd.DataFrame(flips)


def compare(off: Path, on: Path | None = None) -> dict:
    """Compare the rule-on derivatives (``on``) with the rule-off ones and write the tables."""
    on = Path(on or C.DERIV_DIR)
    out = bids_io.deriv_dir(*OUT[:1]) / OUT[1]
    out.mkdir(parents=True, exist_ok=True)
    ft = flagged_trials(on, off)
    fa = flagged_accuracy(ft)
    gc = gate_changes(on, off)
    kr = key_results(on, off)
    ps, flips = pvalue_changes(on, off)
    for name, df in (("flagged_trials", ft), ("flagged_accuracy", fa), ("gate_changes", gc),
                     ("key_results", kr), ("pvalues", ps), ("significance_flips", flips)):
        df.to_csv(out / f"{name}.tsv", sep="\t", index=False, na_rep="n/a")
    meta = json.loads((on / "qc" / FLAGS_FILE).with_suffix(".json")
                      .read_text(encoding="utf-8"))
    summary = {
        "Description": "The pipeline with the presentation rule on (the results) against "
                       "off (exclusion.py compare).",
        "Date": dt.date.today().isoformat(),
        "Dataset": {"participants": int(ft["participant"].nunique()), "blocks": len(ft),
                    "main_trials": int(ft["n_main"].sum())},
        "PresentationSettings": meta.get("Settings"),
        "FlaggedMainTrials": meta.get("Summary"),
        "GateChanges": gc[["recording", "behaviour_rule_on", "behaviour_rule_off"]]
        .to_dict("records"),
        "SignificanceFlips": len(flips),
        "PValuesCrossing0.05": int(ps["crosses_0.05"].sum()) if len(ps) else 0,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=2, default=str),
                                         encoding="utf-8")
    pd.set_option("display.width", 200)
    print(f"Flagged: {summary['FlaggedMainTrials']}")
    print("\nFlagged vs kept accuracy:\n" + fa.to_string(index=False))
    print("\nGate changes:\n" + (gc.to_string(index=False) if len(gc) else "none"))
    print("\nKey results:\n" + kr.drop(columns="source").to_string(index=False))
    print(f"\n{len(flips)} significance flags differ; {summary['PValuesCrossing0.05']} "
          "p-values cross 0.05")
    print(f"Comparison -> {out}")
    return summary


def comparison_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reuse", type=Path, help="an existing rule-off copy (its project root)")
    ap.add_argument("--keep", action="store_true", help="keep the rule-off copy")
    args = ap.parse_args()
    if not C.APPLY_PRESENTATION_RULE:
        sys.exit("Run this with the presentation rule on: the main derivatives are the "
                 "rule-on side of the comparison.")
    if args.reuse:
        compare(args.reuse / Path(C.BIDS_DIR).relative_to(C.PROJECT_ROOT) / "derivatives")
        return
    dst = Path(tempfile.mkdtemp(prefix="ill_presentation_off_"))
    try:
        compare(run_rule_off(dst))
    finally:
        if args.keep:
            print(f"Rule-off copy kept at {dst}")
        else:
            shutil.rmtree(dst, ignore_errors=True)


# ###########################################################################
# Pipeline stage
# ###########################################################################
def main(quiet: bool = False) -> None:
    """Pipeline stage ``gate``: the presentation flags first (the block rule is judged on
    the trials they keep), then the gate file."""
    presentation_main(quiet=quiet)
    rules_main(quiet=quiet)


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        sys.argv.pop(1)
        comparison_main()
    else:
        main()
