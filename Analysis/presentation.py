"""Presentation validity: trials shown while the head tracking failed.

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

Usage::

    python -m Analysis.presentation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import bids_io
    from . import clean
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import clean
    import config as C

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
        "Description": "Presentation-validity flags (Analysis/presentation.py): one row per "
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
    main = df[df["is_training"] == 0].assign(condition=lambda d: d["recording"].map(condition))
    for cond, g in [("all", main)] + list(main.groupby("condition")):
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


def main(quiet: bool = False) -> pd.DataFrame:
    """Build the flags file (pipeline stage ``gate``, before the block rule)."""
    if not bids_io.find_recordings(include_excluded=True):
        print("No BIDS recordings found. Run Analysis/bidsify.py first.")
        return pd.DataFrame()
    return build(quiet=quiet)


if __name__ == "__main__":
    main()
