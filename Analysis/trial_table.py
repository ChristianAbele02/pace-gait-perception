"""Every trial in the dataset in one long table, for analyses outside the pipeline.

The stages assemble trial data in memory and keep only their results. This stage writes
the trial level to disk, so questions the pipeline does not ask (standing vs walking,
visit 1 vs visit 2, time on task, response bias) can be answered from one file:

    derivatives/trial_table/trials.tsv    one row per trial, every block of every visit
    derivatives/trial_table/trials.json   column descriptions and the settings used

Each row carries:

* Design: participant, visit, condition, run, the block's position in its visit
  (``block_in_visit``, from ``acq_time``), the counterbalance and the gap since the
  previous block ended.
* The trial as logged: difficulty, tilt, stimulus side, response, correctness, reaction
  time. ``timeout`` marks an unanswered trial, whose ``correct`` is n/a (accuracy is over
  answered trials). ``time_in_visit_s`` puts every onset on one clock per visit.
* ``rt_valid``: the RT filter of ``phase_analysis`` (correct, answered, slower than
  ``RT_MIN_S``, within ``RT_MAD_K`` robust SDs of the block's log RT), over the trials
  with a valid presentation.
* Walking only: the step the onset falls in (``step``, joins ``step_table/steps.tsv``),
  stride phase by ``config.PHASE_METHOD``, phase within the step, and the stance foot
  with its evidence. Onsets in a data gap, or in a step too long to be one, get n/a, as in
  ``gait.phase_at``. Feet are "foot A"/"foot B" until ``config.LATERAL_SIGN_VERIFIED`` is
  set, and n/a in blocks whose foot labels gait.py rated unreliable.
* The exclusion gate's behaviour and gait decisions with the reason, and the verdict of
  the calibration the block ran on.
* Presentation validity: ``presentation_valid`` = 0 marks a trial shown while the head
  tracking froze, jumped or dropped out (``presentation.py``), and ``presentation_flag``
  names the artefact. The pipeline drops these trials from every trial-based analysis.

Excluded blocks, training trials and trials with an invalid presentation are included
and flagged, so an analysis filters rather than rebuilds.

Usage::

    python -m Analysis.trial_table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import bids_io
    from . import config as C
    from . import gait as gait_mod
    from . import phase_analysis as pa
    from . import presentation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import gait as gait_mod
    import phase_analysis as pa
    import presentation

TABLE_FILE = "trials.tsv"
UNTRUSTED_FOOT = ("unreliable", "unusable")   # gait verdicts whose foot labels are not used

COLUMNS = {
    "participant": {"Description": "Participant label."},
    "visit": {"Description": "Visit number (from scans.tsv). Not the BIDS session: ses-01 "
                             "holds the calibrations and ses-02 the blocks of every visit."},
    "condition": {"Description": "Block condition.", "Levels": {
        "standing": "standing in the CAVE", "walking": "walking on the treadmill"}},
    "run": {"Description": "BIDS run index of the block within its condition (runs of a "
                           "re-run block are 1xx)."},
    "recording": {"Description": "Recording label of the block; names its files in every "
                                 "derivative folder and joins step_table/steps.tsv."},
    "block_in_visit": {"Description": "Position of the block among all blocks of the visit, "
                                      "both conditions, in recording order (1 = first)."},
    "condition_order_first": {"Description": "Condition run first in this visit."},
    "condition_position": {"Description": "Whether this block's condition ran first or "
                                          "second in the visit."},
    "acq_time": {"Description": "Start of the block's recording (scans.tsv)."},
    "gap_before_block_s": {"Description": "Time from the end of the previous block of the "
                                          "visit to the start of this one; includes any "
                                          "calibration run in between. n/a for the first "
                                          "block.", "Units": "s"},
    "trial": {"Description": "Trial number within the block, training trials included "
                             "(1 = first)."},
    "is_training": {"Description": "1 = training trial at the start of the block."},
    "onset_s": {"Description": "Stimulus onset from the start of the recording.", "Units": "s"},
    "time_in_visit_s": {"Description": "Stimulus onset from the start of the visit's first "
                                       "block.", "Units": "s"},
    "difficulty": {"Description": "Tilt level drawn for the trial.", "Levels": {
        "THR": "calibrated ~75 % tilt", "EASY": "~85 % tilt", "HARD": "~65 % tilt"}},
    "tilt_deg": {"Description": "Gabor tilt presented.", "Units": "degree"},
    "stimulus_side": {"Description": "Direction of the tilt.", "Levels": {
        "left": "tilted left", "right": "tilted right"}},
    "response": {"Description": "Key pressed; n/a for a timeout."},
    "timeout": {"Description": "1 = no response within the response window."},
    "correct": {"Description": "1 = response matched the tilt, 0 = it did not; n/a for a "
                               "timeout."},
    "rt_s": {"Description": "Reaction time from stimulus onset; n/a for a timeout.",
             "Units": "s"},
    "rt_valid": {"Description": "1 = the reaction time passes phase_analysis's RT rule: "
                                "answered correctly, slower than RT_MIN_S, and within "
                                "RT_MAD_K robust SDs of the block's log10 RT (main trials "
                                "with a valid presentation)."},
    "presentation_valid": {"Description": "1 = no head-tracking artefact from "
                                          "PRESENTATION_PRE_S before onset to "
                                          "PRESENTATION_POST_S after the Gabor's offset; "
                                          "0 = flagged, left out of every trial-based "
                                          "analysis (presentation.py)."},
    "presentation_flag": {"Description": "Artefact kinds in the presentation window "
                                         "(spike, jump, freeze, gap), ';'-joined; '-' if "
                                         "none."},
    "step": {"Description": "Walking: step (trough-to-trough interval) the onset falls in, "
                            "1 = first; joins step_table/steps.tsv on (recording, step). "
                            "n/a in a data gap or a step too long to be one."},
    "stride_phase_pct": {"Description": "Walking: stride phase at onset by "
                                        "config.PHASE_METHOD, 0 = a double-support trough.",
                         "Units": "%"},
    "step_phase_pct": {"Description": "Walking: position of the onset within its step.",
                       "Units": "%"},
    "stance_foot": {"Description": "Walking: foot in stance at onset ('foot A'/'foot B' "
                                   "until the rig's sway sign is verified, then "
                                   "'right'/'left'); n/a when the block's foot labels are "
                                   "unreliable."},
    "foot_margin": {"Description": "Walking: signed evidence for the stance-foot label of the "
                                   "step, in robust-SD units (> 0 favours foot A); |margin| < "
                                   "0.25 is an ambiguous step."},
    "foot_label_verdict": {"Description": "Walking: gait.py's rating of the block's foot "
                                          "labels (reliable / usable (check) / unreliable / "
                                          "unusable)."},
    "calibration_verdict": {"Description": "Verdict on the calibration the block ran on "
                                           "(qc/gate.tsv)."},
    "behaviour": {"Description": "Exclusion gate, behaviour scope (analyses of the "
                                 "responses).", "Levels": {"include": "use",
                                                           "exclude": "leave out"}},
    "gait": {"Description": "Exclusion gate, gait scope (analyses of the walking alone).",
             "Levels": {"include": "use", "exclude": "leave out"}},
    "exclusion_reason": {"Description": "Why the gate excludes the block in either scope; "
                                        "'include' if it does not."},
}


# ===========================================================================
# Building blocks
# ===========================================================================
def stance_at(times: np.ndarray, troughs: np.ndarray, step_foot: np.ndarray | None = None,
              step_margin: np.ndarray | None = None) -> dict:
    """Step number, % step, stance foot and foot evidence for each event time.

    The step is the trough-to-trough interval the time falls in, under the same validity
    rule as ``gait.phase_at`` (whose % step this reuses): a time outside the troughs, or
    inside a step longer than ``MAX_STEP_RATIO`` x the median step, has no step. So every
    trial with a stride phase has a step and vice versa.
    """
    times = np.asarray(times, float)
    troughs = np.asarray(troughs, float)
    pct_step, _ = gait_mod.phase_at(times, troughs)
    ok = np.isfinite(pct_step)
    idx = np.searchsorted(troughs, times) - 1
    step = np.where(ok, idx, -1)
    foot = np.full(times.size, -1)
    margin = np.full(times.size, np.nan)
    if step_foot is not None and np.size(step_foot):
        sf = np.asarray(step_foot, int)
        lab = ok & (idx < sf.size)
        foot[lab] = sf[idx[lab]]
        if step_margin is not None and np.size(step_margin) == sf.size:
            margin[lab] = np.asarray(step_margin, float)[idx[lab]]
    return {"step": step, "pct_step": pct_step, "foot": foot, "margin": margin}


def rt_valid(rt: np.ndarray, correct: np.ndarray, main: np.ndarray) -> np.ndarray:
    """True where a reaction time passes ``phase_analysis``'s RT rule.

    Correct and answered, slower than ``RT_MIN_S`` (faster is an anticipation), and within
    ``RT_MAD_K`` x 1.4826 MAD of the median log10 RT over the block's main trials that
    pass the first two conditions. The same rule as ``phase_analysis._clean_rt``, so an
    RT analysis on this table uses the trials the pipeline's RT tests use.
    """
    rt = np.asarray(rt, float)
    base = (np.asarray(main, bool) & (np.asarray(correct, float) == 1.0)
            & np.isfinite(rt) & (rt > pa.RT_MIN_S))
    out = np.zeros(rt.size, bool)
    if not base.any():
        return out
    lr = np.log10(rt[base])
    med = np.median(lr)
    mad = 1.4826 * np.median(np.abs(lr - med)) or 1e-9
    out[np.flatnonzero(base)[np.abs(lr - med) <= pa.RT_MAD_K * mad]] = True
    return out


def _seconds(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def block_schedule(recs, bids_dir: Path | None = None) -> dict[str, dict]:
    """Per block: its scans.tsv row, its place in the visit and the gap before it.

    Blocks are ordered within (participant, visit) by ``acq_time``. A visit with a block
    missing its ``acq_time`` falls back to the logged condition order and the run number,
    and then has no gaps (they need both clocks).
    """
    meta = {r.label(): {"rec": r, **(bids_io.scan_meta(r, bids_dir) or {})} for r in recs}
    visits: dict[tuple, list[str]] = {}
    for lab, m in meta.items():
        visits.setdefault((m["rec"].sub, m.get("visit")), []).append(lab)
    out = {}
    for labs in visits.values():
        times = {lab: pd.to_datetime(meta[lab].get("acq_time"), errors="coerce") for lab in labs}
        timed = all(pd.notna(t) for t in times.values())
        if timed:
            order = sorted(labs, key=lambda lab: times[lab])
        else:
            order = sorted(labs, key=lambda lab: (meta[lab].get("condition_position") != "first",
                                                  int(meta[lab]["rec"].run or 0)))
        start = times[order[0]] if timed else None
        for k, lab in enumerate(order):
            gap = float("nan")
            if timed and k:
                prev = order[k - 1]
                gap = ((times[lab] - times[prev]).total_seconds()
                       - _seconds(meta[prev].get("duration_s")))
            out[lab] = {**{k_: v for k_, v in meta[lab].items() if k_ != "rec"},
                        "block_in_visit": k + 1, "gap_before_block_s": gap,
                        "offset_in_visit_s": ((times[lab] - start).total_seconds()
                                              if timed else float("nan"))}
    return out


def _gait_arrays(rec, gait_ok: bool) -> tuple[dict | None, str]:
    """The block's gait .npz arrays and foot-label verdict (run gait if the file is missing
    and the gate lets the block into the gait scope)."""
    d = bids_io.deriv_dir("gait", rec.sub, rec.ses)
    npz = d / f"{rec.label()}_gait.npz"
    if not npz.exists() and gait_ok:
        gait_mod.gait_recording(rec)
    if not npz.exists():
        return None, "n/a"
    gj = d / f"{rec.label()}_gait.json"
    verdict = "unknown"
    if gj.exists():
        verdict = str(json.loads(gj.read_text(encoding="utf-8")).get("foot_label_verdict")
                      or "unknown")
    g = np.load(npz)
    return {k: g[k] for k in g.files}, verdict


def block_trials(rec, sched: dict, gate_row: dict, gait_ok: bool) -> pd.DataFrame | None:
    """Every trial of one block as rows of the table."""
    ev = bids_io.load_events(rec, presentation_rule=False)     # every trial, flag as a column
    if ev.empty:
        return None
    n = len(ev)
    onset = ev["onset"].to_numpy(float)
    training = ev["is_training"].fillna(0).to_numpy(int)
    correct = ev["response_correct"].to_numpy(float)
    rt = ev["response_time"].to_numpy(float)
    timeout = ev["response"].isna().to_numpy()
    pflags = presentation.trial_flags(rec, ev)
    pvalid = pflags["presentation_valid"].to_numpy(int)

    step = np.full(n, -1)
    stride_pct, step_pct, margin = (np.full(n, np.nan) for _ in range(3))
    foot_txt = np.full(n, None, dtype=object)
    verdict = "n/a"
    if rec.condition == "walking":
        g, verdict = _gait_arrays(rec, gait_ok)
        if g is not None and np.size(g["troughs"]) >= 2:
            sf = g["step_foot"] if np.size(g.get("step_foot", [])) else None
            st = stance_at(onset, g["troughs"], sf, g.get("step_margin"))
            step, step_pct = st["step"], st["pct_step"]
            tg = g["t"] if np.size(g.get("t", [])) else None
            stride_pct = pa._phase_of(onset, g["troughs"], sf, tg, pa.continuous_phase(g))
            if not verdict.startswith(UNTRUSTED_FOOT):
                margin = st["margin"]
                foot_txt = np.array([gait_mod.foot_name(f) if f >= 0 else None
                                     for f in st["foot"]], dtype=object)

    def _na(v):
        return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v

    return pd.DataFrame({
        "participant": rec.sub, "visit": sched.get("visit"), "condition": rec.condition,
        "run": int(rec.run) if rec.run else None, "recording": rec.label(),
        "block_in_visit": sched.get("block_in_visit"),
        "condition_order_first": sched.get("condition_order_first"),
        "condition_position": sched.get("condition_position"),
        "acq_time": sched.get("acq_time"),
        "gap_before_block_s": _na(round(sched.get("gap_before_block_s", np.nan), 1)),
        "trial": np.arange(1, n + 1), "is_training": training,
        "onset_s": np.round(onset, 3),
        "time_in_visit_s": np.round(onset + sched.get("offset_in_visit_s", np.nan), 3),
        "difficulty": ev["difficulty"].to_numpy(), "tilt_deg": ev["tilt_angle_deg"].to_numpy(),
        "stimulus_side": ev["stimulus_side"].to_numpy(),
        "response": ev["response"].to_numpy(), "timeout": timeout.astype(int),
        "correct": pd.array(np.where(timeout, np.nan, correct), dtype="Int64"),
        "rt_s": np.round(rt, 4),
        "rt_valid": rt_valid(rt, correct, (training == 0) & (pvalid == 1)).astype(int),
        "presentation_valid": pvalid,
        "presentation_flag": pflags["presentation_flag"].to_numpy(),
        "step": pd.array(np.where(step >= 0, step + 1, np.nan), dtype="Int64"),
        "stride_phase_pct": np.round(stride_pct, 2), "step_phase_pct": np.round(step_pct, 2),
        "stance_foot": foot_txt, "foot_margin": np.round(margin, 3),
        "foot_label_verdict": verdict if rec.condition == "walking" else None,
        "calibration_verdict": gate_row.get("calibration_verdict"),
        "behaviour": gate_row.get("behaviour"), "gait": gate_row.get("gait"),
        "exclusion_reason": gate_row.get("reason"),
    })


def concat_blocks(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Stack per-block frames into the table, in ``columns`` order.

    A standing block has no stride columns at all (every value n/a); those all-empty
    columns are dropped before stacking and restored after, so they cannot decide the
    dtype of a column the walking blocks fill.
    """
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat([f.dropna(axis=1, how="all") for f in frames],
                     ignore_index=True).reindex(columns=columns)


# ===========================================================================
# Stage
# ===========================================================================
def _sidecar(n_rows: int) -> dict:
    return {
        "Description": "Every trial of every block, one row each, with the design, the "
                       "stride at onset and the exclusion gate's decisions "
                       "(Analysis/trial_table.py). Excluded blocks are included; filter on "
                       "'behaviour' or 'gait'.",
        "Settings": {"PhaseMethod": getattr(C, "PHASE_METHOD", "trough"),
                     "RTMinS": pa.RT_MIN_S, "RTMadK": pa.RT_MAD_K,
                     "MaxStepRatio": gait_mod.MAX_STEP_RATIO,
                     "FootLabelsAnatomical": bool(getattr(C, "LATERAL_SIGN_VERIFIED", False)),
                     "GateId": bids_io.gate_id(), "Rows": n_rows},
        **COLUMNS,
    }


def _print(df: pd.DataFrame) -> None:
    for (sub, visit), d in df.groupby(["participant", "visit"], dropna=False):
        main = d[d["is_training"] == 0]
        walk = main[main["condition"] == "walking"]
        kept = main[main["behaviour"] == "include"]
        print(f"  {sub} visit {visit}: {d['recording'].nunique()} blocks, {len(main)} main "
              f"trials ({int(main['timeout'].sum())} timeouts, {len(kept)} in the behaviour "
              f"gate); walking {len(walk)}, {int(walk['step'].notna().sum())} with a step")


def main() -> pd.DataFrame:
    """Write derivatives/trial_table/trials.tsv (+ .json) from every block in the dataset."""
    recs = bids_io.find_recordings(include_excluded=True)
    gate = bids_io.load_gate()
    rows = ({r["recording"]: r for r in gate[gate["unit"] == "block"].to_dict("records")}
            if not gate.empty else {})
    gait_out = bids_io.excluded_recordings(scope="gait")
    sched = block_schedule(recs)
    frames = [f for rec in recs
              if (f := block_trials(rec, sched.get(rec.label(), {}), rows.get(rec.label(), {}),
                                    rec.label() not in gait_out)) is not None]
    df = concat_blocks(frames, list(COLUMNS))
    df = df.sort_values(["participant", "visit", "block_in_visit", "trial"],
                        kind="stable").reset_index(drop=True)
    bids_io.gate_report("trial_table", condition=("standing", "walking"),
                        n_analysed=df.loc[df["behaviour"] == "include", "participant"].nunique())
    out = bids_io.deriv_dir("trial_table")
    df.to_csv(out / TABLE_FILE, sep="\t", index=False, na_rep="n/a")
    (out / TABLE_FILE).with_suffix(".json").write_text(
        json.dumps(_sidecar(len(df)), indent=2, ensure_ascii=False), encoding="utf-8")
    _print(df)
    print(f"Trial table: {len(df)} trials from {df['recording'].nunique()} blocks -> "
          f"{out / TABLE_FILE}")
    return df


if __name__ == "__main__":
    main()
