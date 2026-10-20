"""Every step of every walking block in one long table, for analyses outside the pipeline.

gait.py finds the steps (double-support troughs of the head bob) and labels the stance
foot, but keeps them per block in .npz arrays. This stage writes them as one table, for
questions such as step-time asymmetry between the feet and its stability across blocks
and visits:

    derivatives/step_table/steps.tsv    one row per step, every walking block
    derivatives/step_table/steps.json   column descriptions and the settings used

A step is the interval between two successive troughs. Steps longer than
``MAX_STEP_RATIO`` x the block's median span a data gap or a missed trough; they stay in
the table with ``valid_step`` = 0 and no sway or bob measures, so step numbers do not
shift. Filtering on ``valid_step`` reproduces gait.py's own step statistics.

Feet are "foot A"/"foot B" until ``config.LATERAL_SIGN_VERIFIED`` is set, and n/a in
blocks whose foot labels gait.py rated unreliable or unusable. ``(recording, step)``
joins ``trial_table/trials.tsv``.

Usage::

    python -m Analysis.step_table
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
    from . import trial_table as tt
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import gait as gait_mod
    import trial_table as tt

TABLE_FILE = "steps.tsv"

COLUMNS = {
    "participant": tt.COLUMNS["participant"],
    "visit": tt.COLUMNS["visit"],
    "run": tt.COLUMNS["run"],
    "recording": {"Description": "Recording label of the walking block; joins "
                                 "trial_table/trials.tsv."},
    "block_in_visit": tt.COLUMNS["block_in_visit"],
    "acq_time": tt.COLUMNS["acq_time"],
    "step": {"Description": "Step number within the block (1 = first trough-to-trough "
                            "interval); joins trial_table/trials.tsv on (recording, step)."},
    "t_start_s": {"Description": "Trough that starts the step, from the start of the "
                                 "recording (the clock of the events' onsets).", "Units": "s"},
    "duration_s": {"Description": "Step duration, trough to trough.", "Units": "s"},
    "valid_step": {"Description": "1 = duration at most MAX_STEP_RATIO x the block's median "
                                  "step; 0 = the interval spans a data gap or a missed trough "
                                  "and is not one step."},
    "stance_foot": {"Description": "Foot in stance during the step ('foot A'/'foot B' until "
                                   "the rig's sway sign is verified, then 'right'/'left'); "
                                   "n/a when the block's foot labels are unreliable."},
    "foot_margin": {"Description": "Signed evidence for the label, in robust-SD units (> 0 "
                                   "favours foot A); lateral sway, fused with head roll when "
                                   "the two agree."},
    "ambiguous_foot": {"Description": "1 = |foot_margin| below gait.AMBIGUOUS_MARGIN: the "
                                      "step carries little evidence of its own and its label "
                                      "comes mostly from the alternation constraint."},
    "stride": {"Description": "Stride number: a stride starts at each foot-A step, 0 = steps "
                              "before the first one. n/a when the foot labels are unreliable."},
    "lateral_sway_m": {"Description": "Mean band-passed lateral head position over the step "
                                      "(signed: the side the head leans to).", "Units": "m"},
    "lateral_range_m": {"Description": "Peak-to-peak band-passed lateral head position within "
                                       "the step.", "Units": "m"},
    "vertical_range_m": {"Description": "Peak-to-peak band-passed vertical head position "
                                        "within the step (the bob).", "Units": "m"},
    "foot_label_verdict": tt.COLUMNS["foot_label_verdict"],
    "gait": tt.COLUMNS["gait"],
    "exclusion_reason": tt.COLUMNS["exclusion_reason"],
}


# ===========================================================================
# Building blocks
# ===========================================================================
def _step_ranges(tg: np.ndarray, sig: np.ndarray, troughs: np.ndarray) -> np.ndarray:
    """Peak-to-peak of a band-passed channel within each step (NaN if empty)."""
    idx = np.searchsorted(tg, troughs)
    out = np.full(max(0, len(troughs) - 1), np.nan)
    for i in range(out.size):
        seg = sig[idx[i]:min(idx[i + 1], sig.size)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.max() - seg.min()
    return out


def steps_frame(tg: np.ndarray, x_filt: np.ndarray | None, y_filt: np.ndarray | None,
                troughs: np.ndarray, step_foot: np.ndarray | None = None,
                step_margin: np.ndarray | None = None, *,
                trust_feet: bool = True) -> pd.DataFrame:
    """The steps of one block from its gait arrays, one row per step.

    Uses the trough times and labels exactly as ``gait.py`` saved them; the sway and bob
    measures are read off the same band-passed signals the labels were decoded from.
    """
    troughs = np.asarray(troughs, float)
    n = max(0, troughs.size - 1)
    dur = np.diff(troughs)
    valid = dur <= gait_mod.MAX_STEP_RATIO * np.median(dur) if n else np.zeros(0, bool)
    nan = np.full(n, np.nan)

    def _measure(fn, sig):
        if sig is None or not np.size(sig):
            return nan
        return np.where(valid, fn(np.asarray(tg, float), np.asarray(sig, float), troughs), np.nan)

    sway = _measure(gait_mod._step_means, x_filt)
    lat_range = _measure(_step_ranges, x_filt)
    vert_range = _measure(_step_ranges, y_filt)

    foot = np.full(n, None, dtype=object)
    margin, ambiguous, stride = nan, np.full(n, np.nan), np.full(n, np.nan)
    if trust_feet and step_foot is not None and np.size(step_foot) == n:
        sf = np.asarray(step_foot, int)
        foot = np.array([gait_mod.foot_name(f) for f in sf], dtype=object)
        stride = np.cumsum(sf == 0).astype(float)
        if step_margin is not None and np.size(step_margin) == n:
            margin = np.asarray(step_margin, float)
            ambiguous = (np.abs(margin) < gait_mod.AMBIGUOUS_MARGIN).astype(float)

    return pd.DataFrame({
        "step": np.arange(1, n + 1), "t_start_s": np.round(troughs[:-1], 3),
        "duration_s": np.round(dur, 4), "valid_step": valid.astype(int),
        "stance_foot": foot, "foot_margin": np.round(margin, 3),
        "ambiguous_foot": pd.array(ambiguous, dtype="Int64"),
        "stride": pd.array(stride, dtype="Int64"),
        "lateral_sway_m": np.round(sway, 5), "lateral_range_m": np.round(lat_range, 5),
        "vertical_range_m": np.round(vert_range, 5),
    })


def block_steps(rec, sched: dict, gate_row: dict, gait_ok: bool) -> pd.DataFrame | None:
    """Every step of one walking block as rows of the table, or None without gait output."""
    g, verdict = tt._gait_arrays(rec, gait_ok)
    if g is None or np.size(g["troughs"]) < 2:
        return None
    df = steps_frame(g["t"], g.get("x_filt"), g.get("y_filt"), g["troughs"],
                     g.get("step_foot"), g.get("step_margin"),
                     trust_feet=not verdict.startswith(tt.UNTRUSTED_FOOT))
    head = {"participant": rec.sub, "visit": sched.get("visit"),
            "run": int(rec.run) if rec.run else None, "recording": rec.label(),
            "block_in_visit": sched.get("block_in_visit"), "acq_time": sched.get("acq_time")}
    for k, v in reversed(head.items()):
        df.insert(0, k, v)
    df["foot_label_verdict"] = verdict
    df["gait"] = gate_row.get("gait")
    df["exclusion_reason"] = gate_row.get("reason")
    return df


# ===========================================================================
# Stage
# ===========================================================================
def _sidecar(n_rows: int) -> dict:
    return {
        "Description": "Every step of every walking block, one row each, with the stance "
                       "foot and the head sway and bob over the step (Analysis/"
                       "step_table.py). Blocks the gait scope excludes are listed when gait "
                       "output exists for them; filter on 'gait'.",
        "Settings": {"MaxStepRatio": gait_mod.MAX_STEP_RATIO,
                     "AmbiguousMargin": gait_mod.AMBIGUOUS_MARGIN,
                     "FootLabelsAnatomical": bool(getattr(C, "LATERAL_SIGN_VERIFIED", False)),
                     "GateId": bids_io.gate_id(), "Rows": n_rows},
        **COLUMNS,
    }


def main() -> pd.DataFrame:
    """Write derivatives/step_table/steps.tsv (+ .json) from every walking block."""
    recs = [r for r in bids_io.find_recordings(include_excluded=True) if r.condition == "walking"]
    gate = bids_io.load_gate()
    rows = ({r["recording"]: r for r in gate[gate["unit"] == "block"].to_dict("records")}
            if not gate.empty else {})
    gait_out = bids_io.excluded_recordings(scope="gait")
    sched = tt.block_schedule(bids_io.find_recordings(include_excluded=True))
    frames = []
    for rec in recs:
        f = block_steps(rec, sched.get(rec.label(), {}), rows.get(rec.label(), {}),
                        rec.label() not in gait_out)
        if f is None:
            print(f"  {rec.label()}: no gait output - not in the table")
            continue
        frames.append(f)
        print(f"  {rec.label()}: {len(f)} steps ({int((f['valid_step'] == 0).sum())} too long "
              f"to be one), median {f.loc[f['valid_step'] == 1, 'duration_s'].median():.3f} s, "
              f"foot labels {f['foot_label_verdict'].iloc[0]}")
    df = tt.concat_blocks(frames, list(COLUMNS))
    df = df.sort_values(["participant", "visit", "block_in_visit", "step"],
                        kind="stable").reset_index(drop=True)
    bids_io.gate_report("step_table", "gait", condition="walking",
                        n_analysed=df.loc[df["gait"] == "include", "participant"].nunique())
    out = bids_io.deriv_dir("step_table")
    df.to_csv(out / TABLE_FILE, sep="\t", index=False, na_rep="n/a")
    (out / TABLE_FILE).with_suffix(".json").write_text(
        json.dumps(_sidecar(len(df)), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Step table: {len(df)} steps from {df['recording'].nunique()} walking blocks -> "
          f"{out / TABLE_FILE}")
    return df


if __name__ == "__main__":
    main()
