"""Data-quality overview: what was recorded, kept and excluded, with standard reasons.

Reads the exclusion gate (``derivatives/qc/gate.tsv``, built by ``block_rules.py``) and the
planned runs (``config.PLANNED_RUNS``) and sums them up per participant x condition, per
condition and overall, at block and at trial level. Every exclusion carries one label
from a fixed set, so the counts compare across participants:

  * blocks, automatic rule (``block_rules.py``): ``Chance-level accuracy``,
    ``Too many timeouts``, ``Reversed responses``;
  * blocks, manual list (``config.EXCLUDED_RECORDINGS``, labelled in
    ``config.EXCLUSION_LABELS``): ``Technical failure``, ``No valid walking``,
    ``Manual exclusion`` (fallback);
  * calibrations: ``Unreliable calibration``;
  * planned blocks without a recording: ``Not recorded``, or ``Visit pending`` when the
    visit lies after the participant's last recorded one.

A block failing several criteria gets the most specific as its label (manual, then
reversal, timeouts, chance) and all of them in ``all_labels``. Trial counts are main
trials; the training trials that open every block are never analysed and are counted
apart. Kept trials are the answered main trials of the blocks the behaviour scope keeps;
timeouts in those blocks are counted separately.

Outputs (under ``derivatives/summary/``):
  * ``data_quality.md`` — the overview tables, with the label legend.
  * ``data_quality_summary.tsv`` — the same counts, one row per participant x condition,
    per condition and in total.
  * ``data_quality_units.tsv`` — one row per block, calibration and missing planned
    block, with its label.

``report.py`` (the pipeline's last stage) writes them on every run.

Usage::

    python -m Analysis.data_quality
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

try:
    from . import bids_io
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C

MANUAL_LABELS = ("Technical failure", "No valid walking", "Manual exclusion")
LEGEND = [
    ("Chance-level accuracy", "block",
     "accuracy over the answered trials not above 50 % (one-sided binomial, "
     "config.BLOCK_RULE_ALPHA)"),
    ("Too many timeouts", "block",
     "at least config.BLOCK_RULE_MAX_TIMEOUT_FRACTION of the main trials unanswered"),
    ("Reversed responses", "block",
     "accuracy significantly below 50 %: swapped keys or an invalid stimulus"),
    ("Technical failure", "block (manual)",
     "a rig or tracking malfunction invalidated the block (operator report)"),
    ("No valid walking", "block (manual)",
     "a usable standing block of a participant without any valid walking block, left out "
     "so the participant does not enter the walking-vs-standing contrasts"),
    ("Manual exclusion", "block (manual)",
     "any other manual exclusion; the reason is in config.EXCLUDED_RECORDINGS"),
    ("Unreliable calibration", "calibration",
     "the QUEST+ fit failed its reliability check; leaves the pooled calibration refit "
     "only, not the blocks run on it"),
    ("Training trial", "trial", "the practice trials opening every block; never analysed"),
    ("Timeout", "trial", "no response in the response window; left out of accuracy"),
    ("Not recorded", "planned block", "a planned run with no recording"),
    ("Visit pending", "planned block", "a planned run of a visit not yet recorded"),
]
_RULE_LABELS = (("reversal", "Reversed responses"), ("timeouts", "Too many timeouts"),
                ("chance", "Chance-level accuracy"))


def _rule_labels(rule: str) -> list[str]:
    """Standard labels of the automatic-rule criteria a unit failed, most specific first."""
    keys = {part.split(":", 1)[0].strip() for part in str(rule).split(";")}
    return [lab for key, lab in _RULE_LABELS if key in keys]


def _int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def unit_rows(gate: pd.DataFrame, training: dict[str, int] | None = None,
              planned: dict | None = None, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per block, calibration and missing planned block, with its standard label.

    ``gate`` is the gate table as ``bids_io.load_gate`` returns it, ``training`` the
    number of training trials per block, ``planned`` and ``labels`` default to
    ``config.PLANNED_RUNS`` and ``config.EXCLUSION_LABELS``.
    """
    training = training or {}
    planned = C.PLANNED_RUNS if planned is None else planned
    labels = getattr(C, "EXCLUSION_LABELS", {}) if labels is None else labels
    rows = []
    for _, g in gate.iterrows():
        run = re.search(r"_run-(\d+)", g["recording"])
        manual = g.get("manual", "-") not in ("-", "", "n/a")
        crit = ([labels.get(g["recording"], "Manual exclusion")] if manual else [])
        if g["unit"] == "block":
            crit += _rule_labels(g.get("rule", ""))
        elif str(g.get("rule", "")).startswith("calibration"):
            crit.append("Unreliable calibration")
        excluded = g["behaviour"] == "exclude"
        n_test, n_answered = _int(g.get("n_trials")), _int(g.get("n_answered"))
        is_block = g["unit"] == "block"
        rows.append({
            "unit": g["unit"], "status": "recorded", "recording": g["recording"],
            "participant": g["participant"], "condition": g["condition"],
            "visit": _int(g.get("visit")) or "n/a",
            "run": int(run.group(1)) if run else "n/a",
            "n_training": training.get(g["recording"], "n/a") if is_block else "n/a",
            "n_trials": n_test,
            "n_answered": n_answered if is_block else n_test,
            "n_timeouts": n_test - n_answered if is_block else 0,
            "n_correct": _int(g.get("n_correct")),
            "behaviour": g["behaviour"], "gait": g.get("gait", "n/a"),
            "label": crit[0] if excluded and crit else "-",
            "all_labels": "; ".join(crit) if excluded else "-",
        })
    df = pd.DataFrame(rows)
    blocks = df[df["unit"] == "block"] if not df.empty else df
    for sub, sb in (blocks.groupby("participant") if not blocks.empty else []):
        recorded_visits = [v for v in sb["visit"] if v != "n/a"]
        last = max(recorded_visits, default=0)
        for cond, visits in planned.items():
            have = set(sb.loc[sb["condition"] == cond, "run"])
            for visit, runs in visits.items():
                for run in runs:
                    if run in have:
                        continue
                    lab = "Visit pending" if visit > last else "Not recorded"
                    rows.append({
                        "unit": "block", "status": "missing", "recording": "n/a",
                        "participant": sub, "condition": cond, "visit": visit, "run": run,
                        "n_training": "n/a", "n_trials": 0, "n_answered": 0,
                        "n_timeouts": 0, "n_correct": 0, "behaviour": "n/a", "gait": "n/a",
                        "label": lab, "all_labels": lab})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_u"] = out["unit"].map({"block": 0, "calibration": 1})
    out["_r"] = pd.to_numeric(out["run"], errors="coerce")
    out = out.sort_values(["_u", "participant", "condition", "_r"]).drop(columns=["_u", "_r"])
    return out.reset_index(drop=True)


def _count_labels(labels: pd.Series) -> str:
    c = labels[labels != "-"].value_counts()
    return "; ".join(f"{n}x {lab}" for lab, n in c.items()) or "-"


def _summarise(b: pd.DataFrame, cal: pd.DataFrame) -> dict:
    rec = b[b["status"] == "recorded"]
    kept = rec[rec["behaviour"] == "include"]
    excl = rec[rec["behaviour"] == "exclude"]
    answered = kept["n_answered"].sum()
    return {
        "blocks_recorded": len(rec), "blocks_kept": len(kept), "blocks_excluded": len(excl),
        "blocks_gait_kept": int((rec["gait"] == "include").sum()),
        "blocks_missing": int((b["status"] == "missing").sum()),
        "trials_recorded": int(rec["n_trials"].sum()),
        "trials_kept": int(answered),
        "trials_excluded_with_block": int(excl["n_trials"].sum()),
        "timeouts_in_kept_blocks": int(kept["n_timeouts"].sum()),
        "accuracy_kept": round(kept["n_correct"].sum() / answered, 3) if answered else "n/a",
        "calibrations_recorded": len(cal),
        "calibrations_kept": int((cal["behaviour"] == "include").sum()),
        "block_exclusions": _count_labels(excl["label"]),
        "missing": _count_labels(b.loc[b["status"] == "missing", "label"]),
    }


def summary_rows(units: pd.DataFrame) -> pd.DataFrame:
    """Counts per participant x condition, per condition and in total."""
    blocks = units[units["unit"] == "block"]
    cals = units[units["unit"] == "calibration"]
    rows = []
    for (sub, cond), b in blocks.groupby(["participant", "condition"]):
        c = cals[(cals["participant"] == sub) & (cals["condition"] == cond)]
        rows.append({"level": "participant", "participant": sub, "condition": cond,
                     **_summarise(b, c)})
    for cond, b in blocks.groupby("condition"):
        rows.append({"level": "condition", "participant": "all", "condition": cond,
                     **_summarise(b, cals[cals["condition"] == cond])})
    rows.append({"level": "total", "participant": "all", "condition": "all",
                 **_summarise(blocks, cals)})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols: dict[str, str]) -> str:
    """A GitHub-flavoured Markdown table of ``df[cols]`` under the given headers."""
    lines = ["| " + " | ".join(cols.values()) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            cells.append(f"{v:.1%}" if k == "accuracy_kept" and isinstance(v, float)
                         else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown(units: pd.DataFrame, summary: pd.DataFrame, gate_id: str = "") -> str:
    """The overview as a Markdown document."""
    counts = {
        "participant": "Participant", "condition": "Condition",
        "blocks_recorded": "Blocks", "blocks_kept": "Kept", "blocks_excluded": "Excl.",
        "trials_recorded": "Trials", "trials_kept": "Kept (answered)",
        "trials_excluded_with_block": "Excl. with block",
        "timeouts_in_kept_blocks": "Timeouts (kept blocks)",
        "accuracy_kept": "Accuracy (kept)", "block_exclusions": "Exclusion labels",
    }
    tot = summary[summary["level"] == "total"].iloc[0]
    blocks = units[(units["unit"] == "block") & (units["status"] == "recorded")]
    excl = blocks[blocks["behaviour"] == "exclude"]
    cal = units[units["unit"] == "calibration"]
    cal_x = cal[cal["behaviour"] == "exclude"]
    miss = units[units["status"] == "missing"]
    n_train = pd.to_numeric(blocks["n_training"], errors="coerce").sum()
    gait_out = blocks[blocks["gait"] == "exclude"]
    walk = blocks[blocks["condition"] == "walking"]
    walk_gait = int((walk["gait"] == "include").sum())

    md = ["# Data quality overview", "",
          "Generated by `Analysis/data_quality.py` from the exclusion gate "
          f"(`derivatives/qc/gate.tsv`{', gate ' + gate_id if gate_id else ''}) on every "
          "pipeline run; do not edit by hand. Counts are for the behaviour scope (the "
          "analyses of the responses). Trials are main trials; kept trials are the "
          "answered trials of kept blocks.", "",
          f"**Overall:** {tot['blocks_kept']}/{tot['blocks_recorded']} blocks and "
          f"{tot['trials_kept']}/{tot['trials_recorded']} trials kept, accuracy "
          f"{tot['accuracy_kept']:.1%} on the kept trials; "
          f"{tot['calibrations_kept']}/{tot['calibrations_recorded']} calibrations kept; "
          f"{int(n_train)} training trials not analysed. The gait analyses keep "
          f"{walk_gait}/{len(walk)} walking blocks: only manual exclusions apply there.", "",
          "## By participant and condition", "",
          _md_table(summary[summary["level"] == "participant"], counts), "",
          "## By condition", "",
          _md_table(summary[summary["level"] != "participant"], counts), "",
          "## Excluded blocks", ""]
    if excl.empty:
        md.append("None.")
    else:
        e = excl.assign(accuracy=[f"{c / a:.1%}" if a else "n/a"
                                  for c, a in zip(excl["n_correct"], excl["n_answered"])])
        md.append(_md_table(e, {"participant": "Participant", "condition": "Condition",
                                "visit": "Visit", "run": "Run", "accuracy": "Accuracy",
                                "n_timeouts": "Timeouts", "label": "Label",
                                "all_labels": "All criteria failed", "gait": "Gait scope"}))
    md += ["", "## Calibrations", "",
           f"{len(cal) - len(cal_x)}/{len(cal)} kept for the pooled calibration refit."]
    if not cal_x.empty:
        md += ["", _md_table(cal_x, {"participant": "Participant", "condition": "Condition",
                                     "visit": "Visit", "label": "Label"})]
    md += ["", "## Missing planned blocks", ""]
    if miss.empty:
        md.append("None.")
    else:
        m = (miss.groupby(["participant", "condition", "visit", "label"])["run"]
             .apply(lambda r: ", ".join(f"{int(x):02d}" for x in r)).reset_index())
        md.append(_md_table(m, {"participant": "Participant", "condition": "Condition",
                                "visit": "Visit", "run": "Runs", "label": "Label"}))
    if not gait_out.empty:
        md += ["", "Excluded from the gait analyses as well: "
               + ", ".join(gait_out["recording"]) + "."]
    md += ["", "## Labels", "",
           _md_table(pd.DataFrame(LEGEND, columns=["label", "unit", "meaning"]),
                     {"label": "Label", "unit": "Applies to", "meaning": "Meaning"}), ""]
    return "\n".join(md)


def _training_counts() -> dict[str, int]:
    out = {}
    for rec in bids_io.find_recordings(include_excluded=True, apply_block_rule=False):
        ev = bids_io.load_events(rec)
        if not ev.empty and "is_training" in ev:
            out[rec.label()] = int((pd.to_numeric(ev["is_training"], errors="coerce") == 1).sum())
    return out


def main(quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the overview (Markdown + two TSVs) under ``derivatives/summary/``."""
    gate = bids_io.load_gate()
    if gate.empty:
        print("No exclusion gate found. Run the pipeline first.")
        return pd.DataFrame(), pd.DataFrame()
    units = unit_rows(gate, _training_counts())
    summary = summary_rows(units)
    out = bids_io.deriv_dir("summary")
    units.to_csv(out / "data_quality_units.tsv", sep="\t", index=False, na_rep="n/a")
    summary.to_csv(out / "data_quality_summary.tsv", sep="\t", index=False, na_rep="n/a")
    (out / "data_quality.md").write_text(markdown(units, summary, bids_io.gate_id()),
                                         encoding="utf-8")
    if not quiet:
        tot = summary[summary["level"] == "total"].iloc[0]
        print(f"  data quality: {tot['blocks_kept']}/{tot['blocks_recorded']} blocks, "
              f"{tot['trials_kept']}/{tot['trials_recorded']} trials kept; "
              f"exclusions: {tot['block_exclusions']}")
        print(f"Data quality -> {out / 'data_quality.md'}")
    return units, summary


if __name__ == "__main__":
    main()
