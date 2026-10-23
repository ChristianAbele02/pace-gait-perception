"""Sensitivity analysis: the pipeline with and without the presentation rule.

Results are computed with the presentation rule on (``presentation.py``). This tool
reruns the whole pipeline with the rule off (``ILL_PRESENTATION_RULE=0``) on a temporary
copy of the project and compares the two runs, so the effect of the rule can be reported.

Run it after a full pipeline run, so ``derivatives/`` is current with the rule on::

    python -m Analysis.presentation_comparison               # rule-off run + comparison
    python -m Analysis.presentation_comparison --keep        # keep the rule-off copy
    python -m Analysis.presentation_comparison --reuse DIR   # compare against a kept copy

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
removing trials earlier redraws the null of every later block. Four exploratory stages
(``NONDETERMINISTIC``) seed per process and differ between any two runs; their rows are
marked.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
from scipy.stats import norm

try:
    from . import bids_io
    from . import config as C
    from . import presentation
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import presentation

OUT = ("sensitivity", "presentation_rule")
SKIP = ("qc/", "gait/", "gait_", "head_stability", "step_table", "trial_table", "planning/",
        "sensitivity/")
NONDETERMINISTIC = ("ddm_phase", "onset_response", "sdt", "stance_foot")
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
    ("GLMM walking - standing accuracy (descriptive)", "population/population_glmm.json",
     "condition_model.locomotion_effect.approx_acc_delta"),
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
    flags = pd.read_csv(on / "qc" / presentation.FLAGS_FILE, sep="\t",
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
        main = (ev["is_training"] == 0).to_numpy()
        valid = f["presentation_valid"].to_numpy(int) == 1
        corr = pd.to_numeric(ev["response_correct"], errors="coerce").to_numpy(float)
        ans = np.isfinite(corr)
        kept, bad = main & valid & ans, main & ~valid & ans
        kinds = f["presentation_flag"].to_numpy(str)
        rows.append({
            "recording": rec.label(), "participant": rec.sub, "condition": rec.condition,
            "n_main": int(main.sum()), "n_flagged": int((main & ~valid).sum()),
            **{k: int(sum(k in s for s in kinds[main & ~valid])) for k in presentation.REASONS},
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
        nondet = rel.startswith(NONDETERMINISTIC)
        for k in sorted(set(a) | set(b)):
            x, y = a.get(k), b.get(k)
            last = k.rsplit(".", 1)[-1]
            if "signific" in last.lower() and (isinstance(x, bool) or isinstance(y, bool)):
                if x != y:
                    flips.append({"file": rel, "key": k, "rule_on": x, "rule_off": y,
                                  "nondeterministic_stage": nondet})
            elif _PKEY.search(last):
                xa, yb = _num(x), _num(y)
                if np.isfinite(xa) and np.isfinite(yb) and min(xa, yb) < 0.10:
                    ps.append({"file": rel, "key": k, "rule_on": xa, "rule_off": yb,
                               "crosses_0.05": (xa < 0.05) != (yb < 0.05),
                               "nondeterministic_stage": nondet})
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
    meta = json.loads((on / "qc" / presentation.FLAGS_FILE).with_suffix(".json")
                      .read_text(encoding="utf-8"))
    summary = {
        "Description": "The pipeline with the presentation rule on (the results) against "
                       "off (presentation_comparison.py).",
        "Date": dt.date.today().isoformat(),
        "Dataset": {"participants": int(ft["participant"].nunique()), "blocks": len(ft),
                    "main_trials": int(ft["n_main"].sum())},
        "PresentationSettings": meta.get("Settings"),
        "FlaggedMainTrials": meta.get("Summary"),
        "GateChanges": gc[["recording", "behaviour_rule_on", "behaviour_rule_off"]]
        .to_dict("records"),
        "SignificanceFlips": {"total": len(flips),
                              "deterministic_stages": int((~flips["nondeterministic_stage"]).sum())
                              if len(flips) else 0},
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


def main() -> None:
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


if __name__ == "__main__":
    main()
