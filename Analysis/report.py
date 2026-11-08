"""Dataset summary, data-quality overview and the HTML report.

Usage::

    python -m Analysis.report


Dataset summary and HTML report
===============================

Dataset summary tables and one HTML report.

Combines the per-recording derivatives (QC, gait, stride phase) and the population results
into one table and one HTML page.

Outputs (``derivatives/summary/``):
  * ``dataset_summary.tsv``: one row per recording with key metrics;
  * ``report.html``: the table, dataset-level figures embedded, per-recording figures
    linked (see ``_img_tag``);
  * ``exclusions.tsv``, ``block_inventory.tsv``, ``counterbalance_qc.tsv``,
    ``provenance.json``;
  * ``data_quality.md`` and its two TSVs: recorded / kept / excluded counts with
    standard exclusion labels, written by ``report.data_quality_main``.


Data-quality overview
=====================

Data-quality overview: what was recorded, kept and excluded, with standard reasons.

Reads the exclusion gate (``derivatives/qc/gate.tsv``, built by ``exclusion.py``) and the
planned runs (``config.PLANNED_RUNS``) and sums them up per participant x condition, per
condition and overall, at block and at trial level. Every exclusion carries one label
from a fixed set, so the counts compare across participants:

  * blocks, automatic rule (``exclusion.py``): ``Low frame rate``,
    ``Chance-level accuracy``, ``Too many timeouts``, ``Reversed responses``;
  * blocks, manual list (``config.EXCLUDED_RECORDINGS``, labelled in
    ``config.EXCLUSION_LABELS``): ``Technical failure``, ``No valid walking``,
    ``Manual exclusion`` (fallback); every block and calibration of a participant rated
    "not useful": ``Participant not useful``;
  * calibrations: ``Unreliable calibration``;
  * planned blocks without a recording: ``Not recorded``, or ``Visit pending`` when the
    visit lies after the participant's last recorded one (``Visit cancelled`` for a
    participant in ``config.VISITS_CANCELLED``).

The participant roster gives each participant a status from the gate:

  * ``not useful``: no block kept in the behaviour scope, or rated "not useful";
  * ``pending``: a planned visit is still to come;
  * ``useful``: every planned block recorded and kept;
  * ``partial``: anything else (a block lost, a visit cancelled).

Next to it stands the experimenter's own rating (``config.PARTICIPANT_RATINGS``: very good,
good, weak, not useful), which bidsify also writes to ``participants.tsv``.

A block failing several criteria gets the most specific as its label (manual, then frame
rate, reversal, timeouts, chance) and all of them in ``all_labels``. Trial counts are main
trials; the training trials that open every block are never analysed and are counted
apart. Kept trials are the answered main trials of the blocks the behaviour scope keeps,
whose presentation was valid; timeouts in those blocks, and their trials removed by the
presentation rule (``exclusion.py``, label ``Presentation artefact``), are counted
separately.

Outputs (under ``derivatives/summary/``):
  * ``data_quality.md`` — the overview tables, with the label legend.
  * ``data_quality_summary.tsv`` — the same counts, one row per participant x condition,
    per condition and in total.
  * ``data_quality_units.tsv`` — one row per block, calibration and missing planned
    block, with its label.
  * ``participant_roster.tsv`` — one row per participant: status, rating, visits, and
    blocks recorded and kept per condition.

``report.py`` (the pipeline's last stage) writes them on every run.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as _md
from pathlib import Path

import pandas as pd

try:
    from . import bids_io
    from . import config as C
    from . import exclusion
    from . import phase_analysis as pa
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bids_io
    import config as C
    import exclusion
    import phase_analysis as pa


# ###########################################################################
# Dataset summary and HTML report
# ###########################################################################
def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


# --- Provenance -------------------------------------------------------------
def _pkg_version(name: str) -> str:
    try:
        return _md.version(name)
    except Exception:
        return "?"


def _git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=str(C.PROJECT_ROOT),
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def provenance() -> dict:
    """Run metadata: git commit and dirty flag, Python and package versions, and the
    config.py constants that affect results."""
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "packages": {p: _pkg_version(p)
                     for p in ("numpy", "pandas", "scipy", "matplotlib")},
        "config": {
            "NOMINAL_FS": C.NOMINAL_FS,
            "STIM_DURATION_S": C.STIM_DURATION_S,
            "RESPONSE_WINDOW_S": C.RESPONSE_WINDOW_S,
            "PHASE_METHOD": C.PHASE_METHOD,
            "LOCAL_POS_VALID_RADIUS_M": C.LOCAL_POS_VALID_RADIUS_M,
            "MAX_PLAUSIBLE_SPEED_MPS": C.MAX_PLAUSIBLE_SPEED_MPS,
        },
    }


def _provenance_html(prov: dict) -> str:
    pkgs = ", ".join(f"{k} {v}" for k, v in prov["packages"].items())
    cfg = ", ".join(f"{k}={v}" for k, v in prov["config"].items())
    dirty = " <b>· uncommitted changes</b>" if prov["git_dirty"] else ""
    return (f"<h2>Provenance</h2><p class='note'>generated {prov['generated_utc']} &middot; "
            f"git <code>{prov['git_sha'][:10]}</code> ({prov['git_branch']}){dirty}<br>"
            f"Python {prov['python']} &middot; {pkgs}<br>config: {cfg}</p>")


# --- Inclusion / exclusion ---------------------------------------------------
# Exclusion decisions come from the gate (exclusion.py). Other issues (unusable head
# data, side bias, low L/R alternation, unreliable foot labels) are non-blocking warnings.


# --- Counterbalance QC ------------------------------------------------------
# Each participant has two visits with crossed-over condition order and a set interval.
# Both depend on the operator (`visit` field on the setup screen); an error would bring
# back the time-on-task confound, so they are checked here.
INTERSESSION_MIN_DAYS = 2      # recovery from fatigue
INTERSESSION_MAX_DAYS = 21     # beyond this, threshold drift makes pooling hard


def _counterbalance_qc() -> list[dict]:
    out = []
    by_sub: dict[str, list[dict]] = {}
    for rec in bids_io.find_recordings(include_excluded=True):
        m = bids_io.scan_meta(rec)
        if m:
            by_sub.setdefault(rec.sub, []).append(m)
    for sub, metas in sorted(by_sub.items()):
        visits = {m["visit"]: m for m in metas if m.get("visit")}
        orders = {v: m.get("condition_order_first") for v, m in visits.items()}
        times = {}
        for v, m in visits.items():
            t = m.get("acq_time")
            if t:
                try:
                    times[v] = pd.Timestamp(t)
                except (ValueError, TypeError):
                    pass
        gap = None
        if len(times) >= 2:
            gap = round(float((max(times.values()) - min(times.values())).days), 1)
        problems = []
        if len(visits) < 2:
            problems.append(f"only {len(visits)} visit(s) recorded")
        elif len(set(o for o in orders.values() if o)) < 2:
            problems.append(f"order did NOT cross over (both visits {list(orders.values())[0]}"
                            "-first) — H2 stays confounded with time-on-task")
        if gap is not None and not (INTERSESSION_MIN_DAYS <= gap <= INTERSESSION_MAX_DAYS):
            problems.append(f"inter-visit gap {gap} d outside "
                            f"[{INTERSESSION_MIN_DAYS}, {INTERSESSION_MAX_DAYS}]")
        out.append({"participant": sub, "n_visits": len(visits),
                    "order_visit1": orders.get(1, "n/a"), "order_visit2": orders.get(2, "n/a"),
                    "crossed_over": len(set(o for o in orders.values() if o)) == 2,
                    "intersession_days": gap if gap is not None else "n/a",
                    "ok": not problems, "problem": "; ".join(problems) or "ok"})
    return out


def _run_number(rec) -> int:
    return int(rec.run) if rec.run else 0


def block_inventory() -> list[dict]:
    """Planned blocks present per participant and condition (``config.PLANNED_RUNS``).

    Blocks can be missing (visit not held, aborted block, lost log) or truncated (crash,
    disconnect). Analyses pool whatever exists, but trial counts change, so this is
    listed. Runs outside the plan (re-runs under a new Trial number, or ``run-1xx`` from a
    reused Trial number) are listed as extra and analysed normally.
    """
    planned = getattr(C, "PLANNED_RUNS", {})
    # Excluded = manual list (out of all analyses). Behaviour-only exclusions are listed
    # separately.
    excluded = bids_io.excluded_recordings(scope="gait")
    beh_only = set(bids_io.excluded_recordings(scope="behaviour")) - set(excluded)
    recs_by: dict[tuple, dict[int, object]] = {}
    excl_by: dict[tuple, list[int]] = {}
    beh_by: dict[tuple, list[int]] = {}
    subs = set()
    for rec in bids_io.find_recordings(include_excluded=True):
        subs.add(rec.sub)
        if rec.label() in excluded:           # recorded but excluded from all analyses
            excl_by.setdefault((rec.sub, rec.condition), []).append(_run_number(rec))
            continue
        if rec.label() in beh_only:
            beh_by.setdefault((rec.sub, rec.condition), []).append(_run_number(rec))
        recs_by.setdefault((rec.sub, rec.condition), {})[_run_number(rec)] = rec
    calib: dict[tuple, set] = {}
    for p in bids_io.find_calibration_tables():
        sub = p.parts[-4]
        subs.add(sub)
        m = re.search(r"_acq-(walking|standing)(?:_run-(\d+))?_beh", p.name)
        if m:
            calib.setdefault((sub, m.group(1)), set()).add(int(m.group(2) or 0))
    out = []
    for sub in sorted(subs):
        for cond, visits in planned.items():
            present = recs_by.get((sub, cond), {})
            plan = sorted({r for runs in visits.values() for r in runs})
            excl = sorted(excl_by.get((sub, cond), []))
            missing = [r for r in plan if r not in present and r not in excl]
            lost_visits = [v for v, runs in visits.items()
                           if all(r in missing or r in excl for r in runs)]
            truncated, n_trials, n_phased = [], 0, 0
            for run, rec in sorted(present.items()):
                if (bids_io.scan_meta(rec) or {}).get("log_status") == "truncated":
                    truncated.append(run)
                qc = _load(bids_io.deriv_dir("qc", rec.sub, rec.ses) / f"{rec.label()}_qc.json")
                n_trials += int(qc.get("behaviour", {}).get("n_analysis_trials") or 0)
            if cond == "walking":
                ses = next(iter(present.values())).ses if present else C.EXPERIMENT_SESSION
                pooled = _load(bids_io.deriv_dir("phase", sub, ses)
                               / f"{pa.subject_walk_label(sub, ses)}_phase.json")
                n_phased = int(pooled.get("n_trials") or 0)
            no_calib = [v for v in visits if v not in calib.get((sub, cond), set())]
            status = ("none" if not present else
                      "complete" if not (missing or truncated or excl) else "incomplete")
            out.append({
                "participant": sub, "condition": cond, "status": status,
                "runs_planned": len(plan),
                "runs_present": len([r for r in present if r in plan]),
                "missing_runs": ",".join(map(str, missing)) or "-",
                "missing_visits": ",".join(map(str, lost_visits)) or "-",
                "excluded_runs": ",".join(map(str, excl)) or "-",
                "behaviour_excluded_runs": ",".join(
                    map(str, sorted(beh_by.get((sub, cond), [])))) or "-",
                "truncated_runs": ",".join(map(str, truncated)) or "-",
                "extra_runs": ",".join(str(r) for r in sorted(present) if r not in plan) or "-",
                "calibration_missing_visits": ",".join(map(str, no_calib)) or "-",
                "n_trials": n_trials,
                "n_trials_with_phase": n_phased if cond == "walking" else "n/a",
            })
    return out


def _gate_rows() -> dict[str, dict]:
    """``{unit: gate row}`` from ``qc/gate.tsv`` (rebuilt if stale)."""
    g = bids_io.load_gate()
    return {} if g.empty else {r["recording"]: r.to_dict() for _, r in g.iterrows()}


def _exclusions(rows: list[dict], gate: dict[str, dict] | None = None) -> list[dict]:
    """One row per block: gate decision per scope with reason, and non-blocking warnings."""
    out = []
    gate = _gate_rows() if gate is None else gate
    for r in rows:
        cov = r.get("clean_coverage_pct")
        warns = []
        if r.get("clean_verdict") == "severe" or (cov is not None and cov < 60):
            warns.append(f"head data mostly unusable (clean coverage {cov}%)")
        bias = r.get("response_right_bias")
        if bias is not None and abs(float(bias) - 0.5) > 0.2:
            warns.append(f"response side-bias {bias}")
        lr = r.get("lr_alternation_pct")
        if lr is not None and float(lr) < 70:
            warns.append(f"low L/R alternation {lr}%")
        fv = r.get("foot_label_verdict")
        if fv and (str(fv).startswith("unreliable") or fv == "unusable"):
            # Only response_bias.py uses foot labels and it skips such runs; the 2/4 cps test
            # does not depend on them.
            warns.append(f"stance-foot labels {fv} (stance_foot analysis skips this run)")
        out.append({
            "recording": r.get("recording"),
            "participant": r["participant"], "condition": r["condition"],
            "visit": r.get("visit"),
            "behaviour": r.get("gate_behaviour"), "gait": r.get("gate_gait"),
            "reason": gate.get(r.get("recording"), {}).get("reason", "n/a"),
            "rule_verdict": gate.get(r.get("recording"), {}).get("rule", "n/a"),
            "clean_coverage_pct": cov, "calibration_verdict": r.get("calibration_verdict"),
            "warnings": "; ".join(warns),
        })
    return out


def _ledger_html() -> str:
    """HTML table of N per stage after the gate (qc/gate_ledger.tsv)."""
    f = C.DERIV_DIR / "qc" / bids_io.GATE_LEDGER_FILE
    if not f.exists():
        return "<p><em>no stage has reported an N yet</em></p>"
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    df["current"] = df["gate_id"] == bids_io.gate_id()
    return df.to_html(index=False, na_rep="n/a", border=0, classes="summary")


def _rows():
    rows = []
    gate = _gate_rows()
    for rec in bids_io.find_recordings(include_excluded=True):
        sub, ses = rec.sub, rec.ses
        g = gate.get(rec.label(), {})
        qc = _load(bids_io.deriv_dir("qc", sub, ses) / f"{rec.label()}_qc.json")
        # a stage's result is shown only for blocks the gate lets into that stage, so an
        # output left over from an earlier run for an excluded block is never reported
        gait = (_load(bids_io.deriv_dir("gait", sub, ses) / f"{rec.label()}_gait.json")
                if g.get("gait") == "include" else {})
        phase = (_load(bids_io.deriv_dir("phase", sub, ses) / f"{rec.label()}_phase.json")
                 if g.get("behaviour") == "include" else {})
        beh = qc.get("behaviour", {})
        tim = qc.get("timing", {})
        ori = qc.get("tracking", {}).get("orientation_verdict", {})
        clean_agg = qc.get("position_cleaning", {}).get("aggregate", {})
        acc_meas = phase.get("measures", {}).get("accuracy", {})
        n_targets = phase.get("n_trials", beh.get("n_analysis_trials", 0))
        # visit, to match the block with its calibration (exclusion.calibration_verdict)
        visit = (bids_io.scan_meta(rec) or {}).get("visit")
        rows.append({
            "recording": rec.label(),
            "participant": sub, "session": ses, "condition": rec.condition,
            "visit": visit,
            "gate_behaviour": g.get("behaviour", "n/a"), "gate_gait": g.get("gait", "n/a"),
            "n_samples": tim.get("n_samples"), "fs_hz": tim.get("fs_effective_hz"),
            "dropped_frame_pct": tim.get("dropped_frame_pct"),
            "invalid_pos_pct": qc.get("tracking", {}).get("invalid_pos_pct"),
            "clean_coverage_pct": clean_agg.get("clean_coverage_pct"),
            "clean_verdict": clean_agg.get("verdict"),
            "orient_tracked_channels": len(ori.get("tracked_channels", [])),
            "n_targets": n_targets,
            "accuracy": beh.get("accuracy_overall"),
            "median_rt_s": beh.get("median_rt_s"),
            "response_right_bias": beh.get("response_right_bias"),
            "calibration_verdict": exclusion.calibration_verdict(sub, rec.condition, visit),
            "gait_freq_hz": gait.get("dominant_freq_hz"),
            "stride_dur_s": gait.get("stride_duration_s"),
            "lr_alternation_pct": gait.get("lr_alternation_pct"),
            "foot_label_verdict": gait.get("foot_label_verdict"),
            "roll_agreement_pct": gait.get("roll_agreement_pct"),
            "ambiguous_step_pct": gait.get("ambiguous_step_pct"),
            "hilbert_trough_corr": gait.get("hilbert_trough_phase_corr"),
            "best_phase_freq_cps": acc_meas.get("best_freq_cps"),
            "best_phase_r2": acc_meas.get("best_r2"),
            "phase_significant": acc_meas.get("significant"),
        })
    return rows


def _img_tag(path: Path, width=720, *, embed: bool = True) -> str:
    """One figure, base64-embedded if ``embed``, else linked by relative path.

    Embedding all per-recording figures gave a 229 MB page at 12 participants (~380 MB at
    N = 20). Dataset-level figures are embedded; per-recording figures are linked relative
    to ``derivatives/summary/``.
    """
    if not path.exists():
        return f"<p><em>missing: {path.name}</em></p>"
    if embed:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" width="{width}"><br>'
    try:
        rel = os.path.relpath(path, C.DERIV_DIR / "summary").replace("\\", "/")
    except ValueError:                                  # different drive on Windows
        rel = path.as_uri()
    return f'<a href="{rel}"><img src="{rel}" width="{width}" loading="lazy"></a><br>'


def _figures_html() -> str:
    def fig(stage: str, name: str, *parts: str) -> Path:
        return C.DERIV_DIR.joinpath(stage, *parts, "figures", name)

    blocks = []
    # one sheet per block and stage; a stage's sheet only for blocks the gate lets into it
    gate = _gate_rows()
    scope = {"qc": None, "gait": "gait", "phase": "behaviour", "clocktime": "behaviour"}
    for rec in bids_io.find_recordings(include_excluded=True):
        g = gate.get(rec.label(), {})
        present = [f for f in (fig(stage, f"{rec.label()}_{stage}.png", rec.sub, rec.ses)
                               for stage, sc in scope.items()
                               if sc is None or g.get(sc) == "include")
                   if f.exists()]
        if present:
            blocks.append(f"<h3>{rec.label()} ({rec.condition})</h3>"
                          + "".join(_img_tag(f, embed=False) for f in present))
    # Dataset-level figures grouped by role (all dataset-level figures of the pipeline).
    groups = [
        ("Confirmatory", [
            ("Phase-agnostic population test (PRIMARY)",
             fig("population", "population_agnostic.png")),
        ]),
        ("Population — secondary", [
            ("Common-phase population test (Hotelling T2)",
             fig("population", "population_common_phase.png")),
            ("Population prevalence (Fig 4)", fig("population", "population_fig4_prevalence.png")),
            ("Per-participant phase (Fig 5)", fig("population", "population_fig5_phase.png")),
        ]),
        ("Conditions and thresholds", [
            ("Standing vs walking (Fig 2)",
             fig("conditions", "conditions_standing_vs_walking.png")),
            ("Discrimination threshold, standing vs walking (H5)",
             fig("conditions", "threshold_by_condition.png")),
        ]),
        ("Controls and supplements", [
            ("Stride duration vs oscillation strength (Supp 7)",
             fig("population", "population_suppl7_stridedur_vs_fits.png")),
            ("Oscillation frequency (Hz) vs stride duration (Supp 8)",
             fig("population", "population_suppl8_hz_vs_stridedur.png")),
            ("Pooled stride-phase modulation (exploratory aggregate)",
             fig("phase", "pooled_phase.png")),
        ]),
        ("Descriptive", [
            ("Behaviour overview", fig("descriptive", "descriptive_behaviour.png")),
            ("Gait and tracking quality", fig("descriptive", "descriptive_quality.png")),
        ]),
        ("Exploratory add-ons", [
            ("Stance-foot contrast (which foot is down)",
             fig("stance_foot", "stance_foot_population.png")),
            ("Response direction by stance foot",
             fig("stance_foot", "stance_foot_response_population.png")),
            ("Sensitivity vs criterion over the stride (SDT decomposition)",
             fig("sdt", "sdt_population.png")),
            ("Head roll and the left/right response", fig("roll_bias", "roll_bias_population.png")),
            ("Button presses over the stride", fig("press_phase", "press_phase_population.png")),
            ("Group-average-curve analysis, calibrated on our design",
             fig("group_curve", "group_curve.png")),
        ]),
        ("Exploratory gait", [
            ("Head stabilisation over the stride", fig("head_stability", "head_stability_profiles.png")),
            ("Head stabilisation summary", fig("head_stability", "head_stability_summary.png")),
            ("Head kinematics over the stride", fig("head_kinematics", "head_kinematics.png")),
            ("Stride-to-stride gait dynamics", fig("gait_dynamics", "gait_dynamics.png")),
            ("Gait phase-response curve", fig("gait_prc", "gait_prc.png")),
        ]),
    ]
    ds_blocks = []
    for heading, items in groups:
        shown = [(t, f) for t, f in items if f.exists()]
        if not shown:
            continue
        ds_blocks.append(f"<h3>{heading}</h3>")
        for title, f in shown:
            ds_blocks.append(f"<h4>{title}</h4>" + _img_tag(f, width=820))
    return "\n".join(ds_blocks + blocks)


def _common_phase_summary() -> str:
    """The common-phase Hotelling result and the paired walking - standing test."""
    html = ""
    cp = _load(C.DERIV_DIR / "population" / "population_common_phase.json")
    if cp:
        html += "<h2>Common-phase population test &mdash; Hotelling T&sup2; (secondary)</h2>"
        for meas, m in (cp.get("measures") or {}).items():
            rows = []
            for fk, d in (m.get("by_freq") or {}).items():
                if not d.get("estimable"):
                    rows.append(f"{fk}: not estimable at n={d.get('n_participants')}")
                    continue
                star = " *" if d.get("significant") else ""
                rows.append(f"{fk}: T&sup2;={d['t2']:.2f}, F(2,{d['df2']})={d['f']:.2f}, "
                            f"amplitude {d['amplitude']:.4f}, "
                            f"Holm p={d['p_holm']:.4f}{star}")
            html += (f"<p><b>{meas}</b> (n={m.get('n_participants')}): "
                     + "; ".join(rows) + "</p>")
        li = cp.get("locomotion_interaction")
        if li:
            rows = []
            for fk, d in li.items():
                dd = d["difference"]
                if not dd.get("estimable"):
                    rows.append(f"{fk}: not estimable")
                    continue
                star = " *" if dd.get("significant") else ""
                verdict = f" ({dd['verdict']})" if dd.get("verdict") else ""
                rows.append(f"{fk}: F(2,{dd['df2']})={dd['f']:.2f}, "
                            f"Holm p={dd['p_holm']:.4f}{star}{verdict}")
            html += ("<p><b>Locomotion specificity</b> (paired walking &minus; standing): "
                     + "; ".join(rows) + "</p>")
        html += ("<p class='note'>Secondary. Asks whether the population oscillates at a "
                 "common phase. When preferred phases differ between participants the "
                 "vector mean cancels much of the signal, so a null here alongside a "
                 "positive phase-agnostic result is the expected pattern, not a "
                 "contradiction.</p>")
    return html


def main():
    """Write the dataset summary tables and the HTML report."""
    rows = _rows()
    if not rows:
        print("No recordings found. Run the pipeline first.")
        return
    df = pd.DataFrame(rows)
    out_dir = bids_io.deriv_dir("summary")
    df.to_csv(out_dir / "dataset_summary.tsv", sep="\t", index=False, na_rep="n/a")

    # Provenance and inclusion/exclusion table.
    prov = provenance()
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    excl = _exclusions(rows)
    edf = pd.DataFrame(excl)
    edf.to_csv(out_dir / "exclusions.tsv", sep="\t", index=False, na_rep="n/a")
    n_incl = int((edf["behaviour"] == "include").sum()) if not edf.empty else 0
    n_gait = int((edf["gait"] == "include").sum()) if not edf.empty else 0
    excl_table_html = edf.to_html(index=False, na_rep="n/a", border=0, classes="summary")

    inv = pd.DataFrame(block_inventory())
    if not inv.empty:
        inv.to_csv(out_dir / "block_inventory.tsv", sep="\t", index=False, na_rep="n/a")
    inv_html = (inv.to_html(index=False, na_rep="n/a", border=0, classes="summary")
                if not inv.empty else "<p><em>no participants</em></p>")

    _, dq = data_quality_main(quiet=True)
    dq_html = (dq.to_html(index=False, na_rep="n/a", border=0, classes="summary")
               if not dq.empty else "<p><em>no gate</em></p>")

    cbdf = pd.DataFrame(_counterbalance_qc())
    if not cbdf.empty:
        cbdf.to_csv(out_dir / "counterbalance_qc.tsv", sep="\t", index=False, na_rep="n/a")

    table_html = df.to_html(index=False, na_rep="n/a", border=0,
                            classes="summary", float_format=lambda v: f"{v:.3f}")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ILL dataset report</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; color: #222; }}
 table.summary {{ border-collapse: collapse; font-size: 12px; }}
 table.summary th, table.summary td {{ border: 1px solid #ccc; padding: 3px 6px; text-align: right; }}
 table.summary th {{ background: #f0f0f4; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 28px; }}
 h3 {{ font-size: 13px; color: #444; margin-top: 18px; }}
 img {{ border: 1px solid #ddd; margin: 4px 0; }}
 .note {{ color: #666; font-size: 12px; }}
</style></head><body>
<h1>ILL CAVE locomotion — dataset report</h1>
<p class="note">Auto-generated by <code>Analysis/report.py</code>. One row per recording.</p>
{_provenance_html(prov)}
<h2>Exclusion gate ({n_incl}/{len(edf)} blocks in the behavioural analyses, {n_gait}/{len(edf)} in the gait analyses)</h2>
<p class="note">Every analysis reads one file, <code>derivatives/qc/gate.tsv</code>
(Analysis/exclusion.py). <code>behaviour</code>: blocks used by analyses of the responses;
excluded are the manual list (config.EXCLUDED_RECORDINGS) and, while
<code>config.APPLY_BLOCK_RULE</code> is on (now {getattr(C, "APPLY_BLOCK_RULE", False)}), blocks
logged at too low a frame rate, blocks whose accuracy on the answered trials is significantly
below chance or with 25 % or more of the trials unanswered, and all blocks of a participant
in a condition whose pooled accuracy on the answered trials is not above chance (an
unreliable calibration leaves only the pooled calibration refit). <code>gait</code>: blocks used
by analyses of the walking alone; only the manual list and the frame-rate criterion exclude
there. Warnings never exclude anything.</p>
{excl_table_html}
<h2>Data quality overview</h2>
<p class="note">Blocks and trials recorded, kept and excluded per participant x condition,
per condition and in total, with standard exclusion labels (Analysis/report.py;
legend and per-block register in <code>summary/data_quality.md</code>).</p>
{dq_html}
<h2>N per analysis stage</h2>
<p class="note">What each stage was left with after the gate (<code>qc/gate_ledger.tsv</code>):
participants and blocks the gate let in, and <code>n_analysed</code>, the participants that
reached the stage's result (lower when the stage drops some itself, e.g. too few trials).
<code>current</code> is False for a row written under an older gate.</p>
{_ledger_html()}
<h2>Block inventory</h2>
<p class="note">Planned runs per participant and condition (config.PLANNED_RUNS) against those
present. Excluded runs (config.EXCLUDED_RECORDINGS) are recorded but not analysed; the
reason is in scans.tsv. Missing or truncated blocks are analysed as they are: every stage pools the runs a
participant has, so a lost block costs trials, not the participant.
<code>n_trials_with_phase</code> is what the walking stride-phase tests actually use; trials in
gaps in the head data get no stride phase.</p>
{inv_html}
<h2>Summary table</h2>
{table_html}
{_common_phase_summary()}
<h2>Figures</h2>
{_figures_html()}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")

    print("=== Dataset summary ===")
    cols = ["participant", "session", "condition", "n_targets", "accuracy",
            "gait_freq_hz", "phase_significant"]
    print(df[cols].to_string(index=False))
    print(f"\n  gate: {n_incl}/{len(edf)} blocks in the behavioural analyses, "
          f"{n_gait}/{len(edf)} in the gait analyses (see exclusions.tsv); "
          f"git {prov['git_sha'][:10]}"
          f"{' (dirty)' if prov['git_dirty'] else ''}.")
    if not inv.empty:
        inc = inv[inv["status"] != "complete"]
        print(f"\n  blocks: {len(inv) - len(inc)}/{len(inv)} participant x condition cells "
              "have every planned run, complete.")
        for _, r in inc.iterrows():
            bits = [f"missing runs {r['missing_runs']}" if r["missing_runs"] != "-" else "",
                    f"(whole visit {r['missing_visits']})" if r["missing_visits"] != "-" else "",
                    f"excluded {r['excluded_runs']}" if r["excluded_runs"] != "-" else "",
                    f"truncated {r['truncated_runs']}" if r["truncated_runs"] != "-" else "",
                    f"extra {r['extra_runs']}" if r["extra_runs"] != "-" else ""]
            print(f"    ! {r['participant']} {r['condition']}: "
                  + " ".join(b for b in bits if b)
                  + f"; {r['n_trials']} trials"
                  + (f", {r['n_trials_with_phase']} with stride phase"
                     if r["condition"] == "walking" else ""))
    if not cbdf.empty:
        bad = cbdf[~cbdf["ok"]]
        print(f"\n  counterbalance: {int(cbdf['ok'].sum())}/{len(cbdf)} participants have "
              f"two visits with the order crossed over and an inter-visit gap in "
              f"[{INTERSESSION_MIN_DAYS}, {INTERSESSION_MAX_DAYS}] d.")
        for _, r in bad.iterrows():
            print(f"    ! {r['participant']}: {r['problem']}")
    print(f"\nSummary TSV  -> {out_dir/'dataset_summary.tsv'}")
    print(f"Exclusions   -> {out_dir/'exclusions.tsv'}")
    if not dq.empty:
        tot = dq[dq["level"] == "total"].iloc[0]
        print(f"Data quality -> {out_dir/'data_quality.md'} ({tot['blocks_kept']}/"
              f"{tot['blocks_recorded']} blocks, {tot['trials_kept']}/"
              f"{tot['trials_recorded']} trials kept)")
    if not inv.empty:
        print(f"Blocks       -> {out_dir/'block_inventory.tsv'}")
    if not cbdf.empty:
        print(f"Counterbal.  -> {out_dir/'counterbalance_qc.tsv'}")
    print(f"Provenance   -> {out_dir/'provenance.json'}")
    print(f"HTML report  -> {out_dir/'report.html'}")


# ###########################################################################
# Data-quality overview
# ###########################################################################
MANUAL_LABELS = ("Technical failure", "No valid walking", "Participant not useful",
                 "Manual exclusion")
LEGEND = [
    ("Low frame rate", "block",
     "the head node logged below config.BLOCK_RULE_MIN_SAMPLING_HZ or kept the stimulus on "
     "screen over config.BLOCK_RULE_MAX_STIM_FACTOR x its nominal time; out of the gait "
     "analyses as well"),
    ("Chance-level accuracy", "block",
     "the participant's accuracy over the answered trials of all their blocks in that "
     "condition not above 50 % (one-sided binomial, config.BLOCK_RULE_ALPHA)"),
    ("Too many timeouts", "block",
     "at least config.BLOCK_RULE_MAX_TIMEOUT_FRACTION of the main trials unanswered"),
    ("Reversed responses", "block",
     "accuracy significantly below 50 %: swapped keys or an invalid stimulus"),
    ("Technical failure", "block (manual)",
     "a rig or tracking malfunction invalidated the block (operator report)"),
    ("No valid walking", "block (manual)",
     "a usable standing block of a participant without any valid walking block, left out "
     "so the participant does not enter the walking-vs-standing contrasts"),
    ("Participant not useful", "block, calibration (manual)",
     "the participant is rated \"not useful\" (config.PARTICIPANT_RATINGS) and left out "
     "of every analysis, gait included"),
    ("Manual exclusion", "block (manual)",
     "any other manual exclusion; the reason is in config.EXCLUDED_RECORDINGS"),
    ("Unreliable calibration", "calibration",
     "the QUEST+ fit failed its reliability check; leaves the pooled calibration refit "
     "only, not the blocks run on it"),
    ("Training trial", "trial", "the practice trials opening every block; never analysed"),
    ("Timeout", "trial", "no response in the response window; left out of accuracy"),
    ("Presentation artefact", "trial",
     "the head tracking froze, jumped or dropped out between 0.3 s before onset and 0.1 s "
     "after the Gabor went off, so the view may have moved while it was shown "
     "(exclusion.py); left out of every trial-based analysis"),
    ("Not recorded", "planned block", "a planned run with no recording"),
    ("Visit pending", "planned block", "a planned run of a visit not yet recorded"),
    ("Visit cancelled", "planned block",
     "a planned run of a visit that will not be recorded (config.VISITS_CANCELLED)"),
]
STATUS_LEVELS = ("useful", "partial", "pending", "not useful")
_RULE_LABELS = (("frame rate", "Low frame rate"), ("reversal", "Reversed responses"),
                ("timeouts", "Too many timeouts"), ("chance", "Chance-level accuracy"))


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
              planned: dict | None = None, labels: dict[str, str] | None = None,
              cancelled: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per block, calibration and missing planned block, with its standard label.

    ``gate`` is the gate table as ``bids_io.load_gate`` returns it, ``training`` the
    number of training trials per block, ``planned``, ``labels`` and ``cancelled`` default
    to ``config.PLANNED_RUNS``, ``config.EXCLUSION_LABELS`` and ``config.VISITS_CANCELLED``.
    """
    training = training or {}
    planned = C.PLANNED_RUNS if planned is None else planned
    labels = getattr(C, "EXCLUSION_LABELS", {}) if labels is None else labels
    cancelled = getattr(C, "VISITS_CANCELLED", {}) if cancelled is None else cancelled
    rows = []
    for _, g in gate.iterrows():
        run = re.search(r"_run-(\d+)", g["recording"])
        manual = g.get("manual", "-") not in ("-", "", "n/a")
        man_label = labels.get(g["recording"]) or (
            "Participant not useful" if g.get("manual") == bids_io.NOT_USEFUL_REASON
            else "Manual exclusion")
        crit = [man_label] if manual else []
        if g["unit"] == "block":
            crit += _rule_labels(g.get("rule", ""))
        elif str(g.get("rule", "")).startswith("calibration"):
            crit.append("Unreliable calibration")
        excluded = g["behaviour"] == "exclude"
        n_test, n_answered = _int(g.get("n_trials")), _int(g.get("n_answered"))
        is_block = g["unit"] == "block"
        n_flag = _int(g.get("n_presentation_flagged")) if is_block else 0
        rows.append({
            "unit": g["unit"], "status": "recorded", "recording": g["recording"],
            "participant": g["participant"], "condition": g["condition"],
            "visit": _int(g.get("visit")) or "n/a",
            "run": int(run.group(1)) if run else "n/a",
            "n_training": training.get(g["recording"], "n/a") if is_block else "n/a",
            "n_trials": n_test + n_flag,
            "n_presentation_flagged": n_flag,
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
                    lab = ("Not recorded" if visit <= last else
                           "Visit cancelled" if sub in cancelled else "Visit pending")
                    rows.append({
                        "unit": "block", "status": "missing", "recording": "n/a",
                        "participant": sub, "condition": cond, "visit": visit, "run": run,
                        "n_training": "n/a", "n_trials": 0, "n_presentation_flagged": 0,
                        "n_answered": 0,
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
        "presentation_artefacts_in_kept_blocks": int(kept["n_presentation_flagged"].sum()),
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


def roster_rows(units: pd.DataFrame, ratings: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per participant: status (``STATUS_LEVELS``), the experimenter's rating
    (``ratings``, default ``config.PARTICIPANT_RATINGS``) and the block counts behind them."""
    ratings = getattr(C, "PARTICIPANT_RATINGS", {}) if ratings is None else ratings
    blocks = units[units["unit"] == "block"]
    conds = sorted(blocks["condition"].unique())
    rows = []
    for sub, b in blocks.groupby("participant"):
        rec = b[b["status"] == "recorded"]
        miss = b[b["status"] == "missing"]
        kept = rec[rec["behaviour"] == "include"]
        rating = ratings.get(sub, "unrated")
        if kept.empty or rating == "not useful":
            status = "not useful"
        elif (miss["label"] == "Visit pending").any():
            status = "pending"
        elif miss.empty and len(kept) == len(rec):
            status = "useful"
        else:
            status = "partial"
        row = {"participant": sub, "status": status, "rating": rating,
               "visits_recorded": int(rec["visit"].nunique())}
        for cond in conds:
            row[f"{cond}_recorded"] = int((rec["condition"] == cond).sum())
            row[f"{cond}_kept"] = int((kept["condition"] == cond).sum())
        row["block_exclusions"] = _count_labels(rec.loc[rec["behaviour"] == "exclude", "label"])
        row["missing"] = _count_labels(miss["label"])
        rows.append(row)
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


def markdown(units: pd.DataFrame, summary: pd.DataFrame, gate_id: str = "",
             roster: pd.DataFrame | None = None) -> str:
    """The overview as a Markdown document."""
    roster = roster_rows(units) if roster is None else roster
    roster_cols = {c: h for c, h in (
        ("participant", "Participant"), ("status", "Status"), ("rating", "Rating"),
        ("visits_recorded", "Visits"), ("walking_kept", "Walking kept"),
        ("walking_recorded", "of recorded"), ("standing_kept", "Standing kept"),
        ("standing_recorded", "of recorded"), ("block_exclusions", "Exclusions"),
        ("missing", "Missing")) if c in roster.columns}
    counts = {
        "participant": "Participant", "condition": "Condition",
        "blocks_recorded": "Blocks", "blocks_kept": "Kept", "blocks_excluded": "Excl.",
        "trials_recorded": "Trials", "trials_kept": "Kept (answered)",
        "trials_excluded_with_block": "Excl. with block",
        "presentation_artefacts_in_kept_blocks": "Presentation artefacts (kept blocks)",
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
          "Generated by `Analysis/report.py` from the exclusion gate "
          f"(`derivatives/qc/gate.tsv`{', gate ' + gate_id if gate_id else ''}) on every "
          "pipeline run; do not edit by hand. Counts are for the behaviour scope (the "
          "analyses of the responses). Trials are main trials; kept trials are the "
          "answered trials of kept blocks whose presentation was valid (no head-tracking "
          "artefact around the stimulus; `qc/presentation_flags.tsv`).", "",
          f"**Overall:** {tot['blocks_kept']}/{tot['blocks_recorded']} blocks and "
          f"{tot['trials_kept']}/{tot['trials_recorded']} trials kept, accuracy "
          f"{tot['accuracy_kept']:.1%} on the kept trials; "
          f"{tot['presentation_artefacts_in_kept_blocks']} trials of kept blocks removed by "
          "the presentation rule; "
          f"{tot['calibrations_kept']}/{tot['calibrations_recorded']} calibrations kept; "
          f"{int(n_train)} training trials not analysed. The gait analyses keep "
          f"{walk_gait}/{len(walk)} walking blocks: only manual exclusions and the "
          "frame-rate criterion apply there.", "",
          "## Participant roster", "",
          "Status from the gate: **useful** = every planned block recorded and kept; "
          "**partial** = some lost or a visit cancelled; **pending** = a visit still to "
          "come; **not useful** = no block kept, or rated so. Rating = the experimenter's "
          "own (`config.PARTICIPANT_RATINGS`, also in `participants.tsv`); a participant "
          "rated not useful is left out of every analysis, gait included.", "",
          _md_table(roster, roster_cols), "",
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
        ev = bids_io.load_events(rec, presentation_rule=False)
        if not ev.empty and "is_training" in ev:
            out[rec.label()] = int((pd.to_numeric(ev["is_training"], errors="coerce") == 1).sum())
    return out


def data_quality_main(quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the overview (Markdown + two TSVs) under ``derivatives/summary/``."""
    gate = bids_io.load_gate()
    if gate.empty:
        print("No exclusion gate found. Run the pipeline first.")
        return pd.DataFrame(), pd.DataFrame()
    units = unit_rows(gate, _training_counts())
    summary = summary_rows(units)
    roster = roster_rows(units)
    out = bids_io.deriv_dir("summary")
    units.to_csv(out / "data_quality_units.tsv", sep="\t", index=False, na_rep="n/a")
    summary.to_csv(out / "data_quality_summary.tsv", sep="\t", index=False, na_rep="n/a")
    roster.to_csv(out / "participant_roster.tsv", sep="\t", index=False, na_rep="n/a")
    (out / "data_quality.md").write_text(markdown(units, summary, bids_io.gate_id(), roster),
                                         encoding="utf-8")
    if not quiet:
        tot = summary[summary["level"] == "total"].iloc[0]
        print(f"  data quality: {tot['blocks_kept']}/{tot['blocks_recorded']} blocks, "
              f"{tot['trials_kept']}/{tot['trials_recorded']} trials kept; "
              f"exclusions: {tot['block_exclusions']}")
        print("  roster: " + "; ".join(f"{n} {s}" for s, n in
                                       roster["status"].value_counts().items()))
        print(f"Data quality -> {out / 'data_quality.md'}")
    return units, summary


if __name__ == "__main__":
    main()
