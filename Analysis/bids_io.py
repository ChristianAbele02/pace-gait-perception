"""Loader for the generated Motion-BIDS dataset.

Discovery and loading functions, so analysis scripts read the BIDS dataset instead of the
raw Unity CSVs.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C

# Spec order is task, tracksys, acq, run. Datasets written before 2026-09-21 put acq
# before tracksys; that order is still accepted when reading.
_REC_RE = re.compile(
    r"^(?P<sub>sub-\d+)_(?P<ses>ses-\d+)_task-(?P<task>[a-zA-Z0-9]+)"
    r"(?:_acq-(?P<acq_old>[a-zA-Z0-9]+))?"
    r"_tracksys-(?P<tracksys>[a-zA-Z0-9]+)"
    r"(?:_acq-(?P<acq>[a-zA-Z0-9]+))?"
    r"(?:_run-(?P<run>\d+))?"
    r"_motion\.tsv$"
)


@dataclass
class Recording:
    """One motion recording (a sub/ses[/acq][/run]) and the paths to its BIDS files."""
    sub: str
    ses: str
    task: str
    tracksys: str
    motion_tsv: Path
    acq: str = field(default="")
    run: str = field(default="")           # BIDS run- index; "" for single/legacy
    condition: str = field(default="unknown")
    old_order: bool = field(default=False)  # acq- before tracksys- (older files)

    @property
    def _acq_tag(self) -> str:
        return f"_acq-{self.acq}" if self.acq else ""

    @property
    def _run_tag(self) -> str:
        return f"_run-{self.run}" if self.run else ""

    @property
    def base(self) -> str:
        """Full BIDS stem of the motion files, entities in spec order.

        Includes ``tracksys-``; channels, JSON and events sidecars use this stem.
        """
        if self.old_order:
            return (f"{self.sub}_{self.ses}_task-{self.task}{self._acq_tag}"
                    f"_tracksys-{self.tracksys}{self._run_tag}")
        return (f"{self.sub}_{self.ses}_task-{self.task}_tracksys-{self.tracksys}"
                f"{self._acq_tag}{self._run_tag}")

    @property
    def events_tsv(self) -> Path:
        """Trial table for this recording (one row per stimulus)."""
        # events include tracksys (BIDS Motion recommendation)
        return self.motion_tsv.parent / f"{self.base}_events.tsv"

    @property
    def channels_tsv(self) -> Path:
        """Channel description sidecar (one row per motion column)."""
        return self.motion_tsv.parent / f"{self.base}_channels.tsv"

    @property
    def motion_json(self) -> Path:
        """Motion metadata sidecar (sampling frequency, task, tracking system)."""
        return self.motion_tsv.parent / f"{self.base}_motion.json"

    def label(self) -> str:
        """Short stem used to name this recording's derivatives and figures.

        Like :attr:`base` without ``tracksys-``; keeps ``acq-`` and ``run-`` so each block
        has its own outputs.
        """
        return f"{self.sub}_{self.ses}_task-{self.task}{self._acq_tag}{self._run_tag}"


def _condition_for(sub: str, ses: str, bids_dir: Path) -> str:
    tsv = bids_dir / sub / f"{sub}_sessions.tsv"
    if tsv.exists():
        s = pd.read_csv(tsv, sep="\t")
        if "condition" in s.columns:
            hit = s.loc[s["session_id"] == ses, "condition"]
            if len(hit):
                return str(hit.iloc[0])
    return "unknown"


# ---------------------------------------------------------------------------
# The exclusion gate
# ---------------------------------------------------------------------------
# derivatives/qc/gate.tsv decides which blocks and calibrations each analysis may use.
# block_rules.py writes it (stage "gate", after the calibration verdicts); stages read it
# via find_recordings() / excluded_calibrations() and report their N with gate_report().
# One decision column per scope, since a participant at chance can still have good gait:
#   behaviour - analyses of responses (accuracy, RT, thresholds, SDT, DDM...): manual list
#               plus the automatic rule while config.APPLY_BLOCK_RULE is on.
#   gait      - analyses of walking only: manual list only.
# Below block level, load_events() drops the trials the presentation rule flags
# (presentation.py, derivatives/qc/presentation_flags.tsv) in both scopes. The block
# rule is judged on the remaining trials.
GATE_FILE = "gate.tsv"
GATE_LEDGER_FILE = "gate_ledger.tsv"
SCOPES = ("behaviour", "gait")


def _qc_dir(bids_dir: Path | None = None) -> Path:
    return Path(bids_dir or C.BIDS_DIR) / "derivatives" / "qc"


def gate_settings() -> dict:
    """Config values the gate depends on; a change makes the gate stale."""
    # Bump RuleVersion when block_rules changes its decisions.
    try:
        from . import presentation
    except ImportError:
        import presentation
    return {"RuleVersion": 5,
            "RuleActive": bool(getattr(C, "APPLY_BLOCK_RULE", False)),
            "Alpha": float(getattr(C, "BLOCK_RULE_ALPHA", 0.05)),
            "MaxTimeoutFraction": float(getattr(C, "BLOCK_RULE_MAX_TIMEOUT_FRACTION", 0.25)),
            "Manual": dict(sorted(getattr(C, "EXCLUDED_RECORDINGS", {}).items())),
            "PresentationRule": {
                "Active": bool(getattr(C, "APPLY_PRESENTATION_RULE", False)),
                **presentation.settings()}}


def calibration_label(path: Path) -> str:
    """Gate label of a calibration table: its file name without ``.tsv``."""
    return path.name[:-len(".tsv")]


def gate_staleness(bids_dir: Path | None = None) -> str | None:
    """Reason the gate file is out of date, or None if current.

    Stale if missing, if blocks or calibrations were added or removed, if its config
    changed, or if an input (events, calibration tables, calibration verdicts, the
    presentation flags) is newer.
    """
    f = _qc_dir(bids_dir) / GATE_FILE
    meta = f.with_suffix(".json")
    if not f.exists() or not meta.exists():
        return "no gate file yet"
    try:
        gate = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        settings = json.loads(meta.read_text(encoding="utf-8")).get("Settings")
    except (OSError, ValueError, pd.errors.ParserError):
        return "gate file unreadable"
    recs = find_recordings(bids_dir, include_excluded=True)
    cals = find_calibration_tables(bids_dir)
    on_disk = {r.label() for r in recs} | {calibration_label(c) for c in cals}
    if set(gate.get("recording", pd.Series(dtype=str))) != on_disk:
        return "blocks or calibrations added or removed since it was built"
    if settings != json.loads(json.dumps(gate_settings())):
        return "exclusion settings in config.py changed"
    deriv = Path(bids_dir or C.BIDS_DIR) / "derivatives" / "calibration"
    inputs = ([r.events_tsv for r in recs] + list(cals)
              + list(deriv.glob("sub-*/*/*_calibration_qc.json"))
              + [_qc_dir(bids_dir) / "presentation_flags.tsv"])
    newest = max((q.stat().st_mtime for q in inputs if q.exists()), default=0.0)
    if newest > f.stat().st_mtime:
        return "inputs changed since it was built"
    return None


def load_gate(bids_dir: Path | None = None, refresh: bool = True) -> pd.DataFrame:
    """Gate table, rebuilt first if stale.

    Only the dataset at ``config.BIDS_DIR`` is rebuilt; others are read as they are.
    """
    bids_dir = Path(bids_dir or C.BIDS_DIR)
    if refresh and bids_dir.resolve() == Path(C.BIDS_DIR).resolve():
        why = gate_staleness(bids_dir)
        if why:
            print(f"  [gate] rebuilding {GATE_FILE}: {why}")
            try:
                from . import block_rules
            except ImportError:
                import block_rules
            block_rules.main(quiet=True)
    f = _qc_dir(bids_dir) / GATE_FILE
    return pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) if f.exists() else pd.DataFrame()


def _check_scope(scope: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"unknown gate scope {scope!r}; use one of {SCOPES}")


def excluded_recordings(bids_dir: Path | None = None,
                        apply_block_rule: bool | None = None,
                        scope: str = "behaviour") -> dict[str, str]:
    """``{recording label: reason}`` for blocks excluded from ``scope``.

    Always includes ``config.EXCLUDED_RECORDINGS``. For ``behaviour`` also the blocks the
    automatic rule flags, if ``config.APPLY_BLOCK_RULE`` (or ``apply_block_rule``) is True.
    """
    _check_scope(scope)
    out = {k: f"manual: {v}" for k, v in getattr(C, "EXCLUDED_RECORDINGS", {}).items()}
    if apply_block_rule is None:
        apply_block_rule = bool(getattr(C, "APPLY_BLOCK_RULE", False))
    if apply_block_rule and scope == "behaviour":
        gate = load_gate(bids_dir)
        if not gate.empty:
            hit = gate[(gate["unit"] == "block") & (gate["rule_exclude"].str.lower() == "true")]
            for _, r in hit.iterrows():
                out.setdefault(r["recording"], f"block rule: {r['rule']}")
    return out


def excluded_calibrations(bids_dir: Path | None = None) -> dict[str, str]:
    """``{calibration label: reason}`` for UNRELIABLE calibrations, excluded from pooled
    calibration fits while ``config.APPLY_BLOCK_RULE`` is True."""
    if not getattr(C, "APPLY_BLOCK_RULE", False):
        return {}
    gate = load_gate(bids_dir)
    if gate.empty:
        return {}
    hit = gate[(gate["unit"] == "calibration") & (gate["rule_exclude"].str.lower() == "true")]
    return {r["recording"]: f"calibration rule: {r['rule']}" for _, r in hit.iterrows()}


def gate_id(bids_dir: Path | None = None) -> str:
    """Short fingerprint of the gate decisions, stored in each ledger row."""
    meta = (_qc_dir(bids_dir) / GATE_FILE).with_suffix(".json")
    try:
        return str(json.loads(meta.read_text(encoding="utf-8")).get("GateId", "n/a"))
    except (OSError, ValueError):
        return "n/a"


def _brief(label: str) -> str:
    """sub-02_ses-02_task-ILL_acq-walking_run-02 -> sub-02 walking run-02."""
    m = re.match(r"(sub-\d+)_ses-\d+_task-[^_]+(?:_tracksys-[^_]+)?(?:_acq-([a-z]+))?"
                 r"(?:_run-(\d+))?", label)
    if not m:
        return label
    return " ".join(x for x in (m.group(1), m.group(2),
                                f"run-{m.group(3)}" if m.group(3) else None) if x)


def gate_report(stage: str, scope: str = "behaviour", *, condition=None,
                unit: str = "blocks", n_analysed: int | None = None,
                analysed_subjects=None, prune: tuple[str, ...] = ()) -> dict:
    """Print and log how many participants and blocks the gate admits to ``stage``.

    ``condition`` restricts the count ("walking" or a tuple). ``unit="calibrations"``
    counts calibration tables. ``n_analysed`` is the number of participants in the stage's
    result; if lower than the gate's N, the stage dropped some (too few trials, no gait).

    ``analysed_subjects`` can replace ``n_analysed``; the dropped participants are then
    named.

    ``prune`` lists derivative folders with per-participant or per-block outputs. Files of
    excluded units, and of participants the stage dropped (if ``analysed_subjects`` is
    given), are deleted so outdated results are not picked up later.

    The row is upserted into ``derivatives/qc/gate_ledger.tsv`` (one per stage and unit)
    with the gate fingerprint, so rows from an older gate are visible.
    """
    _check_scope(scope)
    conds = (condition,) if isinstance(condition, str) else tuple(condition or ())
    if unit == "calibrations":
        allu = {calibration_label(c): c.parts[-4] for c in find_calibration_tables()
                if not conds or any(f"_acq-{c_}" in c.name for c_ in conds)}
        excl = excluded_calibrations() if scope == "behaviour" else {}
    else:
        allu = {r.label(): r.sub for r in find_recordings(include_excluded=True)
                if not conds or r.condition in conds}
        excl = excluded_recordings(scope=scope)
    out_units = {u: excl[u] for u in allu if u in excl}
    subs_all = sorted(set(allu.values()))
    subs_in = sorted({s for u, s in allu.items() if u not in out_units})
    subs_out = [s for s in subs_all if s not in subs_in]

    by_sub: dict[str, list[str]] = {}
    for u in out_units:
        by_sub.setdefault(allu[u], []).append(u)
    detail = []
    for s in sorted(by_sub):
        kinds = sorted({out_units[u].split(":", 1)[0] for u in by_sub[s]})
        detail.append(f"{s} x{len(by_sub[s])} ({', '.join(kinds)})" if len(by_sub[s]) > 1
                      else f"{_brief(by_sub[s][0])} ({kinds[0]})")
    what = unit if not conds else f"{'/'.join(conds)} {unit}"
    line = (f"  N [{stage} | {scope} gate]: {len(subs_in)}/{len(subs_all)} participants, "
            f"{len(allu) - len(out_units)}/{len(allu)} {what}")
    if out_units:
        line += f"; excluded {'; '.join(detail)}"
    if subs_out:
        line += f"; out entirely: {', '.join(subs_out)}"
    dropped: list[str] = []
    if analysed_subjects is not None:
        analysed_subjects = set(analysed_subjects)
        n_analysed = len(analysed_subjects)
        dropped = [s for s in subs_in if s not in analysed_subjects]
    if n_analysed is not None:
        line += f" -> analysed n={n_analysed}"
        if dropped:
            line += f" (dropped by this stage: {', '.join(dropped)})"
        elif n_analysed < len(subs_in):
            line += f" ({len(subs_in) - n_analysed} more dropped by this stage)"
    print(line)

    if prune:
        _prune_outputs(prune, set(out_units), set(subs_out) | set(dropped))

    row = {"stage": stage, "scope": scope, "unit": unit,
           "condition": "/".join(conds) or "all",
           "participants_total": len(subs_all), "participants_included": len(subs_in),
           "participants_excluded": ",".join(subs_out) or "-",
           "units_total": len(allu), "units_included": len(allu) - len(out_units),
           "units_excluded": len(out_units),
           "n_analysed": n_analysed if n_analysed is not None else "n/a",
           "excluded_detail": "; ".join(detail) or "-",
           "gate_id": gate_id()}
    ledger = _qc_dir() / GATE_LEDGER_FILE
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row]).astype(str)
    if ledger.exists():
        old = pd.read_csv(ledger, sep="\t", dtype=str, keep_default_na=False)
        same = (old["stage"] == stage) & (old["unit"] == unit)
        if same.any():
            old.loc[same, list(row)] = new.iloc[0].to_numpy()
            new = old
        else:
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(ledger, sep="\t", index=False)
    return row


def clear_stale(kind: str, *names: str) -> None:
    """Delete a stage's previous group-level outputs when none can be computed at the
    current N, so outdated results are not left in place."""
    for n in names:
        for f in (C.DERIV_DIR / kind / n, C.DERIV_DIR / kind / "figures" / n):
            if f.exists():
                f.unlink()
                print(f"  [gate] removed {f.relative_to(C.DERIV_DIR).as_posix()}: "
                      "not estimable at the current N")


def _prune_outputs(kinds: tuple[str, ...], labels: set[str], subs_out: set[str]) -> None:
    """Delete outputs of excluded units from the given derivative folders."""
    removed = 0
    for kind in kinds:
        root = C.DERIV_DIR / kind
        if not root.exists():
            continue
        for f in sorted(p for p in root.rglob("*") if p.is_file()):
            sub_of = f.name.split("_", 1)[0]
            if sub_of in subs_out or any(f.name.startswith(lb + "_") for lb in labels):
                f.unlink()
                removed += 1
        for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            if not any(d.iterdir()):
                d.rmdir()
    if removed:
        print(f"  [gate] removed {removed} output file(s) of excluded units from "
              f"{', '.join(kinds)}")


def find_recordings(bids_dir: Path | None = None, *, include_excluded: bool = False,
                    apply_block_rule: bool | None = None,
                    scope: str = "behaviour") -> list[Recording]:
    """All motion recordings in the dataset as :class:`Recording` objects.

    Recordings excluded from ``scope`` (:func:`excluded_recordings`) are dropped unless
    ``include_excluded`` is set (used by QC, the gate and the block inventory). The default
    scope is ``behaviour``; walking-only analyses pass ``scope="gait"``.

    One entry per block, not per participant: the two-visit design gives eight walking and
    two standing recordings per participant, distinguished by ``run-``. Callers needing
    participants must pool by ``(sub, ses)``.

    Condition comes from ``acq-``, or from ``sessions.tsv`` for older recordings.
    """
    bids_dir = Path(bids_dir or C.BIDS_DIR)
    recs = []
    for tsv in sorted(bids_dir.glob("sub-*/ses-*/motion/*_motion.tsv")):
        m = _REC_RE.match(tsv.name)
        if m:
            d = m.groupdict()
            acq_old = d.pop("acq_old") or ""
            acq = d.pop("acq") or acq_old
            run = d.pop("run") or ""
            d["old_order"] = bool(acq_old)
            # condition from acq-, else from sessions.tsv
            condition = acq if acq in ("walking", "standing") else _condition_for(d["sub"], d["ses"], bids_dir)
            recs.append(Recording(motion_tsv=tsv, acq=acq, run=run, condition=condition, **d))
    if not include_excluded:
        excluded = excluded_recordings(bids_dir, apply_block_rule, scope)
        recs = [r for r in recs if r.label() not in excluded]
    return recs


def scan_meta(rec: Recording, bids_dir: Path | None = None) -> dict:
    """This recording's row from ``sub-XX_scans.tsv``, as a plain dict.

    Includes the counterbalance logged since 2026-08-18: ``visit``,
    ``condition_order_first``, ``condition_position`` (first or second in its visit) and
    ``acq_time``. Older recordings lack them or have ``n/a``; callers must handle a
    missing ``condition_position``.
    """
    bids_dir = Path(bids_dir or C.BIDS_DIR)
    tsv = bids_dir / rec.sub / f"{rec.sub}_scans.tsv"
    if not tsv.exists():
        return {}
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    if "filename" not in df.columns:
        return {}
    try:
        rel = rec.motion_tsv.relative_to(bids_dir / rec.sub).as_posix()
    except ValueError:
        return {}
    hit = df.loc[df["filename"] == rel]
    if not len(hit):
        return {}
    row = {k: (None if pd.isna(v) or str(v) == "n/a" else str(v))
           for k, v in hit.iloc[0].to_dict().items()}
    if row.get("visit") is not None:
        try:
            row["visit"] = int(float(row["visit"]))
        except ValueError:
            row["visit"] = None
    return row


def find_calibration_tables(bids_dir: Path | None = None) -> list[Path]:
    """Paths to every QUEST+ calibration table (``*_beh.tsv``) in the dataset.

    One per condition and visit, i.e. four per participant after both visits.
    """
    bids_dir = Path(bids_dir or C.BIDS_DIR)
    return sorted(bids_dir.glob("sub-*/ses-*/beh/*_task-*_beh.tsv"))


def load_motion(rec: Recording) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (motion_df, channels_df, motion_json_meta).

    Use the ``t`` column (true sample time) for timing, not the row index.

    ``_motion.tsv`` has no header; column names come from ``_channels.tsv``. Files written
    before 2026-09-21 have a header row, which is detected.
    """
    channels = pd.read_csv(rec.channels_tsv, sep="\t", na_values=["n/a"])
    names = channels["name"].astype(str).tolist()
    with open(rec.motion_tsv, encoding="utf-8") as f:
        has_header = f.readline().rstrip("\r\n").split("\t") == names
    motion = pd.read_csv(rec.motion_tsv, sep="\t", na_values=["n/a"],
                         header=0 if has_header else None,
                         names=None if has_header else names)
    meta = json.loads(rec.motion_json.read_text(encoding="utf-8"))
    return motion, channels, meta


def load_events(rec: Recording, presentation_rule: bool | None = None) -> pd.DataFrame:
    """Trial table for ``rec``, or an empty frame if it has none.

    Onsets are seconds from recording start on the clock of ``motion_df["t"]``.

    With the presentation rule on (``config.APPLY_PRESENTATION_RULE``, or the
    ``presentation_rule`` argument), trials whose presentation window overlaps a
    head-tracking artefact are dropped and the index is reset (``presentation.py``).
    Every trial-based analysis reads events this way. QC, the gait sheets and the trial
    table pass ``presentation_rule=False`` to see every trial.
    """
    if not rec.events_tsv.exists():
        return pd.DataFrame()
    ev = pd.read_csv(rec.events_tsv, sep="\t", na_values=["n/a"])
    if presentation_rule is None:
        presentation_rule = bool(getattr(C, "APPLY_PRESENTATION_RULE", False))
    if presentation_rule and not ev.empty:
        try:
            from . import presentation
        except ImportError:
            import presentation
        ev = ev[presentation.valid_mask(rec, ev)].reset_index(drop=True)
    return ev


# One folder per stage under derivatives/, grouped by role. Each is a BIDS-derivative
# dataset: deriv_dir() writes its dataset_description.json, and bidsify lists the groups in
# derivatives/README.md.
DERIVATIVE_GROUPS = {
    "Preprocessing and quality": {
        "qc": "One QC sheet per block: head position raw vs cleaned with the size of every "
              "correction, belt speed, target onsets and accuracy by level; frame timing, "
              "tracking and vigilance in the JSON (qc.py). Also the exclusion gate, the "
              "N per stage, and the presentation flags of every trial (block_rules.py, "
              "presentation.py).",
        "calibration": "QUEST+ calibration per condition and visit: staircase figures, "
                       "reliability checks, and the offline slope/lapse refit "
                       "(calibration.py, calibration_refit.py).",
        "gait": "Steps and stride phase per walking block: one gait sheet per block (stride "
                "template and durations, head path with the onsets, stance-foot decode), "
                "time series in the .npz (gait.py).",
    },
    "Tables for further analysis": {
        "trial_table": "Every trial of every block in one table: design, response, RT "
                       "validity, stride phase and stance foot at onset, and the exclusion "
                       "gate's decisions (trial_table.py).",
        "step_table": "Every step of every walking block in one table: duration, stance "
                      "foot and its evidence, head sway and bob over the step "
                      "(step_table.py).",
    },
    "Confirmatory and secondary tests": {
        "phase": "Stride-phase modulation of accuracy (answered trials), reaction time and the "
                 "timeout rate: one sheet per walking block, the per-participant "
                 "confirmatory 2 and 4 cps tests, and the pooled "
                 "exploratory aggregate (phase_analysis.py).",
        "population": "Group-level tests across participants: phase-agnostic test, common "
                      "phase, GLMM, prevalence (group_agnostic.py, common_phase.py, glmm.py, "
                      "population.py).",
        "conditions": "Standing versus walking: accuracy (answered trials), reaction time "
                      "and thresholds "
                      "(conditions.py, threshold_by_condition.py).",
        "clocktime": "Control analysis on clock time instead of stride phase, one figure per "
                     "block (clocktime_control.py).",
        "bayes_glmm": "Bayesian mixed models (NumPyro) of stride phase on accuracy and log "
                      "RT with standing as the control: shared-phase and per-person "
                      "modulation, Bayes factors, walking main effect (bayes_glmm.py).",
    },
    "Descriptive": {
        "descriptive": "Descriptive tables and overview figures of behaviour and data "
                       "quality (descriptive.py).",
        "calibration_transfer": "Whether each calibrated threshold holds in the blocks "
                                "run on it: accuracy at the calibrated tilt and the "
                                "blocks' own threshold against a sampling null "
                                "(calibration_transfer.py).",
    },
    "Exploratory: responses over the stride": {
        "onset_response": "Responses locked to stimulus onset (onset_response.py).",
        "gam_phase": "GAM fits and mutual information over stride phase (gam_phase.py).",
        "ddm_phase": "Drift-diffusion decomposition over stride phase (ddm_phase.py).",
        "sdt": "Signal-detection measures over stride phase (sdt_phase.py).",
        "stance_foot": "Stance-foot (foot A / foot B) contrast (stance_foot.py).",
    },
    "Exploratory: the walking itself": {
        "head_stability": "Head pitch against the vertical bob over the stride; head "
                          "fixation point and gaze compensation (head_stability.py).",
        "gait_dynamics": "Stride-time variability, DFA long-range correlations and local "
                         "dynamic stability per walking block (gait_dynamics.py).",
        "gait_prc": "Gait phase shifts after stimulus onsets and button presses (phase "
                    "response curve; gait_prc.py).",
        "gait_phase_methods": "Trough, Hilbert and multi-channel stride phase scored on "
                              "held-out head signals (gait_phase_methods.py).",
    },
    "Planning and summary": {
        "planning": "Power and prevalence planning simulations (prevalence_planning.py).",
        "summary": "Dataset summary tables, block inventory and the HTML report (report.py).",
        "sensitivity": "Sensitivity analyses run on demand, not by the pipeline: the "
                       "pipeline with and without the presentation rule, compared "
                       "(presentation_comparison.py).",
    },
}
DERIVATIVE_FOLDERS = {k: v for group in DERIVATIVE_GROUPS.values() for k, v in group.items()}


def describe_derivative(name: str, deriv_root: Path | None = None) -> None:
    """Write derivatives/<name>/dataset_description.json (BIDS-Derivatives)."""
    d = Path(deriv_root or C.DERIV_DIR) / name
    d.mkdir(parents=True, exist_ok=True)
    text = json.dumps({
        "Name": f"ILL analysis derivatives: {name}",
        "BIDSVersion": "1.11.1",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "ILL analysis pipeline (Analysis/run_pipeline.py)",
                         "Description": DERIVATIVE_FOLDERS.get(name, "n/a")}],
        "DatasetLinks": {"raw": "../../"},
        "SourceDatasets": [{"URL": "bids:raw:"}],
    }, indent=2, ensure_ascii=False)
    f = d / "dataset_description.json"
    if not f.exists() or f.read_text(encoding="utf-8") != text:
        f.write_text(text, encoding="utf-8")


def deriv_dir(name: str, sub: str | None = None, ses: str | None = None) -> Path:
    """Path under derivatives/<name>[/sub-XX[/ses-YY]], created on demand."""
    describe_derivative(name)
    d = C.DERIV_DIR / name
    if sub:
        d = d / sub
    if ses:
        d = d / ses
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig_dir(name: str, sub: str | None = None, ses: str | None = None) -> Path:
    """``deriv_dir(...)/figures``, created on demand."""
    d = deriv_dir(name, sub, ses) / "figures"
    d.mkdir(exist_ok=True)
    return d


def uniform_resample(t: np.ndarray, y: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate (t, y) onto a uniform grid at ``fs`` Hz using true timestamps.

    Unity samples are unevenly spaced; filtering and FFT need a uniform grid.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 2:
        return t, y
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return grid, np.interp(grid, t, y)
