"""Run the full analysis pipeline.

    python -m Analysis.run_pipeline          # bidsify -> calibration -> gate -> qc -> gait -> ...
    python Analysis/run_pipeline.py --skip-bids
    python Analysis/run_pipeline.py --only qc

Each stage can also be run as its own script.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from . import (bidsify, calibration, calibration_refit, qc, block_rules, gait, phase_analysis,
                   glmm, population, group_agnostic, clocktime_control, conditions,
                   threshold_by_condition, report, descriptive, onset_response,
                   gam_phase, ddm_phase, prevalence_planning, stance_foot, sdt_phase,
                   common_phase, head_stability, gait_dynamics, gait_prc,
                   gait_phase_methods, calibration_transfer, bayes_glmm, bids_io,
                   trial_table, step_table, stage_cache, config as C)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bidsify, calibration, calibration_refit, qc, block_rules, gait, phase_analysis
    import glmm, population, group_agnostic, clocktime_control, conditions
    import threshold_by_condition, report, descriptive, onset_response
    import gam_phase, ddm_phase, prevalence_planning, stance_foot, sdt_phase
    import common_phase
    import head_stability, gait_dynamics, gait_prc, gait_phase_methods
    import calibration_transfer, bayes_glmm, stage_cache
    import trial_table, step_table
    import bids_io
    import config as C

STAGES = ["bids", "calibration", "gate", "calib_refit", "qc", "gait", "trial_table",
          "step_table", "phase",
          "agnostic", "common_phase", "glmm", "bayes_glmm", "population", "planning", "clocktime",
          "conditions",
          "threshold", "calib_transfer", "descriptive", "onset_response", "gam", "ddm", "stance_foot",
          "sdt", "phase_methods", "head_stability", "gait_dynamics", "gait_prc", "report"]


_TIMINGS: list[tuple[str, float]] = []
_RUNNING: list = [None, 0.0]          # [running stage title, start time]


def _lap(next_title: str | None) -> None:
    """Stop timing the current stage (if any) and start timing ``next_title``."""
    now = time.perf_counter()
    if _RUNNING[0] is not None:
        _TIMINGS.append((_RUNNING[0], now - _RUNNING[1]))
        print(f"  [stage took {now - _RUNNING[1]:.1f} s]")
    _RUNNING[0], _RUNNING[1] = next_title, now


def _print_timings() -> None:
    """Wall-clock time per stage, longest first."""
    if not _TIMINGS:
        return
    total = sum(t for _, t in _TIMINGS)
    print(f"\n########## Time per stage (total {total / 60:.1f} min) ##########")
    for title, t in sorted(_TIMINGS, key=lambda x: -x[1]):
        print(f"  {t:7.1f} s  {title}")


def _list_figures() -> list[Path]:
    return sorted(C.DERIV_DIR.glob("**/*.png")) if C.DERIV_DIR.exists() else []


def _print_ledger() -> None:
    """Print the N per stage after the exclusion gate, from qc/gate_ledger.tsv."""
    import pandas as pd
    f = C.DERIV_DIR / "qc" / bids_io.GATE_LEDGER_FILE
    if not f.exists():
        return
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    cur = bids_io.gate_id()
    print(f"\n########## N per stage after the exclusion gate (gate {cur}) ##########")
    for _, r in df.iterrows():
        stale = "" if r["gate_id"] == cur else "  [from an older gate]"
        print(f"  {r['stage']:<15} {r['scope']:<9} {r['condition']:<16} "
              f"participants {r['participants_included']}/{r['participants_total']}, "
              f"{r['unit']} {r['units_included']}/{r['units_total']}, "
              f"analysed n={r['n_analysed']}{stale}")
    print(f"  ledger -> {f}")


def main() -> None:
    """Parse arguments and run the requested pipeline stages in order."""
    # On a redirected Windows console, replace non-cp1252 characters instead of crashing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="ILL analysis pipeline")
    ap.add_argument("--skip-bids", action="store_true", help="reuse existing BIDS dataset")
    ap.add_argument("--only", choices=STAGES, help="run a single stage")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute everything, ignoring .pipeline_cache (stage_cache.py)")
    args = ap.parse_args()
    if args.no_cache:
        stage_cache.ENABLED = False

    steps = STAGES if not args.only else [args.only]
    if args.skip_bids and "bids" in steps:
        steps.remove("bids")

    if "bids" in steps:
        _lap('1. BIDS conversion')
        print("\n########## 1. BIDS conversion ##########")
        bidsify.convert_all()
    if "calibration" in steps:
        _lap('2. Calibration figures')
        print("\n########## 2. Calibration figures ##########")
        calibration.main()
    if "gate" in steps:
        # Built right after the calibration verdicts it depends on. Later stages read it and
        # print their N; bids_io.load_gate rebuilds it if stale (e.g. with --only).
        _lap('2b. Exclusion gate (derivatives/qc/gate.tsv)')
        print("\n########## 2b. Exclusion gate (derivatives/qc/gate.tsv) ##########")
        block_rules.main()
    if "calib_refit" in steps:
        _lap('2c. Calibration slope/lapse refit (b8)')
        print("\n########## 2c. Calibration slope/lapse refit (b8) ##########")
        calibration_refit.main()
    if "qc" in steps:
        _lap('3. Quality control')
        print("\n########## 3. Quality control ##########")
        qc.main()
    if "gait" in steps:
        _lap('4. Gait extraction')
        print("\n########## 4. Gait extraction ##########")
        gait.main()
    if "trial_table" in steps:
        _lap('4a. Trial table (derivatives/trial_table/trials.tsv)')
        print("\n########## 4a. Trial table (derivatives/trial_table/trials.tsv) ##########")
        trial_table.main()
    if "step_table" in steps:
        _lap('4b. Step table (derivatives/step_table/steps.tsv)')
        print("\n########## 4b. Step table (derivatives/step_table/steps.tsv) ##########")
        step_table.main()
    if "phase" in steps:
        _lap('5. Stride-phase modulation (per participant)')
        print("\n########## 5. Stride-phase modulation (per participant) ##########")
        phase_analysis.main()
    if "agnostic" in steps:
        _lap('6. CONFIRMATORY phase-agnostic population test')
        print("\n########## 6. CONFIRMATORY phase-agnostic population test ##########")
        group_agnostic.main()
    if "common_phase" in steps:
        _lap('6a. Common-phase population test - Hotelling T2 (secondary)')
        print("\n########## 6a. Common-phase population test - Hotelling T2 (secondary) ##########")
        common_phase.main()
    if "glmm" in steps:
        _lap('6a. Hierarchical GLMM — common-phase (secondary)')
        print("\n########## 6a. Hierarchical GLMM — common-phase (secondary) ##########")
        glmm.main()
    if "bayes_glmm" in steps:
        _lap('6a2. Bayesian mixed models: stride phase x condition (secondary)')
        print("\n########## 6a2. Bayesian mixed models: stride phase x condition (secondary) ##########")
        bayes_glmm.main([])
    if "population" in steps:
        _lap('6b. Population prevalence & phase (Fig 4/5)')
        print("\n########## 6b. Population prevalence & phase (Fig 4/5) ##########")
        population.main()
    if "planning" in steps:
        _lap('6d. Prevalence precision planning')
        print("\n########## 6d. Prevalence precision planning ##########")
        prevalence_planning.main()
    if "clocktime" in steps:
        _lap('6e. Clock-time control (Supp Fig 5/6: Hz vs cps)')
        print("\n########## 6e. Clock-time control (Supp Fig 5/6: Hz vs cps) ##########")
        clocktime_control.main()
    if "conditions" in steps:
        _lap('7. Standing vs walking (Fig 2)')
        print("\n########## 7. Standing vs walking (Fig 2) ##########")
        conditions.main()
    if "threshold" in steps:
        _lap('7a. Threshold standing vs walking (H5)')
        print("\n########## 7a. Threshold standing vs walking (H5) ##########")
        threshold_by_condition.main()
    if "calib_transfer" in steps:
        _lap('7a2. Calibration transfer: calibrated vs block threshold')
        print("\n########## 7a2. Calibration transfer: calibrated vs block threshold ##########")
        calibration_transfer.main()
    if "descriptive" in steps:
        _lap('7b. Descriptive dataset overview (exploratory)')
        print("\n########## 7b. Descriptive dataset overview (exploratory) ##########")
        descriptive.main()
    if "onset_response" in steps:
        _lap('7c. Onset vs response locus (exploratory)')
        print("\n########## 7c. Onset vs response locus (exploratory) ##########")
        onset_response.main()
    if "gam" in steps:
        _lap('7d. Cyclic-spline GAM + mutual information (exploratory)')
        print("\n########## 7d. Cyclic-spline GAM + mutual information (exploratory) ##########")
        gam_phase.main()
    if "ddm" in steps:
        _lap('7e. Drift-diffusion decomposition (exploratory)')
        print("\n########## 7e. Drift-diffusion decomposition (exploratory) ##########")
        ddm_phase.main()
    if "stance_foot" in steps:
        _lap('7f. Stance-foot contrast — which foot is down (exploratory)')
        print("\n########## 7f. Stance-foot contrast — which foot is down (exploratory) ##########")
        stance_foot.main()
    if "sdt" in steps:
        _lap('7g. Sensitivity vs criterion — SDT decomposition (exploratory)')
        print("\n########## 7g. Sensitivity vs criterion — SDT decomposition (exploratory) ##########")
        sdt_phase.main()
    if "phase_methods" in steps:
        _lap('7h. Stride-phase methods on held-out signals (exploratory)')
        print("\n########## 7h. Stride-phase methods on held-out signals (exploratory) ##########")
        gait_phase_methods.main()
    if "head_stability" in steps:
        _lap('7i. Head stabilisation over the stride (exploratory)')
        print("\n########## 7i. Head stabilisation over the stride (exploratory) ##########")
        head_stability.main()
    if "gait_dynamics" in steps:
        _lap('7j. Stride-to-stride gait dynamics (exploratory)')
        print("\n########## 7j. Stride-to-stride gait dynamics (exploratory) ##########")
        gait_dynamics.main()
    if "gait_prc" in steps:
        _lap('7k. Gait phase response to task events (exploratory)')
        print("\n########## 7k. Gait phase response to task events (exploratory) ##########")
        gait_prc.main()
    if "report" in steps:
        _lap('8. Dataset summary + HTML report (b6/b7)')
        print("\n########## 8. Dataset summary + HTML report (b6/b7) ##########")
        report.main()

    _lap(None)
    _print_timings()
    if len(steps) > 1:
        _print_ledger()

    figs = _list_figures()
    print(f"\n########## Figures ({len(figs)}, in each stage's figures/ folders) ##########")
    for stage, n in sorted(Counter(f.relative_to(C.DERIV_DIR).parts[0] for f in figs).items()):
        print(f"  {stage:<20} {n}")
    print(f"\nAll outputs under: {C.DERIV_DIR}")


if __name__ == "__main__":
    main()
