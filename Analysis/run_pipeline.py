"""Run the full analysis pipeline, stage by stage in dependency order.

    python -m Analysis.run_pipeline              # all stages
    python -m Analysis.run_pipeline --skip-bids  # reuse the BIDS dataset
    python -m Analysis.run_pipeline --only qc    # one stage
    python -m Analysis.run_pipeline --no-cache   # recompute everything

Each stage is also a module that runs on its own (``python -m Analysis.<module>``).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

try:
    from . import (bayes_glmm, bids_io, bidsify, calibration, clocktime_control, conditions,
                   descriptive, exclusion, gait, gait_dynamics, group_curve, head_motion,
                   phase_analysis, population, power, press_phase, qc, report, response_bias,
                   stage_cache, tables, threshold_phase, config as C)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bayes_glmm, bids_io, bidsify, calibration, clocktime_control, conditions
    import descriptive, exclusion, gait, gait_dynamics, group_curve, head_motion
    import phase_analysis, population, power, press_phase, qc, report, response_bias
    import stage_cache, tables, threshold_phase
    import config as C

# (stage, title, function), in dependency order. bids_io.load_gate and the presentation
# flags also rebuild themselves whenever they are stale, e.g. for a stage run with --only.
PIPELINE = [
    ("bids", "BIDS conversion", lambda: bidsify.convert_all()),
    ("calibration", "Calibration check and figures", lambda: calibration.check_main()),
    # Straight after the calibration verdicts it uses, before any analysis. The
    # presentation flags come first: the block rule is judged on the trials they keep.
    ("gate", "Presentation rule + exclusion gate (derivatives/qc/)", lambda: exclusion.main()),
    ("calib_refit", "Calibration slope/lapse refit", lambda: calibration.refit_main()),
    ("qc", "Quality control", lambda: qc.main()),
    ("gait", "Gait extraction and stride phase", lambda: gait.main()),
    ("tables", "Trial and step tables", lambda: tables.main()),
    ("phase", "Stride-phase modulation per participant", lambda: phase_analysis.main()),
    ("population", "Population tests: CONFIRMATORY phase-agnostic, common phase, prevalence",
     lambda: population.main()),
    ("bayes_glmm", "Bayesian mixed models: stride phase x condition (secondary)",
     lambda: bayes_glmm.main([])),
    ("clocktime", "Clock-time control (Hz vs cps)", lambda: clocktime_control.main()),
    ("conditions", "Standing vs walking: performance, H5 threshold, calibration transfer",
     lambda: conditions.main()),
    ("threshold_phase", "Stride phase on the threshold scale (secondary)",
     lambda: threshold_phase.main()),
    # After the calibration transfer, the phase stages and the common-phase test it reads.
    ("power", "Power: calibration cost and a shared oscillation", lambda: power.main()),
    ("descriptive", "Descriptive dataset overview", lambda: descriptive.main()),
    # Reads the walking thresholds of the conditions stage for its head-roll figure.
    ("response_bias", "Stance foot, SDT and head roll (exploratory)", lambda: response_bias.main()),
    ("press_phase", "Button presses over the stride (exploratory)", lambda: press_phase.main()),
    # After the population stage and press_phase, whose results it sets beside its own.
    ("group_curve", "Group-average-curve analysis, calibrated (exploratory)",
     lambda: group_curve.main()),
    ("head_motion", "Head stabilisation and kinematics (exploratory)", lambda: head_motion.main()),
    ("gait_dynamics", "Gait dynamics and phase response (exploratory)",
     lambda: gait_dynamics.main()),
    ("report", "Dataset summary + HTML report", lambda: report.main()),
]
STAGES = [k for k, _, _ in PIPELINE]


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

    steps = [args.only] if args.only else list(STAGES)
    if args.skip_bids and "bids" in steps:
        steps.remove("bids")

    for n, (key, title, run) in enumerate(PIPELINE, start=1):
        if key not in steps:
            continue
        _lap(f"{n}. {title}")
        print(f"\n########## {n}. {title} ##########")
        run()

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
