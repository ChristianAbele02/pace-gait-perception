"""Simulate participants and run the pipeline on them as a readiness check.

Writes raw CSVs in the rig's logger format for N synthetic participants (standing and
walking calibration, then several walking and standing blocks), with condition tokens in
the file names (`pXX-bY-tZZ_walking.csv`, `_standing.csv`, `_calibration_standing.csv`)
and the onset/mocap columns. Then converts them and runs the pipeline stages listed in
``main`` on a separate `bids_sim/` dataset; the real dataset is not touched.

Checks end to end that:
  * logger output converts to BIDS with correct acq- names and run indices (no block of
    a condition overwrites another);
  * walking runs are pooled per participant, and the population stages (GLMM, de-biased
    prevalence) and the standing-vs-walking comparison work with n > 1 and both conditions.

Generative model. Trials come from a two-boundary drift-diffusion process, so choice and
RT are consistent, which the DDM, GAM and onset-vs-response modules need. Drift depends on
difficulty (HARD/THR/EASY: ~0.65/0.75/0.85 correct). Some walking participants are
responders with a stride-locked modulation based on Davidson et al. (2024) (``DAVIDSON``,
``davidson_power/effect.py``): 2 or 4 cps, individual preferred phase, hit-rate amplitude
~0.04-0.08. The modulation acts on three parameters to give a known ground truth:
  * drift rate, peaking at ``phi_v`` (sensory: higher accuracy and shorter RT), locked to
    target onset;
  * non-decision time, small modulation in phase with ``phi_v`` (motor timing);
  * an omission gate at a different phase ``phi_resp``, so response likelihood peaks away
    from accuracy (Davidson's swing vs heel-strike split).
Non-responders and all standing trials are flat. The ground truth is written to
``sim_truth.json``.

Usage::

    python -m Analysis.simulate_participants            # 6 participants, default
    python -m Analysis.simulate_participants --n 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C

FS = 60.0
STRIDE_HZ = 0.9              # ~1.11 s stride; vertical bob at 2x = 1.8 Hz
BOB_M, SWAY_M = 0.03, 0.03
ISI_S = 4.0                  # inter-target interval, jittered per trial
ISI_JITTER_S = 0.75         # half-range of the uniform ISI jitter
RNG = np.random.default_rng(2024)

# --- Drift-diffusion generative parameters ---------------------------------
# Symmetric two-boundary Wiener process (start at midpoint, unit diffusion). With bounds at
# ±a/2, P(correct) = sigmoid(v*a), so the drifts below give ~0.65/0.75/0.85 for a = 1.4.
# Trials slower than RESPONSE_WINDOW_S become timeouts.
A_BOUND = 1.4               # boundary separation a
T0_S = 0.30                 # baseline non-decision time; RTs ~0.8 s
DDM_DT = 0.01               # Euler step (s); RT resolution
DRIFT = {"HARD": 0.442, "THR": 0.785, "EASY": 1.239}   # v: sigmoid(v*1.4) ≈ .65/.75/.85
# Hit-rate amplitude to relative drift amplitude m_v at threshold:
# ΔP ≈ [P(1-P)·a]·(m_v·v_THR), so m_v = ΔP / (P(1-P)·a·v_THR).
_DPDV_THR = 0.75 * 0.25 * A_BOUND                       # dP/dv at threshold ≈ 0.2625
ACC_AMP_TO_MV = 1.0 / (_DPDV_THR * DRIFT["THR"])        # ≈ 4.85

# --- Davidson (2024) empirical calibration ---------------------------------
# From davidson_power/effect.py on Davidson's OSF data (Nat Commun 2024; walking, target
# onset, accuracy x stride, N=36):
#   * baseline accuracy ≈ 0.74; median independent walking targets per participant ≈ 836
#     (the published curve has ~1670 rows because each target appears twice);
#   * debiased amplitude: population RMS ≈ 0.026 at 2 cps, p75 of max(2, 4 cps) ≈ 0.051,
#     median of the 8 flagged responders ≈ 0.062;
#   * 8/36 ≈ 22 % significant, but the test's actual false-positive rate is ~15-17 %, so
#     this is not a clean prevalence estimate;
#   * peak frequency is 2 or 4 cps.
# The amplitudes below are at or above the empirical upper end so the synthetic data have
# a detectable effect. They are not the design's effect size; the planning value is the
# 0.026 RMS (davidson_power/).
DAVIDSON = dict(baseline_acc=0.75, targets_per_pp=836,
                amp_acc_responder=(0.04, 0.08), sig_prevalence=8.0 / 36.0,
                peak_freqs=(2.0, 4.0))


# ---------------------------------------------------------------------------
# Weibull (as in calibration_refit), used for the calibration log
# ---------------------------------------------------------------------------
def _weibull(x, alpha, beta=3.2, lapse=0.03, guess=0.5):
    f = 1.0 - np.exp(-np.power(np.clip(x, 1e-9, None) / alpha, beta))
    return guess + (1.0 - guess - lapse) * f


# ---------------------------------------------------------------------------
# Drift-diffusion trial sampler (Euler-Maruyama, unit diffusion)
# ---------------------------------------------------------------------------
def _ddm_sample(rng, v: float, a: float, t0: float,
                dt: float = DDM_DT, max_rt: float = 2.0):
    """One 2-AFC drift-diffusion trial. Returns (correct, rt), or (None, None) on timeout.

    Evidence starts between bounds at ±a/2, drifts at v toward the correct bound with unit
    diffusion; the first crossing gives choice and decision time, plus ``t0``. No decision
    before ``max_rt`` is a timeout.
    """
    half = a / 2.0
    x = 0.0
    sd = np.sqrt(dt)
    max_steps = int(max((max_rt - t0) / dt, 0))
    for i in range(max_steps):
        x += v * dt + rng.normal(0.0, sd)
        if x >= half:
            return True, t0 + (i + 1) * dt
        if x <= -half:
            return False, t0 + (i + 1) * dt
    return None, None


# ---------------------------------------------------------------------------
# One experiment run (walking or standing) as a raw logger dataframe
# ---------------------------------------------------------------------------
def _motion_run(condition: str, threshold: float, dur_s: float, *,
                mod: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """One experiment run. ``mod`` is the stride modulation of a responder, or ``None``
    for a flat observer (non-responders and all standing runs)."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    t = np.arange(n) / FS + np.cumsum(rng.normal(0, 0.0005, n))   # frame jitter
    walking = condition == "walking"

    if walking:
        local_y = BOB_M * np.cos(2 * np.pi * 2 * STRIDE_HZ * t) + rng.normal(0, 0.002, n)
        local_x = SWAY_M * np.sin(2 * np.pi * STRIDE_HZ * t) + rng.normal(0, 0.002, n)
        speed = np.full(n, 1.0) + rng.normal(0, 0.02, n)
        dist = np.cumsum(speed) / FS
    else:
        local_y = rng.normal(0, 0.004, n)
        local_x = rng.normal(0, 0.004, n)
        speed = np.abs(rng.normal(0, 0.01, n))
        dist = np.zeros(n)

    easy, hard = threshold * 1.26, threshold * 0.76
    df = pd.DataFrame({
        "timestamp": t,
        "clock-hires": t,
        "head-global-x": local_x, "head-global-y": 1.7 + local_y, "head-global-z": dist,
        "head-local-x": local_x, "head-local-y": local_y, "head-local-z": rng.normal(0, 0.002, n),
        "head-global-rot-x": rng.normal(0, 1, n), "head-global-rot-y": rng.normal(0, 3, n),
        "head-global-rot-z": rng.normal(0, 1, n),
        "head-local-rot-x": rng.normal(0, 1, n), "head-local-rot-y": rng.normal(0, 2, n),
        "head-local-rot-z": rng.normal(0, 1, n),
        # 6-DOF mocap pose (a2/a3) with pitch, roll and yaw
        "head-mocap-x": local_x, "head-mocap-y": 1.7 + local_y, "head-mocap-z": dist,
        "head-mocap-qx": rng.normal(0, 0.01, n), "head-mocap-qy": rng.normal(0, 0.01, n),
        "head-mocap-qz": rng.normal(0, 0.01, n), "head-mocap-qw": np.ones(n),
        "head-mocap-rot-x": 3 * np.cos(2 * np.pi * 2 * STRIDE_HZ * t) if walking else rng.normal(0, 0.5, n),
        "head-mocap-rot-y": rng.normal(0, 3, n),
        "head-mocap-rot-z": 2 * np.sin(2 * np.pi * STRIDE_HZ * t) if walking else rng.normal(0, 0.5, n),
        "head-mocap-valid": 1,
        "mocap-frame-advanced": 1,
        "treadmill-speed": speed,
        "distance-on-path": dist,
        "tracking-framenumber": np.arange(n),
        "tracking-timestamp": t,
        "task-visible": 0, "stimulus-onset": 0, "response-onset": 0,
        "reaction-time-precise": -1.0, "checkpoint-dropped": -1,
        "participant-id": 0, "block-id": 1 if walking else 0, "trial-id": 1,
        # counterbalance columns, set per run by generate() (see ALLOCATION)
        "visit": 1, "counterbalance-slot": 0, "condition-order-first": "n/a",
        "session-start-iso": "n/a",
        "checkpoint-type": "", "checkpoint-index": -1, "is-training": -1,
        "difficulty-level": "", "visual-tilt": -1, "visual-angle": 0.0,
        "expected-response": -1, "checkpoint-response": -1, "response-correct": -1,
        "checkpoint-reaction-time": -1.0,
        "threshold-angle": threshold, "easy-angle": easy, "hard-angle": hard,
    })

    # --- trials every ~ISI seconds from the drift-diffusion model ---
    diffs = {"THR": threshold, "EASY": easy, "HARD": hard}   # tilt (deg) per level
    weights = {"THR": 0.7, "EASY": 0.15, "HARD": 0.15}
    onset = 5.0
    idx_for = lambda tt: int(np.searchsorted(t, tt))
    ci = 0
    while onset < dur_s - max(5.0, C.RESPONSE_WINDOW_S + 0.5):
        lvl = rng.choice(list(weights), p=list(weights.values()))
        ang = diffs[lvl]
        direction = int(rng.integers(0, 2))          # correct button (0/1)
        sf = (STRIDE_HZ * onset) % 1.0               # stride fraction at target onset

        # DDM parameters, stride-modulated for responders
        v, t0, gate = DRIFT[lvl], T0_S, 1.0
        if mod is not None:
            a_v = 2 * np.pi * mod["k"] * (sf - mod["phi_v"])
            v = DRIFT[lvl] * (1.0 + mod["m_v"] * np.cos(a_v))          # drift (sensory)
            t0 = T0_S * (1.0 - mod["m_t0"] * np.cos(a_v))             # non-decision time (motor)
            # omission gate at phi_resp, a different phase from accuracy
            gate = 1.0 - mod["amp_resp"] * np.sin(np.pi * mod["k"] * (sf - mod["phi_resp"])) ** 2
        correct_bound, rt = _ddm_sample(rng, max(v, 0.02), A_BOUND, t0,
                                        max_rt=C.RESPONSE_WINDOW_S)
        responded = (rt is not None) and (rng.uniform() < gate)

        oi = idx_for(onset)
        df.loc[oi:idx_for(onset + C.STIM_DURATION_S), "task-visible"] = 1
        df.loc[oi, "stimulus-onset"] = 1
        if responded:
            correct = int(correct_bound)
            resp = direction if correct else 1 - direction
            res_i = idx_for(onset + rt)
            if res_i >= n:
                break
            df.loc[res_i, "response-onset"] = 1
            df.loc[res_i, "reaction-time-precise"] = rt
            row = [condition, ci, 0, lvl, direction, ang, direction, resp, correct, rt]
        else:
            # Timeout: the row records the target (checkpoint-index >= 0) with -1 for
            # response and correctness (n/a downstream), as the rig does.
            res_i = idx_for(onset + C.RESPONSE_WINDOW_S)
            if res_i >= n:
                break
            row = [condition, ci, 0, lvl, direction, ang, direction, -1, -1, -1.0]
        df.loc[res_i, ["checkpoint-type", "checkpoint-index", "is-training", "difficulty-level",
                       "visual-tilt", "visual-angle", "expected-response", "checkpoint-response",
                       "response-correct", "checkpoint-reaction-time"]] = row
        onset += ISI_S + float(rng.uniform(-ISI_JITTER_S, ISI_JITTER_S))
        ci += 1
    return df


# ---------------------------------------------------------------------------
# Calibration run as a raw QUEST+-style log
# ---------------------------------------------------------------------------
def _calibration_run(threshold: float, n_trials: int = 80, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    levels = threshold * np.geomspace(0.4, 2.2, 7)
    rows = []
    for i in range(n_trials):
        x = float(rng.choice(levels))
        d = int(rng.integers(0, 2))
        correct = int(rng.uniform() < _weibull(x, threshold))
        rows.append({"tracker": "A" if i % 2 else "B", "global_trial": i,
                     "tracker_trial": i // 2, "presented_tilt_deg": round(x, 4),
                     "tilt_direction": d, "response": d if correct else 1 - d,
                     "correct": correct, "post_mean_threshold_deg": round(threshold, 4),
                     "post_sd_log": 0.1})
    thr = {"threshold_deg": threshold, "easy_deg": threshold * 1.26, "hard_deg": threshold * 0.76}
    return pd.DataFrame(rows), thr


def _write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, sep=";", index=False)


def _write_calibration_csv(df: pd.DataFrame, thr: dict, path: Path):
    header = (f"# threshold_deg={thr['threshold_deg']};easy_deg={thr['easy_deg']};"
              f"hard_deg={thr['hard_deg']}\n# quest+ params gamma=0.5;beta=3.2;lambda=0.05\n"
              f"# trackerA_deg={thr['threshold_deg']};trackerB_deg={thr['threshold_deg']};"
              f"pooled_deg={thr['threshold_deg']};n_scored_trials={len(df)};converged=1\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
    df.to_csv(path, sep=";", index=False, mode="a")


# ---------------------------------------------------------------------------
def _draw_modulation(p: int) -> dict:
    """Stride modulation for one responder, based on Davidson (2 or 4 cps, individual phase,
    accuracy amplitude ~0.04-0.08).

    Acts on drift (``phi_v``), non-decision time (in phase with ``phi_v``) and an omission
    gate (``phi_resp``), giving the ground truth for the onset-vs-response and DDM modules."""
    lo, hi = DAVIDSON["amp_acc_responder"]
    amp_acc = float(RNG.uniform(lo, hi))
    phi_v = float(RNG.uniform(0.0, 1.0))                         # preferred stride fraction
    phi_resp = float((phi_v + RNG.uniform(0.20, 0.35)) % 1.0)   # response phase, offset from phi_v
    return {"responder": True, "k": float(RNG.choice(DAVIDSON["peak_freqs"])),
            "amp_acc": amp_acc, "m_v": float(min(amp_acc * ACC_AMP_TO_MV, 0.6)),
            "m_t0": float(RNG.uniform(0.03, 0.07)), "phi_v": phi_v,
            "amp_resp": float(RNG.uniform(0.12, 0.22)), "phi_resp": phi_resp}


# --- Counterbalance, identical to ExperimentConductor.ALLOCATION ------------
# Permuted blocks of 4 (two 'S' = standing first, two 'W' = walking first), so S and W
# never differ by more than 2 during enrolment. Must match the Unity constant exactly.
ALLOCATION = "WSSWSWWSSSWWWSSWWWSSSWSWWSSWSWSWSWWSWSWSWSSWSWWSWSSWWWSSWSWSSWSW"


def standing_first(slot: int, visit: int) -> bool:
    """Condition order for a (slot, visit), with the visit-2 crossover applied."""
    base = ALLOCATION[(max(slot, 1) - 1) % len(ALLOCATION)] == "S"
    return (not base) if visit % 2 == 0 else base


def generate(n_participants: int, raw_dir: Path, walk_s=360.0, stand_s=220.0,
             n_walk=3, n_stand=2, responder_frac: float = 0.5, visits: int = 1):
    """Write raw logs for ``n_participants``: one standing and one walking calibration,
    then ``n_walk`` walking and ``n_stand`` standing blocks in each visit.

    Each block has its own trial number (BIDS run-); block-id selects the condition
    (0 = standing, 1 = walking).

    ``responder_frac`` of participants are responders with a stride-locked modulation, the
    rest flat. This is higher than Davidson's ~22 % so a small simulated sample still has
    signal. Ground truth goes to ``sim_truth.json``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_resp = int(np.ceil(responder_frac * n_participants))
    truth = {}
    for p in range(1, n_participants + 1):
        thr_stand = float(np.clip(RNG.normal(5.0, 1.0), 2.5, 9.0))
        thr_walk = float(np.clip(thr_stand * RNG.normal(1.3, 0.1), 2.5, 12.0))  # walking threshold usually higher
        mod = _draw_modulation(p) if p <= n_resp else None
        truth[C.sub_label(p)] = mod if mod is not None else {"responder": False}
        # calibrations (ses-01), one per condition
        cds, ths = _calibration_run(thr_stand, seed=p)
        _write_calibration_csv(cds, ths, raw_dir / f"p{p:02d}-b0-t01_calibration_standing.csv")
        cdw, thw = _calibration_run(thr_walk, seed=100 + p)
        _write_calibration_csv(cdw, thw, raw_dir / f"p{p:02d}-b1-t01_calibration_walking.csv")
        # Experiment blocks. With visits=2 the order crosses over between visits as on the
        # rig, so threshold_by_condition._crossover_model is estimable. Run indices count
        # across visits (both in ses-02).
        for visit in range(1, visits + 1):
            s_first = standing_first(p, visit)
            # visits a week apart; blocks consecutive within a visit
            clock = datetime(2026, 9, 1, 10, 0, 0) + timedelta(days=7 * (visit - 1))
            seq = ([("standing", n_stand)] + [("walking", n_walk)] if s_first
                   else [("walking", n_walk)] + [("standing", n_stand)])
            for cond, n_blocks in seq:
                clock += timedelta(minutes=8)             # calibration
                for i in range(1, n_blocks + 1):
                    r = (visit - 1) * n_blocks + i        # run index across visits
                    if cond == "walking":
                        df = _motion_run("walking", thr_walk, walk_s, mod=mod,
                                         seed=1000 + 10 * p + r)
                        bid, dur = 1, walk_s
                    else:
                        df = _motion_run("standing", thr_stand, stand_s, mod=None,
                                         seed=2000 + 10 * p + r)
                        bid, dur = 0, stand_s
                    df["participant-id"] = p
                    df["visit"] = visit
                    df["counterbalance-slot"] = p
                    df["condition-order-first"] = "standing" if s_first else "walking"
                    df["session-start-iso"] = clock.strftime("%Y-%m-%dT%H:%M:%S")
                    _write_csv(df, raw_dir / f"p{p:02d}-b{bid}-t{r:02d}_{cond}.csv")
                    clock += timedelta(seconds=dur + 120)   # block and break
    (raw_dir / "sim_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    n_r = min(n_resp, n_participants)
    print(f"  generated {n_participants} participants ({n_r} responders / "
          f"{n_participants - n_r} flat; 2 per-condition calibrations + {n_walk} walking + "
          f"{n_stand} standing blocks each) -> {raw_dir}")


def main():
    """Generate a synthetic cohort and run the pipeline on it."""
    # On a redirected Windows console, replace non-cp1252 characters instead of crashing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of simulated participants")
    ap.add_argument("--keep", action="store_true", help="keep the bids_sim dataset afterwards")
    args = ap.parse_args()

    sim_root = C.PROJECT_ROOT / "bids_sim"
    raw_dir = C.PROJECT_ROOT / "_sim_raw"
    for d in (sim_root, raw_dir):
        if d.exists():
            shutil.rmtree(d)

    # Point config at the simulated dataset; each pNN is its own subject.
    C.BIDS_DIR = sim_root
    C.SOURCEDATA_DIR = sim_root / "sourcedata"
    C.DERIV_DIR = sim_root / "derivatives"
    C.RAW_DIR = raw_dir
    C.MERGE_TO_SUBJECT = None

    print(f"=== Simulated-participant readiness check (n={args.n}) ===")
    generate(args.n, raw_dir)

    # Stages read C.BIDS_DIR / C.DERIV_DIR at call time, so they use bids_sim/.
    try:
        from . import (bidsify, bids_io, qc, block_rules, calibration, calibration_refit, gait,
                       phase_analysis, group_agnostic, glmm, population,
                       prevalence_planning, clocktime_control, conditions,
                       threshold_by_condition, descriptive, report,
                       onset_response, gam_phase, ddm_phase, stance_foot, sdt_phase,
                       common_phase, gait_phase_methods, head_stability, gait_dynamics,
                       gait_prc)
    except ImportError:
        import bidsify, bids_io, qc, block_rules, calibration, calibration_refit, gait
        import phase_analysis, group_agnostic, glmm, population, prevalence_planning
        import clocktime_control, conditions, threshold_by_condition, descriptive, report
        import onset_response, gam_phase, ddm_phase, stance_foot, sdt_phase
        import common_phase
        import gait_phase_methods, head_stability, gait_dynamics, gait_prc

    print("\n-- convert --")
    bidsify.convert_all(raw_dir=raw_dir, bids_dir=sim_root, prune_loose=True)
    print("\n-- qc --")
    for r in bids_io.find_recordings(sim_root, include_excluded=True):
        qc.qc_recording(r)
    print("\n-- calibration --");        calibration.main()
    print("\n-- exclusion gate --");     block_rules.main()
    print("\n-- calibration refit --");  calibration_refit.main()
    print("\n-- gait --");        gait.main()
    print("\n-- phase --");       phase_analysis.main()
    # --- confirmatory population test ---
    print("\n-- phase-agnostic population test (CONFIRMATORY) --")
    group_agnostic.main()
    print("\n-- common-phase Hotelling T2 (secondary) --")
    common_phase.main()
    print("\n-- glmm (descriptive estimates only) --");  glmm.main()
    print("\n-- population prevalence --");           population.main()
    print("\n-- prevalence planning --");             prevalence_planning.main()
    print("\n-- clocktime --");   clocktime_control.main()
    print("\n-- conditions --");  conditions.main()
    print("\n-- threshold standing vs walking (H5) --")
    threshold_by_condition.main()
    print("\n-- descriptive --"); descriptive.main()
    # --- exploratory analyses ---
    print("\n-- onset-vs-response (exploratory) --");  onset_response.main()
    print("\n-- GAM + mutual information (exploratory) --");  gam_phase.main()
    print("\n-- drift-diffusion decomposition (exploratory) --");  ddm_phase.main()
    print("\n-- stance-foot contrast (exploratory) --");  stance_foot.main()
    print("\n-- sensitivity vs criterion (exploratory) --");  sdt_phase.main()
    print("\n-- stride-phase methods (exploratory) --");  gait_phase_methods.main()
    print("\n-- head stabilisation (exploratory) --");  head_stability.main()
    print("\n-- gait dynamics (exploratory) --");  gait_dynamics.main()
    print("\n-- gait phase response (exploratory) --");  gait_prc.main()
    print("\n-- report --");      report.main()

    print(f"\nSim dataset at {sim_root}.")
    if not args.keep:
        print("(deleting bids_sim/ and _sim_raw/; re-run with --keep to retain them.)")
        shutil.rmtree(sim_root, ignore_errors=True)
        shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
