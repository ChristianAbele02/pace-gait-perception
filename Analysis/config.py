"""Shared configuration for the ILL analysis pipeline.

Paths are relative to the repository root, so scripts work from any directory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIDS_DIR = PROJECT_ROOT / "data"  # BIDS dataset root
# Raw Unity logs are copied into data/ from the head node. After conversion bidsify
# moves them unchanged to data/sourcedata/raw/.
RAW_DIR = BIDS_DIR
SOURCEDATA_DIR = BIDS_DIR / "sourcedata"   # zipped raw Unity output with BIDS names
DERIV_DIR = BIDS_DIR / "derivatives"       # QC / gait / phase / calibration outputs

# --- BIDS naming -----------------------------------------------------------
TASK_LABEL = "ILL"                 # one task label for the whole study
TRACKING_SYSTEM = "optitrack"      # OptiTrack + VRPN
MANUFACTURER = "NaturalPoint Inc. (OptiTrack), Corvallis, OR, USA"
# Motion capture as documented in Hofmann (2023, PhD thesis, Bielefeld University, ch. 2);
# camera count as currently installed (16, per the lab). Update if hardware changes.
MOCAP_MODEL = "OptiTrack Flex 13 (16 cameras on the CAVE support cage)"
MOCAP_SOFTWARE = "OptiTrack Motive 1.9"

# Institution fields for the BIDS sidecars (recommended by the spec). Empty = omitted.
INSTITUTION_NAME = "Bielefeld University"
INSTITUTION_ADDRESS = "Universitätsstraße 25, 33615 Bielefeld, Germany"
INSTITUTION_DEPARTMENT = "Cognitive Neuroscience, Biology Faculty"

# dataset_description.json: AUTHORS created the dataset; SUPERVISORS go under
# Acknowledgements (BIDS has no supervisor field).
AUTHORS = ["Christian Abele-Kretz"]
SUPERVISORS = ["Prof. Dr. Christoph Kayser", "Dr. Norbert Bödekker"]

# Sessions: calibration is ses-01, experiment blocks are ses-02. The condition is the acq-
# entity (acq-walking / acq-standing), so both share ses-02; sub-XX_scans.tsv also lists
# it. Blocks of the same condition are separated by run-, taken from the raw trial field
# (run_for). Both visits go into ses-02 and differ only by run-.
CALIBRATION_SESSION = "ses-01"
EXPERIMENT_SESSION = "ses-02"

# Planned runs per visit (INSTRUCTIONS.md §6). Only used to report missing blocks
# (report.block_inventory); analyses pool whatever runs exist, so a lost block or visit
# costs trials, not the participant.
PLANNED_RUNS = {"walking": {1: (1, 2, 3, 4), 2: (5, 6, 7, 8)},
                "standing": {1: (1,), 2: (2,)}}

# Recordings excluded from all analyses, keyed by Recording.label(), with the reason. They
# stay in the BIDS dataset (reason in scans.tsv) and still go through QC; the exclusion gate
# (derivatives/qc/gate.tsv) removes them from both scopes.
# Example:
#   "sub-XX_ses-02_task-ILL_acq-walking_run-NN":
#       "Tracking failure flipped the CAVE view (operator report); responses reversed.",
EXCLUDED_RECORDINGS: dict[str, str] = {}

# Standard label for each manual exclusion above, used by the data-quality overview
# (data_quality.py) next to the labels the automatic rule produces. One of
# data_quality.MANUAL_LABELS; a recording missing here is labelled "Manual exclusion".
# Example: {"sub-XX_ses-02_task-ILL_acq-walking_run-NN": "Technical failure"}
EXCLUSION_LABELS: dict[str, str] = {}

# Setup-screen values logged wrongly, keyed by (subject, visit), with the reason. bidsify
# applies them to scans.tsv, participants.tsv and the calibration beh tables; raw logs in
# sourcedata/ are left unchanged.
# Example:
#   ("sub-XX", 2): {"participant": XX, "counterbalance_slot": N,
#                   "condition_order_first": "standing",
#                   "reason": "The setup screen still held the previous participant's values."},
SETUP_CORRECTIONS: dict[tuple[str, int], dict] = {}

# Automatic block exclusion rule (block_rules.py): frame rate, reversal and timeouts per
# block, chance per participant x condition, unreliable calibration. Verdicts go to
# derivatives/qc/gate.tsv. The frame-rate check applies to every analysis; the rest to
# behavioural analyses only, so gait analyses keep blocks answered at chance.
APPLY_BLOCK_RULE = True
BLOCK_RULE_ALPHA = 0.05
BLOCK_RULE_MAX_TIMEOUT_FRACTION = 0.25    # exclude blocks with >= 25 % unanswered
BLOCK_RULE_MIN_SAMPLING_HZ = 30.0         # the head node normally runs at ~60 Hz
BLOCK_RULE_MAX_STIM_FACTOR = 2.0          # exclude blocks with median on-screen time > 2 x nominal

# Presentation rule (presentation.py): a trial is dropped from every trial-based analysis
# when a head-tracking artefact overlaps its presentation window. The tracked head drives
# the CAVE perspective, so a frozen or jumping pose moves the view while the Gabor is shown.
# Window = [onset - PRE, onset + measured on-screen time + POST]. The values were fixed
# before data collection was complete and must not be tuned to results.
# ILL_PRESENTATION_RULE=0 turns the rule off (presentation_comparison.py uses this).
APPLY_PRESENTATION_RULE = os.environ.get("ILL_PRESENTATION_RULE", "1") not in ("0", "false", "no")
PRESENTATION_PRE_S = 0.30         # lead: the render eye is smoothed at 1.8 Hz (~0.3 s to settle)
PRESENTATION_POST_S = 0.10        # margin after the Gabor's measured offset
PRESENTATION_STALE_MIN_S = 0.10   # a frozen pose this long counts; single repeats do not
PRESENTATION_GAP_S = 0.25         # no head sample for longer than this counts (gait.GAP_S)

# None: each raw pNN maps to sub-NN. An integer puts all input under one subject.
MERGE_TO_SUBJECT: int | None = None

# --- Acquisition constants (README / StaircaseManager) ---------------------
NOMINAL_FS = 60.0          # Unity target frame rate (Hz); actual dt is logged per frame
STIM_DURATION_S = 0.10     # taskDuration: Gabor display time
RESPONSE_WINDOW_S = 2.0    # responseWindow (lowered from 3 s; RTs are ~0.8-0.9 s)

# Stride-phase method for the modulation analysis (b3):
#   "trough"  -> linear interpolation between double-support troughs (Davidson).
#   "hilbert" -> continuous analytic-signal phase; independent of trough detection, origin
#                from lateral sway (no half-stride ambiguity).
#   "phaser"  -> multi-channel phase (vertical bob, lateral sway, head roll; phaser.py,
#                after Revzen & Guckenheimer 2008). 2-3x lower error than one channel on
#                synthetic data, but no better on real head data (no more held-out variance
#                than troughs, slipped cycles). Compared in gait_phase_methods.py; not
#                recommended.
# gait.py saves all three; this picks the one all stages bin on. The confirmatory analysis
# uses "trough", as specified in the design.
PHASE_METHOD = "trough"      # "trough" (confirmatory) | "hilbert" | "phaser" (multi-channel)

# Local head positions further than this (m) from the median are treated as dropouts or
# pre-initialisation samples (e.g. the -505 frames at the start).
LOCAL_POS_VALID_RADIUS_M = 2.0
# Belt speeds above this are sensor glitches (e.g. a 309 m/s spike).
MAX_PLAUSIBLE_SPEED_MPS = 5.0


# --- block id to condition -------------------------------------------------
def block_to_condition(block_id: int) -> str:
    """block 0 == standing, block > 0 == walking."""
    return "standing" if int(block_id) == 0 else "walking"


def acq_for(condition: str | None) -> str | None:
    """BIDS ``acq-`` label for walking/standing; None for calibration or unknown."""
    return condition if condition in ("walking", "standing") else None


def run_for(ids: dict) -> str | None:
    """BIDS ``run-`` label from the raw file's trial field.

    Each block of a condition gets its own trial number (1, 2, 3, ...), stored as
    ``run-01``, ``run-02``, ... Both visits are in ses-02, so run numbers count up per
    condition across the study.

    Trial ``0`` (legacy, single run) gets no run- entity. Operators number blocks from 1 (INSTRUCTIONS.md).
    """
    t = int(ids["trial"])
    return f"run-{t:02d}" if t >= 1 else None


# --- Raw filename parsing ---------------------------------------------------
# Examples: p01-b1-t01.csv, p01-b1-t01-1.csv (collision suffix), p00-b0-t00_calibration.csv,
# and gzipped logs with a condition token:
#   p01-b1-t01_walking.csv.gz, p01-b0-t00_standing.csv.gz, p01-b1-t01_walking-1.csv.gz
# Per-condition calibrations:
#   p05-b0-t00_calibration_standing.csv.gz, p05-b0-t00_calibration_walking.csv.gz
# (the older bare ..._calibration.csv is still accepted).
# Since 2026-09-16 the visit follows the participant:
#   p05-v2-b1-t05_walking.csv.gz, p05-v2-b0-t02_calibration_standing.csv
# Older names without it parse with visit None.
# The condition token and the -N suffix may come in either order.
_RAW_RE = re.compile(
    r"^p(?P<participant>\d+)(?:-v(?P<visit>\d+))?-b(?P<block>\d+)-t(?P<trial>\d+)"
    r"(?:-(?P<dup>\d+))?"
    r"(?:_(?P<cond>calibration_standing|calibration_walking|walking|standing|calibration))?"
    r"(?:-(?P<dup2>\d+))?"
    r"\.csv(?:\.gz)?$",
    re.IGNORECASE,
)


def parse_raw_name(name: str) -> dict | None:
    """Parse a raw CSV filename into its identifiers, or None if it doesn't match."""
    m = _RAW_RE.match(Path(name).name)
    if not m:
        return None
    g = m.groupdict()
    cond = (g["cond"] or "").lower()
    is_calib = cond.startswith("calibration")
    # Condition from the token if present, else from the block id.
    #   "calibration_standing"/"calibration_walking": calibration measured while standing/walking.
    #   bare "calibration": legacy single calibration without a condition.
    if cond in ("walking", "standing"):
        condition = cond
    elif cond in ("calibration_standing", "calibration_walking"):
        condition = cond.split("_", 1)[1]          # standing / walking
    elif is_calib:
        condition = "calibration"
    else:
        condition = block_to_condition(int(g["block"]))
    return {
        "participant": int(g["participant"]),
        "visit": int(g["visit"]) if g["visit"] else None,
        "block": int(g["block"]),
        "trial": int(g["trial"]),
        "dup": int(g["dup"] or g["dup2"]) if (g["dup"] or g["dup2"]) else None,
        "is_calibration": is_calib,
        "condition": condition,
    }


def sub_label(participant: int) -> str:
    """Zero-padded BIDS subject label for a participant number (7 -> ``sub-07``)."""
    return f"sub-{participant:02d}"


def subject_for(ids: dict) -> str:
    """BIDS subject label for a parsed raw file (honours MERGE_TO_SUBJECT)."""
    return sub_label(MERGE_TO_SUBJECT if MERGE_TO_SUBJECT is not None else ids["participant"])


def session_for(ids: dict) -> str:
    """BIDS session label: calibration -> ses-01, experiment -> ses-02."""
    return CALIBRATION_SESSION if ids["is_calibration"] else EXPERIMENT_SESSION


# --- Lateral sway to anatomical side (stance-foot labels) -------------------
# ``head_local_pos_x`` is ``trackingOrigin.InverseTransformPoint(head.position).x``
# (ILLLoggerController.cs). Unity is left-handed, so facing the rig's +z, local +x is the
# participant's right. The head and trunk shift toward the stance limb in single support,
# so a step with positive mean lateral sway is a right-stance step.
#
# This rests on an assumption about the rig's mounting, not a measurement, so it is gated:
#
#   * LATERAL_SIGN_VERIFIED = False (default)
#       Steps are still labelled consistently, so the stride origin, odd harmonics and
#       the direction-agnostic stance-foot test (``stance_foot.py``) remain valid. Labels
#       read "foot A"/"foot B", and the direction-specific (common-sign) population test
#       is reported as provisional.
#   * LATERAL_SIGN_VERIFIED = True
#       Labels read "right"/"left" and the common-sign test has an anatomical meaning.
#
# To verify: record a walk-in-place with a known first step (called by the operator), run
# ``python -m Analysis.gait`` and check that step's label in panel (e) of the block's
# ``*_gait.png``. If inverted, flip LATERAL_POSITIVE_IS_RIGHT; do not change signs in the
# analysis code.
LATERAL_POSITIVE_IS_RIGHT = True    # +x sway during a step means that foot is in stance
LATERAL_SIGN_VERIFIED = False       # set True after checking against a known first step

# Head pitch sign (head_stability.py): positive = nose down, assuming a room frame of
# x right / y up / z forward. The measured step-rate pitch lies ~130-180 deg from the
# vertical bob, i.e. anti-compensatory. Pozzo et al. (1990) and Hirasaki et al. (1999)
# report compensatory pitch in walking, so a mirrored rotation (e.g. in the Motive to Unity
# conversion) is more likely. Until verified, the sign of fixation point and compensation is
# not interpreted. To verify: nod (chin down) and check that the pitch goes positive.
HEAD_PITCH_SIGN_VERIFIED = False


# --- Outcome measures and multiplicity -------------------------------------
# Holm in ``phase_analysis`` / ``group_agnostic`` corrects over the two confirmatory
# frequencies (2 and 4 cps) within a measure, not across measures. Each additional
# confirmatory measure would inflate the family-wise error rate.
#
# Accuracy is the single pre-specified primary outcome: the power analysis is based on it
# (``davidson_power/power.py``), Davidson et al. (2024) report it as the main effect, and
# its scale is comparable across participants without normalisation. It is computed over
# the answered trials: a timeout is a missing response, not a wrong one. Reaction time is
# secondary, without family-wise error control.
#
# Timeouts are a secondary measure too: P(no response) over all trials at their onset
# phase, with the same binned fit and trial-level permutation as accuracy.
#
# Response likelihood ("response_counts") was removed. In Davidson's detection task a
# response meant a detection, so the count histogram measured sensitivity. Here every trial
# requires a left/right press, so a response only means no timeout. In addition, the
# observed histogram binned response phases (onset + RT) while the null sampled onset
# phases; the response-phase distribution is the onset distribution smoothed by RT spread.
# The test was therefore overly conservative and degenerate (p = 1) when timeouts were rare.
# The timeouts measure asks the same question correctly: outcome and null both use onset
# phases, and it runs only for participants with at least ``phase_analysis.MIN_TIMEOUTS``.
PRIMARY_MEASURE = "accuracy"
SECONDARY_MEASURES = ("reaction_time", "timeouts")

# --- Estimator choices for the binned Fourier fit --------------------------
# Both on by default. Each only reduces variance and cannot create an effect, since the
# permutation null goes through the same code.
#
# WEIGHT_BINS_BY_N: weight each phase bin by its trial count in the cosine fit. A bin mean
#   has variance sigma^2 / n_bin, and bin counts vary because onset coverage is not fully
#   uniform (the checkpoint jitter is a reflected random walk, residual Rayleigh R ~ 0.07).
#   Set False to reproduce Davidson's unweighted fit.
WEIGHT_BINS_BY_N = True

# RESIDUALISE_DIFFICULTY: subtract each difficulty level's mean before binning and add the
#   grand mean back (scale stays hit rate / log10 s). The three tilt levels (mixed
#   0.85 / 0.10 / 0.05) are independent of stride phase, so their differences are noise in
#   the binned curve. Removing them recovers the ~5 points of power the mix costs in the
#   power analysis. Set False to fit raw accuracy / RT.
RESIDUALISE_DIFFICULTY = True
