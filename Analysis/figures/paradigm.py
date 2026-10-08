"""Paradigm constants for the methods schematics, each copied from its Unity field.

Shared by the trial timeline, trial flow and adopted-design figures so they stay consistent.
"""
from __future__ import annotations

STIM_MS = 100                  # CheckpointManager.taskDuration    = 0.1 s
RESP_S = 2.0                   # CheckpointManager.responseWindow  = 2 s
ISI_S = 4.0                    # CheckpointManager.targetISI       = 4 s
ISI_JIT = 0.75                 # CheckpointManager.isiJitter (half-range of the uniform jitter)
MIX = (85, 10, 5)              # thresholdWeight / easyWeight / hardWeight, in %
TRIALS_PER_BLOCK = 180
N_TRAIN = 5                    # CheckpointManager.trainingCheckpoints
N_WARMUP = 5                   # StaircaseManager.warmupTrials
WARM_HI, WARM_LO = 12, 6       # warmupStartTiltDeg -> warmupEndTiltDeg

# Blocks per condition and visit. The confirmatory test is powered on walking trials; one standing block gives H2 power 0.994 at a 1.10x ratio.
N_STAND, N_WALK = 1, 4

BLOCK_MIN = round(TRIALS_PER_BLOCK * ISI_S / 60)   # 12 min at the mean ISI
CAL_MIN = 5                    # one QUEST+ calibration, warm-up included
WALK_BREAK_MIN = 5             # break after every second walking block
