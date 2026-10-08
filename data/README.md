# ILL: visual perception while walking (Motion-BIDS dataset)

Participants judge whether a briefly shown Gabor patch is tilted left or right
(two-alternative forced choice) while walking on a treadmill or standing still in a
CAVE, with their head tracked by OptiTrack. The study replicates and extends Davidson,
Verstraten & Alais (2024), Nat. Commun. 15:2027.

This file is written by `Analysis/bidsify.py`. Change the generator, not this file.

## The lab

Recorded in the Interactive Locomotion Lab (ILL) at Bielefeld University, described in
Hofmann (2023, PhD thesis, ch. 2):

- **Treadmill:** 2.5 x 1.5 m belt (Ingenieurbüro Achilles) on a Moog six-axis motion
  platform, which stays level and still in this study.
- **Display:** six 65-inch 4K panels (Sony Bravia X85, 60 Hz) in a half hexagon, three
  columns by two rows, covering about 180 x 64 degrees. Rendered by Unity 2017.4 with
  UniCave on one Linux head node and three Windows render nodes, with off-axis projection
  from the tracked head.
- **Tracking:** 16 OptiTrack Flex 13 cameras at 120 Hz (Motive), following a rigid body of
  reflective markers on the participant's head.
- **Timing:** a stimulus appears on the panels about 40 ms after the frame on which it is
  logged, which is the panels' input lag (Hofmann 2023, App. C).

The full stimulus and display specification is in `task-ILL_stimulus.json`.

## What is where

```
data/
├── README.md                  this overview
├── dataset_description.json   dataset metadata
├── participants.tsv / .json   one row per participant: thresholds, belt speed, counterbalance slot
├── task-ILL_stimulus.json     the Gabor stimulus in retinal units (custom file, see .bidsignore)
├── sub-XX/
│   ├── sub-XX_sessions.tsv    what each session contains
│   ├── sub-XX_scans.tsv       one row per data file: condition, visit, start time,
│   │                          order, trials, duration, effective sampling rate
│   ├── ses-01/beh/            QUEST+ calibrations, one table per condition and visit
│   └── ses-02/motion/         experiment blocks, one set of files per block:
│         *_motion.tsv           per-frame head tracking and task signals (no header row)
│         *_channels.tsv/.json   the columns of *_motion.tsv, in order, and their frames
│         *_motion.json          recording metadata (nominal and effective sampling rate)
│         *_events.tsv/.json     one row per Gabor trial
├── sourcedata/
│   ├── raw/                   the original rig logs, byte for byte, original names
│   └── sub-XX/ses-YY/...      the same logs gzipped and BIDS-named (read by the pipeline)
└── derivatives/               analysis outputs, one folder per stage (see its README.md)
```

## File names

`sub-01_ses-02_task-ILL_tracksys-optitrack_acq-walking_run-03_motion.tsv`

| Part | Meaning |
|---|---|
| `sub-01` | participant (`p01` on the rig) |
| `ses-01` / `ses-02` | calibration / experiment. Sessions are parts of the protocol, **not visits** |
| `task-ILL` | the only task |
| `tracksys-optitrack` | tracking system (motion files only) |
| `acq-walking` / `acq-standing` | condition |
| `run-03` | block number within that condition, counted across both visits; for a calibration, the visit number |

The rig's name for the same block is `p01-v1-b1-t03_walking.csv.gz`: participant 01,
visit 1, block field 1 (walking), trial field 3, which becomes `run-03`.

## Visits

Every participant comes twice, with the order of standing and walking reversed on the
second visit. Both visits' calibrations are in `ses-01` and both visits' blocks in
`ses-02`; the `visit` column of `sub-XX_scans.tsv` says which visit every file belongs
to, and `condition_position` whether that condition came first or second in the visit.

## Time and sampling

The rig logs one row per rendered frame, nominally 60 Hz. Frames are not evenly spaced,
so always use the `t` channel for timing. `t` and all event times count seconds from
the first sample of the recording. `SamplingFrequency` in `*_motion.json` is the nominal
60 Hz and `SamplingFrequencyEffective` the rate the recording actually reached; the same
effective rate is listed per file in `sub-XX_scans.tsv`. A recording far below 60 Hz was
slowed down by rendering: its timing is coarser and its stimuli stayed on screen longer
than the nominal 0.1 s (the measured time is in the `duration` column of `*_events.tsv`).
Check such recordings before analysis.

## Which head channels to use

| Channels | What they are |
|---|---|
| `head_local_pos_*` | head relative to the CAVE rig: `x` lateral sway, `y` vertical bob, `z` fore-aft. Used for gait |
| `head_mocap_*` | the same head rigid body read straight from OptiTrack, including its orientation (`head_mocap_quat_*`) |
| `head_global_*` | Unity world space, which includes the travel along the virtual path |
| `head_local_ori_*`, `head_global_ori_*` | Unity transform angles. They do not carry head rotation (only the path heading shows in `head_global_ori_y`); use `head_mocap_quat_*` |

The OptiTrack stream has been seen to return, for single samples, a different tracked
object under the head's ID (the position drops by about 1.4 m and the orientation jumps
by more than 100 degrees). Filter `head_mocap_*` for such jumps before use. The
reference frames (`global`, `local`, `mocap`) are defined in every `*_channels.json`.

Missing values are written as `n/a`.

## Credits

Dataset by Christian Abele-Kretz (Cognitive Neuroscience, Biology Faculty, Bielefeld University), supervised by Prof. Dr. Christoph Kayser and Dr. Norbert Bödekker.
