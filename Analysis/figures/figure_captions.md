# Manuscript figures — captions and supporting tables

Complete, self-contained captions for the methods figures. Journals typeset the caption as
text beneath the figure, so **the figures carry no baked-in caption**; the wording lives here.

For the same reason the figures carry **no baked-in tables**. The parameter tables are at the
bottom of this file, as real tables that can be searched, copied and restyled.

Each figure is authored at 183 mm (double-column) and written as a 300 dpi PNG and a
vector PDF. **Submit the PDF.**

| # | File | One-line message |
|---|---|---|
| 1 | `figures/adopted_design` | What a participant does, across two visits. |
| S1 | `figures/apparatus` | What the participant is standing in. |
| S2 | `figures/trial_timeline` | What happens in a trial, and when. |
| S3 | `figures/trial_flow` | How a session runs, and how the two visits fit together. |

---

## Figure 1 — The adopted design

> **Figure 1. The adopted collection design: two visits with the condition order crossed
> over.** Session schedule for both arms of the allocation. The condition leading a
> participant's first visit is read from a pre-generated permuted-block sequence
> (`ExperimentConductor.ALLOCATION`: blocks of four holding two standing-first and two
> walking-first assignments) at that participant's enrolment slot, so lanes marked
> "slot S" lead with standing and "slot W" with walking. Each condition is calibrated
> immediately before its
> own blocks, which then run contiguously; a condition is never revisited, so no
> re-calibration is needed and the walking blocks stay in one sequence. The condition
> order crosses over between a participant's two visits. That crossover is the reason the
> design uses two visits: it gives every participant all four (condition × session-half)
> cells, making the standing-versus-walking threshold contrast a within-participant
> comparison free of time-on-task rather than one merely balanced across the sample.
> Each visit yields 720 walking and 180 standing trials, pooling to 1440 and 360.

## Figure S1 — Apparatus

> **Figure S1. The ILL CAVE apparatus.** A three-quarter view into the CAVE. The
> participant walks in place on a treadmill centred in front of the front wall,
> surrounded by three projection walls. Each wall is physically two stacked 1080p panels
> exposed to its render node as a single 1920 × 2160 output; the seam is drawn on each
> wall and the six resulting half-screens are numbered 1–6. Head position is tracked by
> optical cameras (red), and each wall's view frustum is computed off-axis from the
> tracked eye position, so the projected geometry stays correct as the participant moves
> rather than being correct only from a fixed nominal viewpoint. The Gabor target is
> shown on the front wall at eye height, rendered from the stimulus shader's own
> equations at its true spatial frequency (3 cpd), envelope (σ = 0.70°) and contrast
> (0.7); **its tilt is exaggerated here for visibility** — at threshold it is a few
> degrees. Apparatus dimensions and machine topology are in Tables S1 and S4.

## Figure S2 — Trial timeline

> **Figure S2. The trial timeline, drawn to scale.** Two consecutive trials. On each, a
> Gabor patch is presented for 100 ms and the observer reports its tilt direction within
> a 2 s response window; the two patches shown are tilted left and right respectively,
> illustrating the two-alternative judgement. The response given is written inside the
> window that recorded it: the observer answers "left" on both trials, which is correct
> for the first patch (green, tick) and an error for the second (red, cross). The
> patches are rendered from the stimulus
> shader's own equations at their true spatial frequency, envelope and contrast, but
> **with the tilt exaggerated for visibility** — at threshold it is a few degrees and
> would not be resolvable at this size. Successive stimulus onsets are separated by 4.0 s
> with uniform jitter of ± 0.75 s, so the response window always closes before the next
> stimulus. In walking blocks the jitter accumulates as a bounded random walk (± 3 s)
> rather than being drawn independently each trial, so that stride-phase coverage of
> stimulus onsets does not depend on the participant's cadence.

## Figure S3 — Session and trial flow

> **Figure S3. The software flow of a session.** The three Unity scenes and the paths
> between them. Setup takes the participant number, the visit number and the
> counterbalance slot, from which the leading condition is resolved and displayed; the
> per-condition standing and walking threshold fields override calibration when set.
> Each condition is calibrated once per visit in the QUEST+ scene, whose lead-in is five
> warm-up trials excluded from the fit and whose output is written as a BIDS `ses-01`
> `_beh.tsv`. A block then runs its trial loop: a Gabor is presented for 100 ms, a
> response window of up to 2 s follows (it opens at stimulus onset and closes on the
> response), and the trial is written to Motion-BIDS. Standing blocks are timer-triggered
> and walking blocks are triggered by spatial checkpoints, but both realise the same
> nominal 4.0 ± 0.75 s onset-to-onset interval. Blocks end by returning to setup. The
> structure these blocks sit in — two visits with the condition order crossed over — is
> Figure 5.

---

# Supporting tables

## Table S1 — Apparatus

| Apparatus | Specification |
|---|---|
| Engine | Unity 2017.4.25f1 + UniCave |
| Head node | Linux (Ubuntu 24.04) |
| Render nodes | 3 × Windows, one per wall |
| Projection surfaces | 3 walls, each split top / bottom |
| Cameras / projection planes | 13 / 7 |
| Panel active area | 142 × 80 cm (65″) |
| Resolution per wall | 3840 × 2160 px |
| Eye-to-front-wall distance | ~100 cm |
| Eye height | 170 cm (Hofmann 2023 reference) |
| Pixels per degree | ~47 (15.7 px per cycle at 3 cpd) |
| Projection model | off-axis, tracked eye |
| Tracking | OptiTrack → VRPN → Unity |

## Table S2 — Trial and block parameters

| Parameter | Value |
|---|---|
| Task | 2-AFC Gabor orientation discrimination |
| Stimulus duration | 100 ms |
| Response window | 2.0 s |
| Inter-stimulus interval | 4.0 s ± 0.75 s (uniform) |
| Walking ISI jitter | accumulates, bounded ± 3 s |
| Trials per block | 180 |
| Training trials per block | 5 (excluded) |
| Difficulty mix (threshold / easy / hard) | 0.85 / 0.10 / 0.05 |
| Difficulty levels | 65 % / 75 % / 85 % correct |
| Standing blocks | timer-triggered |
| Walking blocks | checkpoint-triggered |
| Max same tilt direction in a row | 3 |

## Table S3 — Adopted design

| Parameter | Adopted value |
|---|---|
| Sessions per participant | 2 visits, ≥ 24 h apart |
| Blocks per visit | 4 walking + 1 standing |
| Walking trials (per visit / pooled) | 720 / 1440 |
| Standing trials (per visit / pooled) | 180 / 360 |
| Session duration | ~80 min |
| Condition order | permuted-block allocation |
| Order between visits | crosses over (within-participant) |
| Calibration | QUEST+, per condition, per visit |
| Calibration lead-in | 5 warm-up trials (12° → 6°) |
| Recruitment | 20 participants (minimum 12 analysable) |
| Confirmatory population test | phase-agnostic Σ\|z\|² at 2 and 4 cps |

## Table S4 — Machine topology

| Stage | Role |
|---|---|
| OptiTrack | optical head tracking; pose delivered over VRPN |
| Head node (Linux) | owns tracking, synchronises render nodes, commands the treadmill over Modbus |
| 3 × render node (Windows) | one per CAVE wall; renders that wall's off-axis view |
| 3 × CAVE wall | each two stacked 1080p panels as one 1920 × 2160 output |
