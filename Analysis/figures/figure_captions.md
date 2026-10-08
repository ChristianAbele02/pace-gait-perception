# Manuscript figures — captions and supporting tables

Complete, self-contained captions for every ILL manuscript figure, written to be pasted
into the thesis. Journals typeset the caption as text beneath the figure, so **the
figures carry no baked-in caption**; the wording lives here.

For the same reason the figures carry **no baked-in tables**. The parameter tables that
used to be rendered into the images are at the bottom of this file, as real tables that
can be searched, copied and restyled.

Each figure is authored at 183 mm (double-column) and written as a 300 dpi PNG and a
vector PDF. **Submit the PDF.**

| # | File | One-line message |
|---|---|---|
| 1 | `davidson_power/fig1_effect_size` | The effect size is 0.026, not 0.071. |
| 2 | `davidson_power/fig2_per_participant_power` | Per-participant detection is unreachable. |
| 3 | `davidson_power/fig3_population_power` | The choice of population test decides the study. |
| 4 | `davidson_power/fig4_trials_vs_participants` | How trials trade against sample size. |
| 5 | `figures/adopted_design` | What a participant does, across two visits. |
| 5b | `figures/design_tradeoff` | Why the walking/standing split is 8/2. |
| 6 | `davidson_power/fig6_one_vs_two_visits` | Why there are two visits and not one. |
| S1 | `figures/apparatus` | What the participant is standing in. |
| S2 | `figures/trial_timeline` | What happens in a trial, and when. |
| S3 | `figures/trial_flow` | How a session runs, and how the two visits fit together. |

---

## Figure 1 — Effect size

> **Figure 1. The empirical stride-phase effect size, corrected.**
> **(A)** Distribution of per-participant oscillation amplitude in Davidson et al.'s
> (2024) walking data (N = 36), taken as the better of 2 and 4 cps and de-biased against a
> noise floor bootstrapped through Davidson's own binning, including its double entry of
> every target. Bars are stacked by whether Davidson's per-participant test flagged the
> participant. The broken line marks the population root-mean-square amplitude at 2 cps
> used for power analysis (0.026); the dotted line marks 0.071, the value this project
> assumed before the effect was re-derived from the raw data.
> **(B)** The three inflations that produced 0.071, removed in turn: the correct data slice
> (the combined foot split Davidson actually fit, rather than the RLR-only subset); all 36
> participants rather than the 8 his test flagged, a flag whose false-positive rate is
> ≈ 17 % against a nominal 5 %; and the pre-specified 2 cps rather than each participant's
> better of 2 and 4 cps. The corrected value is a factor of 2.8 below the superseded one.

## Figure 2 — Per-participant power

> **Figure 2. Per-participant detection is out of reach at any feasible trial count.**
> Detection power of the pipeline's confirmatory per-participant test (40 stride-phase
> bins, difficulty-residualised accuracy, occupancy-weighted fit, R² at 2 and 4 cps against
> 1000 trial-level permutations, Holm-corrected pair), against walking trials per
> participant, at the three amplitudes of Figure 1. Shaded bands are 95 % Monte-Carlo
> intervals (1000 simulated participants per point). At the population amplitude (0.026)
> power is 20 % at the adopted 1440 trials and 26 % at 2160. Reaching 80 % needs an
> amplitude only the selected flagged subset shows.

## Figure 3 — Population power

> **Figure 3. Idiosyncratic phases make the choice of population test decide the study.**
> **(A)** Davidson's 36 per-participant Fourier coefficients at 2 cps in the (cos, sin)
> plane. Each participant oscillates at their own phase. The broken circle has the RMS
> amplitude (0.026); the red arrow is the mean vector, whose length relative to the mean
> magnitude is the phase coherence R = 0.33.
> **(B)** Population power against sample size at the adopted 1440 walking trials per
> participant. "Phase-agnostic" is the confirmatory statistic Σ|zᵢ|²/sᵢ² against its null
> (`Analysis/group_agnostic.py`); "common-phase" is the one-sample Hotelling T² on the
> mean coefficient (`Analysis/common_phase.py`). Both are Holm-corrected over 2 and 4 cps,
> with 4000 simulated studies per point and 95 % Monte-Carlo bands. Vector-averaging
> coefficients that point in different directions cancels most of the effect, so at
> N = 12 the two tests reach 91 % and 15 %. The shaded band is the planned recruitment
> range.

## Figure 4 — Trials against participants

> **Figure 4. Trials per participant and sample size trade off.**
> Power (%) of the confirmatory phase-agnostic test over the simulated grid of
> participants × walking blocks (180 trials per block, 4000 studies per cell). The black
> staircase separates cells at or above 80 % from those below, drawn along cell edges
> without interpolation. The orange box is the adopted design: eight blocks at the planned
> N of 12–20, at 91–99 % power. Six blocks clear 80 % from N = 12 and four blocks only
> from N = 24.

## Figure 5 — The adopted design

> **Figure 5. The adopted collection design: two visits with the condition order crossed
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

## Figure 6 — One visit or two

> **Figure 6. Why the design uses two visits.** Population power and minimum detectable
> effect of the confirmatory phase-agnostic test (`Analysis/group_agnostic.py`:
> Σ|zᵢ|²/sᵢ² at 2 and 4 cps, Holm-corrected over the pair). The adopted two-visit design
> (1440 walking trials per participant, eight blocks over two ~80 min visits) is compared
> with the single-visit alternative a ~90 min session can hold (810 trials, four to five
> blocks). Both trial counts are simulated directly.
> **(A)** Power against sample size at Davidson's measured per-participant amplitude
> (0.026), with 95 % Monte-Carlo bands. The two-visit design reaches 80 % power at N ≈ 7,
> the single-visit design at N ≈ 19. What the second visit buys beyond power is the
> crossover of condition order, which makes the standing-versus-walking threshold
> contrast within-participant (Figure 5). A single visit cannot provide that at any
> sample size.
> **(B)** The smallest oscillation amplitude each design detects with 80 % power, from
> simulating the whole effect at multiples of the measured amplitude and reading off the
> 80 % crossing; the dotted line marks the measured value. At N = 20 the two-visit design
> still detects amplitudes down to 0.020, the single-visit design only to 0.026.

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
| Eye height | ~164 cm |
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
| Target N | 12–20 participants |
| Confirmatory population test | phase-agnostic Σ\|z\|² at 2 and 4 cps |
| Population power at N = 12 → 20 | 0.92 → 0.99 |

## Table S4 — Machine topology

| Stage | Role |
|---|---|
| OptiTrack | optical head tracking; pose delivered over VRPN |
| Head node (Linux) | owns tracking, synchronises render nodes, commands the treadmill over Modbus |
| 3 × render node (Windows) | one per CAVE wall; renders that wall's off-axis view |
| 3 × CAVE wall | each two stacked 1080p panels as one 1920 × 2160 output |

---

## Figure 5b — Design trade-off

> **Figure 5b. Reallocating the trial budget from standing to walking.**
> **(A)** Power of the confirmatory phase-agnostic population test against walking trials
> per participant, at three cohort sizes, simulated at the effect size measured from
> Davidson et al.'s (2024) raw data (4000 simulations per cell, Holm-corrected over 2 and 4
> cycles per stride). The adopted design moves from six walking blocks to eight, taking
> power at N = 12 from 0.81 to 0.93 and at N = 15 from 0.87 to 0.96.
> **(B)** What the standing blocks were buying: the standard deviation of the standing 75 %
> discrimination threshold, as a percentage of the threshold, against standing trials.
> Estimated with the fixed-slope Weibull maximum-likelihood fit the pipeline applies
> (slope 2.5 and lapse 0.05 fixed, the three tilt levels at the 65 / 75 / 85 % points mixed
> 0.05 / 0.85 / 0.10, true threshold 6.0°, 4000 simulations). Halving the standing budget
> widens the threshold standard deviation from 4.0 % to 5.7 % of threshold.
>
> The two arms are not comparable in size. The comparison the standing blocks exist for,
> the walking-versus-standing threshold contrast, is **within-participant**, so it cancels
> the between-participant threshold spread and retains only estimation error. At the
> adopted one standing block per visit its power is already 0.994 for a 1.10× threshold
> ratio and 1.000 for 1.15× (N = 12). Total session length is unchanged at about
> 80 minutes per visit. Underlying values are generated by
> `python -m Analysis.davidson_power.power tradeoff` into
> `davidson_power/design_tradeoff.json`, which the figure and the READMEs read.
