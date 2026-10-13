# PACE — analysis pipeline

Turns the raw Unity head-node CSV logs into a **Motion-BIDS** dataset, runs
head-tracking **quality control**, plots the **QUEST+ calibration**, extracts
**gait / stride phase**, and tests **stimulus onsets against stride phase** (the
Davidson, Verstraten & Alais 2024 analysis).

The repository ships without participant data. To try the pipeline end to end, generate
synthetic participants (`python -m Analysis.simulate_participants --n 8 --keep`, see Run).

```
raw  pXX-vV-bY-tZZ*.csv[.gz]     (dropped into data/, straight off the rig)
  │  bidsify.py — archives raw into sourcedata/ as .csv.gz, moves the original to sourcedata/raw/
  ▼
data/                            (BIDS root = config.BIDS_DIR)
  ├── sourcedata/sub-XX/ses-YY/  raw Unity output, gzipped + BIDS-named
  └── sub-XX/
        ├── ses-01/beh/          *_acq-{standing,walking}[_run-NN]_task-ILL_beh.tsv   QUEST+ calibration
        └── ses-02/motion/       *_acq-{walking,standing}_tracksys-optitrack[_run-NN]_motion.tsv
                                 *_…_events.tsv   one row per stimulus
  │  calibration / qc / gait / phase / agnostic / glmm / population / conditions / report
  ▼
data/derivatives/                                README.md indexes the folders by role
  ├── qc/ calibration/ gait/                       preprocessing and quality, per subject × session
  ├── phase/ population/ conditions/ clocktime/    confirmatory and secondary tests
  ├── descriptive/                                 descriptive overview
  ├── onset_response/ gam_phase/ ddm_phase/ sdt/ stance_foot/        exploratory: responses
  ├── head_stability/ gait_dynamics/ gait_prc/ gait_phase_methods/   exploratory: walking
  ├── planning/ summary/                           planning; report.html, inventory, provenance
  └── qc/  gate.tsv (exclusion gate)  gate_ledger.tsv (N per stage)

<stage>/sub-XX/ses-YY/            data: *_<stage>.json / .npz, named after the recording
<stage>/sub-XX/ses-YY/figures/    one sheet per block: <recording>_<stage>.png
<stage>/figures/                  figures across participants
```

## Naming conventions

- One task label for the whole study: **`task-ILL`**.
- **`ses-01`** = QUEST+ calibration, **`ses-02`** = the experiment. Both visits fold into
  `ses-02`.
- **Condition is the `acq-` entity** (`acq-standing` / `acq-walking`), in the filename and
  in every derivative. Each condition is calibrated separately and uses its own threshold.
- **Blocks are the `run-` entity**, taken from the raw file's *trial* field
  (`config.run_for`), so multiple blocks per condition never overwrite each other. A
  legacy single block (trial `0`) carries no `run-`.
- Population tests **pool all of a participant's walking runs**, so a participant counts
  once on their full trial budget.
- **A reused Trial number** makes the rig write a `-1` file (`p06-v1-b1-t04_walking-1.csv.gz`).
  `bidsify` keeps both blocks: the original keeps `run-04`, the re-run gets `run-104`
  (`100·k + trial`, clear of the operator's 1–8), with a warning. The same applies to a
  visit-2 block that repeats a visit-1 number.
- Operator naming: [`INSTRUCTIONS.md`](../INSTRUCTIONS.md).

## Missing and partial data

Nothing in the pipeline needs a complete set of blocks. Every stage pools the runs a
participant has, so a lost block or a lost visit costs trials, not the participant.

- **Data quality.** `summary/data_quality.md` sums up what was recorded, kept and excluded,
  per participant, condition, block and trial, with one standard label per exclusion.
- **Missing blocks.** `report.py` writes `summary/block_inventory.tsv`. It lists, per participant
  and condition, the planned runs (`config.PLANNED_RUNS`) against those present, the missing
  runs, whole missing visits, truncated logs, runs outside the plan, missing calibrations, and
  the walking trials that actually got a stride phase. The confirmatory test already copes
  with unequal trial counts: each participant is studentised by their own permutation null
  (`group_agnostic.agnostic_test`), so fewer trials give a wider null, not a larger weight. A
  participant with no walking blocks is left out of the stride-phase tests; one with no
  standing blocks keeps them. The H5 threshold crossover (`threshold_by_condition`) needs
  both visits, because with one visit condition is confounded with time-on-task. A
  participant missing a visit is left out of those contrasts and counted in `n_incomplete`.
- **Unusable blocks** are listed with their reason in `config.EXCLUDED_RECORDINGS`. They stay in
  the BIDS dataset (`scans.tsv` column `exclusion_reason`) and QC still runs on them, which
  keeps the evidence for the decision. The exclusion gate (below) leaves them out of both
  scopes, so every analysis stage skips them. Typical reasons: a tracking failure that made the
  stimulus invalid, or a standing block whose participant has no usable walking blocks, which
  would otherwise enter the standing-vs-walking contrasts on one side only.
- **Exclusion gate** (`block_rules.py`, stage `gate`, run straight after the calibration
  verdicts and before any analysis). One file, `derivatives/qc/gate.tsv`, holds a row for every
  block and every calibration, and every analysis stage reads it. Each block is checked
  against three criteria, and each calibration against a fourth:
  1. **Chance:** accuracy over the answered main trials not above chance, one-sided binomial
     against 0.5 (`config.BLOCK_RULE_ALPHA`). Timeouts are left out rather than counted as
     wrong: a missed response says nothing about whether the tilt was seen, and with timeouts
     scored wrong a guesser's expected accuracy falls below 0.5, so the test would compare
     against the wrong null. With ~150–170 answered trials a block needs ~57 % to pass. This
     removes only blocks with no evidence the task was done. It is deliberately not a cut-off
     like 60–70 %, which would drop blocks that were merely hard and bias the
     standing-vs-walking comparison.
  2. **Reversal:** accuracy over the answered trials significantly *below* chance, which
     points to reversed answers or an invalid stimulus, not guessing.
  3. **Timeouts:** 25 % or more of the main trials unanswered
     (`config.BLOCK_RULE_MAX_TIMEOUT_FRACTION`). A block with that many missed responses was
     not done as instructed, and its accuracy would rest on a self-selected subset of trials.
  4. **Calibration:** an UNRELIABLE calibration is left out of the pooled calibration refit
     and the slope-invariance test. Its blocks are not excluded for it; they are judged by
     1–3 on their own responses, and the verdict is recorded with them in the gate (the
     latest calibration of that condition and visit counts). An off-target threshold makes
     the levels harder or easier than planned, which the difficulty residualisation absorbs.

     A calibration is UNRELIABLE when `calibration.py` raises a blocking flag: threshold at
     the grid ceiling (no real threshold) or near the floor, a difficulty level clamped to a
     grid bound, a wide posterior, inattention on the catch trials, or high-tilt lapses.
     High-tilt lapses (errors above 2× threshold) count only when there are significantly
     more of them than the QUEST+ model's own 5 % lapse rate predicts (one-sided
     Poisson-binomial, α = 0.05).

  Easy-trial accuracy is deliberately not a criterion. At ~17 easy trials a binomial test on
  them wrongly fails an engaged participant 10–40% of the time, so it stays a QC warning.

  The gate has two **scopes**, because the rule judges the *responses*:
  - `behaviour` — every analysis of the responses (stride-phase tests, GLMM, population,
    clock-time, conditions, H5 thresholds, descriptive, onset/response, GAM, DDM, stance
    foot, SDT). Excluded: the manual list plus every block the rule fails.
  - `gait` — analyses of the walking alone (gait extraction, phase methods, head stability,
    gait dynamics, gait PRC). Excluded: the manual list only. A participant answering at
    chance still contributes their walking.

  `bids_io.find_recordings(scope=...)` applies it (default `behaviour`, the strict one). The
  rule is switched on by `config.APPLY_BLOCK_RULE`; with it off the verdicts are still
  written and only the manual list excludes. Which blocks are currently out, and why, is in
  `gate.tsv` and `summary/exclusions.tsv`.

  **Kept current.** `bids_io.load_gate` rebuilds the gate before use whenever it is stale:
  blocks or calibrations added or removed, inputs newer than the gate, or the exclusion
  settings in `config.py` changed. A stage run alone (`--only phase`) therefore never works
  from an old gate.

  **N per stage.** Every analysis stage prints one line, e.g.
  `N [phase | behaviour gate]: 2/3 participants, 7/17 walking blocks; excluded ... -> analysed n=2`,
  and writes it to `derivatives/qc/gate_ledger.tsv` (participants and blocks the gate let in,
  what it excluded, and `n_analysed`, the participants that reached the stage's result; lower
  when the stage drops some itself). Each row carries the gate's fingerprint, so a row from an
  older gate shows. The full pipeline prints the ledger at the end, and the report shows it.
  Stages that write per-participant files also delete those of excluded participants, so an
  old result cannot reappear in the report. `summary/exclusions.tsv` lists each block's
  decision for both scopes, with the non-blocking warnings (poor head data, side bias, low
  L/R alternation, unreliable foot labels).
- **A block that ends early** (crash, disconnect, a copy cut off). `bidsify` keeps every
  whole frame before the break and drops a half-written last row. It reads a cut-off
  `.csv.gz` up to the cut, and marks the file `log_status = truncated` in `scans.tsv`. A log
  with no usable frames (empty, header only) is skipped with a warning; it no longer stops
  the rest of the dataset converting.
- **Gaps inside a block** (tracking dropout, stale mocap frames, a logger hitch, the belt
  stopping). The resampler bridges a gap with a straight line, so `gait.py` treats any stretch
  with no real head sample for > 0.25 s (at least 3 frame intervals) as a gap. Troughs and
  Hilbert phase within 0.5 s of it are discarded. `phase_at` gives no phase inside a
  step longer than 1.6× the block's median step, which is either a gap or a missed trough.
  Trials there are left out of the stride-phase tests only. Their count per block is
  `n_trials_no_phase` in the phase JSON; the block's gap count and length are `n_data_gaps` /
  `data_gap_total_s` in the gait JSON.
- **Mocap diagnostic channels not recorded.** Stale frames are normally read from
  `mocap_frame_advanced`. In some sessions that flag and the other
  `head_mocap_*` channels are logged as zeros while the head position itself streams
  normally. Reading such a flag literally would mark every frame stale and discard the block.
  `clean.stale_frames` therefore uses the flag only when at least half its rows are fresh;
  otherwise it finds freezes as runs of identical head positions lasting ≥ 0.1 s. On sessions
  with a working flag this catches every frame the flag marks. QC reports which source was
  used (`stale_pose.source`). Without the mocap quaternion there is no head orientation, so
  `head_stability.py` and `gait_phase_methods.py` skip such blocks.

## Run

```bash
pip install -r Analysis/requirements.txt
python -m Analysis.run_pipeline          # all 28 stages, in dependency order
python -m Analysis.run_pipeline --no-cache   # the same, recomputing everything

# or stage by stage:
python -m Analysis.bidsify               # raw CSV → BIDS + sourcedata gz
python -m Analysis.calibration           # QUEST+ convergence + reliability check
python -m Analysis.calibration_refit     # offline slope/lapse refit (--bids for real data)
python -m Analysis.qc                    # tracking / behaviour QC
python -m Analysis.gait                  # gait template, stride cycle, head trace
python -m Analysis.phase_analysis        # per-participant stride-phase modulation
python -m Analysis.group_agnostic        # CONFIRMATORY phase-agnostic population test
python -m Analysis.common_phase          # SECONDARY common-phase test (Hotelling T2)
python -m Analysis.glmm                  # GLMM amplitude/phase estimates (no p-values)
python -m Analysis.population            # prevalence & phase + stride-duration controls
python -m Analysis.clocktime_control     # clock-time (Hz) vs stride (cps) control
python -m Analysis.conditions            # standing vs walking
python -m Analysis.threshold_by_condition  # H5: per-condition 75 % threshold, paired
python -m Analysis.descriptive           # exploratory dataset overview
python -m Analysis.onset_response        # EXPLORATORY: onset vs response locus
python -m Analysis.gam_phase             # EXPLORATORY: cyclic-spline GAM + mutual information
python -m Analysis.ddm_phase             # EXPLORATORY: EZ-diffusion decomposition
python -m Analysis.stance_foot           # EXPLORATORY: stance-foot (left/right) contrast
python -m Analysis.sdt_phase             # EXPLORATORY: d-prime vs criterion decomposition
python -m Analysis.report                # summary TSV + HTML report
```

`gait` reuses an existing `*_gait.npz`; delete it (or the `gait/` derivatives) to rebuild the
stance-foot labels from scratch.

```bash
python -m Analysis.simulate_participants --n 8 --keep   # end-to-end check on synthetic data
python -m pytest Analysis/tests -q       # inject-and-recover validation suite
ruff check Analysis/                     # lint gate (F + E9); CI runs this
```

**Skipping unchanged work** (`stage_cache.py`). Two steps reuse earlier work when nothing
they depend on has changed: bidsify does not rewrite a `_motion.tsv` whose table is
identical by value, and `bayes_glmm` reuses its fit when the trial arrays, its code, the
priors, the sampler settings and the library versions are all the same. A step is reused
only when that content fingerprint matches AND every output it recorded is still on disk,
unmodified; otherwise it is recomputed. Nothing is ever deleted by the cache. Its manifests
live in `.pipeline_cache/` next to `data/` (git-ignored; deleting it costs one full run).
Use `--no-cache` (or `ILL_NO_CACHE=1`) for a clean full recompute, e.g. before reporting
final results. Each run ends with the time per stage.

Figures are written as PNGs into a `figures/` folder next to the data they show (headless
`Agg` backend). A block gets one sheet per stage (`qc`, `gait`, `phase`, `clocktime`), and
`summary/report.html` shows them all on one page; the run prints the count per stage at the
end. `report.py` also stamps a `provenance.json` (git SHA, package
versions, config snapshot) so each figure is traceable to exact code and parameters.

## Modules

| File | Role |
|---|---|
| `config.py` | paths, constants, filename parsing, session/subject/condition/run mapping, `PHASE_METHOD`. |
| `bidsify.py` | raw Unity CSV → Motion-BIDS, keyed on `(sub, ses, datatype, condition, run)`; archives raw into `sourcedata/`; carries 6-DOF mocap channels and onset pulses. |
| `bids_io.py` | discover and load BIDS recordings; uniform resampling. |
| `clean.py` | head-position spike/dropout detect-and-repair. Short gaps interpolated, long gaps **excised, never interpolated across**. Runs **before** the gait band-pass — a sharp spike otherwise rings through the filter at ~30× the real bob and wrecks trough detection. |
| `calibration.py` | QUEST+ convergence figure and reliability check: wide posterior, A/B divergence, high-tilt lapses beyond the model lapse rate, catch-trial inattention, ceiling and floor rails. |
| `calibration_convergence_sim.py` | re-implementation of the dual-tracker QUEST+; sets and validates the runtime stop and reliability rules. |
| `calibration_refit.py` | offline MLE refit of slope + lapse via a joint fit (lapse free or fixed), with rail detection; `--bids` fits per condition (QUEST+ and catch trials, warm-up left out) and runs a slope-invariance likelihood-ratio test. The pipeline step simulates that refit at the rig's calibration design (QUEST+ placement, real threshold spread and trial counts, catch trials; 200 repetitions for 12 and 24 calibrations, cached) and compares three lapse strategies in `calibration_refit_simulation.png`. |
| `plot_calibration.py` | standalone CLI: convergence figure from a raw calibration CSV. |
| `qc.py` | timing, tracking-channel, cleaning, treadmill and behaviour QC; writes one QC sheet per block (`*_qc.png`: head position raw vs cleaned with the size of every correction, belt speed, onsets, accuracy by level). |
| `gait.py` | band-pass, trough (double-support) detection, stride cycle, gait template, `phase_at()`, Hilbert phase, effective-fs QC, and the **stance-foot labelling** (`label_steps`): per-step lateral-sway evidence, decoded under an alternation prior (Viterbi) and cross-checked against an independent rigid-body roll channel. Writes one gait sheet per block (`*_gait.png`: stride, head path, stance-foot decode). |
| `phase_analysis.py` | 40-bin stride binning, Fourier fit 0.1–10 cps, **weighted by bin occupancy** and run on **difficulty-adjusted** accuracy (primary) and RT (secondary) (§4b). **Primary per-participant test = confirmatory at 2 and 4 cps (Holm)**; the full-grid max-R² scan is exploratory. Significance by 1000× trial-level permutation with re-binning. |
| `group_agnostic.py` | **THE CONFIRMATORY POPULATION TEST.** `S(f) = Σᵢ\|zᵢ\|²/sᵢ²` at 2 and 4 cps, Holm-corrected, **studentised** by each participant's own null so no one dominates the sum by being noisy (under H0 each contributes 1, so `S ≈ n`). Runs on the **primary** measure (accuracy) plus two secondary ones; only accuracy carries a family-wise claim. Invariant to rotating any individual `zᵢ`, so it detects oscillation whatever phase each participant prefers — the point at Davidson's coherence of R = 0.33, where it reaches 91 % power at N = 12 × 1440 against a common-phase contrast's 15 %. Its null reuses `phase_analysis`'s own permutations, so it is exact by construction. |
| `common_phase.py` | **THE COMMON-PHASE POPULATION TEST (secondary).** One-sample **Hotelling T²** on the per-participant (cos, sin) coefficients at 2 and 4 cps, Holm. Exact in small samples; measured FPR **0.05** in the idiosyncratic-phase case where the old GLMM Wald scored **0.60**; bit-identical to the statistic `davidson_power` simulated, so the pipeline now agrees with its own power analysis. Carries the locomotion-specificity test as a **paired** T² on walking − standing. Secondary by design: at R = 0.33 the vector mean cancels most of the effect. |
| `glmm.py` | trial-level hierarchical GLMM (participant random intercept + random cos/sin slopes). **Descriptive only and now p-value-free** — amplitude, preferred phase, and the condition and difficulty breakdowns. Its variational Wald was removed rather than annotated (§4d); the same model with valid uncertainty, RT and a standing control is `bayes_glmm.py`. |
| `bayes_glmm.py` | **Bayesian mixed models (secondary), NumPyro NUTS.** One model per measure — accuracy (Bernoulli-logit, answered trials) and log RT (Student-t) — over walking and standing trials: participant random intercept and walking slope, difficulty, and per condition the 2 and 4 cps cos/sin terms with a population mean (shared phase) and isotropic participant slopes (per-person modulation at their own phase). Standing trials carry a virtual stride phase (onset on a clock at the participant's walking stride time, seeded random start), which keeps the stimulus timing and reads no motion signal; a head-sway phase is not a clean null, because its zero-phase filter lets stimulus- and response-evoked head movement leak back to the onset. Reports amplitudes in hit-rate and RT units with 95 % HDIs, Savage–Dickey Bayes factors for the shared phase (2-D) and the per-person scale (boundary, reflected KDE), the walking-minus-standing contrasts and the walking main effect. Phase priors are anchored on Davidson's per-participant amplitude (N(0, 0.2) logit per component, τ ~ HalfNormal(0.2)). `--validate` checks the Bayes factors by simulation on the real design: how often BF10 > 3 appears with no effect, and how often a Davidson-sized (and a three times larger) effect at idiosyncratic phases is detected. |
| `power.py` | per-participant detection power by Monte-Carlo over the same confirmatory test, disk-cached. |
| `population.py` | grouped R² spectra, Bayesian prevalence de-biased with the measured sensitivity (Ince 2022) plus an identifiability guard, preferred-phase polar + Rayleigh, and the stride-duration controls (Supp 7/8). |
| `clocktime_control.py` | is the oscillation locked to the stride or to any clock? Folds onsets onto a fixed temporal period (Hz) with the same machinery (Supp 5/6). |
| `conditions.py` | standing vs walking hit rate / RT / by-difficulty. |
| `threshold_by_condition.py` | H5 — fits a fixed-slope Weibull to each condition's experiment trials, reads off the 75 % threshold and compares the two in a paired test. Slope and lapse come from `calibration_refit`'s per-condition joint fit (three tilt levels cannot identify a slope), so "walking shifts threshold, not slope" is tested rather than assumed. |
| `calibration_transfer.py` | whether each calibrated threshold holds in the blocks run on it. Per calibration (participant × condition × visit, behaviour-gate blocks pooled), the blocks' own 75 % threshold (QUEST+ slope, lapse and guess fixed, profile 95 % CI) against the calibrated one, with a simulated sampling null at the blocks' tilts; counts the calibrations outside their null range (test against 5 %) and estimates the calibration error from the excess spread of log(block / calibrated). A fit at ceiling or chance is a bound only and stays out of the spread. Figure: accuracy at the calibrated tilt for every block, and calibrated vs block threshold. |
| `descriptive.py` | exploratory dataset overview with Wilson CIs: behaviour, gait and tracking quality. No inferential stats. |
| `onset_response.py` | EXPLORATORY — stimulus-onset vs response-onset locus. Both re-expressed onset-referenced; the dissociation is the circular Δψ with a trial-bootstrap CI. |
| `gam_phase.py` | EXPLORATORY — cyclic-spline GAM (flags non-sinusoidal structure via AIC against a single cosine) plus mutual information as a model-free omnibus. |
| `ddm_phase.py` | EXPLORATORY — per-bin EZ-diffusion maps (accuracy, RT) → (v, a, Ter) to identify the carrier: sensory drift, caution, or motor timing. A screening decomposition. |
| `stance_foot.py` | EXPLORATORY — does performance depend on **which foot is in stance**? A one-d.o.f. contrast on the labels `gait.label_steps` provides, tested against a **stride-flip permutation null** (§4a). Population level mirrors `group_agnostic`: a direction-agnostic `S = Σ zᵢ²` (primary, valid even before the rig's sway-sign convention is verified) and a direction-specific `T = mean zᵢ` (secondary, provisional until it is). |
| `sdt_phase.py` | EXPLORATORY — **is an accuracy oscillation sensory or decisional?** Phase-resolved `d'` and criterion `c` from `stimulus_side` × `response` (§4c), same 2/4 cps confirmatory machinery, permutation null that shuffles the (stimulus, response) pair. Population level reuses `group_agnostic.agnostic_test` verbatim. |
| `phaser.py` | multi-channel stride phase (vertical bob + lateral sway + head roll; protophase-corrected, precision-weighted), after Revzen & Guckenheimer (2008). Selectable with `PHASE_METHOD = "phaser"`; not recommended on current data (§5). |
| `gait_phase_methods.py` | EXPLORATORY — scores trough, Hilbert and multi-channel phase on head signals none of them uses (§5). |
| `head_stability.py` | EXPLORATORY — head pitch against the vertical bob over the stride: step-band coupling, head fixation point, gaze compensation at the 2 m stimulus distance (§5). |
| `gait_dynamics.py` | EXPLORATORY — stride-time CV, DFA long-range correlations and short-term local dynamic stability per walking block (§5). |
| `gait_prc.py` | EXPLORATORY — do stimulus onsets or button presses shift the gait rhythm? Phase response curve against a frame-locked random-time null (§5). |
| `block_rules.py` | builds the exclusion gate `qc/gate.tsv` (chance / reversal / timeouts per block, unreliable calibrations out of the refit, plus the manual list), with a behaviour and a gait scope; read by every analysis stage, which logs its N to `qc/gate_ledger.tsv` (see Missing and partial data). |
| `report.py` | dataset summary TSV + self-contained HTML report; writes `provenance.json`, `exclusions.tsv` (each block's gate decision for both scopes, plus non-blocking warnings) and the N-per-stage ledger. |
| `data_quality.py` | data-quality overview, run by `report.py`: blocks and trials recorded, kept and excluded per participant x condition, per condition and in total, each exclusion with a standard label (chance-level accuracy, too many timeouts, reversed responses, technical failure, no valid walking, unreliable calibration; missing blocks as not recorded or visit pending). Writes `summary/data_quality.md` and two TSVs; manual exclusions are labelled in `config.EXCLUSION_LABELS`. |
| `prevalence_planning.py` | prevalence-precision (HPDI width) vs N planning simulation. |
| `simulate_participants.py` | N synthetic participants in the two-visit topology into a separate `bids_sim/`, run through the whole pipeline. Trials come from a Davidson-calibrated generative drift-diffusion model; the injected truth is written to `sim_truth.json`. |
| `tests/test_pipeline.py` | inject-and-recover tests for the Fourier test, GLMM, Hilbert-vs-trough agreement, calibration refit, prevalence guard, spike cleaning, BIDS naming, the confirmatory phase-agnostic test, and each exploratory add-on. |
| `gabor_preview.py` | stimulus design aid (not part of the pipeline): renders the Gabor at true visual angle and sweeps each parameter. |
| `run_pipeline.py` | orchestrates all 28 stages, prints the time each took, and lists the figures produced; `--no-cache` recomputes everything. |
| `stage_cache.py` | content-fingerprint cache that lets a step skip work whose inputs, code and outputs are unchanged (see Run). |
| `davidson_power/` | effect size measured from Davidson's raw data + power simulations of this pipeline's own tests. See [its README](davidson_power/README.md). |

## Stance-foot identity (which foot is down)

The vertical head bob says *where in the step* a target fell; it cannot say *which foot was
down*. That distinction matters for exactly one reason, and it is worth being precise about
it, because the obvious motivation is the wrong one:

> **It does not affect the confirmatory result.** The confirmatory test runs at **2 and 4
> cps**. A half-stride shift is one whole cycle at 2 cps and two at 4 cps, so the even
> harmonics are *exactly invariant* to which foot is which. Swapping every foot label would
> leave `phase_analysis` and `group_agnostic` bit-identical.

What foot identity actually unlocks is the **odd harmonics (1, 3 cps)** — which *are* the
left/right asymmetry of the stride — and the direct question `stance_foot.py` asks.

### Labelling (`gait.label_steps`)

Lateral sway oscillates **once** per stride, because the head and trunk displace toward the
stance limb through single support (the CoM has to pass over the base of support). The sign
of the mean sway over a step therefore names the foot. Three things turn that from a hint
into a usable label:

| Step | What it does | Why |
|---|---|---|
| Robust scaling (`_normalise`) | per-step means ÷ MAD | sway amplitude varies several-fold between participants and with speed; this makes "one unit of evidence" mean the same thing everywhere, so one fixed decoder gain is defensible |
| Evidence clip (`MARGIN_CLIP = 2.0`) | bound \|per-step evidence\| in MAD units | beyond a couple of MADs a bigger number is not stronger physiology, it is an artefact — the band-pass edge transient inflates the first and last step's mean by ~6× the 99th percentile of the real distribution |
| **Alternation decode** (`_viterbi_alternate`) | 2-state HMM, `ALT_PRIOR = 0.9995`, Viterbi | real gait alternates, but sign-of-sway labels each step *independently*, so one noisy step flips on its own — physiologically impossible, and the error that destroys odd-harmonic power |
| **Roll cross-check** (`_pick_roll`) | an independent rigid-body orientation DOF | corroboration from a different measurement, not a restatement of the same one. Fused into the evidence only at ≥ `ROLL_FUSE_MIN_AGREE` (70 %) agreement; otherwise reported and left out |

The roll channel is **not** assumed. `ROLL_CHANNELS` prefers the mocap rigid-body Euler over
the Unity head Transform — the logger documents that Transform as possibly a "yaw-only
container" — and which *axis* carries mediolateral tilt depends on the Motive rigid-body
definition, so `_pick_roll` selects on the signal's own frequency signature: roll oscillates
once per stride, whereas anything riding on the vertical bob oscillates twice. Selecting on
frequency rather than on agreement with sway is what keeps the cross-check honest — the
criterion never looks at the sway signal's *phase*, which is what carries the label.

### The left/right naming is gated

`head_local_pos_x` is `trackingOrigin.InverseTransformPoint(head.position).x`. Unity is
left-handed, so for a participant facing +z, +x is their right — and since the head leans
toward the stance limb, positive sway means right stance. **That chain is a geometric
assumption about the rig, not a measurement**, so `config.LATERAL_SIGN_VERIFIED` gates it:

- **`False` (current).** Steps are still separated reliably and named **"foot A" / "foot B"**.
  Everything that needs only *the two feet told apart* stays valid — the stride origin, the
  odd harmonics, and the **direction-agnostic** population test.
- **`True`.** Labels are named left/right and the direction-specific test becomes an
  anatomical claim.

The three decoder constants are set together to give **one statable guarantee**, asserted
numerically in `test_no_single_step_can_break_the_alternation` so a future tweak cannot
silently void it:

- **an isolated flip is impossible** — flipping one step against its neighbours turns two
  transitions into "stay" (cost `2·log(p/(1−p))` = 15.2) while a single step's evidence can
  buy at most `2·EMISSION_GAIN·MARGIN_CLIP` = 12;
- **a sustained parity shift stays cheap** — a real missed trough shifts the parity of
  everything after it, pays that penalty *once* (7.6), and is then supported by every
  subsequent step.

**To verify:** record one walk-in where the operator calls the first step, run
`python -m Analysis.gait`, and read the first shaded step off panel (e) of
the block's `*_gait.png` sheet. If it is inverted, flip `LATERAL_POSITIVE_IS_RIGHT` — never a sign
somewhere in the analysis code.

Per-recording diagnostics (`*_gait.json`, and the `foot_label_verdict` gate): raw vs decoded
alternation, steps relabelled, % low-evidence steps, the roll channel used and its agreement.
`lr_alternation_pct` deliberately keeps its old meaning — the **raw**, unconstrained rate —
because the decoded rate is ~100 % by construction and would certify itself.

### 4a. The stance-foot contrast (`stance_foot.py`, EXPLORATORY)

**Statistic.** `Δ = mean(y | foot A) − mean(y | foot B)` on a participant's pooled walking
trials; accuracy primary, log₁₀ RT secondary. One degree of freedom — far better powered
than a 1 cps R² fit, and it pools across participants without a phase-alignment problem.

**Null: flip the foot label of whole strides.** Under H0 the two half-strides of a stride are
exchangeable labels, so this is an exact randomisation — and it is what makes the test clean:
a per-stride flip leaves every trial's **within-step phase**, its **time on task** and the
**trial count** untouched, and moves only the foot bit. A 2 cps oscillation, a fatigue drift
or any block structure therefore *cannot* manufacture an effect (asserted in
`test_stance_foot_immune_to_within_step_oscillation`). It is also the exact null of the
ambiguity the labels resolve: randomising the foot bit is what the pipeline was implicitly
forced to do before the labels existed.

**Population**, mirroring `group_agnostic`'s reasoning and for the same reason (Davidson's
cross-participant coherence is only R = 0.33):

| Test | Statistic | Asks | Status |
|---|---|---|---|
| direction-**agnostic** | `S = Σᵢ zᵢ²`, `zᵢ = Δᵢ / sd(nullᵢ)` | does *anyone* show an asymmetry, each in whatever direction | **primary** — rotation-free, so valid even while the sway sign is unverified |
| direction-**specific** | `T = mean zᵢ` | does it point the same way in everyone | secondary, and **PROVISIONAL** until `LATERAL_SIGN_VERIFIED`: before then only the magnitude means anything, not the sign |

Both nulls are assembled from the participants' own permutations, so they are exact by
construction rather than assumed. Each participant is standardised by their own null SD
before pooling, so a longer run does not dominate the sum for being longer.

Also reported per participant: a **confident-steps-only** sensitivity analysis (trials whose
step carried strong labelling evidence — a real effect should survive it), and the two feet's
within-step phase coverage as a visible balance check. Runs whose `foot_label_verdict` comes
back unreliable are dropped by this module and **only** by this module; nothing confirmatory
depends on foot identity, so a bad label is never a reason to exclude a run.


### 4b. Estimator choices in the binned fit (`config.WEIGHT_BINS_BY_N`, `RESIDUALISE_DIFFICULTY`)

Two variance reductions, both **on by default**, both switchable in `config.py`. Neither can
create an effect: each is applied identically to the observed data and to every permutation
in the null, so the test stays exact — the unit tests pin exactly that.

**Bins are weighted by their trial count.** A bin mean has variance `σ²/n`, so an unweighted
fit lets a thinly populated bin inject as much noise as a full one. Bin occupancy really does
vary: the checkpoint jitter is a *reflected random walk* precisely so onset phase decorrelates
from cadence, and its stationary distribution leaves a residual Rayleigh `R ≈ 0.07`. Set
`WEIGHT_BINS_BY_N = False` to reproduce Davidson's unweighted fit exactly. (Count mode is
never weighted — there the bin count *is* the dependent variable.)

**Difficulty levels are centred before binning.** The design mixes threshold / easy / hard at
0.85 / 0.10 / 0.05 and draws the level independently of stride phase, so which level a trial
happened to get is pure nuisance variance in the binned curve — it shifts a bin mean according
to that bin's accidental difficulty composition and carries no phase information. Each level's
own mean is subtracted and the grand mean added back, so the curve stays on the hit-rate (or
log₁₀ s) scale and cross-participant coefficients stay comparable. The power analysis prices
the difficulty mix at roughly 5 points; this recovers most of that for no trials. It matters
more for RT than for accuracy, since an easy tilt is answered markedly faster.

> **Not a substitute for a difficulty-resolved analysis.** Adjusting removes the level
> *offsets*; it does not test H3 (is modulation largest at threshold?). At 1440 walking trials the
> mix yields ~144 easy and ~72 hard trials, which cannot support a 2 cps fit. H3 needs its
> own design, and the README's hypothesis list should not imply otherwise.

### 4c. Sensitivity vs criterion (`sdt_phase.py`, EXPLORATORY)

Accuracy cannot say *what* oscillates. In a left/right discrimination a hit rate is the product
of **sensitivity** (how well the tilt is seen) and **criterion** (a standing preference for one
response), and a criterion that swings over the stride depresses accuracy exactly as a
sensitivity dip does. Davidson et al. leave the decomposition open, and a response-side bias is
exactly what the criterion measures.

Nothing extra has to be recorded: `events.tsv` already carries `stimulus_side` and `response`,
and the direction pool is balanced 50/50 by design. Treating "tilt right" as the signal,

```
H = P(said right | tilt right)     F = P(said right | tilt left)
d' = z(H) − z(F)                   c = −½ (z(H) + z(F))
```

with the **log-linear correction** (Hautus 1995: +0.5 to each frequency, +1 to each denominator,
applied unconditionally) so every bin's `d'` is finite and the estimator stays continuous.

| Choice | Why |
|---|---|
| **20 bins**, not 40 | each bin must support *two* rates, so it needs ~2× the trials; at ~1440 pooled walking trials that is ~72/bin, ~36 per cell |
| **pooled walking runs only** | a single ~180-trial run would put ~4 trials in each cell, and no correction rescues that |
| **timeouts excluded** | a trial with no press is neither a hit nor a false alarm; there is no defensible cell for it. (Whether *timing out* is phase-dependent is a separate question, not tested here; see 4e.) |
| null shuffles the **(stimulus, response) pair** | preserves every trial's outcome and both marginals, and moves only *when* each outcome occurred — the exact randomisation of "neither sensitivity nor criterion depends on phase" |
| population test is **`group_agnostic.agnostic_test` itself** | reused verbatim, not reimplemented, so the two modules cannot drift apart |

Validated by injection: a pure `d'` oscillation is detected by the `d'` test and **never** by
the criterion test, and a pure criterion oscillation the other way round — the selectivity that
makes the decomposition worth reading. False-positive rate on a biased but non-oscillating
observer is at or below nominal.

Status is **exploratory**: it identifies what an accuracy effect is made of, it does not test
whether one exists.


### 4d. Common-phase inference: Hotelling T², not the GLMM Wald

The GLMM reports **no** fixed-effect Wald test, because a p-value in a results file is read
as a p-value however it is captioned, and this one would be invalid.

The defect is structural. `BinomialBayesMixedGLM` fits by variational inference; mean-field
posteriors understate variance, and the fixed-effect SEs on (cos, sin) cannot absorb the
random-slope variance carried on those same two regressors.

| test | null, no slope variance | null, **idiosyncratic phases** |
|---|---|---|
| GLMM variational Wald | 10 % | **60 %** |
| **Hotelling T² (`common_phase.py`)** | **4 %** | **9 %** |

The second column is the one that matters — idiosyncratic preferred phase is the situation
this study is designed around (R = 0.33).

`common_phase.py` runs a one-sample Hotelling T² on the per-participant coefficient pairs,
at 2 and 4 cps with Holm over the pair. Two properties beyond the error rate: it is **exact
in small samples** (F reference, not asymptotic χ², which matters at N = 12–20), and it is
**the statistic `davidson_power/power.py` already simulated** — the 15 % power figure
in the design comes from it, so the pipeline and its own power analysis now use one
estimator. A regression test pins the two implementations to 1e-12.

Coefficients enter **unstandardised** here — the opposite of `group_agnostic`'s choice, and
deliberately so. There the sum of squares needs every participant on a common precision
scale; here `S` is the *between-participant* covariance and already carries the
heterogeneity, so a participant with a wider individual null widens `S` and makes the test
conservative rather than anti-conservative.

Locomotion specificity is the **paired** form: walking minus standing coefficient per
participant, against the origin. Standing's phase axis stays the matched Hilbert surrogate;
pairing removes each participant's own offset.

> **Secondary by design.** This asks whether the population oscillates at a *common* phase.
> At R = 0.33 the vector mean cancels most of the signal — 15 % power against the
> phase-agnostic test's 91 %. A null here beside a positive `group_agnostic` result is the
> **expected** pattern, not a contradiction, and the console output says so where it prints.

### 4e. Two measures, and what is not analysed

The pipeline analyses **accuracy** (primary) and **reaction time** (secondary).
**Response likelihood is not analysed.** Two independent reasons:

*Wrong quantity.* In Davidson's **detection** task "responded" meant "detected", which made
the response histogram a sensitivity measure. This is a **forced-choice discrimination** —
every trial demands a press — so "responded" means only "did not time out". The measure was
carried over by name, not by meaning.

*Mis-specified.* Binning **response** phases (onset + RT) against a null drawn from
**onset** phases compares two different distributions. A response-phase distribution is the
onset distribution convolved with the RT spread, hence systematically smoother, so the
observed R² sits below its own null by construction — conservative to the point of being
inert, and outright degenerate (p ≡ 1) when the timeout rate approaches zero.

Whether *timing out* is phase-dependent is a real question that stays untested here. It
needs a test built for it.

**H3 (modulation largest at threshold) is not tested.** Not for want of theory: the
0.85 / 0.10 / 0.05 difficulty weighting exists to serve H1, and it leaves ~144 EASY and ~72
HARD trials at 1440 walking trials — too few for a 2-cps fit. The weighting that serves H1 is what makes H3
untestable, and H1 is the study. `glmm.fit_difficulty_model` reports amplitude per level as
description, with the trial count printed above each bar.

## 5. What the walking itself shows (exploratory gait analyses)

Four analyses use only the head motion and the event *times*, never the answers. None is
confirmatory; each is a description of how people walk in the CAVE while doing the task.

**5a. Does the task reset the gait rhythm? (`gait_prc.py`)** Walking behaves like a
limit-cycle oscillator, and an oscillator's response to a brief input depends on where in
its cycle the input lands: the phase response curve (PRC). Funato et al. (2016) measured
the walking PRC with brief treadmill-belt perturbations. Here the inputs are the task's own
events. For each Gabor onset and each button press, the pre-event rhythm (mean of the
last four steps) predicts when the next troughs should come. The shift after four steps
(two full strides), as a % of a step, is how much earlier (−) or later (+) they came. It
must be an even number of steps: an odd window holds one more step of one foot than of
the other, so ordinary left/right asymmetry would make the shift depend on stride phase
even at random times. The null is the same
measure at random *logged-frame* times in the same block: events can only happen on a
logged frame, and at a low frame rate trough times are biased toward frame times, so a
continuous-time null would differ for a sampling reason. Tests: mean shift against random
sets of the same size, and PRC shape (2-harmonic R²) against the same. Onset and press are
~0.8 s apart and not independent.

**5b. Head stabilisation (`head_stability.py`).** The head pitches against the vertical bob
so the line of sight stays on a point ahead (Pozzo et al., 1990): the **head fixation point**
of Hirasaki et al. (1999). Pitch is measured as rotation relative to the block's median head
pose, about the horizontal room axis that carries the most step-frequency power (the rigid
body's own axes are not aligned with the room). Reported per block: step-band coherence and
phase between bob and pitch, the fixation-point distance, and the compensation of gaze error
for a target at the 2 m stimulus distance, plus stride-phase profiles. **The pitch sign is not
verified** (`config.HEAD_PITCH_SIGN_VERIFIED`): the observed step-band phase (~125–170°) reads
as anti-compensatory, which contradicts the literature, so a mirrored rotation sense is the
likelier explanation. Compensation is therefore also reported for the mirrored sign. This
measures the head, not the retinal motion of the Gabor, which is head-locked and drawn from a
render eye that keeps only head motion below 1.8 Hz.

**5c. Stride-to-stride dynamics (`gait_dynamics.py`).** Stride-time CV; detrended fluctuation
analysis (Peng et al., 1995) of the stride-interval series, whose healthy value shows
long-range correlations (Hausdorff et al., 1995) that treadmill walking alters (Terrier &
Dériaz, 2011); and the short-term local dynamic stability λS (Rosenstein et al., 1993;
Dingwell & Cusumano, 2000; reviewed by Bruijn et al., 2013) of time-normalised 3-D head
velocity. DFA boxes run 16..N/4 strides, fixed rather than tuned (Damouras et al., 2010).
Tracking gaps break the stride series; each unbroken run is integrated on its own and never
spliced. Stride times come from head-bob troughs, so CV includes event-detection noise:
compare it within participants, not with foot-sensor studies.

**5d. Which stride phase is best? (`gait_phase_methods.py`, `phaser.py`)** A Phaser-style
multi-channel phase (Revzen & Guckenheimer, 2008) combines the bob, lateral sway and head roll,
each corrected from protophase to phase (Kralemann et al., 2008) and weighted by its precision.
On synthetic data it is 2–3× more accurate than any single channel. On the real head data it
was not better. Scored on held-out signals (head pitch and fore-aft head position, which no
method uses), the trough phase explained as much or more phase-locked variance in almost every
block, and the multi-channel phase slipped cycles (stride-time CV 9–23 % against 7–9 %). Lateral
sway and roll are much less clean oscillators than the vertical bob. The confirmatory analysis
therefore uses troughs.

References for §5:
- Bruijn SM, Meijer OG, Beek PJ, van Dieën JH (2013). Assessing the stability of human locomotion: a review of current measures. *J R Soc Interface* 10:20120999.
- Damouras S, Chang MD, Sejdić E, Chau T (2010). An empirical examination of detrended fluctuation analysis for gait data. *Gait Posture* 31:336–340.
- Dingwell JB, Cusumano JP (2000). Nonlinear time series analysis of normal and pathological human walking. *Chaos* 10:848–863.
- Funato T, Yamamoto Y, Aoi S, et al. (2016). Evaluation of the phase-dependent rhythm control of human walking using phase response curves. *PLoS Comput Biol* 12:e1004950.
- Hausdorff JM, Peng C-K, Ladin Z, Wei JY, Goldberger AL (1995). Is walking a random walk? Evidence for long-range correlations in stride interval of human gait. *J Appl Physiol* 78:349–358.
- Hirasaki E, Moore ST, Raphan T, Cohen B (1999). Effects of walking velocity on vertical head and body movements during locomotion. *Exp Brain Res* 127:117–130.
- Kralemann B, Cimponeriu L, Rosenblum M, Pikovsky A, Mrowka R (2008). Phase dynamics of coupled oscillators reconstructed from data. *Phys Rev E* 77:066205.
- Peng C-K, Havlin S, Stanley HE, Goldberger AL (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos* 5:82–87.
- Pozzo T, Berthoz A, Lefort L (1990). Head stabilization during various locomotor tasks in humans. I. Normal subjects. *Exp Brain Res* 82:97–106.
- Revzen S, Guckenheimer JM (2008). Estimating the phase of synchronized oscillators. *Phys Rev E* 78:051907.
- Rosenstein MT, Collins JJ, De Luca CJ (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D* 65:117–134.
- Terrier P, Dériaz O (2011). Kinematic variability, fractal dynamics and local dynamic stability of treadmill walking. *J NeuroEng Rehabil* 8:12.

## Figures

`plotstyle.py` is the one house style for every figure in the pipeline: palette by role,
typography, and the print constants. `S.save_pub()` writes a **300 dpi PNG and a vector
PDF** at exactly the authored size and audits the figure for undersized type.

| Property | Value |
|---|---|
| Width | `COL2_IN` = 7.2 in / **183 mm** (double column); `COL1_IN` = 89 mm |
| Smallest type | `MIN_PT` = **8 pt**, enforced by `save_pub()` |
| Panel labels | bold capitals, top-left, via `S.panel()` |
| Palette | `plotstyle` roles; colour-blind aware, no red/green pairing carries meaning alone |

**Author at final print size.** Type is set in points, so it only survives if the figure is
placed 1:1 — a 18 in figure dropped into a 183 mm column prints its 8 pt labels at ~3 pt.
`save_pub()` writes with `savefig.bbox="standard"` rather than `"tight"` so the saved width
cannot silently drift.

**Schematics share one look, defined in `diagram_kit`.** Anything diagram-shaped — boxes,
chips, timelines, flow — reads its colours and metrics from there rather than re-choosing
them:

| Constant | What it fixes |
|---|---|
| `FLOW` | the fills: a light tint with its own hue as the edge and dark text on top (`blue`, `green`, `walk`, `orange`, `red`, `rest`, `panel`). Never a saturated fill with reversed-out white type — that loses its labels when printed small. |
| `FLOW_INK` / `FLOW_MUTED` | primary and secondary text |
| `FLOW_LW` | line weights (`box` 1.4, `pill` 1.1, `arrow` 1.2, `icon` 1.15), set for 183 mm |
| `FLOW_PT` | type sizes (`title` 10.5, `body` 8.5, `small` 8.0) |
| `corner_radius(ax)` | round corners on an axes whose two axes carry different quantities — a single radius in data units otherwise comes out as a lopsided ellipse |

Data figures (the `davidson_power` curves) keep `plotstyle`'s semantic palette — a
flowchart fill means nothing on a line plot — but share the same typography and ink.

`figures/` holds the six **methods schematics** — the CAVE apparatus, the machine setup, the
trial timeline, the session/trial flow, the adopted collection design and the design
trade-off — drawn from
matplotlib primitives (`diagram_kit.py` for shapes and the shared `FLOW` palette,
`icons_svg.py` + `icons/` for the flat Tabler line icons) rather than exported artwork, so
they stay vector. Every paradigm number they draw (timings, trial mix, blocks per visit)
comes from one place, `paradigm.py`, which names the Unity field each value mirrors. It also holds
[`figures/figure_captions.md`](figures/figure_captions.md): the full manuscript captions for
**every** figure, plus the parameter tables that were pulled out of the images.

```bash
python Analysis/figures/make_apparatus.py
python Analysis/figures/make_setup_diagram.py
python Analysis/figures/make_trial_timeline.py
python Analysis/figures/make_trial_flow.py
python Analysis/figures/make_adopted_design.py
python Analysis/figures/make_design_tradeoff.py
```

Each figure carries **one message** and contains no table and no baked-in caption — those
are text, and a table rendered as pixels cannot be searched, copied or corrected.

## Davidson (2024) figure replication

| Davidson figure | Our output |
|---|---|
| 1d / 3b (gait template) | `gait/…/figures/*_gait.png`, panels a–b |
| 1c / 3a (head trace) | `gait/…/figures/*_gait.png`, panels c–d |
| 3c–f (phase modulation) | `phase/…/figures/*_phase.png`, panels c, e–f |
| 3g (R² spectrum) | `phase/…/figures/*_phase.png`, panel d |
| 4 (prevalence) | `population/figures/population_fig4_prevalence.png` |
| 5 (per-participant phase) | `population/figures/population_fig5_phase.png` |
| Supp 5/6 (clock-time control) | `clocktime/…/figures/*_clocktime.png` |
| Supp 7/8 (stride duration) | `population/figures/population_suppl{7,8}_*.png` |
| 2 (standing vs walking) | `conditions/figures/conditions_standing_vs_walking.png` |
| 1a/1b/2-task (VE & task UI) | not replicated |

**Method fidelity (verified against the paper):** troughs in vertical head position =
double-support; stride = 2 steps, so the gait fundamental is **2 cps**; **40 stride bins**;
Fourier fit **0.1–10 cps in 0.1 steps**; significance by **1000× trial-level permutation**,
retaining the max R² across frequencies as the null for the exploratory grid scan; Bayesian
prevalence as MAP + HPDI.

## BIDS notes (verified against BIDS 1.11.1)

- **Calibration is `_beh.tsv`, not `_events.tsv`** — spec-mandated, because the QUEST+ log
  has no `onset`/`duration` columns.
- **Motion `_channels.tsv`** uses the required column order `name, component, type,
  tracked_point, units`, and carries the 6-DOF mocap pose channels, the stimulus/response
  onset pulses and the `mocap_frame_advanced` flag.
- **Events** take onset from the explicit `stimulus-onset` pulse when present (else the
  `task-visible` rising edge); response time prefers the precise hi-res RT.
- `task-ILL_stimulus.json` uses a custom `_stimulus` suffix (not the reserved `_stim`) and
  is listed in `.bidsignore`.
- **Validation:** the schema-based `@bids/validator` is authoritative. The legacy npm
  `bids-validator@1.x` flags every motion file as `NOT_INCLUDED` — a known false positive,
  it predates Motion-BIDS.

## Caveats

- **Re-runs are safe.** `sourcedata/*.csv.gz` is the single source of truth; re-running
  reads from it. Condition comes from the CSV's `block-id`, so it stays correct whether the
  input is a loose log or a re-read archive.
- **Sampling.** Unity logs are frame-timed (~60 Hz, variable dt, dropped frames). The true
  per-sample time is preserved in the `t` channel; downstream code resamples on that.
- **Coordinate frames.** `*_global_*` = Unity world space; `*_local_*` = relative to the
  moving CAVE rig. The local vertical (`head_local_pos_y`) is the gait signal.
- **Sentinel frames.** Pre-init / dropout samples carry out-of-range positions (e.g. −505);
  `qc._valid_pos_mask` excludes them.
- **Spike cleaning is conservative and verified.** Thresholds sit 15–75× above the real
  ~0.02 m gait bob, so clean recordings come through byte-for-byte unchanged (asserted by
  unit test) while genuine dropouts — ~1 % of samples — are removed. Exclude a recording on
  the cleaning verdict only if clean coverage drops below 60 %.
- **Half-stride ambiguity.** Lateral sway labels the stance foot at ~93 % alternation,
  fixing an absolute stride origin. If alternation is low for a participant, `phase_at`
  falls back to step-index parity and the ±half-stride ambiguity returns.
- **Pooled-trials results (`pooled_*`) are exploratory, not population inference.** Pooling
  ignores between-participant variance, so a pooled effect could be one participant.
- **Power.** Below the planned sample the modules are method-faithful but their
  significance is not interpretable. `s_hat` in the summary is *computed*, not a constant —
  it sits at α when the test has no power at the available trial counts, which is exactly
  when prevalence is declared unidentifiable. See
  [`davidson_power/`](davidson_power/README.md).
