# Davidson-anchored power analysis

Measures the stride-phase effect size in the raw data of Davidson et al. (2024)
(<https://osf.io/8djtq/>, *Nat Commun*), then simulates **this pipeline's own tests** at
our trial counts.

**Result.** Per-participant RMS amplitude **0.026**, coherent group amplitude **0.010**,
phase coherence **R = 0.33**. The 0.071 that earlier design decisions rested on was
inflated ~2.8×. Consequences:

* Per-participant detection and Bayesian prevalence are out of reach at any feasible trial
  count, and are **exploratory**.
* The confirmatory test is the phase-agnostic [`group_agnostic.py`](../group_agnostic.py):
  **91 %** power at N = 12 × 1440 walking trials, where the common-phase test reaches **15 %**.
* The design is **8 walking + 2 standing blocks** over two visits, N 12–20 analysable,
  rather than 6 + 4 (*Design trade-off* below).

## Code

| Module | Does | Writes |
|---|---|---|
| `effect.py` | measures the effect size from the OSF `.mat` files; `--download` fetches them (~49 MB, git-ignored), `--verify` evidences the extraction choices (§1) | `davidson_effect.json` |
| `power.py` | one simulator of the pipeline's estimator chain and every power analysis on it: `validate`, `checks`, `sweep`, `population`, `robustness`, `mda`, `tradeoff` | `sweep_results.json`, `group_power_results.json`, `design_compare_results.json`, `design_tradeoff.json` |
| `figures.py` | fig1–4, fig6 | `fig*.{png,pdf}` |

### The simulator

`power.py` draws each participant's walking trials and passes them through the pipeline's
own chain, and `validate` checks each step against the pipeline function itself:

| Step | Pipeline function | Agreement |
|---|---|---|
| difficulty residualisation | `phase_analysis.difficulty_adjust` | 2e-16 |
| 40 bins, fit weighted by bin occupancy | `phase_analysis._spectrum_with_coef` | 4e-16 |
| per-participant R² at 2 & 4 cps vs 1000 trial-level permutations, Holm | `phase_analysis.analyse_measure` | 25/25 identical p-values on shared permutations |
| population `S = Σ|zᵢ|²/sᵢ²`, Holm | `group_agnostic.agnostic_test` | power 90.7 % vs 90.7 % on real permutations |
| common-phase Hotelling T², Holm | `common_phase.hotelling_t2` | 2e-16 |
| fixed-slope Weibull ML threshold | `threshold_by_condition.fit_alpha` | 1e-5 relative |

Three shortcuts make it fast, and each is exact:

* The estimators see a participant's trials only through the (stride bin × difficulty
  level) table, and inside a cell the trials are independent Bernoulli draws. So the
  40 × 3 table is drawn directly (multinomial counts, binomial hits), not 1440 trials.
* `sᵢ²`, which the pipeline estimates from its 1000 shuffles, is computed as the
  permutation expectation `n/(n−1)·Var(v)·[(G⁻¹)₁₁ + (G⁻¹)₂₂]`.
* The population null is the statistic's distribution over studies simulated without an
  effect. Trials are exchangeable under H0, so this is what the permutation null estimates.

Effect model: participant *i*'s coefficient at frequency *f* is `μ_f + u_if`,
`u_if ~ CN(0, RMS_f² − μ_f²)`, with RMS and the coherent μ read from
`davidson_effect.json`. Both frequencies are carried in the same dataset. The difficulty
mix uses the rig's own levels: QUEST+ places them at the 75 / 85 / 65 % points of
`P = γ + (1−γ−λ)(1 − e^{−(x/α)^β})`. A threshold oscillation then reaches each level with
gain ∝ (1−q)(−ln(1−q)), i.e. 1.00 / 0.93 / 0.75, weighted 0.98.

## 1. Effect size — `effect.py`

Inputs: `GFX_Data_inGaits.mat` (accuracy in 40 stride bins) and
`GFX_Data_inGaits_FourierFits.mat` (Davidson's per-participant R² spectra), 36 participants.

1. **Data slice.** `GFX_TargPosData`'s second index is the foot split
   `{LRL, RLR, combined}`, not the gait type. Only index 2 reproduces his stored spectra
   (to 7e-14). The index 1 used before is off by 0.18–0.41.
2. **Noise floor.** His pipeline enters every target twice, once per foot
   parameterisation, so ~1670 rows come from ~836 independent trials. The floor is
   bootstrapped through that duplication. `--verify` places the LRL/RLR offset jitter at
   about ±2 percentiles, and the RMS amplitude moves < 0.0003 across ±1–3.
3. **No selection.** Under a null that keeps the duplication, his per-participant flag
   has a **~17 % FPR**. Amplitudes are therefore neither conditioned on the flag nor read
   at each participant's peak frequency; they are read at the pre-specified 2.01 / 4.01 cps.
4. **Coherent amplitude, debiased by the variance of the mean.** `|z̄|² − Σ|zᵢ − z̄|²/(N(N−1))`
   removes measurement noise *and* the between-participant scatter. Subtracting only the
   noise alone would leave `var_between/N` in and gave 0.011 instead of 0.010.

| Quantity | Value |
|---|---|
| independent walking targets / participant | median **836** (309–982) |
| baseline accuracy | 0.741 |
| **per-participant RMS amplitude @2 cps** | **0.026** (95 % CI 0.011–0.035) |
| per-participant RMS amplitude @4 cps | 0.020 |
| **coherent group amplitude @2 cps** | **0.010** (95 % CI 0.000–0.022) |
| phase coherence R | **0.33** @2 cps, 0.21 @4 cps |
| FPR of Davidson's per-participant flag | ~17 % (nominal 5 %) |

His **group-level** result stands: max R² 0.473 at 2.01 cps against his own threshold
0.362. The duplication is common to his observed and shuffled group curves.

## 2. Per-participant power — `power.py sweep`

The confirmatory per-participant test, run in full with 1000 permutations per simulated
participant and 1000 participants per cell, half oscillating at 2 cps and half at 4 cps.
FPR at amplitude 0 is 5.3 %, and the MC SE ≤ 1.6 points.

| walking trials (blocks) | amp **0.026** (typical) | 0.051 (upper quartile) | 0.062 (Davidson's flagged 8) |
|---|---|---|---|
| 360 (2) | 7 % | 16 % | 24 % |
| 540 (3) | 10 % | 26 % | 36 % |
| 836 (Davidson's own) | 12 % | 43 % | 56 % |
| 1080 (6, former) | 15 % | 53 % | 71 % |
| **1440 (8, adopted)** | **20 %** | **67 %** | **84 %** |
| 2160 (12) | 26 % | 87 % | 97 % |

Even 12 blocks reach only ~26 % at a typical amplitude, which is why per-participant
inference is exploratory.

## 3. Population power — `power.py population`

Two tests, both Holm-combined over 2 and 4 cps, 4000 simulated studies per cell against a
10 000-study null:

* **A, common-phase.** One-sample Hotelling T² on the (cos, sin) pairs: a population *mean*
  oscillation.
* **B, phase-agnostic** (confirmatory). `Σᵢ|zᵢ|²/sᵢ²`: do people oscillate at all, each at
  their own phase.

FPR: A 4.9 %, B 5.4 %. Cells are `A / B`:

| participants | 540 trials | 1080 (former) | **1440 (adopted)** | 2160 |
|---|---|---|---|---|
| 8 | 9 % / 36 % | 8 % / 67 % | 11 % / 83 % | 12 % / 96 % |
| **12** | 10 % / 44 % | 13 % / 81 % | **15 % / 91 %** | 17 % / 99 % |
| 16 | 13 % / 50 % | 18 % / 89 % | 22 % / 96 % | 26 % / 100 % |
| **20** | 16 % / 57 % | 23 % / 94 % | **28 % / 99 %** | 31 % / 100 % |
| 24 | 20 % / 64 % | 29 % / 96 % | 34 % / 100 % | 41 % / 100 % |
| 30 | 24 % / 72 % | 36 % / 98 % | 44 % / 100 % | 52 % / 100 % |
| 36 | 30 % / 77 % | 45 % / 99 % | 53 % / 100 % | 62 % / 100 % |

R = 0.33 is why test A is so much weaker: a common-phase contrast is the wrong shape for a
phase-idiosyncratic effect.

`power.py robustness` re-runs B on a spike-and-slab population (a share γ oscillate, the
rest are flat) matched on E|c|². At 12 × 1440 it gives 92 % (Gaussian), 90 % (γ = 0.5),
83 % (0.25) and 73 % (0.12). The advantage over A holds even for a sparse population.

## 4. Smallest detectable effect — `power.py mda`

The design question asked the other way round: how small an oscillation could each
design still catch with 80 % power? `mda` sweeps B over multiples of the measured
amplitude and reads off the crossing.

| N | one visit (810 trials) | two visits (1440 trials) |
|---|---|---|
| 12 | 0.0299 | 0.0225 |
| 20 | 0.0256 | 0.0196 |
| 30 | 0.0229 | 0.0171 |

Against the measured 0.026, two visits are at 80 % power from N ≈ 7 and one visit from
N ≈ 19.

## 5. Checks — `power.py checks`, `effect.py --verify`

* The closed-form null scale matches the simulated `E|z|²` (and `4·Var/N`) within MC error.
* Amplitude recovery is unbiased (0.026 → 0.0256 after the 0.4 % bin-averaging loss).
* The per-participant test is calibrated: FPR 4.4–5.0 % at 360–2160 trials.
* ISI 4 s ± 0.75 s against a ~1.05 s stride leaves target phases at chance non-uniformity
  at 2 and 4 cps. Confirm on real data with `onset_phase_coverage`.
* The difficulty mix costs little: per-participant power at amplitude 0.05 and 1440 trials
  is 67 % with every trial at threshold, 64 % with the rig mix, and 54 % if only threshold
  trials carry the effect.

## 6. Reproduce

```
python -m Analysis.davidson_power.effect --download   # the OSF .mat files (~49 MB)
python -m Analysis.davidson_power.effect              # -> davidson_effect.json
python -m Analysis.davidson_power.effect --verify     # §1
python -m Analysis.davidson_power.power validate      # §Code, ~4 min
python -m Analysis.davidson_power.power checks        # §5, ~3 min
python -m Analysis.davidson_power.power all           # §2-4 + trade-off, ~45 min on one core
python -m Analysis.davidson_power.figures             # fig1-4, fig6
```

The analyses are independent, so `sweep`, `population`, `mda`, `tradeoff` and
`robustness` can run as parallel processes (~20 min, bounded by `sweep`).

## 7. Figures

House style and captions: [`../README.md`](../README.md#figures) and
[`../figures/figure_captions.md`](../figures/figure_captions.md). Figure 5, the adopted
design, is a methods schematic in [`../figures`](../figures/); filenames track manuscript
numbers.

| Figure | Message |
|---|---|
| `fig1_effect_size` | The effect size is 0.026, not 0.071: three inflations removed in turn. |
| `fig2_per_participant_power` | Per-participant detection is out of reach at any feasible trial count. |
| `fig3_population_power` | Phases are idiosyncratic (R = 0.33), so the choice of population test decides the study: 91 % vs 15 % at N = 12. |
| `fig4_trials_vs_participants` | Power over participants × walking blocks, with the 80 % boundary and the adopted design. |
| `fig6_one_vs_two_visits` | Why two visits: 80 % power from N ≈ 7 instead of N ≈ 19. |

### Design comparison — the table behind Figure 6

| Parameter | Two visits (adopted) | One visit |
|---|---|---|
| Walking blocks / trials | **8 (4 per visit) / 1440** | 4–5 / ~810 |
| N for 80 % population power | **≈ 7** | **≈ 19** |
| Population power, phase-agnostic | 0.92 (N = 12) → 0.99 (N = 20) | 0.64 (N = 12) → 0.81 (N = 20) |
| Smallest detectable amplitude at N = 20 | 0.020 | 0.026 |
| Order crossover between visits | yes | not possible |
| H2 free of time-on-task? | yes (within-participant) | no (between only) |

The planned N of 12–20 is set by attrition, H2 and the exploratory analyses, not by the
confirmatory test, which the two-visit design already powers at N ≈ 7.

---

## Design trade-off — the 8/2 walking/standing split

`power.py tradeoff` writes `design_tradeoff.json`, the measurements behind the 8/2 split, and
`../figures/make_design_tradeoff.py` draws them.

**Gain:** confirmatory power against walking trials, 4000 simulations per cell.

| walking trials | blocks | N = 12 | N = 15 | N = 20 |
|---|---|---|---|---|
| 720 | 4 | 0.581 | 0.646 | 0.751 |
| 900 | 5 | 0.711 | 0.778 | 0.879 |
| 1080 | 6 *(former)* | 0.814 | 0.875 | 0.938 |
| 1260 | 7 | 0.860 | 0.919 | 0.966 |
| **1440** | **8 *(adopted)*** | **0.925** | **0.957** | **0.986** |

**Cost:** SD of the standing 75 % threshold, estimated by `threshold_by_condition`'s
fixed-slope Weibull ML with the rig's three levels (true threshold 6.0°, 4000 simulations).

| standing trials | blocks | SD | % of threshold |
|---|---|---|---|
| 180 | 1 | 0.509° | 8.5 % |
| **360** | **2 *(adopted)*** | **0.340°** | **5.7 %** |
| 540 | 3 | 0.279° | 4.7 % |
| 720 | 4 *(former)* | 0.243° | 4.0 % |

**Why the trade is lopsided.** H2 is the crossover model's within-participant contrast,
so between-participant threshold spread cancels and only estimation error remains. At
the adopted 1 standing + 4 walking blocks per visit, H2 has power **0.994** for a 1.10×
threshold ratio and 1.000 for 1.15× (N = 12, 2000 simulations). Standing was buying
precision that the comparison it exists for cannot use.

The JSON is the single source that the figure and the READMEs read.
