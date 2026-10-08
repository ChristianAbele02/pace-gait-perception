using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// =============================================================================
//  StaircaseManager  —  QUEST+ calibration for the Gabor tilt-discrimination task
// =============================================================================
//  Replaces the former 1-up/2-down adaptive staircase with a Bayesian adaptive
//  method (QUEST+, Watson 2017, J. Vision 17(3):10). The task is a 2-AFC
//  orientation discrimination: a Gabor is tilted left or right by a magnitude
//  (degrees) and the observer reports the tilt direction. QUEST+ maintains a
//  posterior over the threshold parameter and places each trial at the tilt
//  that minimises the expected entropy of that posterior.
//
//  Configuration (single free parameter: threshold) follows the canonical
//  reduction of QUEST+ to threshold-only estimation — slope (beta) and lapse
//  (lambda) are FIXED to literature values, which is the robust choice for the
//  ~40-trial budget typical of a calibration block (estimating slope/lapse
//  reliably needs hundreds of trials; Paire et al. 2022, Behav Res Methods).
//
//  Psychometric model (Weibull, Watson's QUEST+ parameterisation, on log10 deg):
//      P(correct | x) = gamma + (1 - gamma - lambda) * (1 - exp(-10^(beta*(x - alpha))))
//    x      = log10(tilt magnitude in deg)         (stimulus)
//    alpha  = log10(threshold tilt in deg)         (estimated parameter)
//    beta   = slope                                (fixed)
//    gamma  = guess rate = 0.5                      (2-AFC)
//    lambda = lapse rate                            (fixed)
//
//  Cited default parameter values (orientation discrimination, Gabor):
//    gamma  = 0.5     2-AFC guess rate.
//    lambda = 0.05    Lapse (fixed). RAISED from the nominal 0.02: too low a
//                     lapse rate makes adaptive placement RUN AWAY on the
//                     occasional error at a clearly-suprathreshold tilt — the
//                     model treats such an error as near-impossible and inflates
//                     the threshold estimate (observed: a tracker chasing a 9 deg
//                     lapse up to ~8 deg). 0.05 absorbs sparse lapses robustly;
//                     orientation-discrimination lapses are 0.01-0.03 (Wichmann &
//                     Hill 2001), so 0.05 is mildly conservative on purpose.
//    beta   = 2.5     Discrimination Weibull slopes fall in ~1.4-3 (Wallis et
//                     al. 2013, Vis Res; May & Solomon 2013 "Four Theorems"); ~2.5 is
//                     consistent with the orientation threshold/slope coupling
//                     (threshold ~1.5-2 deg => cum-Gaussian sigma ~2.5 deg).
//                     NOTE: underestimating the slope (assuming it shallower) is
//                     the safer error for threshold bias (PLOS ONE,
//                     "Efficient estimation of stereo thresholds").
//
//  IMPORTANT — the source paradigm (Davidson, Verstraten & Alais 2024, Nature
//  Communications 15:2027) adapts CONTRAST in a detection task and used QUEST
//  with beta = 3.5, lambda = 0.01. We instead keep the tilt-DISCRIMINATION
//  variable, so the orientation-discrimination values above apply. Walking is
//  expected to shift the THRESHOLD but not the slope (Davidson et al. 2024:
//  threshold change p < 0.001, slope change n.s. p = 0.063), so the same slope
//  prior is appropriate for standing and walking; the threshold is what gets
//  re-estimated per condition.
//
//  Output: a single POOLED posterior over all trials (both trackers' data, weak
//  prior) gives the final threshold (alpha), written to ExperimentConductor as
//  thresholdAngle with easyAngle / hardAngle derived from the fitted PF.
//
//  CONVERGENCE is PRECISION-based: both trackers run interleaved until the pooled
//  posterior SD <= convergenceSDLog (after minScoredTrials), with a hard cap
//  (maxTotalTrials) and an early-abort for hopeless runs (abortCheckTrials).
//
//  RELIABILITY (the operator's green DONE / red UNRELIABLE banner) gates on three
//  things: (1) converged (precise); (2) attentive — easy-anchor CATCH trials
//  interleaved at ~catchTrialInterval test for lapses/inattention/side-bias and are
//  NOT fed to QUEST+; (3) a real, non-railed threshold (not run away to the grid
//  ceiling). A/B tracker agreement is logged and warned, but does NOT gate the
//  banner (at ~20 trials/tracker two posteriors differ by > agreementLogTol ~30% of
//  the time even for good observers). See Analysis/calibration_convergence_sim.py
//  for the simulation that set these rules. The main experiment then presents
//  Gabors balanced 1/3 threshold, 1/3 easy, 1/3 hard.
//
//  WARM-UP (practice) lead-in: the run opens with warmupTrials clearly-suprathreshold
//  practice trials on a gentle descending ramp (warmupStartTiltDeg -> warmupEndTiltDeg,
//  both easier than the staircase start tilts) so the observer re-learns the 2-AFC
//  mapping BEFORE QUEST+ starts placing trials — the first few QUEST+ trials dominate
//  the posterior, so absorbing the
//  task-relearning phase up front keeps early mistakes from biasing the threshold.
//  This mirrors the experiment's training checkpoints (CheckpointManager.training-
//  Checkpoints, also 5) so the threshold is measured in the same warmed-up state it
//  is tested in. Warm-up trials are NOT fed to QUEST+/pooled and do NOT count toward
//  the trial budget or the catch cadence (tagged "warmup" in the CSV, ignored by the
//  analysis, exactly like catch trials).
// =============================================================================

public class StaircaseManager : MonoBehaviour
{
    [Header("QUEST+ Psychometric Parameters (cited defaults)")]
    [Tooltip("Guess rate. 2-AFC tilt discrimination => 0.5.")]
    public float guessRate = 0.5f;
    [Tooltip("Lapse rate (fixed). Raised to 0.05 for robustness: too low a lapse makes QUEST+ run away on suprathreshold errors. Orientation-discrimination lapses ~0.01-0.03.")]
    public float lapseRate = 0.05f;
    [Tooltip("Weibull slope on log10 axis (fixed). Discrimination ~1.4-3; default 2.5. Err shallow.")]
    public float slopeBeta = 2.5f;

    [Header("Stimulus / Threshold Grid (degrees)")]
    [Tooltip("Smallest tilt the grid can present / smallest threshold it can estimate.")]
    public float minAngle = 0.3f;
    [Tooltip("Largest tilt the grid can present / largest threshold it can estimate.")]
    public float maxAngle = 25.0f;
    [Tooltip("Number of candidate tilt magnitudes QUEST+ may choose from.")]
    public int numStimLevels = 41;
    [Tooltip("Number of candidate threshold values in the posterior grid.")]
    public int numThresholdLevels = 61;

    [Header("Dual Interleaved Trackers (prior centres, degrees)")]
    [Tooltip("Prior centre of tracker A (started above expected ~2 deg threshold).")]
    public float startAngleA = 4.0f;
    [Tooltip("Prior centre of tracker B (started below, classic 'from below').")]
    public float startAngleB = 2.5f;
    [Tooltip("Prior width in log10 units (~0.5 ≈ factor 3.2). Broad = weak prior = less anchoring to the start angle, which reduces A/B divergence.")]
    public float priorLogSD = 0.5f;
    [Tooltip("[LEGACY — no longer gates] Was the per-tracker base budget. The stop is now "
        + "precision-based (see Convergence Gate); both trackers run interleaved until the "
        + "pooled posterior is precise enough. Kept only as a display reference.")]
    public int trialsPerTracker = 25;
    [Tooltip("[LEGACY — no longer gates] Was the per-tracker base-phase early stop; "
        + "superseded by the pooled-precision stop (convergenceSDLog).")]
    public float earlyStopLogSD = 0.08f;

    [Header("Convergence Gate (precision-based)")]
    [Tooltip("Calibration FINISHES once the POOLED posterior SD (log10) is at or below "
        + "this — i.e. the estimate is precise enough. The pooled posterior uses BOTH "
        + "trackers' data, so it tightens faster than either tracker and lets us stop as "
        + "soon as the threshold is actually well-determined. Keep this <= the analysis "
        + "QC threshold (Analysis MAX_FINAL_SD_LOG = 0.12) so a converged run is never "
        + "then flagged 'wide posterior'. 0.10 log10 ≈ ±26% (1 SD).")]
    public float convergenceSDLog = 0.10f;
    [Tooltip("Minimum scored (answered) trials before the precision stop may trigger — "
        + "guards against a premature stop on a transiently-tight early posterior.")]
    public int minScoredTrials = 30;
    [Tooltip("Hard cap on total scored trials across both trackers. On cap the pooled "
        + "estimate is used and the run is flagged not-converged (recalibrate).")]
    public int maxTotalTrials = 100;
    [Tooltip("Early-abort: if by this many trials the pooled SD is still above abortSDLog "
        + "the run clearly is not converging (e.g. an inconsistent / side-biased "
        + "responder); stop early rather than grinding to the cap. <=0 disables.")]
    public int abortCheckTrials = 80;
    [Tooltip("Pooled SD (log10) above which, once abortCheckTrials is reached, the run is "
        + "aborted as non-converging.")]
    public float abortSDLog = 0.18f;
    [Tooltip("Sanity rail: a final pooled threshold at/above this (deg) is treated as 'no "
        + "real threshold found' — QUEST+ ran to the grid ceiling (e.g. a random / "
        + "non-responding participant) — and the run is flagged UNRELIABLE however narrow "
        + "the posterior. Default = half the grid max; keep well above any plausible "
        + "threshold (~6-8 deg).")]
    public float maxReliableThresholdDeg = 12.5f;
    [Tooltip("QC ROBUSTNESS FLAG ONLY (no longer the stop gate). The two trackers start "
        + "from different priors and should END within this |alpha| gap in log10 "
        + "(~0.15 = factor 1.4); a larger gap is logged as a reliability flag (it means "
        + "the responder was inconsistent), but does not by itself extend the run.")]
    public float agreementLogTol = 0.15f;

    [Header("Difficulty Levels (target proportion correct on the fitted PF)")]
    // The three main-experiment difficulty levels are defined as target accuracies
    // on the fitted psychometric function (not tilt multiples), then inverted to
    // tilt angles. This lands each level at a KNOWN accuracy for every participant
    // regardless of their slope, keeps all three off ceiling/floor, and puts the
    // threshold level at the steepest (most perturbation-sensitive) part of the PF.
    [Tooltip("Hard level target proportion correct (steep band, off floor). Default 0.65.")]
    public float targetHardPC = 0.65f;
    [Tooltip("Threshold level target proportion correct (~steepest, most sensitive point). Default 0.75.")]
    public float targetThresholdPC = 0.75f;
    [Tooltip("Easy level target proportion correct (off ceiling). Default 0.85.")]
    public float targetEasyPC = 0.85f;

    [Header("Catch / Lapse Trials (easy anchors)")]
    [Tooltip("Intersperse occasional VERY EASY suprathreshold trials. They are visually "
        + "identical to normal trials but at a large, unmistakable tilt, so a wrong answer "
        + "is almost certainly a lapse/inattention (not a perceptual error). Catch trials "
        + "are NOT fed to QUEST+ (no posterior update, do not count toward the trial "
        + "budget) — they only measure attentiveness, an independent reliability signal.")]
    public bool enableCatchTrials = true;
    [Tooltip("Average number of SCORED trials between catch trials (jittered by "
        + "catchIntervalJitter so they are unpredictable). ~6 => >= minCatchToJudge catches "
        + "even in a minimum-length (30-trial) calibration, so fast-but-bogus convergence "
        + "(e.g. a random responder) cannot escape the attentiveness check.")]
    public int catchTrialInterval = 6;
    [Tooltip("± jitter (scored trials) on the catch interval, so catches are not on a "
        + "predictable cadence the participant could learn.")]
    public int catchIntervalJitter = 2;
    [Tooltip("Tilt magnitude (deg) of a catch trial — large enough to be near-ceiling "
        + "(~100% correct) for any plausible threshold. Clamped to [minAngle, maxAngle].")]
    public float catchTiltDeg = 12.0f;
    [Tooltip("If the catch error rate exceeds this (over >= minCatchToJudge answered "
        + "catches), the run is flagged UNRELIABLE even if the threshold posterior "
        + "converged — a precise threshold from an inattentive observer is untrustworthy.")]
    public float maxCatchLapseRate = 0.10f;
    [Tooltip("Minimum answered catch trials needed before the catch lapse rate is allowed "
        + "to flag a run unreliable (avoids judging on 1-2 catches).")]
    public int minCatchToJudge = 4;

    [Header("Warm-up (practice) Trials")]
    [Tooltip("Lead the calibration with this many fixed, EASY practice trials before "
        + "QUEST+ starts measuring — mirrors the experiment's training checkpoints (also 5). "
        + "They re-establish the 2-AFC mapping so the staircase is not contaminated by "
        + "early task-relearning errors (the first QUEST+ trials dominate the posterior). "
        + "Like the experiment's training trials they are NOT fed to QUEST+/pooled, do NOT "
        + "count toward the trial budget, and do NOT advance the catch cadence. Set 0 to "
        + "disable; default 5 to match CheckpointManager.trainingCheckpoints.")]
    public int warmupTrials = 5;
    [Tooltip("Warm-up ramp START tilt (deg) — the FIRST, easiest practice trial. Large and "
        + "clearly suprathreshold (near-100% correct). The warm-up tilt then ramps DOWN "
        + "toward warmupEndTiltDeg across the lead-in. Clamped to [minAngle, maxAngle].")]
    public float warmupStartTiltDeg = 12.0f;
    [Tooltip("Warm-up ramp END tilt (deg) — the LAST practice trial, just before QUEST+ "
        + "takes over. Kept EASIER (larger) than the staircase start tilts (startAngleA/B "
        + "~2.5-4 deg) so even the final warm-up trial stays comfortably suprathreshold. "
        + "Clamped to [minAngle, maxAngle].")]
    public float warmupEndTiltDeg = 6.0f;

    [Header("Timing")]
    public float taskDuration = 0.1f;
    [Tooltip("ONSET-to-ONSET ISI target (s) — the next Gabor appears this long after the "
        + "PREVIOUS onset, NOT after the response. The stimulus + response-window time is "
        + "subtracted out so the onset cadence is fixed regardless of how fast the "
        + "participant answers. Set to match the EXPERIMENT's spatially-generated ISI (4 s).")]
    public float interTrialDelay = 4.0f;
    [Tooltip("Uniform ISI jitter HALF-RANGE (s): ISI = interTrialDelay +/- this, drawn "
        + "uniformly => [3.25, 4.75] s. Match the experiment's isiJitter.")]
    public float isiJitter = 0.75f;

    // Onset-to-onset ISI state. The jitter is now a single uniform draw (matching the
    // experiment's uniform ISI), so no per-trial jitter state is needed beyond the last
    // onset time used to hold the onset cadence fixed regardless of response speed.
    private float lastOnsetTime;
    [Tooltip("Max time (s) to wait for a response after stimulus onset. No response within this window is logged as a timeout (response = -1) and is NOT fed to QUEST+ (the posterior and trial count are left unchanged), so the trial is simply re-placed and calibration still collects the full number of real responses.")]
    public float responseWindow = 2.0f;
    [Tooltip("Seconds to hold the green DONE / red UNRELIABLE result on the HEAD-NODE "
        + "operator overlay after calibration finishes, before returning to setup. This "
        + "is shown only on the operator screen, never to the participant.")]
    public float resultDisplaySeconds = 5.0f;

    public FollowerPlane visualTaskPlane;

    [Header("Per-condition Calibration")]
    [Tooltip("Locomotor condition this calibration measures. Normally set at runtime from "
        + "ExperimentConductor.calibrationCondition; the serialized value is the fallback. "
        + "'standing' = treadmill off; 'walking' = walk-in-place on the treadmill at block speed.")]
    public string calibrationCondition = "standing";
    [Tooltip("Treadmill controller — REQUIRED for the WALKING calibration (walk-in-place). "
        + "Leave null for standing; auto-found at Start if one is present in the scene.")]
    public TreadmillController treadmillController;

    // --- State ---
    private QuestPlusTracker trackerA;
    private QuestPlusTracker trackerB;
    private QuestPlusTracker pooled;   // posterior over ALL trials (weak prior) -> final estimate
    private QuestPlusTracker currentTracker;
    private readonly Queue<QuestPlusTracker> trialQueue = new Queue<QuestPlusTracker>();

    private bool isCalibrationStarted = false;
    private bool waitingForResponse = false;

    // Calibration outcome, exposed for the HEAD-NODE operator overlay (FPSCounter) to
    // show a green DONE / red UNRELIABLE banner. Operator-only — not shown to the
    // participant (FPSCounter runs only on the head node).
    private bool calibrationFinished = false;
    private bool calibrationReliable = false;
    public bool CalibrationFinished { get { return calibrationFinished; } }
    public bool CalibrationReliable { get { return calibrationReliable; } }
    private int currentTiltDirection; // 0 = left, 1 = right
    private float currentAngle;       // tilt magnitude presented this trial
    private NetworkingSync networkingSync;
    private Coroutine responseWindowCoroutine;

    // --- Overlay / run statistics ---
    private int totalTrials = 0;      // scored (answered) trials
    private int totalCorrect = 0;
    private int totalTimeouts = 0;    // missed (no-response) trials

    // Catch-trial (lapse) state. catchAnswered = answered catches; catchErrors = wrong
    // answers on those (= lapses); catchTimeouts = catches with no response. catchCountdown
    // counts down scored trials until the next catch fires.
    private bool currentIsCatch = false;
    private int catchAnswered = 0;
    private int catchErrors = 0;
    private int catchTimeouts = 0;
    private int catchCountdown = 0;

    // Warm-up (practice) state. warmupRemaining counts down the leading easy practice
    // presentations; like catch trials they are scored separately and never fed to QUEST+.
    private bool currentIsWarmup = false;
    private int warmupRemaining = 0;
    private int warmupAnswered = 0;
    private int warmupCorrect = 0;
    private int warmupTimeouts = 0;

    private int loggedPresentations = 0; // monotonic index for the CSV global_trial column
    private bool hasLastTrial = false;
    private bool lastTrialCorrect = false;
    private bool lastTrialTimedOut = false;
    private float lastTrialAngle = 0f;
    private int lastTrialDir = 0;
    private string lastTrackerName = "-";

    // Per-trial calibration log rows (written to CSV at the end for the
    // convergence figure). Header/metadata are prepended in WriteCalibrationCsv.
    private readonly List<string> calibrationRows = new List<string>();

    void Start()
    {
        networkingSync = FindObjectOfType<NetworkingSync>();

        // Per-condition calibration: which locomotor state are we calibrating? The guided
        // runner sets ExperimentConductor.calibrationCondition before launching scene 2.
        if (ExperimentConductor.instance != null
            && !string.IsNullOrEmpty(ExperimentConductor.instance.calibrationCondition))
            calibrationCondition = ExperimentConductor.instance.calibrationCondition;
        if (treadmillController == null) treadmillController = FindObjectOfType<TreadmillController>();

        trackerA = new QuestPlusTracker("A", minAngle, maxAngle, numStimLevels, numThresholdLevels,
                                        slopeBeta, lapseRate, guessRate, startAngleA, priorLogSD);
        trackerB = new QuestPlusTracker("B", minAngle, maxAngle, numStimLevels, numThresholdLevels,
                                        slopeBeta, lapseRate, guessRate, startAngleB, priorLogSD);

        // Pooled tracker: weak (uniform) prior, updated with EVERY scored trial from
        // both trackers -> the single all-data posterior used for the final threshold.
        // priorLogSD = 0 makes the constructor fall back to a uniform prior.
        pooled = new QuestPlusTracker("POOL", minAngle, maxAngle, numStimLevels, numThresholdLevels,
                                      slopeBeta, lapseRate, guessRate, startAngleA, 0f);

        warmupRemaining = Mathf.Max(0, warmupTrials); // easy practice lead-in before QUEST+ starts
        catchCountdown = RandomCatchInterval();   // first catch ~catchTrialInterval scored trials in

        if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            Debug.Log("QUEST+ Calibration ready. Press Space/PageDown to start.");
    }

    // Scored-trial countdown to the next catch trial, jittered so it is unpredictable.
    private int RandomCatchInterval()
    {
        int lo = Mathf.Max(2, catchTrialInterval - catchIntervalJitter);
        int hi = catchTrialInterval + catchIntervalJitter;
        return UnityEngine.Random.Range(lo, hi + 1);
    }

    // Warm-up tilt for the idx-th (0-based) practice trial: a gentle descending ramp from
    // warmupStartTiltDeg down to warmupEndTiltDeg across the warm-up lead-in, interpolated
    // in log10 space (the same axis the staircase grid lives on). Eases the observer in from
    // a very easy tilt toward — but staying above — the staircase start before QUEST+ takes
    // over. With a single warm-up trial it just uses the start tilt.
    private float WarmupTiltForIndex(int idx)
    {
        if (warmupTrials <= 1) return warmupStartTiltDeg;
        float f = Mathf.Clamp01((float)idx / (warmupTrials - 1));
        float lo = Mathf.Log10(Mathf.Max(1e-3f, warmupStartTiltDeg));
        float hi = Mathf.Log10(Mathf.Max(1e-3f, warmupEndTiltDeg));
        return Mathf.Pow(10f, Mathf.Lerp(lo, hi, f));
    }

    void Update()
    {
        if (System.Environment.MachineName != MasterTrackingData.HeadNodeMachineName) return;

        if (networkingSync == null)
        {
            networkingSync = FindObjectOfType<NetworkingSync>();
            if (networkingSync == null) return;
        }

        if (!isCalibrationStarted)
        {
            if (InputHandler.instance.GetStartAction())
            {
                isCalibrationStarted = true;
                Debug.Log("QUEST+ Calibration started!");
                StartCoroutine(RunLoop());
            }
            return;
        }

        if (waitingForResponse)
        {
            int response = -1;
            if (InputHandler.instance.GetResponseLeft()) response = 0;
            if (InputHandler.instance.GetResponseRight()) response = 1;
            if (response != -1) ProcessResponse(response);
        }
    }

    // --- Rich status overlay for the calibration scene (head node only) ---
    public string GetStatusReport()
    {
        if (trackerA == null || trackerB == null) return "CALIBRATION: initializing...";
        if (!isCalibrationStarted) return "CALIBRATION (QUEST+): waiting for Start (Space/PageDown)...";

        float thrA = trackerA.ThresholdDeg();
        float thrB = trackerB.ThresholdDeg();
        float aLog = pooled.ThresholdLog();                  // final estimate = pooled posterior
        float combined = Mathf.Clamp(AngleForPercentCorrect(aLog, targetThresholdPC), minAngle, maxAngle);
        float easy = Mathf.Clamp(AngleForPercentCorrect(aLog, targetEasyPC), minAngle, maxAngle);
        float hard = Mathf.Clamp(AngleForPercentCorrect(aLog, targetHardPC), minAngle, maxAngle);

        float acc = (totalTrials > 0) ? (100f * totalCorrect / totalTrials) : 0f;
        float abRatio = Mathf.Max(thrA, thrB) / Mathf.Max(1e-3f, Mathf.Min(thrA, thrB));
        float pooledSD = pooled.PosteriorSDLog();
        bool agree = TrackersAgree();
        bool finished = CalibrationDone();

        var sb = new System.Text.StringBuilder();
        sb.AppendFormat("=== CALIBRATION (QUEST+ tilt 2-AFC) [{0}] ===\n",
            calibrationCondition.ToUpper());
        sb.AppendFormat("Model: gamma {0:F2} | beta {1:F1} | lambda {2:F2}\n", guessRate, slopeBeta, lapseRate);
        sb.AppendFormat("Tracker A: thr {0:F2}deg +/-{1:F2}  ({2} trials)\n",
            thrA, trackerA.ThresholdSDDeg(), trackerA.Trials);
        sb.AppendFormat("Tracker B: thr {0:F2}deg +/-{1:F2}  ({2} trials)\n",
            thrB, trackerB.ThresholdSDDeg(), trackerB.Trials);
        sb.AppendFormat("Pooled thr: {0:F2}deg (easy {1:F2} / hard {2:F2}) | A/B {3:F2}x {4}\n",
            combined, easy, hard, abRatio, agree ? "AGREE" : "DISAGREE");
        sb.AppendFormat("Convergence: pooled SD {0:F3} / target {1:F2} (log10) | {2} trials "
            + "(min {3}, cap {4}) | Acc {5:F0}% | Timeouts {6}\n",
            pooledSD, convergenceSDLog, totalTrials, minScoredTrials, maxTotalTrials, acc, totalTimeouts);
        if (enableCatchTrials)
        {
            float catchRate = (catchAnswered > 0) ? (100f * catchErrors / catchAnswered) : 0f;
            bool catchBad = catchAnswered >= minCatchToJudge && catchErrors >= 2
                            && catchRate > 100f * maxCatchLapseRate;
            sb.AppendFormat("Catch (easy): {0}/{1} wrong = {2:F0}% lapses{3} | {4} missed\n",
                catchErrors, catchAnswered, catchRate,
                catchBad ? " [!! INATTENTIVE]" : "", catchTimeouts);
        }
        if (warmupTrials > 0)
            sb.AppendFormat("Warm-up (practice): {0}/{1} done{2} | {3}/{4} correct{5}\n",
                Mathf.Max(0, warmupTrials - warmupRemaining), warmupTrials,
                warmupRemaining > 0 ? " [PRACTICE]" : "", warmupCorrect, warmupAnswered,
                warmupTimeouts > 0 ? string.Format(" | {0} missed", warmupTimeouts) : "");

        if (hasLastTrial)
        {
            if (lastTrialTimedOut)
                sb.AppendFormat("Last [{0}]: {1:F2}deg {2} -> NO RESPONSE\n",
                    lastTrackerName, lastTrialAngle, (lastTrialDir == 0 ? "LEFT" : "RIGHT"));
            else
                sb.AppendFormat("Last [{0}]: {1:F2}deg {2} -> {3}\n",
                    lastTrackerName, lastTrialAngle, (lastTrialDir == 0 ? "LEFT" : "RIGHT"),
                    (lastTrialCorrect ? "CORRECT" : "WRONG"));
        }
        else
            sb.Append("Last: -\n");

        if (waitingForResponse)
            sb.AppendFormat("NOW: {0:F2}deg {1}{2}  (awaiting response)\n",
                currentAngle, (currentTiltDirection == 0 ? "LEFT" : "RIGHT"),
                currentIsWarmup ? " [WARM-UP]" : (currentIsCatch ? " [CATCH]" : ""));
        else if (finished)
            sb.Append(Converged() ? "DONE (converged: pooled estimate precise) - returning to setup.\n"
                                  : "DONE (stopped without convergence - recalibration recommended).\n");
        else
            sb.Append("(inter-trial interval)\n");

        return sb.ToString();
    }

    private IEnumerator RunLoop()
    {
        // Walking calibration: start the belt so the participant is already walking
        // (walk-in-place, static scene) before the first tilt is shown.
        StartWalkingTreadmillIfNeeded();
        yield return new WaitForSeconds(1.0f);
        TriggerNextTrial();
    }

    // Walk-in-place: drive the treadmill at the block speed for the WALKING calibration,
    // so the threshold is measured in the same locomotor state as the walking block.
    // Head-node only (the treadmill server is driven from the head node).
    private void StartWalkingTreadmillIfNeeded()
    {
        if (calibrationCondition != "walking") return;
        if (System.Environment.MachineName != MasterTrackingData.HeadNodeMachineName) return;
        if (treadmillController == null)
        {
            Debug.LogWarning("StaircaseManager: WALKING calibration but no TreadmillController in the "
                + "scene — the participant cannot walk. Add a TreadmillController to scene 2.");
            return;
        }
        float speed = (ExperimentConductor.instance != null) ? ExperimentConductor.instance.experimentSpeed : 1.0f;
        // MANUAL control only — do NOT ResumeTreadmill() here. The calibration scene is static
        // (no splineWalker translating the rig), so automatic movement-derived speed would read
        // ~0 and the belt would stop. Walk-in-place needs the belt held at the fixed block speed,
        // so manual control stays ON for the whole walking calibration (ended by StopTreadmill).
        treadmillController.ForceTreadmillSpeed(speed);
        Debug.Log(string.Format("WALKING calibration: treadmill walk-in-place (manual) at {0:F2} m/s.", speed));
    }

    private void StopTreadmillIfNeeded()
    {
        if (System.Environment.MachineName != MasterTrackingData.HeadNodeMachineName) return;
        if (treadmillController != null) treadmillController.StopTreadmill();
    }

    // The two trackers started from different priors; once they agree on the
    // threshold (within agreementLogTol in log10) the estimate is robust. This is now
    // a QC ROBUSTNESS FLAG, not the stop gate.
    private bool TrackersAgree()
    {
        return Mathf.Abs(trackerA.ThresholdLog() - trackerB.ThresholdLog()) <= agreementLogTol;
    }

    // CONVERGED = the pooled posterior (both trackers' data, the actual final estimate)
    // is precise enough. This is the new definition of "converged": it targets the
    // precision of the number we actually use, and is aligned with the analysis QC
    // threshold so a converged run is not later flagged "wide posterior".
    private bool Converged()
    {
        return totalTrials >= minScoredTrials && pooled.PosteriorSDLog() <= convergenceSDLog;
    }

    // Stop when (a) the pooled estimate is precise enough (converged), (b) the hard cap
    // is reached, or (c) it is clearly not converging by abortCheckTrials (early-abort).
    private bool CalibrationDone()
    {
        if (totalTrials >= maxTotalTrials) return true;                 // hard cap
        if (Converged()) return true;                                  // precise enough
        if (abortCheckTrials > 0 && totalTrials >= abortCheckTrials     // hopeless -> abort
            && pooled.PosteriorSDLog() > abortSDLog) return true;
        return false;
    }

    // Both trackers run interleaved throughout (balanced data for the pooled posterior
    // and the A/B agreement check) until CalibrationDone().
    private bool IsActive(QuestPlusTracker t)
    {
        return true;
    }

    private void TriggerNextTrial()
    {
        if (CalibrationDone())
        {
            FinishCalibration();
            return;
        }

        // Warm-up (practice) trials lead the calibration: the first warmupTrials
        // presentations are fixed, clearly-suprathreshold tilts that let the participant
        // re-learn the 2-AFC mapping BEFORE QUEST+ starts measuring. Like the experiment's
        // training checkpoints they are NOT fed to QUEST+/pooled and do NOT count toward
        // the trial budget or the catch cadence.
        currentIsWarmup = warmupRemaining > 0;

        // Is this presentation a catch (easy-anchor) trial? Fires when the scored-trial
        // countdown has elapsed (never during the warm-up lead-in). A catch is NOT a tracker
        // trial: it skips QUEST+, uses a big fixed tilt, and is scored separately for
        // attentiveness only.
        currentIsCatch = !currentIsWarmup && enableCatchTrials && catchCountdown <= 0;

        if (currentIsWarmup)
        {
            int warmupIndex = warmupTrials - warmupRemaining; // 0-based, before decrement
            warmupRemaining--;
            currentTracker = null;
            currentAngle = Mathf.Clamp(WarmupTiltForIndex(warmupIndex), minAngle, maxAngle);
        }
        else if (currentIsCatch)
        {
            catchCountdown = RandomCatchInterval();
            currentTracker = null;
            currentAngle = Mathf.Clamp(catchTiltDeg, minAngle, maxAngle);
        }
        else
        {
            // Refill the interleaving queue with whichever trackers are still active,
            // shuffled (Fisher-Yates) so A/B order is balanced and unpredictable.
            if (trialQueue.Count == 0)
            {
                List<QuestPlusTracker> batch = new List<QuestPlusTracker>();
                if (IsActive(trackerA)) batch.Add(trackerA);
                if (IsActive(trackerB)) batch.Add(trackerB);

                for (int i = 0; i < batch.Count; i++)
                {
                    int r = UnityEngine.Random.Range(i, batch.Count);
                    var tmp = batch[i]; batch[i] = batch[r]; batch[r] = tmp;
                }
                foreach (var t in batch) trialQueue.Enqueue(t);
            }

            if (trialQueue.Count > 0) currentTracker = trialQueue.Dequeue();
            else { FinishCalibration(); return; }

            // QUEST+ chooses the next tilt magnitude (minimum expected entropy).
            currentAngle = currentTracker.GetNextStimulusDeg();
        }

        currentTiltDirection = (UnityEngine.Random.value > 0.5f) ? 1 : 0;

        if (networkingSync == null) networkingSync = FindObjectOfType<NetworkingSync>();
        if (networkingSync != null)
        {
            if (visualTaskPlane == null) visualTaskPlane = FindObjectOfType<FollowerPlane>();
            if (visualTaskPlane != null)
            {
                visualTaskPlane.gameObject.SetActive(true);
                networkingSync.BroadcastShowTask(currentTiltDirection, currentAngle);
            }
            else
            {
                Debug.LogError("[CRITICAL ERROR] VisualTaskPlane (Gabor Patch) is MISSING!");
            }
        }

        lastOnsetTime = Time.realtimeSinceStartup; // stimulus onset, for onset-to-onset ISI
        StartCoroutine(HideTaskAfterDelay(taskDuration));
        waitingForResponse = true;

        // Open the response window: if no answer arrives within responseWindow it is
        // logged as a timeout and the trial is re-placed (see CloseResponseWindow).
        if (responseWindowCoroutine != null) StopCoroutine(responseWindowCoroutine);
        responseWindowCoroutine = StartCoroutine(CloseResponseWindow());
    }

    private void ProcessResponse(int response)
    {
        waitingForResponse = false;
        if (responseWindowCoroutine != null) { StopCoroutine(responseWindowCoroutine); responseWindowCoroutine = null; }
        if (networkingSync != null) networkingSync.BroadcastResetTask();

        bool isCorrect = (response == currentTiltDirection);

        hasLastTrial = true;
        lastTrialCorrect = isCorrect;
        lastTrialTimedOut = false;
        lastTrialAngle = currentAngle;
        lastTrialDir = currentTiltDirection;

        if (currentIsWarmup)
        {
            // Warm-up trial: practice only — NOT fed to QUEST+/pooled, not counted toward
            // the trial budget or the catch cadence. Logged with tracker tag "warmup".
            warmupAnswered++;
            if (isCorrect) warmupCorrect++;
            lastTrackerName = "warmup";
            RecordWarmupRow(response, isCorrect ? 1 : 0);
            StartCoroutine(WaitAndTrigger());
            return;
        }

        if (currentIsCatch)
        {
            // Catch trial: scored for attentiveness only — NOT fed to QUEST+/pooled and
            // not counted toward the trial budget. A wrong answer here is a lapse.
            catchAnswered++;
            if (!isCorrect) catchErrors++;
            lastTrackerName = "catch";
            RecordCatchRow(response, isCorrect ? 1 : 0);
            StartCoroutine(WaitAndTrigger());
            return;
        }

        currentTracker.Update(isCorrect);
        pooled.UpdateAt(currentAngle, isCorrect);   // pooled posterior sees every scored trial

        // Track stats for the overlay.
        totalTrials++;
        catchCountdown--;                            // count down to the next catch
        if (isCorrect) totalCorrect++;
        lastTrackerName = currentTracker.name;

        // Record this trial for the calibration CSV / convergence plot.
        RecordCalibrationRow(response, isCorrect ? 1 : 0);

        StartCoroutine(WaitAndTrigger());
    }

    // No response arrived within responseWindow. Mirrors the experiment's timeout
    // handling: the trial is logged as a miss (response = -1, correct = -1). Crucially
    // the QUEST+ tracker is NOT updated — a missed response carries no perceptual
    // information, so the posterior and the tracker's trial count are left unchanged
    // and the trial is simply re-placed on the next iteration.
    private void OnResponseTimeout()
    {
        waitingForResponse = false;
        responseWindowCoroutine = null;
        if (networkingSync != null) networkingSync.BroadcastResetTask();

        hasLastTrial = true;
        lastTrialTimedOut = true;
        lastTrialAngle = currentAngle;
        lastTrialDir = currentTiltDirection;

        if (currentIsWarmup)
        {
            // Missed warm-up practice trial: not scored, not fed to QUEST+. Counted only
            // for the operator overlay so a disengaged participant is still visible.
            warmupTimeouts++;
            lastTrackerName = "warmup";
            RecordWarmupRow(-1, -1);
            Debug.Log("Calibration WARM-UP trial timed out — no response (practice, ignored).");
            StartCoroutine(WaitAndTrigger());
            return;
        }

        if (currentIsCatch)
        {
            // Missed catch: also an inattention signal, but counted separately from a
            // lapse error (no answer != wrong answer).
            catchTimeouts++;
            lastTrackerName = "catch";
            RecordCatchRow(-1, -1);
            Debug.Log("Calibration CATCH trial timed out — no response (inattention).");
            StartCoroutine(WaitAndTrigger());
            return;
        }

        totalTimeouts++;
        lastTrackerName = currentTracker.name;

        RecordCalibrationRow(-1, -1); // response = -1, correct = -1

        Debug.Log("Calibration response window timed out — no response (posterior unchanged, trial re-placed).");

        StartCoroutine(WaitAndTrigger());
    }

    // Appends one row to the per-trial calibration log. The tracker's current
    // estimate/SD are recorded as-is (for a timeout these are unchanged, so the
    // convergence line stays flat across the missed trial).
    private void RecordCalibrationRow(int response, int correct)
    {
        loggedPresentations++;
        calibrationRows.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture,
            "{0};{1};{2};{3:F4};{4};{5};{6};{7:F4};{8:F4}",
            currentTracker.name, loggedPresentations, currentTracker.Trials, currentAngle,
            currentTiltDirection, response, correct,
            currentTracker.ThresholdDeg(), currentTracker.PosteriorSDLog()));
    }

    // Logs a catch trial with tracker name "catch". The posterior columns carry the
    // current POOLED estimate (unchanged by the catch) so the analysis can ignore these
    // rows for the convergence line while still reading the catch result.
    private void RecordCatchRow(int response, int correct)
    {
        loggedPresentations++;
        calibrationRows.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture,
            "{0};{1};{2};{3:F4};{4};{5};{6};{7:F4};{8:F4}",
            "catch", loggedPresentations, catchAnswered, currentAngle,
            currentTiltDirection, response, correct,
            pooled.ThresholdDeg(), pooled.PosteriorSDLog()));
    }

    // Logs a warm-up (practice) trial with tracker name "warmup". Like catch rows the
    // posterior columns carry the current POOLED estimate (unchanged by the warm-up), so
    // the analysis ignores these rows for both the convergence line and the threshold fit.
    private void RecordWarmupRow(int response, int correct)
    {
        loggedPresentations++;
        calibrationRows.Add(string.Format(System.Globalization.CultureInfo.InvariantCulture,
            "{0};{1};{2};{3:F4};{4};{5};{6};{7:F4};{8:F4}",
            "warmup", loggedPresentations, warmupAnswered, currentAngle,
            currentTiltDirection, response, correct,
            pooled.ThresholdDeg(), pooled.PosteriorSDLog()));
    }

    private IEnumerator CloseResponseWindow()
    {
        float end = Time.realtimeSinceStartup + responseWindow;
        while (Time.realtimeSinceStartup < end)
        {
            if (!waitingForResponse) yield break; // answered in time
            yield return null;
        }
        if (waitingForResponse) OnResponseTimeout();
    }

    private IEnumerator WaitAndTrigger()
    {
        // ONSET-to-ONSET ISI matching the experiment's distribution: the next stimulus
        // appears interTrialDelay seconds after the PREVIOUS onset, with the same UNIFORM
        // jitter the experiment uses — a SINGLE uniform draw on
        // [interTrialDelay - isiJitter, + isiJitter] = [3.25, 4.75] s. We subtract the time
        // already spent on the stimulus + response so the onset cadence is fixed regardless
        // of how fast the participant answered. PROVISIONAL shape — see the jitter-shape
        // memory (validate coverage).
        float calculatedISI = interTrialDelay + UnityEngine.Random.Range(-isiJitter, isiJitter);

        float elapsedSinceOnset = Time.realtimeSinceStartup - lastOnsetTime;
        float wait = Mathf.Max(0.5f, calculatedISI - elapsedSinceOnset);
        yield return new WaitForSeconds(wait);
        TriggerNextTrial();
    }

    private void FinishCalibration()
    {
        // Final estimate from the POOLED posterior over all trials (both trackers,
        // weak prior) — more stable than averaging two possibly-divergent means.
        // alpha is the ~80.3%-correct point; we then invert the fitted PF at each
        // target accuracy to get the three difficulty tilts (the experimental
        // "threshold" level sits at the steeper targetThresholdPC).
        float aLog = pooled.ThresholdLog();
        bool converged = Converged();             // precision-based (the new definition)
        bool agree = TrackersAgree();             // QC robustness flag (tracker consistency)
        float pooledSD = pooled.PosteriorSDLog();
        float ratio = Mathf.Max(trackerA.ThresholdDeg(), trackerB.ThresholdDeg())
                      / Mathf.Max(1e-3f, Mathf.Min(trackerA.ThresholdDeg(), trackerB.ThresholdDeg()));
        // Attentiveness from the catch trials. Flag inattentive only with >= 2 errors AND
        // a rate over threshold (one chance lapse in a handful of catches is not evidence).
        float catchLapseRate = (catchAnswered > 0) ? (float)catchErrors / catchAnswered : 0f;
        bool catchOK = !(catchAnswered >= minCatchToJudge && catchErrors >= 2
                         && catchLapseRate > maxCatchLapseRate);
        // Ceiling rail: a pooled "threshold" at/above maxReliableThresholdDeg means QUEST+
        // never found a real threshold (e.g. a random responder running to the grid max) —
        // never reliable, however narrow the posterior.
        bool thresholdRailed = pooled.ThresholdDeg() >= maxReliableThresholdDeg;

        float thr  = Mathf.Clamp(AngleForPercentCorrect(aLog, targetThresholdPC), minAngle, maxAngle);
        float easy = Mathf.Clamp(AngleForPercentCorrect(aLog, targetEasyPC),      minAngle, maxAngle);
        float hard = Mathf.Clamp(AngleForPercentCorrect(aLog, targetHardPC),      minAngle, maxAngle);

        Debug.Log(string.Format(
            "QUEST+ Calibration complete ({10}; pooled SD {13:F3} / target {14:F2}).\n" +
            "A: {0:F2} deg (±{1:F2})  B: {2:F2} deg (±{3:F2})  POOLED: {11:F2} deg  ({12} scored trials)\n" +
            "Difficulty tilts -> HARD({4:P0})={5:F2}  THR({6:P0})={7:F2}  EASY({8:P0})={9:F2} deg",
            trackerA.ThresholdDeg(), trackerA.ThresholdSDDeg(),
            trackerB.ThresholdDeg(), trackerB.ThresholdSDDeg(),
            targetHardPC, hard, targetThresholdPC, thr, targetEasyPC, easy,
            converged ? "CONVERGED" : "NOT CONVERGED", pooled.ThresholdDeg(), totalTrials,
            pooledSD, convergenceSDLog));

        if (!converged)
            Debug.LogWarning(string.Format(
                "[CALIBRATION] Did not reach target precision (pooled SD {0:F3} > {1:F2} log10) in " +
                "{2} trials. Using pooled estimate {3:F2} deg; RECALIBRATION recommended.",
                pooledSD, convergenceSDLog, totalTrials, pooled.ThresholdDeg()));
        if (!agree)
            Debug.LogWarning(string.Format(
                "[CALIBRATION] Trackers disagree {0:F2}x (A={1:F2}, B={2:F2} deg) — responder may be " +
                "inconsistent / side-biased. Pooled estimate used; treat the threshold with caution.",
                ratio, trackerA.ThresholdDeg(), trackerB.ThresholdDeg()));

        Debug.Log(string.Format(
            "[CALIBRATION] Catch trials: {0}/{1} wrong (lapse rate {2:P0}); {3} missed.",
            catchErrors, catchAnswered, catchLapseRate, catchTimeouts));
        if (!catchOK)
            Debug.LogWarning(string.Format(
                "[CALIBRATION] High catch-trial lapse rate ({0:P0}, {1} errors over {2} easy anchors " +
                "> {3:P0}) — participant was inattentive; the threshold is UNRELIABLE despite the " +
                "posterior. RECALIBRATION recommended.",
                catchLapseRate, catchErrors, catchAnswered, maxCatchLapseRate));
        if (thresholdRailed)
            Debug.LogWarning(string.Format(
                "[CALIBRATION] Pooled threshold {0:F2} deg is at the grid ceiling (>= {1:F1}) — no real " +
                "threshold was found (random/non-responding). UNRELIABLE; RECALIBRATION required.",
                pooled.ThresholdDeg(), maxReliableThresholdDeg));

        // Stop the walk-in-place belt now that calibration is finished.
        StopTreadmillIfNeeded();

        // Store into this condition's per-condition set (and refresh the active set).
        if (ExperimentConductor.instance != null)
            ExperimentConductor.instance.StoreCalibration(calibrationCondition, thr, easy, hard);

        WriteCalibrationCsv(thr, easy, hard, converged);

        // RELIABLE = precise (converged) AND attentive (catch OK) AND a real, non-railed
        // threshold. A/B agreement is logged/warned above but does NOT gate the banner: at
        // ~20 trials/tracker two independent posteriors differ by > agreementLogTol ~30% of
        // the time even for attentive observers, so gating on it would falsely demote good
        // runs (see Analysis/calibration_convergence_sim.py). Surface it for the head-node
        // operator banner, then hold briefly before returning to setup.
        calibrationReliable = converged && catchOK && !thresholdRailed;
        calibrationFinished = true;
        StartCoroutine(HoldResultThenReturn());
    }

    // Keep the calibration scene up for resultDisplaySeconds so the operator sees the
    // green DONE / red UNRELIABLE banner, then return everyone to the setup scene.
    private IEnumerator HoldResultThenReturn()
    {
        yield return new WaitForSeconds(resultDisplaySeconds);
        if (networkingSync != null) networkingSync.BroadcastReturnToScene(0);
    }

    // Writes the per-trial calibration data to a CSV alongside the main experiment
    // log (same folder = ExperimentConductor.dataStoragePath, same identifier base
    // p##-v#-b#-t##, suffixed "_calibration_<condition>"). Feeds plot_calibration.py. The
    // output folder is the global, editable ExperimentConductor.dataStoragePath.
    private void WriteCalibrationCsv(float thr, float easy, float hard, bool converged)
    {
        try
        {
            string dir = (ExperimentConductor.instance != null && !string.IsNullOrEmpty(ExperimentConductor.instance.dataStoragePath))
                ? ExperimentConductor.instance.dataStoragePath
                : Application.persistentDataPath;
            Directory.CreateDirectory(dir);

            ExperimentConductor ec = ExperimentConductor.instance;
            string id = (ec != null) ? ec.GetFileIdentifier() : "calib";
            string baseName = Path.Combine(dir, id + "_calibration_" + calibrationCondition);
            string path = baseName + ".csv";
            int n = 0;
            while (File.Exists(path)) { path = baseName + "-" + (++n) + ".csv"; }

            var ic = System.Globalization.CultureInfo.InvariantCulture;
            var lines = new List<string>();
            // Metadata (parsed by the plot script for the title / reference lines).
            lines.Add(string.Format(ic, "# threshold_deg={0:F4};easy_deg={1:F4};hard_deg={2:F4}", thr, easy, hard));
            lines.Add(string.Format(ic, "# guess={0:F2};slope_beta={1:F2};lapse={2:F2};targetPC_hard={3:F2};targetPC_thr={4:F2};targetPC_easy={5:F2};min_angle_deg={6:F3};max_angle_deg={7:F3}",
                guessRate, slopeBeta, lapseRate, targetHardPC, targetThresholdPC, targetEasyPC, minAngle, maxAngle));
            // Convergence metadata: per-tracker finals, pooled estimate + its SD (the
            // precision-based convergence signal), scored-trial count, converged flag
            // (precision met), and the A/B agreement robustness flag.
            lines.Add(string.Format(ic, "# trackerA_deg={0:F4};trackerB_deg={1:F4};pooled_deg={2:F4};pooled_sd_log={3:F4};n_scored_trials={4};converged={5};agree={6}",
                trackerA.ThresholdDeg(), trackerB.ThresholdDeg(), pooled.ThresholdDeg(),
                pooled.PosteriorSDLog(), totalTrials, converged ? 1 : 0, TrackersAgree() ? 1 : 0));
            // Catch-trial (attentiveness) metadata: answered catches, lapse errors, missed.
            float catchRate = (catchAnswered > 0) ? (float)catchErrors / catchAnswered : 0f;
            lines.Add(string.Format(ic, "# n_catch={0};catch_errors={1};catch_timeouts={2};catch_lapse_rate={3:F4};catch_tilt_deg={4:F2}",
                catchAnswered, catchErrors, catchTimeouts, catchRate, Mathf.Clamp(catchTiltDeg, minAngle, maxAngle)));
            // Warm-up (practice) lead-in metadata: answered/correct/missed and the ramp endpoints.
            lines.Add(string.Format(ic, "# n_warmup={0};warmup_correct={1};warmup_timeouts={2};warmup_start_tilt_deg={3:F2};warmup_end_tilt_deg={4:F2}",
                warmupAnswered, warmupCorrect, warmupTimeouts,
                Mathf.Clamp(warmupStartTiltDeg, minAngle, maxAngle),
                Mathf.Clamp(warmupEndTiltDeg, minAngle, maxAngle)));
            // Column header + data rows. The identity columns repeat on every row so the
            // table names its participant, visit and block even when read without its file
            // name (pandas skips the '#' lines; BIDS conversion renames the file).
            string idCols = (ec != null)
                ? string.Format(ic, ";{0};{1};{2};{3};{4};{5};{6}", ec.participant, ec.visit,
                    ec.CounterbalanceSlot, ec.block, ec.trial, calibrationCondition, ec.ConditionOrderFirst)
                : ";-1;-1;-1;-1;-1;" + calibrationCondition + ";n/a";
            lines.Add("tracker;global_trial;tracker_trial;presented_tilt_deg;tilt_direction;response;correct;post_mean_threshold_deg;post_sd_log"
                + ";participant;visit;counterbalance_slot;block;trial;condition;condition_order_first");
            foreach (string row in calibrationRows) lines.Add(row + idCols);

            File.WriteAllLines(path, lines.ToArray());
            Debug.Log("Calibration data saved to \"" + path + "\" (" + calibrationRows.Count + " trials)");
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to write calibration CSV: " + e.Message);
        }
    }

    // Invert the Weibull psychometric function: returns the tilt (deg) at which the
    // fitted PF reaches the target proportion correct, given the threshold parameter
    // alpha (log10 deg). Solves
    //   pc = gamma + (1-gamma-lambda)*(1 - exp(-10^(beta*(x-alpha))))
    // => x = alpha + log10(-ln(1-q)) / beta,   q = (pc - gamma)/(1 - gamma - lambda)
    private float AngleForPercentCorrect(float alphaLog, float pc)
    {
        float denom = 1f - guessRate - lapseRate;
        if (denom <= 1e-4f) return Mathf.Pow(10f, alphaLog);
        float q = (pc - guessRate) / denom;
        q = Mathf.Clamp(q, 1e-4f, 1f - 1e-4f);
        double inner = -System.Math.Log(1.0 - q);          // > 0 for q in (0,1)
        float xLog = alphaLog + (float)(System.Math.Log10(inner)) / slopeBeta;
        return Mathf.Pow(10f, xLog);
    }

    private IEnumerator HideTaskAfterDelay(float duration)
    {
        float end = Time.realtimeSinceStartup + duration;
        while (Time.realtimeSinceStartup < end) yield return null;
        if (networkingSync != null) networkingSync.BroadcastResetTask();
    }

    // -------------------------------------------------------------------------
    //  QUEST+ tracker — single-parameter (threshold) Bayesian adaptive method.
    //  Minimum-expected-entropy stimulus placement (Watson 2017). The grids and
    //  per-trial likelihood are precomputed; per-trial cost is O(stim*param),
    //  trivial within the multi-second inter-trial interval.
    // -------------------------------------------------------------------------
    private class QuestPlusTracker
    {
        private const double EPS = 1e-6;

        public string name;
        private readonly float gamma, lambda, beta;
        private readonly float[] stimLog;    // candidate stimuli, log10(deg)
        private readonly float[] paramLog;   // candidate thresholds alpha, log10(deg)
        private readonly double[] posterior; // over paramLog
        private readonly double[,] pCorrect; // [stim, param] = P(correct | stim, alpha)
        private int lastStimIdx = 0;
        private int trials = 0;

        public int Trials { get { return trials; } }

        public QuestPlusTracker(string name, float minDeg, float maxDeg, int nStim, int nParam,
                                float beta, float lambda, float gamma, float priorCenterDeg, float priorLogSD)
        {
            this.name = name;
            this.beta = beta;
            this.lambda = lambda;
            this.gamma = gamma;

            float loMin = Mathf.Log10(Mathf.Max(1e-3f, minDeg));
            float loMax = Mathf.Log10(Mathf.Max(minDeg + 1e-3f, maxDeg));

            stimLog = LinSpace(loMin, loMax, Mathf.Max(2, nStim));
            paramLog = LinSpace(loMin, loMax, Mathf.Max(2, nParam));

            // Precompute the likelihood P(correct | stim, alpha) over the full grid.
            pCorrect = new double[stimLog.Length, paramLog.Length];
            for (int s = 0; s < stimLog.Length; s++)
                for (int a = 0; a < paramLog.Length; a++)
                    pCorrect[s, a] = Clamp01(Weibull(stimLog[s], paramLog[a]));

            // Prior over alpha: Gaussian in log10 space centred on priorCenterDeg.
            // priorLogSD <= 0 falls back to a uniform prior.
            posterior = new double[paramLog.Length];
            float centerLog = Mathf.Log10(Mathf.Clamp(priorCenterDeg, minDeg, maxDeg));
            double sum = 0.0;
            for (int a = 0; a < paramLog.Length; a++)
            {
                double p;
                if (priorLogSD > 0f)
                {
                    double z = (paramLog[a] - centerLog) / priorLogSD;
                    p = Math.Exp(-0.5 * z * z);
                }
                else p = 1.0;
                posterior[a] = p;
                sum += p;
            }
            Normalize(posterior, sum);
        }

        private float Weibull(float xLog, float aLog)
        {
            double t = Math.Pow(10.0, beta * (xLog - aLog));
            double f = 1.0 - Math.Exp(-t);
            return (float)(gamma + (1.0 - gamma - lambda) * f);
        }

        // Choose the stimulus minimising the expected posterior entropy.
        public float GetNextStimulusDeg()
        {
            int bestIdx = 0;
            double bestEH = double.MaxValue;

            for (int s = 0; s < stimLog.Length; s++)
            {
                double pc = 0.0;
                for (int a = 0; a < paramLog.Length; a++)
                    pc += posterior[a] * pCorrect[s, a];
                pc = Clamp01(pc);
                double pi = 1.0 - pc;

                double Hc = EntropyAfter(s, true, pc);
                double Hi = EntropyAfter(s, false, pi);
                double eh = pc * Hc + pi * Hi;

                if (eh < bestEH) { bestEH = eh; bestIdx = s; }
            }

            lastStimIdx = bestIdx;
            return Mathf.Pow(10f, stimLog[bestIdx]);
        }

        // Entropy of the posterior that would result from a given outcome.
        private double EntropyAfter(int s, bool correct, double norm)
        {
            if (norm <= EPS) return 0.0;
            double H = 0.0;
            for (int a = 0; a < paramLog.Length; a++)
            {
                double like = correct ? pCorrect[s, a] : (1.0 - pCorrect[s, a]);
                double post = posterior[a] * like / norm;
                if (post > EPS) H -= post * Math.Log(post);
            }
            return H;
        }

        // Bayesian update with the observed outcome at the last presented stimulus.
        public void Update(bool correct)
        {
            double sum = 0.0;
            for (int a = 0; a < paramLog.Length; a++)
            {
                double like = correct ? pCorrect[lastStimIdx, a] : (1.0 - pCorrect[lastStimIdx, a]);
                posterior[a] *= like;
                sum += posterior[a];
            }
            Normalize(posterior, sum);
            trials++;
        }

        // Update with an outcome observed at an ARBITRARY presented tilt (deg), not
        // necessarily one this tracker would have chosen. Used by the pooled tracker,
        // which is fed every trial from both interleaved trackers. The likelihood is
        // evaluated at the exact presented tilt (no grid-snap needed).
        public void UpdateAt(float stimDeg, bool correct)
        {
            float xLog = Mathf.Log10(Mathf.Max(1e-3f, stimDeg));
            double sum = 0.0;
            for (int a = 0; a < paramLog.Length; a++)
            {
                double pc = Clamp01(Weibull(xLog, paramLog[a]));
                double like = correct ? pc : (1.0 - pc);
                posterior[a] *= like;
                sum += posterior[a];
            }
            Normalize(posterior, sum);
            trials++;
        }

        // Posterior-mean threshold (alpha) in log10 degrees.
        public float ThresholdLog()
        {
            double meanLog = 0.0;
            for (int a = 0; a < paramLog.Length; a++) meanLog += posterior[a] * paramLog[a];
            return (float)meanLog;
        }

        public float ThresholdDeg()
        {
            return Mathf.Pow(10f, ThresholdLog());
        }

        public float PosteriorSDLog()
        {
            double meanLog = 0.0;
            for (int a = 0; a < paramLog.Length; a++) meanLog += posterior[a] * paramLog[a];
            double var = 0.0;
            for (int a = 0; a < paramLog.Length; a++)
            {
                double d = paramLog[a] - meanLog;
                var += posterior[a] * d * d;
            }
            return (float)Math.Sqrt(Math.Max(0.0, var));
        }

        // Approximate +/- SD in degrees (mean*10^sdLog - mean), for the status overlay.
        public float ThresholdSDDeg()
        {
            float thr = ThresholdDeg();
            float sdLog = PosteriorSDLog();
            return thr * (Mathf.Pow(10f, sdLog) - 1f);
        }

        private static float[] LinSpace(float lo, float hi, int n)
        {
            float[] v = new float[n];
            if (n == 1) { v[0] = lo; return v; }
            float step = (hi - lo) / (n - 1);
            for (int i = 0; i < n; i++) v[i] = lo + step * i;
            return v;
        }

        private static double Clamp01(double x)
        {
            if (x < EPS) return EPS;
            if (x > 1.0 - EPS) return 1.0 - EPS;
            return x;
        }

        private static void Normalize(double[] p, double sum)
        {
            if (sum <= 0.0)
            {
                double u = 1.0 / p.Length;
                for (int i = 0; i < p.Length; i++) p[i] = u;
                return;
            }
            for (int i = 0; i < p.Length; i++) p[i] /= sum;
        }
    }
}
