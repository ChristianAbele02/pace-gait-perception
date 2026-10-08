using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Text;

/// <summary>
/// Runs one experimental BLOCK: lays out the trial sequence, drives the trial state
/// machine, and bookends the block with the two matched baselines.
///
/// Trials are "checkpoints" placed along the walking spline (or fired on a timer when
/// standing). Layout is decided up front and deterministically from <c>mapSeed</c>, so
/// every render node reconstructs an identical sequence: difficulty is drawn from a pool
/// WEIGHTED toward threshold (where the stride-phase oscillation lives), and tilt
/// direction from a CONSTRAINED shuffle that forbids more than <c>maxSameTiltRun</c>
/// identical directions in a row (an unconstrained 50/50 draw produces long runs that
/// participants notice and exploit).
///
/// Onset timing is what makes or breaks the stride-phase analysis, so see the placement
/// loop in <c>GenerateCheckpoints</c>: walking onsets accumulate their ISI jitter into a
/// bounded random walk rather than sitting on a fixed grid, which is what keeps stride-
/// phase coverage uniform at every cadence.
///
/// Block structure: a 30 s standing baseline, then five easy training trials, then the
/// experimental trials, then a matched 30 s cool-down. The two baselines are equal by
/// design so they can be contrasted directly.
/// </summary>
public class CheckpointManager : MonoBehaviour
{
    public static CheckpointManager instance;

    [Header("References")]
    public BezierSpline spline;
    public GameObject checkpointPrefab;
    public Transform checkpointsParent;
    public GameObject playerObj;
    public TreadmillController treadmillController;
    public FollowerPlane visualTaskPlane;
    private SplineWalker splineWalker;
    private NetworkingSync networkingSync;
    private ExperimentConductor experimentConductor;




    [Header("Checkpoint Settings")]
    public int maxCheckpoints = 8;
    [Tooltip("Uniform y-position for all checkpoints.")]
    public float checkpointHeight = 1.0f;
    [Tooltip("Max jitter along path, as fraction of path (0.01–0.05 is typical).")]
    public float tJitter = 0.05f;
    [Tooltip("How long (in seconds) to wait at each speed step.")]
    public float rampDuration = 1.0f;
    [Header("Resume Settings")]
    public float resumeDelay = 3f;
    public float responseWindow = 2f;
    public float taskDuration = 0.1f;

    [Header("On-screen Messages (head-height text)")]
    [Tooltip("Duration (s) of BOTH the end-of-run cool-down AND the matched pre-experiment "
        + "baseline shown at the start. Kept as one value so the ECG baseline and the "
        + "cool-down are the same length by construction.")]
    public float cooldownSeconds = 30f;
    [Tooltip("Distance (m) before the end of the path at which the 'treadmill will stop "
        + "soon' warning appears (walking blocks).")]
    public float treadmillWarnDistanceM = 8f;
    [TextArea] public string baselineMessage =
        "The experiment will start soon.\nPlease stand still and relax.";
    [TextArea] public string treadmillWarnMessage =
        "The treadmill will stop soon.";
    [TextArea] public string cooldownMessage =
        "Cool-down period.\nPlease remain standing calmly.";

    [Header("Debug Stimulus Hold (debug mode only)")]
    // In debug mode the stimulus normally uses the real taskDuration so timing is
    // faithful. Enable the hold below to keep the Gabor visible longer for visual
    // inspection — this has NO effect in normal (non-debug) runs.
    [Tooltip("Debug mode only: hold the Gabor visible longer than taskDuration for inspection. No effect in normal runs.")]
    public bool debugHoldStimulus = false;
    [Tooltip("Seconds to hold the stimulus when Debug Hold Stimulus is enabled (debug mode only).")]
    public float debugStimulusHoldSeconds = 1.0f;

    [Header("Generation Settings")]
    [Tooltip("Target time (seconds) between stimuli. 4 s (with the 2 s response window) "
        + "packs more trials into the same path/duration for denser stride-phase sampling.")]
    public float targetISI = 4.0f;
    [Tooltip("Uniform ISI jitter HALF-RANGE (seconds): the ISI is target +/- this, drawn "
        + "uniformly. 0.75 => ISI in [3.25, 4.75] s. Applies the same way to standing/"
        + "calibration (a single uniform draw per interval) and to walking, where the draws "
        + "ACCUMULATE into a random walk on onset time (see the placement loop) so that "
        + "onset stride-phase coverage does not depend on the participant's cadence.")]
    public float isiJitter = 0.75f;
    [Tooltip("Bound (seconds) on the accumulated walking ISI drift; the running offset is "
        + "reflected at +/- this value so the checkpoint train cannot wander off the end of "
        + "the spline. 3 s = 3 m of path at 1 m/s, well inside the spare path beyond endT. "
        + "Only needs raising if a block is very long AND you want the walk to mix faster.")]
    public float isiDriftBoundSeconds = 3.0f;
    public int trainingCheckpoints = 5;

    [Header("Difficulty Mix (a7)")]
    [Tooltip("Relative share of THRESHOLD-difficulty experimental trials. The stride-"
        + "phase (Davidson) oscillation lives almost entirely in near-threshold trials "
        + "- accuracy at EASY sits near ceiling and at HARD near floor, so both carry "
        + "little modulation. Weighting toward threshold concentrates trials where the "
        + "effect can actually appear. These three weights are normalised, so only "
        + "their ratio matters (default 0.85 / 0.10 / 0.05). Raised from 0.70/0.15/0.15 "
        + "on the Davidson-anchored power analysis: at the trial budgets we can collect, "
        + "the +21% near-threshold yield is needed to reach usable per-participant power "
        + "(see Analysis/davidson_power/). Applied to BOTH conditions so standing and "
        + "walking blocks - and the per-condition QUEST+ calibration they mimic - stay "
        + "comparable.")]
    public float thresholdWeight = 0.85f;
    [Tooltip("Relative share of EASY trials (engagement / attention check).")]
    public float easyWeight = 0.10f;
    [Tooltip("Relative share of HARD trials (engagement / floor check).")]
    public float hardWeight = 0.05f;

    [Header("Sequence Constraints (a8)")]
    [Tooltip("Maximum number of identical tilt DIRECTIONS allowed in a row. The shuffled "
        + "direction sequence is reordered so no run of left/right exceeds this, which "
        + "stops long same-side streaks from manufacturing a response side-bias. "
        + "Default 2 (i.e. never 3+ in a row).")]
    public int maxSameTiltRun = 2;

    [Header("Map Generation")]
    [Tooltip("Sets the same seed on all machines to get identical checkpoint layouts.")]
    public int mapSeed = 12345;

    [Header("Logger")]
    public ILLLoggerController logger; // Reference to logger for saving responses

    #region Internal State
    // store checkpoint info and objects
    private List<CheckpointInfo> checkpointInfos = new List<CheckpointInfo>();
    private List<GameObject> spawnedCheckpoints = new List<GameObject>();
    
    // Constants for Logging
    private const string LOG_TYPE_STANDING = "standing";
    private const string LOG_TYPE_WALKING = "walking";

    // Checkpoint Coroutine Tracking
    private Coroutine responseWindowCoroutine;
    
    // Needed for controlling movement
    private float defaultSpeed;
    // Track which checkpoints have been triggered
    private HashSet<int> triggeredCheckpointIndices = new HashSet<int>();
    // Response tracking
    private bool waitingForResponse = false;
    private int lastResponse = -1; // -1 means no response yet
    private float stimulusOnsetTimestamp; //Timestamp of reaching checkpoint
    private bool wasStanding;
    private bool isGameStarted = false;
    private bool isHeadNode = false;
    // Visual Task State
    // Response mapping: the participant presses the side the Gabor leans toward,
    // so the CORRECT response equals the tilt direction (0 = Left, 1 = Right).
    private int expectedResponse = -1;
    private bool isCheckpointSequenceActive = false;
    private float currentVisualAngle = 0f;
    private int currentTiltDirection; //0= left and 1=right
    private int currentCheckpointIndex = -1; // ball to hide at stimulus onset
    public bool isTrainingTrial = false;

    // --- Overlay / run statistics (experimental trials only) ---
    private int respTotal = 0;
    private int respCorrect = 0;
    private bool hasLastResp = false;
    private bool lastRespCorrect = false;
    private float lastRespRT = -1f;
    private float lastRespAngle = 0f;
    private int lastRespDir = 0;

    // Cached Waits
    private WaitForSeconds waitRampDuration;
    private WaitForSeconds waitResumeDelay;
    private WaitForSeconds waitResponseWindow;
    private WaitForSeconds waitCooldown;

    // Encapsulate checkpoint state for replay or logging
    private struct CheckpointInfo
    {
        public Vector3 pos;
        public Vector3 rot;
        public float distanceOnPath;
        public bool triggersStop; // true = standing checkpoint, triggers pause
        public float visualAngle;
        public int tiltDirection;
        public bool isTraining;
    }
    #endregion

    void Awake()
    {
        if (instance == null)
            instance = this;
        else
        {
            Destroy(gameObject);
        }
        experimentConductor = ExperimentConductor.instance;
        if (experimentConductor == null)
        {
            experimentConductor = FindObjectOfType<ExperimentConductor>();
        }
    }

    void Start()
    {
        Time.timeScale = 1.0f;
		networkingSync = FindObjectOfType<NetworkingSync>();
        isHeadNode = (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName);        
        defaultSpeed = (ExperimentConductor.instance != null) ? ExperimentConductor.instance.experimentSpeed : 1.0f;
        
        Debug.Log("Setting Speed to:" + defaultSpeed);
        
        // Robustly Find SplineWalker ---
        if (splineWalker == null) splineWalker = FindObjectOfType<SplineWalker>();
        if (splineWalker != null && playerObj == null) 
        {
            playerObj = splineWalker.gameObject;
        }
        
        if (splineWalker == null && playerObj != null)
        {
            splineWalker = playerObj.GetComponent<SplineWalker>();
        }

        if (treadmillController == null) treadmillController = FindObjectOfType<TreadmillController>();
        if (spline == null) spline = FindObjectOfType<BezierSpline>();
        if (visualTaskPlane == null) visualTaskPlane = FindObjectOfType<FollowerPlane>();

        // --- VALIDATION LOGS ---
        if (isHeadNode)
        {
            if (splineWalker == null) Debug.LogError("CRITICAL FAILURE: SplineWalker Script NOT found in scene!");
            if (treadmillController == null) Debug.LogError("CRITICAL FAILURE: TreadmillController NOT found!");
        }

        // --- PAUSE CHECK (Run on ALL nodes) ---
        // Ensure we start paused until "Start Game" signal is received
        if (splineWalker != null) 
        {
            splineWalker.speed = 0f;
            splineWalker.isMovementPaused = true;
        }

        if (isHeadNode)
        {
            if (treadmillController != null) 
            {
                treadmillController.StopTreadmill();
            }
        }
        // Generate mapSeed out of participant number and experiment number
        if (ExperimentConductor.instance != null)
        {
            // Get the unique identifier string (e.g., "p05-b1-t03")
            string experimentID = ExperimentConductor.instance.GetExperimentIdentifier();

            // Convert to a seed with a DETERMINISTIC hash (not string.GetHashCode(),
            // which is not guaranteed identical across platforms/runtimes — the
            // Linux head node and Windows render nodes could otherwise derive
            // different seeds and generate divergent checkpoint jitter/layouts).
            // FNV-1a over the char codes is pure integer math: identical on every
            // node and stable across sessions => reproducible, matched layouts.
            this.mapSeed = DeterministicSeed(experimentID);

            Debug.Log("CheckpointManager: Generated mapSeed '" + this.mapSeed + "' from Experiment ID '" + experimentID + "'");
        }
        else
        {
            Debug.LogError("CheckpointManager: ExperimentConductor not found! Using default Inspector seed: " + this.mapSeed);
        }


        if (this.IsDebugMode)
        {
            Debug.Log("CheckpointManager: DEBUG MODE ENABLED. Checkpoints visible, data NOT saved. Stimulus uses real taskDuration (" + taskDuration + "s).");
        }

        // Cache Waits
        waitRampDuration = new WaitForSeconds(rampDuration);
        waitResumeDelay = new WaitForSeconds(resumeDelay);
        waitResponseWindow = new WaitForSeconds(responseWindow);
        waitCooldown = new WaitForSeconds(cooldownSeconds);

        // Per-condition calibration: load THIS block's difficulty set (standing vs walking)
        // into the active thresholds BEFORE generating checkpoints, so each block presents
        // tilts around its OWN threshold (a walking threshold is typically larger than
        // standing). The logger then also records the block's own threshold-angle.
        if (experimentConductor != null)
        {
            string blockCondition = (experimentConductor.block == 0) ? LOG_TYPE_STANDING : LOG_TYPE_WALKING;
            experimentConductor.SetActiveCondition(blockCondition);
        }

        GenerateCheckpoints();
    }

    public int GetTotalCheckpointCount()
    {
        return checkpointInfos.Count;
    }

    // Platform-stable hash (FNV-1a, 32-bit) over a string's UTF-16 code units.
    // Pure integer arithmetic => identical on Linux (head node) and Windows (render
    // nodes) and stable across runs, unlike string.GetHashCode(). Used so every
    // node derives the SAME mapSeed and thus identical checkpoint layouts.
    private int DeterministicSeed(string s)
    {
        unchecked
        {
            uint hash = 2166136261u;        // FNV offset basis
            const uint prime = 16777619u;   // FNV prime
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                hash = (hash ^ (byte)(c & 0xFF)) * prime;
                hash = (hash ^ (byte)((c >> 8) & 0xFF)) * prime;
            }
            return (int)hash;
        }
    }

    public int GetTriggeredCheckpointCount()
    {
        return triggeredCheckpointIndices.Count;
    }

    // Show/hide a checkpoint marker's mesh. Called locally and via RPC so the ball
    // disappears on every node the instant its stimulus is shown. In non-debug mode
    // the meshes are already disabled, so this is a harmless no-op there.
    public void SetCheckpointVisible(int index, bool visible)
    {
        if (index < 0 || index >= spawnedCheckpoints.Count) return;
        GameObject cp = spawnedCheckpoints[index];
        if (cp == null) return;
        foreach (var r in cp.GetComponentsInChildren<MeshRenderer>()) r.enabled = visible;
    }

    // How long the Gabor stays on screen. Normally the real taskDuration; in debug
    // mode, optionally held longer (debugStimulusHoldSeconds) for visual inspection.
    private float EffectiveStimulusDuration()
    {
        if (IsDebugMode && debugHoldStimulus && debugStimulusHoldSeconds > 0f)
            return debugStimulusHoldSeconds;
        return taskDuration;
    }

    // Rich status overlay for the main experiment (head node only).
    public string GetExperimentStatusReport()
    {
        var ec = (experimentConductor != null) ? experimentConductor : ExperimentConductor.instance;
        var sb = new StringBuilder();

        sb.Append("=== EXPERIMENT (tilt 2-AFC) ===\n");
        if (ec != null)
        {
            string mode = (ec.block == 0) ? "STANDING" : "WALKING";
            sb.AppendFormat("P {0} | B {1} | T {2}  [{3}]{4}\n",
                ec.participant, ec.block, ec.trial, mode, IsDebugMode ? " (DEBUG)" : "");
            sb.AppendFormat("Thr {0:F2} | Easy {1:F2} | Hard {2:F2} deg\n",
                ec.thresholdAngle, ec.easyAngle, ec.hardAngle);
        }

        int triggered = GetTriggeredCheckpointCount();
        int total = GetTotalCheckpointCount();
        float pct = (total > 0) ? 100f * triggered / total : 0f;
        sb.AppendFormat("Progress: {0}/{1} ({2:F0}%)  [{3} training]\n",
            triggered, total, pct, trainingCheckpoints);

        float acc = (respTotal > 0) ? 100f * respCorrect / respTotal : 0f;
        sb.AppendFormat("Accuracy: {0:F0}% ({1}/{2} scored)\n", acc, respCorrect, respTotal);

        if (hasLastResp)
            sb.AppendFormat("Last: {0} ({1:F3}s) @ {2:F2}deg [{3}] {4}\n",
                lastRespCorrect ? "CORRECT" : "WRONG", lastRespRT, lastRespAngle,
                DifficultyLabel(lastRespAngle), lastRespDir == 0 ? "LEFT" : "RIGHT");
        else
            sb.Append("Last: -\n");

        if (waitingForResponse)
            sb.AppendFormat("NOW: {0:F2}deg [{1}] {2} (awaiting){3}\n",
                currentVisualAngle, DifficultyLabel(currentVisualAngle),
                currentTiltDirection == 0 ? "LEFT" : "RIGHT",
                isTrainingTrial ? " TRAINING" : "");

        return sb.ToString();
    }

    private string DifficultyLabel(float angle)
    {
        var ec = (experimentConductor != null) ? experimentConductor : ExperimentConductor.instance;
        if (ec == null) return "?";
        float dT = Mathf.Abs(angle - ec.thresholdAngle);
        float dE = Mathf.Abs(angle - ec.easyAngle);
        float dH = Mathf.Abs(angle - ec.hardAngle);
        if (dT <= dE && dT <= dH) return "THR";
        if (dE <= dH) return "EASY";
        return "HARD";
    }

    //Helper for FPSCounter
    public string GetNextCheckpointStats(float currentDist, float currentSpeed)
    {
        if (checkpointInfos == null || checkpointInfos.Count == 0) return "N/A";
        
        // Find first checkpoint ahead of us
        foreach (var info in checkpointInfos)
        {
            if (info.distanceOnPath > currentDist) 
            {
                 float distRemaining = info.distanceOnPath - currentDist;
                 
                 string timeStr = "Inf";
                 if (currentSpeed > 0.01f)
                 {
                     float timeRemaining = distRemaining / currentSpeed;
                     timeStr = timeRemaining.ToString("F1") + "s";
                 }
                 else if (experimentConductor != null && experimentConductor.block == 0)
                 {
                     // In Standing Mode (Block 0), distance doesn't matter because it's time based.
                     // But we don't easily track "time to next" here without more complex logic.
                     // So maybe just show distance or "Time-Based"
                     timeStr = "Timer";
                 }
                 
                 return string.Format("{0:F1}m ({1})", distRemaining, timeStr);
            }
        }
        return "Done";
    }

   public void GenerateCheckpoints()
    {
        Random.InitState(mapSeed);
        
        // Cleanup old
        foreach (var cp in spawnedCheckpoints) if (cp != null) DestroyImmediate(cp);
        spawnedCheckpoints.Clear();
        checkpointInfos.Clear();
        triggeredCheckpointIndices.Clear();

        if (checkpointsParent == null) 
        {
            GameObject newParent = new GameObject("Checkpoints");
            checkpointsParent = newParent.transform;
            checkpointsParent.SetParent(transform);
        } 
        else 
        {
            int childCount = checkpointsParent.childCount;
            for (int i = childCount - 1; i >= 0; i--) DestroyImmediate(checkpointsParent.GetChild(i).gameObject);
        }

        // 1. Calculate Count — distance-driven so the WALKING block (distance
        //    method) and the STANDING block (time method) share the SAME mean ISI.
        //    Checkpoints sit on a uniform grid every (targetISI * speed) metres
        //    along the usable path [startT..endT]; at the configured speed that
        //    spatial spacing equals targetISI seconds, matching the standing
        //    block's time-based targetISI wait. Training checkpoints are PART of
        //    this grid (not added on top), so they no longer compress the ISI.
        if (targetISI <= 0)
        {
            Debug.LogError("CheckpointManager: targetISI must be greater than 0! Using default value of 8.0s.");
            targetISI = 8.0f;
        }

        float startT = 0.05f;
        float endT = 0.95f;

        // Speed used purely to convert the time ISI into a spatial spacing. Guard
        // against 0 (e.g. if a standing block was configured with speed 0) so the
        // count stays sensible; the standing block still waits targetISI in time.
        float speedForSpacing = defaultSpeed;
        if (speedForSpacing < 0.01f)
        {
            Debug.LogWarning("CheckpointManager: defaultSpeed ~0; using 1.0 m/s for checkpoint spacing.");
            speedForSpacing = 1.0f;
        }

        float usableLength = (endT - startT) * spline.Length;
        float spacingMeters = targetISI * speedForSpacing;

        int totalCount = Mathf.FloorToInt(usableLength / spacingMeters);
        // Experimental = grid minus the training checkpoints. Difficulty is now mixed by
        // weight (a7) rather than equal thirds, so no multiple-of-3 rounding is needed.
        int experimentalCount = Mathf.Max(0, totalCount - trainingCheckpoints);
        totalCount = trainingCheckpoints + experimentalCount;

        Debug.Log(string.Format(
            "Generating {0} checkpoints ({1} training + {2} experimental). " +
            "Spacing {3:F2} m => ISI {4:F2} s at {5:F2} m/s. Usable path {6:F1} m of {7:F1} m.",
            totalCount, trainingCheckpoints, experimentalCount,
            spacingMeters, spacingMeters / speedForSpacing, speedForSpacing,
            usableLength, spline.Length));

        // 2. Create Balanced Pools
        List<float> diffPool = new List<float>();
        List<int> dirPool = new List<int>();

        if (experimentConductor != null)
        {
            // A. Difficulty pool, WEIGHTED toward threshold (a7). Counts are the
            //    normalised weights x experimentalCount; the remainder goes to
            //    threshold so the pool always holds exactly experimentalCount entries
            //    (and the trial budget is spent where the oscillation can show up).
            float wSum = Mathf.Max(1e-6f, thresholdWeight + easyWeight + hardWeight);
            int nEasy = Mathf.FloorToInt(experimentalCount * (easyWeight / wSum));
            int nHard = Mathf.FloorToInt(experimentalCount * (hardWeight / wSum));
            int nThr  = experimentalCount - nEasy - nHard;   // remainder -> threshold
            for (int i = 0; i < nThr;  i++) diffPool.Add(experimentConductor.thresholdAngle);
            for (int i = 0; i < nEasy; i++) diffPool.Add(experimentConductor.easyAngle);
            for (int i = 0; i < nHard; i++) diffPool.Add(experimentConductor.hardAngle);

            Debug.Log(string.Format("Difficulty mix (a7): {0} threshold / {1} easy / {2} hard "
                + "(weights {3:F2}/{4:F2}/{5:F2}).", nThr, nEasy, nHard,
                thresholdWeight / wSum, easyWeight / wSum, hardWeight / wSum));

            // B. Direction Pool (50/50 split)
            int half = experimentalCount / 2;
            for (int i = 0; i < half; i++) dirPool.Add(0); // Left
            for (int i = 0; i < (experimentalCount - half); i++) dirPool.Add(1); // Right
        }
        else
        {
            // Fallback for testing without conductor
            for (int i = 0; i < experimentalCount; i++) { diffPool.Add(45f); dirPool.Add(Random.Range(0,2)); }
        }

        // 3. Shuffle. Difficulty is shuffled freely; the DIRECTION sequence uses a
        //    constrained shuffle so no more than maxSameTiltRun identical tilts occur
        //    in a row (a8). Both draw from the seeded UnityEngine.Random, so every node
        //    still derives an identical, reproducible layout.
        diffPool = diffPool.OrderBy(x => Random.value).ToList();
        dirPool = ConstrainedShuffleNoLongRun(dirPool, Mathf.Max(1, maxSameTiltRun));

        // 4. Place Checkpoints on the uniform grid. step is the normalized path
        //    distance corresponding to spacingMeters (=> targetISI at this speed).
        float step = spacingMeters / spline.Length;

        // Walking jitter: CUMULATIVE (a random walk on onset time), not a per-node offset
        // around a fixed grid. Each successive checkpoint is placed one ISI further plus a
        // fresh uniform draw of +/-isiJitter, and that draw ACCUMULATES into a running
        // offset.
        //
        // Why not the old fixed-grid + independent offset: with a fixed grid the absolute
        // onset time is i*ISI + u_i, so whenever the participant's stride period T divides
        // the ISI (T = 4/k s — notably T = 1.0 s, i.e. a very ordinary 120 steps/min), the
        // grid term i*ISI mod T is the SAME for every trial and contributes no phase spread
        // at all. All that is left is the jitter, and a +/-0.75 s draw spans 1.5 stride
        // periods there — 1.5 cycles wrapped modulo one cycle is not uniform. Measured
        // Rayleigh R of onset stride-phase was 0.18-0.22 at T = 1.0 s against a 0.129
        // significance threshold, i.e. detectably non-uniform coverage at a common cadence.
        // Accumulating the jitter decorrelates absolute onset time from ANY stride period,
        // so coverage no longer depends on the participant's cadence (measured R ~ 0.07).
        //
        // Two side benefits: the inter-checkpoint SPACING is now a single draw, ISI +/-
        // isiJitter = [3.25, 4.75] s, rather than the difference of two independent offsets
        // ([2.5, 5.5] s) — tighter, and further clear of the response window; and it matches
        // what the analysis simulator has always generated.
        //
        // The running offset is REFLECTED at +/-isiDriftBoundSeconds so the walk cannot
        // wander off the end of the spline. Reflection costs nothing in coverage (it only
        // shortens a step that would cross the bound, so spacing stays inside ISI +/-
        // isiJitter) but caps the total displacement at a known, small number of metres.
        float maxJitterSeconds = isiJitter;
        float maxJitter = (speedForSpacing * maxJitterSeconds) / spline.Length;   // normalised

        // Spacing safety: the MINIMUM inter-onset gap is now targetISI - isiJitter. It must
        // still clear the response window, or a trial could open before the previous one
        // closed (which CheckpointManager would have to drop as an overlap).
        float minGapSeconds = targetISI - isiJitter;
        if (minGapSeconds <= responseWindow)
        {
            Debug.LogWarning(string.Format(
                "ISI jitter {0:F2}s leaves a minimum gap of {1:F2}s, which does not clear the "
                + "{2:F2}s response window. Clamping jitter.", isiJitter, minGapSeconds, responseWindow));
            maxJitterSeconds = Mathf.Max(0f, targetISI - responseWindow - 0.25f);
            maxJitter = (speedForSpacing * maxJitterSeconds) / spline.Length;
        }

        // Drift bound, normalised. Keep the walk inside the path we actually have: the grid
        // ends at endT, so the spare headroom is (1 - endT) of the spline.
        float driftBound = (speedForSpacing * Mathf.Max(0f, isiDriftBoundSeconds)) / spline.Length;
        float headroom = (1f - endT) * 0.9f;
        if (driftBound > headroom)
        {
            Debug.LogWarning(string.Format(
                "ISI drift bound {0:F2}s exceeds the spare path beyond endT; clamping to {1:F2}s.",
                isiDriftBoundSeconds, headroom * spline.Length / speedForSpacing));
            driftBound = headroom;
        }

        // Running (accumulated) jitter offset, in normalised path units. Exactly one
        // Random.Range call per checkpoint, in order, so every render node still derives an
        // identical layout from the shared mapSeed.
        float driftOffset = 0f;

        for (int i = 0; i < totalCount; i++)
        {
            float tBase = startT + (i * step);
            driftOffset += Random.Range(-maxJitter, maxJitter);
            // Reflect into [-driftBound, +driftBound]. Closed-form triangle fold (period
            // 4*driftBound, identity inside the bound) rather than a reflect-until-inside
            // loop: constant time whatever the bound, so a mis-set tiny bound cannot stall
            // scene generation mid-session.
            if (driftBound > 0f)
            {
                float y = Mathf.Repeat(driftOffset + driftBound, 4f * driftBound);
                driftOffset = (y <= 2f * driftBound) ? (y - driftBound) : (3f * driftBound - y);
            }
            float tJ = Mathf.Clamp01(tBase + driftOffset);

            bool isTraining = (i < trainingCheckpoints);
            float angle;
            int direction;

            if (isTraining)
            {
                // Training: Always Easy, Random Direction
                angle = (experimentConductor != null) ? experimentConductor.easyAngle : 45f;
                direction = Random.Range(0, 2);
            }
            else
            {
                // Experimental: Pop from shuffled pools
                int poolIndex = i - trainingCheckpoints;
                angle = diffPool[poolIndex];
                direction = dirPool[poolIndex];
            }

            CreateCheckpointAt(checkpointsParent, tBase, tJ, angle, direction, isTraining);
        }
    }

    // Returns a permutation of `pool` in which no value repeats more than `maxRun`
    // times in a row (a8). Greedy: at each step, among the values that still have
    // copies AND wouldn't break the run limit, pick the one with the most remaining
    // copies (ties broken randomly via the seeded UnityEngine.Random). Always picking
    // the most-plentiful eligible value is what guarantees the algorithm never strands
    // a leftover that can only be placed as a too-long run. Deterministic for a given
    // mapSeed, so all render nodes reproduce the identical sequence.
    private List<int> ConstrainedShuffleNoLongRun(List<int> pool, int maxRun)
    {
        Dictionary<int, int> counts = new Dictionary<int, int>();
        foreach (int v in pool) { if (!counts.ContainsKey(v)) counts[v] = 0; counts[v]++; }

        List<int> result = new List<int>(pool.Count);
        int lastVal = int.MinValue;
        int runLen = 0;

        for (int placed = 0; placed < pool.Count; placed++)
        {
            // Find the largest remaining count among values not blocked by the run limit.
            int bestCount = -1;
            bool blockedOnly = true;
            foreach (var kv in counts)
            {
                if (kv.Value <= 0) continue;
                if (kv.Key == lastVal && runLen >= maxRun) continue; // would exceed run
                blockedOnly = false;
                if (kv.Value > bestCount) bestCount = kv.Value;
            }

            int chosen;
            if (blockedOnly)
            {
                // Everything left is the run-limited value (degenerate, e.g. a single
                // value remaining). Place it rather than loop forever.
                chosen = int.MinValue;
                foreach (var kv in counts) { if (kv.Value > 0) { chosen = kv.Key; break; } }
            }
            else
            {
                // Collect all eligible values tied for the most remaining copies, pick one.
                List<int> cands = new List<int>();
                foreach (var kv in counts)
                {
                    if (kv.Value != bestCount) continue;
                    if (kv.Key == lastVal && runLen >= maxRun) continue;
                    cands.Add(kv.Key);
                }
                chosen = cands[Random.Range(0, cands.Count)];
            }

            result.Add(chosen);
            counts[chosen]--;
            if (chosen == lastVal) runLen++; else { lastVal = chosen; runLen = 1; }
        }
        return result;
    }

    public bool IsDebugMode
    {
        get
        {
            if (experimentConductor == null) return false;
            return experimentConductor.isDebugMode;
        }
    }

    /// Instantiates a checkpoint prefab, places it at a position corresponding to tJittered,
    /// orients it along the path direction at tBase, and adds trigger behaviour.
    private void CreateCheckpointAt(Transform parent, float tBase, float tJittered, float angle, int direction, bool isTraining)
    {
        Vector3 dir = spline.GetDirectionT(tBase).normalized;
        Vector3 pos = spline.GetPointT(tJittered);
        pos.y += checkpointHeight;

        GameObject checkpoint = Instantiate(checkpointPrefab);
        checkpoint.transform.position = pos;
        checkpoint.transform.rotation = Quaternion.LookRotation(dir, Vector3.up);
        checkpoint.transform.SetParent(parent);

        // Default: Experiment Mode -> Invisible (disable MeshRenderer)
        // Debug Mode -> Visible (keep MeshRenderer enabled)
        MeshRenderer[] renderers = checkpoint.GetComponentsInChildren<MeshRenderer>();
        foreach(var r in renderers)
        {
            r.enabled = this.IsDebugMode;
        }

        float distOnPath = tJittered * spline.Length;
        bool triggersStop = (experimentConductor != null && experimentConductor.block == 0);

        CheckpointInfo info = new CheckpointInfo()
        {
            pos = checkpoint.transform.position,
            rot = checkpoint.transform.eulerAngles,
            distanceOnPath = distOnPath,
            triggersStop = triggersStop,
            visualAngle = angle,
            tiltDirection = direction,
            isTraining = isTraining
        };

        var trigger = checkpoint.AddComponent<CheckpointAction>();
        trigger.manager = this;
        trigger.checkpointIndex = checkpointInfos.Count;

        checkpointInfos.Add(info);
        spawnedCheckpoints.Add(checkpoint);
    }

    public void OnPlayerCheckpoint(int index)
    {
        // Only run this if we are NOT in the Standing Block Sequence (avoid double triggers)
        if (experimentConductor != null && experimentConductor.block == 0) return;
        if (index < 0 || index >= checkpointInfos.Count) return;
        if (!isHeadNode) return;
        // A1.S5: this checkpoint overlapped a still-open response window. At the
        // configured 8 s ISI / 3 s window this should never happen, but if jitter or a
        // slow response causes it, COUNT the overlap (logger) instead of silently
        // swallowing the stimulus, so it shows up in QC and trial accounting.
        if (waitingForResponse)
        {
            if (logger != null) logger.MarkDroppedCheckpoint(index);
            Debug.LogWarning("CheckpointManager: checkpoint " + index +
                " reached while still awaiting a response - logged as dropped/overlap.");
            return;
        }

        if (visualTaskPlane != null && networkingSync != null)
        {          
            isCheckpointSequenceActive = true;  
            CheckpointInfo info = checkpointInfos[index];
        
            this.currentVisualAngle = info.visualAngle;
            this.currentTiltDirection = info.tiltDirection;
            this.isTrainingTrial = info.isTraining;
            this.wasStanding = info.triggersStop;
            this.currentCheckpointIndex = index;

            triggeredCheckpointIndices.Add(index);

            // The full trial is logged once at resolution (response or timeout).
            // The response window + waitingForResponse are set inside TriggerVisualTask.
            if (this.wasStanding)
                StartCoroutine(HandleStandingStop());
            else
                StartCoroutine(HandleWalkingPass());
        }
    }

    void Update()
    {
        if (!isGameStarted)
        {
            if (isHeadNode && InputHandler.instance.GetStartAction())
            {
				if (networkingSync == null)
                {
					networkingSync = FindObjectOfType<NetworkingSync>();
				}
                if (networkingSync != null) 
                {
                    networkingSync.BroadcastStartGame();
                }
            }
        }
        else if (waitingForResponse) // Game is running
        {
            if (isHeadNode)
            {
                int response = -1;
                if (InputHandler.instance.GetResponseLeft())
                {
                    response = 0; 
                }
                else if (InputHandler.instance.GetResponseRight())
                {
                    response = 1; 
                }
                
                if (response != -1)
                {
                    waitingForResponse = false;
                    // Stamp the response onset on the logger's hi-res clock the instant
                    // the button is detected, for an explicit response-onset pulse and a
                    // precise reaction time (a1/a9).
                    if (logger != null) logger.MarkResponseOnset();
                    if (responseWindowCoroutine != null) StopCoroutine(responseWindowCoroutine);
                    if (!this.wasStanding) { treadmillController.ResumeTreadmill();}
                    networkingSync.BroadcastResponseReceived(response);
                }
            }
        }
    }

    public void StartGame()
    {
        if (isGameStarted) return; // Prevent double-starts
        isGameStarted = true;
        Debug.Log("--- EXPERIMENT STARTED (baseline) ---");

        // Hold everyone still for the pre-experiment ECG baseline before the block
        // actually begins (treadmill off, no movement).
        if (splineWalker != null) { splineWalker.speed = 0f; splineWalker.isMovementPaused = true; }
        if (isHeadNode && treadmillController != null) treadmillController.StopTreadmill();
        StartCoroutine(BaselineThenBegin());
    }

    // Pre-experiment baseline: show the "starting soon" notice for cooldownSeconds (same
    // length as the end cool-down), participant standing still, then start the block.
    private IEnumerator BaselineThenBegin()
    {
        if (isHeadNode && networkingSync != null)
            networkingSync.BroadcastShowMessage(baselineMessage);

        // Lock the Hybrid projection reference distance to the participant's settled
        // start pose now, while they stand still for the baseline, so the no-wobble
        // lens is calibrated to where they actually are for the whole block. Harmless
        // in OffAxis mode; the operator can still re-press R live.
        if (isHeadNode && networkingSync != null)
            networkingSync.BroadcastRecalibrateHybrid();

        yield return waitCooldown;   // == cooldownSeconds (matched to the end cool-down)

        if (isHeadNode && networkingSync != null)
            networkingSync.BroadcastHideMessage();

        BeginBlock();
    }

    private void BeginBlock()
    {
        Debug.Log("--- BLOCK BEGINS ---");

        // Determine Mode globally
        bool isStandingBlock = (experimentConductor != null && experimentConductor.block == 0);

        if (isStandingBlock)
        {
            // --- STANDING MODE ---
            // 1. Ensure Movement is OFF on ALL nodes
            if (splineWalker != null) 
            {
                splineWalker.speed = 0f; 
                splineWalker.isMovementPaused = true; 
            }

            // 2. Head Node Specifics
            if (isHeadNode)
            {
                Debug.Log("Mode: STANDING. Starting Timer Sequence.");
                if (treadmillController != null) { treadmillController.StopTreadmill(); }
                StartCoroutine(RunStandingBlockSequence());
            }
        }
        else
        {
            // --- WALKING MODE ---
            // 1. Activate Movement on ALL nodes
            if (splineWalker != null) 
            { 
                splineWalker.speed = defaultSpeed; 
                splineWalker.isMovementPaused = false; 
            }
            
            // 2. Head Node Specifics
            if (isHeadNode)
            {
                Debug.Log("Mode: WALKING. Activating SplineWalker.");

                // Enable movement (Collision-Based Logic)
                if (treadmillController != null) { treadmillController.ResumeTreadmill(); }
            }
        }
    }

    private IEnumerator RunStandingBlockSequence()
    {
        // Initial buffer
        yield return new WaitForSeconds(2.0f);

        for (int i = 0; i < checkpointInfos.Count; i++)
        {
            CheckpointInfo info = checkpointInfos[i];

            // Load Data
            this.currentVisualAngle = info.visualAngle;
            this.currentTiltDirection = info.tiltDirection;
            this.isTrainingTrial = info.isTraining;
            this.wasStanding = true; // Always true in standing block
            this.currentCheckpointIndex = i;

            triggeredCheckpointIndices.Add(i);

            // The full trial is logged once at resolution (response or timeout).
            TriggerVisualTask();

            // --- JITTER LOGIC (uniform) ---
            // Each ISI is a SINGLE uniform draw on [targetISI - isiJitter, + isiJitter]
            // (= [3.25, 4.75] s at the defaults; min 3.25 s clears the 2 s response window +
            // stimulus). Uniform flattens the interval distribution, and because successive
            // ISIs are independent (no telescoping anti-correlation) onset times random-walk
            // so stride-phase coverage accumulates. PROVISIONAL — validate via
            // onset_phase_coverage (see the jitter-shape memory).
            float calculatedISI = targetISI + Random.Range(-isiJitter, isiJitter);

            // Safety clamp
            float finalWaitTime = Mathf.Max(0.5f, calculatedISI);

            // Custom Wait Loop
            float t = 0f;
            while (t < finalWaitTime)
            {
                t += Time.deltaTime;
                yield return null;
            }
        }

        StartCooldown();
    }

    public void OnResponseReceived(int response)
    {
        this.lastResponse = response;
        float reactionTime = Time.realtimeSinceStartup - stimulusOnsetTimestamp;
        // Save response in logger if assigned


        if (isHeadNode) // Only Head Node scores & logs
        {
            // Correct response = the side the Gabor leaned toward (expectedResponse).
            bool isCorrect = (response == expectedResponse);

            // Running accuracy (experimental trials only).
            if (!this.isTrainingTrial)
            {
                respTotal++;
                if (isCorrect) respCorrect++;
                hasLastResp = true;
                lastRespCorrect = isCorrect;
                lastRespRT = reactionTime;
                lastRespAngle = currentVisualAngle;
                lastRespDir = currentTiltDirection;
            }

            // Log the complete trial (all trials, with the training flag).
            LogTrialOutcome(response, reactionTime);

            Debug.Log("Checkpoint response: " + response + " (Correct: " + isCorrect + ", RT: " + reactionTime.ToString("F3") + "s)");

            if (visualTaskPlane != null && networkingSync != null)
                networkingSync.BroadcastResetTask();
        }
    }

    // Writes one complete trial row to the logger. response = -1 => no response (timeout).
    private void LogTrialOutcome(int response, float reactionTime)
    {
        if (!isHeadNode || logger == null) return;
        int correct = (response >= 0) ? ((response == expectedResponse) ? 1 : 0) : -1;
        string type = wasStanding ? LOG_TYPE_STANDING : LOG_TYPE_WALKING;
        string difficulty = DifficultyLabel(currentVisualAngle);
        logger.RecordTrial(type, currentCheckpointIndex, isTrainingTrial, currentTiltDirection,
                           currentVisualAngle, difficulty, expectedResponse, response, correct,
                           (response >= 0) ? reactionTime : -1f);
    }
    
    private IEnumerator HandleStandingStop()
    {
        yield return StartCoroutine(SlowDown());

        TriggerVisualTask();

        yield return waitResumeDelay;
        yield return StartCoroutine(SpeedUp());
    }

    private IEnumerator HandleWalkingPass()
    {
        treadmillController.ForceTreadmillSpeed(defaultSpeed);
        
        TriggerVisualTask();
        
        yield return null;
    }

    private void TriggerVisualTask()
    {
        stimulusOnsetTimestamp = Time.realtimeSinceStartup;
        
        int directionToUse = this.currentTiltDirection;
        float angleToUse = this.currentVisualAngle;

        // Correct response = the side the Gabor leans toward.
        this.expectedResponse = directionToUse;

        // Show Visuals
        if (visualTaskPlane != null && networkingSync != null)
        {
            networkingSync.BroadcastShowTask(directionToUse, angleToUse);

            // Hide the checkpoint marker at the SAME instant the stimulus appears.
            // Local call covers the head node; the broadcast covers render nodes.
            SetCheckpointVisible(currentCheckpointIndex, false);
            networkingSync.BroadcastHideCheckpoint(currentCheckpointIndex);
        }

        StartCoroutine(HideTaskAfterDelay(EffectiveStimulusDuration()));
        lastResponse = -1;
        waitingForResponse = true;
        
        if (responseWindowCoroutine != null) StopCoroutine(responseWindowCoroutine);
        responseWindowCoroutine = StartCoroutine(CloseResponseWindow());
    }

    
    private IEnumerator HideTaskAfterDelay(float delayInSeconds)
    {
        // Wait one frame to ensure the task was actually rendered
        yield return null; 
        
        float startTime = Time.realtimeSinceStartup;
        float endTime = startTime + delayInSeconds;

        while (Time.realtimeSinceStartup < endTime)
        {
            yield return null;
        }
        
        float actualDuration = Time.realtimeSinceStartup - startTime;
        // Frame-aware tolerance: allow up to ~2 frames of overshoot so we only warn
        // on genuine stalls, not normal frame quantisation at a 50 ms stimulus.
        float tolerance = Mathf.Max(0.04f, 2f * Time.unscaledDeltaTime);
        if (actualDuration > delayInSeconds + tolerance)
        {
             Debug.LogWarning(string.Format("Stimulus Timing Violation! Target: {0}s, Actual: {1}s (Delta: {2}s)",
                 delayInSeconds, actualDuration, actualDuration - delayInSeconds));
        }
        
        if (isHeadNode && visualTaskPlane != null && networkingSync != null)
        {
            networkingSync.BroadcastResetTask();
        }
    }
    
    private IEnumerator CloseResponseWindow()
    {
        yield return waitResponseWindow;
        
        if (waitingForResponse)
        {
            waitingForResponse = false;
            Debug.Log("Response window timed out - no response for this checkpoint");
            LogTrialOutcome(-1, -1f); // record the missed trial as a complete row
            if (isHeadNode && !this.wasStanding) { treadmillController.ResumeTreadmill(); }
        }
    }
    
    private IEnumerator SlowDown()
    {
        float startSpeed = splineWalker.speed;
        treadmillController.ForceTreadmillSpeed(startSpeed);
        
        float speedStep1 = startSpeed * 0.75f;
        splineWalker.speed = speedStep1;
        treadmillController.manualTargetSpeed = speedStep1;
        yield return waitRampDuration; 

        float speedStep2 = startSpeed * 0.25f;
        splineWalker.speed = speedStep2;
        treadmillController.manualTargetSpeed = speedStep2;
        yield return waitRampDuration;

        splineWalker.speed = 0f;
        treadmillController.manualTargetSpeed = 0f;
        
    }

    private IEnumerator SpeedUp()
    {
        yield return waitResumeDelay;
        float targetSpeed = defaultSpeed;
                
        float speedStep1 = targetSpeed * 0.25f;
        splineWalker.speed = speedStep1;
        treadmillController.manualTargetSpeed = speedStep1;
        yield return waitRampDuration;
        
        float speedStep2 = targetSpeed * 0.75f;
        splineWalker.speed = speedStep2;
        treadmillController.manualTargetSpeed = speedStep2;
        yield return waitRampDuration;

        splineWalker.speed = targetSpeed;
        treadmillController.manualTargetSpeed = targetSpeed;
        yield return waitRampDuration; 
        
        treadmillController.ResumeTreadmill();
    }

    private bool treadmillWarnShown = false;

    // Called by SplineWalker shortly before the end of the path (walking blocks) so the
    // participant is warned the treadmill is about to stop. Fires once.
    public void WarnTreadmillStopping()
    {
        if (!isHeadNode || treadmillWarnShown) return;
        treadmillWarnShown = true;
        if (networkingSync != null) networkingSync.BroadcastShowMessage(treadmillWarnMessage);
    }

    public void StartCooldown()
    {
        if (!isHeadNode) return;

        StartCoroutine(SlowDown());
        StartCoroutine(CooldownCoroutine());
    }

    private IEnumerator CooldownCoroutine()
    {
        // Treadmill is ramping to a stop (SlowDown); replace any "stopping soon" warning
        // with the cool-down notice and hold it for the (ECG-baseline-matched) duration.
        if (networkingSync != null) networkingSync.BroadcastShowMessage(cooldownMessage);

        yield return waitCooldown;

        if (networkingSync != null)
        {
            networkingSync.BroadcastHideMessage();
            networkingSync.BroadcastReturnToScene(0);
        }
    }
}
