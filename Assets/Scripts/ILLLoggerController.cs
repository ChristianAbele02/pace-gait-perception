using System;
using System.IO;
using System.IO.Compression;
using System.Collections.Generic;
using UnityEngine;
using Optitrack;


/// <summary>
/// Writes the per-frame recording that the whole analysis pipeline is built on: one CSV
/// row per rendered frame, gzipped at session end.
///
/// Columns are registered as name + value-getter pairs (see <c>AddColumn</c>), so the
/// header and the rows cannot drift apart. What is logged is chosen so the analysis never
/// has to infer timing it could have measured:
///   * full 6-DOF pose STRAIGHT FROM THE RIGID BODY (not the Unity head Transform, which
///     may be a yaw-only container), as quaternion AND Euler to avoid gimbal ambiguity;
///   * <c>mocap-frame-advanced</c>, marking the first Unity row to sample each new mocap
///     frame, so the effective tracking rate and dropped frames are measurable instead of
///     being inferred from the jittery render clock;
///   * explicit <c>stimulus-onset</c> / <c>response-onset</c> pulses and a hi-res
///     <c>reaction-time-precise</c> measured against RENDERED onset;
///   * the counterbalance (visit, slot, order) and the session start time, so condition
///     order is recorded DATA rather than something the analysis re-derives;
///   * <c>checkpoint-dropped</c>, so a trial suppressed by an overlapping response window
///     is counted rather than vanishing.
///
/// Logging is disabled in debug mode — a debug-mode session produces no data at all.
/// </summary>
public class ILLLoggerController : MonoBehaviour
{
    [Header("OptiTrack Setup")]
    public string skeletonName = "Skeleton1";
    public OptitrackStreamingClient optitrack;
    private int skeletonId;

    [Tooltip("Motive 'User ID' (Streaming ID) of the HEAD rigid body. Used to log the "
        + "full 6-DOF mocap pose (position + orientation quaternion) straight from the "
        + "streaming client, independent of the Unity head Transform. This is the robust "
        + "fix for the 'only yaw is non-zero' orientation bug: the Unity Transform may be "
        + "a yaw-only container, but the rigid body carries true pitch/roll/yaw.")]
    public int headRigidBodyId = 1;

    public Transform trackingOrigin; // Assign 'ILL-UniCave' or parent of CameraHolder here

    [Header("Object References (Assign in Inspector)")]
    public CheckpointManager CheckpointManager;
    public GameObject ill; // Your main player/camera object
    public Transform head;
    public Terrain terrain;

    [Header("Lateral sway monitor (tracking-loss guard / data cleaning)")]
    [Tooltip("Absolute lateral head offset (m, local X / sway axis) beyond which the "
        + "participant is WARNED to re-center. Logged as 'lateral-sway-flag'=1 and shown "
        + "as an amber alarm in the FPS heads-up.")]
    public float lateralSwayWarnMeters = 0.35f;
    [Tooltip("Larger lateral offset (m) flagged as CRITICAL (lateral-sway-flag=2, red "
        + "alarm): the participant is far enough off-center that the OptiTrack is likely "
        + "to lose the head and produce the position spikes seen in QC. Use the flag to "
        + "mask those frames offline.")]
    public float lateralSwayCriticalMeters = 0.50f;

    // Lateral-sway level derived each row from the local head X (sway axis):
    // 0 = OK, 1 = WARN (re-center), 2 = CRITICAL (tracking-dropout risk). Exposed for
    // the FPS heads-up alarm and written to the 'lateral-sway-flag' column for cleaning.
    public int LateralSwayLevel { get; private set; }

    // --- Private logging variables ---
    private delegate string getValue();
    private List<string> columnNames;
    private List<getValue> columnValueGetters;
    private StreamWriter outputStream;
    private float lastFlush;
    private string tempFileName;

    // Wall-clock start of this recording, stamped once when the columns are set up and
    // then written unchanged on every row. Feeds BIDS scans.tsv `acq_time`, which the
    // analysis uses for time-on-task within a session and for the interval between a
    // participant's two visits. Sortable round-trip format, local time.
    private string sessionStartIso = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss");

    // --- Checkpoint Event Data ---
    // These are "consumed" (reset) every frame after logging
    private int lastCheckpointResponse = -1;
    private string lastCheckpointType = "";
    public float lastReactionTime = -1f;
    private int lastVisualTilt = -1;
    public int lastResponseCorrect = -1;
    private float lastVisualAngle = 0f;
    private int lastCheckpointIndex = -1;
    private int lastExpectedResponse = -1;
    private string lastDifficulty = "";
    private int lastIsTraining = -1;

    // --- Precise timing & event-onset markers (a1 / a9) -------------------------
    // A high-resolution MONOTONIC clock. Time.realtimeSinceStartup is a float that
    // loses sub-millisecond resolution after a few minutes and is only updated once
    // per frame; the Stopwatch gives a stable, drift-free reference shared by the
    // stimulus-onset and response-onset markers so reaction times are measured on a
    // single clock. Started in OnEnable.
    private System.Diagnostics.Stopwatch hiResClock;

    // stimulus-onset / response-onset are logged as single-frame PULSES (1 on the
    // exact frame the event happens, 0 otherwise) so the analysis pipeline can read
    // onsets straight from the motion stream instead of reconstructing them from
    // (trial-resolution - reaction-time). Onset is defined as the frame the Gabor is
    // actually RENDERED (task-visible rising edge), not the frame it was requested.
    private int stimulusOnsetPulse = 0;
    private int responseOnsetPulse = 0;
    private double stimulusOnsetHires = -1.0;  // hi-res time of the current stimulus onset
    private double responseOnsetHires = -1.0;  // hi-res time of the most recent response
    private double preciseReactionTime = -1.0; // responseOnsetHires - stimulusOnsetHires
    private bool lastTaskVisible = false;       // for rising-edge detection
    private int droppedCheckpointIndex = -1;    // overlap-dropped checkpoint, one frame (a5)

    // --- Native mocap sampling QC (a2) -----------------------------------------
    // The logger writes one row per Unity frame (~60 Hz). OptiTrack streams faster
    // (120-240 Hz) and updates latestFrameData on the NatNet thread. We log the
    // native frame number/timestamp and a "frame-advanced" flag (1 if the mocap
    // frame number changed since the last row) so the pipeline can (a) resample on
    // the true mocap clock, (b) drop duplicate-sampled rows, and (c) compute the
    // effective mocap rate and dropped-frame count. True native-rate capture would
    // require a dedicated sampling thread reading GetLatestRigidBodyState() (the
    // mocap RigidBody, unlike a Unity Transform, is safe to read off the main
    // thread) -- documented as a follow-up; this additive change already removes the
    // orientation bug and gives the pipeline the clock data it needs.
    private int lastTrackingFrame = -1;

    // --- Cached component refs (avoid GetComponent in the per-frame logging path) ---
    private TreadmillController treadmillController;
    private SplineWalker splineWalker;

    // --- Cached head-derived pose, recomputed once per row. The local position and the
    //     local rotation each need a transform inverse that was previously evaluated 3x
    //     (once per component getter); doing it once per row removes that redundant work. ---
    private Vector3 headGlobalPos;
    private Vector3 headGlobalEuler;
    private Vector3 headLocalPos;
    private Vector3 headLocalEuler;

    // True only once the header has been written and per-frame logging is really
    // running (head node, non-debug, all refs valid). OnDisable uses this to decide
    // between SAVING the log and DISCARDING an empty temp file — previously a
    // debug-mode run (or a missing-reference abort) still published an empty
    // .csv.gz that claimed the participant's canonical file name, pushing the
    // real run's log to a "-1" suffix.
    private bool loggingActive = false;

    private void Awake()
    {
        if (Environment.MachineName != MasterTrackingData.HeadNodeMachineName)
        {
            Debug.Log("ILLLoggerController: Not on Head Node. Destroying self.");
            // Disable BEFORE the deferred Destroy so OnEnable/Start never run this
            // frame — otherwise every render node opened a temp stream and wrote a
            // junk .csv.gz to its own disk on teardown.
            enabled = false;
            Destroy(gameObject);
        }
    }

    void OnEnable()
    {
        columnNames = new List<string>();
        columnValueGetters = new List<getValue>();
        lastFlush = Time.realtimeSinceStartup;

        // Monotonic high-resolution clock for onset / reaction-time measurement (a9).
        hiResClock = System.Diagnostics.Stopwatch.StartNew();

        // --- open output file
        tempFileName = Path.GetTempFileName();
        Debug.Log("Logging to: " + tempFileName);
        outputStream = File.CreateText(tempFileName);
    }

    void Start()
    {
        if (optitrack == null)
        {
            optitrack = FindObjectOfType<OptitrackStreamingClient>();
        }
        if (ExperimentConductor.instance != null && ExperimentConductor.instance.isDebugMode)
        {
            Debug.Log("ILLLoggerController: Debug Mode enabled. Logging is DISABLED.");
            this.enabled = false;
            return;
        }

        // Validate all critical references that MUST be set in the Inspector
        if (ill == null || terrain == null || CheckpointManager == null || head == null)
        {
            Debug.LogError("ILLLoggerController: A critical reference is missing! Please assign all fields in the Inspector. Disabling logger.");
            this.enabled = false; // Disable the script
            return;
        }

        // Cache the components the per-frame logger reads, so the hot path never calls
        // GetComponent (which was run every row otherwise).
        treadmillController = ill.GetComponent<TreadmillController>();
        splineWalker = ill.GetComponent<SplineWalker>();

        assembleOutputs();
        loggingActive = true; // header written; OnDisable may now SAVE (not discard)
    }

    // Record one COMPLETE trial outcome — called once per trial when the response
    // arrives or the response window times out. Writing everything in one place
    // guarantees a single, self-consistent row per trial.
    //   response = -1  -> no response (timeout)
    //   correct  = 1/0 -> correct/incorrect, or -1 if no response
    public void RecordTrial(string type, int checkpointIndex, bool isTraining, int tiltDirection,
                            float angleDeg, string difficulty, int expectedResponse, int response,
                            int correct, float reactionTime)
    {
        lastCheckpointType = type;
        lastCheckpointIndex = checkpointIndex;
        lastIsTraining = isTraining ? 1 : 0;
        lastVisualTilt = tiltDirection;
        lastVisualAngle = angleDeg;
        lastDifficulty = difficulty;
        lastExpectedResponse = expectedResponse;
        lastCheckpointResponse = response;
        lastResponseCorrect = correct;
        lastReactionTime = reactionTime;
    }

    // Called by CheckpointManager the instant a response button is detected, so the
    // motion stream carries an explicit response-onset pulse and a precise reaction
    // time measured on the shared hi-res clock (a1 / a9). RT is taken relative to the
    // RENDERED stimulus onset (task-visible rising edge), removing the request-to-
    // render latency that inflated the old realtimeSinceStartup-based RT.
    public void MarkResponseOnset()
    {
        if (hiResClock == null) return;
        responseOnsetHires = hiResClock.Elapsed.TotalSeconds;
        responseOnsetPulse = 1;
        preciseReactionTime = (stimulusOnsetHires >= 0.0)
            ? (responseOnsetHires - stimulusOnsetHires)
            : -1.0;
    }

    // Called when a checkpoint is reached while a previous response window is still
    // open, so the overlapping stimulus is COUNTED (in the motion stream) rather than
    // silently swallowed (a5). Records the dropped checkpoint index for one frame.
    public void MarkDroppedCheckpoint(int checkpointIndex)
    {
        droppedCheckpointIndex = checkpointIndex;
    }

    // Reads the head rigid body straight from the streaming client (full 6-DOF mocap
    // pose). Cached once per row so the 11 pose columns don't each re-query/relock.
    private Vector3 mocapPos = Vector3.zero;
    private Quaternion mocapRot = Quaternion.identity;
    private bool mocapValid = false;
    private int mocapFrame = -1;
    private double mocapTimestamp = 0.0;
    private int trackingFrameAdvanced = 0;

    private void SampleMocap()
    {
        mocapValid = false;
        if (optitrack == null) return;
        try
        {
            FrameData.RigidBody rb = optitrack.GetLatestRigidBodyState(headRigidBodyId);
            mocapPos = rb.position;
            mocapRot = rb.orientation;
            mocapValid = rb.trackingValid;
        }
        catch (Exception)
        {
            // GetLatestRigidBodyState throws if the id isn't in the latest frame
            // (e.g. brief dropout). Leave mocapValid = false for this row.
            mocapValid = false;
        }
        mocapFrame = optitrack.GetLatestFrameNumber();
        mocapTimestamp = optitrack.GetLatestTimestamp();
        trackingFrameAdvanced = (mocapFrame != lastTrackingFrame) ? 1 : 0;
        lastTrackingFrame = mocapFrame;
    }

    // Cache the head-derived pose once per row. The local position/rotation each require a
    // transform inverse that was previously evaluated 3x (once per component getter); doing
    // it once here removes that redundant per-frame work. Values are identical to before.
    private void SampleHeadDerived()
    {
        if (head == null) return;
        headGlobalPos = head.position;
        headGlobalEuler = head.eulerAngles;
        if (trackingOrigin != null)
        {
            headLocalPos = trackingOrigin.InverseTransformPoint(head.position);
            headLocalEuler = (Quaternion.Inverse(trackingOrigin.rotation) * head.rotation).eulerAngles;
        }
        else
        {
            // Matches the old "0" fallback when no trackingOrigin is assigned.
            headLocalPos = Vector3.zero;
            headLocalEuler = Vector3.zero;
        }

        // Lateral-sway level from the local X (sway axis). Computed here so the logged
        // 'lateral-sway-flag' column and the FPS heads-up alarm share the SAME thresholds.
        float lat = Mathf.Abs(headLocalPos.x);
        LateralSwayLevel = (lat >= lateralSwayCriticalMeters) ? 2
                         : (lat >= lateralSwayWarnMeters) ? 1 : 0;
    }

    void assembleOutputs()
    {
        // --- assemble output
        AddColumn("timestamp", () => Time.realtimeSinceStartup.ToString());
        // Monotonic hi-res clock (seconds) -- the reference clock for onsets / RTs (a9).
        AddColumn("clock-hires", () => (hiResClock != null) ? hiResClock.Elapsed.TotalSeconds.ToString("F6") : "0");

        // Global Head Position (cached once per row in SampleHeadDerived).
        AddColumn("head-global-x", () => headGlobalPos.x.ToString("F4"));
        AddColumn("head-global-y", () => headGlobalPos.y.ToString("F4"));
        AddColumn("head-global-z", () => headGlobalPos.z.ToString("F4"));

        // Local Head Position (relative to trackingOrigin; cached once per row).
        AddColumn("head-local-x", () => headLocalPos.x.ToString("F4"));
        AddColumn("head-local-y", () => headLocalPos.y.ToString("F4"));
        AddColumn("head-local-z", () => headLocalPos.z.ToString("F4"));

        // Global Head Rotation (Euler Angles; cached once per row).
        AddColumn("head-global-rot-x", () => headGlobalEuler.x.ToString("F4"));
        AddColumn("head-global-rot-y", () => headGlobalEuler.y.ToString("F4"));
        AddColumn("head-global-rot-z", () => headGlobalEuler.z.ToString("F4"));

        // Local Head Rotation (Euler Angles relative to trackingOrigin; cached once per row).
        AddColumn("head-local-rot-x", () => headLocalEuler.x.ToString("F4"));
        AddColumn("head-local-rot-y", () => headLocalEuler.y.ToString("F4"));
        AddColumn("head-local-rot-z", () => headLocalEuler.z.ToString("F4"));

        // Lateral-sway guard flag (0 OK / 1 WARN / 2 CRITICAL) from the local head X.
        // Logged so the analysis can mask frames where the participant swayed far enough
        // off-center to risk an OptiTrack dropout (the source of head-position spikes).
        AddColumn("lateral-sway-flag", () => LateralSwayLevel.ToString());

        // --- Full 6-DOF mocap pose, straight from the streaming client (a2 / a3) ---
        // Logged from the rigid body, NOT the Unity head Transform, so true pitch/roll/
        // yaw are captured even if the Transform is a yaw-only container. Quaternion is
        // logged alongside Euler to avoid gimbal/wrap ambiguity in downstream analysis.
        AddColumn("head-mocap-x", () => mocapPos.x.ToString("F4"));
        AddColumn("head-mocap-y", () => mocapPos.y.ToString("F4"));
        AddColumn("head-mocap-z", () => mocapPos.z.ToString("F4"));
        AddColumn("head-mocap-qx", () => mocapRot.x.ToString("F5"));
        AddColumn("head-mocap-qy", () => mocapRot.y.ToString("F5"));
        AddColumn("head-mocap-qz", () => mocapRot.z.ToString("F5"));
        AddColumn("head-mocap-qw", () => mocapRot.w.ToString("F5"));
        AddColumn("head-mocap-rot-x", () => mocapRot.eulerAngles.x.ToString("F4"));
        AddColumn("head-mocap-rot-y", () => mocapRot.eulerAngles.y.ToString("F4"));
        AddColumn("head-mocap-rot-z", () => mocapRot.eulerAngles.z.ToString("F4"));
        AddColumn("head-mocap-valid", () => mocapValid ? "1" : "0");
        // 1 only on the first Unity row that sampled a NEW mocap frame (a2). Lets the
        // pipeline drop duplicate-sampled rows, measure the effective mocap rate and
        // count dropped frames instead of trusting the jittery render clock.
        AddColumn("mocap-frame-advanced", () => trackingFrameAdvanced.ToString());

        // --- Event onset pulses (a1) ---
        // 1 on the exact frame the Gabor is rendered / a response is pressed, else 0.
        AddColumn("stimulus-onset", () => stimulusOnsetPulse.ToString());
        AddColumn("response-onset", () => responseOnsetPulse.ToString());
        // Precise reaction time (s) on the shared hi-res clock, relative to RENDERED
        // onset. -1 until a response resolves. Complements checkpoint-reaction-time.
        AddColumn("reaction-time-precise", () => preciseReactionTime.ToString("F6"));
        // Index of a checkpoint dropped because it overlapped an open response window
        // (a5); -1 otherwise. Lets QC count overlaps instead of them vanishing.
        AddColumn("checkpoint-dropped", () => droppedCheckpointIndex.ToString());

        // 1 = Task Visible, 0 = Hidden. Use this to slice your data!
        AddColumn("task-visible", () => (FollowerPlane.instance != null && FollowerPlane.instance.isTaskVisible) ? "1" : "0");

        // treadmill speed (cached reference; no per-frame GetComponent)
        AddColumn("treadmill-speed", () => (treadmillController != null) ? treadmillController.speed.ToString() : "0");

        // path progress (cached reference; no per-frame GetComponent)
        AddColumn("distance-on-path", () => (splineWalker != null) ? splineWalker.progress.ToString() : "0");
    
        // Native mocap frame number / timestamp (cached this row by SampleMocap, so they
        // line up exactly with mocap-frame-advanced and the head-mocap-* pose columns).
        AddColumn("tracking-framenumber", () => mocapFrame.ToString());
        AddColumn("tracking-timestamp", () => mocapTimestamp.ToString("F6"));
        
        AddColumn("total-checkpoints", () => CheckpointManager.GetTotalCheckpointCount().ToString());
        AddColumn("checkpoints-triggered", () => CheckpointManager.GetTriggeredCheckpointCount().ToString());
        
        AddColumn("participant-id", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.participant.ToString() : "-1");
        AddColumn("block-id", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.block.ToString() : "-1");
        AddColumn("trial-id", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.trial.ToString() : "-1");
        // --- Counterbalance (recorded so condition ORDER is data, not something the
        // analysis has to re-derive from the participant number; needed to model the
        // order x condition interaction in threshold_by_condition.py) ---
        AddColumn("visit", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.visit.ToString() : "-1");
        AddColumn("counterbalance-slot", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.CounterbalanceSlot.ToString() : "-1");
        AddColumn("condition-order-first", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.ConditionOrderFirst : "n/a");
        // Wall-clock start of this recording (ISO-8601, local). Populates BIDS
        // scans.tsv acq_time -> lets the analysis compute time-on-task within a session
        // and the inter-session interval between the two visits.
        AddColumn("session-start-iso", () => sessionStartIso);
        // --- Trial-event columns (populated on the frame a trial resolves) ---
        AddColumn("checkpoint-type", () => lastCheckpointType);              // standing / walking
        AddColumn("checkpoint-index", () => lastCheckpointIndex.ToString());
        AddColumn("is-training", () => lastIsTraining.ToString());           // 1 = training, 0 = experimental, -1 = no trial this frame
        AddColumn("difficulty-level", () => lastDifficulty);                 // THR / EASY / HARD
        AddColumn("visual-tilt", () => lastVisualTilt.ToString());           // 0 = left, 1 = right
        AddColumn("visual-angle", () => lastVisualAngle.ToString("F2"));     // tilt magnitude (deg)
        AddColumn("expected-response", () => lastExpectedResponse.ToString()); // correct button (0/1)
        AddColumn("checkpoint-response", () => lastCheckpointResponse.ToString()); // pressed button (0/1, -1 = none)
        AddColumn("response-correct", () => lastResponseCorrect.ToString());  // 1/0, -1 = no response
        AddColumn("checkpoint-reaction-time", () => lastReactionTime.ToString());
        // --- Calibration / difficulty reference ---
        AddColumn("threshold-angle", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.thresholdAngle.ToString("F4") : "0");
        AddColumn("easy-angle", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.easyAngle.ToString("F4") : "0");
        AddColumn("hard-angle", () => (ExperimentConductor.instance != null) ? ExperimentConductor.instance.hardAngle.ToString("F4") : "0");
        
        // --- write header
        foreach (var name in columnNames) {
            outputStream.Write(name);
            outputStream.Write(';');
        }
        outputStream.Write('\n');
    }

    void Update ()
    {
        // --- flush every while (shorter interval: data rate is tiny, so a crash now
        //     loses at most a few seconds instead of up to 30 s).
        var now = Time.realtimeSinceStartup;
        if (now - lastFlush > 5) {
            outputStream.Flush();
            lastFlush = now;
        }

        // --- sample the mocap rigid body for this row (a2/a3).
        SampleMocap();
        // --- cache the head-derived pose (global/local pos+rot) once for this row (perf).
        SampleHeadDerived();

        // --- stimulus-onset pulse (a1): fire on the RENDERED rising edge of the task,
        //     i.e. the first frame the Gabor is actually visible. This is the onset the
        //     pipeline bins on, and the reference for the precise reaction time (a9).
        bool taskVisible = (FollowerPlane.instance != null && FollowerPlane.instance.isTaskVisible);
        if (taskVisible && !lastTaskVisible)
        {
            stimulusOnsetPulse = 1;
            stimulusOnsetHires = (hiResClock != null) ? hiResClock.Elapsed.TotalSeconds : -1.0;
        }
        lastTaskVisible = taskVisible;

        // --- write line
        foreach (var getValue in columnValueGetters) {
            outputStream.Write(getValue());
            outputStream.Write(';');
        }
        outputStream.Write('\n');

        // Reset event-based data so it's only logged for this one frame.
        ResetEventData();
        // Onset pulses are single-frame markers too.
        stimulusOnsetPulse = 0;
        responseOnsetPulse = 0;
        preciseReactionTime = -1.0;  // only non -1 on the row a response resolves
        droppedCheckpointIndex = -1; // single-frame marker
    }

    private void ResetEventData()
    {
        lastCheckpointResponse = -1;
        lastCheckpointType = "";
        lastReactionTime = -1f;
        lastVisualTilt = -1;
        lastResponseCorrect = -1;
        lastCheckpointIndex = -1;
        lastExpectedResponse = -1;
        lastDifficulty = "";
        lastIsTraining = -1;
    }

    void OnDisable()
    {
        // Check if stream was ever created (in case Start() failed)
        if (outputStream == null) return;

        // Logging never actually started (debug mode / missing references): discard
        // the empty temp file instead of publishing a junk .csv.gz that would claim
        // the participant's canonical file name (pushing the real run to "-1").
        if (!loggingActive)
        {
            try { outputStream.Close(); } catch (Exception) {}
            outputStream = null;
            try { if (File.Exists(tempFileName)) File.Delete(tempFileName); } catch (Exception) {}
            Debug.Log("ILLLoggerController: logging never started — discarded empty temp file (no data file written).");
            return;
        }

        outputStream.Flush();
        outputStream.Close();

        // move temporary file to final position
        var trialNumber = 0;

        // Resolve the output folder; fall back to persistentDataPath (mirroring
        // StaircaseManager.WriteCalibrationCsv) so a misconfigured/empty
        // dataStoragePath can never strand the recorded data in the temp folder.
        string dir = (ExperimentConductor.instance != null && !string.IsNullOrEmpty(ExperimentConductor.instance.dataStoragePath))
            ? ExperimentConductor.instance.dataStoragePath
            : Application.persistentDataPath;

        // Make sure the directory exists before trying to move the file
        try
        {
            Directory.CreateDirectory(dir);
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to create data storage directory: " + e.Message);
            Debug.LogWarning("Uncompressed log file is still at: " + tempFileName);
            return;
        }


        // Tag the raw log with the locomotion condition (block 0 = standing, else
        // walking) so walking/standing runs are distinguishable straight off the rig;
        // the BIDS converter turns this into the acq-<condition> entity. Calibration is
        // written separately by StaircaseManager and is unaffected.
        string condition = (ExperimentConductor.instance != null && ExperimentConductor.instance.block != 0)
            ? "walking" : "standing";
        // File name carries participant, visit, block and run (p05-v2-b1-t03); the
        // seed identifier without the visit is left to CheckpointManager.
        string id = (ExperimentConductor.instance != null)
            ? ExperimentConductor.instance.GetFileIdentifier() : "log";
        string baseName = Path.Combine(dir, id + "_" + condition);

        // The session is recorded as a plain-text temp file (durable: a crash
        // mid-run loses at most the last flush, and the rest stays readable).
        // We gzip it once here, at the end, into the final .csv.gz - small
        // artifact, ~1 s, with no loss of in-run crash resilience.
        string outputPath = baseName + ".csv.gz";
        while (File.Exists(outputPath))
        {
            trialNumber++;
            outputPath = baseName + "-" + trialNumber + ".csv.gz";
        }

        Debug.Log("Saving as \"" + outputPath + "\"");
        try
        {
            using (var input = File.OpenRead(tempFileName))
            using (var outFs = File.Create(outputPath))
            using (var gz = new GZipStream(outFs, CompressionMode.Compress))
            {
                // Manual copy loop: Stream.CopyTo is not in the .NET 3.5 profile.
                byte[] buffer = new byte[1 << 16];
                int read;
                while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                    gz.Write(buffer, 0, read);
            }
            File.Delete(tempFileName);
        }
        catch (Exception e)
        {
            Debug.LogError("Failed to gzip log file: " + e.Message);
            Debug.LogWarning("Uncompressed log file is still at: " + tempFileName);
        }
    }

    void AddColumn(string name, getValue getter) {
        columnNames.Add(name);
        columnValueGetters.Add(getter);
    }
    
    //Helper for FPSCounter to get live local position ---
    public Vector3 CurrentLocalHeadPosition
    {
        get
        {
             if (head == null) return Vector3.zero;
             if (trackingOrigin != null) return trackingOrigin.InverseTransformPoint(head.position);
             return head.position; // Fallback if no origin
        }
    }
}