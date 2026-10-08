using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI; 
using UnityEngine.SceneManagement; 

/// <summary>
/// Session-level controller and the operator's setup screen: owns the identifiers that
/// name every output file, resolves the difficulty levels a block will use, and decides
/// which condition runs first.
///
/// Holds participant / visit / block / trial, where <c>block</c> selects the CONDITION
/// (0 = standing, 1 = walking) and <c>trial</c> is the RUN INDEX that keeps repeated
/// blocks of one condition from overwriting each other (it becomes the BIDS <c>run-</c>
/// entity downstream). Reusing a (block, trial) pair silently destroys a block, which is
/// why the operator guide fixes the numbering.
///
/// Condition order comes from a pre-generated permuted-block allocation indexed by
/// <c>counterbalanceSlot</c>, and CROSSES OVER on visit 2 so each participant meets each
/// condition once early and once late in a session — without that, time on task is
/// confounded with condition in the same direction in both visits and survives pooling.
/// The slot is separate from the participant number so an excluded participant's slot can
/// be re-filled while IDs stay unique.
///
/// Thresholds are per condition, measured by the QUEST+ calibration and stored here; the
/// easy and hard levels are derived from the threshold rather than set by hand.
/// </summary>
public class ExperimentConductor : MonoBehaviour
{
    [Header("Debug Mode")]
    public bool isDebugMode;
    public Toggle debugModeToggle;

    [Header("Experiment Data")]
    public int participant;
    public int block;
    public int trial;
    public float experimentSpeed = 1.0f;
    public string dataStoragePath;

    [Header("Counterbalance (condition order)")]
    [Tooltip("Which of the participant's two Option-A visits this is: 1 or 2. The "
        + "condition order CROSSES OVER between visits (visit 2 runs the opposite order "
        + "to visit 1), so within every participant each condition occupies the early "
        + "and the late half of a session exactly once. Without this, time-on-task "
        + "(fatigue/adaptation) is confounded with condition in the SAME direction in "
        + "both visits and the confound survives pooling.")]
    public int visit = 1;
    [Tooltip("Counterbalance slot into the pre-generated allocation sequence below. "
        + "Leave 0 to use the participant number. Set it explicitly to re-fill a slot "
        + "vacated by an excluded participant, which is what keeps the design balanced "
        + "under dropout (participant IDs stay unique; the SLOT is what is reused).")]
    public int counterbalanceSlot = 0;

    // ACTIVE difficulty set — copied from the per-condition set for the current block
    // (SetActiveCondition). The logger and CheckpointManager read these, so the logged
    // threshold-angle and the presented tilts always match the block's own calibration.
    [Header("Calibration Results (active set)")]
    public float thresholdAngle = 15.0f; // The calculated threshold
    public float easyAngle = 20.0f;      // Easier version
    public float hardAngle = 10.0f;      // Harder version

    // Per-condition calibration results. A STANDING and a WALKING calibration each fill
    // their own set; the active set above is switched to the block's condition at run
    // start. A walking tilt threshold is typically larger (harder) than standing, so
    // using the block's own threshold keeps the near-threshold trials — where the
    // stride-phase oscillation lives — actually at threshold in BOTH conditions.
    [Header("Per-condition Calibration Results")]
    public float thresholdStanding = 15.0f, easyStanding = 20.0f, hardStanding = 10.0f;
    public float thresholdWalking  = 15.0f, easyWalking  = 20.0f, hardWalking  = 10.0f;
    [Tooltip("Which condition the NEXT calibration run (scene 2) calibrates. Set by the "
        + "guided runner / operator before launching calibration. 'standing' or 'walking'.")]
    public string calibrationCondition = "standing";

    // Manual Threshold Override:
    // If the Threshold field holds a valid positive number at Start, it is used
    // directly so you can restart a participant with a saved threshold WITHOUT
    // re-running calibration. Easy/Hard are derived via the factors below, which
    // are the model-implied 85%/65% / 75% tilt ratios for the QUEST+ parameters
    // (gamma 0.5, beta 2.5, lambda 0.02) — keep in sync with StaircaseManager.
    [Header("Manual Threshold Override")]
    [Tooltip("easy(85%)/threshold(75%) tilt ratio for the QUEST+ model. Keep in sync with StaircaseManager.")]
    public float easyFactor = 1.26f;
    [Tooltip("hard(65%)/threshold(75%) tilt ratio for the same QUEST+ model.")]
    public float hardFactor = 0.76f;

    [Header("Scene References")]
    public string mainSceneName = "01_MainExperiment";
    public string calibrationSceneName = "02_Calibration";

    [Header("UI (Assign in Inspector)")]
    public GameObject setupUIPanel; 
    public GameObject waitingPanel;
    public InputField participantField;
    public InputField blockField;
    public InputField trialField;
    public InputField speedField;
    public Button startButton; //start experiment
    public InputField thresholdField;
    public Button calibrateButton;    // start calibration

    [Header("Per-condition Calibration UI (Assign in Inspector)")]
    [Tooltip("Wired in CODE (not via the button's OnClick) so they keep working after the "
        + "setup scene reloads between conditions. Just drag the buttons into these slots.")]
    public Button calibrateStandingButton; // -> StartCalibrationStanding()
    public Button calibrateWalkingButton;  // -> StartCalibrationWalking()
    [Tooltip("Read-only operator hint: which condition to calibrate/run FIRST for the "
        + "participant / visit / slot currently in the fields (permuted-block allocation, "
        + "crossed over on visit 2).")]
    public Text calibrationOrderLabel;
    [Tooltip("Optional. Visit number for this session (1 or 2) — drives the crossover. "
        + "Leave unassigned to use the `visit` field value directly.")]
    public InputField visitField;
    [Tooltip("Optional. Counterbalance slot; blank/0 uses the participant number. Set it "
        + "to re-fill the slot of an excluded participant.")]
    public InputField counterbalanceSlotField;
    [Tooltip("Optional. Pre-filled with the stored per-condition thresholds on the setup "
        + "screen; a valid value present at experiment start overrides that condition's "
        + "threshold. Leave unassigned to ignore.")]
    public InputField thresholdStandingField;
    public InputField thresholdWalkingField;

    public static ExperimentConductor instance;
    private NetworkingSync networkingSync;

    public string GetExperimentIdentifier()
    {
        // Example: "p05-b1-t03"
        // This string SEEDS the checkpoint layout (CheckpointManager.mapSeed) on every
        // node, and the render nodes only ever receive participant / block / trial. Do not
        // add the visit here: the nodes would derive different seeds and diverge. File
        // names use GetFileIdentifier() instead.
        return string.Format("p{0:00}-b{1}-t{2:00}", participant, block, trial);
    }

    // Name stem for the files the head node writes (block log and calibration CSV), e.g.
    // "p05-v2-b1-t03": the seed identifier above plus the visit, so every file names its
    // participant, visit, block and run on its own.
    public string GetFileIdentifier()
    {
        return FileIdentifier(participant, visit, block, trial);
    }

    public static string FileIdentifier(int participantId, int visitNo, int blockId, int trialId)
    {
        return string.Format("p{0:00}-v{1}-b{2}-t{3:00}", participantId, visitNo, blockId, trialId);
    }

    // ---------------------------------------------------------------------------
    // Counterbalance
    //
    // Replaces the previous `participant % 2` rule, which had three problems: the
    // allocation was predictable before enrolment (no concealment), it broke silently
    // whenever a participant was excluded or an ID skipped (you can land 11 vs 7), and
    // it was a function of the same number that seeds the checkpoint layout.
    //
    // ALLOCATION is a pre-generated PERMUTED-BLOCK sequence: 16 blocks of 4, each block
    // holding two 'S' (standing first) and two 'W' (walking first) in random order.
    // That keeps strict alternation's best property -- the S/W split can never drift
    // more than 2 apart at ANY point in the collection period, so condition order stays
    // balanced against seasonal/experimenter/hardware drift -- while removing the
    // predictability. Generated once with numpy default_rng(20260818); regenerate with:
    //
    //   rng = np.random.default_rng(20260818)
    //   "".join("".join(rng.permutation(list("SSWW"))) for _ in range(16))
    //
    // Slots are assigned at ENROLMENT, not by ID: if a participant drops out, give the
    // replacement that participant's `counterbalanceSlot` (their participant number
    // stays unique). Both the slot and the resulting order are written to the log.
    const string ALLOCATION = "WSSWSWWSSSWWWSSWWWSSSWSWWSSWSWSWSWWSWSWSWSSWSWWSWSSWWWSSWSWSSWSW";

    // Slot actually in force (0 / unset -> the participant number).
    public int CounterbalanceSlot { get { return counterbalanceSlot > 0 ? counterbalanceSlot : participant; } }

    // The participant's BASE order (visit 1), from the allocation sequence.
    public bool BaseStandingFirst
    {
        get
        {
            int s = CounterbalanceSlot;
            if (s <= 0) return true;                       // degenerate: keep it defined
            return ALLOCATION[(s - 1) % ALLOCATION.Length] == 'S';
        }
    }

    // Order for THIS visit. Crossover: visit 2 (and any even visit) runs the opposite
    // order to visit 1, so each condition sits in the early and the late half of a
    // session exactly once per participant.
    public bool StandingFirst
    {
        get { return (visit % 2 == 0) ? !BaseStandingFirst : BaseStandingFirst; }
    }

    // Machine-readable order tag for the data log ("standing" / "walking" = which
    // condition ran FIRST in this visit).
    public string ConditionOrderFirst { get { return StandingFirst ? "standing" : "walking"; } }

    // ---------------------------------------------------------------------------
    // Visit field
    //
    // The Visit field starts EMPTY on a fresh launch, so the visit has to be typed. Once
    // a run has started, the visit is kept and pre-filled when the setup screen returns,
    // but only for the participant it was used with: typing a different participant asks
    // for the visit again, so one person's visit cannot carry over to the next.
    private bool visitSaved = false;               // a run started with a typed visit
    private int visitOwner = -1;                   // participant the saved visit belongs to
    private bool visitTypedOnThisScreen = false;   // operator edited the field since the screen loaded
    private string startBlockedMessage = null;     // why the last Start / Calibrate was refused

    // Visit for participant p from the Visit field. Valid only if the field holds 1 or 2
    // AND it was typed on this screen or saved for this same participant. Without a wired
    // Visit field the serialized `visit` is used, so an older scene keeps working.
    bool TryReadVisit(int p, out int v, out string problem)
    {
        problem = null;
        v = visit;
        if (visitField == null) return true;
        string s = (visitField.text ?? "").Trim();
        int parsed;
        if (s.Length == 0)
        {
            problem = "Enter the VISIT (1 or 2).";
            return false;
        }
        if (!int.TryParse(s, out parsed) || (parsed != 1 && parsed != 2))
        {
            problem = "The visit must be 1 or 2 (the field holds '" + s + "').";
            return false;
        }
        if (!visitTypedOnThisScreen && visitOwner != p)
        {
            problem = string.Format("The visit in the field ({0}) was used for P{1:00}. Type the visit for P{2:00} again.",
                parsed, visitOwner, p);
            return false;
        }
        v = parsed;
        return true;
    }

    // Refuses to start a run whose files could not be tied to a participant and a visit.
    // Returns true when the run may start; otherwise shows the reason on the setup screen.
    bool CheckParticipantAndVisit()
    {
        int p, v;
        string problem = "Enter the Participant ID.";
        if (participantField != null && int.TryParse((participantField.text ?? "").Trim(), out p)
            && TryReadVisit(p, out v, out problem))
        {
            startBlockedMessage = null;
            return true;
        }
        startBlockedMessage = "NOT STARTED: " + problem;
        Debug.LogWarning("[SETUP] " + startBlockedMessage);
        UpdateCalibrationOrderLabel();
        return false;
    }

    // Remember the visit a run starts with, so it is pre-filled when the setup screen returns.
    void SaveVisitForParticipant()
    {
        if (visitField == null) return;
        visitSaved = true;
        visitOwner = participant;
    }

    // Pull visit / slot out of their optional UI fields (call before a run starts, so
    // the values the logger writes are the ones actually in force).
    void ReadCounterbalanceFields()
    {
        int v;
        string problem;
        if (TryReadVisit(participant, out v, out problem)) visit = v;
        int s;
        if (counterbalanceSlotField != null && int.TryParse(counterbalanceSlotField.text, out s))
            counterbalanceSlot = Mathf.Max(0, s);
        Debug.Log(string.Format(
            "Counterbalance: P{0} visit {1} slot {2} -> base {3}-first, THIS visit {4}-first",
            participant, visit, CounterbalanceSlot,
            BaseStandingFirst ? "standing" : "walking", ConditionOrderFirst));
    }

    // Condition label for a block id (mirrors Analysis/config.block_to_condition and
    // CheckpointManager): block 0 == standing, any other block == walking.
    public static string ConditionForBlock(int blockId) { return blockId == 0 ? "standing" : "walking"; }

    // Copy a condition's stored difficulty set into the ACTIVE set (thresholdAngle/easy/
    // hard) that the logger and CheckpointManager read. Called at block start.
    public void SetActiveCondition(string condition)
    {
        if (condition == "walking")
        {
            thresholdAngle = thresholdWalking; easyAngle = easyWalking; hardAngle = hardWalking;
        }
        else
        {
            thresholdAngle = thresholdStanding; easyAngle = easyStanding; hardAngle = hardStanding;
        }
    }

    // Store a calibration result into the matching per-condition set (and refresh the
    // active set). Called by StaircaseManager.FinishCalibration.
    public void StoreCalibration(string condition, float thr, float easy, float hard)
    {
        if (condition == "walking")
        {
            thresholdWalking = thr; easyWalking = easy; hardWalking = hard;
        }
        else
        {
            thresholdStanding = thr; easyStanding = easy; hardStanding = hard;
        }
        SetActiveCondition(condition);
        Debug.Log(string.Format("Stored {0} calibration: thr {1:F2} / easy {2:F2} / hard {3:F2} deg",
            condition, thr, easy, hard));
    }
    
    private void Awake()
    {
        if (instance == null) {
            instance = this;
            DontDestroyOnLoad(gameObject);
            SceneManager.sceneLoaded += OnSceneLoaded;
        } else if (instance != this) {
            instance.UpdateUIReferences(this);
            Destroy(gameObject);
        }
    }

    void OnDisable()
    {
        if (instance == this) SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        // When returning to Scene 0, re-initialize UI
        networkingSync = FindObjectOfType<NetworkingSync>();
        if (scene.buildIndex == 0) 
        {
			InitializeScene0();
		}
		else
		{
			// HIDE UI in Experiment/Calibration Scenes
            if (setupUIPanel != null) setupUIPanel.SetActive(false);
            if (waitingPanel != null) waitingPanel.SetActive(false);
		}
    }

    public void UpdateUIReferences(ExperimentConductor newContext)
    {
        this.setupUIPanel = newContext.setupUIPanel;
        this.waitingPanel = newContext.waitingPanel;
        this.participantField = newContext.participantField;
        this.blockField = newContext.blockField;
        this.trialField = newContext.trialField;
        this.speedField = newContext.speedField;
        this.startButton = newContext.startButton;
        this.debugModeToggle = newContext.debugModeToggle;
        this.thresholdField = newContext.thresholdField;
        this.calibrateButton = newContext.calibrateButton;
        this.calibrateStandingButton = newContext.calibrateStandingButton;
        this.calibrateWalkingButton = newContext.calibrateWalkingButton;
        this.calibrationOrderLabel = newContext.calibrationOrderLabel;
        this.visitField = newContext.visitField;
        this.counterbalanceSlotField = newContext.counterbalanceSlotField;
        this.thresholdStandingField = newContext.thresholdStandingField;
        this.thresholdWalkingField = newContext.thresholdWalkingField;

        InitializeScene0();
    }

    void InitializeScene0()
    {
        Cursor.visible = true;
        Cursor.lockState = CursorLockMode.None;
        if (networkingSync == null) networkingSync = FindObjectOfType<NetworkingSync>();

        if (setupUIPanel != null && waitingPanel != null)
        {
            if (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName)
            {
                setupUIPanel.SetActive(true);
                waitingPanel.SetActive(false);
                
                if (thresholdField != null) thresholdField.text = thresholdAngle.ToString("F2");

                if (startButton != null) {
                    startButton.onClick.RemoveAllListeners();
                    startButton.onClick.AddListener(StartExperiment);
                }
                if (calibrateButton != null) {
                    calibrateButton.onClick.RemoveAllListeners();
                    calibrateButton.onClick.AddListener(StartCalibration);
                }
                if (calibrateStandingButton != null) {
                    calibrateStandingButton.onClick.RemoveAllListeners();
                    calibrateStandingButton.onClick.AddListener(StartCalibrationStanding);
                }
                if (calibrateWalkingButton != null) {
                    calibrateWalkingButton.onClick.RemoveAllListeners();
                    calibrateWalkingButton.onClick.AddListener(StartCalibrationWalking);
                }

                // Per-condition threshold fields: show the values stored for this participant.
                if (thresholdStandingField != null) thresholdStandingField.text = thresholdStanding.ToString("F2");
                if (thresholdWalkingField != null)  thresholdWalkingField.text  = thresholdWalking.ToString("F2");

                // Schedule hint, refreshed live as participant / visit / slot / block /
                // trial are typed.
                startBlockedMessage = null;
                if (participantField != null) {
                    participantField.onValueChanged.RemoveListener(OnSetupFieldChanged);
                    participantField.onValueChanged.AddListener(OnSetupFieldChanged);
                }
                if (visitField != null) {
                    // Empty on a fresh launch; after a run, the visit that run used (valid
                    // only for the same participant, see TryReadVisit). The listener is
                    // attached after the pre-fill so it doesn't count as typing.
                    visitTypedOnThisScreen = false;
                    visitField.onValueChanged.RemoveListener(OnVisitFieldChanged);
                    visitField.text = visitSaved ? visit.ToString() : "";
                    visitField.onValueChanged.AddListener(OnVisitFieldChanged);
                }
                if (counterbalanceSlotField != null) {
                    counterbalanceSlotField.onValueChanged.RemoveListener(OnSetupFieldChanged);
                    counterbalanceSlotField.onValueChanged.AddListener(OnSetupFieldChanged);
                }
                if (blockField != null) {
                    blockField.onValueChanged.RemoveListener(OnSetupFieldChanged);
                    blockField.onValueChanged.AddListener(OnSetupFieldChanged);
                }
                if (trialField != null) {
                    trialField.onValueChanged.RemoveListener(OnSetupFieldChanged);
                    trialField.onValueChanged.AddListener(OnSetupFieldChanged);
                }
                if (debugModeToggle != null) {
                    debugModeToggle.onValueChanged.RemoveListener(OnDebugToggleChanged);
                    debugModeToggle.onValueChanged.AddListener(OnDebugToggleChanged);
                }
                UpdateCalibrationOrderLabel();
            }
            else
            {
                setupUIPanel.SetActive(false);
                waitingPanel.SetActive(true);
            }
        }
    }

    void Start () { InitializeScene0(); }
    
    // Convenience triggers for the setup UI: wire a "Calibrate Standing" and a
    // "Calibrate Walking" button to these so the operator picks the condition with one
    // click. They set calibrationCondition (read by StaircaseManager in scene 2) then
    // launch calibration. The counterbalanced order (StandingFirst) tells the operator
    // which to run first per participant.
    public void StartCalibrationStanding() { calibrationCondition = "standing"; StartCalibration(); }
    public void StartCalibrationWalking()  { calibrationCondition = "walking";  StartCalibration(); }

    // Live operator hint, refreshed whenever a setup field changes.
    void OnSetupFieldChanged(string _)
    {
        startBlockedMessage = null;
        UpdateCalibrationOrderLabel();
    }

    void OnVisitFieldChanged(string _)
    {
        visitTypedOnThisScreen = true;
        OnSetupFieldChanged(_);
    }

    void OnDebugToggleChanged(bool _) { UpdateCalibrationOrderLabel(); }

    // ---------------------------------------------------------------------------
    // Setup-screen schedule (display only)
    //
    // Spells out INSTRUCTIONS.md §6 for the participant and visit in the fields: the
    // counterbalance group, every step of this visit with the Block and Trial to type,
    // and which step the Block / Trial fields match. Nothing here changes what a run
    // does (the operator still decides what to start), so these numbers only label the
    // steps and must be kept in line with the guide if the design changes.
    const int STANDING_BLOCKS_PER_VISIT = 1;
    const int WALKING_BLOCKS_PER_VISIT = 4;
    const int WALKING_BLOCKS_PER_BREAK = 2;   // a break after every 2nd walking block,
    const int BREAK_MINUTES = 5;              // but not after the visit's last one

    const string COLOR_WARN = "#9A0000";
    const string COLOR_MATCH = "#0B3D91";
    const string COLOR_OK = "#1B5E20";
    const string COLOR_NOTE = "#4A4A4A";

    class ScheduleStep
    {
        public bool calibration;
        public string condition;          // "standing" / "walking"
        public int block, trial;
        public int blockNumber, blockTotal; // blocks only, e.g. walking block 5 of 8
        public bool breakAfter;
    }

    static List<ScheduleStep> BuildSchedule(int visitNo, bool standingFirst)
    {
        var steps = new List<ScheduleStep>();
        AddConditionSteps(steps, visitNo, standingFirst ? "standing" : "walking");
        AddConditionSteps(steps, visitNo, standingFirst ? "walking" : "standing");
        return steps;
    }

    // One condition's part of a visit: its calibration (Trial = visit number), then its
    // blocks (Trial counts on across both visits).
    static void AddConditionSteps(List<ScheduleStep> steps, int visitNo, string condition)
    {
        bool walking = condition == "walking";
        int blockId = walking ? 1 : 0;
        int perVisit = walking ? WALKING_BLOCKS_PER_VISIT : STANDING_BLOCKS_PER_VISIT;
        steps.Add(new ScheduleStep { calibration = true, condition = condition, block = blockId, trial = visitNo });
        for (int i = 1; i <= perVisit; i++)
        {
            int n = (visitNo - 1) * perVisit + i;
            steps.Add(new ScheduleStep {
                condition = condition, block = blockId, trial = n,
                blockNumber = n, blockTotal = 2 * perVisit,
                breakAfter = walking && i % WALKING_BLOCKS_PER_BREAK == 0 && i < perVisit });
        }
    }

    static string Capitalized(string s) { return char.ToUpper(s[0]) + s.Substring(1); }

    static string StepName(ScheduleStep st)
    {
        return st.calibration
            ? "Calibrate " + st.condition.ToUpper()
            : string.Format("{0} block {1} of {2}", Capitalized(st.condition), st.blockNumber, st.blockTotal);
    }

    static string StepButton(ScheduleStep st)
    {
        return st.calibration ? "Start Calibration " + Capitalized(st.condition) : "Start Experiment";
    }

    // Must match what ILLLoggerController / StaircaseManager actually write.
    static string StepFileName(int p, int v, ScheduleStep st)
    {
        string id = FileIdentifier(p, v, st.block, st.trial);
        return st.calibration ? id + "_calibration_" + st.condition + ".csv" : id + "_" + st.condition + ".csv.gz";
    }

    public void UpdateCalibrationOrderLabel()
    {
        if (calibrationOrderLabel == null) return;
        calibrationOrderLabel.supportRichText = true;
        calibrationOrderLabel.text = BuildSetupHint();
    }

    // Stage 1 (participant only): the group and both visits' order, and a prompt for the
    // visit. Stage 2 (participant + visit): this visit's full schedule.
    string BuildSetupHint()
    {
        var sb = new System.Text.StringBuilder();
        if (!string.IsNullOrEmpty(startBlockedMessage))
            sb.AppendFormat("<size=22><color={0}><b>{1}</b></color></size>\n\n", COLOR_WARN, startBlockedMessage);
        if (debugModeToggle != null && debugModeToggle.isOn)
            sb.AppendFormat("<size=22><color={0}><b>DEBUG MODE IS ON: blocks will NOT be logged.</b></color></size> "
                + "Untick Debug Mode for real runs (calibrations are logged either way).\n\n", COLOR_WARN);

        const string FIELD_ORDER = "Fields, left to right:  Participant  |  Visit  |  Block  |  Trial";
        int p;
        if (participantField == null || !int.TryParse((participantField.text ?? "").Trim(), out p))
        {
            sb.Append("<size=24><b>Enter the Participant ID, then the Visit (1 or 2).</b></size>\n");
            sb.Append(FIELD_ORDER);
            return sb.ToString();
        }
        participant = p; // keep the counterbalance + data-file naming in sync with the field
        int s;
        if (counterbalanceSlotField != null && int.TryParse(counterbalanceSlotField.text, out s))
            counterbalanceSlot = Mathf.Max(0, s);

        bool baseStanding = BaseStandingFirst;
        string group = baseStanding ? "S" : "W";
        string visit1First = baseStanding ? "standing" : "walking";
        string visit2First = baseStanding ? "walking" : "standing";

        int v;
        string problem;
        if (!TryReadVisit(p, out v, out problem))
        {
            sb.AppendFormat("<size=24><b>P{0:00}   |   Group {1}</b></size>     (counterbalance slot {2})\n",
                p, group, CounterbalanceSlot);
            sb.AppendFormat("Visit 1: {0} first.      Visit 2: {1} first (the order crosses over).\n\n",
                visit1First.ToUpper(), visit2First.ToUpper());
            sb.AppendFormat("<size=22><color={0}><b>{1}</b></color></size>\n", COLOR_WARN, problem);
            sb.Append("The step-by-step schedule for this visit appears once the visit is set.\n\n");
            sb.AppendFormat("<color={0}>{1}</color>", COLOR_NOTE, FIELD_ORDER);
            return sb.ToString();
        }
        visit = v;
        bool standingFirst = StandingFirst;

        sb.AppendFormat("<size=24><b>P{0:00}   |   VISIT {1}   |   Group {2}</b></size>     (counterbalance slot {3})\n",
            p, v, group, CounterbalanceSlot);
        sb.AppendFormat("Group {0}: visit 1 {1} first, visit 2 {2} first.     <b>This visit: {3} FIRST{4}</b>\n\n",
            group, visit1First, visit2First, standingFirst ? "STANDING" : "WALKING",
            v == 2 ? " (crossed over)" : "");
        sb.AppendFormat("<b>Steps for visit {0}</b>: type Block and Trial as listed, then press the button in [ ]\n", v);

        int blockIn = -1, trialIn = -1;
        bool haveFields = blockField != null && trialField != null
            && int.TryParse((blockField.text ?? "").Trim(), out blockIn)
            && int.TryParse((trialField.text ?? "").Trim(), out trialIn);

        List<ScheduleStep> steps = BuildSchedule(v, standingFirst);
        var matches = new List<int>();
        for (int i = 0; i < steps.Count; i++)
        {
            ScheduleStep st = steps[i];
            string line = string.Format("{0}.    Block {1}    Trial {2}        {3}    [{4}]",
                i + 1, st.block, st.trial, StepName(st), StepButton(st));
            if (haveFields && st.block == blockIn && st.trial == trialIn)
            {
                matches.Add(i);
                // colour only, no bold, so the Block / Trial columns stay aligned
                line = string.Format("<color={0}>{1}      <b>= fields</b></color>", COLOR_MATCH, line);
            }
            sb.Append(line).Append('\n');
            if (st.breakAfter)
                sb.AppendFormat("<color={0}>                ----- {1} min break -----</color>\n", COLOR_NOTE, BREAK_MINUTES);
        }
        sb.Append('\n');

        if (!haveFields)
        {
            sb.Append("<b>Next:</b> type the Block and Trial of the step you are about to run.\n");
        }
        else if (matches.Count > 0)
        {
            foreach (int i in matches)
                sb.AppendFormat("<color={0}>Fields = step {1} ({2}):  [{3}] writes <b>{4}</b></color>\n",
                    COLOR_OK, i + 1, StepName(steps[i]), StepButton(steps[i]), StepFileName(p, v, steps[i]));
        }
        else
        {
            sb.AppendFormat("<color={0}><b>Block {1}, Trial {2} is not a step of visit {3} for P{4:00}.</b> "
                + "Check the fields (fine only for a re-run under a new Trial number).</color>\n",
                COLOR_WARN, blockIn, trialIn, v, p);
            sb.AppendFormat("[Start Experiment] would write {0}_{1}.csv.gz\n",
                FileIdentifier(p, v, blockIn, trialIn), ConditionForBlock(blockIn));
        }

        sb.AppendFormat("\n<color={0}>{1}\nBlock = condition (0 standing, 1 walking). Each calibration uses Trial = visit number.\n"
            + "Blocks count on per condition across both visits (standing 1-{2}, walking 1-{3}). Never reuse a Block/Trial pair.</color>",
            COLOR_NOTE, FIELD_ORDER, 2 * STANDING_BLOCKS_PER_VISIT, 2 * WALKING_BLOCKS_PER_VISIT);
        return sb.ToString();
    }

    public void StartCalibration()
    {
        if (!CheckParticipantAndVisit()) return;

        // Parse participant/block/trial first so the calibration data file is named
        // to match the main log (p##-v#-b#-t##_calibration_<condition>.csv).
        if (participantField != null) int.TryParse(participantField.text, out participant);
        if (blockField != null) int.TryParse(blockField.text, out block);
        if (trialField != null) int.TryParse(trialField.text, out trial);
        ReadCounterbalanceFields();
        SaveVisitForParticipant();
        if (!string.IsNullOrEmpty(dataStoragePath) && !Directory.Exists(dataStoragePath))
            Directory.CreateDirectory(dataStoragePath);

        Debug.Log("Starting Calibration...");
        if (networkingSync == null) networkingSync = FindObjectOfType<NetworkingSync>();
        if (networkingSync != null)
        {
            // Broadcasts command to load Scene 2 (Index 2)
            networkingSync.BroadcastReturnToScene(2);
        }
    }

    public void StartExperiment()
    {
        if (!CheckParticipantAndVisit()) return;

        int.TryParse(participantField.text, out participant);
        int.TryParse(blockField.text, out block);
        int.TryParse(trialField.text, out trial);
        ReadCounterbalanceFields();
        SaveVisitForParticipant();
        if (speedField != null && !float.TryParse(speedField.text, out experimentSpeed)) experimentSpeed = 1.0f;

        // Manual threshold override: if a valid positive value is in the Threshold
        // field, use it directly (restart a participant with a saved threshold,
        // no recalibration). Easy/Hard are derived from it with the model factors.
        if (thresholdField != null)
        {
            float parsedThreshold;
            if (float.TryParse(thresholdField.text, out parsedThreshold) && parsedThreshold > 0f)
            {
                // Manual override seeds BOTH conditions as a coarse fallback (e.g. a restart
                // without recalibration); a per-condition calibration overrides its own set.
                thresholdStanding = thresholdWalking = parsedThreshold;
                easyStanding = easyWalking = parsedThreshold * easyFactor;
                hardStanding = hardWalking = parsedThreshold * hardFactor;
                thresholdAngle = parsedThreshold;
                easyAngle = parsedThreshold * easyFactor;
                hardAngle = parsedThreshold * hardFactor;
                Debug.Log(string.Format(
                    "Using manual threshold {0:F2} deg -> easy {1:F2}, hard {2:F2} (both conditions)",
                    thresholdAngle, easyAngle, hardAngle));
            }
        }

        // Optional per-condition manual overrides. A valid value in either field overrides
        // just that condition's stored threshold (easy/hard via the model factors). These
        // run AFTER the single Threshold field's both-condition seed, so they win — and,
        // pre-filled from the stored calibrations, they keep the single field from clobbering
        // the other condition during the interleaved workflow.
        if (thresholdStandingField != null)
        {
            float ts;
            if (float.TryParse(thresholdStandingField.text, out ts) && ts > 0f)
            {
                thresholdStanding = ts; easyStanding = ts * easyFactor; hardStanding = ts * hardFactor;
            }
        }
        if (thresholdWalkingField != null)
        {
            float tw;
            if (float.TryParse(thresholdWalkingField.text, out tw) && tw > 0f)
            {
                thresholdWalking = tw; easyWalking = tw * easyFactor; hardWalking = tw * hardFactor;
            }
        }

		if (debugModeToggle != null) isDebugMode = debugModeToggle.isOn;
        if (!string.IsNullOrEmpty(dataStoragePath)) {
            if (!Directory.Exists(dataStoragePath)) Directory.CreateDirectory(dataStoragePath);
        }

        if (setupUIPanel != null) setupUIPanel.SetActive(false);
        if (networkingSync == null) networkingSync = FindObjectOfType<NetworkingSync>();
        if (networkingSync != null)
            networkingSync.BroadcastExperimentData(participant, block, trial, experimentSpeed, mainSceneName, isDebugMode);     
    }
}
