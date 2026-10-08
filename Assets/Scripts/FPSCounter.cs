using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

public class FPSCounter : MonoBehaviour
{
    private float startTime;
    private float totalElapsed;
    private float frameCount;

    // We can hold references to either manager
    public CheckpointManager checkpointManager;
    public ILLLoggerController logger;
    public StaircaseManager staircaseManager;

    private SplineWalker splineWalker;
    private GameObject skeleton;
    
    float deltaTime = 0.0f;

    void Start()
    {
        if (System.Environment.MachineName != MasterTrackingData.HeadNodeMachineName)
        {
            Destroy(this);
            return;
        }

        // Try to find components if not assigned
        splineWalker = GetComponent<SplineWalker>();
        
        // Auto-Detect which scene we are in ---
        if (checkpointManager == null) checkpointManager = FindObjectOfType<CheckpointManager>();
        if (staircaseManager == null) staircaseManager = FindObjectOfType<StaircaseManager>();
        if (logger == null) logger = FindObjectOfType<ILLLoggerController>();

        startTime = Time.realtimeSinceStartup;

        if (!Application.isEditor)
        {
            Cursor.lockState = CursorLockMode.Locked;
        }
    }

    void Update()
    {
        if (skeleton == null)
        {
            skeleton = GameObject.Find("OptiTrack Skeleton - Skeleton1");
        }
        
        totalElapsed = Time.realtimeSinceStartup - startTime;
        frameCount = Time.frameCount;
        
        deltaTime += (Time.deltaTime - deltaTime) * 0.1f;
    }

    void OnGUI()
    {
        int w = Screen.width, h = Screen.height;

        var fps = 1.0f / Mathf.Max(0.0001f, deltaTime);
        var msec = deltaTime * 1000.0f;

        var builder = new StringBuilder();

        // --- Global telemetry (always shown) ---
        int mins = (int)(totalElapsed / 60f);
        int secs = (int)(totalElapsed % 60f);
        builder.AppendFormat("t {0:00}:{1:00} | {2:0.} fps ({3:0.0} ms) | frame {4}\n",
            mins, secs, fps, msec, (int)frameCount);

        // Active projection model (global, synced across nodes; F1-F7 switch, =/- tune,
        // R re-locks hybrid dRef). Lets the operator confirm the rig is in the intended
        // mode (OffAxis by default, swayGain 0 / smoothing 1.8 Hz) at a glance.
        builder.AppendFormat("proj {0}", ProjectionPlane.mode);
        if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.HybridWide)
            builder.AppendFormat(" | widen {0:F2}", ProjectionPlane.hybridWidenFactor);
        else if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.Partial)
            builder.AppendFormat(" | blend {0:F2}", ProjectionPlane.partialBlend);
        else if (ProjectionPlane.mode == ProjectionPlane.ProjectionMode.OffAxis)
            builder.AppendFormat(" | sway {0:F2} smooth {1:F1}Hz",
                ProjectionPlane.swayGain, ProjectionPlane.swaySmoothingHz);
        builder.Append("\n");

        // --- Scene-specific block ---
        if (checkpointManager != null)
        {
            builder.Append(checkpointManager.GetExperimentStatusReport());

            // Movement / next-task line (kept here: FPSCounter holds the SplineWalker)
            if (splineWalker != null)
            {
                string status = splineWalker.isMovementPaused ? "WAITING" : "MOVING";
                builder.AppendFormat("Speed: {0:0.0} m/s ({1})", splineWalker.speed, status);
                if (checkpointManager.spline != null)
                {
                    float currentDist = splineWalker.progress * checkpointManager.spline.Length;
                    builder.AppendFormat(" | Next: {0}",
                        checkpointManager.GetNextCheckpointStats(currentDist, splineWalker.speed));
                }
                builder.Append("\n");
            }
        }
        else if (staircaseManager != null)
        {
            builder.Append(staircaseManager.GetStatusReport());
        }

        // --- Tracking telemetry + lateral-sway alarm ---
        // Sway level is computed from the SAME thresholds the logger uses (read off the
        // logger), and computed here from the LIVE head position so the alarm fires even
        // if logging is disabled (e.g. debug mode). Direction comes from the sign of X.
        int swayLevel = 0;
        float lateral = 0f;
        if (logger != null)
        {
            Vector3 hp = logger.CurrentLocalHeadPosition;
            lateral = hp.x;
            builder.AppendFormat("Head[loc] X{0:F2} Y{1:F2} Z{2:F2}\n", hp.x, hp.y, hp.z);
            float a = Mathf.Abs(lateral);
            swayLevel = (a >= logger.lateralSwayCriticalMeters) ? 2
                      : (a >= logger.lateralSwayWarnMeters) ? 1 : 0;
        }
        builder.AppendFormat("Skeleton: {0}\n", skeleton != null ? "OK" : "MISSING");
        if (swayLevel > 0)
        {
            // +X = participant's right (flip the labels here if the rig is mirrored).
            string side = (lateral > 0f) ? "RIGHT" : "LEFT";
            if (swayLevel >= 2)
                builder.AppendFormat("[!!! SWAY {0} {1:F2} m] RE-CENTER NOW - OPTITRACK MAY LOSE TRACKING",
                    side, Mathf.Abs(lateral));
            else
                builder.AppendFormat("[!! sway {0} {1:F2} m] re-center participant",
                    side, Mathf.Abs(lateral));
        }

        // --- Render (drop shadow + color; panel turns amber on WARN, red on CRITICAL) ---
        var style = new GUIStyle();
        style.alignment = TextAnchor.UpperLeft;
        style.fontSize = Mathf.Max(12, h * 2 / 100);
        var rect = new Rect(12, 10, w - 20, h * 95 / 100);

        string text = builder.ToString();
        Color main = (swayLevel >= 2) ? Color.red
                   : (swayLevel == 1) ? new Color(1f, 0.75f, 0.1f, 1f)            // amber
                   : new Color(0.85f, 0.9f, 1.0f, 1.0f);

        // shadow for legibility against bright scenes
        style.normal.textColor = new Color(0f, 0f, 0f, 0.75f);
        GUI.Label(new Rect(rect.x + 2, rect.y + 2, rect.width, rect.height), text, style);
        style.normal.textColor = main;
        GUI.Label(rect, text, style);

        // --- Calibration result banner (HEAD-NODE operator only; FPSCounter does not run
        //     on render nodes, so this is never shown to the participant) ---
        if (staircaseManager != null && staircaseManager.CalibrationFinished)
        {
            bool ok = staircaseManager.CalibrationReliable;
            string banner = ok ? "DONE" : "UNRELIABLE";
            Color bcol = ok ? new Color(0.25f, 0.95f, 0.35f) : new Color(1f, 0.30f, 0.25f);

            var bstyle = new GUIStyle();
            bstyle.alignment = TextAnchor.MiddleCenter;
            bstyle.fontStyle = FontStyle.Bold;
            bstyle.fontSize = Mathf.Max(40, h * 9 / 100);
            var brect = new Rect(0, h * 0.36f, w, h * 0.18f);
            bstyle.normal.textColor = new Color(0f, 0f, 0f, 0.85f);      // shadow
            GUI.Label(new Rect(brect.x + 4, brect.y + 4, brect.width, brect.height), banner, bstyle);
            bstyle.normal.textColor = bcol;
            GUI.Label(brect, banner, bstyle);

            var sub = new GUIStyle();
            sub.alignment = TextAnchor.MiddleCenter;
            sub.fontSize = Mathf.Max(16, h * 25 / 1000);
            sub.normal.textColor = bcol;
            GUI.Label(new Rect(0, h * 0.53f, w, h * 0.08f),
                ok ? "calibration reliable" : "recalibration recommended", sub);
        }
    }
}