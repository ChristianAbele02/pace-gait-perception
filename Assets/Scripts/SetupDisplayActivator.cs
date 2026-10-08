using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Makes the SETUP scene fill every connected monitor instead of leaving the second
/// (top) stacked CAVE monitor black.
///
/// The setup scene has no UniCave/DualPipe rig — it renders through a single
/// <c>StaticCamera</c> to display 0, so on a render node whose two stacked monitors are
/// exposed as two separate displays only the first (bottom) monitor shows anything; the
/// second (top) display is never activated and stays black. The main experiment does not
/// have this problem because UniCave activates and drives both displays.
///
/// This runs automatically at startup (no GameObject/scene wiring needed). It:
///   1. Activates every connected display (display 0 is always active), and
///   2. ONLY in the setup scene, clones the StaticCamera onto each extra display so the
///      same setup view (background + instructions) is shown there too.
///
/// It deliberately does nothing in the experiment scene (UniCave owns the displays), and
/// the mirror cameras are plain scene objects that are destroyed on the next scene load,
/// so they cannot leak into the experiment. On the HEAD NODE it is a no-op regardless of
/// monitor count: its secondary monitor(s) are intentionally left as a normal desktop so
/// the operator can run other applications / keep files open there while Unity runs (the
/// head node only uses display 0 for the operator view + overlay). The Debug.Log line reports how many displays
/// were found, which is the quickest way to confirm on the rig whether the two stacked
/// monitors are exposed as two displays (the assumption here) or one spanned display.
/// </summary>
public static class SetupDisplayActivator
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Init()
    {
        // The HEAD NODE keeps its secondary monitor(s) as a normal desktop so the
        // operator can run other applications / keep files open there while Unity runs;
        // only display 0 (operator view + overlay) is used. The RENDER NODES still need
        // every CAVE screen, so they activate and mirror as before.
        bool isHeadNode = (System.Environment.MachineName == MasterTrackingData.HeadNodeMachineName);

        // 1) Activate every connected display (render nodes only). Idempotent and
        //    harmless: display 0 is always active; the experiment needs the extra
        //    displays active anyway.
        int activated = 0;
        for (int i = 1; i < Display.displays.Length && !isHeadNode; i++)
        {
            if (!Display.displays[i].active)
            {
                Display.displays[i].Activate();
                activated++;
            }
        }
        Debug.Log("SetupDisplayActivator: " + Display.displays.Length
                  + " display(s) detected; activated " + activated + " extra"
                  + (isHeadNode ? " (head node: extra monitors left free for desktop use)." : "."));

        // 2) Only the setup scene needs mirror cameras (the experiment uses UniCave), and
        //    only on the render nodes — the head node leaves its extra monitor(s) free.
        if (isHeadNode) return;
        Scene scene = SceneManager.GetActiveScene();
        if (!scene.name.Contains("Setup")) return;
        if (Display.displays.Length <= 1) return;   // single display: nothing to mirror

        Camera src = FindSetupCamera();
        if (src == null)
        {
            Debug.LogWarning("SetupDisplayActivator: no StaticCamera/main camera found to mirror.");
            return;
        }

        for (int i = 1; i < Display.displays.Length; i++)
        {
            if (DisplayAlreadyRendered(i)) continue;   // something already targets this display

            var go = new GameObject("SetupMirrorCamera_" + i);
            var cam = go.AddComponent<Camera>();
            cam.CopyFrom(src);                          // same clear/FOV/culling as the StaticCamera
            cam.targetDisplay = i;                      // ...but onto the extra display
            go.transform.position = src.transform.position;
            go.transform.rotation = src.transform.rotation;
            Debug.Log("SetupDisplayActivator: added mirror camera for display " + i);
        }
    }

    private static Camera FindSetupCamera()
    {
        GameObject named = GameObject.Find("StaticCamera");
        if (named != null)
        {
            Camera c = named.GetComponent<Camera>();
            if (c != null) return c;
        }
        if (Camera.main != null) return Camera.main;
        return (Camera.allCamerasCount > 0) ? Camera.allCameras[0] : null;
    }

    private static bool DisplayAlreadyRendered(int display)
    {
        foreach (Camera c in Camera.allCameras)
            if (c.targetDisplay == display) return true;
        return false;
    }
}
