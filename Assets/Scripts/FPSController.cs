using UnityEngine;

/// <summary>
/// Pins the target frame rate and vsync. Frame pacing matters here beyond smoothness:
/// stimulus duration is counted in rendered frames, so an unstable rate changes how long
/// the Gabor is actually on screen.
/// </summary>
public class FPSController : MonoBehaviour
{
    public int targetFrameRate = 60;
    public string targetLevel;

    void Start ()
    {
        QualitySettings.vSyncCount = 0;

        string[] names = QualitySettings.names;

        for (var i = 0; i < names.Length; i++)
        {
            if (names[i] == targetLevel)
            {
                Debug.Log("setting quality level to " + names[i]);
                QualitySettings.SetQualityLevel(i, true);
            }
        }
    }

    void Update()
    {
        if (targetFrameRate != Application.targetFrameRate)
        {
            Application.targetFrameRate = targetFrameRate;
        }
    }
}
