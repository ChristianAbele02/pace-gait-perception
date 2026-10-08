using UnityEngine;

/// <summary>
/// Prints received key codes. Diagnostic aid for the response box and the CAVE's input
/// forwarding; not used during a real session.
/// </summary>
public class KeyDebugger : MonoBehaviour 
{
    void OnGUI()
    {
        Event e = Event.current;
        if (e.isKey && e.type == EventType.KeyDown)
        {
            Debug.Log("Detected Key: " + e.keyCode);
        }
    }
}