using UnityEngine;

// Diagnostic helper: spawns N marker cubes at fixed depths in front of an anchor,
// then logs each marker's normalized-device-coordinate (NDC) position per active
// CAVE camera once per second. Use to verify parallax magnitude empirically.
//
// Add this component to any active GameObject in the scene. If `anchor` is left
// empty, the script's own transform is used as the origin. Markers are placed at
// anchor.position + (0, heightOffset, depth) for each depth in `depths`.
//
// Log line format (one per marker per camera per tick):
//   [PMarker:<camName>|D=<depth>] world=(x,y,z) viewZ=<v> ndc=(x,y)
//
// To extract per-depth NDC shift across a head motion, grep the per-camera log
// at two different head poses and compare ndc.x for each depth.

public class ParallaxMarkerLogger : MonoBehaviour
{
    [Tooltip("Depths (meters) in +Z from the anchor at which to spawn markers.")]
    public float[] depths = new float[] { 2f, 5f, 10f, 30f, 100f };

    [Tooltip("Anchor transform. If null, uses this GameObject's transform.")]
    public Transform anchor;

    [Tooltip("Height offset above anchor so markers are visible at eye level.")]
    public float heightOffset = 0.2f;

    [Tooltip("Cube size at nearest depth (scales by sqrt(depth) to stay visible).")]
    public float nearMarkerSize = 0.5f;

    [Tooltip("Log every N frames.")]
    public int logInterval = 60;

    GameObject[] markers;

    void Start()
    {
        Transform a = anchor != null ? anchor : transform;
        markers = new GameObject[depths.Length];
        for (int i = 0; i < depths.Length; i++)
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = "ParallaxMarker_" + depths[i].ToString("F0") + "m";
            cube.transform.position = a.position + new Vector3(0f, heightOffset, depths[i]);
            float s = nearMarkerSize * Mathf.Sqrt(depths[i] / Mathf.Max(depths[0], 0.01f));
            cube.transform.localScale = new Vector3(s, s, s);

            var rend = cube.GetComponent<Renderer>();
            float t = depths.Length == 1 ? 0f : (float)i / (depths.Length - 1);
            rend.material.color = Color.Lerp(Color.red, Color.cyan, t);

            var col = cube.GetComponent<Collider>();
            if (col != null) Destroy(col);

            markers[i] = cube;
        }
        Debug.LogFormat("[PMarkers] Spawned {0} markers at depths: {1}",
            markers.Length,
            string.Join(", ", System.Array.ConvertAll(depths, d => d.ToString("F1"))));
    }

    void LateUpdate()
    {
        if (markers == null) return;
        if (Time.frameCount % logInterval != 0) return;

        foreach (var cam in Camera.allCameras)
        {
            if (cam == null || !cam.isActiveAndEnabled) continue;
            var parent = cam.transform.parent;
            if (parent == null) continue;
            // Only the CAVE projection cameras have a ProjectionPlane on the parent.
            if (parent.GetComponent<ProjectionPlane>() == null) continue;

            Matrix4x4 view = cam.worldToCameraMatrix;
            Matrix4x4 vp = cam.projectionMatrix * view;

            for (int i = 0; i < markers.Length; i++)
            {
                if (markers[i] == null) continue;
                Vector3 worldPos = markers[i].transform.position;
                Vector4 clip = vp * new Vector4(worldPos.x, worldPos.y, worldPos.z, 1f);
                if (clip.w == 0f) continue;
                float ndc_x = clip.x / clip.w;
                float ndc_y = clip.y / clip.w;
                float viewZ = view.MultiplyPoint(worldPos).z;

                Debug.LogFormat("[PMarker:{0}|D={1:F0}|{6}] world={2} viewZ={3:F3} ndc=({4:F4},{5:F4})",
                    parent.name, depths[i], worldPos.ToString("F2"), viewZ, ndc_x, ndc_y,
                    ProjectionPlane.mode);
            }
        }
    }
}
