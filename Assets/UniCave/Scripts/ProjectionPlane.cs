//MIT License
//Copyright 2016-Present
//Ross Tredinnick
//Brady Boettcher
//Living Environments Laboratory
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
//to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
//sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

using UnityEngine;
using System.Collections;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class ProjectionPlane : MonoBehaviour {

    public GameObject plane;
    public GameObject trackedRotation;
    public bool forceRight = false;

    public string MachineName;

    // Eye separation scale: 1.0 = physically accurate IPD. Adjust live with [ / ] keys.
    public static float parallaxScale = 1.0f;

    // Projection mode. Set globally via NetworkingSync RPC (F1-F7 keys).
    //   OffAxis        = full CAVE off-axis (Kooima): the only TRUE window -- walls tile
    //                    perfectly, world locked to the physical room, and the Gabor renders
    //                    at its exact calibrated angular size/shape. This is the production
    //                    DEFAULT. Its one weakness -- the head-sway wobble -- is handled by
    //                    the render-eye sway damping below (swayGain/swaySmoothingHz), NOT by
    //                    distorting the projection (which would warp the calibrated Gabor).
    //   Hybrid         = symmetric per-wall frustum, fixed dRef captured at entry.
    //                    Kills the wobble and keeps depth parallax, but does NOT stitch
    //                    the wall corners, so a multi-wall rig shows corner gaps.
    //   HybridWide     = Hybrid with horizontal FOV * hybridWidenFactor (default 1.10),
    //                    overlaps adjacent walls' content to hide the corner gap. Wobble-free
    //                    but the widening slightly distorts the Gabor, so it is an
    //                    alternative for testing, not the default.
    //   Partial        = off-axis math with horizontal shear scaled by partialBlend,
    //                    blends between Hybrid feel (0) and full OffAxis (1).
    //   HybridSeamless = Hybrid feel, but each wall extends only the edges that face a
    //                    neighbour wall (inner-edge half-span = n*tan(gamma/2)) so
    //                    adjacent walls' frustum rays meet EXACTLY at the physical
    //                    corner. Neighbour edges are auto-detected from the plane mesh
    //                    GameObject name ("front wall ...", "left wall ...",
    //                    "right wall ..."). Geometrically exact corners but the edge
    //                    stretch can read as a slight fisheye.
    public enum ProjectionMode { OffAxis, Hybrid, HybridWide, Partial, HybridSeamless }
    public static ProjectionMode mode = ProjectionMode.OffAxis;
    public static float hybridWidenFactor = 1.1f;
    public static float partialBlend = 0.5f;

    // Angle (degrees) between this wall's normal and an adjacent wall's normal.
    // 78 deg for this CAVE. Single static - the CAVE is geometrically regular.
    public static float neighborAngleDeg = 78f;

    // Per-frame diagnostic log (throttled to once/sec/screen). OFF in production to
    // keep render-node output_log.txt clean; flip true (or via a debug build) when
    // empirically checking frustum/parallax values.
    public static bool debugProjectionLog = false;

    // --- Render-eye sway stabilization (OffAxis comfort; RENDERING ONLY) ---
    // The off-axis frustum faithfully renders every head sway, so gait-induced lateral
    // sway makes the background slosh (the "wobble"). This low-passes the head's LOCAL
    // position (its physical position inside the CAVE, i.e. sway/bob -- world locomotion
    // lives on the parent rig and is left untouched) and adds back only swayGain of the
    // fast residual. The TRACKED head data itself is never modified, so the logger / gait
    // analysis still receive the raw 6-DOF trace. swayGain 1 = full veridical parallax
    // (old behaviour); 0 = fully stabilized. swaySmoothingHz = low-pass cutoff dividing
    // "sway" from deliberate motion. Live keys in NetworkingSync (=/- and , / .).
    public static float swayGain = 0.0f;         // 0: fully stabilized (no fast-sway parallax)
    public static float swaySmoothingHz = 1.8f;  // Hz; motion below this is kept as real
                                                 // (1.8: shorter lag than 1.2 -> less smear,
                                                 //  still attenuates the ~2 Hz gait bob)

    public void RecalibrateHybridDRef() { hybridDRef = -1f; }

    // Kept for backward compatibility with NetworkingSync's P-toggle RPC.
    public static bool useStandardPerspective {
        get { return mode == ProjectionMode.Hybrid || mode == ProjectionMode.HybridWide; }
        set { mode = value ? ProjectionMode.Hybrid : ProjectionMode.OffAxis; }
    }

    private Camera leftEye = null;
    private Camera rightEye = null;
    private Vector3 pa, pb, pc;

    // Cached eye-to-wall distance captured at the first hybrid frame. Used so
    // lateral head motion does not differentially scale side-wall FOVs.
    private float hybridDRef = -1f;

    // Low-passed head LOCAL position for render-eye sway stabilization.
    private Vector3 smoothHeadLocal;
    private bool hasSmoothHead = false;

	void Start () {
        string mn = System.Environment.MachineName;

#if UNITY_EDITOR
        if (!UnityEditor.EditorApplication.isPlaying)
#endif
        {
            if (!string.Equals(mn, MachineName, System.StringComparison.OrdinalIgnoreCase))
            {
                //remove extra cameras in case when machine shouldn't be using them..
                Transform lc = transform.GetChild(0);
                lc.parent = null;
                Destroy(lc.gameObject);
                if (transform.childCount > 0)
                {
                    Transform rc = transform.GetChild(0);
                    rc.parent = null;
                    Destroy(rc.gameObject);
                }
                Destroy(this.gameObject);
            }
        }
	}

	void LateUpdate () {

        string mn = System.Environment.MachineName;
#if UNITY_EDITOR
        if (string.Equals(mn, MachineName, System.StringComparison.OrdinalIgnoreCase) || UnityEditor.EditorApplication.isPlaying)
#else
        if (string.Equals(mn, MachineName, System.StringComparison.OrdinalIgnoreCase))
#endif
        {
            if (plane != null)
            {
                // Index-based camera lookup. The head node's plane has a single
                // child literally named "Camera"; render-node planes have two
                // children (stereo). Name-based lookup ("LeftEye"/"RightEye")
                // would miss the head node entirely - see commit 397e92d.
                leftEye = null;
                rightEye = null;
                if (forceRight)
                {
                    rightEye = transform.GetChild(0).GetComponent<Camera>();
                }
                else
                {
                    leftEye = transform.GetChild(0).GetComponent<Camera>();
                }

                if (transform.childCount > 1)
                {
                    rightEye = transform.GetChild(1).GetComponent<Camera>();
                }
            }

            Quaternion trackerRotation = Quaternion.identity;
            if (trackedRotation != null)
            {
                trackerRotation = trackedRotation.transform.rotation;
            }

            // Stabilized RENDER eye: low-pass the head's LOCAL position (sway/bob inside the
            // CAVE; locomotion is on the parent rig and untouched) and add back only swayGain
            // of the fast residual, then map back to world. swayGain==1 reproduces the raw
            // tracked head exactly. The tracked transform itself is never written, so logging
            // still sees the raw pose.
            Transform headT = transform.parent;
            Vector3 headLocal = headT.localPosition;
            if (!hasSmoothHead) { smoothHeadLocal = headLocal; hasSmoothHead = true; }
            else
            {
                float a = Mathf.Clamp01(1f - Mathf.Exp(
                    -2f * Mathf.PI * Mathf.Max(0.01f, swaySmoothingHz) * Mathf.Max(1e-4f, Time.deltaTime)));
                smoothHeadLocal = Vector3.Lerp(smoothHeadLocal, headLocal, a);
            }
            Vector3 renderLocal = smoothHeadLocal + (headLocal - smoothHeadLocal) * Mathf.Clamp01(swayGain);
            Vector3 trackedHead = (headT.parent != null) ? headT.parent.TransformPoint(renderLocal) : renderLocal;

            pa = plane.transform.TransformPoint(plane.GetComponent<MeshFilter>().mesh.vertices[0]);
            pb = plane.transform.TransformPoint(plane.GetComponent<MeshFilter>().mesh.vertices[2]);
            pc = plane.transform.TransformPoint(plane.GetComponent<MeshFilter>().mesh.vertices[3]);

            // Screen coordinate axes from the vertex positions. Same axes used by
            // getAsymProjMatrix, so the view matrix and projection matrix stay
            // consistent in one frame.
            Vector3 vr = (pb - pa).normalized;
            Vector3 vu = (pc - pa).normalized;
            Vector3 vn = Vector3.Cross(vr, vu).normalized;

            // Orient the camera to face the screen so the skybox and any non-custom
            // rendering also use the correct screen orientation.
            transform.rotation = Quaternion.LookRotation(vn, vu);

            bool useHybrid = (mode == ProjectionMode.Hybrid
                              || mode == ProjectionMode.HybridWide
                              || mode == ProjectionMode.HybridSeamless);
            float widen = (mode == ProjectionMode.HybridWide) ? hybridWidenFactor : 1f;
            float horizBlend = (mode == ProjectionMode.Partial) ? partialBlend : 1f;
            // HybridSeamless auto-detects which edges meet a neighbour wall from the
            // plane mesh GameObject's name. Convention in the ILL-UniCave prefab:
            //   "front wall ..." -> both edges meet side walls
            //   "left wall ..."  -> right edge meets front wall
            //   "right wall ..." -> left edge meets front wall
            // Any other name falls back to false/false -> renders as plain Hybrid.
            bool seamless = (mode == ProjectionMode.HybridSeamless);
            bool extL = false, extR = false;
            if (seamless && plane != null)
            {
                string n_ = plane.name.ToLower();
                if (n_.Contains("front wall")) { extL = true; extR = true; }
                else if (n_.Contains("left wall"))  { extR = true; }
                else if (n_.Contains("right wall")) { extL = true; }
            }
            float neighborHalfRad = neighborAngleDeg * 0.5f * Mathf.Deg2Rad;

            // For hybrid modes, capture a reference eye-to-wall distance on the first
            // call so subsequent lateral motion doesn't differentially scale wall
            // FOVs. Re-captured whenever hybrid is freshly entered (or via R /
            // CheckpointManager's start-baseline recalibration).
            float hybridD = -1f;
            if (useHybrid)
            {
                Vector3 refEye = trackedHead + (trackerRotation * MasterTrackingData.LeftEyeOffset * parallaxScale);
                float curD = Vector3.Dot(pa - refEye, vn);
                if (hybridDRef <= 0f && curD > 0.05f) hybridDRef = curD;
                hybridD = hybridDRef > 0f ? hybridDRef : curD;
            }
            else
            {
                hybridDRef = -1f;
            }

            if (leftEye != null)
            {
                Vector3 eyePos = trackedHead + (trackerRotation * MasterTrackingData.LeftEyeOffset * parallaxScale);
                leftEye.transform.position = eyePos;
                leftEye.worldToCameraMatrix = getViewMatrix(eyePos, vr, vu, vn);
                leftEye.projectionMatrix = useHybrid
                    ? getHybridProjMatrix(pa, pb, pc, eyePos, leftEye, hybridD, widen, extL, extR, neighborHalfRad)
                    : getAsymProjMatrix(pa, pb, pc, eyePos, leftEye, horizBlend);
                leftEye.SetStereoProjectionMatrix(Camera.StereoscopicEye.Left, leftEye.projectionMatrix);
            }

            if (rightEye != null)
            {
                Vector3 eyePos = trackedHead + (trackerRotation * MasterTrackingData.RightEyeOffset * parallaxScale);
                rightEye.transform.position = eyePos;
                rightEye.worldToCameraMatrix = getViewMatrix(eyePos, vr, vu, vn);
                rightEye.projectionMatrix = useHybrid
                    ? getHybridProjMatrix(pa, pb, pc, eyePos, rightEye, hybridD, widen, extL, extR, neighborHalfRad)
                    : getAsymProjMatrix(pa, pb, pc, eyePos, rightEye, horizBlend);
                rightEye.SetStereoProjectionMatrix(Camera.StereoscopicEye.Right, rightEye.projectionMatrix);
            }

            // ---- DIAGNOSTIC LOG (throttled to once/sec per screen; off by default) ----
            if (debugProjectionLog && Time.frameCount % 60 == 0 && leftEye != null)
            {
                Vector3 eyePos = leftEye.transform.position;
                Vector3 va = pa - eyePos;
                Vector3 vb = pb - eyePos;
                Vector3 vc = pc - eyePos;
                float n = leftEye.nearClipPlane;
                float d = Vector3.Dot(va, vn);
                float l = Vector3.Dot(vr, va) * n / d;
                float r = Vector3.Dot(vr, vb) * n / d;
                float bo = Vector3.Dot(vu, va) * n / d;
                float t = Vector3.Dot(vu, vc) * n / d;

                Matrix4x4 setProj = leftEye.projectionMatrix;
                Matrix4x4 setView = leftEye.worldToCameraMatrix;

                Debug.LogFormat(
                    "[PP:{0}] mode={20} head={1} eye={2} d={3:F3} l={4:F4} r={5:F4} b={6:F4} t={7:F4} | proj_row0=({8:F3},{9:F3},{10:F3},{11:F3}) proj_row1=({12:F3},{13:F3},{14:F3},{15:F3}) | view_row2=({16:F3},{17:F3},{18:F3},{19:F3})",
                    gameObject.name,
                    trackedHead.ToString("F3"),
                    eyePos.ToString("F3"),
                    d, l, r, bo, t,
                    setProj[0,0], setProj[0,1], setProj[0,2], setProj[0,3],
                    setProj[1,0], setProj[1,1], setProj[1,2], setProj[1,3],
                    setView[2,0], setView[2,1], setView[2,2], setView[2,3],
                    mode);
            }
        }
    }

    // Builds a view matrix from the screen's coordinate axes and eye position.
    // Row 2 is negated because Unity's view space has -Z pointing into the scene
    // (OpenGL convention), while vn points into the screen (same direction).
    static Matrix4x4 getViewMatrix(Vector3 pe, Vector3 vr, Vector3 vu, Vector3 vn)
    {
        Matrix4x4 m = new Matrix4x4();
        m[0, 0] =  vr.x; m[0, 1] =  vr.y; m[0, 2] =  vr.z; m[0, 3] = -Vector3.Dot(vr, pe);
        m[1, 0] =  vu.x; m[1, 1] =  vu.y; m[1, 2] =  vu.z; m[1, 3] = -Vector3.Dot(vu, pe);
        m[2, 0] = -vn.x; m[2, 1] = -vn.y; m[2, 2] = -vn.z; m[2, 3] =  Vector3.Dot(vn, pe);
        m[3, 0] = 0;     m[3, 1] = 0;     m[3, 2] = 0;     m[3, 3] = 1;
        return m;
    }

    // "In-game" projection: symmetric frustum (no eye-position shear -> close objects
    // shift more on screen than far ones, like a regular 3D camera), FOV scaled by a
    // fixed reference distance dRef captured when hybrid mode was entered. Using a
    // fixed dRef (not live d) keeps all walls' FOVs constant during lateral head
    // motion, so front and side walls react consistently. Top/bottom cameras render
    // their assigned vertical half via the b/t offset.
    // extendLeft / extendRight (HybridSeamless): if true, that edge's half-span is
    // forced to n*tan(neighborHalfRad) so the frustum ray at that edge meets the
    // neighbouring wall's frustum ray exactly at the physical corner - closing the
    // missing-wedge gap. False = keep the natural planeWidth*0.5*n/dRef span on that
    // side (no distortion on edges with no neighbour). For non-seamless modes, pass
    // extendLeft=extendRight=false and the matrix collapses to symmetric Hybrid /
    // HybridWide behaviour.
    static Matrix4x4 getHybridProjMatrix(Vector3 pa, Vector3 pb, Vector3 pc, Vector3 pe, Camera theCam, float dRef, float widenFactor,
                                         bool extendLeft, bool extendRight, float neighborHalfRad)
    {
        Vector3 vr = (pb - pa).normalized;
        Vector3 vu = (pc - pa).normalized;
        Vector3 vn = Vector3.Cross(vr, vu).normalized;

        Vector3 va = pa - pe;
        Vector3 vc = pc - pe;

        float n = theCam.nearClipPlane;
        float f = theCam.farClipPlane;
        float dLive = Vector3.Dot(va, vn);
        // Fallback to live d if caller didn't supply a positive reference.
        float dFov = dRef > 0f ? dRef : dLive;
        if (float.IsNaN(dFov) || float.IsInfinity(dFov) || dFov <= 0.01f) dFov = 1.0f; // guard
        float planeWidth = Vector3.Dot(vr, pb - pa); // full width of the wall

        // Vertical: TRUE off-axis placement from THIS camera's own mesh corners relative
        // to the eye, so the horizon sits at the participant's REAL eye height instead of
        // being pinned to the physical screen seam (the cause of the "everything sits a
        // little low" shift). The n/dFov cancels in the m[1,2] centre term below, so the
        // vertical PLACEMENT matches OffAxis exactly; the m[1,1] SCALE still uses the
        // fixed dRef, so there is no vertical zoom/breathe with depth. Adjacent top/bottom
        // halves share the split edge (same eye, same plane distance), so the seam stays
        // seamless and no top/bottom classification is needed.
        float b = Vector3.Dot(vu, va) * n / dFov; // bottom edge elevation vs eye
        float t = Vector3.Dot(vu, vc) * n / dFov; // top edge elevation vs eye

        // Horizontal frustum, per edge.
        //   default (no neighbour on that edge): planeWidth*0.5*n/dRef, widened by widenFactor.
        //   neighboured edge (HybridSeamless):   n*tan(gamma/2), the angle that makes this
        //   wall's edge ray hit the same world ray as the neighbour wall's matching edge
        //   ray - closing the corner gap exactly.
        float baseHalf  = planeWidth * 0.5f * n / dFov * widenFactor;
        float cornerHalf = n * Mathf.Tan(neighborHalfRad);
        float lHalf = extendLeft  ? cornerHalf : baseHalf;
        float rHalf = extendRight ? cornerHalf : baseHalf;
        float l = -lHalf;
        float r = +rHalf;

        Matrix4x4 m = new Matrix4x4();
        m[0, 0] = 2.0f * n / (r - l);
        m[0, 1] = 0;
        m[0, 2] = (r + l) / (r - l); // 0 if both sides symmetric; non-zero if only one edge extended
        m[0, 3] = 0;
        m[1, 0] = 0;
        m[1, 1] = 2.0f * n / (t - b);
        m[1, 2] = (t + b) / (t - b); // preserves top/bottom wall slice
        m[1, 3] = 0;
        m[2, 0] = 0;
        m[2, 1] = 0;
        m[2, 2] = -(f + n) / (f - n);
        m[2, 3] = (-2.0f * f * n) / (f - n);
        m[3, 0] = 0;
        m[3, 1] = 0;
        m[3, 2] = -1.0f;
        m[3, 3] = 0;
        return m;
    }

    static Matrix4x4 getAsymProjMatrix(Vector3 pa, Vector3 pb, Vector3 pc, Vector3 pe, Camera theCam, float horizBlend = 1f)
    {
        Vector3 vu, vr, vn, va, vb, vc;
        float l, r, b, t;

        //compute orthonormal basis for the screen - could pre-compute this...
        vr = (pb - pa).normalized;
        vu = (pc - pa).normalized;
        vn = Vector3.Cross(vr, vu).normalized;

        //compute screen corner vectors
        va = pa - pe;
        vb = pb - pe;
        vc = pc - pe;

        //find the distance from the eye to screen plane
        float n = theCam.nearClipPlane;
        float f = theCam.farClipPlane;
        float d = Vector3.Dot(va, vn); // distance from eye to screen

        // Guard against inverted matrix if eye passes through or behind the screen plane.
        if (float.IsNaN(d) || float.IsInfinity(d) || d <= 0.01f)
        {
            Vector3 screenCenter = pa + ((pb - pa) * 0.5f) + ((pc - pa) * 0.5f);
            Vector3 fallbackPos = screenCenter - (vn * 1.5f);
            va = pa - fallbackPos;
            vb = pb - fallbackPos;
            vc = pc - fallbackPos;
            d = Vector3.Dot(va, vn);
            if (d <= 0.01f) d = 1.0f;
        }

        l = Vector3.Dot(vr, va) * n / d;
        r = Vector3.Dot(vr, vb) * n / d;
        b = Vector3.Dot(vu, va) * n / d;
        t = Vector3.Dot(vu, vc) * n / d;

        //put together the matrix
        //from http://forum.unity3d.com/threads/using-projection-matrix-to-create-holographic-effect.291123/
        // horizBlend in [0,1] scales the off-axis horizontal shear; 1 = full off-axis
        // CAVE, 0 = symmetric horizontal (in-game feel). Used by Partial to blend
        // between CAVE and in-game.
        Matrix4x4 m = new Matrix4x4();
        m[0, 0] = 2.0f * n / (r - l);
        m[0, 1] = 0;
        m[0, 2] = horizBlend * (r + l) / (r - l);
        m[0, 3] = 0;
        m[1, 0] = 0;
        m[1, 1] = 2.0f * n / (t - b);
        m[1, 2] = (t + b) / (t - b);
        m[1, 3] = 0;
        m[2, 0] = 0;
        m[2, 1] = 0;
        m[2, 2] = -(f + n) / (f - n);
        m[2, 3] = (-2.0f * f * n) / (f - n);
        m[3, 0] = 0;
        m[3, 1] = 0;
        m[3, 2] = -1.0f;
        m[3, 3] = 0;

        return m;
    }
}
