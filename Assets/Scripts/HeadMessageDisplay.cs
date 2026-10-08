using UnityEngine;

/// <summary>
/// A world-space text message shown at head height, in front of the participant,
/// on every CAVE node. Used for the pre-experiment baseline notice, the
/// "treadmill will stop soon" warning, and the cool-down notice.
///
/// Mirrors how <see cref="FollowerPlane"/> shows the Gabor: it lives in world space
/// and is driven over the network (via NetworkingSync.BroadcastShowMessage /
/// BroadcastHideMessage -> RPCMode.All), so each render node displays it on its own
/// screens. The object and its TextMesh are created at runtime (no scene editing
/// needed); it is recreated per scene as required.
///
/// Legibility &amp; look: white glyphs alone wash out against the bright nature
/// environment, and the messages sat far enough away (2 m) to read small. The text
/// now floats closer and is drawn on top of a tidy rounded-rectangle panel - a soft
/// dark slate fill with anti-aliased rounded corners and a thin accent frame - so it
/// is clearly legible and looks designed rather than a black blob. The panel texture
/// is generated procedurally at runtime (no art assets) and 9-sliced so the corners
/// stay crisp at any message width.
/// </summary>
public class HeadMessageDisplay : MonoBehaviour
{
    private static HeadMessageDisplay _instance;
    public static HeadMessageDisplay Instance
    {
        get
        {
            // Unity's overloaded == treats a destroyed object as null, so this also
            // re-creates the display after a scene change.
            if (_instance == null)
            {
                GameObject go = new GameObject("HeadMessageDisplay");
                _instance = go.AddComponent<HeadMessageDisplay>();
            }
            return _instance;
        }
    }

    [Tooltip("Metres in front of the head the message floats. Closer = larger/more "
        + "readable; kept modest so it does not crowd the participant.")]
    public float distance = 1.5f;
    [Tooltip("Vertical offset from head height (m); 0 = at eye level.")]
    public float heightOffset = 0.0f;
    [Tooltip("World size of a character (TextMesh characterSize).")]
    public float characterSize = 0.009f;
    public int fontSize = 200;
    [Tooltip("If the text appears mirrored/back-to-front on the CAVE walls, toggle this.")]
    public bool flipFacing = false;

    [Header("Backing panel (legibility & look)")]
    [Tooltip("Draw a rounded panel behind the text so it stays legible against the "
        + "bright environment and looks tidy.")]
    public bool showBackground = true;
    [Tooltip("Panel fill colour and opacity (alpha). A dark slate at ~0.72 alpha reads "
        + "well over both the sky and the ground.")]
    public Color backgroundColor = new Color(0.07f, 0.10f, 0.14f, 0.72f);
    [Tooltip("Thin accent frame drawn around the panel edge.")]
    public Color frameColor = new Color(0.50f, 0.68f, 0.82f, 0.95f);
    [Tooltip("Corner rounding radius, in world metres.")]
    public float cornerRadius = 0.045f;
    [Tooltip("Accent frame thickness, in world metres.")]
    public float frameThickness = 0.006f;
    [Tooltip("Extra panel margin around the text, in world metres (x = sides, y = top/bottom).")]
    public Vector2 backgroundPadding = new Vector2(0.32f, 0.20f);

    private TextMesh textMesh;
    private MeshRenderer meshRenderer;
    private Transform head;
    private bool visible = false;

    // Rounded backing panel (child sprite), sized to the text each frame it is shown.
    private Transform background;
    private SpriteRenderer backgroundRenderer;

    // Procedural panel texture geometry. The texture is a fixed small bitmap; the
    // sprite border (for 9-slicing) and the pixels-per-unit map it to world size.
    private const int TEX = 96;          // panel texture resolution (square)
    private const float PPU = 600f;      // pixels per world metre for the sprite
    private float borderPx;              // 9-slice corner size in texture pixels

    void Awake()
    {
        if (_instance == null) { _instance = this; }
        else if (_instance != this) { Destroy(gameObject); return; }
        BuildText();
        BuildBackground();
        Hide();
    }

    private void BuildText()
    {
        textMesh = GetComponent<TextMesh>();
        if (textMesh == null) textMesh = gameObject.AddComponent<TextMesh>();
        textMesh.text = "";
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.fontSize = fontSize;
        textMesh.characterSize = characterSize;
        textMesh.color = Color.white;
        textMesh.font = Resources.GetBuiltinResource<Font>("Arial.ttf");

        // Adding a TextMesh auto-adds a MeshRenderer, but be defensive.
        meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer == null) meshRenderer = gameObject.AddComponent<MeshRenderer>();
        if (textMesh.font != null) meshRenderer.sharedMaterial = textMesh.font.material;
    }

    // Build the rounded panel as a child SpriteRenderer using a procedurally generated,
    // 9-sliced sprite so the corners and frame stay crisp at any size.
    private void BuildBackground()
    {
        if (background != null) return;

        GameObject panel = new GameObject("MessageBackground");
        background = panel.transform;
        background.SetParent(transform, false);
        background.localRotation = Quaternion.identity;
        // Sit just behind the glyphs (text +Z points away from the viewer) so the
        // text always draws in front of the panel.
        background.localPosition = new Vector3(0f, 0f, 0.02f);

        backgroundRenderer = panel.AddComponent<SpriteRenderer>();
        backgroundRenderer.sprite = BuildPanelSprite();
        backgroundRenderer.drawMode = SpriteDrawMode.Sliced;   // 9-slice: corners stay fixed
        backgroundRenderer.color = Color.white;                // colour is baked into the texture
    }

    // Generate the rounded-rectangle panel texture (dark fill + anti-aliased rounded
    // corners + thin accent frame) and wrap it in a 9-sliced sprite.
    private Sprite BuildPanelSprite()
    {
        float radiusPx = Mathf.Clamp(cornerRadius * PPU, 4f, TEX * 0.5f - 2f);
        float framePx = Mathf.Max(0f, frameThickness * PPU);
        borderPx = Mathf.Min(radiusPx + framePx, TEX * 0.5f - 1f);
        const float feather = 1.5f;  // edge anti-alias width (px)

        var tex = new Texture2D(TEX, TEX, TextureFormat.RGBA32, false)
        {
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };

        Color fill = backgroundColor;
        Color line = frameColor;
        var px = new Color[TEX * TEX];

        for (int y = 0; y < TEX; y++)
        {
            for (int x = 0; x < TEX; x++)
            {
                float cx = x + 0.5f;
                float cy = y + 0.5f;
                // Distance to the nearest panel edge, with rounded corners.
                float fx = Mathf.Min(cx, TEX - cx);
                float fy = Mathf.Min(cy, TEX - cy);
                float edge;
                if (fx < radiusPx && fy < radiusPx)
                {
                    float dx = radiusPx - fx;
                    float dy = radiusPx - fy;
                    edge = radiusPx - Mathf.Sqrt(dx * dx + dy * dy);
                }
                else
                {
                    edge = Mathf.Min(fx, fy);
                }

                // Outside-the-rounded-shape -> transparent, with a feathered edge.
                float coverage = Mathf.Clamp01(edge / feather);
                // Within framePx of the boundary -> accent frame colour.
                float frameMix = (framePx <= 0f) ? 0f
                    : Mathf.Clamp01(1f - (edge - framePx) / feather);

                Color baseCol = Color.Lerp(fill, line, frameMix);
                px[y * TEX + x] = new Color(baseCol.r, baseCol.g, baseCol.b, baseCol.a * coverage);
            }
        }

        tex.SetPixels(px);
        tex.Apply(false, false);

        var rect = new Rect(0, 0, TEX, TEX);
        var pivot = new Vector2(0.5f, 0.5f);
        var border = new Vector4(borderPx, borderPx, borderPx, borderPx);
        return Sprite.Create(tex, rect, pivot, PPU, 0, SpriteMeshType.FullRect, border);
    }

    private Transform ResolveHead()
    {
        if (head != null) return head;
        if (FollowerPlane.instance != null && FollowerPlane.instance.headTransform != null)
            head = FollowerPlane.instance.headTransform;
        else if (Camera.main != null)
            head = Camera.main.transform;
        return head;
    }

    public void ShowMessage(string message)
    {
        if (textMesh == null) BuildText();
        if (background == null) BuildBackground();
        textMesh.text = message;
        if (meshRenderer != null) meshRenderer.enabled = true;
        if (backgroundRenderer != null) backgroundRenderer.enabled = showBackground;
        visible = true;
        PositionInFront();
        SizeBackground();
    }

    public void Hide()
    {
        visible = false;
        if (meshRenderer != null) meshRenderer.enabled = false;
        if (backgroundRenderer != null) backgroundRenderer.enabled = false;
    }

    void LateUpdate()
    {
        if (visible)
        {
            PositionInFront();
            SizeBackground();
        }
    }

    private void PositionInFront()
    {
        Transform h = ResolveHead();
        if (h == null) return;

        // Place the text a fixed distance ahead along the head's horizontal heading,
        // at eye level, so it stays centred in the participant's view.
        Vector3 fwd = h.forward; fwd.y = 0f;
        if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
        fwd.Normalize();

        Vector3 pos = h.position + fwd * distance;
        pos.y = h.position.y + heightOffset;
        transform.position = pos;

        // Billboard toward the participant. TextMesh reads correctly from one side;
        // if it looks reversed on the walls, set flipFacing in the Inspector.
        Vector3 look = flipFacing ? (h.position - pos) : (pos - h.position);
        transform.rotation = Quaternion.LookRotation(look, Vector3.up);
    }

    // Resize the backing panel to wrap the current text plus padding. The TextMesh
    // mesh has no thickness and is held upright (LookRotation up = world up), so the
    // world AABB gives the true height directly and the horizontal spread gives the
    // width. The sprite is 9-sliced, so we set its world size directly (the rounded
    // corners/frame keep their proportions). Clamp so corners never overlap.
    private void SizeBackground()
    {
        if (background == null || backgroundRenderer == null || !showBackground) return;
        if (meshRenderer == null || textMesh == null || string.IsNullOrEmpty(textMesh.text)) return;

        Bounds b = meshRenderer.bounds;
        float width = Mathf.Sqrt(b.size.x * b.size.x + b.size.z * b.size.z) + backgroundPadding.x;
        float height = b.size.y + backgroundPadding.y;

        float minDim = 2f * (borderPx / PPU) + 0.002f;
        width = Mathf.Max(width, minDim);
        height = Mathf.Max(height, minDim);

        backgroundRenderer.size = new Vector2(width, height);
    }
}
