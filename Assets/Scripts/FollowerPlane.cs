using UnityEngine;

[RequireComponent(typeof(Renderer))]
public class FollowerPlane : MonoBehaviour
{
    public static FollowerPlane instance;

    [Header("Visual Task Settings")]
    public Color defaultColor = Color.grey;
    public Color taskColor = Color.red;
    
    [Header("Positioning & Size")]
    public Transform headTransform;
    public float taskScale = 0.5f;

    [Header("Gabor Settings")]
    // Davidson-comparable Gabor: 3 cpd grating, Gaussian envelope σ = 0.70°, on the
    // 0.7 m quad at 2 m (the quad subtends ~19.85°). The VISIBLE patch is shrunk to a
    // small disk (gaborDiskRadius) that tightly frames the Gabor — so it reads as a
    // small Gabor on a small grey disk (close to Davidson's HMD presentation) instead
    // of a tiny Gabor lost in a 19.85° grey disk. The quad mesh is unchanged, so the
    // 3 cpd / 0.70° calibration is preserved exactly. See README ("Stimulus revision")
    // and Analysis/gabor_preview.py.
    public float gaborFrequency = 59.6f;     // cycles across the quad = 3 cpd × 19.85°
    public float gaborSigma = 0.0353f;       // Gaussian SD in quad-width units = 0.70° / 19.85°
    public float gaborDiskRadius = 0.22f;    // visible disk radius (UV, 0–0.5); diameter° ≈ 39.7×r, so 0.22 ≈ an 8.7° patch
    public float gaborContrast = 0.7f;       // Michelson contrast (Davidson et al. 2024)

    [Header("Audio Feedback Settings")]
    public AudioClip taskSound;
    private AudioSource audioSource;

    // Internal reference
    private Renderer planeRenderer;
    private MaterialPropertyBlock propBlock;
    private int rotationPropertyID;
    private int frequencyPropertyID;
    private int sigmaPropertyID;
    private int diskRadiusPropertyID;
    private int contrastPropertyID;

    public bool isTaskVisible = false;

    void Awake()
    {
        if (instance == null)
            instance = this;
        else
            Destroy(gameObject);
    }


    void Start()
    {
        // Get the Renderer component
        planeRenderer = GetComponent<Renderer>();
        
        propBlock = new MaterialPropertyBlock();
        rotationPropertyID = Shader.PropertyToID("_Rotation");
        frequencyPropertyID = Shader.PropertyToID("_Frequency");
        sigmaPropertyID = Shader.PropertyToID("_Sigma");
        diskRadiusPropertyID = Shader.PropertyToID("_DiskRadius");
        contrastPropertyID = Shader.PropertyToID("_Contrast");

        // Audio Setup
        audioSource = gameObject.AddComponent<AudioSource>();
        audioSource.playOnAwake = false;
        audioSource.spatialBlend = 0f; //0f = 2D Sound

        // Pre-warm shader for lower GPU usage later
        planeRenderer.enabled = true;
        planeRenderer.enabled = false;
        isTaskVisible = false;
    }

    public void ShowTask(int direction, float angleMagnitude)
    {
        if (planeRenderer != null)
        {            
            float rotationAngle = (direction == 0) ? -angleMagnitude : angleMagnitude;
            
            planeRenderer.GetPropertyBlock(propBlock);
            propBlock.SetFloat(rotationPropertyID, rotationAngle);
            propBlock.SetFloat(frequencyPropertyID, gaborFrequency);
            propBlock.SetFloat(sigmaPropertyID, gaborSigma);
            propBlock.SetFloat(diskRadiusPropertyID, gaborDiskRadius);
            propBlock.SetFloat(contrastPropertyID, gaborContrast);
            planeRenderer.SetPropertyBlock(propBlock);
            
            // Ensure visible
            planeRenderer.enabled = true;
            isTaskVisible = true;
                        
            if (taskSound != null && audioSource != null)
            {
                audioSource.PlayOneShot(taskSound);
            }
        }
    }

    public void ResetTask()
    {
        if (planeRenderer != null)
        {
            planeRenderer.enabled = false;           
            isTaskVisible = false;
        }
    }
}
