using UnityEngine;

/// <summary>
/// Moves an object along a <see cref="BezierSpline"/> at a set speed, exposing normalised
/// progress. In the walking condition this is what carries the participant's viewpoint
/// down the path, and its progress is logged as <c>distance-on-path</c>.
/// Adapted from Jasper Flick's Catlike Coding spline tutorial.
/// </summary>
public class SplineWalker : MonoBehaviour
{
   public BezierSpline spline;
   public CheckpointManager checkpointManager;

   [Tooltip("Speed in m/s")]
   public float speed;
   
   public bool lookForward = true;

   [Tooltip("Constant offset from path in m")]
   public Vector3 offset;

   [Tooltip("Progress in m")]
   public float progress;
   
   [Tooltip("Pitch in deg")]
   public float pitch;


   public SplineWalkerMode mode = SplineWalkerMode.Once;
   
   public bool isMovementPaused = false;
   private bool isFinishing = false;
   private bool warnedStopping = false;   // "treadmill will stop soon" fired once

   public enum SplineWalkerMode
   {
      Once,
      Loop
   }
   
   private void Start()
   {
      // Pre-initialise position and rotation before the first Update so there is no
      // one-frame snap when isMovementPaused flips to false on experiment start.
      if (spline == null) return;
      float f0 = (spline.Length > 0f) ? Mathf.Clamp01(progress / spline.Length) : 0f;
      transform.position = spline.GetPoint3DT(f0, offset.x, offset.y, offset.z);
      if (lookForward)
         transform.rotation = spline.GetRotationT(f0);
   }

   private void Update()
   {
      // Update logic should run on all nodes
      if (isMovementPaused)
      {
		  return;
	  }

      if (Input.GetKeyDown(KeyCode.KeypadPlus)) speed += 0.1f;
      if (Input.GetKeyDown(KeyCode.KeypadMinus)) speed -= 0.1f;

      speed = Mathf.Max(0.0f, Mathf.Min(3f, speed));

      progress += Time.deltaTime * speed;

      var f = progress / spline.Length;

      // Warn the participant shortly before the end of the path that the treadmill is
      // about to stop (fires once; head node broadcasts the message).
      if (!warnedStopping && checkpointManager != null && spline.Length > 0f)
      {
         if (spline.Length - progress <= checkpointManager.treadmillWarnDistanceM)
         {
            warnedStopping = true;
            checkpointManager.WarnTreadmillStopping();
         }
      }

      // handle overflow
      if (f >= 1f)
        {
            switch (mode)
            {
                case SplineWalkerMode.Once:
                    progress = spline.Length;
                    f = 1f;
                    
                    if (!isFinishing)
                    {
                        isFinishing = true;
                        if (checkpointManager != null)
                        {
                            // Tell the CheckpointManager to start the cooldown
                            checkpointManager.StartCooldown();
                        }
                    }
                    break;
                    
                case SplineWalkerMode.Loop:
                    progress -= spline.Length;
                    f -= 1f;
                    break;
                
            }
        }

      // set position from spline
      transform.position = spline.GetPoint3DT(f, offset.x, offset.y, offset.z);
      
      // set direction from spline
      if (lookForward)
      {
         transform.rotation = spline.GetRotationT(f);
      }
   }
}
