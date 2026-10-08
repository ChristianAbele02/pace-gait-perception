using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

[Serializable]
/// <summary>
/// The walking path: a chain of cubic Bezier curves with an arc-length parameterisation.
///
/// The arc-length table is what makes the paradigm's timing correct — sampling a Bezier
/// by its raw parameter t advances at a non-uniform SPEED, so trials placed on raw t
/// would not be evenly spaced in time. <c>Length</c> and the T-based accessors convert to
/// distance along the path, which is how CheckpointManager turns an inter-stimulus
/// interval in seconds into a spacing in metres.
/// Adapted from Jasper Flick's Catlike Coding spline tutorial.
/// </summary>
public class BezierSpline : MonoBehaviour
{
   [SerializeField]
   private Vector3[] points;

   [SerializeField] 
   private float[] tilts;

   [SerializeField]
   private BezierControlPointMode[] modes;

   [SerializeField] 
   private bool loop;

   private static int distanceSteps = 128;
   private float[] distances = {};
   private float length = 0f;


   public void Set(Vector3[] points, float[] tilts, BezierControlPointMode[] modes, bool loop)
   {
      this.points = points;
      this.tilts = tilts;
      this.modes = modes;
      this.loop = loop;
      for (var i = 0; i < this.points.Length; i++)
      {
         EnforceMode(i);
      }
      RecalculateDistances();
   }
   

   public bool Loop
   {
      get { return loop; }
      set
      {
         loop = value;
         if (!value) return;
         modes[modes.Length - 1] = modes[0];
         SetControlPoint(0, points[0]);
      }
   }

   public int ControlPointCount
   {
      get { return points.Length; }
   }

   public Vector3 GetControlPoint(int index)
   {
      return points[index];
   }

   public void SetControlPoint(int index, Vector3 point)
   {
      if (index % 3 == 0)
      {
         Vector3 delta = point - points[index];
         if (loop)
         {
            if (index == 0)
            {
               points[1] += delta;
               points[points.Length - 2] += delta;
               points[points.Length - 1] = point;
            }
            else if (index == points.Length - 1)
            {
               points[0] = point;
               points[1] += delta;
               points[index - 1] += delta;
            }
            else
            {
               points[index - 1] += delta;
               points[index + 1] += delta;
            }
            
         }
         else
         {
            if (index > 0)
            {
               points[index - 1] += delta;
            }

            if (index + 1 < points.Length)
            {
               points[index + 1] += delta;
            }
         }
      }
      points[index] = point;
      EnforceMode(index);
      RecalculateDistances();
   }

   public BezierControlPointMode GetControlPointMode(int index)
   {
      return modes[(index + 1) / 3];
   }

   public void SetControlPointMode(int index, BezierControlPointMode mode)
   {
      int modeIndex = (index + 1) / 3;
      modes[modeIndex] = mode;
      if (loop)
      {
         if (modeIndex == 0)
         {
            modes[modes.Length - 1] = mode;
         }
         else if (modeIndex == modes.Length - 1)
         {
            modes[0] = mode;
         }
      }
      EnforceMode(index);
      RecalculateDistances();
   }

   public float GetControlPointTilt(int index)
   {
      return tilts[(index + 1) / 3];
   }

   public void SetControlPointTilt(int index, float tilt)
   {
      int tiltIndex = (index + 1) / 3;
      tilts[tiltIndex] = tilt;
      if (loop)
      {
         if (tiltIndex == 0)
         {
            tilts[tilts.Length - 1] = tilt;
         }
         else if (tiltIndex == modes.Length - 1)
         {
            tilts[0] = tilt;
         }
      }
   }
   
   private void EnforceMode(int index)
   {
		
      int modeIndex = (index + 1) / 3;
      BezierControlPointMode mode = modes[modeIndex];

	  // skip enforcing if mode is free or if it's the first or last control point
	  if (mode == BezierControlPointMode.Free || modeIndex == 0 || modeIndex == modes.Length - 1)
      {
         return;
      }

      int middleIndex = modeIndex * 3;
      int fixedIndex, enforcedIndex;
      if (index <= middleIndex)
      {
         fixedIndex = middleIndex - 1;
         if (fixedIndex < 0) fixedIndex = points.Length - 2;
         enforcedIndex = middleIndex + 1;
         if (enforcedIndex >= points.Length) enforcedIndex = 1;
      }
      else
      {
         fixedIndex = middleIndex + 1;
         if (fixedIndex >= points.Length) fixedIndex = 1;
         enforcedIndex = middleIndex - 1;
         if (enforcedIndex < 0) enforcedIndex = points.Length - 2;
         
      }

      Vector3 middle = points[middleIndex];
      Vector3 enforcedTangent = middle - points[fixedIndex];
      if (mode == BezierControlPointMode.Aligned)
      {
         enforcedTangent = enforcedTangent.normalized * Vector3.Distance(middle, points[enforcedIndex]);
      }
      points[enforcedIndex] = middle + enforcedTangent;
      
   }

   public Vector3 GetPoint(float t)
   {
      int i;
      if (t >= 1f)
      {
         t = 1f;
         i = points.Length - 4;
      }
      else
      {
         t = Mathf.Clamp01(t) * CurveCount;
         i = (int) t;
         t -= i;
         i *= 3;
      }

      return transform.TransformPoint(Bezier.GetPoint(
         points[i], points[i + 1], points[i + 2], points[i + 3], t));
   }

   public Vector3 GetPoint3D(float t, float x, float y, float z)
   {
      var pos = GetPoint(t);
      var dir = GetDirection(t);
      var tilt = GetTilt(t);
      
      var right = Quaternion.AngleAxis(tilt, dir) * Vector3.Cross(Vector3.up, dir).normalized;
      var up   = Vector3.Cross(dir, right).normalized;

      return pos + x * right + y * up + z * dir;
   }

   public float GetTilt(float t)
   {
      int i;
      if (t >= 1f)
      {
         t = 1f;
         i = tilts.Length - 2;
      }
      else
      {
         t = Mathf.Clamp01(t) * CurveCount;
         i = (int) t;
         t -= i;
      }

      // Tilt is interpolated linearly between control points. (An abandoned cubic variant
      // used to sit here as unreachable code; it also over-ran the end of `tilts`.)
      return Mathf.Lerp(tilts[i], tilts[i + 1], t);
   }

   public Vector3 GetVelocity(float t)
   {
      int i;
      if (t >= 1f)
      {
         t = 1f;
         i = points.Length - 4;
      }
      else
      {
         t = Mathf.Clamp01(t) * CurveCount;
         i = (int) t;
         t -= i;
         i *= 3;
      }
      return transform.TransformPoint(
                Bezier.GetFirstDerivative(points[i], points[i + 1], points[i + 2], points[i + 3], t)) - transform.position;
   }

   public Vector3 GetVelocityT(float t)
   {
      return GetVelocity(mapT(t));
   }

   public Vector3 GetDirection(float t)
   {
      return GetVelocity(t).normalized;
   }

   public Vector3 GetDirectionT(float t)
   {
      return GetDirection(mapT(t));
   }

   public void Reset()
   {
      points = new[]
      {
         new Vector3(1f, 0f, 0f), 
         new Vector3(2f, 0f, 0f), 
         new Vector3(3f, 0f, 0f), 
         new Vector3(4f, 0f, 0f)
      };

      modes = new[]
      {
         BezierControlPointMode.Free,
         BezierControlPointMode.Free
      };

      tilts = new[]
      {
         0f,
         0f
      };
      
      RecalculateDistances();
   }

   public void AddCurve()
   {
      Vector3 point = points[points.Length - 1];
      
      Array.Resize(ref points, points.Length + 3);
      point.x += 1f;
      points[points.Length - 3] = point;
      point.x += 1f;
      points[points.Length - 2] = point;
      point.x += 1f;
      points[points.Length - 1] = point;
      
      Array.Resize(ref modes, modes.Length + 1);
      modes[modes.Length - 1] = modes[modes.Length - 2];
      EnforceMode(points.Length - 4);
      
      Array.Resize(ref tilts, tilts.Length + 1);
      tilts[tilts.Length - 1] = tilts[tilts.Length - 2];

      if (loop)
      {
         points[points.Length - 1] = points[0];
         modes[modes.Length - 1] = modes[0];
         tilts[tilts.Length - 1] = tilts[0];
         EnforceMode(0);
      } 
      
      RecalculateDistances();
   }

   public void RemoveCurve(int index)
   {
      if (points.Length == 0) return;

      int modeIndex = (index + 1) / 3;
      
      int middleIndex = modeIndex * 3; // rounding to multiple of 3

      
      // copy all elements after the deleted curve three elements to the front
      if (middleIndex == 0)
      {
         // special case first element
         Array.Copy(points, 3, points, 0, points.Length - 3);
         Array.Resize(ref points, points.Length - 3);
      } else if (middleIndex >= points.Length - 2)
      {
         // special case last element
         Array.Resize(ref points, points.Length - 3);
      }
      else
      {
         Array.Copy(points, middleIndex + 2, points, middleIndex - 1, points.Length - (middleIndex + 2));
         Array.Resize(ref points, points.Length - 3);
      }
      
      
      Array.Copy(modes, modeIndex, modes, modeIndex + 1, modes.Length - (modeIndex + 1));
      Array.Resize(ref modes, modes.Length - 1);
      
      Array.Copy(tilts, modeIndex, tilts, modeIndex + 1, tilts.Length - (modeIndex + 1));
      Array.Resize(ref tilts, tilts.Length - 1);
      
      if (loop)
      {
         points[points.Length - 1] = points[0];
         modes[modes.Length - 1] = modes[0];
         tilts[tilts.Length - 1] = tilts[0];
         EnforceMode(0);
      } 
      
      RecalculateDistances();
      
   }

   public int CurveCount
   {
      get { return (points.Length - 1) / 3; }
   }

   
   
   // length approximation as per:
   // https://gamedev.stackexchange.com/questions/5373/moving-ships-between-two-planets-along-a-bezier-missing-some-equations-for-acce/5427#5427
   public float Length
   {
      get 
      { 
          // --- FIX: Force calculation if it hasn't happened yet ---
          if (distances == null || distances.Length == 0 || length == 0f)
          {
              RecalculateDistances();
          }
          return length; 
      }
   }

   public float mapT(float d)
   {
      // make sure the distance array is populated
      if(distances.Length == 0) RecalculateDistances();
      
      var targetLength = d * distances[distances.Length - 1];

      var searchResult = Array.BinarySearch(distances, targetLength);
      var minIndex = searchResult < 0 ? ~searchResult - 1 : searchResult;
      var maxIndex = searchResult < 0 ? ~searchResult : searchResult;

      var f = Mathf.InverseLerp(distances[minIndex], distances[maxIndex], targetLength);

      var fIndex = Mathf.Lerp(minIndex, maxIndex, f) / (distances.Length - 1);

      return fIndex;
   }

   public Vector3 GetPointT(float t)
   {
      return GetPoint(mapT(t));
   }
   public Vector3 GetPoint3DT(float t, float x, float y, float z)
   {
      return GetPoint3D(mapT(t), x, y, z);
   }

   public Quaternion GetRotationT(float t)
   {
      // map into equidistant space
      var mappedT = mapT(t);
      
      // get forward direction and tilt
      var dir = GetDirection(mappedT);
      var tilt = GetTilt(mappedT);
      
      var right = Quaternion.AngleAxis(tilt, dir) * Vector3.Cross(Vector3.up, dir).normalized;
      var up   = Vector3.Cross(dir, right).normalized;

      return Quaternion.LookRotation(dir, up);

   }

   private void RecalculateDistances()
   {
      var clen = 0f;
      
      distances = new float[distanceSteps * CurveCount];
      
      // set initial
      var lastP = GetPoint(0f);
      distances[0] = 0;
      
      // iterate points
      for (var i = 1; i < distances.Length; i++)
      {
         var t = i / (float) (distances.Length - 1);
         var p = GetPoint(t);

         clen += Vector3.Distance(p, lastP);

         distances[i] = clen;


         lastP = p;
      }

      length = distances[distances.Length - 1];

   }

   public Vector3 ClosestPoint(Vector3 a)
   {
      // first step: find closest control point:
      var goodGuess = points.OrderBy(b => Vector3.Distance(a, b)).First();

      var goodGuessIdx = Array.IndexOf(points, goodGuess);
      var goodGuessD = distances[goodGuessIdx];
      var goodGuessF = goodGuessD / length;


      var lastGuess = goodGuessF;
      var thisGuess = goodGuessF;

      do
      {
         lastGuess = thisGuess;
         
         var upperF = lastGuess + 0.01f;
         var lowerF = lastGuess - 0.01f;

         var upper = GetPointT(upperF);
         var lower = GetPointT(lowerF);
         
         var upperD = Vector3.Distance(a, upper);
         var lowerD = Vector3.Distance(a, lower);
         
         float[] choices = {lastGuess, upperD, lowerD};
         thisGuess = choices.OrderBy(f => Vector3.Distance(a,GetPointT(f))).First();



      } while(!Mathf.Approximately(lastGuess, thisGuess));


      return GetPointT(thisGuess);


   }
}
