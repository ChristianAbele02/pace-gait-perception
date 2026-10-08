using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;
using System.Linq;

/// <summary>
/// Editor helper that lays a Bezier spline out as a closed circle — the default walking
/// path. A closed loop means a block never runs out of path, so block length is set by
/// trial count rather than by geometry.
/// </summary>
public class CirclePathGenerator : MonoBehaviour
{
    public BezierSpline spline;
    
    public float a = 10f;

    public float spiralLength = 10f;
    public int numPoints = 100;

    public float maxHeight = 0f;
    public float maxRandomHeight = 0f;
    
    public float tangentScale = 5f;

    public void ApplyPath()
    {
        var points = new List<Vector3>();
        var tilts = new List<float>();
        var modes = new List<BezierControlPointMode>();

        Vector3[] targetPoints =
        {
            new Vector3(100, 0, 100), 
            new Vector3(100, 0, 200),
            new Vector3(100, 0, 300),
            new Vector3(200, 0, 300), 
            new Vector3(200, 0, 200),
            new Vector3(200, 0, 100),
            new Vector3(300, 0, 100), 
            new Vector3(300, 0, 200),
            new Vector3(300, 0, 300),
        };
        var targetAngles = new List<float> {0, 0, 0, 0, 0};
        
        // convert degrees to rad and shuffle so that every participants has a unique experience
        var shuffledAngles = targetAngles.OrderBy(x => Random.value).ToList();
        
        // add flat tracks in begining and end
        shuffledAngles.Insert(0, 0f);
        shuffledAngles.Add(0f);
        shuffledAngles.Add(0f);


        var pointOffset = 0.1f;
        var manipulatorOffset = 0.05f;
        
        // declare some commonly used variable names (just to satisfy the compiler -.-)
        Vector3 p;
        
        // set heights of target Points
        // (skip first point)
        for (var i = 1; i < targetPoints.Length; i++)
        {
            // point and previous point
            p = targetPoints[i];
            
            // set new target point
            targetPoints[i] = new Vector3(p.x, 0f, p.z);
        }

        // first point is special
        p = targetPoints[0];
        var m0 = targetPoints[0] + (targetPoints[1] - targetPoints[0]) * manipulatorOffset;
        
        // add first point
        modes.Add(BezierControlPointMode.Free);
        points.Add(p);
        tilts.Add(0f);
        
        // add first manipulator
        points.Add(m0);
        tilts.Add(0f);

        // add points
        for (var i = 1; i < targetPoints.Length - 1; i++)
        {
            // vectors from last and to next point
            var v0 = targetPoints[i] - targetPoints[i - 1];
            var v1 = targetPoints[i + 1] - targetPoints[i];
            
            // point positions
            var p0 = targetPoints[i - 1] + v0 * (1.0f - pointOffset);
            var p1 = targetPoints[i] + v1 * (0.0f + pointOffset);
            
            // manipulator positions
            var m00 = targetPoints[i-1] + v0 * (1.0f - pointOffset - manipulatorOffset);
            var m01 = targetPoints[i-1] + v0 * (1.0f - pointOffset + manipulatorOffset);
            var m10 = targetPoints[i] + v1 * (0.0f + pointOffset - manipulatorOffset);
            var m11 = targetPoints[i] + v1 * (0.0f + pointOffset + manipulatorOffset);

            // add points to output
            modes.Add(BezierControlPointMode.Free);
            points.Add(m00); 
            tilts.Add(0f);
            points.Add(p0); 
            tilts.Add(0f);
            points.Add(m01);
            tilts.Add(0f);
            
            modes.Add(BezierControlPointMode.Free);
            points.Add(m10); 
            tilts.Add(0f);
            points.Add(p1); 
            tilts.Add(0f);
            points.Add(m11);
            tilts.Add(0f);
        }
        
        // last point is special
        var lastIdx = targetPoints.Length - 1;
        p = targetPoints[lastIdx];
        m0 = targetPoints[lastIdx - 1] + (targetPoints[lastIdx] - targetPoints[lastIdx - 1]) * (1.0f - manipulatorOffset);
        
        // add last manipulator
        points.Add(m0);
        tilts.Add(0f);
        
        // add last point
        modes.Add(BezierControlPointMode.Free);
        points.Add(p);
        tilts.Add(0f);

        // apply
        spline.Set(
            points.ToArray(),
            tilts.ToArray(),
            modes.ToArray(),
            false
            );

    }

    public void Preview()
    {

        var points = GeneratePoints();

        var lastPoint = points[0];

        for (var i = 1; i < points.Count; i++)
        {

            var point = points[i];
            
            Debug.DrawLine(
                lastPoint,
                point,
                Color.red,
                10f
                );

            lastPoint = point;

        }

    }

    private List<Vector3> GeneratePoints()
    {
        var points = new List<Vector3>();
        
        for (var i = -numPoints; i < numPoints; i++)
        {
            var f = i / (float)numPoints;
            var theta = spiralLength * f;

            var r = Mathf.Sign(theta) * Mathf.Sqrt(Mathf.Abs(theta));
            
            var x = a * r * Mathf.Cos(theta);
            var y = 0f;
            var z = Mathf.Sign(theta) * a * r * Mathf.Sin(theta);
            
            points.Add(new Vector3(x, y, z));
        }

        return points;
    }
}
