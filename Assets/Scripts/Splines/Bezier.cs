using UnityEngine;

/// <summary>
/// Cubic and quadratic Bezier evaluation (point and first derivative).
/// Adapted from Jasper Flick's Catlike Coding spline tutorial.
/// </summary>
public static class Bezier
{
    public static Vector3 GetPoint(Vector3 p0, Vector3 p1, Vector3 p2, float t)
    {
        t = Mathf.Clamp01(t);
        return (1f - t) * (1f - t) * p0 + 2f * (1f - t) * t * p1 + t * t * p2;
    }

    public static Vector3 GetFirstDerivative(Vector3 p0, Vector3 p1, Vector3 p2, float t)
    {
        return 2f * (1f - t) * (p1 - p0) + 2f * t * (p2 - p1);
    }


    public static Vector3 GetPoint(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        t = Mathf.Clamp01(t);
        float oneMinusT = 1f - t;
        return oneMinusT * oneMinusT * oneMinusT * p0 +
               3f * oneMinusT * oneMinusT * t * p1 +
               3f * oneMinusT * t * t * p2 +
               t * t * t * p3;
    }

    public static Vector3 GetFirstDerivative(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
    {
        t = Mathf.Clamp01(t);
        float oneMinusT = 1f - t;
        return
            3f * oneMinusT * oneMinusT * (p1 - p0) +
            6f * oneMinusT * t * (p2 - p1) +
            3f * t * t * (p3 - p2);
    }

    public static float GetCubic(float x0, float x1, float x2, float x3, float t)
    {
        t = Mathf.Clamp01(t);
        var oneMinusT = 1f - t;
        return oneMinusT * oneMinusT * oneMinusT * x0 +
               3f * oneMinusT * oneMinusT * t * x1 +
               3f * oneMinusT * t * t * x2 +
               t * t * t * x3;
    }
}