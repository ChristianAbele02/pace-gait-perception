
using System.IO;
using UnityEditor;
using UnityEngine;

// as per: http://catlikecoding.com/unity/tutorials/curves-and-splines/

[CustomEditor(typeof(BezierSpline))]
public class BezierSplineInspector : Editor
{
    private BezierSpline spline;
    private Transform handleTransform;
    private Quaternion handleRotation;

    private const float directionScale = 2.0f;
    private const int stepsPerCurve = 10;

    private const float handleSize = 0.04f;
    private const float pickSize = 0.06f;

    private int selectedIndex = -1;

    private static Color[] modeColors =
    {
        new Color(0.11f, 0.62f, 0.47f),
        new Color(0.85f, 0.37f, 0.01f),
        new Color(0.46f, 0.44f, 0.7f)
    };

    private void OnSceneGUI()
    {
        spline = target as BezierSpline;
        handleTransform = spline.transform;
        handleRotation = Tools.pivotRotation == PivotRotation.Local ? handleTransform.rotation : Quaternion.identity;

        var p0 = ShowPoint(0);
        for (var i = 1; i < spline.ControlPointCount; i += 3)
        {
            var p1 = ShowPoint(i + 0);
            var p2 = ShowPoint(i + 1);
            var p3 = ShowPoint(i + 2);

            Handles.color = Color.gray;
            Handles.DrawLine(p0, p1);
            Handles.DrawLine(p2, p3);

            Handles.DrawBezier(p0, p3, p1, p2, Color.white, null, 3f);
            p0 = p3;
        }

        ShowDirections();
    }

    private void ShowDirections() {
        Handles.color = Color.white;
        var point = spline.GetPointT(0f);
        
        Handles.DrawLine(point, point + spline.GetDirection(0f) * directionScale);
        int steps = stepsPerCurve * spline.CurveCount;
        for (int i = 1; i <= steps; i++)
        {
            point = spline.GetPointT(i / (float) steps);
            
            Handles.DrawAAPolyLine(3f,
                spline.GetPoint3DT(i / (float) steps, -directionScale, 0, -directionScale),
                point,
                spline.GetPoint3DT(i / (float) steps,  directionScale, 0, -directionScale)
                );
        }


    }


    private Vector3 ShowPoint(int index)
    {
        Vector3 point = handleTransform.TransformPoint(spline.GetControlPoint(index));
        float size = HandleUtility.GetHandleSize(point);
        
        // increase size for first point in spline
        if (index == 0) size *= 2f;

        Handles.color = modeColors[(int) spline.GetControlPointMode(index)];
        if (Handles.Button(point, handleRotation, size * handleSize, size * pickSize, Handles.DotCap))
        {
            selectedIndex = index;
            Repaint();
        }

        if (selectedIndex == index)
        {
            EditorGUI.BeginChangeCheck();
            point = Handles.DoPositionHandle(point, handleRotation);
            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(spline, "Move Point");
                EditorUtility.SetDirty(spline);
                spline.SetControlPoint(index, handleTransform.InverseTransformPoint(point));
            }
        }

        return point;
    }

    public override void OnInspectorGUI()
    {
        spline = target as BezierSpline;
        EditorGUILayout.LabelField("Length (approx)", spline.Length.ToString());
        
        EditorGUI.BeginChangeCheck();
        bool loop = EditorGUILayout.Toggle("Loop", spline.Loop);
        if (EditorGUI.EndChangeCheck())
        {
            Undo.RecordObject(spline, "Toggle Loop");
            EditorUtility.SetDirty(spline);
            spline.Loop = loop;
        }
        if (selectedIndex >= 0 && selectedIndex < spline.ControlPointCount)
        {
            DrawSelectedPointInspector();
        }
        if (GUILayout.Button("Add Curve"))
        {
            Undo.RecordObject(spline, "Add Curve");
            spline.AddCurve();
            EditorUtility.SetDirty(spline);
        }

        if (GUILayout.Button("Reset"))
        {
            Undo.RecordObject(spline, "Reset");
            spline.Reset();
            EditorUtility.SetDirty(spline);
        }


        if (GUILayout.Button("Export"))
        {
            // Ask where to save: the old hard-coded "/tmp/path.csv" silently landed in
            // C:\tmp on Windows, where the editor actually runs.
            var savePath = EditorUtility.SaveFilePanel(
                "Export spline path", "", "path.csv", "csv");
            if (!string.IsNullOrEmpty(savePath))
            {
                using (var file = new StreamWriter(savePath))
                {
                    file.WriteLine("distance,px,py,pz,rx,ry,rz");
                    for (float d = 0; d < spline.Length; d += 1)
                    {
                        var p = spline.GetPointT(d / spline.Length);
                        var q = spline.GetRotationT(d / spline.Length).eulerAngles;
                        file.WriteLine("{0},{1},{2},{3},{4},{5},{6}", d, p.x, p.y, p.z, q.x, q.y, q.z);
                    }
                }
                Debug.Log("Exported spline path to " + savePath);
            }
        }
        
        
    }

    private void DrawSelectedPointInspector()
    {
        GUILayout.Label("Selected Point");
        
        EditorGUI.BeginChangeCheck();
        Vector3 point = EditorGUILayout.Vector3Field("Position", spline.GetControlPoint(selectedIndex));
        if (EditorGUI.EndChangeCheck())
        {
            Undo.RecordObject(spline, "Move Point");
            EditorUtility.SetDirty(spline);
            spline.SetControlPoint(selectedIndex, point);
        }
        
        EditorGUI.BeginChangeCheck();
        BezierControlPointMode mode = (BezierControlPointMode)
            EditorGUILayout.EnumPopup("Mode", spline.GetControlPointMode(selectedIndex));
        if (EditorGUI.EndChangeCheck())
        {
            Undo.RecordObject(spline, "Change Point Mode");
            spline.SetControlPointMode(selectedIndex, mode);
            EditorUtility.SetDirty(spline);
        }
        
        EditorGUI.BeginChangeCheck();
        float tilt = EditorGUILayout.FloatField("Tilt", spline.GetControlPointTilt(selectedIndex));
        if (EditorGUI.EndChangeCheck())
        {
            Undo.RecordObject(spline, "Change Point Tilt");
            spline.SetControlPointTilt(selectedIndex, tilt);
            EditorUtility.SetDirty(spline);
        }

        if (GUILayout.Button("Remove Selected Curve"))
        {
            Undo.RecordObject(spline, "Remove Curve");
            spline.RemoveCurve(selectedIndex);
            EditorUtility.SetDirty(spline);
        }
        
        
    }
}