using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(TerrainFromSpline))]
public class TerrainFromSplineInspector : Editor
{
    private TerrainFromSpline tfs;

    public override void OnInspectorGUI()
    {
        tfs = target as TerrainFromSpline;

        DrawDefaultInspector();

        if (GUILayout.Button("Recalculate Terrain"))
        {
            // Undo.RecordObject(Terrain, "Recalculated Terrain From Spline");
            tfs.RecalculateTerrain();
        }
    }

}
