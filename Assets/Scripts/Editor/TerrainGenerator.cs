using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

/// <summary>Editor utility that bakes the ground mesh from the walking spline.</summary>
public class TerrainGenerator : EditorWindow
{
    
    private float hillHeight = 5;
    private float valleyDepth = -5;
    private float cornerTilt = 10;

    private GameObject waypoints;
    
    [MenuItem(("Window/TerrainGenerator"))]
    public static void ShowWindow()
    {
        GetWindow((typeof(TerrainGenerator)));
    }

    public void OnGUI()
    {
        GUILayout.Label("Terrain Settings", EditorStyles.boldLabel);

        hillHeight  = EditorGUILayout.Slider("Hill Height", hillHeight, -10, 10);
        valleyDepth = EditorGUILayout.Slider("Valley Depth", valleyDepth, -10, 10);
        cornerTilt = EditorGUILayout.Slider("Corner Tilt", cornerTilt, -10, 10);

        waypoints = (GameObject) EditorGUILayout.ObjectField("Waypoints", waypoints, typeof(GameObject), true);

        if (GUILayout.Button("Create Terrain"))
        {
            Debug.Log("Halleluja!");
        }
    }
}
