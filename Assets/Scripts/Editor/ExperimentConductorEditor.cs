using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Adds a "Randomize Path" button to the ExperimentConductor inspector, which rebuilds the
/// walking route: a fresh random spline, terrain re-fitted to it, and scenery re-scattered.
/// Editor-only — nothing here runs during an experimental session.
/// </summary>
[CustomEditor(typeof(ExperimentConductor))]
public class ExperimentConductorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        if (GUILayout.Button("Randomize Path"))
        {
            var pathObject = GameObject.Find("Path");

            Undo.RecordObject(pathObject, "RandomizePath");

            // Clear the previous route wholesale. The generators below re-create every child
            // from scratch, so removing them all is simpler and safer than tracking which
            // objects belong to which generator.
            for (var i = pathObject.transform.childCount - 1; i >= 0; i--)
                DestroyImmediate(pathObject.transform.GetChild(i).gameObject);

            // Generate a new random path.
            var cpg = pathObject.GetComponent<CirclePathGenerator>();
            cpg.ApplyPath();

            // Fit the terrain to it.
            var tfs = pathObject.GetComponent<TerrainFromSpline>();
            tfs.RecalculateTerrain();

            // Re-scatter the scenery.
            var dc = pathObject.GetComponent<DecorationGenerator>();
            dc.generateWaystones();
            dc.generateTrees();
            dc.generateBushes();

            EditorSceneManager.MarkAllScenesDirty();
        }
    }
}
