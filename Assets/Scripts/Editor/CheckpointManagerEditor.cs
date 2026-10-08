using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(CheckpointManager))]
/// <summary>
/// Inspector for CheckpointManager: adds a "Generate Checkpoints" button so a trial
/// layout can be previewed in the editor without entering play mode. The layout is
/// seeded, so the preview is the same sequence the run will use.
/// </summary>
public class CheckpointManagerInspector : Editor
{
    public override void OnInspectorGUI()
    {
        var myTarget = (CheckpointManager)target;

        DrawDefaultInspector();

        if (GUILayout.Button("Generate Checkpoints"))
        {
            Undo.RecordObject(myTarget, "Generate Checkpoints");
            myTarget.GenerateCheckpoints();
            EditorUtility.SetDirty(myTarget);
        }
    }
}
