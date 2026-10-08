using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(CirclePathGenerator))]
/// <summary>Inspector for CirclePathGenerator: adds the "generate circular path" button.</summary>
public class CirclePathGeneratorInspector : Editor
{
    public override void OnInspectorGUI()
    {
        var myTarget = (CirclePathGenerator) target;

        DrawDefaultInspector();

        if (GUILayout.Button("Generate Track"))
        {
            Undo.RecordObject(myTarget, "GenerateTrack");
            myTarget.ApplyPath();
            EditorUtility.SetDirty(myTarget);
        }

        if (GUILayout.Button("Preview"))
        {
            myTarget.Preview();
        }

    }
}
