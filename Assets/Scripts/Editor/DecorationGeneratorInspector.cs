using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(DecorationGenerator))]
/// <summary>Inspector for DecorationGenerator: adds the scatter/clear scenery buttons.</summary>
public class DecorationGeneratorInspector : Editor
{
    public override void OnInspectorGUI()
    {
        var myTarget = (DecorationGenerator) target;

        DrawDefaultInspector();

        if (GUILayout.Button("Generate Hurdles"))
        {
            Undo.RecordObject(myTarget, "GenerateHurdles");
            EditorUtility.SetDirty(myTarget);
        }

        if (GUILayout.Button("Generate Waystones"))
        {
            Undo.RecordObject(myTarget, "GenerateWaystones");
            myTarget.generateWaystones();
            EditorUtility.SetDirty(myTarget);
        }

        if (GUILayout.Button("Generate Vegetation"))
        {
            Undo.RecordObject(myTarget, "GenerateVegetation");
            myTarget.generateTrees();
            myTarget.generateBushes();
            EditorUtility.SetDirty(myTarget);
            
        }

    }
}
