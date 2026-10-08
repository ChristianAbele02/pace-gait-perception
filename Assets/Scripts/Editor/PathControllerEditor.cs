using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(PathController))]
public class PathControllerEditor : Editor
{
	public override void OnInspectorGUI()
	{

		DrawDefaultInspector ();

// 		if (GUILayout.Button ("Reset Hurdles")) {
// 			foreach (Transform h in ((PathController)target).GetHurdles().transform) {
// 				HurdleController c = h.gameObject.GetComponent<HurdleController> ();
// 			}
// //			HurdleController c = ((PathController)target).GetComponent<HurdleController> ();
// 		}

	}
}

