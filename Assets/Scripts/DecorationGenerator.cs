using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

/// <summary>
/// Scatters scenery props along the path. Purely visual: it gives optic flow something to
/// act on so walking feels like travel, and it touches nothing the experiment measures.
/// </summary>
public class DecorationGenerator : MonoBehaviour
{

	public BezierSpline spline;
		
	public GameObject[] treePrefabs;
	public GameObject[] bushPrefabs;

	public int numWaystones = 1000;
	public float pathWidth = 2.8f;

	public int numTrees = 1000;
	public int numBushes = 2000;

	public void generateWaystones() {
		Debug.Log ("creating waystones");

		var wayStones = new GameObject("Waystones");

		for (var i = 0; i < numWaystones; i++)
		{
			var f = i / (float)numWaystones;

			var posLeft = spline.GetPoint3DT(f, pathWidth / 2f, 0f, 0f);
			var dirLeft = spline.GetDirectionT(f);
			
			var cubeLeft = GameObject.CreatePrimitive (PrimitiveType.Cube);
			cubeLeft.transform.SetParent (wayStones.transform);
			cubeLeft.transform.LookAt(posLeft, dirLeft);
			cubeLeft.transform.position = posLeft;
			cubeLeft.transform.localScale = new Vector3 (0.1f, 0.1f, 0.1f);

			var posRight = spline.GetPoint3DT(f, - pathWidth / 2f, 0f, 0f);
			var dirRight = spline.GetDirectionT(f);
			
			var cubeRight = GameObject.CreatePrimitive (PrimitiveType.Cube);
			cubeRight.transform.SetParent (wayStones.transform);
			cubeRight.transform.LookAt(posRight, dirRight);
			cubeRight.transform.position = posRight;
			cubeRight.transform.localScale = new Vector3 (0.1f, 0.1f, 0.1f);
		}
		
		wayStones.transform.SetParent(transform);
	}

	private List<GameObject> generateVegetation(GameObject[] prefabs, int num, float offset)
	{

		var objs = new List<GameObject>();
		var terrain = Terrain.activeTerrain;

		while (objs.Count < num)
		{
			// random position on path
			var f = Random.value;

			// random offset to the side
			float offsetX = Random.value * (offset * 2) - offset;
			
			// propability that there is a tree at this offset
			float prop = (-Mathf.Cos(offsetX * Mathf.PI / offset) + 1f) / 2f;
			
			// simple way to sample from a distribution
			if (Random.value > prop) continue;
					
			// create object
			var obj = Instantiate(prefabs[(int) (Random.value * prefabs.Length)]);
			
			// set position relative to path
			obj.transform.position = spline.GetPoint3DT(f, offsetX, 0, 0);
			
			// move to terrain height
			obj.transform.position = new Vector3(
				obj.transform.position.x,
				terrain.SampleHeight(obj.transform.position),
				obj.transform.position.z
			);
			
			// rotate by a random angle
			obj.transform.Rotate(Vector3.up, Random.value * 2 * Mathf.PI);

			// add to output list
			objs.Add(obj);
		}

		return objs;
	}

	public void generateTrees()
	{
		var treeObjs = generateVegetation(treePrefabs, numTrees, 50);
		
		var trees = new GameObject("Trees");

		foreach (var tree in treeObjs)
		{
			tree.transform.SetParent(trees.transform);
		}
		
		trees.transform.SetParent(transform);
	}
	
	public void generateBushes()
	{

		var bushObjs = generateVegetation(bushPrefabs, numBushes, 25);
		
		var bushes = new GameObject("Bushes");

		foreach (var bush in bushObjs)
		{
			bush.transform.SetParent(bushes.transform);
		}
		
		bushes.transform.SetParent(transform);

	}
}
