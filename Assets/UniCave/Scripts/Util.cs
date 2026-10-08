using UnityEngine;

public class Util {



	public static float heightFromTerrain(Terrain terrain, Vector3 pos) {
		// get height and normal from terrain
		Vector3 terrainLocalPos = pos - terrain.transform.position;
		Vector2 normalizedPos = new Vector2 (Mathf.InverseLerp (0.0f, terrain.terrainData.size.x, terrainLocalPos.x)
			, Mathf.InverseLerp (0.0f, terrain.terrainData.size.z, terrainLocalPos.z));
		return terrain.terrainData.GetInterpolatedHeight (normalizedPos.x, normalizedPos.y) / terrain.terrainData.size.y;
	}

	public static Vector3 replaceHeightFromTerrain(Terrain terrain, Vector3 pos) {
		float height = heightFromTerrain (terrain, pos);
		return new Vector3 (pos.x, height, pos.z);
	}
}
