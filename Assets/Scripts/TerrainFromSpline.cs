using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Timeline;
using Random = UnityEngine.Random;

/// <summary>
/// Generates the ground mesh that follows the walking spline. Visual only — no part of
/// the paradigm or the logged data depends on it.
/// </summary>
public class TerrainFromSpline : MonoBehaviour
{

	private BezierSpline _spline;
	public Terrain Terrain;

	public float PathWidth = 1.2f;
	public float HeightOffset = 0.5f;

	public int smoothingPasses = 3;
	public int smoothingSize = 5;
	public int smoothingSigma = 1;
	
	public float perlinMuliplier = 10f;

	private float[,] heights;
	private bool[,] dirty;
	
	

    public void RecalculateTerrain()
    {
		_spline = GetComponent<BezierSpline>();
	    
		var res = Terrain.terrainData.heightmapResolution;
		// heights = Terrain.terrainData.GetHeights(0,0, res,res);
	    
	    heights = new float[Terrain.terrainData.heightmapHeight, Terrain.terrainData.heightmapWidth];
	    dirty = new bool[Terrain.terrainData.heightmapHeight, Terrain.terrainData.heightmapWidth];
	    
	    // initialize arrays
	    for (var y = 0; y < dirty.GetLength(0); y++)
	    for (var x = 0; x < dirty.GetLength(1); x++)
	    {
		    heights[y, x] = 0.5f;
		    dirty[y, x] = false;
	    }


	    var lastSplineA = _spline.GetPoint3DT(0, -PathWidth * 0.5f, HeightOffset, 0f);
	    var lastSplineB = _spline.GetPoint3DT(0, +PathWidth * 0.5f, HeightOffset, 0f);
	    var lastTerrainA = globalToTerrainCoordinates(lastSplineA);
	    var lastTerrainB = globalToTerrainCoordinates(lastSplineB);
	    
	    
	    const int steps = 1000;
	    for (var i = 1; i < steps; i++)
	    {
		    var t = i / (float) (steps - 1);

		    var splineA = _spline.GetPoint3DT(t, -PathWidth * 0.5f, HeightOffset, 0f);
		    var splineB = _spline.GetPoint3DT(t, +PathWidth * 0.5f, HeightOffset, 0f);
		    
		    var terrainA = globalToTerrainCoordinates(splineA);
		    var terrainB = globalToTerrainCoordinates(splineB);
		    
		    
		    DrawTriangle(lastTerrainA, lastTerrainB, terrainB);
		    DrawTriangle(lastTerrainA, terrainA, terrainB);
		    
		    lastTerrainA = terrainA;
		    lastTerrainB = terrainB;

	    }

	    SmoothTerrain();
		
	    // Undo.RecordObject(Terrain, "Recalculated Terrain From Spline");
		Terrain.terrainData.SetHeights(0,0, heights);
	    
		// update grass maps
		// GenerateGrassDetail();
		
		// update textures
		//GenerateTextures();

    }


    private void DrawTriangle(Vector3 a, Vector3 b, Vector3 c)
	{
		// as per: http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html#algo3

		// projecting vectors to 2d
		var a2d = new Vector2(a.x, a.z);
		var b2d = new Vector2(b.x, b.z);
		var c2d = new Vector2(c.x, c.z);
		
		var maxX = Mathf.Max(a2d.x, Mathf.Max(b2d.x, c2d.x));
		var minX = Mathf.Min(a2d.x, Mathf.Min(b2d.x, c2d.x));
		var maxZ = Mathf.Max(a2d.y, Mathf.Max(b2d.y, c2d.y));
		var minZ = Mathf.Min(a2d.y, Mathf.Min(b2d.y, c2d.y));

		var ab2d = b2d - a2d;
		var ac2d = c2d - a2d;

		for (var x = Mathf.FloorToInt(minX); x <= Mathf.CeilToInt(maxX); x++)
		{
			for (var y = Mathf.FloorToInt(minZ); y <= Mathf.CeilToInt(maxZ); y++)
			{
				var q = new Vector2	(x - a2d.x, y - a2d.y);
				
				// "crossproduct" as per: https://www.gamedev.net/forums/topic/289972-cross-product-of-2d-vectors/
				var s = (ab2d.x * q.y - ab2d.y * q.x) / (ab2d.x * ac2d.y - ab2d.y * ac2d.x);
				var t = (q.x * ac2d.y - q.y * ac2d.x) / (ab2d.x * ac2d.y - ab2d.y * ac2d.x);

				// break if not inside
				if (!(s >= 0) || !(t >= 0) || !(s + t <= 1)) continue;
				
				var ac3d = c - a;
				var ab3d = b - a;

				heights[y, x] = (a + t * ab3d + s * ac3d).y;
				dirty[y, x] = true;
			}
		}
	}

	private Vector3 globalToTerrainCoordinates(Vector3 global)
	{
		var terrainLocalPos = global - Terrain.transform.position;
		var res = Terrain.terrainData.heightmapResolution;
		var size = Terrain.terrainData.size;
		
		//Debug.Log("Size:" + size);
		
		var uvCoordinates = new Vector3 (
			Mathf.InverseLerp (0.0f, size.x, terrainLocalPos.x) * res,
			Mathf.InverseLerp (0.0f, size.y, terrainLocalPos.y),
			Mathf.InverseLerp (0.0f, size.z, terrainLocalPos.z) * res
			);

		return uvCoordinates;

	}

	private static float[,] gaussKernel(int size, float sigma)
	{
		var kernel = new float[size, size];
		var sum = 0f;
		
		// set values
		for (var y = 0; y < size; y++)
		{
			for (var x = 0; x < size; x++)
			{
				var offX = x - size / 2f;
				var offY = y - size / 2f;
				
				var v = Mathf.Exp(-(offX * offX + offY * offY) / (2f * sigma * sigma));
				kernel[y, x] = v;
				sum += v;
			}
		}

		// normalize
		for (var y = 0; y < size; y++)
		{
			for (var x = 0; x < size; x++)
			{
				kernel[y, x] /= sum;
			}
		}

		return kernel;

	}
	
	private void SmoothTerrain()
	{
		
		
		//var kernel = new[,]
		//{
		//	{0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f},
		//	{0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
		//	{0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f},
		//	{0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
		//	{0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f}
		//};
		
		// ensure that the kernel size is uneven
		if (smoothingSize % 2 == 0) smoothingSize -= 1;

		Debug.Log("creating gauss kernel (size: " + smoothingSize + ", sigma: " + smoothingSigma + ")");
		var kernel = gaussKernel(smoothingSize, smoothingSigma);
		
		// half kernel size
		var hks = kernel.GetLength(0) / 2;
		
		
		// add some random hills
		for (var y = 0; y < heights.GetLength(0); y++)
		{
			for (var x = 0; x < heights.GetLength(0); x++) {
				
				// don't touch heights generated by path
				if (dirty[y, x]) continue;
			
				var fx = perlinMuliplier * x / heights.GetLength(1);
				var fy = perlinMuliplier * y / heights.GetLength(0);

				var h = 0.5f * Mathf.PerlinNoise(fx, fy) 
				        + 0.25f * Mathf.PerlinNoise(2 * fx, 2 * fy) 
				        + 0.125f * Mathf.PerlinNoise(4 * fx, 4 * fy);
				
				heights[y,x] = h;
			}
			
		}
		// for (var i = 0; i < 100; i++)
		// {
		// 	var x = (int)(Random.value * heights.GetLength(1));
		// 	var y = (int)(Random.value * heights.GetLength(0));

		// 	if(dirty[y,x]) continue;
		// 	heights[y, x] = Random.value * 0.2f;
		// 	dirty[y, x] = true;
		// }

		// run smoothing passes
		for (var i = 0; i < smoothingPasses; i++)
		{

			// ignore outer edge, start at half kernel size (hks)
			for (var y = hks; y < heights.GetLength(0) - hks; y++)
			{
				var yOffset = y - hks;
				
				for (var x = hks; x < heights.GetLength(1) - hks; x++)
				{
				
					var xOffset = x - hks;
					
					
					// don't touch heights generated by path
					if (dirty[y, x]) continue;
					
					var hAcc = 0f;
					var fAcc = 0f;
					
					// for (var k = 0; k < kernel.GetLength(0); k++)
					// {
					// 	var ky = yOffset + k;
					// 	//if (ky < 0 || ky >= heights.GetLength(0)) continue;
					// 	
					// 	
					// 	for (var j = 0; j < kernel.GetLength(1); j++)
					// 	{
					// 		var jx = xOffset + j;
					// 		//if (jx < 0 || jx >= heights.GetLength(1)) continue;

					// 		// fAcc += kernel[k, j];
					// 		hAcc += heights[ky, jx]; // kernel[k, j]; // * heights[ky, jx];

					// 	}
					// 	
					// }
					
					
                    // hand unrolled:
                    hAcc += heights[yOffset + 0, xOffset + 0];
                    hAcc += heights[yOffset + 0, xOffset + 1];
                    hAcc += heights[yOffset + 0, xOffset + 2];
                    hAcc += heights[yOffset + 1, xOffset + 0];
                    hAcc += heights[yOffset + 1, xOffset + 1];
                    hAcc += heights[yOffset + 1, xOffset + 2];
                    hAcc += heights[yOffset + 2, xOffset + 0];
                    hAcc += heights[yOffset + 2, xOffset + 1];
                    hAcc += heights[yOffset + 2, xOffset + 2];


					// normalize value
					hAcc /= 9f; // fAcc;
					
					// set value
					heights[y, x] = hAcc;
				}
				
			}
			
			
		}

	}

	private void GenerateTextures()
	{
		var terrainData = Terrain.terrainData;

		var width = terrainData.alphamapWidth;
		var height = terrainData.alphamapHeight;
		var layers = terrainData.alphamapLayers;

		float[,,] splatmapData = new float[width, height, layers];


		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				splatmapData[x, y, 0] = 1.0f;
				splatmapData[x, y, 1] = 0.0f;
				splatmapData[x, y, 2] = 0.0f;
			}
		}
		
        terrainData.SetAlphamaps(0, 0, splatmapData);


	}

	// from here: https://answers.unity.com/questions/893506/procedural-grass-placement.html
    private void GenerateGrassDetail()
    {
	    
	    var terrainData = Terrain.terrainData;

	    var detailWidth = terrainData.detailWidth;
	    var detailHeight = terrainData.detailHeight;
	    
	    int[,] details0 = new int[ detailWidth, detailHeight ];
        int[,] details1 = new int[ detailWidth, detailHeight ];
 
        int x, y, strength;
 
        for ( x = 0; x < detailWidth / 4; x ++ ) // divided by 4 just to show a test patch
        {
			for ( y = 0; y < detailHeight / 4; y ++ ) // test patch
            {
	            strength = x % 2 == 0 ? x / 2 % 17 : 0; // just to spread the grass out a bit to see the difference

	            var w = new Vector3(
		            Terrain.GetPosition().x + terrainData.size.x * x / detailWidth,
		            Terrain.GetPosition().y,
		            Terrain.GetPosition().z + terrainData.size.z * y / detailHeight
	            );
	            _spline.ClosestPoint(w);


	            if (y % 4 == 0) // set detail layer 0 for every first rpline
		            details0[y, x] = 10; // strength;
                else if (y % 4 == 2) // set detail layer 1 for every third row in 4
		            details1[y, x] = 5; // strength;
            }
        }
 
        terrainData.SetDetailLayer( 0, 0, 0, details0 );
        terrainData.SetDetailLayer( 0, 0, 1, details1 );
	    
    }
}
