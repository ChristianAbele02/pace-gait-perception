using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

public class PathController : MonoBehaviour {

	public bool EnableHurdles;
	public bool EnableWaystones;

	public float pathWidth = 1.5f;
	
	public BezierSpline spline;

	public int numberOfHurdles = 20;
	public int numberOfWaystones = 200;

	// hurdle seed to be send to clients
	private int hurdleSeed;

	void Start () {

		// initialize
		hurdleSeed = (int) (Random.value * 4096);

		// statistics
		Debug.Log ("Total path length: " + spline.Length);

	}


	void OnPlayerConnected(NetworkPlayer player)
	{
		Debug.Log ("creating track for player: " + player.ipAddress);
//		GetComponent<NetworkView> ().RPC ("createWaystones", RPCMode.AllBuffered);

	}

}
