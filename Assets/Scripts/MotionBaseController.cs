using System.Net;
using System.Net.Sockets;
using System.Runtime.Serialization;
using UnityEngine;
using MsgPack.Serialization;



/// <summary>
/// Client for the motion-base rig. Retained from earlier work on this CAVE and NOT used
/// by the ILL locomotion paradigm, which uses the treadmill (see TreadmillController).
/// </summary>
public class MotionBaseController : MonoBehaviour {


	public string mbsAddress;
	public int mbsPort;

	[Tooltip("Enable rotation around the X axis (pitch)")]
	public bool enableX; 
	[Tooltip("Enable rotation around the Y axis (yaw)")]
	public bool enableY; 
	[Tooltip("Enable rotation around the Z axis (roll)")]
	public bool enableZ; 

	private Socket socket;
	private IPEndPoint endPoint;

	private MessagePackSerializer<IRDataPackage> serializer;




	void Start () {

		// initialize socket
		socket = new Socket(
			AddressFamily.InterNetwork, 
			SocketType.Dgram,
			ProtocolType.Udp
		);

		// initialize endpoint
		endPoint = new IPEndPoint(IPAddress.Parse(mbsAddress), mbsPort);

		// initialize serializer
		serializer = MessagePackSerializer.Get<IRDataPackage> ();
	} 
	
	void Update () {
		
		// convert transform to data package and let msgpack serialize it
		var irPkg = ToIRDataPackage ();

		try {
			if (irPkg == null) return;
			
			// serialize package
			var serializedPkg = serializer.PackSingleObject(irPkg);
			
			// send serialized package to mbc
			socket.SendTo(serializedPkg, endPoint);

		} catch (SerializationException e)
		{
			Debug.LogErrorFormat("Failed to Serialize: {0}", e);
		}

	}

	/** 
	 * convert Transform into IRDataPackage
	 */
	private IRDataPackage ToIRDataPackage()
	{

		// get rotation from transform and clamp to intuitive angles
		var rotation = transform.eulerAngles;
		rotation.x = rotation.x > 180f ? rotation.x - 360f : rotation.x;
		rotation.y = rotation.y > 180f ? rotation.y - 360f : rotation.y;
		rotation.z = rotation.z > 180f ? rotation.z - 360f : rotation.z;
		
		// disable rotations
		rotation.x = enableX ? rotation.x : 0f;
		rotation.y = enableY ? rotation.y : 0f;
		rotation.z = enableZ ? rotation.z : 0f;
		
		// create package
		return new IRDataPackage
		{
			// degrees to radians
			roll = -rotation.z * 0.0174533f,
			pitch = -rotation.x * 0.0174533f,
			yaw = rotation.y * 0.0174533f,
			
			// FIXME movement of the platform is not used yet
			surge = 0f,
			sway = 0f,
			heave = 0f,
			
            // this was solved by restricting the maximal velocities and accelerations
            // on the MBC which is a better solution than controlling them outside the
            // tight feedback loop it provides
			maxLinVel = 0,
			maxAngVel = 0,
			maxLinAcc = 0,
			maxAngAcc = 0
		};
	}

	/**
	 * IRDataPackage compatible with MBS
	 */
	public class IRDataPackage {
		[MessagePackMember( 0 )] public float roll;
		[MessagePackMember( 1 )] public float pitch;
		[MessagePackMember( 2 )] public float yaw;

		[MessagePackMember( 3 )] public float surge;
		[MessagePackMember( 4 )] public float sway;
		[MessagePackMember( 5 )] public float heave;

		[MessagePackMember( 6 )] public float maxLinVel;
		[MessagePackMember( 7 )] public float maxAngVel;

		[MessagePackMember( 8 )] public float maxLinAcc;
		[MessagePackMember( 9 )] public float maxAngAcc;

		public override string ToString() {
			return string.Format ("IRDataPackage({0},{1},{2},{3},{4},{5},{6},{7},{8},{9})", 
				roll, pitch, yaw,
				surge, sway, heave,
				maxLinVel, maxAngVel,
				maxLinAcc, maxAngAcc
			);
		}
	}
}
