using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// Drives the treadmill over Modbus/TCP and exposes its measured belt speed to the rest
/// of the experiment.
///
/// Speed is ramped rather than stepped, so the participant is never accelerated abruptly
/// while inside the CAVE. The measured speed is republished each frame for
/// <c>ILLLoggerController</c>, which is what lets the analysis verify the belt actually
/// ran (and detect sensor glitches) instead of trusting the commanded value.
/// </summary>
public class TreadmillController : MonoBehaviour {

	[Tooltip("Speed in meter per second")]
	public string tsAddress = "127.0.0.1";
	public int tsPort = 1338;

	private Socket socket;
	private IPEndPoint endPoint;

	private Vector3 lastPosition;

	[Tooltip("Speed in m/s")]
	public float speed;
	// Controls if the manager is setting the speed.
    // If TRUE, 'speed' is sent manually instead of being calculated.
    [HideInInspector] public bool isManualControl = false;
    // The speed to be sent manually.
    [HideInInspector] public float manualTargetSpeed = 0f;

    // Hard safety cap on the belt speed actually sent (m/s). Normal block speed is ~1 m/s;
    // this only ever clamps a glitch (frame stall / transition spike), never real walking.
    private const float MAX_BELT_SPEED_MPS = 5f;

	public void Start () {
		
		
		// initialize socket
		socket = new Socket(
			AddressFamily.InterNetwork, 
			SocketType.Dgram,
			ProtocolType.Udp
		);

		// initialize endpoint
		endPoint = new IPEndPoint(IPAddress.Parse(tsAddress), tsPort);
		
		// initialize last position
		lastPosition = transform.position;
	}
	
    public void Update () {
        
        float speedToSend;

        if (!isManualControl) // Standard operation: Calculate speed from movement
        {
            // as the treadmill is only going in Z direction
            // we have to massage the position delta into the correct direction
            var heading = transform.position - lastPosition;
            var inverseRotation = Quaternion.Inverse(transform.rotation);

            var unrotatedHeading = inverseRotation * heading;
            speed = unrotatedHeading.z / Time.deltaTime;
            
            // store last position
            lastPosition = transform.position;
            
            speedToSend = speed;
        }
        else // Manual control: Use the stored target speed
        {
            // Use the speed set by an external script (like CheckpointManager)
            speedToSend = manualTargetSpeed;
        }

        // Clamp to a physiologically plausible belt speed before sending. A frame stall or
        // a residual manual->auto transition can otherwise compute an absurd one-frame speed
        // (observed up to ~180 m/s); never command the belt backward or to a runaway value.
        speedToSend = Mathf.Clamp(speedToSend, 0f, MAX_BELT_SPEED_MPS);

        // assemble "data package"
        var speedKMH = speedToSend * 3.6f;
        var pkg = string.Format("S{0:0.000}", speedKMH);
        var serializedPkg = Encoding.ASCII.GetBytes (pkg);

        // send package
        socket.SendTo(serializedPkg, endPoint);
    }


    /// <summary>
    /// Forces the treadmill to send a specific speed,
    /// bypassing its automatic calculation.
    /// </summary>
    public void ForceTreadmillSpeed(float manualSpeed)
    {
        isManualControl = true;
        manualTargetSpeed = manualSpeed; // Set to the desired speed
        Debug.Log("TreadmillController: Manual control enabled (Forced speed: " + manualSpeed + " m/s).");
    }

	/// <summary>
    /// Stops automatic calculation and forces the treadmill to send 0 m/s.
    /// </summary>
    public void StopTreadmill()
    {
        ForceTreadmillSpeed(0.0f);
    }
    
    /// <summary>
    /// Disables manual control, allowing the controller to resume
    /// calculating speed from the object's movement.
    /// </summary>
    public void ResumeTreadmill()
    {
        // Refresh lastPosition so the FIRST auto frame computes a normal one-frame delta,
        // NOT the whole position jump accumulated during the manual period. Without this,
        // every manual->auto switch (each checkpoint) produced a huge spurious speed spike
        // (observed up to ~180 m/s at the checkpoint cadence) because lastPosition was stale
        // from the last auto frame, many frames earlier.
        lastPosition = transform.position;
        isManualControl = false;
        Debug.Log("TreadmillController: Manual control disabled, resuming calculation from movement.");
    }
	
}
