using UnityEngine;
using System.Runtime.InteropServices;

public static class VRPN
{
    [DllImport("unityVrpn")]
    private static extern double vrpnAnalogExtern(string address, int channel, int frameCount);

    [DllImport("unityVrpn")]
    private static extern bool vrpnButtonExtern(string address, int channel, int frameCount);

    [DllImport("unityVrpn")]
    private static extern double vrpnTrackerExtern(string address, int channel, int component, int frameCount);

    public static double vrpnAnalog(string address, int channel)
    {
        return vrpnAnalogExtern(address, channel, Time.frameCount);
    }

    public static bool vrpnButton(string address, int channel)
    {
        return vrpnButtonExtern(address, channel, Time.frameCount);
    }

    //todo - need to allow different transforms here...
    //need to adjust two below functions to match up with your own tracking system's transform
    public static Vector3 vrpnTrackerPos(string address, int channel)
    {
		float vrpnX = (float)vrpnTrackerExtern (address, channel, 0, Time.frameCount);
		float vrpnY = (float)vrpnTrackerExtern (address, channel, 1, Time.frameCount);
		float vrpnZ = (float)vrpnTrackerExtern (address, channel, 2, Time.frameCount);

		return new Vector3(vrpnX, vrpnY, -vrpnZ) + MasterTrackingData.TrackingSystemOffset;
    }

    public static Quaternion vrpnTrackerQuat(string address, int channel)
    {
		float vrpnX = (float)vrpnTrackerExtern (address, channel, 3, Time.frameCount);
		float vrpnY = (float)vrpnTrackerExtern (address, channel, 4, Time.frameCount);
		float vrpnZ = (float)vrpnTrackerExtern (address, channel, 5, Time.frameCount);
		float vrpnW = (float)vrpnTrackerExtern (address, channel, 6, Time.frameCount);

		return new Quaternion(-vrpnX, -vrpnY, vrpnZ, vrpnW);
    }
}
