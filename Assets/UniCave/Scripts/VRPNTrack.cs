//MIT License
//Copyright 2016-Present 
//Ross Tredinnick
//Brady Boettcher
//Living Environments Laboratory
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
//to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, 
//sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
//INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

using UnityEngine;
using System.Collections;

public class VRPNTrack : MonoBehaviour
{
    [SerializeField]
    private string trackerAddress = "Isense900@C6_V1_HEAD";
    [SerializeField]
    private int channel = 0;
    [SerializeField]
    private bool trackPosition = true;
    [SerializeField]
    private bool trackRotation = true;

    [SerializeField]
    public bool onlyHeadNode = true;

    public bool debugOutput = false;

    public int Channel
    {
        get { return channel; }
        set
        {
            channel = value;
        }
    }

    public bool TrackPosition
    {
        get { return trackPosition; }
        set
        {
            trackPosition = value;
            StopCoroutine("Position");
            if (trackPosition && Application.isPlaying)
            {
                StartCoroutine("Position");
            }
        }
    }

    public bool TrackRotation
    {
        get { return trackRotation; }
        set
        {
            trackRotation = value;
            StopCoroutine("Rotation");
            if (trackRotation && Application.isPlaying)
            {
                StartCoroutine("Rotation");
            }
        }
    }

    [SerializeField]
    public Vector3 trackScale = Vector3.one;

    [Header("Stability: outlier rejection / dropout hold")]
    [Tooltip("Reject a head-POSITION sample that implies a speed faster than this (m/s) "
        + "as a marker swap / tracking dropout, and hold the last good position instead. "
        + "Matches the analysis pipeline's plausibility bound (config.MAX_PLAUSIBLE_SPEED_MPS).")]
    public float maxPlausibleSpeed = 5.0f;
    [Tooltip("Reject a head-ROTATION sample whose angular step exceeds this (deg/frame) "
        + "as an orientation flip, holding the last good rotation instead.")]
    public float maxRotationStepDeg = 45.0f;
    [Tooltip("If this many samples in a row are rejected, accept the next one anyway so a "
        + "GENUINE large move / tracker re-init isn't held forever. Non-finite samples are "
        + "never force-accepted.")]
    public int maxConsecutiveRejections = 30;

    // --- last-good pose held across dropouts (stability gating) ---
    private Vector3 lastGoodLocalPos;
    private bool hasLastGoodPos = false;
    private int posRejectStreak = 0;
    private Quaternion lastGoodLocalRot = Quaternion.identity;
    private bool hasLastGoodRot = false;
    private int rotRejectStreak = 0;

    private void Start()
    {
        //this gets rid of this object from non-head nodes...we only want this running on the machine that connects to VRPN...
        //this assumes a distributed type setup, where one machine connects to the tracking system and distributes information
        //to other machines...
        //some setups may try to connect each machine to vrpn...
        //in that case, we wouldn't want to destroy this object..
        if (System.Environment.MachineName != MasterTrackingData.HeadNodeMachineName && onlyHeadNode)
        {
                Debug.Log("Removing tracker settings from " + gameObject.name + " on " + System.Environment.MachineName);
                Destroy(this);
                return;
        }

        if (trackPosition)
        {
            StartCoroutine("Position");
        }

        if (trackRotation)
        {
            StartCoroutine("Rotation");
        }
    }

    private IEnumerator Position()
    {
        while (true)
        {
            Vector3 rawPos = VRPN.vrpnTrackerPos(trackerAddress, channel);
            Vector3 posWithoutOffset = rawPos - MasterTrackingData.TrackingSystemOffset;

            // Apply scale to the relative position
            Vector3 scaledPos = Vector3.Scale(posWithoutOffset, trackScale);
            Vector3 candidate = scaledPos + MasterTrackingData.TrackingSystemOffset;

            // --- stability gate: reject glitch frames, hold the last good pose ---
            // A non-finite sample is always garbage (held, never re-acquired). A finite
            // sample is rejected only if it implies an implausible speed; a sustained run
            // of such samples is treated as a real teleport / tracker re-init and is
            // re-acquired after maxConsecutiveRejections so we never stick to a stale pose.
            // When a sample is ACCEPTED the result is byte-for-byte what the old code wrote,
            // so this only changes behaviour on glitch frames (no effect on clean data).
            bool accept;
            if (!IsFinite(candidate))
            {
                accept = false;
            }
            else if (!hasLastGoodPos)
            {
                accept = true;
            }
            else
            {
                float dt = (Time.deltaTime > 0f) ? Time.deltaTime : (1f / 60f);
                bool plausible = Vector3.Distance(candidate, lastGoodLocalPos) <= maxPlausibleSpeed * dt;
                accept = plausible || posRejectStreak >= maxConsecutiveRejections;
            }

            if (accept)
            {
                transform.localPosition = candidate;
                lastGoodLocalPos = candidate;
                hasLastGoodPos = true;
                posRejectStreak = 0;
            }
            else
            {
                posRejectStreak++;
                if (hasLastGoodPos) transform.localPosition = lastGoodLocalPos;
                if (debugOutput) Debug.LogWarning("VRPNTrack: held head position (rejected glitch sample #" + posRejectStreak + ").");
            }
            yield return null;
        }
    }

    private IEnumerator Rotation()
    {
        while (true)
        {
            Quaternion candidate = VRPN.vrpnTrackerQuat(trackerAddress, channel);

            // --- stability gate (mirrors Position): drop NaN / zero-norm quaternions and
            // single-frame orientation flips, holding the last good rotation. ---
            bool accept;
            if (!IsValidRotation(candidate))
            {
                accept = false;
            }
            else if (!hasLastGoodRot)
            {
                accept = true;
            }
            else
            {
                bool plausible = Quaternion.Angle(candidate, lastGoodLocalRot) <= maxRotationStepDeg;
                accept = plausible || rotRejectStreak >= maxConsecutiveRejections;
            }

            if (accept)
            {
                transform.localRotation = candidate;
                lastGoodLocalRot = candidate;
                hasLastGoodRot = true;
                rotRejectStreak = 0;
            }
            else
            {
                rotRejectStreak++;
                if (hasLastGoodRot) transform.localRotation = lastGoodLocalRot;
                if (debugOutput) Debug.LogWarning("VRPNTrack: held head rotation (rejected glitch sample #" + rotRejectStreak + ").");
            }
            yield return null;
        }
    }

    // True if every component is finite (not NaN / Inf).
    private static bool IsFinite(Vector3 v)
    {
        return !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z)
              || float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));
    }

    // True if the quaternion is finite AND has non-zero norm. VRPN returns the zero
    // quaternion when a tracker reports no data, which is not a valid rotation and would
    // make Quaternion.Angle undefined.
    private static bool IsValidRotation(Quaternion q)
    {
        if (float.IsNaN(q.x) || float.IsNaN(q.y) || float.IsNaN(q.z) || float.IsNaN(q.w)) return false;
        if (float.IsInfinity(q.x) || float.IsInfinity(q.y) || float.IsInfinity(q.z) || float.IsInfinity(q.w)) return false;
        return (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) > 1e-6f;
    }
}
