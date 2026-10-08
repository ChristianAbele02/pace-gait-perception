//======================================================================================================
// Copyright 2016, NaturalPoint Inc.
//======================================================================================================

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using NaturalPoint;
using NaturalPoint.NatNetLib;
using Optitrack;


/// <summary>Skeleton naming conventions supported by OptiTrack Motive.</summary>
public enum OptitrackBoneNameConvention
{
    Motive,
    FBX,
    BVH,
}


public static class OptitrackHiResTimer
{
    public struct Timestamp
    {
        internal Int64 m_ticks;

        public float AgeSeconds
        {
            get
            {
                return Now().SecondsSince( this );
            }
        }

        public float SecondsSince( Timestamp reference )
        {
            Int64 deltaTicks = m_ticks - reference.m_ticks;
            return deltaTicks / (float)System.Diagnostics.Stopwatch.Frequency;
        }
    }

    public static Timestamp Now()
    {
        return new Timestamp {
            m_ticks = System.Diagnostics.Stopwatch.GetTimestamp()
        };
    }
}


/// <summary>
/// Connects to a NatNet streaming server and makes the data available in lightweight Unity-friendly representations.
/// </summary>
public class OptitrackStreamingClient : MonoBehaviour
{
    public enum ClientConnectionType
    {
        Multicast,
        Unicast
    }


    public ClientConnectionType ConnectionType;
    public string LocalAddress = "127.0.0.1";
    public string ServerAddress = "127.0.0.1";
    public UInt16 ServerCommandPort = NatNetConstants.DefaultCommandPort;
    public UInt16 ServerDataPort = NatNetConstants.DefaultDataPort;
    public OptitrackBoneNameConvention BoneNamingConvention = OptitrackBoneNameConvention.Motive;


    #region Private fields
    private bool m_receivedFrameSinceConnect = false;
    private OptitrackHiResTimer.Timestamp m_lastFrameDeliveryTimestamp;
    private Coroutine m_connectionHealthCoroutine = null;

    private NatNetClient m_client;
    
    public DataDescription m_dataDescs;
    public FrameData latestFrameData;

    /// <summary>
    /// Lock held during access to fields which are potentially modified by <see cref="OnNatNetFrameReceived"/> (which
    /// executes on a separate thread). Note while the lock is held, any frame updates received are simply dropped.
    /// </summary>
    private object m_frameDataUpdateLock = new object();
    #endregion Private fields


    /// <summary>
    /// Returns the first <see cref="OptitrackStreamingClient"/> component located in the scene.
    /// Provides a convenient, sensible default in the common case where only a single client exists.
    /// Issues a warning if more than one such component is found.
    /// </summary>
    /// <returns>An arbitrary OptitrackClient from the scene, or null if none are found.</returns>
    public static OptitrackStreamingClient FindDefaultClient()
    {
        OptitrackStreamingClient[] allClients = FindObjectsOfType<OptitrackStreamingClient>();

        if ( allClients.Length == 0 )
        {
            Debug.LogError( "Unable to locate any " + typeof( OptitrackStreamingClient ).FullName + " components." );
            return null;
        }
        else if ( allClients.Length > 1 )
        {
            Debug.LogWarning( "Multiple " + typeof( OptitrackStreamingClient ).FullName + " components found in scene; defaulting to first available." );
        }

        return allClients[0];
    }


    /// <summary>Get the most recently received state for the specified rigid body.</summary>
    /// <param name="rigidBodyId">Corresponds to the "User ID" field in Motive.</param>
    /// <returns>The most recent available state, or null if none available.</returns>
    public FrameData.RigidBody GetLatestRigidBodyState( Int32 rigidBodyId )
    {

        lock ( m_frameDataUpdateLock )
        {
            foreach (var rigidBody in latestFrameData.rigidBodies)
            {
                if (rigidBody.id == rigidBodyId)
                {
                    return rigidBody;
                }
            }
        }

        throw new Exception("could not find rigid body by id: " + rigidBodyId);
    }


    /// <summary>Get the most recently received state for the specified skeleton.</summary>
    /// <param name="skeletonId">
    /// Taken from the corresponding <see cref="OptitrackSkeletonDefinition.Id"/> field.
    /// To find the appropriate skeleton definition, use <see cref="GetSkeletonDefinitionByName"/>.
    /// </param>
    /// <returns>The most recent available state, or null if none available.</returns>
    public FrameData.Skeleton GetLatestSkeletonState( Int32 skeletonId )
    {

        lock ( m_frameDataUpdateLock )
        {
            foreach (var skeleton in latestFrameData.skeletons)
            {
                if (skeleton.id == skeletonId)
                {
                    return skeleton;
                }
            }
        }

        throw new Exception("could not find skeleton by id: " + skeletonId);
    }

    public double GetLatestTimestamp()
    {
        lock (m_frameDataUpdateLock)
        {
            return latestFrameData.timestamp;
        }

    }

    public int GetLatestFrameNumber()
    {
        lock (m_frameDataUpdateLock)
        {
            return latestFrameData.frameNumber;
        }
        
    }


    /// <summary>Retrieves the definition of the rigid body with the specified streaming ID.</summary>
    /// <param name="rigidBodyId"></param>
    /// <returns>The specified rigid body definition, or null if not found.</returns>
    public RigidBodyDescription GetRigidBodyDefinitionById( Int32 rigidBodyId )
    {
        foreach (var rbDef in m_dataDescs.rigidBodies)
        {
            if ( rbDef.id == rigidBodyId )
            {
                return rbDef;
            }
        }

        return null;
    }


    /// <summary>Retrieves the definition of the skeleton with the specified asset name.</summary>
    /// <param name="skeletonAssetName">The name of the skeleton for which to retrieve the definition.</param>
    /// <returns>The specified skeleton definition, or null if not found.</returns>
    public SkeletonDescription GetSkeletonDefinitionByName( string skeletonAssetName )
    {
        return m_dataDescs == null ? null : m_dataDescs.skeletons.FirstOrDefault(skelDef => skelDef.name.Equals(skeletonAssetName, StringComparison.InvariantCultureIgnoreCase));
    }


    /// <summary>Request data descriptions from the host, then update our definitions.</summary>
    /// <exception cref="NatNetException">
    /// Thrown by <see cref="NatNetClient.GetDataDescriptions"/> if the request to the server fails.
    /// </exception>
    public void UpdateDefinitions()
    {
        // This may throw an exception if the server request times out or otherwise fails.
        m_dataDescs = m_client.GetDataDescriptions();

        // TODO Translate rigid body definitions.
//        for ( int nativeRbDescIdx = 0; nativeRbDescIdx < m_dataDescs.rigidBodies.Length; ++nativeRbDescIdx )
//        {
//            var nativeRb = m_dataDescs.rigidBodies[nativeRbDescIdx];
//
//            OptitrackRigidBodyDefinition rbDef = new OptitrackRigidBodyDefinition {
//                Id = nativeRb.id,
//                Name = "", // nativeRb.Name,
//                Markers = new List<OptitrackRigidBodyDefinition.MarkerDefinition>( nativeRb.rigidMarkers.Length )
//            };
//
//            // Populate nested marker definitions.
//            for ( int nativeMarkerIdx = 0; nativeMarkerIdx < nativeRb.rigidMarkers.Length; ++nativeMarkerIdx )
//            {
//                int positionOffset = nativeMarkerIdx * Marshal.SizeOf( typeof( MarkerDataVector ) );
//                IntPtr positionPtr = new IntPtr( nativeRb.MarkerPositions.ToInt64() + positionOffset );
//
//                int labelOffset = nativeMarkerIdx * Marshal.SizeOf( typeof( Int32 ) );
//                IntPtr labelPtr = new IntPtr( nativeRb.MarkerRequiredLabels.ToInt64() + labelOffset );
//
//                MarkerDataVector nativePos =
//                    (MarkerDataVector)Marshal.PtrToStructure( positionPtr, typeof( MarkerDataVector ) );
//
//                Int32 nativeLabel = Marshal.ReadInt32( labelPtr );
//
//                OptitrackRigidBodyDefinition.MarkerDefinition markerDef =
//                    new OptitrackRigidBodyDefinition.MarkerDefinition {
//                        Position = new Vector3( nativePos.Values[0], nativePos.Values[1], nativePos.Values[2] ),
//                        RequiredLabel = nativeLabel,
//                    };
//
//                rbDef.Markers.Add( markerDef );
//            }
//
//            m_rigidBodyDefinitions.Add( rbDef );
//        }

//        // Translate skeleton definitions.
//        foreach(var nativeSkel in m_dataDescs.skeletons) 
//        {
//
//            OptitrackSkeletonDefinition skelDef = new OptitrackSkeletonDefinition {
//                Id = nativeSkel.Id,
//                Name = nativeSkel.Name,
//                Bones = new List<OptitrackSkeletonDefinition.BoneDefinition>( nativeSkel.RigidBodyCount ),
//            };
//
//            // Populate nested bone definitions.
//            for ( int nativeBoneIdx = 0; nativeBoneIdx < nativeSkel.RigidBodyCount; ++nativeBoneIdx )
//            {
//                sRigidBodyDescription nativeBone = nativeSkel.RigidBodies[nativeBoneIdx];
//
//                OptitrackSkeletonDefinition.BoneDefinition boneDef =
//                    new OptitrackSkeletonDefinition.BoneDefinition {
//                        Id = nativeBone.Id,
//                        ParentId = nativeBone.ParentId,
//                        Name = nativeBone.Name,
//                        // TODO FIXME?
//                        Offset = new Vector3( -nativeBone.OffsetX, nativeBone.OffsetY, nativeBone.OffsetZ ),
//                    };
//
//                skelDef.Bones.Add( boneDef );
//            }
//
//            m_skeletonDefinitions.Add( skelDef );
//        }
    }


    /// <summary>
    /// (Re)initializes <see cref="m_client"/> and connects to the configured streaming server.
    /// </summary>
    void OnEnable()
    {

        IPAddress serverAddr = Dns.GetHostEntry(ServerAddress).AddressList[0]; // IPAddress.Parse( ServerAddress ));
        IPAddress localAddr = Dns.GetHostEntry(LocalAddress).AddressList[0]; // IPAddress.Parse( LocalAddress ));

        NatNetConnectionType connType;
        switch ( ConnectionType )
        {
            case ClientConnectionType.Unicast:
                connType = NatNetConnectionType.NatNetConnectionType_Unicast;
                break;
            case ClientConnectionType.Multicast:
            default:
                connType = NatNetConnectionType.NatNetConnectionType_Multicast;
                break;
        }

        try
        {
            m_client = new NatNetClient();
            m_client.Connect( connType, localAddr, serverAddr );
            UpdateDefinitions();
        }
        catch ( Exception ex )
        {
            Debug.LogException( ex, this );
            Debug.LogError( GetType().FullName + ": Error connecting to server; check your configuration, and make sure the server is currently streaming.", this );
            this.enabled = false;
            return;
        }

        m_client.NativeFrameReceived += OnNatNetFrameReceived;
        m_connectionHealthCoroutine = StartCoroutine( CheckConnectionHealth() );
    }


    /// <summary>
    /// Disconnects from the streaming server and cleans up <see cref="m_client"/>.
    /// </summary>
    void OnDisable()
    {
        if ( m_connectionHealthCoroutine != null )
        {
            StopCoroutine( m_connectionHealthCoroutine );
            m_connectionHealthCoroutine = null;
        }

        m_client.NativeFrameReceived -= OnNatNetFrameReceived;
        m_client.Disconnect();
        m_client.Dispose();
        m_client = null;
    }


    System.Collections.IEnumerator CheckConnectionHealth()
    {
        const float kHealthCheckIntervalSeconds = 1.0f;
        const float kRecentFrameThresholdSeconds = 5.0f;

        // The lifespan of these variables is tied to the lifespan of a single connection session.
        // The coroutine is stopped on disconnect and restarted on connect.
        YieldInstruction checkIntervalYield = new WaitForSeconds( kHealthCheckIntervalSeconds );
        OptitrackHiResTimer.Timestamp connectionInitiatedTimestamp = OptitrackHiResTimer.Now();
        OptitrackHiResTimer.Timestamp lastFrameReceivedTimestamp;
        bool wasReceivingFrames = false;
        bool warnedPendingFirstFrame = false;

        while ( true )
        {
            yield return checkIntervalYield;

            if ( m_receivedFrameSinceConnect == false )
            {
                // Still waiting for first frame. Warn exactly once if this takes too long.
                if ( connectionInitiatedTimestamp.AgeSeconds > kRecentFrameThresholdSeconds )
                {
                    if ( warnedPendingFirstFrame == false )
                    {
                        Debug.LogWarning( GetType().FullName + ": No frames received from the server yet. Verify your connection settings are correct and that the server is streaming.", this );
                        warnedPendingFirstFrame = true;
                    }

                    continue;
                }
            }
            else
            {
                // We've received at least one frame, do ongoing checks for changes in connection health.
                lastFrameReceivedTimestamp.m_ticks = Interlocked.Read( ref m_lastFrameDeliveryTimestamp.m_ticks );
                bool receivedRecentFrame = lastFrameReceivedTimestamp.AgeSeconds < kRecentFrameThresholdSeconds;

                if ( wasReceivingFrames == false && receivedRecentFrame == true )
                {
                    // Transition: Bad health -> good health.
                    wasReceivingFrames = true;
                    Debug.Log( GetType().FullName + ": Receiving streaming data from the server.", this );
                    continue;
                }
                else if ( wasReceivingFrames == true && receivedRecentFrame == false )
                {
                    // Transition: Good health -> bad health.
                    wasReceivingFrames = false;
                    Debug.LogWarning( GetType().FullName + ": No streaming frames received from the server recently.", this );
                    continue;
                }
            }
        }
    }


    #region Private methods
    /// <summary>
    /// Event handler for NatNet frame delivery. Updates our simplified state representations.
    /// NOTE: This executes in the context of the NatNetLib network service thread!
    /// </summary>
    /// <remarks>
    /// Because the <see cref="sFrameOfMocapData"/> type is expensive to marshal, we instead utilize the
    /// <see cref="NatNetClient.NativeFrameReceivedEventArgs.NativeFramePointer"/>, treating it as as opaque, and
    /// passing it to some helper "accessor" functions to retrieve the subset of data we care about, using only
    /// blittable types which do not cause any garbage to be allocated.
    /// </remarks>
    /// <param name="sender"></param>
    /// <param name="eventArgs"></param>
    private void OnNatNetFrameReceived( object sender, FrameData frame)
    {
        // In the event of contention, drop the frame being delivered and return immediately.
        // We don't want to stall NatNetLib's internal network service thread.
        if ( ! Monitor.TryEnter( m_frameDataUpdateLock ) )
        {
            return;
        }

        try
        {
            // Update health markers.
            m_receivedFrameSinceConnect = true;
            Interlocked.Exchange( ref m_lastFrameDeliveryTimestamp.m_ticks, OptitrackHiResTimer.Now().m_ticks );
            
            // fix bone ids
            foreach (var skeleton in frame.skeletons)
            {
                for (var b = 0; b < skeleton.bones.Length; b++)
                {
                    skeleton.bones[b].id = skeleton.bones[b].id & 0xFFFF;

                }
            }

            // set current frame
            latestFrameData = frame;
        }
        catch ( Exception ex )
        {
            Debug.LogError( GetType().FullName + ": OnNatNetFrameReceived encountered an exception.", this );
            Debug.LogException( ex, this );
        }
        finally
        {
            Monitor.Exit( m_frameDataUpdateLock );
        }
    }




    public void _EnterFrameDataUpdateLock()
    {
        Monitor.Enter( m_frameDataUpdateLock );
    }


    public void _ExitFrameDataUpdateLock()
    {
        Monitor.Exit( m_frameDataUpdateLock );
    }
#endregion Private methods
}
