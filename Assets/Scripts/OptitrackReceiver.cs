using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using UnityEngine;

using System.Runtime.InteropServices;
using Optitrack;

/// <summary>
/// Receives OptiTrack/NatNet rigid-body streams and republishes them as Unity poses.
///
/// The head rigid body is identified by its Motive streaming ID; poses are exposed with a
/// validity flag and the native frame number/timestamp so <c>ILLLoggerController</c> can
/// record whether each Unity frame saw genuinely NEW tracking data. Tracking dropouts are
/// surfaced rather than silently held at the last pose, because a held pose is
/// indistinguishable from a motionless participant in the gait analysis.
/// </summary>
public class OptitrackReceiver : MonoBehaviour {

  private UdpClient client;
  private IPEndPoint localDataEp;
  private IPAddress multicastaddress;

  private UdpClient commandClient;
  private IPEndPoint localCommandEp;

  private IPEndPoint remoteCommandEp;

  public string multicastServer  = "239.255.42.99";
  public int    multicastCommandPort    = 1510;
  public int    multicastDataPort    = 1511;

  public string commandServer = "tracking-pc";


  public string serverVersion = "XXX";
  public string natNetVersion = "XXX";

  public FrameData frameData;
  public DataDescription dataDescription;

  public GameObject skeleton;
  public Dictionary<int, GameObject> skeletonMap;

  // CONSTANTS
  private const int MAX_NAMELENGTH =            256;

  private const int NAT_CONNECT =               0;
  private const int NAT_SERVERINFO =            1;
  private const int NAT_REQUEST =               2;
  private const int NAT_RESPONSE =              3;
  private const int NAT_REQUEST_MODELDEF =      4;
  private const int NAT_MODELDEF =              5;
  private const int NAT_REQUEST_FRAMEOFDATA =   6;
  private const int NAT_FRAMEOFDATA =           7;
  private const int NAT_MESSAGESTRING =         8;
  private const int NAT_UNRECOGNIZED_REQUEST =  100;
  /* private const int UNDEFINED =                 999999.9999f; */

  /* private const long MAX_PACKETSIZE =            100000;  // max size of packet (actual packet size is dynamic) */


  [StructLayout(LayoutKind.Explicit)]
  public struct SPacket
  {
    [FieldOffset(0)]
    public ushort iMessage;                // message ID (e.g. NAT_FRAMEOFDATA)
    [FieldOffset(2)]
    public ushort nDataBytes;              // Num bytes in payload
  }

  
  void Start () {

    // initialize command socket
    commandClient = new UdpClient();
    commandClient.Client.ReceiveBufferSize = 0x100000;
    var localCommandEp = new IPEndPoint(IPAddress.Any, 0);
    commandClient.Client.Bind(localCommandEp);
    commandClient.EnableBroadcast = true;

    commandClient.BeginReceive(CommandCallback, this);

    // initialize data socket
    client = new UdpClient();
    localDataEp = new IPEndPoint(IPAddress.Any, multicastDataPort);
    client.Client.Bind(localDataEp);

    multicastaddress = IPAddress.Parse(multicastServer);
    client.JoinMulticastGroup(multicastaddress);

    // server address for commands
    var ips = System.Net.Dns.GetHostAddresses(commandServer);
    remoteCommandEp = new IPEndPoint(ips[0], multicastCommandPort);

    client.BeginReceive(DataCallback, this);


    // send initial connect signal
    SendRequest(new SPacket {iMessage = NAT_CONNECT, nDataBytes = 0});

    // send request for model definition
    SendRequest(new SPacket {iMessage = NAT_REQUEST_MODELDEF, nDataBytes = 0});
  }

  void Update ()
  {

    /* string strData = Encoding.ASCII.GetString(data); */
    /* Debug.Log("length: " + data.Length + ", command: " ); */

    if (dataDescription != null && dataDescription.skeletons.Length > 0 && skeletonMap == null)
    {
      Debug.Log("creating skeleton (name=" + dataDescription.skeletons[0].name + ", id=" + dataDescription.skeletons[0] + ")");
      createUnitySkeleton(dataDescription.skeletons[0]);
    }

    updateUnitySkeleton();

  }


  static void CommandCallback(IAsyncResult ar)
  {
    var us = (OptitrackReceiver) ar.AsyncState;
    var client = us.commandClient;

    var bytes = client.EndReceive(ar, ref us.localCommandEp);
    var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);

    try
    {
      var packet = (SPacket)Marshal.PtrToStructure(handle.AddrOfPinnedObject(), typeof(SPacket));
      var ptr = handle.AddrOfPinnedObject();

      switch (packet.iMessage)
      {
        case NAT_FRAMEOFDATA:
          us.frameData = FrameData.unpack(ptr);
          break;
        case NAT_MODELDEF:
          us.dataDescription = DataDescription.unpack(ptr);
          break;
        case NAT_SERVERINFO:
          var offset = 260;
          us.serverVersion = ParsingUtils.ReadByte(ptr, ref offset) + "." +
                             ParsingUtils.ReadByte(ptr, ref offset) + "." +
                             ParsingUtils.ReadByte(ptr, ref offset) + "." +
                             ParsingUtils.ReadByte(ptr, ref offset);
          us.natNetVersion = ParsingUtils.ReadByte(ptr, ref offset) + "." +
                             ParsingUtils.ReadByte(ptr, ref offset) + "." +
                             ParsingUtils.ReadByte(ptr, ref offset) + "." +
                             ParsingUtils.ReadByte(ptr, ref offset);
          break;

        default:
          Debug.LogError("Unrecognized Optitrack received (id: " + packet.iMessage + ")");
          break;

      }

      //Debug.Log("length: " + packet.nDataBytes + ", command: " + packet.iMessage + ", frame number: " + packet.frameNumber + ", number of markersets: " + packet.nMarkerSets);
    }
    finally
    {
      handle.Free();
    }


    client.BeginReceive(CommandCallback, us);

  }

  static void DataCallback(IAsyncResult ar)
  {

    var us = (OptitrackReceiver) ar.AsyncState;
    var client = us.client;

    var bytes = client.EndReceive(ar, ref us.localDataEp);
    var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);

    if(client.Available < 0 ) {
      client.BeginReceive(DataCallback, us);
      return;
    }


    try
    {
      var ptr = handle.AddrOfPinnedObject();

      // only data will ever be received here
      us.frameData = FrameData.unpack(ptr);



    }
    finally
    {
      handle.Free();
    }

    client.BeginReceive(DataCallback, us);

  }


  private void SendRequest(SPacket request)
  {
    var buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(request));
    Marshal.StructureToPtr(request, buffer, false);

    var managedArray = new byte[Marshal.SizeOf(request)];
    Marshal.Copy(buffer, managedArray, 0, Marshal.SizeOf(request));

    commandClient.Send(managedArray, Marshal.SizeOf(request), remoteCommandEp);
  }


  private void createUnitySkeleton(SkeletonDescription skeleton)
  {

    skeletonMap = new Dictionary<int, GameObject>();

    // create bones as transforms
    foreach (var bone in skeleton.bones)
    {
      var goBone = new GameObject(bone.name);
      GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
      cube.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
      cube.transform.SetParent(goBone.transform);
      // goBone.transform.position = bone.offset;
      skeletonMap.Add(bone.id, goBone);
    }

    // linking bones
    foreach (var bone in skeleton.bones)
    {
      var goBone = skeletonMap[bone.id];

      if(bone.parent == 0) {
        goBone.transform.SetParent(this.transform);
      }
      else
      {
        var parentBone = skeletonMap[bone.parent];
        goBone.transform.SetParent(parentBone.transform);
      }
    }

    // update offsets?
    foreach (var bone in skeleton.bones)
    {
      var goBone = skeletonMap[bone.id];

      if(bone.id == 1) {
        goBone.transform.localPosition = new Vector3(
          bone.offset.x,
          bone.offset.y,
          -bone.offset.z
          ) + MasterTrackingData.TrackingSystemOffset;
      }
      else
      {
        goBone.transform.localPosition = new Vector3(
          bone.offset.x,
          bone.offset.y,
          -bone.offset.z
          );
      }

    }

  }

  private void updateUnitySkeleton()
  {
    if(frameData.skeletons.Length < 1) return;

    var skeleton = frameData.skeletons[0];

    foreach (var bone in skeleton.bones)
    {
      // if (!bone.trackingValid) continue;
      var skeletonId = bone.id >> 16;
      var boneId = bone.id & 0xFFFF;

      var go = skeletonMap[boneId];

      // root bone controls position of whole skeleton
      if(boneId == 1) {

        // mirror position on z axis
        var mirroredPosition = new Vector3(
            bone.position.x,
            bone.position.y,
            -bone.position.z
            );

        // total position is bone position plus tracking system offset
        go.transform.localPosition = mirroredPosition + MasterTrackingData.TrackingSystemOffset;

      }

      // mirror rotation on z axix
      var mirroredRotation = new Quaternion(
            bone.orientation.x,
            bone.orientation.y,
            -bone.orientation.z,
            -bone.orientation.w
            );

      // total rotation is bone rotation "plus" rotation of platform in VR
      go.transform.rotation = transform.localRotation * mirroredRotation;

    }


  }
}
