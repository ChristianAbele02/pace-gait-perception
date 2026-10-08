//======================================================================================================
// Copyright 2016, NaturalPoint Inc.
//======================================================================================================

using System;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using Optitrack;
using UnityEngine;


namespace NaturalPoint.NatNetLib
{
    [Serializable]
    public class NatNetException : System.Exception
    {
        public NatNetException()
        {
        }


        public NatNetException(string message)
            : base(message)
        {
        }


        public NatNetException(string message, Exception inner)
            : base(message, inner)
        {
        }


        internal static void ThrowIfNotOK(NatNetError result, string message)
        {
            if (result != NatNetError.NatNetError_OK)
            {
                throw new NatNetException(message + " (" + result.ToString() + ")");
            }
        }
    }


    internal class NatNetClient : IDisposable
    {
//        public class DataDescriptions
//        {
//            public List<sMarkerSetDescription> MarkerSetDescriptions;
//            public List<sRigidBodyDescription> RigidBodyDescriptions;
//            public List<sSkeletonDescription> SkeletonDescriptions;
//            public List<sForcePlateDescription> ForcePlateDescriptions;
//        }

        // CONSTANTS
        private const int MAX_NAMELENGTH = 256;

        private const int NAT_CONNECT = 0;
        private const int NAT_SERVERINFO = 1;
        private const int NAT_REQUEST = 2;
        private const int NAT_RESPONSE = 3;
        private const int NAT_REQUEST_MODELDEF = 4;
        private const int NAT_MODELDEF = 5;
        private const int NAT_REQUEST_FRAMEOFDATA = 6;
        private const int NAT_FRAMEOFDATA = 7;
        private const int NAT_MESSAGESTRING = 8;
        private const int NAT_UNRECOGNIZED_REQUEST = 100;

        public static Version NatNetLibVersion
        {
            get
            {
                // Fixme ... do something useful here
                return new Version(2, 9, 0, 0);
            }
        }

        public bool Connected { get; private set; }

        /// <summary>
        /// This event is raised when a new frame is received via the network.
        /// IMPORTANT: This executes (via reverse P/Invoke) in the context of
        /// the NatNetLib network service thread.
        /// </summary>
        /// <remarks>
        /// NB: The sFrameOfMocapData native structure is large and expensive
        /// to marshal. In particular, each invocation allocates ~200 KB.
        /// </remarks>
        public event EventHandler<FrameData> NativeFrameReceived;

        //public class NativeFrameReceivedEventArgs : EventArgs
        //{
        //    private sFrameOfMocapData? m_marshaledFrame;
        //    private IntPtr m_nativeFrame;

        //    public IntPtr NativeFramePointer {
        //        get
        //        {
        //            return m_nativeFrame;
        //        }

        //        set
        //        {
        //            // Invalidate lazily-evaluated cached marshaled frame.
        //            m_marshaledFrame = null;

        //            m_nativeFrame = value;
        //        }
        //    }

        //    public sFrameOfMocapData MarshaledFrame {
        //        get {
        //            if ( m_marshaledFrame.HasValue == false )
        //            {
        //                m_marshaledFrame = (sFrameOfMocapData)Marshal.PtrToStructure( NativeFramePointer, typeof( sFrameOfMocapData ) );
        //            }

        //            return m_marshaledFrame.Value;
        //        }
        //    }
        //}


        #region Private fields

        //private IntPtr m_clientHandle = IntPtr.Zero;
        private bool m_disposed = false;
        //private NatNetFrameReceivedCallback m_nativeFrameReceivedHandler;
        //private NativeFrameReceivedEventArgs m_nativeFrameReceivedEventArgs = new NativeFrameReceivedEventArgs();

        private UdpClient dataClient;
        private UdpClient commandClient;

        private IPEndPoint localDataEp;
        private IPEndPoint localCommandEp;
        private IPEndPoint remoteCommandEp;

        #endregion Private fields


        public NatNetClient()
        {
            // NatNetError retval = NatNetLib.NativeMethods.NatNet_Client_Create( out m_clientHandle );
            // NatNetException.ThrowIfNotOK( retval, "NatNet_Client_Create failed." );

            //if ( m_clientHandle == IntPtr.Zero )
            //{
            //    throw new NatNetException( "NatNet_Client_Create returned null handle." );
            //}

            //// This ensures the reverse P/Invoke delegate passed to the native code stays alive.
            //m_nativeFrameReceivedHandler = FrameReceivedNativeThunk;

            // retval = NatNetLib.NativeMethods.NatNet_Client_SetFrameReceivedCallback( m_clientHandle, m_nativeFrameReceivedHandler );
            // NatNetException.ThrowIfNotOK( retval, "NatNet_Client_SetFrameReceivedCallback failed." );
        }


        public void Connect(NatNetConnectionType connType,
            IPAddress localAddress,
            IPAddress serverAddress,
            UInt16 serverCommandPort = NatNetConstants.DefaultCommandPort,
            UInt16 serverDataPort = NatNetConstants.DefaultDataPort,
            string multicastAddress = NatNetConstants.MulticastAddress)
        {
            ThrowIfDisposed();

            // sNatNetClientConnectParams initParams = new sNatNetClientConnectParams {
            // j    ConnectionType = connType,
            // j    ServerCommandPort = serverCommandPort,
            // j    ServerDataPort = serverDataPort,
            // j    LocalAddress = localAddress.ToString(),
            // j    ServerAddress = serverAddress.ToString(),
            // j    MulticastAddress = multicastAddress == null ? null : multicastAddress.ToString()
            // j};

            // NatNetError retval = NatNetLib.NativeMethods.NatNet_Client_Connect( m_clientHandle, ref initParams );
            // NatNetException.ThrowIfNotOK( retval, "NatNet_Client_Connect failed." );


            // initialize command socket
            commandClient = new UdpClient {Client = {ReceiveBufferSize = 0x100000}};
            localCommandEp = new IPEndPoint(IPAddress.Any, 0);
            commandClient.Client.Bind(localCommandEp);
            commandClient.EnableBroadcast = true;

            // server address for commands
            remoteCommandEp = new IPEndPoint(serverAddress, serverCommandPort);

            // send initial handshake
            //var async = commandClient.BeginReceive(CommandCallback, this);
            var version = SendRequest<string>(new OptitrackReceiver.SPacket {iMessage = NAT_CONNECT, nDataBytes = 0});
            Debug.Log("received version: " + version);

            //if (!async.AsyncWaitHandle.WaitOne(1000))
            //{
            //    Debug.LogError("failed to connect to Optitrack, disabling tracking");
            //}


            // initialize data socket
            dataClient = new UdpClient();
            localDataEp = new IPEndPoint(IPAddress.Any, serverDataPort);
            dataClient.Client.Bind(localDataEp);

            dataClient.JoinMulticastGroup(IPAddress.Parse(multicastAddress));
            dataClient.BeginReceive(DataCallback, this);

            Connected = true;
        }


        public void Disconnect()
        {
            ThrowIfDisposed();

            if (Connected)
            {
                // NatNetError retval = NatNetLib.NativeMethods.NatNet_Client_Disconnect( m_clientHandle );
                // NatNetException.ThrowIfNotOK( retval, "NatNet_Client_Disconnect failed." );
                dataClient.Close();
                commandClient.Close();

                Connected = false;
            }
        }


        #region Private fields

        public FrameData floriansFrameData;
        public DataDescription floriansDataDescription;
        private string floriansNatNetVersion;
        private string floriansServerVersion;

        #endregion

        static void DataCallback(IAsyncResult ar)
        {
            var us = (NatNetClient) ar.AsyncState;
            var client = us.dataClient;

            var bytes = client.EndReceive(ar, ref us.localDataEp);
            var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);

            if (client.Available < 0)
            {
                client.BeginReceive(DataCallback, us);
                return;
            }


            try
            {

                // only data will ever be received here
                var ptr = handle.AddrOfPinnedObject();
                us.floriansFrameData = FrameData.unpack(ptr);
            }
            finally
            {
                handle.Free();
            }


            try
            {
                us.ThrowIfDisposed();

                if (us.NativeFrameReceived != null)
                {
                    us.NativeFrameReceived(us, us.floriansFrameData);
                }
            }
            catch (Exception ex)
            {
                // It's important that we consume any exceptions here, since an exception thrown
                // from this reverse P/Invoke delegate would transform into an SEH exception once
                // propagated across the native code boundary, and NatNetLib would blow up.
                System.Diagnostics.Debug.WriteLine("ERROR - Exception occurred in FrameReceivedNativeThunk: " +
                                                   ex.ToString());
            }

            client.BeginReceive(DataCallback, us);
        }

        static void CommandCallback(IAsyncResult ar)
        {
            var us = (NatNetClient) ar.AsyncState;
            var client = us.commandClient;

            var bytes = client.EndReceive(ar, ref us.localCommandEp);
            var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);



            try
            {
                var packet = (OptitrackReceiver.SPacket) Marshal.PtrToStructure(handle.AddrOfPinnedObject(),
                    typeof(OptitrackReceiver.SPacket));
                var ptr = handle.AddrOfPinnedObject();

                Debug.Log(
                    "Command Callback received " + packet.iMessage + " message of " + packet.nDataBytes + " bytes");

                switch (packet.iMessage)
                {
                    case NAT_FRAMEOFDATA:
                        us.floriansFrameData = FrameData.unpack(ptr);
                        break;
                    case NAT_MODELDEF:
                        us.floriansDataDescription = DataDescription.unpack(ptr);
                        break;
                    case NAT_SERVERINFO:
                        var offset = 260;
                        us.floriansServerVersion = ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                                   ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                                   ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                                   ParsingUtils.ReadByte(ptr, ref offset);
                        us.floriansNatNetVersion = ParsingUtils.ReadByte(ptr, ref offset) + "." +
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
        }

        public DataDescription GetDataDescriptions()
        {
            ThrowIfDisposed();

            //IntPtr pDataDescriptions;
            //NatNetError retval = NatNetLib.NativeMethods.NatNet_Client_GetDataDescriptionList( m_clientHandle, out pDataDescriptions );
            //NatNetException.ThrowIfNotOK( retval, "NatNet_Client_GetDataDescriptions failed." );
            floriansDataDescription = SendRequest<DataDescription>(new OptitrackReceiver.SPacket {iMessage = NAT_REQUEST_MODELDEF, nDataBytes = 0});



            return floriansDataDescription;
        }

        private T SendRequest<T>(OptitrackReceiver.SPacket request) where T : class
        {
            var buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(request));
            Marshal.StructureToPtr(request, buffer, false);

            var managedArray = new byte[Marshal.SizeOf(request)];
            Marshal.Copy(buffer, managedArray, 0, Marshal.SizeOf(request));

            commandClient.Send(managedArray, Marshal.SizeOf(request), remoteCommandEp);

            var result = default(T);

//            var bytes = commandClient.Receive(ref localCommandEp);
            var asyncResult = commandClient.BeginReceive(null, null);
            var asyncDone = asyncResult.AsyncWaitHandle.WaitOne(1000);
            
            if (!asyncDone)
            {
                Debug.LogError("Timeout during Sendrequest. ID: " + request.iMessage);
                return null;
            }
            
            var bytes = commandClient.EndReceive(asyncResult, ref remoteCommandEp);
            var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            
            try
            {
                var packet = (OptitrackReceiver.SPacket) Marshal.PtrToStructure(handle.AddrOfPinnedObject(),
                    typeof(OptitrackReceiver.SPacket));
                var ptr = handle.AddrOfPinnedObject();

                switch (packet.iMessage)
                {
                    case NAT_FRAMEOFDATA:
                        result = FrameData.unpack(ptr) as T;
                        break;
                    case NAT_MODELDEF:
                        result = DataDescription.unpack(ptr) as T;
                        break;
                    case NAT_SERVERINFO:
                        var offset = 260;
                        result = ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                 ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                 ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                 ParsingUtils.ReadByte(ptr, ref offset) + "  " +
                                 ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                 ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                 ParsingUtils.ReadByte(ptr, ref offset) + "." +
                                 ParsingUtils.ReadByte(ptr, ref offset) as T;
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
                
            // done
            return result;
        }

        //private void SendRequest(OptitrackReceiver.SPacket request)
        //{
        //    var buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(request));
        //    Marshal.StructureToPtr(request, buffer, false);

        //    var managedArray = new byte[Marshal.SizeOf(request)];
        //    Marshal.Copy(buffer, managedArray, 0, Marshal.SizeOf(request));

        //    commandClient.Send(managedArray, Marshal.SizeOf(request), remoteCommandEp);
        //}

        #region Dispose pattern

        ~NatNetClient()
        {
            Dispose(false);
        }


        /// <summary>Implements IDisposable.</summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Called by both the <see cref="IDisposable.Dispose()"/> override,
        /// as well as the finalizer, to do the actual cleanup work.
        /// </summary>
        /// <param name="disposing">
        /// True if <see cref="Dispose()"/> was called explicitly. False if
        /// running as part of the finalizer. If false, do not attempt to
        /// reference other managed objects, since they may have already been
        /// finalized themselves.
        /// </param>
        protected virtual void Dispose(bool disposing)
        {
            if (m_disposed)
                return;

            // Disconnect if necessary.
            if (Connected)
            {
                // NatNetError disconnectResult = NatNetLib.NativeMethods.NatNet_Client_Disconnect( m_clientHandle );

                // if ( disconnectResult != NatNetError.NatNetError_OK )
                // {
                //     System.Diagnostics.Debug.WriteLine( "NatNet_Client_Disconnect returned " + disconnectResult.ToString() + "." );
                // }

                dataClient.Close();
                commandClient.Close();

                Connected = false;
            }

            // // Now destroy the native client.
            // NatNetError destroyResult = NatNetLib.NativeMethods.NatNet_Client_Destroy( m_clientHandle );

            // if ( destroyResult != NatNetError.NatNetError_OK )
            // {
            //     System.Diagnostics.Debug.WriteLine( "NatNet_Client_Destroy returned " + destroyResult.ToString() + "." );
            // }

            // m_clientHandle = IntPtr.Zero;

            m_disposed = true;
        }


        private void ThrowIfDisposed()
        {
            if (m_disposed)
            {
                throw new ObjectDisposedException(GetType().FullName);
            }
        }

        #endregion Dispose pattern
    }
}
