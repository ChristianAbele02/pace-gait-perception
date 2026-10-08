using System;
using UnityEngine;

namespace Optitrack
{

    [Serializable]
    /// <summary>
    /// One decoded NatNet frame: marker sets, rigid-body poses, and the native frame number
    /// and timestamp. The native identifiers matter — they are logged so the analysis can
    /// tell a genuinely new tracking sample from a Unity frame that re-sampled the old one.
    /// </summary>
    public class FrameData : EventArgs
    {
        [Serializable]
        public struct MarkerSet
        {
            public string szName;
            public Vector3[] markers;
        }

        [Serializable]
        public struct RigidMarker
        {
            public int id;
            public Vector3 position;
            public float size;
        }

        [Serializable]
        public struct RigidBody
        {
            public int id;
            public Vector3 position;
            public Quaternion orientation;
            public RigidMarker[] rigidMarkers;
            public float error;
            public bool trackingValid;
        }

        [Serializable]
        public struct Skeleton
        {
            public int id;
            public RigidBody[] bones;
        }
      
        public short iMessage;
        public short nDataBytes;
        public int   frameNumber;
        public MarkerSet[] markerSets;
        public RigidBody[] rigidBodies;
        public Skeleton[] skeletons;
        public float softwareLatency;
        public double timestamp;
        public bool isRecording;
        public bool trackedModelsChanged;


        public static FrameData unpack(IntPtr ptr)
        {
        
            var data = new FrameData();
            var offset = 0;


            data.iMessage = ParsingUtils.ReadShort(ptr, ref offset);
            data.nDataBytes = ParsingUtils.ReadShort(ptr, ref offset);
            data.frameNumber = ParsingUtils.ReadInt(ptr, ref offset);



            // Loop through number of marker sets and get name and data
            var nMarkerSets = ParsingUtils.ReadInt(ptr, ref offset);
//            Debug.Log("reading " + nMarkerSets + " marker sets");
            
            data.markerSets = new MarkerSet[nMarkerSets];

            for (var i = 0; i < nMarkerSets; i++)
            {
                // markerset name
                data.markerSets[i].szName = ParsingUtils.ReadString(ptr, ref offset);

                // marker data
                var nMarkers = ParsingUtils.ReadInt(ptr, ref offset);
                data.markerSets[i].markers = new Vector3[nMarkers];
                
                for (var j = 0; j < nMarkers; j++)
                {
                    var x = ParsingUtils.ReadFloat(ptr, ref offset);
                    var y = ParsingUtils.ReadFloat(ptr, ref offset);
                    var z = ParsingUtils.ReadFloat(ptr, ref offset);

                    data.markerSets[i].markers[j] = new Vector3(x, y, z);
                }
            }


            // Loop through unlabeled markers
            var nOtherMarkers = ParsingUtils.ReadInt(ptr, ref offset);
//            Debug.Log("reading " + nOtherMarkers + " other markers");
            
            // OtherMarker list is Deprecated (but it apparently has to be read nevertheless)
            for (var i = 0; i < nOtherMarkers; i++)
            {
                var x = ParsingUtils.ReadFloat(ptr, ref offset);
                var y = ParsingUtils.ReadFloat(ptr, ref offset);
                var z = ParsingUtils.ReadFloat(ptr, ref offset);
            }

            // Loop through rigidbodies
            var nRigidBodies = ParsingUtils.ReadInt(ptr, ref offset);
//            Debug.Log("reading " + nRigidBodies + " rigid bodies");
            
            data.rigidBodies = new RigidBody[nRigidBodies];
            
            for (var i = 0; i < nRigidBodies; i++)
            {
                // Rigid body position and orientation
                data.rigidBodies[i].id = ParsingUtils.ReadInt(ptr, ref offset);

                // position
                var x = ParsingUtils.ReadFloat(ptr, ref offset);
                var y = ParsingUtils.ReadFloat(ptr, ref offset);
                var z = ParsingUtils.ReadFloat(ptr, ref offset);

                data.rigidBodies[i].position = new Vector3(x, y, z);

                // orientation 
                var qx = ParsingUtils.ReadFloat(ptr, ref offset);
                var qy = ParsingUtils.ReadFloat(ptr, ref offset);
                var qz = ParsingUtils.ReadFloat(ptr, ref offset);
                var qw = ParsingUtils.ReadFloat(ptr, ref offset);

                data.rigidBodies[i].orientation = new Quaternion(qx, qy, qz, qw);

                // associated marker positions
                var nRigidMarkers = ParsingUtils.ReadInt(ptr, ref offset);
                data.rigidBodies[i].rigidMarkers = new RigidMarker[nRigidMarkers];
                
                for(var j = 0; j < nRigidMarkers; j++)
                {
                    var mx = ParsingUtils.ReadFloat(ptr, ref offset);
                    var my = ParsingUtils.ReadFloat(ptr, ref offset);
                    var mz = ParsingUtils.ReadFloat(ptr, ref offset);

                    data.rigidBodies[i].rigidMarkers[j].position = new Vector3(mx,my,mz);
                }

                // associated marker ids (since version 2.0)
                for(var j = 0; j < nRigidMarkers; j++)
                {
                    data.rigidBodies[i].rigidMarkers[j].id = ParsingUtils.ReadInt(ptr, ref offset);
                }

                // associated marker sizes (since version 2.0)
                for(var j = 0; j < nRigidMarkers; j++)
                {
                    data.rigidBodies[i].rigidMarkers[j].size = ParsingUtils.ReadInt(ptr, ref offset);
                }

                // mean marker error (since natnet version 2.0)
                data.rigidBodies[i].error = ParsingUtils.ReadFloat(ptr, ref offset);

                // parameters (since natnet version 2.6)
                var parameters = ParsingUtils.ReadShort(ptr, ref offset);
                data.rigidBodies[i].trackingValid = (parameters & 0x01) != 0;

            } // next rigid body


            // skeletons (natnet version 2.1 and later)
            var nSkeletons = ParsingUtils.ReadInt(ptr, ref offset);
//            Debug.Log("reading " + nSkeletons + " skeletons");
            
            data.skeletons = new Skeleton[nSkeletons];
                
            for (var i = 0; i < nSkeletons; i++)
            {
                // skeleton id
                data.skeletons[i].id = ParsingUtils.ReadInt(ptr, ref offset);

                // number of bones
                var nBones = ParsingUtils.ReadInt(ptr, ref offset);
                data.skeletons[i].bones = new RigidBody[nBones];

                /* Debug.Log("Rigid Body Count: " + nBones); */

                for (var j = 0; j < nBones ; j++)
                {
                    // id
                    var id = ParsingUtils.ReadInt(ptr, ref offset);
                    // Debug.Log(id);
                    data.skeletons[i].bones[j].id = (int)id - 65536;

                    /* Debug.Log("ID : " + data.skeletons[i].bones[j].id); */

                    // position
                    var x = ParsingUtils.ReadFloat(ptr, ref offset);
                    var y = ParsingUtils.ReadFloat(ptr, ref offset);
                    var z = ParsingUtils.ReadFloat(ptr, ref offset);

                    data.skeletons[i].bones[j].position = new Vector3(x, y, z);

                    /* Debug.Log("pos: " + data.skeletons[i].bones[j].position); */


                    // orientation 
                    var qx = ParsingUtils.ReadFloat(ptr, ref offset);
                    var qy = ParsingUtils.ReadFloat(ptr, ref offset);
                    var qz = ParsingUtils.ReadFloat(ptr, ref offset);
                    var qw = ParsingUtils.ReadFloat(ptr, ref offset);

                    data.skeletons[i].bones[j].orientation = new Quaternion(qx, qy, qz, qw);

                    /* Debug.Log("ori: " + data.skeletons[i].bones[j].orientation); */

                    // associated marker positions
                    var nRigidMarkers = ParsingUtils.ReadInt(ptr, ref offset);
                    data.skeletons[i].bones[j].rigidMarkers = new RigidMarker[nRigidMarkers];

                    for(var k = 0; k < nRigidMarkers; k++)
                    {
                        var mx = ParsingUtils.ReadFloat(ptr, ref offset);
                        var my = ParsingUtils.ReadFloat(ptr, ref offset);
                        var mz = ParsingUtils.ReadFloat(ptr, ref offset);

                        data.skeletons[i].bones[j].rigidMarkers[k].position = new Vector3(mx,my,mz);
                    }

                    // associated marker ids
                    for(var k = 0; k < nRigidMarkers; k++)
                    {
                        data.skeletons[i].bones[j].rigidMarkers[k].id = ParsingUtils.ReadInt(ptr, ref offset);
                    }

                    // associated marker sizes
                    for(var k = 0; k < nRigidMarkers; k++)
                    {
                        data.skeletons[i].bones[j].rigidMarkers[k].size = ParsingUtils.ReadFloat(ptr, ref offset);

                        Debug.Log("Marker " + k + ":" + 
                            " id=" + data.skeletons[i].bones[j].rigidMarkers[k].id + 
                            " size=" + data.skeletons[i].bones[j].rigidMarkers[k].size + 
                            " pos=" + data.skeletons[i].bones[j].rigidMarkers[k].position
                            );

                    }

                    // mean marker error
                    data.skeletons[i].bones[j].error = ParsingUtils.ReadFloat(ptr, ref offset);

                    // parameters
                    var parameters = ParsingUtils.ReadShort(ptr, ref offset);
                    data.skeletons[i].bones[j].trackingValid = (parameters & 0x01) != 0;

                    // Debug.Log("Bone " + data.skeletons[i].bones[j].id + ", pos: " + data.skeletons[i].bones[j].position);

                } // next rigid body


            } // next skeleton


            // labeled markers (since version 2.3) (unused ... still read it)
            var nLabeledMarkers = ParsingUtils.ReadInt(ptr, ref offset);

            for (var i = 0; i < nLabeledMarkers; i++)
            {
                int id = ParsingUtils.ReadInt(ptr, ref offset);
      
                float x = ParsingUtils.ReadFloat(ptr, ref offset);
                float y = ParsingUtils.ReadFloat(ptr, ref offset);
                float z = ParsingUtils.ReadFloat(ptr, ref offset);

                float size = ParsingUtils.ReadFloat(ptr, ref offset);
      
                // marker parameters (since version 2.6)
                short parameters = ParsingUtils.ReadShort(ptr, ref offset);
            }
    
            // force plate data (since version 2.9) (also just read it)
            var nForcePlates = ParsingUtils.ReadInt(ptr, ref offset);

            for (var p = 0; p < nForcePlates; p++)
            {
                int id = ParsingUtils.ReadInt(ptr, ref offset);
                int nChannels = ParsingUtils.ReadInt(ptr, ref offset);

                for (var c = 0; c < nChannels; c++)
                {
                    int nFrames = ParsingUtils.ReadInt(ptr, ref offset);

                    for (var f = 0; f < nFrames; f++)
                    {
                        float val = ParsingUtils.ReadFloat(ptr, ref offset);
                    }
                }
            } // next force plate
    
            // no device data (since 3.0)

            // software latency (till 3.0)
            data.softwareLatency = ParsingUtils.ReadFloat(ptr, ref offset);
    
            // timecode
            var timecode = ParsingUtils.ReadUnsignedInt(ptr, ref offset);
            var timecodeSub = ParsingUtils.ReadUnsignedInt(ptr, ref offset);
    
            // timestamp
            data.timestamp = ParsingUtils.ReadDouble(ptr, ref offset);
    
            // no high res timestamps (since 3.0)

            // frame params
            var frameParameters = ParsingUtils.ReadShort(ptr, ref offset);

            data.isRecording = (frameParameters & 0x01) != 0;
            data.trackedModelsChanged = (frameParameters & 0x02) != 0;
    
    
            // end of data tag
            int eod = ParsingUtils.ReadInt(ptr, ref offset);
    
            // done
            return data;
        }
    }
}
