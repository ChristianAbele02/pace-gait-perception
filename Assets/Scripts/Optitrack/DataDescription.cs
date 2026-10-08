using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Optitrack
{

    [Serializable]
    /// <summary>
    /// NatNet data-description records: the streamed asset list Motive sends once per
    /// connection, mapping streaming IDs to named marker sets and rigid bodies. The head
    /// rigid body is located by the ID configured in the scene.
    /// </summary>
    public struct MarkerSetDescription
    {
        public string name;
        public string[] markerNames;
    }

    [Serializable]
    public class RigidBodyDescription
    {
        public string name;
        public int id;
        public Vector3 offset;
        public int parent;
    }

    [Serializable]
    public struct BoneDescription
    {
        public string name;
        public int id;
        public int parent;
        public Vector3 offset;
    }

    [Serializable]
    public class SkeletonDescription
    {
        public string name;
        public int id;
        public BoneDescription[] bones;
    }


    [Serializable]
    public class DataDescription
    {

        public int iMessage;
        public int nDataBytes;

        public MarkerSetDescription[] markerSets;
        public RigidBodyDescription[] rigidBodies;
        public SkeletonDescription[] skeletons;

        public static DataDescription unpack(IntPtr ptr)
        {

            var data = new DataDescription();
            var offset = 0;

            // header
            data.iMessage = ParsingUtils.ReadShort(ptr, ref offset);
            data.nDataBytes = ParsingUtils.ReadShort(ptr, ref offset);

            // datasets
            int nDatasets = Marshal.ReadInt32(ptr, offset);
            offset += sizeof(int);

            Debug.Log("unpacking " + nDatasets + " datasets");


            var markerSets = new List<MarkerSetDescription>();
            var rigidBodies = new List<RigidBodyDescription>();
            var skeletons = new List<SkeletonDescription>();


            for (var d = 0; d < nDatasets; d++)
            {

                // read description type
                int type = Marshal.ReadInt32(ptr, offset);
                offset += sizeof(int);

                switch (type)
                {
                   case 0: // markerset

                       var markerset = new MarkerSetDescription();

                       // name
                       markerset.name = ParsingUtils.ReadString(ptr, ref offset);

                       // marker data
                       int nMarkers = Marshal.ReadInt32(ptr, offset);
                       offset += sizeof(int);

                       markerset.markerNames = new string[nMarkers];
                       for (var m = 0; m < nMarkers; m++)
                       {
                           markerset.markerNames[m] = ParsingUtils.ReadString(ptr, ref offset);
                       }

                       // done
                       markerSets.Add(markerset);
                       break;

                   case 1: // rigid body

                       var rigidbody = new RigidBodyDescription();

                       // name
                       rigidbody.name = ParsingUtils.ReadString(ptr, ref offset);

                       // id
                       rigidbody.id = ParsingUtils.ReadInt(ptr, ref offset);

                       // parent id
                       rigidbody.parent = ParsingUtils.ReadInt(ptr, ref offset);

                       // offset
                       var ox = ParsingUtils.ReadFloat(ptr, ref offset);
                       var oy = ParsingUtils.ReadFloat(ptr, ref offset);
                       var oz = ParsingUtils.ReadFloat(ptr, ref offset);
                       rigidbody.offset = new Vector3(ox, oy, oz);

                       // done 
                       rigidBodies.Add(rigidbody);
                       break;

                   case 2: // skeleton

                       var skeleton = new SkeletonDescription();

                       // name 
                       skeleton.name = ParsingUtils.ReadString(ptr, ref offset);

                       // id
                       skeleton.id = ParsingUtils.ReadInt(ptr, ref offset);

                       // rigid bodies (bone) count
                       var nBones = ParsingUtils.ReadInt(ptr, ref offset);
                       skeleton.bones = new BoneDescription[nBones];

                       for (var b = 0; b < nBones; b++)
                       {
                           // name
                           skeleton.bones[b].name = ParsingUtils.ReadString(ptr, ref offset);
                           // id
                           skeleton.bones[b].id = ParsingUtils.ReadInt(ptr, ref offset);
                           // parent id
                           skeleton.bones[b].parent = ParsingUtils.ReadInt(ptr, ref offset);
                           // offset
                           var box = ParsingUtils.ReadFloat(ptr, ref offset);
                           var boy = ParsingUtils.ReadFloat(ptr, ref offset);
                           var boz = ParsingUtils.ReadFloat(ptr, ref offset);
                           skeleton.bones[b].offset = new Vector3(box, boy, boz);
                       }

                       // done
                       skeletons.Add(skeleton);
                       break;


                   default:
                       Debug.LogError("Invalid data description type: " + type);
                       break;


                }



            }

            // parsing done
            data.markerSets = markerSets.ToArray();
            data.rigidBodies = rigidBodies.ToArray();
            data.skeletons = skeletons.ToArray();

            // done
            return data;
        }



    }
}
