using System;
using System.Runtime.InteropServices;

namespace Optitrack
{
    /// <summary>
    /// Byte-level readers for the NatNet wire format (little-endian scalars, length-prefixed
    /// strings), used to decode OptiTrack packets.
    /// </summary>
    public static class ParsingUtils
    {

        public static string ReadString(IntPtr ptr, ref int offset)
        {
            var s = Marshal.PtrToStringAnsi(new IntPtr(ptr.ToInt64() + offset));
            offset += s.Length + 1;
            return s;
        }
        
        public static byte ReadByte(IntPtr ptr, ref int offset)
        {
            var i = Marshal.ReadByte(ptr, offset);
            offset += sizeof(byte);
            return i;
        }
        
        public static short ReadShort(IntPtr ptr, ref int offset)
        {
            var i = Marshal.ReadInt16(ptr, offset);
            offset += sizeof(short);
            return i;
        }

        public static int ReadInt(IntPtr ptr, ref int offset)
        {
            var i = Marshal.ReadInt32(ptr, offset);
            offset += sizeof(int);
            return i;
        }
        
        public static uint ReadUnsignedInt(IntPtr ptr, ref int offset)
        {
            uint ui;
            var i = Marshal.ReadInt32(ptr, offset);
            offset += sizeof(uint);

            unsafe
            {
                int* iRef = &i;
                ui = *((uint*) iRef);
            }
            
            return ui;
        }
        
        public static double ReadDouble(IntPtr ptr, ref int offset)
        {
            double d;
            var i = Marshal.ReadInt64(ptr, offset);
            offset += sizeof(double);

            unsafe
            {
              long* iRef = &i;
              d = *((double*)iRef);
            }

            return d;
        }

        public static float ReadFloat(IntPtr ptr, ref int offset)
        {
            float f;
            var i = Marshal.ReadInt32(ptr, offset);
            offset += sizeof(float);

            unsafe
            {
              int* iRef = &i;
              f = *((float*)iRef);
            }

            return f;
        }
    }
}