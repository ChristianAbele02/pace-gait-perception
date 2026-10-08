Shader "Custom/GaborShader"
{
    Properties
    {
        _Frequency ("Frequency (cycles across quad)", Float) = 10.0
        _Sigma ("Sigma (Gaussian SD, quad units)", Float) = 0.0353
        _DiskRadius ("Visible disk radius (UV, 0-0.5)", Float) = 0.151
        _Rotation ("Rotation (Degrees)", Float) = 0.0
        _Phase ("Phase", Float) = 0.0
        _Contrast ("Contrast", Range(0, 1)) = 1.0
    }
    SubShader
    {
        // 1. Tell Unity this object is transparent
        Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
        LOD 100

        // 2. Enable Alpha Blending (Standard transparency math)
        Blend SrcAlpha OneMinusSrcAlpha 
        ZWrite Off 

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            float _Frequency;
            float _Sigma;
            float _DiskRadius;
            float _Rotation;
            float _Phase;
            float _Contrast;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                // Center UVs at (0,0) so the math works from the center out
                o.uv = v.uv - 0.5; 
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // --- Rotation Logic ---
                float rad = _Rotation * 3.14159265 / 180.0;
                float cosA = cos(rad);
                float sinA = sin(rad);
                
                float x = i.uv.x * cosA - i.uv.y * sinA;
                float y = i.uv.x * sinA + i.uv.y * cosA;

                // --- Gabor math (Davidson: Gaussian-windowed grating) ---
                float gaussian = exp(-(x*x + y*y) / (2.0 * _Sigma * _Sigma));
                float grating  = cos(2.0 * 3.14159265 * _Frequency * x + _Phase);
                float val = 0.5 + 0.5 * gaussian * grating * _Contrast;

                // --- Circular patch mask ---
                // The visible disk is _DiskRadius (UV; the quad spans 1.0). The Gaussian
                // has already faded to grey well inside this radius, so the result is a
                // small Gabor on a small grey disk (close to Davidson's HMD presentation)
                // rather than a tiny Gabor lost in the whole 19.85-deg quad. Soft edge.
                float dist = length(i.uv);
                float alpha = 1.0 - smoothstep(_DiskRadius - 0.02, _DiskRadius, dist);

                return fixed4(val, val, val, alpha);
            }
            ENDCG
        }
    }
}