
__kernel void rgba_to_cielab(const __global float4 *input,
                                   __global float4 *output,
                                   int     width)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 in_v = input[y*width+x];

    float R, G, B, X, Y, Z;

    R = in_v.x;
    G = in_v.y;
    B = in_v.z;

    if ( R > 0.04045f ) R = tgamma((R + 0.055f ) / 1.055f);
    else                R = R / 12.92f;
    if ( G > 0.04045f ) G = tgamma((G + 0.055f ) / 1.055f );
    else                G = G / 12.92f;
    if ( B > 0.04045f ) B = tgamma((B + 0.055f ) / 1.055f );
    else                B = B / 12.92f;

    R = R * 100.0f;
    G = G * 100.0f;
    B = B * 100.0f;

    //Observer. = 2°, Illuminant = D65
    X = R * 0.4124f + G * 0.3576f + B * 0.1805f;
    Y = R * 0.2126f + G * 0.7152f + B * 0.0722f;
    Z = R * 0.0193f + G * 0.1192f + B * 0.9505f;

    X = X / 95.047f;   //Observer= 2°, Illuminant= D65
    Y = Y / 100.000f;
    Z = Z / 108.883f;

    if ( X > 0.008856f ) X = cbrt(X);
    else                 X = ( 7.787f * X ) + ( 16.0f / 116.0f );
    if ( Y > 0.008856f ) Y = cbrt(Y);
    else                 Y = ( 7.787f * Y ) + ( 16.0f / 116.0f );
    if ( Z > 0.008856f ) Z = cbrt(Z);
    else                 Z = ( 7.787f * Z ) + ( 16.0f / 116.0f );

    float CIEL = ( 116.0f * Y ) - 16.0f;
    float CIEa = 500.0f * ( X - Y );
    float CIEb = 200.0f * ( Y - Z );

    float4 out_v;

    out_v.x = CIEL;
    out_v.y = CIEa;
    out_v.z = CIEb;
    out_v.w = in_v.w;

    output[y*width+x] = out_v;
}
