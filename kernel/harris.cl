const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP |
                          CLK_FILTER_NEAREST;

float gray(float4 v)
{
  return 0.299f * v.x + 0.587f * v.y + 0.114f * v.z;
}

kernel void sobel(read_only  image2d_t input,
                  write_only image2d_t sobelx,
                  write_only image2d_t sobely)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float h = + 1.0f * gray(read_imagef(input, sampler, (int2)(x-1, y-1))) - 1.0f * gray(read_imagef(input, sampler, (int2)(x+1, y-1)))
            + 2.0f * gray(read_imagef(input, sampler, (int2)(x-1, y  ))) - 2.0f * gray(read_imagef(input, sampler, (int2)(x+1, y  )))
            + 1.0f * gray(read_imagef(input, sampler, (int2)(x-1, y+1))) - 1.0f * gray(read_imagef(input, sampler, (int2)(x+1, y+1)));

  float v = + 1.0f * gray(read_imagef(input, sampler, (int2)(x-1, y-1))) + 2.0f * gray(read_imagef(input, sampler, (int2)(x, y-1))) + 1.0f * gray(read_imagef(input, sampler, (int2)(x+1, y-1)))
            - 1.0f * gray(read_imagef(input, sampler, (int2)(x-1, y+1))) - 2.0f * gray(read_imagef(input, sampler, (int2)(x, y+1))) - 1.0f * gray(read_imagef(input, sampler, (int2)(x+1, y+1)));

  write_imagef(sobelx, (int2)(x, y), h);
  write_imagef(sobely, (int2)(x, y), v);
}

kernel void blurx(read_only  image2d_t sobelx,
                  read_only  image2d_t sobely,
                  write_only image2d_t blurxx_h,
                  write_only image2d_t blurxy_h,
                  write_only image2d_t bluryy_h,
                  constant float * gmask,
                  int radius)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float vxx = 0.0f;
  float vxy = 0.0f;
  float vyy = 0.0f;

  for(int r=-radius; r <= radius; r++)
    {
      float sx = read_imagef(sobelx, sampler, (int2)(x+r, y)).x;
      float sy = read_imagef(sobely, sampler, (int2)(x+r, y)).x;

      vxx += gmask[r+radius] * (sx * sx);
      vxy += gmask[r+radius] * (sx * sy);
      vyy += gmask[r+radius] * (sy * sy);
    }

  write_imagef(blurxx_h, (int2)(x, y), (float4)(vxx));
  write_imagef(blurxy_h, (int2)(x, y), (float4)(vxy));
  write_imagef(bluryy_h, (int2)(x, y), (float4)(vyy));
}

kernel void cornerness(read_only  image2d_t blurxx_h,
                       read_only  image2d_t blurxy_h,
                       read_only  image2d_t bluryy_h,
                       write_only image2d_t corner,
                       constant float * gmask,
                       int radius,
                       float threshold)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float vxx = 0.0f;
  float vxy = 0.0f;
  float vyy = 0.0f;

  for(int r=-radius; r <= radius; r++)
    {
      vxx += gmask[r+radius] * read_imagef(blurxx_h, sampler, (int2)(x, y+r)).x;
      vxy += gmask[r+radius] * read_imagef(blurxy_h, sampler, (int2)(x, y+r)).x;
      vyy += gmask[r+radius] * read_imagef(bluryy_h, sampler, (int2)(x, y+r)).x;
    }

  float det = vxx * vyy - vxy * vxy;
  float trace = vxx + vyy;

  const float k = 0.04f;

  float response = det - k * (trace * trace);

  write_imagef(corner, (int2)(x, y), (response > threshold)? (float4)(response) : (float4)(-FLT_MAX));
}

kernel void cornerness_suppress(read_only  image2d_t corner,
                                write_only image2d_t out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float corner_v = read_imagef(corner, sampler, (int2)(x, y)).x;

  bool maximal = true;
  for (int yy=-1; yy<=1; yy++)
    for (int xx=-1; xx<=1; xx++)
      if (read_imagef(corner, sampler, (int2)(x+xx, y+yy)).x > corner_v)
        maximal = false;

  write_imagef(out, (int2)(x, y), (maximal)? 1.0f : 0.0f);
}
