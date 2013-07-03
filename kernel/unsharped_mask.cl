const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP |
                          CLK_FILTER_NEAREST;

kernel void blurx(read_only  image2d_t in,
                  write_only image2d_t out,
                  constant float * gmask,
                  int radius)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v = 0.0f;

  for(int r=-radius; r <= radius; r++)
    {
      v += gmask[r+radius] * read_imagef(in, sampler, (int2)(x+r, y));
    }

  write_imagef(out, (int2)(x, y), v);
}

kernel void blury(read_only  image2d_t in,
                  write_only image2d_t out,
                  constant float * gmask,
                  int radius)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v = 0.0f;

  for(int r=-radius; r <= radius; r++)
    {
      v += gmask[r+radius] * read_imagef(in, sampler, (int2)(x, y+r));
    }

  write_imagef(out, (int2)(x, y), v);
}

kernel void unsharped_mask(read_only  image2d_t in,
                           read_only  image2d_t blurry,
                           write_only image2d_t out,
                           float detail_thresh,
                           float sharpen)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 in_v = read_imagef(in, sampler, (int2)(x, y));

  float4 detail = read_imagef(blurry, sampler, (int2)(x, y)) - in_v;

  float4 sharpened = sharpen * copysign(fmax(fabs(detail) - detail_thresh, 0.0f), detail) * convert_float4(detail > detail_thresh);

  write_imagef(out, (int2)(x, y), in_v + sharpened);
}
