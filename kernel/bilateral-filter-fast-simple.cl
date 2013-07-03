/* This file is part of GEGL
 *
 * GEGL is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GEGL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GEGL; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright 2013 Victor Oliveira (victormatheus@gmail.com)
 */

#define GRID(x,y,z) grid[x+sw*(y + z * sh)]

__kernel void bilateral_init(__global float8 *grid,
                             int sw,
                             int sh,
                             int depth)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  for (int d=0; d<depth; d++)
    {
      GRID(gid_x,gid_y,d) = (float8)(0.0f);
    }
}

__kernel void bilateral_downsample(__global const float4 *input,
                                   __global       float2 *grid,
                                   int width,
                                   int height,
                                   int sw,
                                   int sh,
                                   int   s_sigma,
                                   float r_sigma)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  for (int ry=0; ry < s_sigma; ry++)
    for (int rx=0; rx < s_sigma; rx++)
      {
        const int x = clamp(gid_x * s_sigma - s_sigma/2 + rx, 0, width -1);
        const int y = clamp(gid_y * s_sigma - s_sigma/2 + ry, 0, height-1);

        const float4 val = input[y * width + x];

        const int4 z = convert_int4(val * (1.0f/r_sigma) + 0.5f);

        grid[4*(gid_x+sw*(gid_y + z.x * sh))+0] += (float2)(val.x, 1.0f);
        grid[4*(gid_x+sw*(gid_y + z.y * sh))+1] += (float2)(val.y, 1.0f);
        grid[4*(gid_x+sw*(gid_y + z.z * sh))+2] += (float2)(val.z, 1.0f);
        grid[4*(gid_x+sw*(gid_y + z.w * sh))+3] += (float2)(val.w, 1.0f);

        barrier (CLK_GLOBAL_MEM_FENCE);
      }
}

__kernel void bilateral_blur_x(__global const float8 *grid,
                               __global       float8 *blurx,
                               int sw,
                               int sh,
                               int depth)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  for (int d=0; d<depth; d++)
    {
      const int xp = max(x - 1, 0);
      const int xn = min(x + 1, sw-1);

      float8 v =        grid[xp+sw*(y + d * sh)] +
                 4.0f * grid[x +sw*(y + d * sh)] +
                        grid[xn+sw*(y + d * sh)];

      blurx[x+sw*(y + d * sh)] = v;
    }
}

__kernel void bilateral_blur_y(__global const float8 *blurx,
                               __global       float8 *blury,
                               int sw,
                               int sh,
                               int depth)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  for (int d=0; d<depth; d++)
    {
      const int yp = max(y - 1, 0);
      const int yn = min(y + 1, sh-1);

      float8 v =        blurx[x+sw*(yp + d * sh)] +
                 4.0f * blurx[x+sw*(y  + d * sh)] +
                        blurx[x+sw*(yn + d * sh)];

      blury[x+sw*(y + d * sh)] = v;
    }
}

__kernel void bilateral_blur_z(__global const float8 *blury,
                               __global       float2 *blurz_r,
                               __global       float2 *blurz_g,
                               __global       float2 *blurz_b,
                               __global       float2 *blurz_a,
                               int sw,
                               int sh,
                               int depth)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  for (int d=0; d<depth; d++)
    {
      const int dp = max(d - 1, 0);
      const int dn = min(d + 1, depth-1);

      float8 v =        blury[x+sw*(y + dp * sh)] +
                 4.0f * blury[x+sw*(y + d  * sh)] +
                        blury[x+sw*(y + dn * sh)];

      blurz_r[x+sw*(y + d * sh)] = v.s01;
      blurz_g[x+sw*(y + d * sh)] = v.s23;
      blurz_b[x+sw*(y + d * sh)] = v.s45;
      blurz_a[x+sw*(y + d * sh)] = v.s67;
    }
}

__kernel void bilateral_interpolate(__global    const float4  *input,
                                    __global    const float2  *blurz_r,
                                    __global    const float2  *blurz_g,
                                    __global    const float2  *blurz_b,
                                    __global    const float2  *blurz_a,
                                    __global          float4  *smoothed,
                                    int   width,
                                    int   sw,
                                    int   sh,
                                    int   depth,
                                    int   s_sigma,
                                    float r_sigma)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const float  xf = (float)(x) / s_sigma;
  const float  yf = (float)(y) / s_sigma;
  const float4 zf = input[y * width + x] / r_sigma;

  float8 val;

  int  x1 = (int)xf;
  int  y1 = (int)yf;
  int4 z1 = convert_int4(zf);

  int  x2 = min(x1+1, sw-1);
  int  y2 = min(y1+1, sh-1);
  int4 z2 = min(z1+1, depth-1);

  float  x_alpha = xf - x1;
  float  y_alpha = yf - y1;
  float4 z_alpha = zf - floor(zf);

  #define BLURZ_R(x,y,z) blurz_r[x+sw*(y+z*sh)]
  #define BLURZ_G(x,y,z) blurz_g[x+sw*(y+z*sh)]
  #define BLURZ_B(x,y,z) blurz_b[x+sw*(y+z*sh)]
  #define BLURZ_A(x,y,z) blurz_a[x+sw*(y+z*sh)]

  val.s04 = mix(mix(mix(BLURZ_R(x1, y1, z1.x), BLURZ_R(x2, y1, z1.x), x_alpha),
                    mix(BLURZ_R(x1, y2, z1.x), BLURZ_R(x2, y2, z1.x), x_alpha), y_alpha),
                mix(mix(BLURZ_R(x1, y1, z2.x), BLURZ_R(x2, y1, z2.x), x_alpha),
                    mix(BLURZ_R(x1, y2, z2.x), BLURZ_R(x2, y2, z2.x), x_alpha), y_alpha), z_alpha.x);

  val.s15 = mix(mix(mix(BLURZ_G(x1, y1, z1.y), BLURZ_G(x2, y1, z1.y), x_alpha),
                    mix(BLURZ_G(x1, y2, z1.y), BLURZ_G(x2, y2, z1.y), x_alpha), y_alpha),
                mix(mix(BLURZ_G(x1, y1, z2.y), BLURZ_G(x2, y1, z2.y), x_alpha),
                    mix(BLURZ_G(x1, y2, z2.y), BLURZ_G(x2, y2, z2.y), x_alpha), y_alpha), z_alpha.y);

  val.s26 = mix(mix(mix(BLURZ_B(x1, y1, z1.z), BLURZ_B(x2, y1, z1.z), x_alpha),
                    mix(BLURZ_B(x1, y2, z1.z), BLURZ_B(x2, y2, z1.z), x_alpha), y_alpha),
                mix(mix(BLURZ_B(x1, y1, z2.z), BLURZ_B(x2, y1, z2.z), x_alpha),
                    mix(BLURZ_B(x1, y2, z2.z), BLURZ_B(x2, y2, z2.z), x_alpha), y_alpha), z_alpha.z);

  val.s37 = mix(mix(mix(BLURZ_A(x1, y1, z1.w), BLURZ_A(x2, y1, z1.w), x_alpha),
                    mix(BLURZ_A(x1, y2, z1.w), BLURZ_A(x2, y2, z1.w), x_alpha), y_alpha),
                mix(mix(BLURZ_A(x1, y1, z2.w), BLURZ_A(x2, y1, z2.w), x_alpha),
                    mix(BLURZ_A(x1, y2, z2.w), BLURZ_A(x2, y2, z2.w), x_alpha), y_alpha), z_alpha.w);

  smoothed[y * width + x] = val.s0123/val.s4567;
}
