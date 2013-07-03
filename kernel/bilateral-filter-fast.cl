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

#define LOCAL_W 8
#define LOCAL_H 8

/* found by trial and error */

#define DEPTH_CHUNK 12

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void bilateral_downsample2(__global const float4 *input,
                                    __global       float8 *grid,
                                    int width,
                                    int height,
                                    int sw,
                                    int sh,
                                    int depth,
                                    int   s_sigma,
                                    float r_sigma)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  __local float8 grid_chunk[DEPTH_CHUNK][LOCAL_H][LOCAL_W];

  if (gid_x > sw || gid_y > sh) return;

  for (int d = 0; d < depth; d+=DEPTH_CHUNK)
    {
      for (int k=0; k < DEPTH_CHUNK; k++)
        {
          grid_chunk[k][get_local_id(1)][get_local_id(0)] = (float8)(0.0f);
        }

      barrier (CLK_LOCAL_MEM_FENCE);

      for (int ry=0; ry < s_sigma; ry++)
        for (int rx=0; rx < s_sigma; rx++)
          {
            const int x = clamp(gid_x * s_sigma - s_sigma/2 + rx, 0, width -1);
            const int y = clamp(gid_y * s_sigma - s_sigma/2 + ry, 0, height-1);

            const float4 val = input[y * width + x];

            const int4 z = convert_int4(val * (1.0f/r_sigma) + 0.5f);

            // z >= d && z < d+DEPTH_CHUNK
            int4 inbounds = (z >= d & z < d+DEPTH_CHUNK);

            if (inbounds.x) grid_chunk[z.x-d][get_local_id(1)][get_local_id(0)].s01 += (float2)(val.x, 1.0f);
            if (inbounds.y) grid_chunk[z.y-d][get_local_id(1)][get_local_id(0)].s23 += (float2)(val.y, 1.0f);
            if (inbounds.z) grid_chunk[z.z-d][get_local_id(1)][get_local_id(0)].s45 += (float2)(val.z, 1.0f);
            if (inbounds.w) grid_chunk[z.w-d][get_local_id(1)][get_local_id(0)].s67 += (float2)(val.w, 1.0f);

            barrier (CLK_LOCAL_MEM_FENCE);
          }

      for (int s=d, e=d+min(DEPTH_CHUNK, depth-d); s < e; s++)
        {
          grid[gid_x+sw*(gid_y + s * sh)] = grid_chunk[s-d][get_local_id(1)][get_local_id(0)];
        }
    }
}

#undef LOCAL_W
#undef LOCAL_H

#define LOCAL_W 16
#define LOCAL_H 16

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void bilateral_blur(__global const float8 *grid,
                             __global       float2 *blurz_r,
                             __global       float2 *blurz_g,
                             __global       float2 *blurz_b,
                             __global       float2 *blurz_a,
                             int sw,
                             int sh,
                             int depth)
{
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  const int lx = get_local_id(0);
  const int ly = get_local_id(1);

  float8 vpp = (float8)(0.0f);
  float8 vp  = (float8)(0.0f);
  float8 v   = (float8)(0.0f);

  __local float8 data[LOCAL_H+2][LOCAL_W+2];

  for (int d=0; d<depth; d++)
    {
        for (int ky=get_local_id(1)-1; ky<LOCAL_H+1; ky+=get_local_size(1))
          for (int kx=get_local_id(0)-1; kx<LOCAL_W+1; kx+=get_local_size(0))
            {
              int xx = clamp((int)get_group_id(0)*LOCAL_W+kx, 0, sw-1);
              int yy = clamp((int)get_group_id(1)*LOCAL_H+ky, 0, sh-1);

              data[ky+1][kx+1] = GRID(xx, yy, d);
            }

        barrier (CLK_LOCAL_MEM_FENCE);

      /* blur x */

        data[ly  ][lx+1] = (data[ly  ][lx] + 2.0f * data[ly  ][lx+1] + data[ly  ][lx+2]) / 4.0f;
        data[ly+1][lx+1] = (data[ly+1][lx] + 2.0f * data[ly+1][lx+1] + data[ly+1][lx+2]) / 4.0f;
        data[ly+2][lx+1] = (data[ly+2][lx] + 2.0f * data[ly+2][lx+1] + data[ly+2][lx+2]) / 4.0f;

        barrier (CLK_LOCAL_MEM_FENCE);

      /* blur y */

      if (d==0) {
        v = (data[ly][lx+1] + 2.0f * data[ly+1][lx+1] + data[ly+2][lx+1]) / 4.0f;
        vpp = v;
        vp  = v;
      }
      else {
        vpp = vp;
        vp  = v;

        v = (data[ly][lx+1] + 2.0f * data[ly+1][lx+1] + data[ly+2][lx+1]) / 4.0f;

        float8 blurred = (vpp + 2.0f * vp + v) / 4.0f;

        if (gid_x < sw && gid_y < sh) {
          blurz_r[gid_x+sw*(gid_y+sh*(d-1))] = blurred.s01;
          blurz_g[gid_x+sw*(gid_y+sh*(d-1))] = blurred.s23;
          blurz_b[gid_x+sw*(gid_y+sh*(d-1))] = blurred.s45;
          blurz_a[gid_x+sw*(gid_y+sh*(d-1))] = blurred.s67;
        }
      }
    }

  vpp = vp;
  vp  = v;

  float8 blurred = (vpp + 2.0f * vp + v) / 4.0f;

  if (gid_x < sw && gid_y < sh) {
    blurz_r[gid_x+sw*(gid_y+sh*(depth-1))] = blurred.s01;
    blurz_g[gid_x+sw*(gid_y+sh*(depth-1))] = blurred.s23;
    blurz_b[gid_x+sw*(gid_y+sh*(depth-1))] = blurred.s45;
    blurz_a[gid_x+sw*(gid_y+sh*(depth-1))] = blurred.s67;
  }
}

#undef LOCAL_W
#undef LOCAL_H

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void bilateral_interpolate(__global    const float4    *input,
                                    __read_only       image3d_t  blurz_r,
                                    __read_only       image3d_t  blurz_g,
                                    __read_only       image3d_t  blurz_b,
                                    __read_only       image3d_t  blurz_a,
                                    __global          float4    *smoothed,
                                    int   width,
                                    int   s_sigma,
                                    float r_sigma)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  const float  xf = (float)(x) / s_sigma;
  const float  yf = (float)(y) / s_sigma;
  const float4 zf = input[y * width + x] / r_sigma;

  float8 val;

  val.s04 = (read_imagef (blurz_r, sampler, (float4)(xf, yf, zf.x, 0.0f))).xy;
  val.s15 = (read_imagef (blurz_g, sampler, (float4)(xf, yf, zf.y, 0.0f))).xy;
  val.s26 = (read_imagef (blurz_b, sampler, (float4)(xf, yf, zf.z, 0.0f))).xy;
  val.s37 = (read_imagef (blurz_a, sampler, (float4)(xf, yf, zf.w, 0.0f))).xy;

  smoothed[y * width + x] = val.s0123/val.s4567;
}
