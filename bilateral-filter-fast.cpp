/* This file is an image processing operation for GEGL
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
 * Copyright 2012 Victor Oliveira <victormatheus@gmail.com>
 */

 /* This is an implementation of a fast approximated bilateral filter
  * algorithm descripted in:
  *
  *  A Fast Approximation of the Bilateral Filter using a Signal Processing Approach
  *  Sylvain Paris and Frédo Durand
  *  European Conference on Computer Vision (ECCV'06)
  */

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdalign.h>
#include <omp.h>

#include "opencl_help.h"

inline static float lerp(float a, float b, float v)
{
  return (1.0f - v) * a + v * b;
}

inline static int clamp(int x, int low, int high)
{
  return (x < low) ? low : ((x > high) ? high : x);
}

#include <stdio.h>

void
bilateral_filter (const float * __restrict__ _input,
                  float       * __restrict__ _output,
                  int   width,
                  int   height,
                  int   s_sigma,
                  float r_sigma)
{
  const int padding_xy = 2;
  const int padding_z  = 2;

  const int channels = 4;

  const int sw = (width -1) / s_sigma + 1 + (2 * padding_xy);
  const int sh = (height-1) / s_sigma + 1 + (2 * padding_xy);
  const int depth = (int)(1.0f / r_sigma) + 1 + (2 * padding_z);

  /* down-sampling */

  float * __restrict__ input  = (float * __restrict__) __builtin_assume_aligned(_input,  32);
  float * __restrict__ output = (float * __restrict__) __builtin_assume_aligned(_output, 32);

  float * __restrict__ grid ;
  float * __restrict__ blurx;
  float * __restrict__ blury;
  float * __restrict__ blurz;

  posix_memalign((void **) &grid , 32, sw * sh * depth * channels * 2 * sizeof(float));
  posix_memalign((void **) &blurx, 32, sw * sh * depth * channels * 2 * sizeof(float));
  posix_memalign((void **) &blury, 32, sw * sh * depth * channels * 2 * sizeof(float));
  posix_memalign((void **) &blurz, 32, sw * sh * depth * channels * 2 * sizeof(float));

  #define INPUT(x,y,c) input[c+channels*(x + width * y)]

  #define  GRID(x,y,z,c,i) grid [i+2*(c+channels*(x+sw*(y+z*sh)))]
  #define BLURX(x,y,z,c,i) blurx[i+2*(c+channels*(x+sw*(y+z*sh)))]
  #define BLURY(x,y,z,c,i) blury[i+2*(c+channels*(x+sw*(y+z*sh)))]
  #define BLURZ(x,y,z,c,i) blurz[i+2*(c+channels*(x+sw*(y+z*sh)))]

  for (int k=0; k < (sw * sh * depth * channels * 2); k++)
    {
      grid [k] = 0.0f;
      blurx[k] = 0.0f;
      blury[k] = 0.0f;
      blurz[k] = 0.0f;
    }

#if 0
  /* in case we want to normalize the color space */

  float input_min[4] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
  float input_max[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

  for(y = 0; y < height; y++)
    for(x = 0; x < width; x++)
      for(c = 0; c < channels; c++)
        {
          input_min[c] = MIN(input_min[c], INPUT(x,y,c));
          input_max[c] = MAX(input_max[c], INPUT(x,y,c));
        }
#endif

  /* downsampling */

  for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
      for(int c = 0; c < channels; c++)
        {
          const float z = INPUT(x,y,c); // - input_min[c];

          const int small_x = (int)((float)(x) / s_sigma + 0.5f) + padding_xy;
          const int small_y = (int)((float)(y) / s_sigma + 0.5f) + padding_xy;
          const int small_z = (int)((float)(z) / r_sigma + 0.5f) + padding_z;

          assert(small_x >= 0 && small_x < sw);
          assert(small_y >= 0 && small_y < sh);
          assert(small_z >= 0 && small_z < depth);

          GRID(small_x, small_y, small_z, c, 0) += INPUT(x, y, c);
          GRID(small_x, small_y, small_z, c, 1) += 1.0f;
        }

  /* blur in x, y and z */
  /* XXX: we could use less memory, but at expense of code readability */

  #pragma omp parallel for
  for (int z = 1; z < depth-1; z++)
    for (int y = 1; y < sh-1; y++)
      for (int x = 1; x < sw-1; x++)
        for(int c = 0; c < channels; c++)
          for (int i=0; i<2; i++)
            BLURX(x, y, z, c, i) = (GRID (x-1, y, z, c, i) + 2.0f * GRID (x, y, z, c, i) + GRID (x+1, y, z, c, i)) / 4.0f;

  #pragma omp parallel for
  for (int z = 1; z < depth-1; z++)
    for (int y = 1; y < sh-1; y++)
      for (int x = 1; x < sw-1; x++)
        for(int c = 0; c < channels; c++)
          for (int i=0; i<2; i++)
            BLURY(x, y, z, c, i) = (BLURX (x, y-1, z, c, i) + 2.0f * BLURX (x, y, z, c, i) + BLURX (x, y+1, z, c, i)) / 4.0f;

  #pragma omp parallel for
  for (int z = 1; z < depth-1; z++)
    for (int y = 1; y < sh-1; y++)
      for (int x = 1; x < sw-1; x++)
        for(int c = 0; c < channels; c++)
          for (int i=0; i<2; i++)
            BLURZ(x, y, z, c, i) = (BLURY (x, y, z-1, c, i) + 2.0f * BLURY (x, y, z, c, i) + BLURY (x, y, z+1, c, i)) / 4.0f;

  /* trilinear filtering */

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      for(int c = 0; c < channels; c++)
        {
          float xf = (float)(x) / s_sigma + padding_xy;
          float yf = (float)(y) / s_sigma + padding_xy;
          float zf = INPUT(x,y,c) / r_sigma + padding_z;

          int x1 = (int)xf;
          int y1 = (int)yf;
          int z1 = (int)zf;

          int x2 = x1+1;
          int y2 = y1+1;
          int z2 = z1+1;

          float x_alpha = xf - x1;
          float y_alpha = yf - y1;
          float z_alpha = zf - z1;

          float interpolated[2];

          assert(xf >= 0 && xf < sw);
          assert(yf >= 0 && yf < sh);
          assert(zf >= 0 && zf < depth);

          for (int i=0; i<2; i++)
              interpolated[i] =
              lerp(lerp(lerp(BLURZ(x1, y1, z1, c, i), BLURZ(x2, y1, z1, c, i), x_alpha),
                        lerp(BLURZ(x1, y2, z1, c, i), BLURZ(x2, y2, z1, c, i), x_alpha), y_alpha),
                   lerp(lerp(BLURZ(x1, y1, z2, c, i), BLURZ(x2, y1, z2, c, i), x_alpha),
                        lerp(BLURZ(x1, y2, z2, c, i), BLURZ(x2, y2, z2, c, i), x_alpha), y_alpha), z_alpha);

          output[channels*(y*width+x)+c] = interpolated[0] / interpolated[1];
        }

  free (grid);
  free (blurx);
  free (blury);
  free (blurz);
}

#include "kernel/bilateral-filter-fast.cl.h"

static cl_kernel *kernel;

void
bilateral_filter_cl_prepare()
{
  const char *kernel_name[] = { "bilateral_init",
                                "bilateral_downsample",
                                "bilateral_blur",
                                "bilateral_interpolate",
                                "bilateral_downsample2",
                                NULL };

  kernel = cl_compile_and_build(bilateral_filter_fast_cl_source, kernel_name);
}

void
bilateral_filter_cl (cl_mem input,
                     cl_mem output,
                     int    width,
                     int    height,
                     int    s_sigma,
                     float  r_sigma)
{
  cl_int cl_err = 0;

  int c;

  const int sw = (width -1) / s_sigma + 1;
  const int sh = (height-1) / s_sigma + 1;
  const int depth = (int)(1.0f / r_sigma) + 1;

  size_t global_ws[2];
  size_t local_ws[2];

  cl_mem grid   = NULL;
  cl_mem blur[4] = {NULL, NULL, NULL, NULL};
  cl_mem blur_tex[4] = {NULL, NULL, NULL, NULL};

  cl_image_format format = {CL_RG, CL_FLOAT};

  grid = clCreateBuffer (cl_state.context,
                         CL_MEM_READ_WRITE,
                         sw * sh * depth * sizeof(cl_float8),
                         NULL,
                         &cl_err);
  CL_CHECK;

  for(c = 0; c < 4; c++)
    {
      blur[c] = clCreateBuffer (cl_state.context,
                                CL_MEM_WRITE_ONLY,
                                sw * sh * depth * sizeof(cl_float2),
                                NULL, &cl_err);
      CL_CHECK;

      blur_tex[c] = clCreateImage3D (cl_state.context,
                                     CL_MEM_READ_ONLY,
                                     &format,
                                     sw, sh, depth,
                                     0, 0, NULL, &cl_err);
      CL_CHECK;
    }

#if 0
  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel[0])
  CL_ARG(cl_mem,   grid)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                       kernel[0], 2,
                                       NULL, global_ws, NULL,
                                       0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel[1])
  CL_ARG(cl_mem,   input)
  CL_ARG(cl_mem,   grid)
  CL_ARG(cl_int,   width)
  CL_ARG(cl_int,   height)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   s_sigma)
  CL_ARG(cl_float, r_sigma)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                       kernel[1], 2,
                                       NULL, global_ws, NULL,
                                       0, NULL, NULL);
  CL_CHECK;
  }
#else
  {
  local_ws[0] = 8;
  local_ws[1] = 8;

  global_ws[0] = ((sw + local_ws[0]-1)/local_ws[0])*local_ws[0];
  global_ws[1] = ((sh + local_ws[1]-1)/local_ws[1])*local_ws[1];

  CL_ARG_START(kernel[4])
  CL_ARG(cl_mem,   input)
  CL_ARG(cl_mem,   grid)
  CL_ARG(cl_int,   width)
  CL_ARG(cl_int,   height)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG(cl_int,   s_sigma)
  CL_ARG(cl_float, r_sigma)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                       kernel[4], 2,
                                       NULL, global_ws, local_ws,
                                       0, NULL, NULL);
  CL_CHECK;
  }
#endif

  {
  local_ws[0] = 16;
  local_ws[1] = 16;

  global_ws[0] = ((sw + local_ws[0]-1)/local_ws[0])*local_ws[0];
  global_ws[1] = ((sh + local_ws[1]-1)/local_ws[1])*local_ws[1];

  CL_ARG_START(kernel[2])
  CL_ARG(cl_mem, grid)
  CL_ARG(cl_mem, blur[0])
  CL_ARG(cl_mem, blur[1])
  CL_ARG(cl_mem, blur[2])
  CL_ARG(cl_mem, blur[3])
  CL_ARG(cl_int, sw)
  CL_ARG(cl_int, sh)
  CL_ARG(cl_int, depth)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                       kernel[2], 2,
                                       NULL, global_ws, local_ws,
                                       0, NULL, NULL);
  CL_CHECK;
  }

  for(c = 0; c < 4; c++)
    {
      const size_t dst_origin[3] = {0, 0, 0};
      const size_t dst_region[3] = {sw, sh, depth};

      cl_err = clEnqueueCopyBufferToImage (cl_state.command_queue,
                                           blur[c],
                                           blur_tex[c],
                                           0, dst_origin, dst_region,
                                           0, NULL, NULL);
      CL_CHECK;
    }

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[3])
  CL_ARG(cl_mem,   input)
  CL_ARG(cl_mem,   blur_tex[0])
  CL_ARG(cl_mem,   blur_tex[1])
  CL_ARG(cl_mem,   blur_tex[2])
  CL_ARG(cl_mem,   blur_tex[3])
  CL_ARG(cl_mem,   output)
  CL_ARG(cl_int,   width)
  CL_ARG(cl_int,   s_sigma)
  CL_ARG(cl_float, r_sigma)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[3], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  CL_CHECK;

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;

  CL_RELEASE(grid);

  for(c = 0; c < 4; c++)
    {
      CL_RELEASE(blur[c]);
      CL_RELEASE(blur_tex[c]);
    }
}

#include "kernel/bilateral-filter-fast-simple.cl.h"

static cl_kernel *kernel_simple;

void
bilateral_filter_cl_simple_prepare()
{
  const char *kernel_name[] = { "bilateral_init",
                                "bilateral_downsample",
                                "bilateral_blur_x",
                                "bilateral_blur_y",
                                "bilateral_blur_z",
                                "bilateral_interpolate",
                                NULL };

  kernel_simple = cl_compile_and_build(bilateral_filter_fast_simple_cl_source, kernel_name);
}

void
bilateral_filter_cl_simple (cl_mem input,
                            cl_mem output,
                            int    width,
                            int    height,
                            int    s_sigma,
                            float  r_sigma)
{
  cl_int cl_err = 0;

  int c;

  const int sw = (width -1) / s_sigma + 1;
  const int sh = (height-1) / s_sigma + 1;
  const int depth = (int)(1.0f / r_sigma) + 1;

  size_t global_ws[2];
  size_t local_ws[2];

  cl_mem grid   = NULL;
  cl_mem blurx  = NULL;
  cl_mem blury  = NULL;
  cl_mem blurz[4] = {NULL, NULL, NULL, NULL};

  grid = clCreateBuffer (cl_state.context,
                         CL_MEM_READ_WRITE,
                         sw * sh * depth * sizeof(cl_float8),
                         NULL,
                         &cl_err);
  CL_CHECK;

  blurx = clCreateBuffer (cl_state.context,
                         CL_MEM_READ_WRITE,
                         sw * sh * depth * sizeof(cl_float8),
                         NULL,
                         &cl_err);
  CL_CHECK;

  blury = clCreateBuffer (cl_state.context,
                         CL_MEM_READ_WRITE,
                         sw * sh * depth * sizeof(cl_float8),
                         NULL,
                         &cl_err);
  CL_CHECK;

  for(c = 0; c < 4; c++)
    {
      blurz[c] = clCreateBuffer (cl_state.context,
                                CL_MEM_WRITE_ONLY,
                                sw * sh * depth * sizeof(cl_float2),
                                NULL, &cl_err);
      CL_CHECK;
    }

  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel_simple[0])
  CL_ARG(cl_mem,   grid)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel_simple[0], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel_simple[1])
  CL_ARG(cl_mem,   input)
  CL_ARG(cl_mem,   grid)
  CL_ARG(cl_int,   width)
  CL_ARG(cl_int,   height)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   s_sigma)
  CL_ARG(cl_float, r_sigma)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel_simple[1], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel_simple[2])
  CL_ARG(cl_mem,   grid)
  CL_ARG(cl_mem,   blurx)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel_simple[2], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel_simple[3])
  CL_ARG(cl_mem,   blurx)
  CL_ARG(cl_mem,   blury)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel_simple[3], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = sw;
  global_ws[1] = sh;

  CL_ARG_START(kernel_simple[4])
  CL_ARG(cl_mem,   blury)
  CL_ARG(cl_mem,   blurz[0])
  CL_ARG(cl_mem,   blurz[1])
  CL_ARG(cl_mem,   blurz[2])
  CL_ARG(cl_mem,   blurz[3])
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel_simple[4], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel_simple[5])
  CL_ARG(cl_mem,   input)
  CL_ARG(cl_mem,   blurz[0])
  CL_ARG(cl_mem,   blurz[1])
  CL_ARG(cl_mem,   blurz[2])
  CL_ARG(cl_mem,   blurz[3])
  CL_ARG(cl_mem,   output)
  CL_ARG(cl_int,   width)
  CL_ARG(cl_int,   sw)
  CL_ARG(cl_int,   sh)
  CL_ARG(cl_int,   depth)
  CL_ARG(cl_int,   s_sigma)
  CL_ARG(cl_float, r_sigma)
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel_simple[5], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  CL_CHECK;

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;

  CL_RELEASE(grid);
  CL_RELEASE(blurx);
  CL_RELEASE(blury);

  for(c = 0; c < 4; c++)
    {
      CL_RELEASE(blurz[c]);
    }
}
