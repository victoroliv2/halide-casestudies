#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdalign.h>
#include <omp.h>

#include "opencl_help.h"

inline static int clamp(int x, int low, int high)
{
  return (x < low) ? low : ((x > high) ? high : x);
}

void
unsharped_mask (float const * __restrict__ _input,
                float       * __restrict__ _output,
                int    width,
                int    height,
                float  sigma,
                float  detail_thresh,
                float  sharpen)

{
  const int radius = int(3*sigma + 1.0f);

  const int channels = 4;

  float gaussian_mask[2 * radius + 1];

  for (int r=-radius; r<=radius; r++)
    gaussian_mask[r+radius] = expf(-(float(r)/sigma)*(float(r)/sigma)*0.5f);

  float * __restrict__ input     = (float * __restrict__) __builtin_assume_aligned(_input,  32);
  float * __restrict__ output    = (float * __restrict__) __builtin_assume_aligned(_output, 32);

  float * __restrict__ blurx;
  float * __restrict__ blury;

  posix_memalign((void **) &blurx, 32, width * height * channels * sizeof(float));
  posix_memalign((void **) &blury, 32, width * height * channels * sizeof(float));

  #define  INPUT(x,y,c)  input[c+channels*(x + width * y)]
  #define OUTPUT(x,y,c) output[c+channels*(x + width * y)]
  #define  BLURX(x,y,c)  blurx[c+channels*(x + width * y)]
  #define  BLURY(x,y,c)  blury[c+channels*(x + width * y)]

  for (int k=0; k < (width * height * channels); k++)
    {
      input[k] = 0.0f;
      blurx[k] = 0.0f;
      blury[k] = 0.0f;
    }

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      for(int c = 0; c < channels; c++)
        {
          for (int r=-radius; r<=radius; r++)
            BLURX(x,y,c) += gaussian_mask[r+radius] * INPUT(clamp(x+r, 0, width-1), y, c);
        }

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      for(int c = 0; c < channels; c++)
        {
          for (int r=-radius; r<=radius; r++)
            BLURY(x,y,c) += gaussian_mask[r+radius] * BLURX(x, clamp(y+r, 0, height-1), c);
        }

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      for(int c = 0; c < channels; c++)
        {
          float detail = BLURY(x,y,c) - INPUT(x,y,c);
          float sharpened = (detail <= detail_thresh)?
                            sharpen * copysign(fmax(fabs(detail) - detail_thresh, 0.0f), detail) :
                            0.0f;
          OUTPUT(x,y,c) = INPUT(x,y,c) + sharpened;
        }
}

#include <stdio.h>

#include "kernel/unsharped_mask.cl.h"

static cl_kernel *kernel;

void
unsharped_mask_cl_prepare()
{
  const char *kernel_name[] = { "blurx",
                                "blury",
                                "unsharped_mask",
                                NULL };

  kernel = cl_compile_and_build(unsharped_mask_cl_source, kernel_name);
}

void
unsharped_mask_cl (cl_mem A,
                   cl_mem B,
                   int    width,
                   int    height,
                   float  sigma,
                   float  detail_thresh,
                   float  sharpen)
{
  cl_int cl_err = 0;

  const int radius = int(3*sigma + 1.0f);

  float gaussian_mask[2 * radius + 1];

  for (int r=-radius; r<=radius; r++)
    gaussian_mask[r+radius] = expf(-(r/sigma)*(r/sigma)*0.5f);

  cl_mem m_gaussian_mask = clCreateBuffer (cl_state.context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (2 * radius + 1) * sizeof(cl_float),
                              gaussian_mask,
                              &cl_err);
  CL_CHECK;

  size_t global_ws[2];

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[0])
  CL_ARG(cl_mem,     A);
  CL_ARG(cl_mem,     B);
  CL_ARG(cl_mem,     m_gaussian_mask);
  CL_ARG(cl_int,     radius);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[0], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[1])
  CL_ARG(cl_mem,     B);
  CL_ARG(cl_mem,     A);
  CL_ARG(cl_mem,     m_gaussian_mask);
  CL_ARG(cl_int,     radius);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[1], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[2])
  CL_ARG(cl_mem,     A);
  CL_ARG(cl_mem,     B);
  CL_ARG(cl_mem,     B);
  CL_ARG(cl_float,   detail_thresh);
  CL_ARG(cl_float,   sharpen);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[2], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;

  CL_RELEASE(m_gaussian_mask);
}
