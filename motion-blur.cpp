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

#include <stdio.h>

void
motion_blur (float const * __restrict__ _input,
             float       * __restrict__ _output,
             int    width,
             int    height,
             float  length,
             float  angle)
{
  const int channels = 4;

  float theta = angle * M_PI / 180.0;
  float offset_x = length * cos(theta);
  float offset_y = length * sin(theta);
  int num_steps = ceil(length) + 1;

  float * __restrict__ input     = (float * __restrict__) __builtin_assume_aligned(_input,  32);
  float * __restrict__ output    = (float * __restrict__) __builtin_assume_aligned(_output, 32);

  #define INPUT(x,y,c) input[c+channels*(x + width * y)]

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      for(int c = 0; c < channels; c++)
        {
          float sum = 0.0;

          for(int step = 0; step < num_steps; step++)
            {
              float t = num_steps == 1 ? 0.0f :
                  step / (float)(num_steps - 1) - 0.5f;

              float xx = x + t * offset_x;
              float yy = y + t * offset_y;

              int ix = (int)(xx);
              int iy = (int)(yy);

              float dx = xx - ix;
              float dy = yy - iy;

              float mixy0, mixy1, pix0, pix1, pix2, pix3;

              pix0 = INPUT(clamp(ix,   0, width-1), clamp(iy,   0, height-1), c);
              pix1 = INPUT(clamp(ix+1, 0, width-1), clamp(iy,   0, height-1), c);
              pix2 = INPUT(clamp(ix,   0, width-1), clamp(iy+1, 0, height-1), c);
              pix3 = INPUT(clamp(ix+1, 0, width-1), clamp(iy+1, 0, height-1), c);

              mixy0 = dy * (pix2 - pix0) + pix0;
              mixy1 = dy * (pix3 - pix1) + pix1;

              sum  += dx * (mixy1 - mixy0) + mixy0;
            }

          output[c+channels*(x + y * width)] = sum / num_steps;
        }
}

#include "kernel/motion-blur.cl.h"

static cl_kernel *kernel;

void
motion_blur_cl_prepare()
{
  const char *kernel_name[] = { "motion_blur",
                                NULL };

  kernel = cl_compile_and_build(motion_blur_cl_source, kernel_name);
}

void
motion_blur_cl (cl_mem input,
                cl_mem output,
                int    width,
                int    height,
                float  length,
                float  angle)
{
  cl_int cl_err = 0;

  float theta = angle * M_PI / 180.0;
  float offset_x = fabs(length * cos(theta));
  float offset_y = fabs(length * sin(theta));
  int num_steps = ceil(length) + 1;

  size_t global_ws[2];

  /* set local size */

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[0])
  CL_ARG(cl_mem,   input);
  CL_ARG(cl_mem,   output);
  CL_ARG(cl_int,   width);
  CL_ARG(cl_int,   height);
  CL_ARG(cl_int,   num_steps);
  CL_ARG(cl_float, offset_x);
  CL_ARG(cl_float, offset_y);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[0], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }
}
