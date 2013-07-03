#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdalign.h>
#include <omp.h>
#include <float.h>

#include "opencl_help.h"

#define SIGMA 1.6f

inline static int clamp(int x, int low, int high)
{
  return (x < low) ? low : ((x > high) ? high : x);
}

#include <stdio.h>

void
harris (float const * __restrict__ _input,
        float       * __restrict__ _output,
        int    width,
        int    height,
        float  threshold)
{
  const int channels = 4;

  float * __restrict__ input     = (float * __restrict__) __builtin_assume_aligned(_input,  32);
  float * __restrict__ output    = (float * __restrict__) __builtin_assume_aligned(_output, 32);

  float * __restrict__ gray;

  float * __restrict__ sobelx_1;
  float * __restrict__ sobely_1;
  float * __restrict__ sobelx_2;
  float * __restrict__ sobely_2;

  float * __restrict__ blurxx_1;
  float * __restrict__ blurxy_1;
  float * __restrict__ bluryy_1;

  float * __restrict__ blurxx_2;
  float * __restrict__ blurxy_2;
  float * __restrict__ bluryy_2;

  float * __restrict__ cornerness;

  posix_memalign((void **) &gray ,    32, width * height * sizeof(float));

  posix_memalign((void **) &sobelx_1, 32, width * height * sizeof(float));
  posix_memalign((void **) &sobely_1, 32, width * height * sizeof(float));
  posix_memalign((void **) &sobelx_2, 32, width * height * sizeof(float));
  posix_memalign((void **) &sobely_2, 32, width * height * sizeof(float));

  posix_memalign((void **) &blurxx_1, 32, width * height * sizeof(float));
  posix_memalign((void **) &bluryy_1, 32, width * height * sizeof(float));
  posix_memalign((void **) &blurxy_1, 32, width * height * sizeof(float));

  posix_memalign((void **) &blurxx_2, 32, width * height * sizeof(float));
  posix_memalign((void **) &bluryy_2, 32, width * height * sizeof(float));
  posix_memalign((void **) &blurxy_2, 32, width * height * sizeof(float));

  posix_memalign((void **) &cornerness, 32, width * height * sizeof(float));

  #define  INPUT(x,y,c) input [c+channels*(x + width * y)]
  #define OUTPUT(x,y,c) output[c+channels*(x + width * y)]

  #define GRAY(x,y)    gray[x + width * y]

  #define SOBELX_1(x,y)  sobelx_1[x + width * y]
  #define SOBELY_1(x,y)  sobely_1[x + width * y]
  #define SOBELX_2(x,y)  sobelx_2[x + width * y]
  #define SOBELY_2(x,y)  sobely_2[x + width * y]

  #define BLURXX_1(x,y)  blurxx_1[x + width * y]
  #define BLURXY_1(x,y)  blurxy_1[x + width * y]
  #define BLURYY_1(x,y)  bluryy_1[x + width * y]

  #define BLURXX_2(x,y)  blurxx_2[x + width * y]
  #define BLURXY_2(x,y)  blurxy_2[x + width * y]
  #define BLURYY_2(x,y)  bluryy_2[x + width * y]

  #define CORNERNESS(x,y)  cornerness[x + width * y]

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      {
        GRAY(x,y) = 0.299f * INPUT(x, y, 0) + 0.587f * INPUT(x, y, 1) + 0.114f * INPUT(x, y, 2);
      }

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      {
        SOBELX_1(x,y) = 0.0f;
        SOBELY_1(x,y) = 0.0f;
        SOBELX_2(x,y) = 0.0f;
        SOBELY_2(x,y) = 0.0f;
        CORNERNESS(x,y) = -FLT_MAX;
      }

  #pragma omp parallel for
  for (int y=1; y < height-1; y++)
    for (int x=1; x < width-1; x++)
      {
        SOBELX_1(x,y) = GRAY(x,y-1) - GRAY(x,y+1);
        SOBELY_1(x,y) = GRAY(x-1,y) + 2.0f*GRAY(x,y) + GRAY(x+1,y);
      }

  #pragma omp parallel for
  for (int y=1; y < height-1; y++)
    for (int x=1; x < width-1; x++)
      {
        SOBELX_2(x,y) = SOBELX_1(x,y-1) + 2.0f*SOBELX_1(x,y) + SOBELX_1(x,y+1);
        SOBELY_2(x,y) = SOBELY_1(x,y-1) - SOBELY_1(x,y+1);
      }

  // truncate to 3 sigma and normalize
  int radius = int(3*SIGMA + 1.0f);

  float gaussian[2*radius+1];

  float sum = 0.0f;

  for (int i=-radius; i<=radius; i++)
    sum += gaussian[i+radius] = exp(-(float(i)/SIGMA)*(float(i)/SIGMA)*0.5f);

  for (int i=-radius; i<=radius; i++)
    gaussian[i+radius] /= sum;

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      {
        BLURXX_1(x,y) = 0.0f;
        BLURXY_1(x,y) = 0.0f;
        BLURYY_1(x,y) = 0.0f;

        for (int i=-radius; i<=radius; i++)
          {
            int xx = clamp(x+i, 0, width-1);
            BLURXX_1(x,y) += gaussian[i+radius] * (SOBELX_2(xx,y) * SOBELX_2(xx,y));
            BLURXY_1(x,y) += gaussian[i+radius] * (SOBELX_2(xx,y) * SOBELY_2(xx,y));
            BLURYY_1(x,y) += gaussian[i+radius] * (SOBELY_2(xx,y) * SOBELY_2(xx,y));
          }
      }

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
      {
        BLURXX_2(x,y) = 0.0f;
        BLURXY_2(x,y) = 0.0f;
        BLURYY_2(x,y) = 0.0f;

        for (int i=-radius; i<=radius; i++)
          {
            int yy = clamp(y+i, 0, height-1);
            BLURXX_2(x,y) += gaussian[i+radius] * BLURXX_1(x,yy);
            BLURXY_2(x,y) += gaussian[i+radius] * BLURXY_1(x,yy);
            BLURYY_2(x,y) += gaussian[i+radius] * BLURYY_1(x,yy);
          }

        float det = BLURXX_2(x,y) * BLURYY_2(x,y) - BLURXY_2(x,y) * BLURXY_2(x,y);
        float trace = BLURXX_2(x,y) + BLURYY_2(x,y);

        const float k = 0.04f;

        float response = det - k * (trace * trace);
        CORNERNESS(x,y) = (response > threshold)? response : -FLT_MAX;
      }

  #pragma omp parallel for
  for (int y=1; y < height-1; y++)
    for (int x=1; x < width-1; x++)
      {
        float corner = CORNERNESS(x,y);

        bool maximal = true;
        for (int yy=-1; yy<=1; yy++)
          for (int xx=-1; xx<=1; xx++)
            if (CORNERNESS(x+xx,y+yy) > corner)
              maximal = false;

        if (maximal)
          {
            OUTPUT(x,y,0) = 1.0f;
            OUTPUT(x,y,1) = 0.0f;
            OUTPUT(x,y,2) = 0.0f;
            OUTPUT(x,y,3) = 0.0f;
          }
        else
          {
            OUTPUT(x,y,0) = INPUT(x,y,0);
            OUTPUT(x,y,1) = INPUT(x,y,1);
            OUTPUT(x,y,2) = INPUT(x,y,2);
            OUTPUT(x,y,3) = INPUT(x,y,3);
          }
      }

  free(gray);
  free(sobelx_1);
  free(sobely_1);
  free(sobelx_2);
  free(sobely_2);
  free(blurxx_1);
  free(bluryy_1);
  free(blurxy_1);
  free(blurxx_2);
  free(bluryy_2);
  free(blurxy_2);
  free(cornerness);
}

#include "kernel/harris.cl.h"

static cl_kernel *kernel;

void
harris_cl_prepare()
{
  const char *kernel_name[] = {"sobel",
                               "blurx",
                               "cornerness",
                               "cornerness_suppress",
                               NULL };

  kernel = cl_compile_and_build(harris_cl_source, kernel_name);
}

void
harris_cl (cl_mem input,
           cl_mem output,
           int    width,
           int    height,
           float  threshold)
{
  cl_int cl_err = 0;

  const int radius = int(3*SIGMA + 1.0f);

  float gaussian_mask[2 * radius + 1];

  for (int r=-radius; r<=radius; r++)
    gaussian_mask[r+radius] = expf(-(float(r)/SIGMA)*(float(r)/SIGMA)*0.5f);

  cl_mem m_gaussian_mask = clCreateBuffer (cl_state.context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (2 * radius + 1) * sizeof(cl_float),
                              gaussian_mask,
                              &cl_err);
  CL_CHECK;

  cl_image_format format = {CL_INTENSITY, CL_FLOAT};

  cl_mem sobelx   = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, NULL);
  cl_mem sobely   = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, NULL);
  cl_mem blurxx_h = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, NULL);
  cl_mem blurxy_h = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, NULL);
  cl_mem bluryy_h = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, NULL);
  cl_mem corner   = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, NULL);

  size_t global_ws[2];

  /* set local size */

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[0])
  CL_ARG(cl_mem,   input);
  CL_ARG(cl_mem,   sobelx);
  CL_ARG(cl_mem,   sobely);
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
  CL_ARG(cl_mem,   sobelx);
  CL_ARG(cl_mem,   sobely);
  CL_ARG(cl_mem,   blurxx_h);
  CL_ARG(cl_mem,   blurxy_h);
  CL_ARG(cl_mem,   bluryy_h);
  CL_ARG(cl_mem,   m_gaussian_mask);
  CL_ARG(cl_int,   radius);
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
  CL_ARG(cl_mem,   blurxx_h);
  CL_ARG(cl_mem,   blurxy_h);
  CL_ARG(cl_mem,   bluryy_h);
  CL_ARG(cl_mem,   corner);
  CL_ARG(cl_mem,   m_gaussian_mask);
  CL_ARG(cl_int,   radius);
  CL_ARG(cl_float, threshold);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[2], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[3])
  CL_ARG(cl_mem,   corner);
  CL_ARG(cl_mem,   output);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[3], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;

  CL_RELEASE(sobelx  );
  CL_RELEASE(sobely  );
  CL_RELEASE(blurxx_h);
  CL_RELEASE(blurxy_h);
  CL_RELEASE(bluryy_h);
  CL_RELEASE(corner  );
  CL_RELEASE(m_gaussian_mask);
}
