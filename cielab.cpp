#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdalign.h>
#include <omp.h>

#include "opencl_help.h"

#include <stdio.h>

#include "kernel/rgba_to_cielab.cl.h"

static cl_kernel *kernel;

void
cielab_cl_prepare()
{
  const char *kernel_name[] = { "rgba_to_cielab",
                                NULL };

  kernel = cl_compile_and_build(rgba_to_cielab_cl_source, kernel_name);
}

void
cielab (float const * __restrict__ _input,
        float       * __restrict__ _output,
             int    width,
             int    height)
{
  const int channels = 4;

  float * __restrict__ input     = (float * __restrict__) __builtin_assume_aligned(_input,  32);
  float * __restrict__ output    = (float * __restrict__) __builtin_assume_aligned(_output, 32);

  #define INPUT(x,y,c) input[c+channels*(x + width * y)]

  #pragma omp parallel for
  for (int y=0; y < height; y++)
    for (int x=0; x < width; x++)
        {
          float R, G, B, A, X, Y, Z;

          R = input[y*width+x+0];
          G = input[y*width+x+1];
          B = input[y*width+x+2];
          A = input[y*width+x+3];

          if ( R > 0.04045f ) R = powf(( ( R + 0.055f ) / 1.055f ), 2.4f);
          else                R = R / 12.92f;
          if ( G > 0.04045f ) G = powf(( ( G + 0.055f ) / 1.055f ), 2.4f);
          else                G = G / 12.92f;
          if ( B > 0.04045f ) B = powf(( ( B + 0.055f ) / 1.055f ), 2.4f);
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

          if ( X > 0.008856f ) X = powf(X, 1.0f/3.0f);
          else                 X = ( 7.787f * X ) + ( 16.0f / 116.0f );
          if ( Y > 0.008856f ) Y = powf(Y, 1.0f/3.0f);
          else                 Y = ( 7.787f * Y ) + ( 16.0f / 116.0f );
          if ( Z > 0.008856f ) Z = powf(Z, 1.0f/3.0f);
          else                 Z = ( 7.787f * Z ) + ( 16.0f / 116.0f );

          float CIEL = ( 116.0f * Y ) - 16.0f;
          float CIEa = 500.0f * ( X - Y );
          float CIEb = 200.0f * ( Y - Z );

          output[y*width+x+0] = CIEL;
          output[y*width+x+1] = CIEa;
          output[y*width+x+2] = CIEb;
          output[y*width+x+3] = A;
        }
}

void
cielab_cl (cl_mem input,
           cl_mem output,
           int    width,
           int    height)
{
  cl_int cl_err = 0;

  size_t global_ws[2];

  /* set local size */

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[0])
  CL_ARG(cl_mem,   input);
  CL_ARG(cl_mem,   output);
  CL_ARG(cl_int,   width);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[0], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;
  }
}
