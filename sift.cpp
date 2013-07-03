#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdalign.h>
#include <omp.h>
#include <string.h>
#include <float.h>

#include "opencl_help.h"

inline static int clamp(int x, int low, int high)
{
  return (x < low) ? low : ((x > high) ? high : x);
}

inline static int max(int a, int b)
{
  return (a < b) ? b : a;
}

inline static int min(int a, int b)
{
  return (a < b) ? a : b;
}

#include <stdio.h>

#define SIGMA 1.6f

void
sift (float const * __restrict__ _input,
      unsigned char * __restrict__ _output,
      int    width,
      int    height,
      int    octaves,
      int    intervals,
      float  curv_thr,
      float  contr_thr)
{
  const int channels = 4;

  float * __restrict__ input = (float * __restrict__) __builtin_assume_aligned(_input,  32);
  unsigned char * __restrict__ output = (unsigned char * __restrict__) __builtin_assume_aligned(_output, 32);

  #define INPUT(x,y,c) input[c+channels*(x + width * y)]
  #define OUTPUT(x,y)  output[x + width * y]

  memset(output, 0, width*height*sizeof(unsigned char));

  float sig[intervals+3], sig_prev, sig_total;

  sig[0] = SIGMA;
  float p = pow( 2.0, 1.0 / intervals );
  for(int i = 1; i < intervals + 3; i++ )
    {
      sig_prev = powf( p, i - 1 ) * SIGMA;
      sig_total = sig_prev * p;
      sig[i] = sqrtf( sig_total * sig_total - sig_prev * sig_prev );
    }

  float * __restrict__ gauss_pyr[octaves][intervals+3];

  #define  GAUSS_PYR(o,i,x,y) gauss_pyr[o][i][y*W+x]

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 3; i++ )
      {
        const int W = width  >> o;
        const int H = height >> o;

        posix_memalign((void **) &gauss_pyr[o][i], 32, W * H * sizeof(float));

        if( o == 0  &&  i == 0 ) {

          #pragma omp parallel for
          for(int y=0; y<H; y++)
            for(int x=0; x<W; x++)
              GAUSS_PYR(o,i,x,y) = 0.299f * INPUT(x, y, 0) + 0.587f * INPUT(x, y, 1) + 0.114f * INPUT(x, y, 2);
        }
        /* base of new octvave is halved image from end of previous octave */
        else if( i == 0 ) {
          float * __restrict__ down;

          posix_memalign((void **) &down, 32, 2 * W * H * sizeof(float));

          #define DOWN(x,y) down[x + W * y]

          #pragma omp parallel for
          for(int y=0; y<2*H; y++)
            for(int x=0; x<W; x++)
              DOWN(x, y) = (GAUSS_PYR(o-1,i, max(2*x-1, 0), y) +
                            3.0f * (GAUSS_PYR(o-1,i, 2*x, y) + GAUSS_PYR(o-1,i, min(2*x+1, 2*W-1), y)) +
                            GAUSS_PYR(o-1,i, min(2*x+2, 2*W-1), y)) / 8.0f;

          #pragma omp parallel for
          for(int y=0; y<H; y++)
            for(int x=0; x<W; x++)
              GAUSS_PYR(o,i,x,y) = (DOWN(x, max(2*y-1, 0)) +
                                    3.0f * (DOWN(x, 2*y) + DOWN(x, min(2*y+1, H-1))) +
                                    DOWN(x, min(2*y+2, H-1))) / 8.0f;

          free(down);
        }
        /* blur the current octave's last image to create the next one */
        else {
          float * __restrict__ blur;

          posix_memalign((void **) &blur, 32, W * H * sizeof(float));

          #define BLUR(x,y) blur[x + W * y]

          float sum = 0.0f;

          const int radius = int(3*sig[i] + 1.0f);

          float gaussian_mask[2 * radius + 1];

          for (int i=-radius; i<=radius; i++)
            sum += gaussian_mask[i+radius] = exp(-(float(i)/sig[i])*(float(i)/sig[i])*0.5f);

          for (int i=-radius; i<=radius; i++)
            gaussian_mask[i+radius] /= sum;

          #pragma omp parallel for
          for(int y=0; y<H; y++)
            for(int x=0; x<W; x++) {
              float v = 0.0f;

              for(int r=-radius; r <= radius; r++)
                v += gaussian_mask[r+radius] * GAUSS_PYR(o,i-1,clamp(x+r, 0, W-1),clamp(y+r, 0, H-1));

              BLUR(x,y) = v;
            }

          #pragma omp parallel for
          for(int y=0; y<H; y++)
            for(int x=0; x<W; x++) {
              float v = 0.0f;

              for(int r=-radius; r <= radius; r++)
                v += gaussian_mask[r+radius] * BLUR(clamp(x+r, 0, W-1),clamp(y+r, 0, H-1));

              GAUSS_PYR(o,i,x,y) = v;
            }

          free(blur);
        }
      }

  float * __restrict__ dog_pyr[octaves][intervals+2];

  #define  DOG_PYR(o,i,x,y) dog_pyr[o][i][y*W+x]

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 2; i++ ) {
      const int W = width  >> o;
      const int H = height >> o;

      posix_memalign((void **) &dog_pyr[o][i], 32, W * H * sizeof(float));

      #pragma omp parallel for
      for(int y=0; y<H; y++)
        for(int x=0; x<W; x++)
          DOG_PYR(o,i,x,y) = GAUSS_PYR(o,i+1,x,y) - GAUSS_PYR(o,i,x,y);
    }

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      {
        const int W = width  >> o;
        const int H = height >> o;

        #pragma omp parallel for
        for(int y=0; y<H; y++)
          for(int x=0; x<W; x++)
            {
              float v[3][3][3];

              for (int ry=-1; ry<=1; ry++)
                for (int rx=-1; rx<=1; rx++)
                  {
                    v[0][ry+1][rx+1] = DOG_PYR(o, i-1, clamp(x+rx, 0, W-1), clamp(y+ry, 0, H-1));
                    v[1][ry+1][rx+1] = DOG_PYR(o, i,   clamp(x+rx, 0, W-1), clamp(y+ry, 0, H-1));
                    v[2][ry+1][rx+1] = DOG_PYR(o, i+1, clamp(x+rx, 0, W-1), clamp(y+ry, 0, H-1));
                  }

              const float vcc = v[1][1][1];

              float dmax = -FLT_MAX;
              float dmin =  FLT_MAX;

              for (int ry=0; ry<=2; ry++)
                for (int rx=0; rx<=2; rx++)
                  {
                    dmax = fmax(dmax,
                           fmax(v[0][ry][rx],
                           fmax(v[1][ry][rx],
                                v[2][ry][rx])));

                    dmin = fmin(dmin,
                           fmin(v[0][ry][rx],
                           fmin(v[1][ry][rx],
                                v[2][ry][rx])));
                  }

              const float prelim_contr_thr = 0.5f * contr_thr / intervals;

              const bool is_extremum = ((fabs(vcc) > prelim_contr_thr) &&
                                        ((vcc <= 0.0f && vcc == dmin) ||
                                         (vcc >  0.0f && vcc == dmax)));

              const float dxx = v[1][1][2] + v[1][1][0] - 2.0f * vcc;
              const float dyy = v[1][2][1] + v[1][0][1] - 2.0f * vcc;
              const float dss = v[2][1][1] + v[0][1][1] - 2.0f * vcc;
              const float dxy = ( v[1][2][2] - v[1][2][0] - v[1][0][2] + v[1][0][0] ) / 4.0f;
              const float dxs = ( v[2][1][2] - v[2][1][0] - v[0][1][2] + v[0][1][0] ) / 4.0f;
              const float dys = ( v[2][2][1] - v[2][0][1] - v[0][2][1] + v[0][0][1] ) / 4.0f;

              const float pc_det = dxx * dyy - 2.0f * dxy;
              const float pc_tr = dxx + dyy;

              float invdet = 1.0f/(  ( dxx * (dyy * dss - dys * dys) )
                                   - ( dxy * (dxy * dss - dys * dxs) )
                                   + ( dxs * (dxy * dys - dyy * dxs) ));

              const float inv_dxx = invdet * (dyy * dss - dys * dys);
              const float inv_dyy = invdet * (dxx * dss - dxs * dxs);
              const float inv_dss = invdet * (dxx * dyy - dxy * dxy);
              const float inv_dxy = invdet * (dxs * dys - dxy * dss);
              const float inv_dxs = invdet * (dxy * dys - dxs * dyy);
              const float inv_dys = invdet * (dxy * dxs - dxx * dys);

              const float dx = (v[1][1][2] - v[1][1][0]) / 2.0f;
              const float dy = (v[1][2][1] - v[1][0][1]) / 2.0f;
              const float ds = (v[2][1][1] - v[0][1][1]) / 2.0f;

              const float interp_x = inv_dxx * dx + inv_dxy * dy + inv_dxs * ds;
              const float interp_y = inv_dxy * dx + inv_dyy * dy + inv_dys * ds;
              const float interp_s = inv_dxs * dx + inv_dys * dy + inv_dss * ds;

              const float interp_contr = interp_x * dx + interp_y * dy + interp_s * ds;

              bool ok = is_extremum &&
                        pc_det > 0.0f;
                        (pc_tr * pc_tr / pc_det < ( curv_thr + 1.0f )*( curv_thr + 1.0f ) / curv_thr) &&
                        fabs(interp_contr) > contr_thr / intervals &&
                        dx < 1.0f &&
                        dy < 1.0f &&
                        ds < 1.0f;

              if (ok) OUTPUT(x << o, y << o) = 1;
            }
      }

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 3; i++ )
      free(gauss_pyr[o][i]);

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 2; i++ )
      free(dog_pyr[o][i]);
}

#include "kernel/sift.cl.h"

static cl_kernel *kernel;

void
sift_cl_prepare()
{
  const char *kernel_name[] = { "downx",
                                "downy",
                                "blurx",
                                "blury",
                                "dog",
                                "isvalid",
                                "gray",
                                NULL };

  kernel = cl_compile_and_build(sift_cl_source, kernel_name);
}

void
sift_cl (cl_mem input,
         cl_mem output,
         int    width,
         int    height,
         int    octaves,
         int    intervals,
         float  curv_thr,
         float  contr_thr)
{
  cl_int cl_err = 0;

  float sig[intervals+3], sig_prev, sig_total;

  cl_image_format format = {CL_INTENSITY, CL_FLOAT};

  sig[0] = SIGMA;
  float p = pow( 2.0, 1.0 / intervals );
  for(int i = 1; i < intervals + 3; i++ )
    {
      sig_prev = powf( p, i - 1 ) * SIGMA;
      sig_total = sig_prev * p;
      sig[i] = sqrtf( sig_total * sig_total - sig_prev * sig_prev );
    }

  void * ptr =  clEnqueueMapBuffer (cl_state.command_queue, output, CL_TRUE, CL_MAP_WRITE,
                                    0, width * height * sizeof(unsigned char),
                                    0, NULL, NULL, &cl_err);
  CL_CHECK;

  memset(ptr, 0, width * height * sizeof(unsigned char));

  cl_err = clEnqueueUnmapMemObject (cl_state.command_queue, output, ptr,
                                    0, NULL, NULL);
  CL_CHECK;

  cl_mem gauss_pyr[octaves][intervals+3];
  cl_mem dog_pyr[octaves][intervals+2];

  size_t global_ws[2];

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 3; i++ )
      {
        int W = width  / (1 << o);
        int H = height / (1 << o);

        gauss_pyr[o][i] = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, W, H, 0, NULL, &cl_err);
        CL_CHECK;

        if( o == 0  &&  i == 0 ) {
            {
              global_ws[0] = W;
              global_ws[1] = H;

              CL_ARG_START(kernel[6])
              CL_ARG(cl_mem,   input);
              CL_ARG(cl_mem,   gauss_pyr[o][i]);
              CL_ARG_END

              cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                              kernel[6], 2,
                                              NULL, global_ws, NULL,
                                              0, NULL, NULL);
              CL_CHECK;
            }
        }
        /* base of new octvave is halved image from end of previous octave */
        else if( i == 0 ) {
          cl_mem down = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, W, H * 2, 0, NULL, &cl_err);
          CL_CHECK;

            {
              global_ws[0] = W;
              global_ws[1] = H * 2;

              CL_ARG_START(kernel[0])
              CL_ARG(cl_mem,   gauss_pyr[o-1][i]);
              CL_ARG(cl_mem,   down);
              CL_ARG_END

              cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                              kernel[0], 2,
                                              NULL, global_ws, NULL,
                                              0, NULL, NULL);
              CL_CHECK;
            }

            {
              global_ws[0] = W;
              global_ws[1] = H;

              CL_ARG_START(kernel[1])
              CL_ARG(cl_mem,   down);
              CL_ARG(cl_mem,   gauss_pyr[o][i]);
              CL_ARG_END

              cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                              kernel[1], 2,
                                              NULL, global_ws, NULL,
                                              0, NULL, NULL);
              CL_CHECK;
            }

          cl_err = clFinish(cl_state.command_queue);
          CL_CHECK;

          CL_RELEASE(down);
        }
        /* blur the current octave's last image to create the next one */
        else {
          cl_mem blur = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, W, H, 0, NULL, &cl_err);
          CL_CHECK;

          const int radius = int(3*sig[i] + 1.0f);

          float gaussian_mask[2 * radius + 1];

          float sum = 0.0f;

          for (int i=-radius; i<=radius; i++)
            sum += gaussian_mask[i+radius] = exp(-(float(i)/sig[i])*(float(i)/sig[i])*0.5f);

          for (int i=-radius; i<=radius; i++)
            gaussian_mask[i+radius] /= sum;

          cl_mem m_gaussian_mask = clCreateBuffer (cl_state.context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      (2 * radius + 1) * sizeof(cl_float),
                                      gaussian_mask,
                                      &cl_err);
          CL_CHECK;

            {
              global_ws[0] = W;
              global_ws[1] = H;

              CL_ARG_START(kernel[2])
              CL_ARG(cl_mem,   gauss_pyr[o][i-1]);
              CL_ARG(cl_mem,   blur);
              CL_ARG(cl_mem,   m_gaussian_mask);
              CL_ARG(cl_int,   radius);
              CL_ARG_END

              cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                              kernel[2], 2,
                                              NULL, global_ws, NULL,
                                              0, NULL, NULL);
              CL_CHECK;
            }

            {
              global_ws[0] = W;
              global_ws[1] = H;

              CL_ARG_START(kernel[3])
              CL_ARG(cl_mem,   blur);
              CL_ARG(cl_mem,   gauss_pyr[o][i]);
              CL_ARG(cl_mem,   m_gaussian_mask);
              CL_ARG(cl_int,   radius);
              CL_ARG_END

              cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                              kernel[3], 2,
                                              NULL, global_ws, NULL,
                                              0, NULL, NULL);
              CL_CHECK;
            }

          cl_err = clFinish(cl_state.command_queue);
          CL_CHECK;

          CL_RELEASE(blur);
          CL_RELEASE(m_gaussian_mask);
        }
      }

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 2; i++ ) {
        int W = width  / (1 << o);
        int H = height / (1 << o);

        dog_pyr[o][i] = clCreateImage2D (cl_state.context, CL_MEM_READ_WRITE, &format, W, H, 0, NULL, &cl_err);
        CL_CHECK;

        {
        global_ws[0] = W;
        global_ws[1] = H;

        CL_ARG_START(kernel[4])
        CL_ARG(cl_mem,   gauss_pyr[o][i+1]);
        CL_ARG(cl_mem,   gauss_pyr[o][i]);
        CL_ARG(cl_mem,   dog_pyr[o][i]);
        CL_ARG_END

        cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                        kernel[4], 2,
                                        NULL, global_ws, NULL,
                                        0, NULL, NULL);
        CL_CHECK;
        }
    }

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      {
        int W = width  / (1 << o);
        int H = height / (1 << o);

        CL_CHECK;

        {
        global_ws[0] = W;
        global_ws[1] = H;

        CL_ARG_START(kernel[5])
        CL_ARG(cl_mem,   dog_pyr[o][i-1]);
        CL_ARG(cl_mem,   dog_pyr[o][i]);
        CL_ARG(cl_mem,   dog_pyr[o][i+1]);
        CL_ARG(cl_mem,   output);
        CL_ARG(cl_float, curv_thr);
        CL_ARG(cl_float, contr_thr);
        CL_ARG(cl_int,   o);
        CL_ARG(cl_int,   intervals);
        CL_ARG_END

        cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                        kernel[5], 2,
                                        NULL, global_ws, NULL,
                                        0, NULL, NULL);
        CL_CHECK;
        }
      }

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 3; i++ )
      CL_RELEASE(gauss_pyr[o][i]);

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 2; i++ )
      CL_RELEASE(dog_pyr[o][i]);
}
