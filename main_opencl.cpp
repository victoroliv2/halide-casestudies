#include <stdio.h>
#include <cassert>
#include <time.h>
#include <iostream>
#include <limits>
#include <sys/time.h>

#include <cfloat>

#include "lodepng.h"
#include "opencl_help.h"

double now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    static bool first_call = true;
    static time_t first_sec = 0;
    if (first_call) {
        first_call = false;
        first_sec = tv.tv_sec;
    }
    assert(tv.tv_sec >= first_sec);
    return (tv.tv_sec - first_sec) + (tv.tv_usec / 1000000.0);
}

#define NTRIES 10

void cielab_cl_prepare();
void bilateral_filter_cl_prepare();
void bilateral_filter_cl_simple_prepare();
void motion_blur_cl_prepare();
void unsharped_mask_cl_prepare();
void harris_cl_prepare();
void sift_cl_prepare();

void
cielab_cl (cl_mem input,
           cl_mem output,
           int    width,
           int    height);

void
motion_blur_cl (cl_mem input,
                cl_mem output,
                int    width,
                int    height,
                float  length,
                float  angle);

void
unsharped_mask_cl (cl_mem A,
                   cl_mem B,
                   int    width,
                   int    height,
                   float  sigma,
                   float  detail_thresh,
                   float  sharpen);

void
bilateral_filter_cl (cl_mem input,
                     cl_mem output,
                     int    width,
                     int    height,
                     int    s_sigma,
                     float  r_sigma);

void
bilateral_filter_cl_simple (cl_mem input,
                            cl_mem output,
                            int    width,
                            int    height,
                            int    s_sigma,
                            float  r_sigma);

void
harris_cl (cl_mem input,
           cl_mem output,
           int    width,
           int    height,
           float  threshold);

void
unsharped_mask (cl_mem A,
                cl_mem B,
                int    width,
                int    height,
                float  sigma,
                float  detail_thresh,
                float  sharpen);

void
sift_cl (cl_mem input,
         cl_mem output,
         int    width,
         int    height,
         int    octaves,
         int    intervals,
         float  curv_thr,
         float  contr_thr);

void
cielab (float const * __restrict__ _input,
        float       * __restrict__ _output,
        int    width,
        int    height);

void
motion_blur (float const * __restrict__ _input,
             float       * __restrict__ _output,
             int    width,
             int    height,
             float  length,
             float  angle);

void
unsharped_mask (float const * __restrict__ _input,
                float       * __restrict__ _output,
                int    width,
                int    height,
                float  sigma,
                float  detail_thresh,
                float  sharpen);

void
bilateral_filter (const float * __restrict__ _input,
                  float       * __restrict__ _output,
                  int   width,
                  int   height,
                  int   s_sigma,
                  float r_sigma);

void
harris (float const * __restrict__ _input,
        float       * __restrict__ _output,
        int    width,
        int    height,
        float  threshold);

void
sift (float const * __restrict__ _input,
      unsigned char * __restrict__ _output,
      int    width,
      int    height,
      int    octaves,
      int    intervals,
      float  curv_thr,
      float  contr_thr);

void save(float * output, int width, int height, const char * out_file)
{
  unsigned int error;

  std::vector<unsigned char> output_png;
  output_png.resize(width * height * 4);

  for(int k=0; k<width * height * 4; k++)
    output_png[k] = 255.0f * output[k];

  error = lodepng::encode(out_file, output_png, width, height);
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

struct Stats
{
  float min;
  float max;
  float elapsed[NTRIES];

  Stats(){
    min =  FLT_MAX;
    max = -FLT_MAX;
    for (int k=0; k<NTRIES; k++) elapsed[k] = FLT_MAX;
  }
};

#define TIME_START(st)                                \
{                                                     \
  double start = now();                               \
  {

#define TIME_END(st, i)                               \
  }                                                   \
  double end   = now();                               \
                                                      \
  st.elapsed[i] = end - start;                        \
  if (st.elapsed[i] < st.min) st.min = st.elapsed[i]; \
  if (st.elapsed[i] > st.max) st.max = st.elapsed[i]; \
}

#define CL_WRITE                                                  \
cl_mem input_tex, output_tex;                                     \
                                                                  \
input_tex = clCreateBuffer (cl_state.context,                     \
                            CL_MEM_READ_ONLY,                     \
                            width * height * sizeof(cl_float4),   \
                            NULL,                                 \
                            NULL);                                \
                                                                  \
clEnqueueWriteBuffer (cl_state.command_queue,                     \
                      input_tex,                                  \
                      CL_FALSE,                                   \
                      0,                                          \
                      width * height * sizeof(cl_float4),         \
                      input,                                      \
                      0, NULL, NULL);                             \
                                                                  \
output_tex = clCreateBuffer (cl_state.context,                    \
                             CL_MEM_WRITE_ONLY,                   \
                             width * height * sizeof(cl_float4),  \
                             NULL,                                \
                             NULL);


#define CL_READ                                          \
clEnqueueReadBuffer (cl_state.command_queue,             \
                     output_tex,                         \
                     CL_TRUE,                            \
                     0,                                  \
                     width * height * sizeof(cl_float4), \
                     output,                             \
                     0, NULL, NULL);                     \
                                                         \
clReleaseMemObject(input_tex);                           \
clReleaseMemObject(output_tex);

#define CL_WRITE_IMAGE2D                                          \
cl_mem input_tex, output_tex;                                     \
                                                                  \
cl_image_format format = {CL_RGBA, CL_FLOAT};                     \
                                                                  \
input_tex = clCreateImage2D (cl_state.context,                    \
                             CL_MEM_READ_WRITE,                   \
                             &format,                             \
                             width,                               \
                             height,                              \
                             0,                                   \
                             NULL,                                \
                             NULL);                               \
                                                                  \
size_t origin[3] = {0,0,0};                                       \
size_t region[3] = {width,height,1};                              \
                                                                  \
clEnqueueWriteImage (cl_state.command_queue,                      \
                     input_tex,                                   \
                     CL_FALSE,                                    \
                     origin,                                      \
                     region,                                      \
                     0,                                           \
                     0,                                           \
                     input,                                       \
                     0,                                           \
                     NULL,                                        \
                     NULL);                                       \
                                                                  \
output_tex = clCreateImage2D (cl_state.context,                   \
                             CL_MEM_READ_WRITE,                   \
                             &format,                             \
                             width,                               \
                             height,                              \
                             0,                                   \
                             NULL,                                \
                             NULL);

#define CL_READ_IMAGE2D                                           \
clEnqueueReadImage (cl_state.command_queue,                       \
                     output_tex,                                  \
                     CL_TRUE,                                     \
                     origin,                                      \
                     region,                                      \
                     0,                                           \
                     0,                                           \
                     output,                                      \
                     0,                                           \
                     NULL,                                        \
                     NULL);                                       \
                                                                  \
clReleaseMemObject(input_tex);                                    \
clReleaseMemObject(output_tex);


#define CL_WRITE_SIFT                                             \
cl_mem input_tex, output_tex;                                     \
                                                                  \
cl_image_format format = {CL_RGBA, CL_FLOAT};                     \
                                                                  \
input_tex = clCreateImage2D (cl_state.context,                    \
                             CL_MEM_READ_WRITE,                   \
                             &format,                             \
                             width,                               \
                             height,                              \
                             0,                                   \
                             NULL,                                \
                             NULL);                               \
                                                                  \
size_t origin[3] = {0,0,0};                                       \
size_t region[3] = {width,height,1};                              \
                                                                  \
clEnqueueWriteImage (cl_state.command_queue,                      \
                     input_tex,                                   \
                     CL_FALSE,                                    \
                     origin,                                      \
                     region,                                      \
                     0,                                           \
                     0,                                           \
                     input,                                       \
                     0,                                           \
                     NULL,                                        \
                     NULL);                                       \
                                                                  \
output_tex = clCreateBuffer (cl_state.context,                    \
                            CL_MEM_READ_WRITE,                    \
                            width * height * sizeof(unsigned char), \
                            NULL,                                 \
                            NULL);

#define CL_READ_SIFT                                     \
clEnqueueReadBuffer (cl_state.command_queue,             \
                     output_tex,                         \
                     CL_TRUE,                            \
                     0,                                  \
                     width * height * sizeof(unsigned char), \
                     output_sift,                        \
                     0, NULL, NULL);                     \
                                                         \
clReleaseMemObject(input_tex);                           \
clReleaseMemObject(output_tex);

int main(int argc, const char *argv[])
{
  assert(argc == 3);

  std::vector<unsigned char> input_png;
  unsigned int width, height, error;

  float * __restrict__ input;
  float * __restrict__ output;
  unsigned char * __restrict__ output_sift;

  struct Stats lab_time, mb_time, unsharped_time, bf_time, harris_time, sift_time;
  struct Stats lab_time_cl, mb_time_cl, unsharped_time_cl, bf_time_cl, bf_simple_time_cl, harris_time_cl, sift_time_cl;

  error = lodepng::decode(input_png, width, height, argv[1]);
  if(error) std::cout << "decoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

  cl_init();

  cielab_cl_prepare();
  motion_blur_cl_prepare();
  unsharped_mask_cl_prepare();
  bilateral_filter_cl_prepare();
  bilateral_filter_cl_simple_prepare();
  harris_cl_prepare();
  sift_cl_prepare();

  posix_memalign((void **) &input , 32, width * height * 4 * sizeof(float));
  posix_memalign((void **) &output, 32, width * height * 4 * sizeof(float));
  posix_memalign((void **) &output_sift, 32, width * height * sizeof(unsigned char));

  for(int k=0; k<width * height * 4; k++)
     input[k] = input_png[k] / 255.0f;

  /* run usual */

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(lab_time)
        {
          cielab(input,
                 output,
                 width,
                 height);
        }
      TIME_END(lab_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(mb_time)
        {
          motion_blur(input,
                      output,
                      width,
                      height,
                      10.0f,
                      45.0f);
        }
      TIME_END(mb_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(unsharped_time)

        {
          unsharped_mask (input,
                          output,
                          width,
                          height,
                          1.5f,
                          0.5f,
                          0.5f);
        }
      TIME_END(unsharped_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(bf_time)
        {
          bilateral_filter(input,
                           output,
                           width,
                           height,
                           8,
                           0.05f);
        }
      TIME_END(bf_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(harris_time)
        {
          harris (input,
                  output,
                  width,
                  height,
                  0.05f);
        }
      TIME_END(harris_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(sift_time)
        {
          sift (input,
                output_sift,
                width,
                height,
                4,
                2,
                10.0f,
                0.04f);
        }
      TIME_END(sift_time, k)
    }

  /* run opencl */

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(lab_time_cl)
        {
          CL_WRITE

          cielab_cl (input_tex,
                     output_tex,
                     width,
                     height);

          CL_READ
        }
      TIME_END(lab_time_cl, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(mb_time_cl)
        {
          CL_WRITE

          motion_blur_cl (input_tex,
                          output_tex,
                          width,
                          height,
                          10.0f,
                          45.0f);

          CL_READ
        }
      TIME_END(mb_time_cl, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(unsharped_time_cl)

        {
          CL_WRITE_IMAGE2D

          unsharped_mask_cl (input_tex,
                             output_tex,
                             width,
                             height,
                             1.5f,
                             0.5f,
                             0.5f);

          CL_READ_IMAGE2D
        }
      TIME_END(unsharped_time_cl, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(bf_time_cl)
        {
          CL_WRITE

          bilateral_filter_cl (input_tex,
                               output_tex,
                               width,
                               height,
                               8,
                               0.05f);

          CL_READ
        }
      TIME_END(bf_time_cl, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(bf_simple_time_cl)
        {
          CL_WRITE

          bilateral_filter_cl_simple (input_tex,
                                      output_tex,
                                      width,
                                      height,
                                      8,
                                      0.05f);

          CL_READ
        }
      TIME_END(bf_simple_time_cl, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(harris_time_cl)
        {
          CL_WRITE_IMAGE2D

          harris_cl (input_tex,
                     output_tex,
                     width,
                     height,
                     0.05f);

          CL_READ_IMAGE2D
        }
      TIME_END(harris_time_cl, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(sift_time_cl)
        {
          CL_WRITE_SIFT

          sift_cl (input_tex,
                   output_tex,
                   width,
                   height,
                   4,
                   2,
                   10.0f,
                   0.04f);

          CL_READ_SIFT

          if (k==0)
            {
              unsigned char * output_png = new unsigned char [4*width*height];

              for(int k=0; k<width * height; k++) {
                assert(output_sift[k] == 0);
                output_png[4*k+0] = 255 * output_sift[k];
                output_png[4*k+1] = 255 * output_sift[k];
                output_png[4*k+2] = 255 * output_sift[k];
                output_png[4*k+3] = 255;
              }

              error = lodepng::encode(argv[2], output_png, width, height);
              if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

              free(output_png);
            }
        }
      TIME_END(sift_time_cl, k)
    }

  /* save output */


  printf("[CPU]\n");

  printf("- CIELAB:         %lf \n", lab_time.min);
  printf("- MOTION-BLUR:    %lf \n", mb_time.min);
  printf("- UNSHARPED-MASK: %lf \n", unsharped_time.min );
  printf("- BIL. FILTER:    %lf \n", bf_time.min);
  printf("- HARRIS:         %lf \n", harris_time.min);
  printf("- SIFT:           %lf \n", sift_time.min);

  printf("[OpenCL]\n");

  printf("- CIELAB:             %lf \n", lab_time_cl.min );
  printf("- MOTION-BLUR:        %lf \n", mb_time_cl.min );
  printf("- UNSHARPED-MASK:     %lf \n", unsharped_time_cl.min );
  printf("- BIL. FILTER:        %lf \n", bf_time_cl.min );
  printf("- BIL. FILTER SIMPLE: %lf \n", bf_simple_time_cl.min);
  printf("- HARRIS:             %lf \n", harris_time_cl.min);
  printf("- SIFT:               %lf \n", sift_time_cl.min);

  return 0;
}
