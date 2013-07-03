#include <stdio.h>
#include <cassert>
#include <time.h>
#include <iostream>
#include <limits>
#include <sys/time.h>

#include <cfloat>

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

#define NTRIES 20

static const char* bw_cl_source =
"kernel void bw_buffer(__global float4 *in,                                 \n"
"                        __global float4 *out,                              \n"
"                        int width)                                         \n"
"{                                                                          \n"
"  const int gid_x = get_global_id(0);                                      \n"
"  const int gid_y = get_global_id(1);                                      \n"
"  out[width*gid_y + gid_x] = in[width*gid_y + gid_x];                      \n"
"}                                                                          \n"
"                                                                           \n"
"const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |                    \n"
"                          CLK_ADDRESS_NONE |                               \n"
"                          CLK_FILTER_NEAREST;                              \n"
"                                                                           \n"
"kernel void bw_image2d(read_only  image2d_t in,                            \n"
"                       write_only image2d_t out)                           \n"
"{                                                                          \n"
"  const int gid_x = get_global_id(0);                                      \n"
"  const int gid_y = get_global_id(1);                                      \n"
"  const float4 pixel = read_imagef(in, sampler, (int2)(gid_x, gid_y));     \n"
"  write_imagef (out, (int2)(gid_x, gid_y), pixel);                         \n"
"}                                                                          \n";

static cl_kernel *kernel;

void
bandwidth_cl_prepare()
{
  const char *kernel_name[] = { "bw_buffer",
                                "bw_image2d",
                                NULL };

  kernel = cl_compile_and_build(bw_cl_source, kernel_name);
}

void
buffer_cl (cl_mem input,
                  cl_mem output,
                  int    width,
                  int    height)
{

  size_t global_ws[2];
  cl_int cl_err;

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

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;
  }
}

void
image2d_cl (cl_mem input,
            cl_mem output,
            int    width,
            int    height)
{

  size_t global_ws[2];
  cl_int cl_err;

  {
  global_ws[0] = width;
  global_ws[1] = height;

  CL_ARG_START(kernel[1])
  CL_ARG(cl_mem,   input);
  CL_ARG(cl_mem,   output);
  CL_ARG_END

  cl_err = clEnqueueNDRangeKernel(cl_state.command_queue,
                                  kernel[1], 2,
                                  NULL, global_ws, NULL,
                                  0, NULL, NULL);
  CL_CHECK;

  cl_err = clFinish(cl_state.command_queue);
  CL_CHECK;
  }
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

#define CL_WRITE_BUFFER                                           \
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
                      CL_TRUE,                                    \
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


#define CL_READ_BUFFER                                   \
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
                             CL_MEM_READ_ONLY,                    \
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
                             CL_MEM_WRITE_ONLY,                   \
                             &format,                             \
                             width,                               \
                             height,                              \
                             0,                                   \
                             NULL,                                \
                             NULL);

#define CL_READ_IMAGE2D                                           \
clEnqueueWriteImage (cl_state.command_queue,                      \
                     output_tex,                                  \
                     CL_FALSE,                                    \
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

int main(int argc, const char *argv[])
{
  unsigned int width = 2048, height = 2048;

  float * __restrict__ input;
  float * __restrict__ output;

  struct Stats bw_buffer, bw_image2d;

  cl_init();

  bandwidth_cl_prepare();

  posix_memalign((void **) &input , 32, width * height * 4 * sizeof(float));
  posix_memalign((void **) &output, 32, width * height * 4 * sizeof(float));

  /* run usual */

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(bw_buffer)
        {
          CL_WRITE_BUFFER

          buffer_cl (input_tex,
                     output_tex,
                     width,
                     height);

          CL_READ_BUFFER
        }
      TIME_END(bw_buffer, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      TIME_START(bw_image2d)
        {
          CL_WRITE_IMAGE2D

          image2d_cl (input_tex,
                      output_tex,
                      width,
                      height);

          CL_READ_IMAGE2D
        }
      TIME_END(bw_image2d, k)
    }

  printf("[Results]\n");

  printf("- Global memory:  %lf GB/s\n", (width * height) / (1024 * 1024 * bw_buffer.min)  );
  printf("- Texture Memory: %lf GB/s\n", (width * height) / (1024 * 1024 * bw_image2d.min) );

  return 0;
}
