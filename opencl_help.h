#include "CL/cl.h"
#include "CL/cl_gl.h"
#include "CL/cl_gl_ext.h"
#include "CL/cl_ext.h"

#define CL_ERROR {fprintf(stderr, "Error in %s:%d@%s - %s\n", __FILE__, __LINE__, __func__, cl_errstring(cl_err)); assert(0);}

#define CL_CHECK {if (cl_err != CL_SUCCESS) CL_ERROR;}

#define CL_ARG_START(KERNEL) \
  { cl_kernel __mykernel=KERNEL; int __p = 0;

#define CL_ARG(TYPE, NAME) \
  { cl_err = clSetKernelArg(__mykernel, __p++, sizeof(TYPE), (void*)& NAME); \
    CL_CHECK; }

#define CL_ARG_END \
  __p = -1; }

#define CL_RELEASE(obj)               \
  { cl_err = clReleaseMemObject(obj); \
    CL_CHECK; }

#define CL_SAFE_CALL(func)                                          \
func;                                                               \
if (errcode != CL_SUCCESS)                                          \
{                                                                   \
  fprintf(stderr, "OpenCL error in %s, Line %u in file %s\nError:%s", \
            #func, __LINE__, __FILE__, cl_errstring(errcode)); \
}

typedef struct
{
  cl_context       context;
  cl_platform_id   platform;
  cl_device_id     device;
  cl_command_queue command_queue;
  cl_ulong         max_mem_alloc;
  cl_ulong         local_mem_size;

  char platform_name   [1024];
  char platform_version[1024];
  char platform_ext    [1024];
  char device_name     [1024];
}
ClState;

const char *cl_errstring(cl_int err);

bool cl_init();

#define CL_BUILD(SOURCE, ...)                                \
      (const char *kernel_name[] = { __VA_ARGS__ , NULL };,  \
       return cl_compile_and_build(SOURCE, kernel_name);)

cl_kernel *
cl_compile_and_build (const char *program_source, const char *kernel_name[]);

#ifdef __CL_INIT_MAIN__

ClState cl_state = {NULL, NULL, NULL, NULL, 0, 0, "", "", "", ""};

#else

extern ClState cl_state;

#endif
