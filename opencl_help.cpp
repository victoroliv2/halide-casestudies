#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define __CL_INIT_MAIN__
#include "opencl_help.h"
#undef __CL_INIT_MAIN__

const char *cl_errstring(cl_int err) {
  static const char* strings[] =
  {
    /* Error Codes */
      "success"                         /*  0  */
    , "device not found"                /* -1  */
    , "device not available"            /* -2  */
    , "compiler not available"          /* -3  */
    , "mem object allocation failure"   /* -4  */
    , "out of resources"                /* -5  */
    , "out of host memory"              /* -6  */
    , "profiling info not available"    /* -7  */
    , "mem copy overlap"                /* -8  */
    , "image format mismatch"           /* -9  */
    , "image format not supported"      /* -10 */
    , "build program failure"           /* -11 */
    , "map failure"                     /* -12 */
    , ""                                /* -13 */
    , ""                                /* -14 */
    , ""                                /* -15 */
    , ""                                /* -16 */
    , ""                                /* -17 */
    , ""                                /* -18 */
    , ""                                /* -19 */
    , ""                                /* -20 */
    , ""                                /* -21 */
    , ""                                /* -22 */
    , ""                                /* -23 */
    , ""                                /* -24 */
    , ""                                /* -25 */
    , ""                                /* -26 */
    , ""                                /* -27 */
    , ""                                /* -28 */
    , ""                                /* -29 */
    , "invalid value"                   /* -30 */
    , "invalid device type"             /* -31 */
    , "invalid platform"                /* -32 */
    , "invalid device"                  /* -33 */
    , "invalid context"                 /* -34 */
    , "invalid queue properties"        /* -35 */
    , "invalid command queue"           /* -36 */
    , "invalid host ptr"                /* -37 */
    , "invalid mem object"              /* -38 */
    , "invalid image format descriptor" /* -39 */
    , "invalid image size"              /* -40 */
    , "invalid sampler"                 /* -41 */
    , "invalid binary"                  /* -42 */
    , "invalid build options"           /* -43 */
    , "invalid program"                 /* -44 */
    , "invalid program executable"      /* -45 */
    , "invalid kernel name"             /* -46 */
    , "invalid kernel definition"       /* -47 */
    , "invalid kernel"                  /* -48 */
    , "invalid arg index"               /* -49 */
    , "invalid arg value"               /* -50 */
    , "invalid arg size"                /* -51 */
    , "invalid kernel args"             /* -52 */
    , "invalid work dimension"          /* -53 */
    , "invalid work group size"         /* -54 */
    , "invalid work item size"          /* -55 */
    , "invalid global offset"           /* -56 */
    , "invalid event wait list"         /* -57 */
    , "invalid event"                   /* -58 */
    , "invalid operation"               /* -59 */
    , "invalid gl object"               /* -60 */
    , "invalid buffer size"             /* -61 */
    , "invalid mip level"               /* -62 */
    , "invalid global work size"        /* -63 */
  };

  return strings[-err];
}

bool cl_init()
{
  cl_int err = clGetPlatformIDs (1, &cl_state.platform, NULL);
  if(err != CL_SUCCESS)
    {
      fprintf(stderr,"Could not create platform\n");
      return false;
    }

  clGetPlatformInfo (cl_state.platform, CL_PLATFORM_NAME,       sizeof(cl_state.platform_name),    cl_state.platform_name,    NULL);
  clGetPlatformInfo (cl_state.platform, CL_PLATFORM_VERSION,    sizeof(cl_state.platform_version), cl_state.platform_version, NULL);
  clGetPlatformInfo (cl_state.platform, CL_PLATFORM_EXTENSIONS, sizeof(cl_state.platform_ext),     cl_state.platform_ext,     NULL);

  err = clGetDeviceIDs (cl_state.platform, CL_DEVICE_TYPE_DEFAULT, 1, &cl_state.device, NULL);
  if(err != CL_SUCCESS)
    {
      fprintf(stderr,"Error: %s\n", cl_errstring(err));
      fprintf(stderr,"Could not create device\n");
      return false;
    }

  clGetDeviceInfo(cl_state.device, CL_DEVICE_NAME, sizeof(cl_state.device_name), cl_state.device_name, NULL);

  clGetDeviceInfo (cl_state.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &cl_state.max_mem_alloc,    NULL);
  clGetDeviceInfo (cl_state.device, CL_DEVICE_LOCAL_MEM_SIZE,     sizeof(cl_ulong), &cl_state.local_mem_size,   NULL);

  printf("Platform Name:%s\n",       cl_state.platform_name);
  printf(" Version:%s\n",            cl_state.platform_version);
  printf("Extensions:%s\n",          cl_state.platform_ext);
  printf("Default Device Name:%s\n", cl_state.device_name);
  printf("Max Alloc: %lu bytes\n",   (unsigned long)cl_state.max_mem_alloc);
  printf("Local Mem: %lu bytes\n",   (unsigned long)cl_state.local_mem_size);

  cl_state.context = clCreateContext(0, 1, &cl_state.device, NULL, NULL, &err);
  if(err != CL_SUCCESS)
    {
      fprintf(stderr,"Could not create context\n");
      return false;
    }

  cl_state.command_queue = clCreateCommandQueue(cl_state.context, cl_state.device, 0, &err);

  if(err != CL_SUCCESS)
    {
      fprintf(stderr,"Could not create command queue\n");
      return false;
    }

  fprintf(stderr,"OK\n");
  return true;
}

cl_kernel *
cl_compile_and_build (const char *program_source, const char *kernel_name[])
{
  cl_int errcode;

  size_t length = strlen(program_source);

  int i;
  int kernel_n = 0;
  while (kernel_name[++kernel_n] != NULL);

  CL_SAFE_CALL( cl_program program = clCreateProgramWithSource(cl_state.context, 1,
                                                               &program_source,
                                                               &length, &errcode) );

  errcode = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errcode != CL_SUCCESS)
    {
      char *msg;
      size_t s;
      cl_int build_errcode = errcode;

      CL_SAFE_CALL( errcode = clGetProgramBuildInfo(program,
                                                    cl_state.device,
                                                    CL_PROGRAM_BUILD_LOG,
                                                    0, NULL, &s) );

      msg = (char*) malloc (s);
      CL_SAFE_CALL( errcode = clGetProgramBuildInfo(program,
                                                    cl_state.device,
                                                    CL_PROGRAM_BUILD_LOG,
                                                    s, msg, NULL) );

      fprintf(stderr, "Build Error:%s\n%s\n\n", cl_errstring(build_errcode), msg);
      free (msg);

      return NULL;
    }
  else
    {
      fprintf(stderr, "Compiling successful\n");
    }

  cl_kernel *kernels = (cl_kernel *) malloc(sizeof(cl_kernel) * kernel_n);

  for (i=0; i<kernel_n; i++)
    {
      CL_SAFE_CALL( kernels[i] = clCreateKernel(program, kernel_name[i], &errcode) );
    }

  return kernels;
}
