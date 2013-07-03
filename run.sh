COMPUTE_PROFILE=1 COMPUTE_PROFILE_CONFIG=prof_config COMPUTE_PROFILE_LOG=profile.csv ./main_opencl wilson-4mpix.png out.png
COMPUTE_PROFILE=1 COMPUTE_PROFILE_CONFIG=prof_config COMPUTE_PROFILE_LOG=profile.csv ./main_opencl wilson2.png out.png

CUDA_PROFILE=1 CUDA=prof_config CUDA_PROFILE_LOG=profile2.csv HL_TARGET=ptx HL_CUDADEVICE=0 ./main_halide wilson-4mpix.png
CUDA_PROFILE=1 CUDA=prof_config CUDA_PROFILE_LOG=profile2.csv HL_TARGET=ptx HL_CUDADEVICE=0 ./harris_halide wilson-4mpix.png out.png
CUDA_PROFILE=1 CUDA=prof_config CUDA_PROFILE_LOG=profile2.csv HL_TARGET=ptx HL_CUDADEVICE=0 ./sift_halide wilson2.png

HL_TARGET=x86_64 HL_NUMTHREADS=4 ./main_halide wilson-4mpix.png
HL_TARGET=x86_64 HL_NUMTHREADS=4 ./harris_halide wilson-4mpix.png out.png
HL_TARGET=x86_64 HL_NUMTHREADS=4 ./sift_halide wilson2.png
