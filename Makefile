CC=g++-4.7

#CFLAGS=-Wall -g -O0 -fopenmp -std=c++0x -march=native -mtune=native -I /home/victorm/Halide/cpp_bindings -I /home/victorm/Halide/support
#CFLAGS=-Wall -g -fopenmp -std=c++0x -march=native -mtune=native -Ofast -funroll-loops -ftree-vectorizer-verbose=0 -I /home/victorm/halide-jrk-december-works-gpu
#CFLAGS=-Wall -g -fopenmp -std=c++0x -march=native -mtune=native -Ofast -funroll-loops -ftree-vectorizer-verbose=0 -I /home/victorm/Halide-stable
CFLAGS=-Wall -g -fopenmp -std=c++0x -march=native -mtune=native -Ofast -funroll-loops -ftree-vectorizer-verbose=0 -I /home/victorm/Halide/cpp_bindings -I /home/victorm/Halide/support

#LDFLAGS=-lOpenCL -lpthread -lrt -L /home/victorm/Halide/cpp_bindings -lHalide -L /usr/lib/x86_64-linux-gnu -ldl -L /usr/local/cuda/lib -lcuda
#LDFLAGS=-lOpenCL -lpthread -lrt -L /home/victorm/halide-jrk-december-works-gpu -lHalide -L /usr/lib/x86_64-linux-gnu -ldl
#LDFLAGS=-lOpenCL -lpthread -lrt -L /home/victorm/Halide-stable -lHalide -L /usr/lib/x86_64-linux-gnu -ldl
LDFLAGS=-lOpenCL -lpthread -lrt -L /home/victorm/Halide/cpp_bindings -lHalide -L /usr/lib/x86_64-linux-gnu -ldl

PNGFLAGS=$(shell libpng-config --ldflags) $(shell libpng-config --cflags)

HEADERS = opencl_help.h
OPENCL_KERNEL = kernel/*.cl.h
OPENCL_SUPPORT = opencl_help.o lodepng.o hdrloader.o
OPENCL_FILTER = cielab.o bilateral-filter-fast.o motion-blur.o unsharped_mask.o harris.o sift.o

%.o: %.cpp $(HEADERS) $(OPENCL_KERNEL)
	$(CC) -c -o $@ $< $(CFLAGS)

all: main_opencl main_halide test_histogram test_downsample test_maximum test_bandwidth test_complex_conv test_depth sift_halide cnn_halide harris_halide blur_halide sobel_halide

main_opencl: $(OPENCL_SUPPORT) $(OPENCL_FILTER) main_opencl.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

main_halide: $(OPENCL_SUPPORT) $(OPENCL_FILTER) main_halide.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

sift_halide: $(OPENCL_SUPPORT) $(OPENCL_FILTER) sift_halide.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

cnn_halide: $(OPENCL_SUPPORT) $(OPENCL_FILTER) cnn_halide.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

harris_halide: $(OPENCL_SUPPORT) $(OPENCL_FILTER) harris_halide.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

blur_halide: $(OPENCL_SUPPORT) $(OPENCL_FILTER) blur_halide.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

sobel_halide: $(OPENCL_SUPPORT) $(OPENCL_FILTER) sobel_halide.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

test_histogram: $(OPENCL_SUPPORT) $(OPENCL_FILTER) test_histogram.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

test_downsample: $(OPENCL_SUPPORT) $(OPENCL_FILTER) test_downsample.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

test_maximum: $(OPENCL_SUPPORT) $(OPENCL_FILTER) test_maximum.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

test_bandwidth: $(OPENCL_SUPPORT) $(OPENCL_FILTER) test_bandwidth.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

test_complex_conv: $(OPENCL_SUPPORT) $(OPENCL_FILTER) test_complex_conv.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

test_depth: $(OPENCL_SUPPORT) $(OPENCL_FILTER) test_depth.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(PNGFLAGS)

clean:
	rm -f *.o
	rm -f main_opencl
	rm -f main_halide
	rm -f test_histogram
	rm -f test_downsample
	rm -f test_maximum
	rm -f test_bandwidth
	rm -f test_complex_conv
	rm -f sift_halide
	rm -f cnn_halide
	rm -f harris_halide
	rm -f blur_halide
	rm -f sobel_halide
