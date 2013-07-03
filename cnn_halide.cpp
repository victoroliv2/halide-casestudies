#include "Halide.h"

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <cfloat>
#include <sys/time.h>

Var x, y, f1, f2, batch;

#define NTRIES 10

char * func_name(const char *s, int line)
{
  static int k = 0;
  char *ss = (char*) malloc(128);
  sprintf(ss, "%s_%d_%d", s, line, k++);
  return ss;
}

#define DECL_FUNC(name) Func name(func_name(#name, __LINE__));

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

Expr tanh(Expr x)
{
  return (1.0f - exp(-2.0f * x)) / (1.0f + exp(-2.0f * x));
}

// Downsample by a factor of (2 x 2)
Func max_pool_tanh(Func f) {
    DECL_FUNC(down)
    DECL_FUNC(blah)

    down(x, y, f1, batch) = max(max(max(f(2*x,   2*y,   f1, batch),
                                        f(2*x+1, 2*y,   f1, batch)),
                                        f(2*x,   2*y+1, f1, batch)),
                                        f(2*x+1, 2*y+1, f1, batch));

    blah(x, y, f1, batch) = tanh( down(x, y, f1, batch) );

    if (use_gpu())
      {
        down.root().cudaTile(x,y,4,4);
        blah.root().cudaTile(x,y,4,4);
      }
    else
      {
        down.root();
        blah.root();
      }

    return down;
}

int main(int argc, char * argv[])
{
#define DATASET_SIZE 1024
// #define KERNEL_0 20
#define KERNEL_0 10
#define KERNEL_1 45 /* 50 doesn'r work, WHY? */

  Image<float> input (28, 28, DATASET_SIZE);
  Image<float> fmask1(5,  5,  KERNEL_0);
  Image<float> fmask2(5,  5,  KERNEL_0, KERNEL_1);

  Func input_f,
       fmask1_f,
       fmask2_f;

  Func layer0_conv("layer0_conv");
  Func layer0_down("layer0_down");

  Func layer1_conv1("layer1_conv1");
  Func layer1_conv2("layer1_conv2");
  Func layer1_down("layer1_down");

  /* input: 28x28x1xN output: 24x24xF1xN */

  // input_f(x,y,batch)  = input(clamp(x, 0, 27), clamp(y, 0, 27), clamp(batch, 0, DATASET_SIZE-1));
  // fmask1_f(x,y,f1)    = fmask1(clamp(x, 0, 4), clamp(y, 0, 4), clamp(f1, 0, KERNEL_0-1));
  // fmask2_f(x,y,f1,f2) = fmask2(clamp(x, 0, 4), clamp(y, 0, 4), clamp(f1, 0, KERNEL_0-1), clamp(f2, 0, KERNEL_1-1));

  RDom R0(0, 5,
          0, 5);

  layer0_conv(x, y, f1, batch) += fmask1(R0.x, R0.y, f1) * input(x+R0.x, y+R0.y, batch);

  /* input: 24x24xF1xN output: 12x12xF1xN */
  layer0_down = max_pool_tanh(layer0_conv);

  /* input: 12x12xF1xN output: 8x8xF2xN */

  layer1_conv1(x, y, f1, f2, batch) += fmask2(R0.x, R0.y, f1, f2) * layer0_down(x+R0.x, y+R0.y, f1, batch);

  RDom R1(0, KERNEL_0);

  layer1_conv2(x, y, f2, batch) += layer1_conv1(x, y, R1, f2, batch);

  /* input: 8x8xF2xN output: 4x4xF2xN */
  layer1_down = max_pool_tanh(layer1_conv2);

  /* here it ends the GPU part */

  /* schedules */

  if (use_gpu())
    {
      layer0_conv.reorder(f1, x, y, batch).cudaTile(x,y,1,1);
      layer0_conv.update().reorder(R0.x, R0.y, f1, x, y).cudaTile(x,y,1,1);

      layer1_conv1.reorder(f1, f2, x, y, batch).cudaTile(x,y,1,1);
      layer1_conv1.update().reorder(R0.x, R0.y, f1, f2).cudaTile(x,y,1,1);

      layer1_conv2.reorder(f2, x, y, batch).cudaTile(x,y,1,1);
      layer1_conv2.update().reorder(R1, f2, x, y, batch).cudaTile(x,y,1,1);
    }
  else
    {
      layer0_conv.root();
      layer1_conv1.root();
      layer1_conv2.root();
    }

  /* track down bug */

  // Image<float> out1(24, 24, KERNEL_0, DATASET_SIZE);
  // layer0_conv.realize(out1);

  // fprintf(stderr, ">>>>>>>>>>>>>> 1\n");

  // Image<float> out2(12, 12, KERNEL_0, DATASET_SIZE);
  // layer0_down.realize(out2);

  // fprintf(stderr, ">>>>>>>>>>>>>> 2\n");

  Image<float> out3(8, 8, KERNEL_1, DATASET_SIZE);
  layer1_conv2.realize(out3);

  // fprintf(stderr, ">>>>>>>>>>>>>> 3\n");

  //Image<float> out4(4, 4, KERNEL_1, DATASET_SIZE);
  //layer1_down.realize(out4);

  fprintf(stderr, ">>>>>>>>>>>>>> 4\n");

  return 0;
}
