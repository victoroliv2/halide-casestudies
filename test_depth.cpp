#include "Halide.h"

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <cfloat>
#include <sys/time.h>

Var x, y, f1, f2, batch;

int main(int argc, char * argv[])
{

#define DATASET_SIZE 1024
#define KERNEL_0 10
// it works if this is 10, but I'm not sure why
#define KERNEL_1 50

  Image<float> input (28, 28, DATASET_SIZE);

  Image<float> fmask2(5,  5,  KERNEL_0, KERNEL_1);

  Func layer1("layer1");

  RDom R(0, 5,
         0, 5,
         0, KERNEL_0);

  layer1(x, y, f2, batch) += fmask2(R.x, R.y, R.z, f2);

  if (use_gpu())
    {
      layer1.root().cudaTile(x,y,4,4);
    }
  else
    {
      layer1.root();
    }

  Image<float> out(8, 8, KERNEL_1, DATASET_SIZE);
  layer1.realize(out);

  return 0;
}
