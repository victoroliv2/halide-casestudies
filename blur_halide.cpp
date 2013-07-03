#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>
#include <memory>
#include <cfloat>
#include <vector>
#include <sys/time.h>

#define NTRIES 10

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

int main(int argc, char **argv) {

  Image<float> input (2050, 2050, 1);

  struct Stats blur_time;

  Func blur_x("blur_x"), blur_y("blur_y");
  Var x("x"), y("y"), yo("yo"), xo("xo"), xi("xi"), yi("yi");

  // The algorithm
  blur_x(x, y) = (input(x, y+1) + input(x+1, y+1) + input(x+2, y+1))/3;
  blur_y(x, y) = (blur_x(x+1, y) + blur_x(x+1, y+1) + blur_x(x+1, y+2))/3;

  int sched = atoi(argv[1]);

  switch(sched)
  {
    case 0:

      blur_x.root();
      blur_y.root();
      break;

    case 1:

      blur_x.root().parallel(y);
      blur_y.root().parallel(y);
      break;

    case 2:

      blur_y.split(y, yo, yi, 4);
      blur_y.parallel(yo);
      blur_y.vectorize(x, 4);

      blur_x.chunk(yo);
      blur_x.vectorize(x, 4);
      break;

    case 3:

      blur_y.tile(x, y, xi, yi, 128, 32);
      blur_y.vectorize(xi, 4);
      blur_y.parallel(y);
      blur_x.chunk(x);
      blur_x.vectorize(x, 4);
      break;

    case 4:

      blur_y.root().parallel(y).vectorize(x, 4);
      break;

    // case 4:

    //  blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    //  blur_x.chunk(y).vectorize(x, 8);
    //  break;
  }

  for(int k=0; k<NTRIES; k++)
    {
      Image<float> out (input.width()-2, input.height()-2, 1);

      TIME_START(blur_time)

      blur_y.realize(out);

      TIME_END(blur_time, k)
    }

  printf("[Halide]\n");
  printf("- BOX-BLUR:       %lf \n", blur_time.min);

  return 0;
}
