#include <Halide.h>

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

    int W = 2050, H = 2050;

  struct Stats sobel_time, sobel_time_gpu;

    Image<float> input(W, H);

    Var x("x"), y("y"), yo("yo"), yi("yi"), xo("xo"), xi("xi");

    Func //clamped("clamped"),
         grad_x_h("grad_x_h"),
         grad_x_v("grad_x_v"),
         grad_y_v("grad_y_v"),
         grad_y_h("grad_y_h"),
         magnitude("magnitude");

    //clamped(x,y) = input(clamp(x, 0, W-1),
    //                     clamp(y, 0, H-1));

    grad_x_h(x,y) = (input(x,   y)
                   - input(x+2, y));

    grad_x_v(x,y) = (grad_x_h(x, y  )
                   + grad_x_h(x, y+1) * 2.0f
                   + grad_x_h(x, y+2));

    grad_y_h(x,y) = (input(x,   y)
                   + input(x+1, y) * 2.0f
                   + input(x+2, y));

    grad_y_v(x,y) = (grad_y_h(x,y  )
                   - grad_y_h(x,y+2));

    magnitude(x,y) = (grad_x_v(x,y) * grad_x_v(x,y)
                    + grad_y_v(x,y) * grad_y_v(x,y));

    int sched = atoi(argv[1]);

    switch(sched)
    {
      case 0:
        grad_x_h .root();
        grad_x_v .root();
        grad_y_h .root();
        grad_y_v .root();
        magnitude.root();
        break;

      case 1:
        grad_x_h .root().parallel(y);
        grad_x_v .root().parallel(y);
        grad_y_h .root().parallel(y);
        grad_y_v .root().parallel(y);
        magnitude.root().parallel(y);
        break;

      case 2:
        grad_x_h .root().parallel(y).vectorize(x, 4);
        grad_x_v .root().parallel(y).vectorize(x, 4);
        grad_y_h .root().parallel(y).vectorize(x, 4);
        grad_y_v .root().parallel(y).vectorize(x, 4);
        magnitude.root().parallel(y).vectorize(x, 4);
        break;

      case 3:
        magnitude.root().parallel(y).vectorize(x, 4);

        grad_x_v.split(y, yo, yi, 4).vectorize(x, 4);
        grad_x_h.chunk(x).vectorize(x, 4);

        grad_y_v.split(y, yo, yi, 4).vectorize(x, 4);
        grad_y_h.chunk(x).vectorize(x, 4);
        break;

      case 4:
        magnitude.root().parallel(y).vectorize(x, 4);

        grad_x_v.split(y, yo, yi, 32).split(x, xo, xi, 128).vectorize(xi, 4);
        grad_x_h.chunk(x).vectorize(x, 4);

        grad_y_v.split(y, yo, yi, 32).split(x, xo, xi, 128).vectorize(xi, 4);
        grad_y_h.chunk(x).vectorize(x, 4);
        break;

      case 5:
        magnitude.root().parallel(y);
        break;

      case 6:
        grad_x_h .vectorize(x, 4);
        grad_x_v .vectorize(x, 4);
        grad_y_h .vectorize(x, 4);
        grad_y_v .vectorize(x, 4);
        magnitude.root().parallel(y).vectorize(x, 4);
        break;

      case 50:
        magnitude.root().parallel(y);

        grad_x_v.split(y, yo, yi, 4);
        grad_x_h.chunk(x);

        grad_y_v.split(y, yo, yi, 4);
        grad_y_h.chunk(x);
        break;

      case 51:
        magnitude.root().parallel(y).vectorize(x, 4);

        grad_x_v.root().split(y, yo, yi, 32).split(x, xo, xi, 128).vectorize(xi, 4).parallel(yo);
        grad_x_h.chunk(yo,yi).vectorize(x, 4);

        grad_y_v.root().split(y, yo, yi, 32).split(x, xo, xi, 128).vectorize(xi, 4).parallel(yo);
        grad_y_h.chunk(yo,yi).vectorize(x, 4);
        break;

      case 52:
        magnitude.root().parallel(y).vectorize(x, 4);

        grad_x_v.vectorize(x, 4);
        grad_x_h.chunk(x).vectorize(x, 4);

        grad_y_v.vectorize(x, 4);
        grad_y_h.chunk(x).vectorize(x, 4);
        break;

      case 53:
        magnitude.split(y, yo, yi, 32).split(x, xo, xi, 128).vectorize(xi, 4).parallel(yo);

        grad_x_v.chunk(yo,yi).vectorize(x, 4);
        grad_x_h.chunk(x).vectorize(x, 4);

        grad_y_v.chunk(yo,yi).vectorize(x, 4);
        grad_y_h.chunk(x).vectorize(x, 4);
        break;

      case 100:
        magnitude.root().cudaTile(x,y,16,16);

      case 101:
        grad_x_h.root().cudaTile(x,y,16,16);
        grad_x_v.root().cudaTile(x,y,16,16);

        grad_y_h.root().cudaTile(x,y,16,16);
        grad_y_v.root().cudaTile(x,y,16,16);

        magnitude.root().cudaTile(x,y,16,16);

        break;

      // case 4:

      //  blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
      //  blur_x.chunk(y).vectorize(x, 8);
      //  break;
    }

    for(int k=0; k<NTRIES; k++)
      {
        Image<float> out (input.width()-2, input.height()-2, 1);

        TIME_START(sobel_time)

        input.markHostDirty(); /* copy CPU -> GPU */

        TIME_START(sobel_time_gpu)
        magnitude.realize(out);
        TIME_END(sobel_time_gpu, k)

        out.copyToHost();      /* copy GPU -> CPU */

        TIME_END(sobel_time, k)
      }

    printf("[Halide]\n");
    printf("- SOBEL:       %lf \n", sobel_time.min);
    printf("- SOBEL GPU:   %lf \n", sobel_time_gpu.min);

    return 0;
}
