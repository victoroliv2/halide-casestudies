#include "Halide.h"

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <memory>
#include <cfloat>
#include <vector>
#include <sys/time.h>

Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");

#define POW2(x) ((x)*(x))

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

Func blur(Func gray, const float sigma) {
  DECL_FUNC(gaussian)

  gaussian(x) = exp(-(x/sigma)*(x/sigma)*0.5f);

  // truncate to 3 sigma and normalize
  int radius = int(3*sigma + 1.0f);
  RDom i(-radius, 2*radius+1);

  DECL_FUNC(normalized)
  normalized(x) = gaussian(x) / sum(gaussian(i)); // Uses an inline reduction

  // Convolve the input using two reductions
  DECL_FUNC(blurx)
  DECL_FUNC(blury)
  blurx(x, y) += gray (x+i, y  ) * normalized(i);
  blury(x, y) += blurx(x,   y+i) * normalized(i);

  if (use_gpu())
    {
      normalized.root().cudaTile(x,2);
      blurx.root().cudaTile(x,y,16,16);
      blury.root().cudaTile(x,y,16,16);
      blurx.update().root().reorder(i,x,y).cudaTile(x,y,16,16);
      blury.update().root().reorder(i,x,y).cudaTile(x,y,16,16);
    }
  else
    {
      normalized.root();
      blurx.update().root().reorder(i,x,y).parallel(y).vectorize(x, 4);
      blury.update().root().reorder(i,x,y).parallel(y).vectorize(x, 4);
    }

  return blury;
}

#define F_PI 3.14159265f

int main(int argc, char * argv[])
{
  #define SIGMA 1.6f

  struct Stats harris_time;

  assert(argc == 3);

  Image<float> input = load<float>(argv[1]);

  printf("(%d %d %d)\n", input.width(), input.height(), input.channels());

  float threshold = atof(argv[2]);

  printf("threshold: %f\n", threshold);

  DECL_FUNC(gray);
  gray(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

  DECL_FUNC(clamped);
  clamped(x, y) = gray(clamp(x, 0, input.width()  - 1),
                       clamp(y, 0, input.height() - 1));

  /* compute Gx and Gy using sobel masks*/

  DECL_FUNC(sobel_Gx_1);
  DECL_FUNC(sobel_Gx_2);
  DECL_FUNC(sobel_Gy_1);
  DECL_FUNC(sobel_Gy_2);

  // sobel x
  sobel_Gx_1(x,y) = clamped(x,y-1) - clamped(x,y+1);
  sobel_Gx_2(x,y) = sobel_Gx_1(x,y-1) + 2.0f*sobel_Gx_1(x,y) + sobel_Gx_1(x,y+1);

  // sobel y
  sobel_Gy_1(x,y) = clamped(x-1,y) + 2.0f*clamped(x,y) + clamped(x+1,y);
  sobel_Gy_2(x,y) = sobel_Gy_1(x,y-1) - sobel_Gy_1(x,y+1);

  DECL_FUNC(sobel_xx)
  DECL_FUNC(sobel_xy)
  DECL_FUNC(sobel_yy)

  sobel_xx(x,y) = sobel_Gx_2(x,y) * sobel_Gx_2(x,y);
  sobel_xy(x,y) = sobel_Gx_2(x,y) * sobel_Gy_2(x,y);
  sobel_yy(x,y) = sobel_Gy_2(x,y) * sobel_Gy_2(x,y);

  DECL_FUNC(blur_xx)
  DECL_FUNC(blur_xy)
  DECL_FUNC(blur_yy)

  blur_xx = blur(sobel_xx, SIGMA);
  blur_xy = blur(sobel_xy, SIGMA);
  blur_yy = blur(sobel_yy, SIGMA);

  DECL_FUNC(det)
  DECL_FUNC(trace)

  det(x,y) = blur_xx(x,y) * blur_yy(x,y) - blur_xy(x,y) * blur_xy(x,y);
  trace(x,y) = blur_xx(x,y) + blur_yy(x,y);

  const float k = 0.04f;

  DECL_FUNC(cornerness)
  cornerness(x,y) = det(x,y) - k * (trace(x,y) * trace(x,y));

  DECL_FUNC(cornerness_thresh)
  cornerness_thresh(x,y) = select(cornerness(x,y) > threshold, cornerness(x,y), -FLT_MAX);

  RDom r(-1,2,-1,2);

  DECL_FUNC(cornerness_maxima)
  cornerness_maxima(x,y) = -FLT_MAX;
  cornerness_maxima(x,y) = max(cornerness_maxima(x,y), cornerness_thresh(x+r.x,y+r.y));

  DECL_FUNC(cornerness_suppression)
  cornerness_suppression(x,y) = (cornerness_thresh(x,y) == cornerness_maxima(x,y)) * 1.0f;

  if (use_gpu())
    {
      //sobel_Gx_1.root().cudaTile(x,y,16,16);
      sobel_Gx_2.root().cudaTile(x,y,16,16);
      //sobel_Gy_1.root().cudaTile(x,y,16,16);
      sobel_Gy_2.root().cudaTile(x,y,16,16);
      sobel_xx.root().cudaTile(x,y,16,16);
      sobel_xy.root().cudaTile(x,y,16,16);
      sobel_yy.root().cudaTile(x,y,16,16);
      //det.root().cudaTile(x,y,16,16);
      //trace.root().cudaTile(x,y,16,16);
      //cornerness.root().cudaTile(x,y,16,16);
      cornerness_thresh.root().cudaTile(x,y,16,16);
      cornerness_maxima.root().cudaTile(x,y,16,16);
      cornerness_maxima.update().root().cudaTile(x,y,16,16);
      cornerness_suppression.root().cudaTile(x,y,16,16);
    }
  else
    {
      sobel_Gx_2.root().parallel(y).vectorize(x, 4);
      sobel_Gx_1.root().parallel(y).vectorize(x, 4);

      sobel_Gy_2.root().parallel(y).vectorize(x, 4);
      sobel_Gy_1.root().parallel(y).vectorize(x, 4);

      blur_xx.root().parallel(y).vectorize(x, 4);
      blur_xy.root().parallel(y).vectorize(x, 4);
      blur_yy.root().parallel(y).vectorize(x, 4);

      cornerness.root().parallel(y).vectorize(x, 4);
      cornerness_thresh.root().parallel(y).vectorize(x, 4);

      cornerness_maxima.root().parallel(y).vectorize(x, 4);
      cornerness_suppression.root().parallel(y).vectorize(x, 4);
    }

  cornerness_suppression.compileJIT();

  for(int k=0; k<NTRIES; k++)
    {
      Image<float> out (input.width(), input.height(), 1);

      TIME_START(harris_time)

      input.markHostDirty(); /* copy CPU -> GPU */

      cornerness_suppression.realize(out);

      out.copyToHost();      /* copy GPU -> CPU */

      TIME_END(harris_time, k)
    }

  printf("[Halide]\n");
  printf("- HARRIS:       %lf \n", harris_time.min);

  return 0;
}
