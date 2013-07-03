#include "Halide.h"

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <cfloat>
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

Expr lerp(Expr a, Expr b, Expr alpha) {
    return (1.0f - alpha)*a + alpha*b;
}

Expr copysign(Expr mag, Expr sig)
{
  Expr s = select(sig < 0.0f, -1.0f, 1.0f);
  return abs(mag) * s;
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

Var x("x"), y("y"), z("z"), c("c"), k("k");

int main(int argc, char * argv[])
{
  assert(argc == 2);

  struct Stats lab_time, mb_time, unsharp_time, bf_time;

  Image<float> input = load<float>(argv[1]);

  printf("(%d %d %d)\n", input.width(), input.height(), input.channels());

  Func clamped("clamped");
  clamped(x, y, c) = input(clamp(x, 0, input.width()  - 1), clamp(y, 0, input.height() - 1), c);

  Func cielab("cielab");

  {
    Var tx("tx"), ty("ty"), xi("xi"), yi("yi");

    Func gamma("gamma");
    gamma(x,y,c) = 100.0f * select(input(x,y,c) > 0.04045f, pow(((input(x,y,c) + 0.055f) / 1.055f), 2.4f),
                                                            input(x,y,c) / 12.92f);

    Func xyz("xyz");

    Expr X = (gamma(x,y,0) * 0.4124f + gamma(x,y,1) * 0.3576f + gamma(x,y,2) * 0.1805f) / 95.047f;
    Expr Y = (gamma(x,y,0) * 0.2126f + gamma(x,y,1) * 0.7152f + gamma(x,y,2) * 0.0722f) / 100.000f;
    Expr Z = (gamma(x,y,0) * 0.0193f + gamma(x,y,1) * 0.1192f + gamma(x,y,2) * 0.9505f) / 108.883f;

    X = select(X > 0.008856f, pow(X, 1.0f/3.0f), ( 7.787f * X ) + ( 16.0f / 116.0f ));
    Y = select(Y > 0.008856f, pow(Y, 1.0f/3.0f), ( 7.787f * Y ) + ( 16.0f / 116.0f ));
    Z = select(Z > 0.008856f, pow(Z, 1.0f/3.0f), ( 7.787f * Z ) + ( 16.0f / 116.0f ));

    xyz(x,y,c) = select(c == 0, X,
                 select(c == 1, Y,
                                Z));

    Expr CIEL = ( 116.0f * xyz(tx,ty,1) ) - 16.0f;
    Expr CIEa = 500.0f * ( xyz(tx,ty,0) - xyz(tx,ty,1) );
    Expr CIEb = 200.0f * ( xyz(tx,ty,1) - xyz(tx,ty,2) );

    cielab(tx,ty,c) = select(c == 0, CIEL,
                      select(c == 1, CIEa,
                      select(c == 2, CIEb,
                                     input(tx,ty,3))));

    if (use_gpu())
      {
        cielab.reorder(c,tx,ty).unroll(c, 4).root().cudaTile(tx,ty,16,16);
      }
    else
      {
        // cielab.vectorize(tx, 4).reorder(c,tx,ty).unroll(c, 4).root().parallel(ty);

        gamma.chunk(tx).vectorize(x, 4).reorder(c, x, y).unroll(c, 3);
        xyz.chunk(tx).vectorize(x, 4).reorder(c, x, y).unroll(c, 3);
        cielab.tile(tx, ty, xi, yi, 128, 32).vectorize(xi, 4).reorder(xi, yi, c, tx, ty);
        cielab.parallel(ty);
      }
  }

  Func motion_blur("motion_blur");

  float length = 10.0f;
  float angle  = 45.0f;

  {
    Func acc_mb("acc_mb"), output("output");

    float theta = angle * (float)M_PI / 180.0f;
    float offset_x = length * std::cos(theta);
    float offset_y = length * std::sin(theta);
    int   num_steps = (int)(length+0.5f) + 1;

    Var tx("tx"), ty("ty"), xi("xi"), yi("yi");

    RDom step(0, num_steps);

    Expr t = (num_steps == 1)? 0.0f : step / (float)(num_steps - 1) - 0.5f;

    Expr xx = x + t * offset_x;
    Expr yy = y + t * offset_y;

    Expr dx = xx - floor(xx);
    Expr dy = yy - floor(yy);

    Expr ix = cast<int>(xx);
    Expr iy = cast<int>(yy);

    Expr mixy0 = dy * (clamped(ix,  iy+1,c) - clamped(ix,  iy,c)) + clamped(ix,  iy,c);
    Expr mixy1 = dy * (clamped(ix+1,iy+1,c) - clamped(ix+1,iy,c)) + clamped(ix+1,iy,c);

    acc_mb(x,y,c) += dx * (mixy1 - mixy0) + mixy0;

    motion_blur(x,y,c) = acc_mb(x,y,c) / float(num_steps);

    if (use_gpu())
      {
        acc_mb.reorder(c,x,y).root().cudaTile(x,y,16,16);
        acc_mb.update().reorder(step,c,x,y).root().cudaTile(x,y,16,16);
        motion_blur.reorder(c,x,y).root().cudaTile(x,y,16,16);
      }
    else
      {

        //acc_mb.update().reorder(c,x,y).chunk(x).vectorize(c, 4);
        //motion_blur.tile(x, y, xi, yi, 128, 32).parallel(y).vectorize(xi, 4);

        // motion_blur.split(y, y, yi, 4).parallel(y).vectorize(x, 4);
        // acc_mb.chunk(y, yi)/* .reorder(c,x,y) */.vectorize(x, 4);

        motion_blur.root().reorder(c,x,y).parallel(y).unroll(c,4).vectorize(x, 4);
        acc_mb.update().reorder(c,x,y).parallel(y).unroll(c,4).vectorize(x, 4);
      }
  }

  Func unsharped_mask("unsharp_mask");

  {
    float sigma = 1.5f;
    Uniform<float> detail_thresh = 0.5f;
    Uniform<float> sharpen = 0.5f;

    Func gaussian("gaussian");
    gaussian(x) = exp(-(x/sigma)*(x/sigma)*0.5f);

    // truncate to 3 sigma and normalize
    int radius = int(3*sigma + 1.0f);
    RDom i(-radius, 2*radius+1);

    Func normalized("normalized");
    normalized(x) = gaussian(x) / sum(gaussian(i)); // Uses an inline reduction

    // Convolve the input using two reductions
    Func blurx("blurx");
    Func blury("blury");
    blurx(x, y, c) += clamped(x+i, y, c) * normalized(i);
    blury(x, y, c) +=   blurx(x, y+i, c) * normalized(i);

    Func detail("detail");
    detail(x, y, c) = blury(x, y, c) - clamped(x, y, c);

    unsharped_mask(x, y, c) = clamped(x, y, c) + select(detail(x, y, c) > detail_thresh,
                                                        sharpen * copysign(max(abs(detail(x,y,c)) - detail_thresh, 0.0f), detail(x,y,c)),
                                                        0.0f);

    if (use_gpu())
      {
        normalized.root().cudaTile(x,2);
        blurx.root().cudaTile(x,y,16,16);
        blury.root().cudaTile(x,y,16,16);
        blurx.update().root().reorder(i,c,x,y).cudaTile(x,y,16,16);
        blury.update().root().reorder(i,c,x,y).cudaTile(x,y,16,16);
        unsharped_mask.root().reorder(c,x,y).cudaTile(x,y,16,16);
      }
    else
      {
        normalized.root();
        blurx.update().root().reorder(i,c,x,y).parallel(y).unroll(c,4).vectorize(x, 4);
        blury.update().root().reorder(i,c,x,y).parallel(y).unroll(c,4).vectorize(x, 4);
        detail.root().reorder(c,x,y).parallel(y).unroll(c,4).vectorize(x, 4);
        unsharped_mask.root().reorder(c,x,y).parallel(y).unroll(c,4).vectorize(x, 4);
      }
  }

  Uniform<float> r_sigma = 0.05f;
  int s_sigma = 8;

  Func smoothed("smoothed");

  {
    // Construct the bilateral grid
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma/2, y * s_sigma + r.y - s_sigma/2, c);
    val = clamp(val, 0.0f, 1.0f);
    Expr zi = cast<int>(val * (1.0f/r_sigma) + 0.5f);
    Func grid("grid");
    grid(x, y, zi, c, k) += select(k == 0, val, 1.0f);

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");

    blurx(x, y, z, c, k) =  grid(x-1, y,   z  , c, k) +  grid(x, y, z, c, k)*4.0f +  grid(x+1, y,   z,   c, k);
    blury(x, y, z, c, k) = blurx(x,   y-1, z  , c, k) + blurx(x, y, z, c, k)*4.0f + blurx(x,   y+1, z,   c, k);
    blurz(x, y, z, c, k) = blury(x,   y,   z-1, c, k) + blury(x, y, z, c, k)*4.0f + blury(x,   y,   z+1, c, k);

    // Take trilinear samples to compute the output
    val = clamp(clamped(x, y, c), 0.0f, 1.0f);
    Expr zv = val * (1.0f/r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x/s_sigma;
    Expr yi = y/s_sigma;

    Func interpolated("interpolated");
    interpolated(x, y, c, k) =
        lerp(lerp(lerp(blurz(xi, yi,   zi  , c, k), blurz(xi+1, yi,   zi  , c, k), xf),
                  lerp(blurz(xi, yi+1, zi  , c, k), blurz(xi+1, yi+1, zi  , c, k), xf), yf),
             lerp(lerp(blurz(xi, yi,   zi+1, c, k), blurz(xi+1, yi,   zi+1, c, k), xf),
                  lerp(blurz(xi, yi+1, zi+1, c, k), blurz(xi+1, yi+1, zi+1, c, k), xf), yf), zf);

    // Normalize
    smoothed(x, y, c) = interpolated(x, y, c, 0)/interpolated(x, y, c, 1);

    if (use_gpu())
      {
        //OK
        Var gridz;
        gridz = grid.arg(2);
        grid.root().cudaTile(x, y, 16, 16);
        grid.update().reorder(k, c, x, y).root().cudaTile(x, y, 16, 16);
        blurx.root().reorder(k, c, x, y).cudaTile(x, y, 8, 8);
        blury.root().reorder(k, c, x, y).cudaTile(x, y, 8, 8);
        blurz.root().reorder(k, c, x, y).cudaTile(x, y, 8, 8);
        smoothed.root().cudaTile(x, y, s_sigma, s_sigma);
      }
    else
      {
        //OK
        grid.root().parallel(z);
        grid.update().reorder(k, c, x, y).parallel(y);
        blurx.root().parallel(z).vectorize(x, 4);
        blury.root().parallel(z).vectorize(x, 4);
        blurz.root().parallel(z).vectorize(x, 4);
        smoothed.root().parallel(y).vectorize(x, 4);
      }
  }

  {
    cielab.compileJIT();
    motion_blur.compileJIT();
    unsharped_mask.compileJIT();
    smoothed.compileJIT();
  }

  for(int k=0; k<NTRIES; k++)
    {
      Image<float> out (input.width(), input.height(), input.channels());

      TIME_START(lab_time)

      input.markHostDirty(); /* copy CPU -> GPU */

      cielab.realize(out);

      out.copyToHost();      /* copy GPU -> CPU */

      TIME_END(lab_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      Image<float> out (input.width(), input.height(), input.channels());

      TIME_START(mb_time)

      input.markHostDirty(); /* copy CPU -> GPU */

      motion_blur.realize(out);

      out.copyToHost();      /* copy GPU -> CPU */

      TIME_END(mb_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      Image<float> out (input.width(), input.height(), input.channels());

      TIME_START(unsharp_time)

      input.markHostDirty(); /* copy CPU -> GPU */

      unsharped_mask.realize(out);

      out.copyToHost();      /* copy GPU -> CPU */

      TIME_END(unsharp_time, k)
    }

  for(int k=0; k<NTRIES; k++)
    {
      Image<float> out (input.width(), input.height(), input.channels());

      TIME_START(bf_time)

      input.markHostDirty(); /* copy CPU -> GPU */

      /* there is some memory leak */
      smoothed.realize(out);

      out.copyToHost();      /* copy GPU -> CPU */

      TIME_END(bf_time, k)
    }

  printf("[Halide]\n");

  printf("- CIELAB:       %lf \n", lab_time.min);
  printf("- MOTION-BLUR:  %lf \n", mb_time.min);
  printf("- UNSHARP-MASK: %lf \n", unsharp_time.min);
  printf("- BIL. FILTER:  %lf \n", bf_time.min);

  return 0;
}
