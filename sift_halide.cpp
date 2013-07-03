#include "Halide.h"

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <memory>
#include <cfloat>
#include <vector>
#include <sys/time.h>

Var x("x"), y("y"), c("c"), k("k"), fi("fi");

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

// Downsample with a 1 3 3 1 filter
Func downsample(Func gray) {
    DECL_FUNC(downx)
    DECL_FUNC(downy)

    downx(x, y) = (gray(2*x-1, y) + 3.0f * (gray(2*x, y) + gray(2*x+1, y)) + gray(2*x+2, y)) / 8.0f;
    downy(x, y) = (downx(x, 2*y-1) + 3.0f * (downx(x, 2*y) + downx(x, 2*y+1)) + downx(x, 2*y+2)) / 8.0f;

    if (use_gpu())
      {
        downx.root().cudaTile(x,y,16,16);
        downy.root().cudaTile(x,y,16,16);
      }
    else
      {
        downx.root().parallel(y).vectorize(x, 4);
        downy.root().parallel(y).vectorize(x, 4);
      }

    return downy;
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
  blurx(x, y) += gray(x+i, y) * normalized(i);
  blury(x, y) += blurx(x, y+i) * normalized(i);

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
      blurx.update().root().parallel(y).vectorize(x, 4);
      blury.update().root().parallel(y).vectorize(x, 4);
    }

  return blury;
}

int main(int argc, char * argv[])
{
  #define SIGMA 1.6f
  const float contr_thr = 0.04f;
  const float curv_thr  = 10.0f;

  const int octaves   = 4;
  const int intervals = 2;

  assert(argc == 2);

  struct Stats sift_time;

  Image<float> input = load<float>(argv[1]);

  printf("(%d %d %d)\n", input.width(), input.height(), input.channels());

  DECL_FUNC(gray);
  gray(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

  DECL_FUNC(clamped);
  clamped(x, y) = gray(clamp(x, 0, input.width()  - 1),
                       clamp(y, 0, input.height() - 1));

  /* precompute gaussian sigmas */

  float sig[intervals+3], sig_prev, sig_total;

  sig[0] = SIGMA;
  float p = pow( 2.0, 1.0 / intervals );
  for(int i = 1; i < intervals + 3; i++ )
    {
      sig_prev = powf( p, i - 1 ) * SIGMA;
      sig_total = sig_prev * p;
      sig[i] = sqrtf( sig_total * sig_total - sig_prev * sig_prev );
    }

  Func gauss_pyr[octaves][intervals+3];

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 3; i++ )
      {
        if( o == 0  &&  i == 0 ) {
          gauss_pyr[o][i] = clamped;
        }
        /* base of new octvave is halved image from end of previous octave */
        else if( i == 0 ) {
          gauss_pyr[o][i] = downsample( gauss_pyr[o-1][i] );
        }
        /* blur the current octave's last image to create the next one */
        else {
          gauss_pyr[o][i] = blur(gauss_pyr[o][i-1], sig[i]);
        }
      }

  /* difference-of-gaussians pyramid */

  Func dog_pyr[octaves][intervals+2];

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 2; i++ ) {
      int W = input.width()  / (1<<o);
      int H = input.height() / (1<<o);

      DECL_FUNC(dog_pyr__)
      dog_pyr[o][i] = dog_pyr__;

      dog_pyr[o][i](x,y) = gauss_pyr[o][i+1](x,y) - gauss_pyr[o][i](x,y);
    }

  Func key[octaves][intervals];

  float prelim_contr_thr = 0.5f * contr_thr / intervals;

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      {
        Expr v = dog_pyr[o][i](x,y);

        RDom r(-1,2,-1,2);

        Expr dmax = max(dog_pyr[o][i+1](x+r.x,y+r.y),
                    max(dog_pyr[o][i  ](x+r.x,y+r.y),
                        dog_pyr[o][i-1](x+r.x,y+r.y)));

        DECL_FUNC(dog_max);
        //dog_max(x,y) = maximum(dmax); /* can't set schedule */
        dog_max(x,y) = -FLT_MAX;
        dog_max(x,y) = max(dog_max(x,y), dmax);

        Expr dmin = min(dog_pyr[o][i+1](x+r.x,y+r.y),
                    min(dog_pyr[o][i  ](x+r.x,y+r.y),
                        dog_pyr[o][i-1](x+r.x,y+r.y)));

        DECL_FUNC(dog_min);
        //dog_min(x,y) = minimum(dmin);
        dog_min(x,y) = FLT_MAX;
        dog_min(x,y) = min(dog_min(x,y), dmin);

        DECL_FUNC(is_extremum);
        is_extremum(x,y) = ((abs(v) > prelim_contr_thr) &&
                            ((v <= 0.0f && v == dog_min(x,y)) ||
                             (v >  0.0f && v == dog_max(x,y))));

        /* 2nd part */

        DECL_FUNC(dx);
        dx(x,y) = (dog_pyr[o][i](x+1, y) - dog_pyr[o][i](x-1, y)) / 2.0f;

        DECL_FUNC(dy);
        dy(x,y) = (dog_pyr[o][i](x, y+1) - dog_pyr[o][i](x, y-1)) / 2.0f;

        DECL_FUNC(ds);
        ds(x,y) = (dog_pyr[o][i+1](x, y) - dog_pyr[o][i-1](x, y)) / 2.0f;

        Expr dxx = dog_pyr[o][i  ](x+1, y  ) + dog_pyr[o][i  ](x-1, y  ) - 2.0f * v;
        Expr dyy = dog_pyr[o][i  ](x,   y+1) + dog_pyr[o][i  ](x,   y-1) - 2.0f * v;
        Expr dss = dog_pyr[o][i+1](x,   y  ) + dog_pyr[o][i-1](x,   y  ) - 2.0f * v;
        Expr dxy = ( dog_pyr[o][i  ](x+1, y+1) - dog_pyr[o][i  ](x-1, y+1) - dog_pyr[o][i  ](x+1, y-1) + dog_pyr[o][i  ](x-1, y-1) ) / 4.0f;
        Expr dxs = ( dog_pyr[o][i+1](x+1, y  ) - dog_pyr[o][i+1](x-1, y  ) - dog_pyr[o][i-1](x+1, y  ) + dog_pyr[o][i-1](x-1, y  ) ) / 4.0f;
        Expr dys = ( dog_pyr[o][i+1](x,   y+1) - dog_pyr[o][i+1](x,   y-1) - dog_pyr[o][i-1](x,   y+1) + dog_pyr[o][i-1](x,   y-1) ) / 4.0f;

        #define HESSIAN_XX 0
        #define HESSIAN_YY 1
        #define HESSIAN_SS 2
        #define HESSIAN_XY 3
        #define HESSIAN_XS 4
        #define HESSIAN_YS 5

        DECL_FUNC(hessian);
        hessian(c,x,y) = select(c == HESSIAN_XX, dxx,
                         select(c == HESSIAN_YY, dyy,
                         select(c == HESSIAN_SS, dss,
                         select(c == HESSIAN_XY, dxy,
                         select(c == HESSIAN_XS, dxs,
                                                 dys)))));

        DECL_FUNC(pc_det);
        pc_det(x,y) = hessian(HESSIAN_XX,x,y) * hessian(HESSIAN_YY,x,y) - 2.0f * hessian(HESSIAN_XY,x,y);

        DECL_FUNC(pc_tr);
        pc_tr(x,y)  = hessian(HESSIAN_XX,x,y) + hessian(HESSIAN_YY,x,y);

        DECL_FUNC(invdet);
        invdet(x,y) = 1.0f/(  ( hessian(HESSIAN_XX,x,y) * (hessian(HESSIAN_YY,x,y) * hessian(HESSIAN_SS,x,y) - hessian(HESSIAN_YS,x,y) * hessian(HESSIAN_YS,x,y)) )
                            - ( hessian(HESSIAN_XY,x,y) * (hessian(HESSIAN_XY,x,y) * hessian(HESSIAN_SS,x,y) - hessian(HESSIAN_YS,x,y) * hessian(HESSIAN_XS,x,y)) )
                            + ( hessian(HESSIAN_XS,x,y) * (hessian(HESSIAN_XY,x,y) * hessian(HESSIAN_YS,x,y) - hessian(HESSIAN_YY,x,y) * hessian(HESSIAN_XS,x,y)) ));

        DECL_FUNC(inv);
        inv(c,x,y) = select(c == HESSIAN_XX, (invdet(x,y)) * (hessian(HESSIAN_YY,x,y) * hessian(HESSIAN_SS,x,y) - hessian(HESSIAN_YS,x,y) * hessian(HESSIAN_YS,x,y)),
                     select(c == HESSIAN_YY, (invdet(x,y)) * (hessian(HESSIAN_XX,x,y) * hessian(HESSIAN_SS,x,y) - hessian(HESSIAN_XS,x,y) * hessian(HESSIAN_XS,x,y)),
                     select(c == HESSIAN_SS, (invdet(x,y)) * (hessian(HESSIAN_XX,x,y) * hessian(HESSIAN_YY,x,y) - hessian(HESSIAN_XY,x,y) * hessian(HESSIAN_XY,x,y)),
                     select(c == HESSIAN_XY, (invdet(x,y)) * (hessian(HESSIAN_XS,x,y) * hessian(HESSIAN_YS,x,y) - hessian(HESSIAN_XY,x,y) * hessian(HESSIAN_SS,x,y)),
                     select(c == HESSIAN_XS, (invdet(x,y)) * (hessian(HESSIAN_XY,x,y) * hessian(HESSIAN_YS,x,y) - hessian(HESSIAN_XS,x,y) * hessian(HESSIAN_YY,x,y)),
                                             (invdet(x,y)) * (hessian(HESSIAN_XY,x,y) * hessian(HESSIAN_XS,x,y) - hessian(HESSIAN_XX,x,y) * hessian(HESSIAN_YS,x,y)))))));

        DECL_FUNC(interp);
        interp(c,x,y) = select(c == 0, inv(HESSIAN_XX, x,y) * dx(x,y) + inv(HESSIAN_XY, x,y) * dy(x,y) + inv(HESSIAN_XS, x,y) * ds(x,y),
                        select(c == 1, inv(HESSIAN_XY, x,y) * dx(x,y) + inv(HESSIAN_YY, x,y) * dy(x,y) + inv(HESSIAN_YS, x,y) * ds(x,y),
                                       inv(HESSIAN_XS, x,y) * dx(x,y) + inv(HESSIAN_YS, x,y) * dy(x,y) + inv(HESSIAN_SS, x,y) * ds(x,y)));

        DECL_FUNC(interp_contr);
        interp_contr(x,y) = interp(0,x,y) * dx(x,y) + interp(1,x,y) * dy(x,y) + interp(2,x,y) * ds(x,y);

        Expr is_valid = is_extremum(x,y) &&
                        pc_det(x,y) > 0.0f &&
                        (pc_tr(x,y) * pc_tr(x,y) / pc_det(x,y) < ( curv_thr + 1.0f )*( curv_thr + 1.0f ) / curv_thr) &&
                        abs(interp_contr(x,y)) > contr_thr / intervals &&
                        dx(x,y) < 1.0f &&
                        dy(x,y) < 1.0f &&
                        ds(x,y) < 1.0f;

        key[o][i-1](x,y) = is_valid;

        if (use_gpu())
          {
            dog_min.root().cudaTile(x,y,16,16);
            dog_max.root().cudaTile(x,y,16,16);
            dog_min.update().root().reorder(r.x,r.y,x,y).cudaTile(x,y,16,16);
            dog_max.update().root().reorder(r.x,r.y,x,y).cudaTile(x,y,16,16);

            // is_extremum.cudaTile(x,y,16,16); /* for some reason, root() doesn't work */

            // dx.root().cudaTile(x,y,16,16);
            // dy.root().cudaTile(x,y,16,16);
            // ds.root().cudaTile(x,y,16,16);

            hessian.root().unroll(c,6).cudaTile(x,y,16,16);
            //pc_det.root().cudaTile(x,y,16,16);
            //pc_tr.root().cudaTile(x,y,16,16);
            invdet.root().cudaTile(x,y,16,16);
            inv.root().unroll(c,6).cudaTile(x,y,16,16);
            //interp.root().unroll(c,3).cudaTile(x,y,16,16);
            interp_contr.root().cudaTile(x,y,16,16);
          }
        else
          {
            dog_min.root().parallel(y).vectorize(x, 4);
            dog_max.root().parallel(y).vectorize(x, 4);

            hessian.root().parallel(y).vectorize(x, 4);
            pc_det.root().parallel(y).vectorize(x, 4);
            pc_tr.root().parallel(y).vectorize(x, 4);
            invdet.root().parallel(y).vectorize(x, 4);
            inv.root().parallel(y).vectorize(x, 4);
            interp.root().parallel(y).vectorize(x, 4);
            interp_contr.root().parallel(y).vectorize(x, 4);
          }
      }

  if (use_gpu())
    {
      gray.cudaTile(x,y,16,16);

      for(int o = 0; o < octaves; o++ )
        for(int i = 0; i < intervals + 2; i++ )
          dog_pyr[o][i].cudaTile(x,y,16,16);

      for(int o = 0; o < octaves; o++ )
        for(int i = 1; i <= intervals; i++ )
          key[o][i-1].cudaTile(x,y,16,16);
    }
  else
    {
      gray.root().parallel(y).vectorize(x, 4);

      for(int o = 0; o < octaves; o++ )
        for(int i = 0; i < intervals + 2; i++ )
          dog_pyr[o][i].root().parallel(y).vectorize(x, 4);

      //for(int o = 0; o < octaves; o++ )
      //  for(int i = 1; i <= intervals; i++ )
      //    key[o][i-1].root(); -- segfault... god knows why
    }

   {
     Expr expr_comb = false;

     for(int o = 0; o < octaves; o++ )
       for(int i = 0; i < intervals; i++ )
         {
           expr_comb = select(x % (1 << o) == 0 && y % (1 << o) == 0,
                           (key[o][i](x / (1 << o), y / (1 << o)) || expr_comb),
                           expr_comb);
         }

     Func comb_ext;
     comb_ext(x,y) = 255 * cast<uint8_t>(expr_comb);

     if (use_gpu())
       comb_ext.root().cudaTile(x,y,16,16);
     else
       comb_ext.root().parallel(y).vectorize(x, 4);

     comb_ext.compileJIT();

     for(int k=0; k<NTRIES; k++)
       {
         Image<uint8_t> out (input.width(), input.height(), 1);

         TIME_START(sift_time)

         input.markHostDirty(); /* copy CPU -> GPU */

         comb_ext.realize(out);

         out.copyToHost();      /* copy GPU -> CPU */

         TIME_END(sift_time, k)

         if (k==0) save(out, "sift_halide_out.png");
       }
   }

  printf("[Halide]\n");

  printf("- SIFT:       %lf \n", sift_time.min);

  return 0;
}
