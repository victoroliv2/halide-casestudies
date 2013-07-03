#include "Halide.h"

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <cfloat>
#include <sys/time.h>

Var x, y, c;

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

// Downsample with a 1 3 3 1 filter
Func downsample(Func f) {
    DECL_FUNC(downx)
    DECL_FUNC(downy)

    downx(x, y) = (f(2*x-1, y) + 3.0f * (f(2*x, y) + f(2*x+1, y)) + f(2*x+2, y)) / 8.0f;
    downy(x, y) = (downx(x, 2*y-1) + 3.0f * (downx(x, 2*y) + downx(x, 2*y+1)) + downx(x, 2*y+2)) / 8.0f;

    if (use_gpu())
      {
        downx.root().cudaTile(x,y,16,16);
        downy.root().cudaTile(x,y,16,16);
      }
    else
      {
        downx.root();
        downy.root();
      }

    return downy;
}

Func blur(Func image, const float sigma) {
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
  blurx(x, y) += image(x+i, y) * normalized(i);
  blury(x, y) += blurx(x, y+i) * normalized(i);

  if (use_gpu())
    {
      normalized.root();
      blurx.root().cudaTile(x,y,16,16);
      blury.root().cudaTile(x,y,16,16);
      blurx.update().root().reorder(i,x,y).cudaTile(x,y,16,16);
      blury.update().root().reorder(i,x,y).cudaTile(x,y,16,16);
    }
  else
    {
      normalized.root();
      blurx.root();
      blury.root();
    }

  return blury;
}

struct
{
  float dx, dy, ds;
}
Keypoint;

int main(int argc, char * argv[])
{
  const float sigma     = 1.6f;
  const float contr_thr = 0.04f;
  const int   curv_thr  = 10;

  const int octaves   = 4;
  const int intervals = 2;

  assert(argc == 2);

  Image<float> input = load<float>(argv[1]);

  printf("(%d %d %d)\n", input.width(), input.height(), input.channels());

  DECL_FUNC(gray);
  gray(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

  DECL_FUNC(clamped);
  clamped(x, y) = gray(clamp(x, 0, input.width()  - 1),
                       clamp(y, 0, input.height() - 1));

  /* gaussian pyramid */

  Func gauss_pyr[octaves][intervals+3];

  /* precompute gaussian sigmas */

  float sig[intervals+3], sig_prev, sig_total;

  sig[0] = sigma;
  float k = pow( 2.0, 1.0 / intervals );
  for(int i = 1; i < intervals + 3; i++ )
    {
      sig_prev = powf( k, i - 1 ) * sigma;
      sig_total = sig_prev * k;
      sig[i] = sqrtf( sig_total * sig_total - sig_prev * sig_prev );
    }

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 3; i++ )
      {
        if( o == 0  &&  i == 0 )
          gauss_pyr[o][i] = clamped;

        /* base of new octvave is halved image from end of previous octave */
        else if( i == 0 )
          gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intervals] );

        /* blur the current octave's last image to create the next one */
        else
          gauss_pyr[o][i] = blur(gauss_pyr[o][i-1], sig[i]);
      }

  /* difference-of-gaussians pyramid */

  Func dog_pyr[octaves][intervals+2];

  for(int o = 0; o < octaves; o++ )
    for(int i = 0; i < intervals + 2; i++ ) {
      DECL_FUNC(dog_pyr__)
      dog_pyr[o][i] = dog_pyr__;
      dog_pyr[o][i](x,y) = gauss_pyr[o][i+1](x,y) - gauss_pyr[o][i](x,y);
    }

  Func is_extremum[octaves][intervals];

  float prelim_contr_thr = 0.5f * contr_thr / intervals;

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      {
        Expr v = dog_pyr[o][i](x,y);

        RDom r(-1,2,-1,2);

        Expr dmax = max(dog_pyr[o][i+1](x+r.x,y+r.y),
                    max(dog_pyr[o][i  ](x+r.x,y+r.y),
                        dog_pyr[o][i-1](x+r.x,y+r.y)));

        // Expr dmax = v;
        // for (int yy=-1; yy<=1; yy++)
        //   for (int xx=-1; xx<=1; xx++)
        //   {
        //     dmax = max(dmax,
        //            max(dog_pyr[o][i+1](x+xx,y+yy),
        //            max(dog_pyr[o][i  ](x+xx,y+yy),
        //                dog_pyr[o][i-1](x+xx,y+yy))));
        //   }

        DECL_FUNC(dog_max);
        dog_max(x,y) = maximum(dmax); /* can't set schedule */
        dog_max(x,y) = -FLT_MAX;
        dog_max(x,y) = max(dog_max(x,y), dmax);

        dog_max(x,y) = dmax;

        Expr dmin = min(dog_pyr[o][i+1](x+r.x,y+r.y),
                    min(dog_pyr[o][i  ](x+r.x,y+r.y),
                        dog_pyr[o][i-1](x+r.x,y+r.y)));

        // Expr dmin = v;
        // for (int yy=-1; yy<=1; yy++)
        //   for (int xx=-1; xx<=1; xx++)
        //   {
        //     dmin = min(dmin,
        //            min(dog_pyr[o][i+1](x+xx,y+yy),
        //            min(dog_pyr[o][i  ](x+xx,y+yy),
        //                dog_pyr[o][i-1](x+xx,y+yy))));
        //   }

        DECL_FUNC(dog_min);
        dog_min(x,y) = minimum(dmin);
        dog_min(x,y) = FLT_MAX;
        dog_min(x,y) = min(dog_min(x,y), dmin);

        dog_min(x,y) = dmin;

        if (use_gpu())
          {
            dog_min.root().cudaTile(x,y,16,16);
            dog_max.root().cudaTile(x,y,16,16);
            dog_min.update().root().reorder(r.x,r.y,x,y).cudaTile(x,y,16,16);
            dog_max.update().root().reorder(r.x,r.y,x,y).cudaTile(x,y,16,16);

            //dog_min.root();
            //dog_max.root();
          }
        else
          {
            dog_min.root();
            dog_min.update().root();
            dog_max.root();
            dog_max.update().root();
          }

        DECL_FUNC(is_extremum__)
        is_extremum[o][i-1] = is_extremum__;
        is_extremum[o][i-1](x,y) = ((abs(v) > prelim_contr_thr) &&
                                    ((v <= 0.0f && v == dog_min(x,y)) ||
                                     (v >  0.0f && v == dog_max(x,y))));

      }

  Func key[octaves][intervals];

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      {
        // dD = deriv_3D( dog_pyr, octv, intvl, r, c );
        // H = hessian_3D( dog_pyr, octv, intvl, r, c );
        // H_inv = cvCreateMat( 3, 3, CV_64FC1 );
        // cvInvert( H, H_inv, CV_SVD );
        // cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 );

        Expr dx = (dog_pyr[o][i]  (x+1, y  ) - dog_pyr[o][i]  (x-1, y  )) / 2.0f;
        Expr dy = (dog_pyr[o][i]  (x  , y+1) - dog_pyr[o][i]  (x,   y-1)) / 2.0f;
        Expr ds = (dog_pyr[o][i+1](x  , y  ) - dog_pyr[o][i-1](x,   y  )) / 2.0f;

        DECL_FUNC(deriv);
        deriv(x,y,c) = select(c == 0, dx,
                       select(c == 1, dy,
                                      ds));

        Expr v = dog_pyr[o][i](x,y);

        Expr dxx = dog_pyr[o][i  ](x+1, y  ) + dog_pyr[o][i  ](x-1, y  ) - 2 * v;
        Expr dyy = dog_pyr[o][i  ](x,   y+1) + dog_pyr[o][i  ](x,   y-1) - 2 * v;
        Expr dss = dog_pyr[o][i+1](x,   y  ) + dog_pyr[o][i-1](x,   y  ) - 2 * v;

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
        hessian(x,y,c) = select(c == HESSIAN_XX, dxx,
                         select(c == HESSIAN_YY, dyy,
                         select(c == HESSIAN_SS, dss,
                         select(c == HESSIAN_XY, dxy,
                         select(c == HESSIAN_XS, dxs,
                                                 dys)))));

        DECL_FUNC(det);
        det(x,y) =   ( hessian(x,y, HESSIAN_XX) * (hessian(x,y, HESSIAN_YY) * hessian(x,y, HESSIAN_SS) - hessian(x,y, HESSIAN_YS) * hessian(x,y, HESSIAN_YS)) )
                   - ( hessian(x,y, HESSIAN_XY) * (hessian(x,y, HESSIAN_XY) * hessian(x,y, HESSIAN_SS) - hessian(x,y, HESSIAN_YS) * hessian(x,y, HESSIAN_XS)) )
                   + ( hessian(x,y, HESSIAN_XS) * (hessian(x,y, HESSIAN_XY) * hessian(x,y, HESSIAN_YS) - hessian(x,y, HESSIAN_YY) * hessian(x,y, HESSIAN_XS)) );

        // if a symmetric matrix is invertible then its inverse is symmetric also

        Expr invdet = 1.0f / det(x,y);

        Expr a[6];

        a[HESSIAN_XX] = (invdet) * (hessian(x,y,HESSIAN_YY) * hessian(x,y,HESSIAN_SS) - hessian(x,y,HESSIAN_YS) * hessian(x,y,HESSIAN_YS));
        a[HESSIAN_YY] = (invdet) * (hessian(x,y,HESSIAN_XX) * hessian(x,y,HESSIAN_SS) - hessian(x,y,HESSIAN_XS) * hessian(x,y,HESSIAN_XS));
        a[HESSIAN_SS] = (invdet) * (hessian(x,y,HESSIAN_XX) * hessian(x,y,HESSIAN_YY) - hessian(x,y,HESSIAN_XY) * hessian(x,y,HESSIAN_XY));
        a[HESSIAN_XY] = (invdet) * (hessian(x,y,HESSIAN_XS) * hessian(x,y,HESSIAN_YS) - hessian(x,y,HESSIAN_XY) * hessian(x,y,HESSIAN_SS));
        a[HESSIAN_XS] = (invdet) * (hessian(x,y,HESSIAN_XY) * hessian(x,y,HESSIAN_YS) - hessian(x,y,HESSIAN_XS) * hessian(x,y,HESSIAN_YY));
        a[HESSIAN_YS] = (invdet) * (hessian(x,y,HESSIAN_XY) * hessian(x,y,HESSIAN_XS) - hessian(x,y,HESSIAN_XX) * hessian(x,y,HESSIAN_YS));

        DECL_FUNC(inv);
        inv(x,y,c) = select(c == HESSIAN_XX, a[HESSIAN_XX],
                     select(c == HESSIAN_YY, a[HESSIAN_YY],
                     select(c == HESSIAN_SS, a[HESSIAN_SS],
                     select(c == HESSIAN_XY, a[HESSIAN_XY],
                     select(c == HESSIAN_XS, a[HESSIAN_XS],
                                             a[HESSIAN_YS])))));

        // matrix product

        Expr f[3];

        f[0] = inv(x,y, HESSIAN_XX) * deriv(x,y,0) + inv(x,y, HESSIAN_XY) * deriv(x,y,1) + inv(x,y, HESSIAN_XS) * deriv(x,y,2);
        f[1] = inv(x,y, HESSIAN_XY) * deriv(x,y,0) + inv(x,y, HESSIAN_YY) * deriv(x,y,1) + inv(x,y, HESSIAN_YS) * deriv(x,y,2);
        f[2] = inv(x,y, HESSIAN_XS) * deriv(x,y,0) + inv(x,y, HESSIAN_YS) * deriv(x,y,1) + inv(x,y, HESSIAN_SS) * deriv(x,y,2);

        DECL_FUNC(interp);
        interp(x,y,c) = select(c == 0, f[0],
                        select(c == 1, f[1],
                                       f[2]));

        DECL_FUNC(interp_contr);
        interp_contr(x,y) = interp(x,y,0) * deriv(x,y,0) + interp(x,y,1) * deriv(x,y,1) + interp(x,y,2) * deriv(x,y,2);


        Expr valid_keypoint = is_extremum[o][i-1](x,y) &&
                              abs(interp_contr(x,y)) > contr_thr / intervals &&
                              deriv(x,y,0) < 1.0f &&
                              deriv(x,y,1) < 1.0f &&
                              deriv(x,y,2) < 1.0f;

        if (use_gpu())
          {
            deriv.root().cudaTile(x,y,16,16);
            hessian.root().cudaTile(x,y,16,16);
            det.root().cudaTile(x,y,16,16);
            inv.root().cudaTile(x,y,16,16);
            interp.root().cudaTile(x,y,16,16);
            interp_contr.root().cudaTile(x,y,16,16);
          }
        else
          {
            deriv.root();
            hessian.root();
            det.root();
            inv.root();
            interp.root();
            interp_contr.root();
          }

        // key-points

        DECL_FUNC(key__)
        key[o][i-1] = key__;
        key[o][i-1](x,y,c) = select(c == 0, select(valid_keypoint, +1.0f, -1.0f),
                             select(c == 1, deriv(x,y,0),
                             select(c == 2, deriv(x,y,1),
                                            deriv(x,y,2))));
      }

  /* here it ends the GPU part */

  /* schedules */

  if (use_gpu())
    {
      gray.cudaTile(x,y,16,16);

      for(int o = 0; o < octaves; o++ )
        for(int i = 0; i < intervals + 2; i++ )
          dog_pyr[o][i].cudaTile(x,y,16,16);

      for(int o = 0; o < octaves; o++ )
        for(int i = 1; i <= intervals; i++ )
          {
            is_extremum[o][i-1].cudaTile(x,y,16,16);
            key[o][i-1].cudaTile(x,y,16,16);
          }
    }
  else
    {
      gray.root();

      for(int o = 0; o < octaves; o++ )
        for(int i = 0; i < intervals + 2; i++ )
          dog_pyr[o][i].root();

      for(int o = 0; o < octaves; o++ )
        for(int i = 0; i < intervals; i++ )
          is_extremum[o][i].root();
    }

  /* do it! */

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      key[o][i-1].compileJIT();

  // Func mark_red[octaves][intervals];
  // {
  //   for(int o = 0; o < octaves; o++ )
  //     for(int i = 1; i <= intervals; i++ )
  //       {
  //           DECL_FUNC(mark_red__)
  //           mark_red[o][i-1] = mark_red__;
  //           mark_red[o][i-1](x,y,c) = select(c == 0, select(key[o][i-1](x,y,0) > 0.0f, 1.0f, gauss_pyr[o][i](x,y)),
  //                                     select(c == 1, select(key[o][i-1](x,y,0) > 0.0f, 0.0f, gauss_pyr[o][i](x,y)),
  //                                     select(c == 2, select(key[o][i-1](x,y,0) > 0.0f, 0.0f, gauss_pyr[o][i](x,y)),
  //                                                    1.0f)));

  //           mark_red[o][i-1].compileJIT();
  //       }
  // }

  std::vector<struct Keypoints> features;

  for(int o = 0; o < octaves; o++ )
    for(int i = 1; i <= intervals; i++ )
      {
        int w = input.width()  / (1 << o);
        int h = input.height() / (1 << o);

        Image<float> out (w, h, 4);

        key[o][i].realize(out);

        float octv_pow = powf(2.0f, o);

        for (int yy=0; yy<h; yy++)
          for (int xx=0; xx<w; xx++)
            {
              if (out(xx,yy,0) > 0.0f)
                {
                  Keypoint k;

                  k.octv  = o;
                  k.intvl = i;
                  k.x  = xx;
                  k.y  = yy;
                  k.ix = (float(xx)+out(xx,yy,1)) * octv_pow;
                  k.iy = (float(yy)+out(xx,yy,2)) * octv_pow;
                  k.is = out(xx,yy,3);

                  k.scl = sigma * powf(2.0f, o + (i + k.is) / intervals);
                  k.scl_octv = sigma * powf(2.0f, (i + k.is) / intervals);

                  features.push_back(k);
                }
            }
      }

  /* end of first part -- have keypoints */

  /* feature orientations */

  for (int j=0; j<features.size(); j++)
    {
      Func dx, dy;
      dx(x,y) = gauss_pyr[feat.octv][feat.intvl](x+1,y) - gauss_pyr[feat.octv][feat.intvl](x-1,y);
      dy(x,y) = gauss_pyr[feat.octv][feat.intvl](x,y-1) - gauss_pyr[feat.octv][feat.intvl](x,y+1);

      Func grad_mag, grad_ori;
      grad_mag(x,y) = sqrt( dx(x,y)*dx(x,y) + dy(x,y)*dy(x,y) );
      grad_ori(x,y) = atan2( dy(x,y), dx(x,y) );

      float exp_denom = 2.0f * (SIFT_ORI_SIG_FCTR * feat.scl_octv) * (SIFT_ORI_SIG_FCTR * feat.scl_octv);
      gaussian(x,y) = exp( -( x*x + y*y ) / exp_denom);

      int rad = round( SIFT_ORI_RADIUS * feat.scl_octv );

      RDom rhist(-rad, rad+1, -rad, rad+1);

      Func hist;
      hist(c,x,y) = 0.0f;

      Expr bin = clamp(round( SIFT_ORI_HIST_BINS * ( grad_ori(x+rhist.x, y+rhist.y) + PI ) / (PI * 2.0f) ), 0, SIFT_ORI_HIST_BINS-1);

      hist(bin,x,y) += gaussian(x+rhist.x, y+rhist.y) * grad_mag(x+rhist.x, y+rhist.y);
    }

  printf("FINISHED - elapsed time: %f sec.\n", min);

  return 0;
}
