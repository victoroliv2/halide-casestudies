const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP |
                          CLK_FILTER_NEAREST;

__kernel void gray(read_only  image2d_t in,
                   write_only image2d_t out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v  = read_imagef(in, sampler, (int2)(x, y));
  float g = 0.299f * v.x + 0.587f * v.y + 0.114f * v.z;

  write_imagef(out, (int2)(x, y), (float4)(g));
}


__kernel void downx(read_only  image2d_t in,
                    write_only image2d_t out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v  =          read_imagef(in, sampler, (int2)(2*x-1, y))
              + 3.0f * read_imagef(in, sampler, (int2)(2*x  , y))
              +        read_imagef(in, sampler, (int2)(2*x+1, y))
              +        read_imagef(in, sampler, (int2)(2*x+2, y));

  write_imagef(out, (int2)(x, y), v / 8.0f);
}

__kernel void downy(read_only  image2d_t in,
                    write_only image2d_t out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v  =          read_imagef(in, sampler, (int2)(x, 2*y-1))
              + 3.0f * read_imagef(in, sampler, (int2)(x, 2*y))
              +        read_imagef(in, sampler, (int2)(x, 2*y+1))
              +        read_imagef(in, sampler, (int2)(x, 2*y+2));

  write_imagef(out, (int2)(x, y), v / 8.0f);
}

kernel void blurx(read_only  image2d_t in,
                  write_only image2d_t out,
                  constant float * gmask,
                  int radius)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v = 0.0f;

  for(int r=-radius; r <= radius; r++)
    {
      v += gmask[r+radius] * read_imagef(in, sampler, (int2)(x+r, y));
    }

  write_imagef(out, (int2)(x, y), v);
}

kernel void blury(read_only  image2d_t in,
                  write_only image2d_t out,
                  constant float * gmask,
                  int radius)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v = 0.0f;

  for(int r=-radius; r <= radius; r++)
    {
      v += gmask[r+radius] * read_imagef(in, sampler, (int2)(x, y+r));
    }

  write_imagef(out, (int2)(x, y), v);
}

kernel void dog(read_only  image2d_t g1,
                read_only  image2d_t g2,
                write_only image2d_t out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float4 v  =   read_imagef(g2, sampler, (int2)(x, y))
              - read_imagef(g1, sampler, (int2)(x, y));

  write_imagef(out, (int2)(x, y), v);
}

kernel void isvalid(read_only  image2d_t dog_p,
                    read_only  image2d_t dog_c,
                    read_only  image2d_t dog_n,
                    global uchar * isvalid,
                    float curv_thr,
                    float contr_thr,
                    int o,
                    int intervals)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  float v[3][3][3];

  for (int ry=-1; ry<=1; ry++)
    for (int rx=-1; rx<=1; rx++)
      {
        v[0][ry+1][rx+1] = read_imagef(dog_p, sampler, (int2)(x+rx, y+ry)).x;
        v[1][ry+1][rx+1] = read_imagef(dog_c, sampler, (int2)(x+rx, y+ry)).x;
        v[2][ry+1][rx+1] = read_imagef(dog_n, sampler, (int2)(x+rx, y+ry)).x;
      }

  const float vcc = v[1][1][1];

  float dmax = -FLT_MAX;
  float dmin =  FLT_MAX;

  for (int ry=0; ry<=2; ry++)
    for (int rx=0; rx<=2; rx++)
      {
        dmax = fmax(dmax,
               fmax(v[0][ry][rx],
               fmax(v[1][ry][rx],
                    v[2][ry][rx])));

        dmin = fmin(dmin,
               fmin(v[0][ry][rx],
               fmin(v[1][ry][rx],
                    v[2][ry][rx])));
      }

  float prelim_contr_thr = 0.5f * contr_thr / intervals;

  bool is_extremum = ((fabs(vcc) > prelim_contr_thr) &&
                      ((vcc <= 0.0f && vcc == dmin) ||
                       (vcc >  0.0f && vcc == dmax)));

  const float dxx = v[1][1][2] + v[1][1][0] - 2.0f * vcc;
  const float dyy = v[1][2][1] + v[1][0][1] - 2.0f * vcc;
  const float dss = v[2][1][1] + v[0][1][1] - 2.0f * vcc;
  const float dxy = ( v[1][2][2] - v[1][2][0] - v[1][0][2] + v[1][0][0] ) / 4.0f;
  const float dxs = ( v[2][1][2] - v[2][1][0] - v[0][1][2] + v[0][1][0] ) / 4.0f;
  const float dys = ( v[2][2][1] - v[2][0][1] - v[0][2][1] + v[0][0][1] ) / 4.0f;

  float pc_det = dxx * dyy - 2.0f * dxy;
  float pc_tr = dxx + dyy;

  float invdet = 1.0f/(  ( dxx * (dyy * dss - dys * dys) )
                       - ( dxy * (dxy * dss - dys * dxs) )
                       + ( dxs * (dxy * dys - dyy * dxs) ));

  const float inv_dxx = invdet * (dyy * dss - dys * dys);
  const float inv_dyy = invdet * (dxx * dss - dxs * dxs);
  const float inv_dss = invdet * (dxx * dyy - dxy * dxy);
  const float inv_dxy = invdet * (dxs * dys - dxy * dss);
  const float inv_dxs = invdet * (dxy * dys - dxs * dyy);
  const float inv_dys = invdet * (dxy * dxs - dxx * dys);

  const float dx = (v[1][1][2] - v[1][1][0]) / 2.0f;
  const float dy = (v[1][2][1] - v[1][0][1]) / 2.0f;
  const float ds = (v[2][1][1] - v[0][1][1]) / 2.0f;

  const float interp_x = inv_dxx * dx + inv_dxy * dy + inv_dxs * ds;
  const float interp_y = inv_dxy * dx + inv_dyy * dy + inv_dys * ds;
  const float interp_s = inv_dxs * dx + inv_dys * dy + inv_dss * ds;

  const float interp_contr = interp_x * dx + interp_y * dy + interp_s * ds;

  bool ok = is_extremum &&
            pc_det > 0.0f;
            (pc_tr * pc_tr / pc_det < ( curv_thr + 1.0f )*( curv_thr + 1.0f ) / curv_thr) &&
            fabs(interp_contr) > contr_thr / intervals &&
            dx < 1.0f &&
            dy < 1.0f &&
            ds < 1.0f;

  int off = y * get_global_size(0) * (1 << o) + x * (1 << o);

  if (ok) isvalid[off] = 1;
}
