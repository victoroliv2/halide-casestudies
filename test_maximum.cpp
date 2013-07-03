#include <Halide.h>

using namespace Halide;

#include "image_io.h"

#include <iostream>
#include <limits>
#include <cfloat>
#include <sys/time.h>

Var x, y;

Func blur(Func image, const float sigma) {
  Func gaussian;

  gaussian(x) = exp(-(x/sigma)*(x/sigma)*0.5f);

  // truncate to 3 sigma and normalize
  int radius = int(3*sigma + 1.0f);
  RDom i(-radius, 2*radius+1);

  Func normalized;
  normalized(x) = gaussian(x) / sum(gaussian(i)); // Uses an inline reduction

  // Convolve the input using two reductions
  Func blurx;
  Func blury;
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

int main(int argc, char **argv) {

    Image<float> input = load<float>(argv[1]);


    Func gray;
    gray(x, y) = 0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2);

    Func clamped;
    clamped(x, y) = gray(clamp(x, 0, input.width()  - 1),
                         clamp(y, 0, input.height() - 1));

    RDom r(-1, 2, -1, 2);

    Func blur1, blur2;
    blur1 = blur(clamped, 3.0f);
    blur2 = blur(blur1,   3.0f);

    Func mmm;

    mmm(x,y) = -FLT_MAX;
    mmm(x,y) = max(mmm(x,y),
               max(blur1(x+r.x, y+r.y),
               max(blur2(x+r.x, y+r.y),
                   clamped(x+r.x, y+r.y))));

    Func key;
    key(x,y) = select(mmm(x,y) == blur1(x, y), 1.0f, 0.0f);

    if (use_gpu()) {
	mmm.root().cudaTile(x, y, 16, 16);
	mmm.update().root().reorder(r.x, r.y, x, y).cudaTile(x, y, 16, 16);
    } else {
        mmm.update().root().parallel(y);
    }

    Image<float> out(input.width(), input.height(), 1);

    key.realize(out);

    save(out, "test_max.png");

    return 0;
}
