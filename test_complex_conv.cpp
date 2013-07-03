#include <Halide.h>

using namespace Halide;

int main(int argc, char **argv) {

    int W = 1024, H = 1024;

    int k = 0;
    Image<float> in(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = ((float) rand() / (RAND_MAX)) + 1;
        }
    }

    Image<int32_t> radius(256);

    for (int i = 0; i < 256; i++) {
      radius(i) = i % 8;
    }

    Var v("v"), x("x"), y("y");

    Expr rexp = clamp(radius(v), 0, 7);
    RDom rr(-rexp, rexp+1, -rexp, rexp+1);

    Func conv("conv");

    conv(x, y, v) += in(clamp(x+rr.x, 0, W-1), clamp(y+rr.y, 0, H-1));

    Image<float> out(W, H, 256);

    conv.realize(out);

    printf("Success!\n");

    return 0;

}
