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

    Var x, y;

    RDom r(0, 16, 0, 16);

    Func down("down");

    down(x,y) += ( in(clamp(x+r.x-8, 0, in.width()-1), clamp(y+r.y-8, 0, in.height()-1)) ) / 16.0f;

    if (use_gpu()) {
	      down.root().cudaTile(x, y, 16, 16);
	      down.update().root().reorder(r.x, r.y, x, y).cudaTile(x, y, 16, 16);
    } else {
        Var xi, yi;

        // Grab a handle to the update step of a reduction for scheduling
        // using the "update()" method.
        down.update().tile(x, y, xi, yi, 32, 32).vectorize(xi, 4).parallel(y);
    }

    Image<float> out(W/16, H/16);

    down.realize(out);

    printf("Success!\n");

    return 0;

}
