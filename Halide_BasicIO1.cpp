/*
this sample show how to use Halide Buffer Load/Save Image
*/
#include "Halide.h"
// Include some support code for loading pngs.
#include "halide_image_io.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace Halide::Tools;
int main(int argc, char **argv) {
    //Load image
    Halide::Buffer<uint8_t> input = load_image("Img/Lena.jpg");
     
    Halide::Func brighter;
    Halide::Var x, y, c;
    Halide::Expr value = input(x, y, c);
    value = Halide::cast<float>(value);
    value = value * 1.5f;
    value = Halide::min(value, 255.0f); 
    value = Halide::cast<uint8_t>(value);
    brighter(x, y, c) = value;
 
    Halide::Buffer<uint8_t> output =
        brighter.realize(input.width(), input.height(), input.channels());
    save_image(output, "brighter.png");
    printf("Success!\n");
    return 0;
  
 
}
