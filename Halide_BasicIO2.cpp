/*
this sample show halide how to co-work with opencv 1 channels
*/
#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace Halide::Tools;
using namespace cv;
int main(int argc, char **argv) {

    //Load image
    Halide::Buffer<uint8_t> input = load_image("Img/Lena.jpg");
   
    uint16_t R2GRAY = 77, G2GRAY = 150, B2GRAY = 29;
    Halide::Func gray;
    Halide::Var x, y;
    Halide::Expr r = Halide::cast<uint16_t>(input(x, y, 0));
    Halide::Expr g = Halide::cast<uint16_t>(input(x, y, 1));
    Halide::Expr b = Halide::cast<uint16_t>(input(x, y, 2));
    gray(x, y) = Halide::cast<uint8_t>((r * R2GRAY +
                                        g * G2GRAY + 
                                        b * B2GRAY) >> 8 );


    Mat GrayMat = Mat::zeros(input.height(),input.width(), CV_8UC1);
    Halide::Buffer<uint8_t> buf(GrayMat.data, input.width(), input.height());
    gray.realize(buf);

    imshow("gray", GrayMat);
    waitKey(0);
 

  
 
    return 0;
}
