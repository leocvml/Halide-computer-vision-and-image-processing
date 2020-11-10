/*
this sample show halide how to co-work with opencv 3 channels

*/
#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace Halide::Tools;
using namespace cv;

void HalideBuffer2Mat(const Halide::Buffer<uint8_t> &src, cv::Mat &dest)
{
    if (dest.empty())
        dest.create(cv::Size(src.width(), src.height()), CV_MAKETYPE(CV_8U, src.channels()));

    const int ch = dest.channels();
    switch (ch)
    {
    case 1:
        for (int j = 0; j < dest.rows; j++)
        {   
            uchar* ptr = dest.ptr<uchar>(j);            
            for (int i = 0; i < dest.cols; i++)
            {
                // dest.at<uchar>(j, i) = src(i, j);
                ptr[i] = src(i, j);

            }
        }
        break;
    case 3:
        for (int j = 0; j < dest.rows; j++)
        {   
            uchar* ptr = dest.ptr<uchar>(j);
            for (int i = 0; i < dest.cols; i++)
            {
                ptr[i * 3 + 0 ] = src(i, j, 2);
                ptr[i * 3 + 1 ] = src(i, j, 1);
                ptr[i * 3 + 2 ] = src(i, j, 0);
            }
        }
        break;
    }
}

int main(int argc, char **argv) {

    //Load image
    Halide::Buffer<uint8_t> input = load_image("Img/Lena.jpg");
   
    Halide::Func ORI;
    Halide::Var x, y, c;
    Halide::Expr value = input(x, y, c);
    value = Halide::cast<float>(value);
    value = value * 1.5f;
    value = Halide::min(value, 255.0f); 
    value = Halide::cast<uint8_t>(value);
    ORI(x, y, c) = value;
  
    Halide::Buffer<uint8_t> buf =  
        ORI.realize(input.width(), input.height(), input.channels());
    

    Mat BrighterMat;
    HalideBuffer2Mat(buf,BrighterMat);
     

    imshow("Brighter", BrighterMat);
    waitKey(0);
  
    return 0;
}
