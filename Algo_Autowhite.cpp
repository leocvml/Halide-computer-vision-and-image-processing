/*
in this sample we will implement binary segmentation using threshold 
*/
#include "Halide.h"
#include "halide_image_io.h"
#include <iostream>
// #include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace Halide::Tools;
using namespace cv;



void HalideBuffer2Mat(const Halide::Buffer<uint8_t> &src, cv::Mat &dest, bool RGBLayout=true)
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
                /*set RGBLayout condition*/
                if(RGBLayout){
                    ptr[i * 3 + 0 ] = src(i, j, 2);
                    ptr[i * 3 + 1 ] = src(i, j, 1);
                    ptr[i * 3 + 2 ] = src(i, j, 0);
                }
                if(!RGBLayout){
                    ptr[i * 3 + 0 ] = src(i, j, 0);
                    ptr[i * 3 + 1 ] = src(i, j, 1);
                    ptr[i * 3 + 2 ] = src(i, j, 2);
                }
            }
        }
        break;
    }
}

int main(int argc, char **argv) {

    //Load image
    cv::Mat image = cv::imread("Img/AWB_before1.jpg");
    Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved(image.data,image.cols, image.rows, image.channels());
    
    
    Halide::Func MeanRGB;
    Halide::Var c;
    Halide::RDom r(0,input.width(), 0, input.height());
    MeanRGB(c) = Halide::cast<int>(0);
    MeanRGB(c) += (input(r.x,r.y,c));
    MeanRGB(c) /=  (input.width() * input.height());
    
    Halide::Func GetkRGB;
    GetkRGB(c) = Halide::cast<float>(0);
    Halide::RDom rc(0,3);
    GetkRGB(c) += (MeanRGB(rc));
    GetkRGB(c) /= 3 * MeanRGB(c);

    Halide::Func AWB;
    Halide::Var x,y;
    Halide::Expr value = input(x, y, c);
    value = Halide::cast<float>(value);
    value = value * GetkRGB(c);
    value = Halide::min(value, 255.0f); 
    value = Halide::cast<uint8_t>(value);
    AWB(x,y,c) = value;

 
 
 
    MeanRGB.compute_root();
    Halide::Buffer<uint8_t> result = AWB.realize(input.width(), input.height(), input.channels());

   

    imshow("before AWB" , image);
    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("after AWB", Result);

   
    waitKey(0);
  
    return 0;
}
