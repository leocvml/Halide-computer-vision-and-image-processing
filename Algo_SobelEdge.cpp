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
    cv::Mat image = cv::imread("Img/AWB_before.jpg");
    Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved(image.data,image.cols, image.rows, image.channels());
    
    Halide::Var x,y,c;


    uint16_t R2GRAY = 77, G2GRAY = 150, B2GRAY = 29;
    Halide::Func gray;
    Halide::Expr r = Halide::cast<uint16_t>(input(x, y, 0));
    Halide::Expr g = Halide::cast<uint16_t>(input(x, y, 1));
    Halide::Expr b = Halide::cast<uint16_t>(input(x, y, 2));
    gray(x, y) = Halide::cast<uint8_t>((r * R2GRAY +
                                        g * G2GRAY + 
                                        b * B2GRAY) >> 8 );

 
    Halide::Func clamped;
    Halide::Expr clamped_x = clamp(x, 0, input.width() - 1);
    Halide::Expr clamped_y = clamp(y, 0, input.height() - 1);
    // Load from input at the clamped coordinates. This means that
    // no matter how we evaluated the Func 'clamped', we'll never
    // read out of bounds on the input. This is a clamp-to-edge
    // style boundary condition, and is the simplest boundary
    // condition to express in Halide.
    clamped(x, y) = gray(clamped_x, clamped_y);

    
    Halide::Func Gradient; 
    Halide::Func kernelY;
    kernelY(x,y) = 0;
    kernelY(-1,-1) = 1,kernelY(-1,0) = 2, kernelY(-1,1) = 1;
    kernelY(0,-1) = 0,kernelY(0,0) = 0, kernelY(0,1) = 0;
    kernelY(1,-1) = -1,kernelY(1,0) = -2, kernelY(1,1) = -1;
    
    Halide::Func kernelX;
    kernelX(x,y) = 0;
    kernelX(-1,-1) = 1,kernelX(-1,0) = 0, kernelX(-1,1) = -1;
    kernelX(0,-1) = 2,kernelX(0,0) = 0, kernelX(0,1) = -2;
    kernelX(1,-1) = 1,kernelX(1,0) = 0, kernelX(1,1) = -1;

    Halide::RDom r_(-1,3,-1,3);
    Gradient(x,y,c) = Halide::cast<uint8_t>(Halide::sqrt(Halide::pow(Halide::sum(clamped(x+r_.x, y+r_.y) * kernelY(r_.x, r_.y)), 2) 
                    +Halide::pow(Halide::sum(clamped(x+r_.x, y+r_.y) * kernelX(r_.x, r_.y)), 2)));

    Halide::Func Sobel;
    Sobel(x,y,c) = Halide::cast<uint8_t>(select(Gradient(x,y,c) > 150,255,0));

    

    Halide::Buffer<uint8_t> result(input.width(), input.height(),3);
    Sobel.realize(result);

    imshow("ORI img" , image);
    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("Sobel img", Result);

   
    waitKey(0);
  
    return 0;
}
