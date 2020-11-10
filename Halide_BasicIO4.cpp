/*
this sample show halide how to co-work with opencv
in all tutorial i will use this I/O template
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
    cv::Mat image = cv::imread("Img/Lena.jpg");
    Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved(image.data,image.cols, image.rows, image.channels());
    std::cout << input.width() <<" "<< input.height() <<" "<<input.channels() <<std::endl; 
    /*
OpenCV Mat memory layout is Interleaved  Like this :
    [4x3 3channels]

    // BGRBGRBGR   
    // BGRBGRBGR
    // BGRBGRBGR
    // BGRBGRBGR

but in Halide::buffer default memory layout is Planar:
    [4x3 3channels]
        [0]     [1]     [2]
    // RRR  // GGG  // BBB
    // RRR  // GGG  // BBB
    // RRR  // GGG  // BBB
    // RRR  // GGG  // BBB

    So we use make interleaved here
*/
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
    
    Mat OriMat;
    HalideBuffer2Mat(buf,OriMat,false);
     

    imshow("gray", OriMat);
    waitKey(0);
  
    return 0;
}
