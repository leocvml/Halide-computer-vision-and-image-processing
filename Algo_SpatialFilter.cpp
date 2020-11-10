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
    
    Halide::Var x,y,c;
    Halide::Func Meanfilter;
    Halide::RDom r(-1,3,-1,3);
    Meanfilter(x,y,c) = Halide::cast<uint8_t>(Halide::sum( Halide::cast<uint16_t>(input(x+r.x, y+r.y,c))) / 9);
    /*this is halide magic!!!
        halide provide some useful function sum max min ....*/
    

    Halide::Buffer<uint8_t> result(input.width()-2, input.height()-2, 3);
    result.set_min(1, 1);   //set min  avoid encounter 0 
    Meanfilter.realize(result);

    imshow("ORI img" , image);
    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("average blur img", Result);

   
    waitKey(0);
  
    return 0;
}
