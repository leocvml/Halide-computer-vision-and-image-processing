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
    cv::Mat image = cv::imread("Img/Lena.jpg");
    Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved(image.data,image.cols, image.rows, image.channels());
    
    uint16_t R2GRAY = 77, G2GRAY = 150, B2GRAY = 29;
    Halide::Func Gray;
    Halide::Var x,y;
    Halide::Expr r = Halide::cast<uint16_t>(input(x, y, 0));
    Halide::Expr g = Halide::cast<uint16_t>(input(x, y, 1));
    Halide::Expr b = Halide::cast<uint16_t>(input(x, y, 2));
    Gray(x, y) = Halide::cast<uint8_t>((r * R2GRAY +
                                        g * G2GRAY + 
                                        b * B2GRAY) >> 8 );
 
    Halide::Func histogram;
    histogram(x) = Halide::cast<float>(0);
    Halide::RDom iter(0, input.width(), 0, input.height());
    histogram(Gray(iter.x,iter.y)) += 1;

    histogram(x) *= Halide::cast<float>(255) / (input.width() * input.height());
    Halide::RDom iter1256(1,256);
    histogram(iter1256) = histogram(iter1256) + histogram(iter1256-1);
 
    Halide::Func HistogramEqualize;
    HistogramEqualize(x,y) = Halide::cast<uint8_t>(histogram(Gray(x,y)));

    //Tell Halide to evaluate all of histogram before any of LUT.
    histogram.compute_root();
    Halide::Buffer<uint8_t> result = HistogramEqualize.realize(input.width(), input.height());
    Halide::Buffer<uint8_t> GrayScale = Gray.realize(input.width(), input.height());
    

    Mat grayimg;
    HalideBuffer2Mat(GrayScale,grayimg,false);
    imshow("Ori gray", grayimg);

    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("Hist Equal", Result);

   
    waitKey(0);
  
    return 0;
}
