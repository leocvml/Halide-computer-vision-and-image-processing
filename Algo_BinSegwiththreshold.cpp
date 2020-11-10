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

    int threshold = 100;
    Halide::Func Segmentation;
    Segmentation(x,y) = Halide::cast<uint8_t>(select(Gray(x, y) > threshold, 255, 0));
   
  
    Halide::Buffer<uint8_t> buf =  Segmentation.realize(input.width(), input.height());
    
    
    Mat OriMat;
    HalideBuffer2Mat(buf,OriMat,false);
     

    imshow("gray", OriMat);
    waitKey(0);
  
    return 0;
}
