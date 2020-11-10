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
    
    std::cout << input.width() << " " << input.height() <<std::endl;
    /*
        choose Local white, calculate kr,kg,kb overthreshold
    */


    Halide::Var x,y,c;

    //How to choose Threshold
    /*  pseudo code you can implement by yourself

        Halide::Func CalSumRGB;
        CalSumRGB(x) = Halide::cast<int>(0);
        Halide::RDom r(input);
        CalSumRGB(Halide::cast<int>(input(r.x,r.y,0)) + 
                Halide::cast<int>(input(r.x,r.y,1)) + 
                Halide::cast<int>(input(r.x,r.y,2))) += 1;
        
        
        int Threshold; int topRatio = 0.8;
        int sum = 0;
        for(int i = 765; i >= 0 ; i--){
            sum += CalSumRGB(i);
            if(sum > (input.width() * input.height() * (1- topRatio))){
                Threshold = i;
                break;
            }
        }
    
    */

    int threshold = 700;
    Halide::Func CalOverThresh;
 
    CalOverThresh(c) = Halide::Tuple{Halide::cast<int>(0),Halide::cast<int>(0)};
 
    Halide::Expr sumpart = Halide::cast<int>(CalOverThresh(c)[0]);
    Halide::Expr countpart = Halide::cast<int>(CalOverThresh(c)[1]);
    
    Halide::RDom r_(input);
    sumpart += select((Halide::cast<int>(input(r_.x,r_.y,0)) 
                      + Halide::cast<int>(input(r_.x,r_.y,1))  
                      + Halide::cast<int>(input(r_.x,r_.y,2))) > threshold , Halide::cast<int>(input(r_.x,r_.y,c)), 0);
    
    countpart += select((Halide::cast<int>(input(r_.x,r_.y,0)) 
                      + Halide::cast<int>(input(r_.x,r_.y,1))  
                      + Halide::cast<int>(input(r_.x,r_.y,2))) > threshold ,1, 0);

    CalOverThresh(c) = Halide::Tuple{sumpart, countpart};
 

    Halide::Func Krgb;
    Krgb(c) = Halide::cast<int>(CalOverThresh(c)[0] / CalOverThresh(c)[1]);

    int Maxval = 255;
    Halide::Func AWB;
    Halide::Expr value = input(x, y, c);
    value = Halide::cast<float>(value);
    value = value * Maxval / Krgb(c);   //another method is calculate Per Max RGB channel in original image
    value = Halide::min(value, 255.0f); 
    value = Halide::cast<uint8_t>(value);
    AWB(x,y,c) = value;

 
    Krgb.compute_root();
    Halide::Buffer<uint8_t> result = AWB.realize(input.width(), input.height(), input.channels());

    
    imshow("before AWB" , image);
    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("after AWB", Result);

   
    waitKey(0);
  
    return 0;
}
