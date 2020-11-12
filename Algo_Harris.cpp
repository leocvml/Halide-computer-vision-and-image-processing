*
in this sample we will implement harris feature detectiio n
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

class HarrisPipeLine{
    public:
        Halide::Buffer<uint8_t> input;
        Halide::Buffer<uint8_t> result;
        Halide::Expr Sum3x3(Halide::Func f,Halide::Var x, Halide::Var y){
            return f(x-1,y-1) + f(x,y-1) + f(x+1,y-1) +
                   f(x-1,y)   + f(x,y)   + f(x+1, y) +
                   f(x-1,y+1) + f(x,y+1) + f(x+1,y+1);
        }
        HarrisPipeLine(Halide::Buffer<uint8_t>in):input(in), result(in.width(), in.height(), in.channels()){
            Halide::Func in_b = Halide::BoundaryConditions::repeat_edge(input);

            gray(x, y) = 0.299f * in_b(x, y, 0) + 0.587f * in_b(x, y, 1) + 0.114f * in_b(x, y, 2);

            Iy(x, y) = gray(x - 1, y - 1) * (-1.0f / 12) + gray(x - 1, y + 1) * (1.0f / 12) +
                    gray(x, y - 1) * (-2.0f / 12) + gray(x, y + 1) * (2.0f / 12) +
                    gray(x + 1, y - 1) * (-1.0f / 12) + gray(x + 1, y + 1) * (1.0f / 12);

            Ix(x, y) = gray(x - 1, y - 1) * (-1.0f / 12) + gray(x + 1, y - 1) * (1.0f / 12) +
                    gray(x - 1, y) * (-2.0f / 12) + gray(x + 1, y) * (2.0f / 12) +
                    gray(x - 1, y + 1) * (-1.0f / 12) + gray(x + 1, y + 1) * (1.0f / 12);

            Ixx(x,y) = Ix(x,y) * Ix(x,y);
            Ixy(x,y) = Ix(x,y) * Iy(x,y);
            Iyy(x,y) = Iy(x,y) * Iy(x,y);
            Sxx(x,y) = Sum3x3(Ixx,x,y);
            Sxy(x,y) = Sum3x3(Ixy,x,y);
            Syy(x,y) = Sum3x3(Iyy,x,y);
            det(x,y) = Sxx(x,y) * Syy(x,y) - Sxy(x,y) * Sxy(x,y);
            trace(x,y) = Sxx(x,y) + Syy(x,y);
            harris(x,y) = det(x,y) - 0.04f * trace(x,y) * trace(x,y);

            PointMask(x,y,c) = select(harris(x,y) > 20 && c == 2,255,input(x,y,0));
            output(x,y,c) = Halide::cast<uint8_t>(PointMask(x,y,c)); 

        }
        Halide::Buffer<uint8_t> execute(){
            schedule();
            output.realize(result);
            return result;
        }
    private:
        void schedule(){
            Halide::Var yo,yi;
            output.split(y,yo,yi,16).parallel(yo);
        }
         
        Halide::Expr r, g, b;
        Halide::Var x,y,c;
        Halide::Func gray, clamped, Ix,Iy,Ixx, 
            Ixy, Iyy, Sxx, Sxy, Syy, det, trace,harris,output, PointMask;
         
};

int main(int argc, char **argv) {

    //Load image
    cv::Mat image = cv::imread("/home/leoyh_cheng/Desktop/Halide/IP&CV/Img/Cheeseboard.jpg");
    Halide::Buffer<uint8_t> input = Halide::Buffer<uint8_t>::make_interleaved(image.data,image.cols, image.rows, image.channels());
    Halide::Buffer<uint8_t> result(input.width(), input.height(), input.channels());
 
    HarrisPipeLine Harris(input);
    result = Harris.execute();
 
    imshow("ORI img" , image);
    Mat Result;
    HalideBuffer2Mat(result,Result,false);
    imshow("EXP img", Result);

    waitKey(0);
  
    return 0;
}